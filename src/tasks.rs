//! Task management: persistent sandboxed sub-agents.
//!
//! Each task has its own workspace, role, frozen LLM config, and
//! conversation history. Tasks run as headless `stray --task <name>`
//! subprocesses.

use crate::config::{global_config_dir, LlmConfig};
use crate::roles::{self, Role};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const TASKS_DIR: &str = "tasks";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskStatus {
    Idle,
    Working,
    Paused,
    Done,
    Failed,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskStatus::Idle => "idle",
            TaskStatus::Working => "working",
            TaskStatus::Paused => "paused",
            TaskStatus::Done => "done",
            TaskStatus::Failed => "failed",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "working" => TaskStatus::Working,
            "paused" => TaskStatus::Paused,
            "done" => TaskStatus::Done,
            "failed" => TaskStatus::Failed,
            _ => TaskStatus::Idle,
        }
    }
}

/// Metadata about a task, loaded from task.toml + progress.txt.
#[derive(Clone)]
pub struct TaskMeta {
    pub name: String,
    pub role_key: String,
    pub goal: String,
    pub status: TaskStatus,
    pub created_at: u64,
    /// Unix seconds of the task's last status change — the idle clock for reaping.
    /// Falls back to `created_at` when absent (migrated/old files).
    pub last_active: u64,
    /// Unix seconds we last nudged the agent to reap this task (0 = never).
    pub last_prodded: u64,
    pub progress: String,
    pub pid: u32,
}

/// Serializable task.toml format.
#[derive(Serialize, Deserialize)]
struct TaskToml {
    role: String,
    // Old "departments" schema stored the goal under `task`; alias migrates it.
    #[serde(alias = "task")]
    goal: String,
    #[serde(default = "default_status")]
    status: String,
    #[serde(default)]
    created_at: u64,
    #[serde(default)]
    last_active: u64,
    #[serde(default)]
    last_prodded: u64,
    #[serde(default)]
    pid: u32,
    llm: TaskLlmToml,
}

/// Unix seconds now.
fn now_secs() -> u64 {
    let mut tv = libc::timeval { tv_sec: 0, tv_usec: 0 };
    unsafe { libc::gettimeofday(&mut tv, std::ptr::null_mut()) };
    tv.tv_sec as u64
}

/// A task is eligible for reaping after this long with no status change.
pub const IDLE_REAP_SECS: u64 = 7 * 24 * 60 * 60; // 7 days

/// Validate a task name is a single safe path segment — no separators, no `..`,
/// no leading dot, no spaces. This is the guard against path traversal: every
/// name-taking operation (create, spawn, delete, load, …) resolves under the
/// tasks base dir, so a name like `../../etc` must never get through.
pub fn valid_task_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Task name must not be empty".into());
    }
    if name.len() > 128 {
        return Err("Task name is too long".into());
    }
    if name.starts_with('.')
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.contains(' ')
        || name.contains('\0')
    {
        return Err(format!(
            "Invalid task name '{name}': use letters, digits and dashes only"
        ));
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct TaskLlmToml {
    api_url: String,
    api_key: String,
    model: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    vision: bool,
    #[serde(default = "default_compact_at")]
    compact_at: usize,
}

fn default_status() -> String {
    "idle".into()
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_compact_at() -> usize {
    80_000 // 80% of 100k default
}

// ---------------------------------------------------------------------------
// TaskManager
// ---------------------------------------------------------------------------

pub struct TaskManager {
    pub base_dir: PathBuf,
}

impl TaskManager {
    /// Create a new manager. Returns None if the global config dir is unavailable.
    pub fn new() -> Option<Self> {
        let base = global_config_dir()?.join(TASKS_DIR);
        // One-time migration from the old "departments" name.
        if !base.exists() {
            let old = global_config_dir()?.join("departments");
            if old.is_dir() {
                let _ = std::fs::rename(&old, &base);
            }
        }
        Some(Self { base_dir: base })
    }

    /// Path to a specific task's directory.
    pub fn task_dir(&self, name: &str) -> PathBuf {
        self.base_dir.join(name)
    }

    /// Create a new task with the given role and task.
    /// The LLM config is resolved: role.llm if set, else snapshot from fallback_llm.
    /// `compact_at` is the context compaction threshold (0 = use default 80k).
    pub fn create(
        &self,
        name: &str,
        role: &Role,
        goal: &str,
        fallback_llm: &LlmConfig,
        compact_at: usize,
    ) -> Result<PathBuf, String> {
        valid_task_name(name)?;

        let dir = self.task_dir(name);
        if dir.exists() {
            return Err(format!("Task '{}' already exists", name));
        }

        // Create directory structure
        let workspace = dir.join("workspace");
        std::fs::create_dir_all(&workspace)
            .map_err(|e| format!("Failed to create workspace: {e}"))?;

        // Resolve LLM config: role's LLM or snapshot global
        let resolved_compact = if compact_at > 0 { compact_at } else { default_compact_at() };
        let llm = match &role.llm {
            Some(l) => TaskLlmToml {
                api_url: l.api_url.clone(),
                api_key: l.api_key.clone(),
                model: l.model.clone(),
                max_tokens: l.max_tokens,
                vision: l.vision,
                compact_at: resolved_compact,
            },
            None => TaskLlmToml {
                api_url: fallback_llm.api_url.clone(),
                api_key: fallback_llm.api_key.clone(),
                model: fallback_llm.model.clone(),
                max_tokens: fallback_llm.max_tokens,
                vision: fallback_llm.vision,
                compact_at: resolved_compact,
            },
        };

        let created_at = now_secs();

        let toml_data = TaskToml {
            role: role.key.clone(),
            goal: goal.to_string(),
            status: "idle".into(),
            created_at,
            last_active: created_at,
            last_prodded: 0,
            pid: 0,
            llm,
        };

        // Write task.toml (atomic)
        let toml_path = dir.join("task.toml");
        atomic_write(&toml_path, &toml::to_string_pretty(&toml_data)
            .map_err(|e| format!("Failed to serialize: {e}"))?)?;

        // Create empty history file (progress.txt and output.md are written by
        // the agent to workspace/ since that's its cwd)
        let _ = std::fs::write(dir.join("history.json"), "[]");

        Ok(dir)
    }

    /// List all tasks with their metadata.
    pub fn list(&self) -> Vec<TaskMeta> {
        let mut tasks = Vec::new();
        let entries = match std::fs::read_dir(&self.base_dir) {
            Ok(e) => e,
            Err(_) => return tasks,
        };

        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(meta) = self.load_meta(&name) {
                tasks.push(meta);
            }
        }

        // Sort by created_at descending (newest first)
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        tasks
    }

    /// Load metadata for a single task.
    pub fn load_meta(&self, name: &str) -> Option<TaskMeta> {
        valid_task_name(name).ok()?;
        let dir = self.task_dir(name);
        let toml_path = dir.join("task.toml");
        let content = std::fs::read_to_string(&toml_path).ok()?;
        let toml_data: TaskToml = toml::from_str(&content).ok()?;

        // Agent writes to ./progress.txt from workspace/ (its cwd)
        let progress = std::fs::read_to_string(dir.join("workspace/progress.txt"))
            .or_else(|_| std::fs::read_to_string(dir.join("progress.txt")))
            .unwrap_or_default()
            .trim()
            .to_string();

        // Old/migrated files have no last_active — fall back to created_at so a
        // 0 doesn't read as "idle since 1970" and trigger an instant false reap.
        let last_active = if toml_data.last_active > 0 {
            toml_data.last_active
        } else {
            toml_data.created_at
        };

        Some(TaskMeta {
            name: name.to_string(),
            role_key: toml_data.role,
            goal: toml_data.goal,
            status: TaskStatus::from_str(&toml_data.status),
            created_at: toml_data.created_at,
            last_active,
            last_prodded: toml_data.last_prodded,
            progress,
            pid: toml_data.pid,
        })
    }

    /// Delete a task (removes entire directory). Hard-scoped: the name is
    /// validated to a single safe segment so `remove_dir_all` can never escape
    /// the tasks base dir.
    pub fn delete(&self, name: &str) -> Result<(), String> {
        valid_task_name(name)?;
        let dir = self.task_dir(name);
        if !dir.exists() {
            return Err(format!("Task '{}' not found", name));
        }

        // Breadcrumb before removal — remember it lived (one line, not the workspace).
        let meta = self.load_meta(name);
        if let Some(m) = &meta {
            self.write_breadcrumb(m);
            // Kill if still running: SIGTERM, wait up to ~1s, then SIGKILL — so the
            // child (which chdir'd into workspace/) is gone before we remove the tree.
            if m.pid > 0 && self.is_alive(m.pid) {
                unsafe { libc::kill(m.pid as i32, libc::SIGTERM); }
                let mut waited = 0;
                while waited < 1000 && self.is_alive(m.pid) {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    waited += 50;
                }
                if self.is_alive(m.pid) {
                    unsafe { libc::kill(m.pid as i32, libc::SIGKILL); }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }

        std::fs::remove_dir_all(&dir)
            .map_err(|e| format!("Failed to delete task: {e}"))
    }

    /// One-line breadcrumb appended to <base>/reaped.log when a task is deleted,
    /// so the agent remembers a task existed and was cleaned up.
    fn write_breadcrumb(&self, meta: &TaskMeta) {
        use std::io::Write;
        let line = format!(
            "[{}] reaped '{}' (role {}, {}) — {}\n",
            crate::date_today(),
            meta.name,
            meta.role_key,
            meta.status.as_str(),
            meta.goal.chars().take(80).collect::<String>(),
        );
        let log = self.base_dir.join("reaped.log");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log) {
            let _ = f.write_all(line.as_bytes());
        }
    }

    /// Spawn a task subprocess. Returns the PID.
    pub fn spawn(&self, name: &str) -> Result<u32, String> {
        valid_task_name(name)?;
        let meta = self.load_meta(name)
            .ok_or_else(|| format!("Task '{}' not found", name))?;

        // Don't spawn if already running
        if meta.status == TaskStatus::Working && meta.pid > 0 && self.is_alive(meta.pid) {
            return Err(format!("Task '{}' is already running (PID {})", name, meta.pid));
        }

        // Remove pause flag if present
        let pause_path = self.task_dir(name).join("pause");
        let _ = std::fs::remove_file(&pause_path);

        let exe = std::env::current_exe()
            .map_err(|e| format!("Cannot find stray binary: {e}"))?;

        // The role's trust level drives the sandbox. A task writes to its own
        // dir (output.md, progress.txt, task.toml, history, inbox) and needs the
        // network for its own LLM calls — so writes are confined to the task dir
        // and network is allowed; a FreeRoam role runs unconfined.
        let trust = roles::find_role(&meta.role_key)
            .map(|r| r.trust)
            .unwrap_or_default();
        let task_dir = self.task_dir(name);
        let argv = vec![
            exe.to_string_lossy().to_string(),
            "--task".to_string(),
            name.to_string(),
        ];
        let mut cmd = crate::sandbox::wrap(trust, &task_dir, &argv, true, true);
        // Redirect stderr to a log file (piped stderr blocks if buffer fills and parent never reads)
        let log_file = std::fs::File::create(self.task_dir(name).join("stderr.log"))
            .map_err(|e| format!("Failed to create stderr log: {e}"))?;
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::from(log_file));

        let child = cmd.spawn()
            .map_err(|e| format!("Failed to spawn task: {e}"))?;

        let pid = child.id();

        // Update task.toml with PID and status
        self.update_status(name, TaskStatus::Working, pid);

        Ok(pid)
    }

    /// Create a pause flag file. The headless runner checks this between rounds.
    pub fn pause(&self, name: &str) {
        if valid_task_name(name).is_err() {
            return;
        }
        let pause_path = self.task_dir(name).join("pause");
        let _ = std::fs::write(&pause_path, "");
    }

    /// Check if a process is alive via kill(pid, 0).
    pub fn is_alive(&self, pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    /// Update status and PID in task.toml (atomic write). Preserves last_prodded.
    pub fn update_status(&self, name: &str, status: TaskStatus, pid: u32) {
        if valid_task_name(name).is_err() {
            return;
        }
        let dir = self.task_dir(name);
        let toml_path = dir.join("task.toml");
        let content = match std::fs::read_to_string(&toml_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut toml_data: TaskToml = match toml::from_str(&content) {
            Ok(d) => d,
            Err(_) => return,
        };

        toml_data.status = status.as_str().to_string();
        toml_data.pid = pid;
        toml_data.last_active = now_secs(); // every status change resets the idle clock

        if let Ok(s) = toml::to_string_pretty(&toml_data) {
            let _ = atomic_write(&toml_path, &s);
        }
    }

    /// Queue a message for a task's headless runner via inbox.jsonl (append-only).
    /// This is a SEPARATE channel from history.json — which the running child
    /// rewrites every round — so a message sent to a live task survives to be
    /// drained on its next round instead of being clobbered. Returns true on success.
    pub fn enqueue_message(&self, name: &str, content: &str) -> bool {
        if valid_task_name(name).is_err() {
            return false;
        }
        use std::io::Write;
        let inbox = self.task_dir(name).join("inbox.jsonl");
        let Ok(line) = serde_json::to_string(&serde_json::json!({ "content": content })) else {
            return false;
        };
        match std::fs::OpenOptions::new().create(true).append(true).open(&inbox) {
            Ok(mut f) => writeln!(f, "{line}").is_ok(),
            Err(_) => false,
        }
    }

    /// True if the task has queued inbox messages not yet drained by its runner.
    /// Also counts a crash-stranded `inbox.draining` (a drain that died before
    /// removing it) so the host's re-spawn net recovers those messages too.
    pub fn has_pending_inbox(&self, name: &str) -> bool {
        if valid_task_name(name).is_err() {
            return false;
        }
        let dir = self.task_dir(name);
        let nonempty = |p: std::path::PathBuf| {
            std::fs::metadata(&p).map(|m| m.len() > 0).unwrap_or(false)
        };
        nonempty(dir.join("inbox.jsonl")) || nonempty(dir.join("inbox.draining"))
    }

    /// Record that we nudged the agent to reap this task. Persisted so the
    /// "re-prod at most weekly" throttle survives Stray restarts.
    pub fn mark_prodded(&self, name: &str) {
        if valid_task_name(name).is_err() {
            return;
        }
        let toml_path = self.task_dir(name).join("task.toml");
        let Ok(content) = std::fs::read_to_string(&toml_path) else { return };
        let Ok(mut toml_data) = toml::from_str::<TaskToml>(&content) else { return };
        toml_data.last_prodded = now_secs();
        if let Ok(s) = toml::to_string_pretty(&toml_data) {
            let _ = atomic_write(&toml_path, &s);
        }
    }
}

// ---------------------------------------------------------------------------
// Atomic file write (tmp + rename)
// ---------------------------------------------------------------------------

fn atomic_write(path: &PathBuf, content: &str) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content)
        .map_err(|e| format!("Failed to write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("Failed to rename {}: {e}", tmp.display()))
}

// ---------------------------------------------------------------------------
// TaskTool — allows Stray to create tasks via XML tool tags
// ---------------------------------------------------------------------------

use crate::tools::Tool;
use std::sync::{Arc, Mutex};

/// Tool that lets the main Stray agent create and check tasks.
/// Stores a reference to the current LLM config for snapshotting.
pub struct TaskTool {
    llm_config: Arc<Mutex<LlmConfig>>,
    description: String,
}

impl TaskTool {
    pub fn new(llm_config: Arc<Mutex<LlmConfig>>) -> Self {
        // Build description with available roles
        let all_roles = roles::load_roles();
        let roles_list: Vec<String> = all_roles.iter().map(|r| {
            let tools = r.tools.join(", ");
            format!("  - {} (tools: {})", r.key, tools)
        }).collect();

        // List existing tasks
        let task_list = if let Some(mgr) = TaskManager::new() {
            let tasks = mgr.list();
            if tasks.is_empty() {
                "  (none)".to_string()
            } else {
                tasks.iter().map(|d| {
                    format!("  - {} [{}]{}", d.name, d.status.as_str(),
                        if d.progress.is_empty() { String::new() }
                        else { format!(" — {}", d.progress) })
                }).collect::<Vec<_>>().join("\n")
            }
        } else {
            "  (unavailable)".to_string()
        };

        let base_path = TaskManager::new()
            .map(|m| m.base_dir.to_string_lossy().to_string())
            .unwrap_or_default();

        let description = format!(
            "Manage sandboxed tasks (sub-agents).\n\n\
             IMPORTANT — Task constraints:\n\
             - Each task has its own workspace at: {base_path}/<name>/workspace/\n\
             - Tasks can READ files anywhere on the system (your project, etc.)\n\
             - Tasks can only WRITE within their own workspace\n\
             - Tasks are best for small/mid-sized tasks — don't copy large directories into them\n\
             - If a task needs project files, tell it to READ them from the original location\n\
             - You (Stray) can write files to a task's workspace using your own write/bash tools\n\
             - You will be AUTOMATICALLY NOTIFIED when a task finishes — no need to poll or check repeatedly\n\n\
             Actions:\n\
             - create: Spin up a new task to work on a goal\n\
             - check: View a task's status, progress, and output\n\
             - message: Send a message to a task (auto-resumes if stopped, queued if running)\n\
             - delete: Remove a finished/spent task and its workspace (hard delete, one-line breadcrumb kept)\n\n\
             Tasks are semi-ephemeral: after 7 days with no activity you'll be nudged to review a task and delete it if its results are already used.\n\n\
             Available roles:\n{}\n\n\
             Existing tasks:\n{}",
            roles_list.join("\n"),
            task_list
        );

        Self { llm_config, description }
    }
}

impl Tool for TaskTool {
    fn name(&self) -> &str { "task" }
    fn description(&self) -> &str {
        &self.description
    }
    fn tag(&self) -> &str { "task" }
    fn usage_hint(&self) -> &str {
        "action: create\nname: my-task\nrole: software-engineer\ntask: Describe the task. The task can read files from anywhere but writes only to its workspace.\n\n\
         action: check\nname: my-task\n\n\
         action: message\nname: my-task\nmessage: Follow-up instructions for the task\n\n\
         action: delete\nname: my-task"
    }

    fn display_action(&self, input: &str) -> String {
        let mut action = "create";
        let mut name = "";
        for line in input.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("action:") {
                action = match rest.trim() {
                    "check" => "check",
                    "message" => "message",
                    "delete" | "reap" => "delete",
                    _ => "create",
                };
            }
            if let Some(rest) = line.strip_prefix("name:") {
                name = rest.trim();
            }
        }
        match action {
            "check" => format!("Checking task '{name}'"),
            "message" => format!("Messaging task '{name}'"),
            "delete" => format!("Reaping task '{name}'"),
            _ => format!("Creating task '{name}'"),
        }
    }

    fn execute(&self, input: &str) -> String {
        // Parse key: value format
        let mut action = "create".to_string();
        let mut name = String::new();
        let mut role_key = "software-engineer".to_string();
        let mut task = String::new();
        let mut message = String::new();

        for line in input.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("action:") {
                action = rest.trim().to_lowercase();
            } else if let Some(rest) = line.strip_prefix("name:") {
                name = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("role:") {
                role_key = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("task:") {
                task = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("goal:") {
                task = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("message:") {
                message = rest.trim().to_string();
            }
        }

        if name.is_empty() {
            return "[error] Task name is required".into();
        }

        let name = name.replace(' ', "-").to_lowercase();
        if let Err(e) = valid_task_name(&name) {
            return format!("[error] {e}");
        }

        match action.as_str() {
            "check" => self.check_task(&name),
            "message" => self.message_task(&name, &message),
            "delete" | "reap" => self.delete_task(&name),
            _ => self.create_task(&name, &role_key, &task),
        }
    }
}

impl TaskTool {
    fn create_task(&self, name: &str, role_key: &str, goal: &str) -> String {
        if goal.is_empty() {
            return "[error] A task goal is required — add a `task:` line describing it".into();
        }

        let role = match roles::find_role(role_key) {
            Some(r) => r,
            None => return format!("[error] Unknown role: {role_key}"),
        };

        let llm_config = match self.llm_config.lock() {
            Ok(c) => c.clone(),
            Err(_) => return "[error] Could not read LLM config".into(),
        };

        let manager = match TaskManager::new() {
            Some(m) => m,
            None => return "[error] Cannot determine config directory".into(),
        };

        if let Err(e) = manager.create(name, &role, goal, &llm_config, 0) {
            return format!("[error] {e}");
        }

        let workspace = manager.task_dir(name).join("workspace");
        let ws_display = workspace.to_string_lossy();

        match manager.spawn(name) {
            Ok(pid) => format!(
                "[Task '{name}' created — role: {}, PID: {pid}, status: working]\n\
                 Workspace: {ws_display}\n\
                 Note: The task can READ files anywhere, but can only WRITE within its workspace.\n\
                 To check on it later, use: <task>\naction: check\nname: {name}\n</task>",
                role.name
            ),
            Err(e) => format!(
                "[Task '{name}' created but failed to start: {e}]"
            ),
        }
    }

    fn check_task(&self, name: &str) -> String {
        let manager = match TaskManager::new() {
            Some(m) => m,
            None => return "[error] Cannot determine config directory".into(),
        };

        let meta = match manager.load_meta(name) {
            Some(m) => m,
            None => return format!("[error] Task '{name}' not found"),
        };

        let status = meta.status.as_str();
        let progress = if meta.progress.is_empty() { "—".to_string() } else { meta.progress };

        // Read output if available
        let dir = manager.task_dir(name);
        let output_path = dir.join("workspace/output.md");
        let output = std::fs::read_to_string(&output_path)
            .unwrap_or_default();
        let output = output.trim();

        let mut result = format!("[Task '{name}' — status: {status}, progress: {progress}]");

        if !output.is_empty() {
            // Truncate to ~1000 chars for context
            let preview = if output.len() > 1000 {
                format!("{}...\n[truncated — full output in workspace/output.md]", &output[..1000])
            } else {
                output.to_string()
            };
            result.push_str(&format!("\n\nOutput:\n{preview}"));
        }

        result
    }

    fn message_task(&self, name: &str, message: &str) -> String {
        if message.is_empty() {
            return "[error] Message is required".into();
        }

        let manager = match TaskManager::new() {
            Some(m) => m,
            None => return "[error] Cannot determine config directory".into(),
        };

        let meta = match manager.load_meta(name) {
            Some(m) => m,
            None => return format!("[error] Task '{name}' not found"),
        };

        // Queue the message on the task's inbox (drained on its next round —
        // survives even while the task is actively running).
        let msg_content = if meta.status == TaskStatus::Done || meta.status == TaskStatus::Failed {
            // Completed task — frame as a new follow-up task
            format!("[{}] [System] You have a new follow-up task. Act on this and update ./output.md and ./progress.txt with your new results:\n{message}", crate::timestamp())
        } else {
            format!("[{}] {message}", crate::timestamp())
        };
        if !manager.enqueue_message(name, &msg_content) {
            return format!("[error] Failed to queue message for task '{name}'");
        }

        // If already running, the message is queued — the runner drains it next round
        let already_running = meta.status == TaskStatus::Working
            && meta.pid > 0
            && manager.is_alive(meta.pid);

        if already_running {
            return format!(
                "[Message queued for task '{name}' (PID {}, currently working). \
                 It will see the message on its next round.]",
                meta.pid
            );
        }

        // Not running — auto-resume with the injected message
        match manager.spawn(name) {
            Ok(pid) => format!(
                "[Task '{name}' messaged and resumed — PID: {pid}, status: working]\n\
                 To check on it later, use: <task>\naction: check\nname: {name}\n</task>"
            ),
            Err(e) => format!("[error] Failed to resume '{name}': {e}"),
        }
    }

    fn delete_task(&self, name: &str) -> String {
        let manager = match TaskManager::new() {
            Some(m) => m,
            None => return "[error] Cannot determine config directory".into(),
        };
        match manager.delete(name) {
            Ok(()) => format!("[Task '{name}' deleted — workspace removed, noted in reaped.log]"),
            Err(e) => format!("[error] {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Headless task runner
// ---------------------------------------------------------------------------

/// History entry for JSON serialization.
#[derive(Serialize, Deserialize)]
struct HistoryEntry {
    role: String,
    content: String,
}

/// Run a task in headless mode (no TUI, no events).
/// Called via: stray --task <name>
pub fn run_headless(name: &str) {
    use crate::{call_llm, formats, Message, Role as MsgRole, tools};

    // Resolve the tasks base dir
    let manager = match TaskManager::new() {
        Some(m) => m,
        None => {
            eprintln!("[task] Cannot determine config directory");
            std::process::exit(1);
        }
    };

    if let Err(e) = valid_task_name(name) {
        eprintln!("[task] {e}");
        std::process::exit(1);
    }

    let dir = manager.task_dir(name);
    if !dir.exists() {
        eprintln!("[task] Task '{}' not found", name);
        std::process::exit(1);
    }

    // Load task.toml
    let toml_path = dir.join("task.toml");
    let toml_content = match std::fs::read_to_string(&toml_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[task] Cannot read task.toml: {e}");
            std::process::exit(1);
        }
    };
    let task_toml: TaskToml = match toml::from_str(&toml_content) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[task] Invalid task.toml: {e}");
            std::process::exit(1);
        }
    };

    // Build LLM config from frozen task config
    let llm_config = LlmConfig {
        api_url: task_toml.llm.api_url,
        api_key: task_toml.llm.api_key,
        model: task_toml.llm.model,
        max_tokens: task_toml.llm.max_tokens,
        vision: task_toml.llm.vision,
    };

    // Load role
    let role = match roles::find_role(&task_toml.role) {
        Some(r) => r,
        None => {
            eprintln!("[task] Unknown role: {}", task_toml.role);
            std::process::exit(1);
        }
    };

    // Build the role's tool registry. This subprocess is already OS-sandboxed
    // (see TaskManager::spawn), so its tools run without a second, nested
    // sandbox — os_wrap = false. The role's trust still scopes write/edit.
    let vision_flag = std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(llm_config.vision)
    );
    let trust_handle = std::sync::Arc::new(std::sync::Mutex::new(role.trust));
    let registry = tools::build_registry(&role.tools, trust_handle, false, vision_flag.clone());

    // Build format + tools JSON
    let format = formats::format_for_model(&llm_config.model, &registry);
    let tools_json = format.format_tools(&registry);
    let tags: Vec<&str> = registry.tags();

    // Change to workspace directory
    let workspace = dir.join("workspace");
    if let Err(e) = std::env::set_current_dir(&workspace) {
        eprintln!("[task] Cannot chdir to workspace: {e}");
        std::process::exit(1);
    }
    let cwd = workspace.to_string_lossy().to_string();

    // Build system prompt — output instructions BEFORE tool docs so they don't get buried
    let system_prompt = format!(
        "{}\n\n\
         You are working in a sandboxed task workspace.\n\
         Today is {}. Working directory: {}\n\n\
         YOUR TASK:\n{}\n\n\
         CRITICAL RULES:\n\
         1. You MUST write your results to ./output.md using the write tool when done. Be concise — key findings and actionable points only, no filler.\n\
         2. Periodically write a ~10 word progress summary to ./progress.txt\n\
         3. Work only within your workspace directory.\n\
         {}",
        role.system_prompt,
        crate::date_today(),
        cwd,
        task_toml.goal,
        format.system_prompt_suffix(&registry)
    );

    // Load or initialize message history
    let history_path = dir.join("history.json");
    let mut messages = load_history(&history_path, &system_prompt);

    // If resuming (history has more than just system message), log it
    if messages.len() > 1 {
        eprintln!("[task] Resuming '{}' with {} messages", name, messages.len());
    } else {
        eprintln!("[task] Starting '{}' with role '{}'", name, role.name);
        // Add initial heartbeat message
        messages.push(Message {
            role: MsgRole::User,
            content: format!("[{}] Begin your task.", crate::timestamp()),
        });
    }

    // Update status to working + write PID
    let pid = std::process::id();
    manager.update_status(name, TaskStatus::Working, pid);

    // Set up SIGTERM/SIGINT handlers for a clean exit.
    unsafe {
        libc::signal(libc::SIGTERM, headless_signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, headless_signal_handler as *const () as libc::sighandler_t);
    }

    let max_rounds = role.max_rounds;
    let compact_at = task_toml.llm.compact_at;
    let mut round: u64 = 0;
    let pause_path = dir.join("pause");
    let progress_path = workspace.join("progress.txt");

    /// Check if progress.txt has meaningful content.
    fn has_progress(path: &std::path::Path) -> bool {
        std::fs::read_to_string(path)
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
    }

    loop {
        // Check pause flag
        if pause_path.exists() {
            eprintln!("[task] Pause requested, saving state");
            save_history(&history_path, &messages);
            manager.update_status(name, TaskStatus::Paused, 0);
            let _ = std::fs::remove_file(&pause_path);
            return;
        }

        // Pick up any messages the host queued while we were working.
        drain_inbox(&dir, &mut messages);

        // Call LLM (headless: no AppState, no event_rx)
        let resp = match call_llm(
            &llm_config, &messages, &tools_json, None, None, &tags, &mut Vec::new()
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[task] LLM error: {e}");
                save_history(&history_path, &messages);
                manager.update_status(name, TaskStatus::Failed, 0);
                return;
            }
        };

        // Parse tool calls
        let (calls, _) = format.parse_response(&resp.content);
        messages.push(Message { role: MsgRole::Assistant, content: resp.content });

        if calls.is_empty() {
            // A message may have landed during this round — pick it up and keep
            // going rather than finishing on top of unseen work.
            if drain_inbox(&dir, &mut messages) > 0 {
                save_history(&history_path, &messages);
                continue;
            }
            // No tool calls — agent considers itself done
            // Finalisation: ensure progress + output are written
            if !has_progress(&progress_path) {
                eprintln!("[task] No progress on finish, requesting finalisation");
                messages.push(Message { role: MsgRole::User,
                    content: "[System] You are finishing. Write a brief final status to ./progress.txt and ensure ./output.md contains your results.".into() });
                // One more round for finalisation
                if let Ok(r) = call_llm(&llm_config, &messages, &tools_json, None, None, &tags, &mut Vec::new()) {
                    let (fin_calls, _) = format.parse_response(&r.content);
                    messages.push(Message { role: MsgRole::Assistant, content: r.content });
                    for call in &fin_calls {
                        if let Some(t) = registry.tools().iter().find(|t| t.name() == call.tool) {
                            let output = t.execute(&call.input);
                            messages.push(Message { role: MsgRole::User,
                                content: format.format_results(&[(call.tool.clone(), call.input.clone(), output)]) });
                        }
                    }
                }
            }
            eprintln!("[task] No tool calls, finishing");
            break;
        }

        // Execute tools synchronously
        let mut results: Vec<(String, String, String)> = Vec::new();
        for call in &calls {
            let tool = registry.tools().iter().find(|t| t.name() == call.tool);
            match tool {
                Some(t) => {
                    eprintln!("[task] {} → {}", t.name(), tools::truncate_middle(&call.input, 60));
                    let output = if let Some(spawn_result) = t.spawn(&call.input) {
                        // Blocking wait for spawned tools
                        match spawn_result {
                            Ok(child) => {
                                match child.wait_with_output() {
                                    Ok(out) => t.format_output(&out),
                                    Err(e) => format!("[error] {e}"),
                                }
                            }
                            Err(e) => e,
                        }
                    } else {
                        t.execute(&call.input)
                    };
                    results.push((call.tool.clone(), call.input.clone(), output));
                }
                None => {
                    results.push((call.tool.clone(), call.input.clone(),
                        format!("[error] Unknown tool: {}", call.tool)));
                }
            }
        }

        // Append tool results
        let mut result_msg = format.format_results(&results);

        // Progress nudges: after round 1, then every 5 rounds
        round += 1;
        if !has_progress(&progress_path) && (round == 1 || round % 5 == 0) {
            result_msg.push_str("\n\n[System] Remember to update ./progress.txt with a brief ~10 word status summary.");
        } else if round % 5 == 0 {
            result_msg.push_str("\n\n[System] Update ./progress.txt with your current status.");
        }

        messages.push(Message {
            role: MsgRole::User,
            content: result_msg,
        });

        // Save history after each round
        save_history(&history_path, &messages);

        // Auto-compact if context is getting large
        let token_count = crate::estimate_tokens(&messages);
        if token_count >= compact_at {
            eprintln!("[task] Context at ~{token_count} tokens, compacting...");
            messages.push(Message {
                role: MsgRole::User,
                content: crate::COMPACT_PROMPT.into(),
            });
            match call_llm(&llm_config, &messages, &None, None, None, &[], &mut Vec::new()) {
                Ok(resp) => {
                    let system = messages.first().cloned()
                        .unwrap_or(Message { role: MsgRole::System, content: String::new() });
                    messages.clear();
                    messages.push(system);
                    messages.push(Message {
                        role: MsgRole::Assistant,
                        content: format!("[Context compacted from ~{token_count} tokens]\n\n{}", resp.content),
                    });
                    let new_tokens = crate::estimate_tokens(&messages);
                    eprintln!("[task] Compacted to ~{new_tokens} tokens ({:.0}% reduction)",
                        (1.0 - new_tokens as f64 / token_count as f64) * 100.0);
                    save_history(&history_path, &messages);
                }
                Err(e) => {
                    eprintln!("[task] Compaction failed: {e}");
                    messages.pop(); // remove the compact prompt
                }
            }
        }

        if max_rounds > 0 && round >= max_rounds {
            eprintln!("[task] Max rounds ({max_rounds}) reached");
            break;
        }
    }

    // Finished — save final state
    save_history(&history_path, &messages);
    manager.update_status(name, TaskStatus::Done, 0);
    eprintln!("[task] Task '{}' completed", name);
}

// ---------------------------------------------------------------------------
// History persistence
// ---------------------------------------------------------------------------

fn load_history(path: &PathBuf, system_prompt: &str) -> Vec<crate::Message> {
    use crate::{Message, Role as MsgRole};

    if let Ok(content) = std::fs::read_to_string(path) {
        if let Ok(entries) = serde_json::from_str::<Vec<HistoryEntry>>(&content) {
            if !entries.is_empty() {
                return entries.iter().map(|e| Message {
                    role: match e.role.as_str() {
                        "system" => MsgRole::System,
                        "user" => MsgRole::User,
                        "assistant" => MsgRole::Assistant,
                        _ => MsgRole::User,
                    },
                    content: e.content.clone(),
                }).collect();
            }
        }
    }

    // Fresh start: system message only
    vec![Message { role: MsgRole::System, content: system_prompt.to_string() }]
}

fn save_history(path: &PathBuf, messages: &[crate::Message]) {
    let entries: Vec<HistoryEntry> = messages.iter().map(|m| HistoryEntry {
        role: m.role.as_str().to_string(),
        content: m.content.clone(),
    }).collect();

    if let Ok(json) = serde_json::to_string_pretty(&entries) {
        let _ = atomic_write(path, &json);
    }
}

/// Move any queued inbox messages (inbox.jsonl) into the live message list.
/// Renames the inbox aside first so a concurrent send — which re-creates
/// inbox.jsonl — isn't lost during the drain. Returns how many were drained.
///
/// Crash recovery: a prior drain that died between the rename and the remove
/// leaves messages stranded in inbox.draining. We re-ingest that leftover first
/// (it's older than the current inbox.jsonl, and — since remove happens before
/// any processing — was never handled), so a mid-drain crash never orphans mail.
///
/// Residual race (accepted): a sender that has already opened its append fd at
/// the instant of the rename writes into the renamed file, which may be read or
/// deleted before that write lands. The window is microseconds at human message
/// pace; a lost message can always be re-sent. If this ever matters, switch to
/// per-message files (maildir-style) or an flock.
fn drain_inbox(dir: &std::path::Path, messages: &mut Vec<crate::Message>) -> usize {
    let inbox = dir.join("inbox.jsonl");
    let staged = dir.join("inbox.draining");
    let mut n = 0;
    // Recover a leftover from a crashed prior drain, oldest-first.
    if staged.exists() {
        n += ingest_staged(&staged, messages);
    }
    // Normal path: atomically move the live inbox aside, then ingest it.
    if inbox.exists() && std::fs::rename(&inbox, &staged).is_ok() {
        n += ingest_staged(&staged, messages);
    }
    if n > 0 {
        eprintln!("[task] drained {n} queued message(s) from inbox");
    }
    n
}

/// Read every JSON line of `staged` into `messages`, then delete it.
fn ingest_staged(staged: &std::path::Path, messages: &mut Vec<crate::Message>) -> usize {
    use crate::{Message, Role as MsgRole};
    let content = std::fs::read_to_string(staged).unwrap_or_default();
    let _ = std::fs::remove_file(staged);
    let mut n = 0;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(c) = v.get("content").and_then(|c| c.as_str()) {
                messages.push(Message { role: MsgRole::User, content: c.to_string() });
                n += 1;
            }
        }
    }
    n
}

// ---------------------------------------------------------------------------
// Signal handling for headless mode
// ---------------------------------------------------------------------------

extern "C" fn headless_signal_handler(_sig: libc::c_int) {
    // A full history save isn't async-signal-safe, so we just exit cleanly.
    // The host marks a killed task Failed via its liveness check; a paused task
    // has already saved history in the loop before setting Paused status.
    unsafe { libc::_exit(0); }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{valid_task_name, TaskToml};

    #[test]
    fn accepts_plain_names() {
        for ok in ["refactor-parser", "audit-gist", "task1", "a"] {
            assert!(valid_task_name(ok).is_ok(), "should accept {ok}");
        }
    }

    #[test]
    fn rejects_path_traversal_and_separators() {
        for bad in ["../foo", "..", "a/b", "a\\b", ".hidden", "with space", "", "a/../../etc"] {
            assert!(valid_task_name(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn deserializes_old_departments_schema() {
        // Pre-Phase-1 files used `task = ...` for the goal and had no
        // last_active/last_prodded. Migration renames the dir but not the file,
        // so the schema must still load (via #[serde(alias)] + defaults) or every
        // migrated task silently vanishes.
        let old = r#"
role = "software-engineer"
task = "refactor the parser"
status = "done"
created_at = 1700000000
pid = 0

[llm]
api_url = "http://localhost:1234/v1"
api_key = "k"
model = "m"
"#;
        let parsed: TaskToml = toml::from_str(old).expect("old schema must still deserialize");
        assert_eq!(parsed.goal, "refactor the parser"); // via #[serde(alias = "task")]
        assert_eq!(parsed.created_at, 1_700_000_000);
        assert_eq!(parsed.last_active, 0); // default — load_meta falls back to created_at
        assert_eq!(parsed.last_prodded, 0);
        assert_eq!(parsed.status, "done");
    }

    #[test]
    fn inbox_drains_and_clears() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("stray-inbox-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let inbox = dir.join("inbox.jsonl");
        {
            let mut f = std::fs::File::create(&inbox).unwrap();
            // Same shape enqueue_message writes.
            writeln!(f, "{}", serde_json::json!({ "content": "first message" })).unwrap();
            writeln!(f, "{}", serde_json::json!({ "content": "second" })).unwrap();
        }

        let mut msgs: Vec<crate::Message> = Vec::new();
        assert_eq!(super::drain_inbox(&dir, &mut msgs), 2);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "first message");
        assert_eq!(msgs[1].content, "second");
        assert!(!inbox.exists(), "inbox must be consumed after draining");
        // Nothing left to drain.
        assert_eq!(super::drain_inbox(&dir, &mut msgs), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inbox_recovers_crash_stranded_draining() {
        use std::io::Write;
        // Simulate a drain that crashed after the rename but before the remove:
        // a leftover inbox.draining sits beside a freshly-arrived inbox.jsonl.
        let dir = std::env::temp_dir().join(format!("stray-inbox-recover-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        {
            let mut f = std::fs::File::create(dir.join("inbox.draining")).unwrap();
            writeln!(f, "{}", serde_json::json!({ "content": "stranded" })).unwrap();
        }
        {
            let mut f = std::fs::File::create(dir.join("inbox.jsonl")).unwrap();
            writeln!(f, "{}", serde_json::json!({ "content": "fresh" })).unwrap();
        }

        let mut msgs: Vec<crate::Message> = Vec::new();
        assert_eq!(super::drain_inbox(&dir, &mut msgs), 2);
        // Recovered leftover comes first (it's older than the live inbox).
        assert_eq!(msgs[0].content, "stranded");
        assert_eq!(msgs[1].content, "fresh");
        assert!(!dir.join("inbox.jsonl").exists());
        assert!(!dir.join("inbox.draining").exists(), "staged file must be removed");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
