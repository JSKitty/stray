//! Department management: persistent sandboxed sub-agents.
//!
//! Each department has its own workspace, role, frozen LLM config, and
//! conversation history. Departments run as headless `stray --department <name>`
//! subprocesses.

use crate::config::{global_config_dir, LlmConfig};
use crate::roles::{self, Role};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const DEPARTMENTS_DIR: &str = "departments";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeptStatus {
    Idle,
    Working,
    Paused,
    Done,
    Failed,
}

impl DeptStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            DeptStatus::Idle => "idle",
            DeptStatus::Working => "working",
            DeptStatus::Paused => "paused",
            DeptStatus::Done => "done",
            DeptStatus::Failed => "failed",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "working" => DeptStatus::Working,
            "paused" => DeptStatus::Paused,
            "done" => DeptStatus::Done,
            "failed" => DeptStatus::Failed,
            _ => DeptStatus::Idle,
        }
    }
}

/// Metadata about a department, loaded from department.toml + progress.txt.
#[derive(Clone)]
pub struct DepartmentMeta {
    pub name: String,
    pub role_key: String,
    pub task: String,
    pub status: DeptStatus,
    pub created_at: u64,
    pub progress: String,
    pub pid: u32,
}

/// Serializable department.toml format.
#[derive(Serialize, Deserialize)]
struct DeptToml {
    role: String,
    task: String,
    #[serde(default = "default_status")]
    status: String,
    #[serde(default)]
    created_at: u64,
    #[serde(default)]
    pid: u32,
    llm: DeptLlmToml,
}

#[derive(Serialize, Deserialize)]
struct DeptLlmToml {
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
// DepartmentManager
// ---------------------------------------------------------------------------

pub struct DepartmentManager {
    pub base_dir: PathBuf,
}

impl DepartmentManager {
    /// Create a new manager. Returns None if the global config dir is unavailable.
    pub fn new() -> Option<Self> {
        let base = global_config_dir()?.join(DEPARTMENTS_DIR);
        Some(Self { base_dir: base })
    }

    /// Path to a specific department's directory.
    pub fn dept_dir(&self, name: &str) -> PathBuf {
        self.base_dir.join(name)
    }

    /// Create a new department with the given role and task.
    /// The LLM config is resolved: role.llm if set, else snapshot from fallback_llm.
    /// `compact_at` is the context compaction threshold (0 = use default 80k).
    pub fn create(
        &self,
        name: &str,
        role: &Role,
        task: &str,
        fallback_llm: &LlmConfig,
        compact_at: usize,
    ) -> Result<PathBuf, String> {
        // Validate name: kebab-case, no path separators
        if name.is_empty() || name.contains('/') || name.contains('\\') || name.contains(' ') {
            return Err("Department name must be non-empty, no spaces or slashes".into());
        }

        let dir = self.dept_dir(name);
        if dir.exists() {
            return Err(format!("Department '{}' already exists", name));
        }

        // Create directory structure
        let workspace = dir.join("workspace");
        std::fs::create_dir_all(&workspace)
            .map_err(|e| format!("Failed to create workspace: {e}"))?;

        // Resolve LLM config: role's LLM or snapshot global
        let resolved_compact = if compact_at > 0 { compact_at } else { default_compact_at() };
        let llm = match &role.llm {
            Some(l) => DeptLlmToml {
                api_url: l.api_url.clone(),
                api_key: l.api_key.clone(),
                model: l.model.clone(),
                max_tokens: l.max_tokens,
                vision: l.vision,
                compact_at: resolved_compact,
            },
            None => DeptLlmToml {
                api_url: fallback_llm.api_url.clone(),
                api_key: fallback_llm.api_key.clone(),
                model: fallback_llm.model.clone(),
                max_tokens: fallback_llm.max_tokens,
                vision: fallback_llm.vision,
                compact_at: resolved_compact,
            },
        };

        // Timestamp
        let created_at = {
            let mut tv = libc::timeval { tv_sec: 0, tv_usec: 0 };
            unsafe { libc::gettimeofday(&mut tv, std::ptr::null_mut()) };
            tv.tv_sec as u64
        };

        let toml_data = DeptToml {
            role: role.key.clone(),
            task: task.to_string(),
            status: "idle".into(),
            created_at,
            pid: 0,
            llm,
        };

        // Write department.toml (atomic)
        let toml_path = dir.join("department.toml");
        atomic_write(&toml_path, &toml::to_string_pretty(&toml_data)
            .map_err(|e| format!("Failed to serialize: {e}"))?)?;

        // Create empty history file (progress.txt and output.md are written by
        // the agent to workspace/ since that's its cwd)
        let _ = std::fs::write(dir.join("history.json"), "[]");

        Ok(dir)
    }

    /// List all departments with their metadata.
    pub fn list(&self) -> Vec<DepartmentMeta> {
        let mut depts = Vec::new();
        let entries = match std::fs::read_dir(&self.base_dir) {
            Ok(e) => e,
            Err(_) => return depts,
        };

        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(meta) = self.load_meta(&name) {
                depts.push(meta);
            }
        }

        // Sort by created_at descending (newest first)
        depts.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        depts
    }

    /// Load metadata for a single department.
    pub fn load_meta(&self, name: &str) -> Option<DepartmentMeta> {
        let dir = self.dept_dir(name);
        let toml_path = dir.join("department.toml");
        let content = std::fs::read_to_string(&toml_path).ok()?;
        let toml_data: DeptToml = toml::from_str(&content).ok()?;

        // Agent writes to ./progress.txt from workspace/ (its cwd)
        let progress = std::fs::read_to_string(dir.join("workspace/progress.txt"))
            .or_else(|_| std::fs::read_to_string(dir.join("progress.txt")))
            .unwrap_or_default()
            .trim()
            .to_string();

        Some(DepartmentMeta {
            name: name.to_string(),
            role_key: toml_data.role,
            task: toml_data.task,
            status: DeptStatus::from_str(&toml_data.status),
            created_at: toml_data.created_at,
            progress,
            pid: toml_data.pid,
        })
    }

    /// Delete a department (removes entire directory).
    pub fn delete(&self, name: &str) -> Result<(), String> {
        let dir = self.dept_dir(name);
        if !dir.exists() {
            return Err(format!("Department '{}' not found", name));
        }

        // Kill if running
        if let Some(meta) = self.load_meta(name) {
            if meta.pid > 0 && self.is_alive(meta.pid) {
                unsafe { libc::kill(meta.pid as i32, libc::SIGTERM); }
                // Brief wait for graceful shutdown
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }

        std::fs::remove_dir_all(&dir)
            .map_err(|e| format!("Failed to delete department: {e}"))
    }

    /// Spawn a department subprocess. Returns the PID.
    pub fn spawn(&self, name: &str) -> Result<u32, String> {
        let meta = self.load_meta(name)
            .ok_or_else(|| format!("Department '{}' not found", name))?;

        // Don't spawn if already running
        if meta.status == DeptStatus::Working && meta.pid > 0 && self.is_alive(meta.pid) {
            return Err(format!("Department '{}' is already running (PID {})", name, meta.pid));
        }

        // Remove pause flag if present
        let pause_path = self.dept_dir(name).join("pause");
        let _ = std::fs::remove_file(&pause_path);

        let exe = std::env::current_exe()
            .map_err(|e| format!("Cannot find stray binary: {e}"))?;

        // Determine if the role is read-only (no write/edit tools)
        let read_only = if let Some(role) = roles::find_role(&meta.role_key) {
            !role.tools.iter().any(|t| t == "write" || t == "edit")
        } else {
            false
        };

        let workspace = self.dept_dir(name).join("workspace");

        // Build command with sandbox wrapping (falls back gracefully)
        let mut cmd = crate::sandbox::wrap_command(&workspace, read_only, name, &exe);
        // Redirect stderr to a log file (piped stderr blocks if buffer fills and parent never reads)
        let log_file = std::fs::File::create(self.dept_dir(name).join("stderr.log"))
            .map_err(|e| format!("Failed to create stderr log: {e}"))?;
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::from(log_file));

        let child = cmd.spawn()
            .map_err(|e| format!("Failed to spawn department: {e}"))?;

        let pid = child.id();

        // Update department.toml with PID and status
        self.update_status(name, DeptStatus::Working, pid);

        Ok(pid)
    }

    /// Create a pause flag file. The headless runner checks this between rounds.
    pub fn pause(&self, name: &str) {
        let pause_path = self.dept_dir(name).join("pause");
        let _ = std::fs::write(&pause_path, "");
    }

    /// Check if a process is alive via kill(pid, 0).
    pub fn is_alive(&self, pid: u32) -> bool {
        if pid == 0 {
            return false;
        }
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }

    /// Update status and PID in department.toml (atomic write).
    pub fn update_status(&self, name: &str, status: DeptStatus, pid: u32) {
        let dir = self.dept_dir(name);
        let toml_path = dir.join("department.toml");
        let content = match std::fs::read_to_string(&toml_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut toml_data: DeptToml = match toml::from_str(&content) {
            Ok(d) => d,
            Err(_) => return,
        };

        toml_data.status = status.as_str().to_string();
        toml_data.pid = pid;

        if let Ok(s) = toml::to_string_pretty(&toml_data) {
            let _ = atomic_write(&toml_path, &s);
        }
    }

    /// Load the frozen LLM config from a department's department.toml.
    pub fn load_llm_config(&self, name: &str) -> Option<LlmConfig> {
        let dir = self.dept_dir(name);
        let toml_path = dir.join("department.toml");
        let content = std::fs::read_to_string(&toml_path).ok()?;
        let toml_data: DeptToml = toml::from_str(&content).ok()?;

        // We need a LlmConfig with the same fields. Build one manually since
        // LlmConfig uses Deserialize and we have a DeptLlmToml.
        // Serialize DeptLlmToml to TOML, wrap in [llm] table, deserialize as partial.
        Some(LlmConfig {
            api_url: toml_data.llm.api_url,
            api_key: toml_data.llm.api_key,
            model: toml_data.llm.model,
            max_tokens: toml_data.llm.max_tokens,
            vision: toml_data.llm.vision,
        })
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
// DepartmentTool — allows Stray to create departments via XML tool tags
// ---------------------------------------------------------------------------

use crate::tools::Tool;
use std::sync::{Arc, Mutex};

/// Tool that lets the main Stray agent create and check departments.
/// Stores a reference to the current LLM config for snapshotting.
pub struct DepartmentTool {
    llm_config: Arc<Mutex<LlmConfig>>,
    description: String,
}

impl DepartmentTool {
    pub fn new(llm_config: Arc<Mutex<LlmConfig>>) -> Self {
        // Build description with available roles
        let all_roles = roles::load_roles();
        let roles_list: Vec<String> = all_roles.iter().map(|r| {
            let tools = r.tools.join(", ");
            format!("  - {} (tools: {})", r.key, tools)
        }).collect();

        // List existing departments
        let dept_list = if let Some(mgr) = DepartmentManager::new() {
            let depts = mgr.list();
            if depts.is_empty() {
                "  (none)".to_string()
            } else {
                depts.iter().map(|d| {
                    format!("  - {} [{}]{}", d.name, d.status.as_str(),
                        if d.progress.is_empty() { String::new() }
                        else { format!(" — {}", d.progress) })
                }).collect::<Vec<_>>().join("\n")
            }
        } else {
            "  (unavailable)".to_string()
        };

        let base_path = DepartmentManager::new()
            .map(|m| m.base_dir.to_string_lossy().to_string())
            .unwrap_or_default();

        let description = format!(
            "Manage sandboxed departments (sub-agents).\n\n\
             IMPORTANT — Department constraints:\n\
             - Each department has its own workspace at: {base_path}/<name>/workspace/\n\
             - Departments can READ files anywhere on the system (your project, etc.)\n\
             - Departments can only WRITE within their own workspace\n\
             - Departments are best for small/mid-sized tasks — don't copy large directories into them\n\
             - If a department needs project files, tell it to READ them from the original location\n\
             - You (Stray) can write files to a department's workspace using your own write/bash tools\n\
             - You will be AUTOMATICALLY NOTIFIED when a department finishes — no need to poll or check repeatedly\n\n\
             Actions:\n\
             - create: Spin up a new department to work on a task\n\
             - check: View a department's status, progress, and output\n\
             - message: Send a message to a department (auto-resumes if stopped, queued if running)\n\n\
             Available roles:\n{}\n\n\
             Existing departments:\n{}",
            roles_list.join("\n"),
            dept_list
        );

        Self { llm_config, description }
    }
}

impl Tool for DepartmentTool {
    fn name(&self) -> &str { "department" }
    fn description(&self) -> &str {
        &self.description
    }
    fn tag(&self) -> &str { "department" }
    fn usage_hint(&self) -> &str {
        "action: create\nname: my-task\nrole: software-engineer\ntask: Describe the task. The department can read files from anywhere but writes only to its workspace.\n\n\
         action: check\nname: my-task\n\n\
         action: message\nname: my-task\nmessage: Follow-up instructions for the department"
    }

    fn display_action(&self, input: &str) -> String {
        let mut action = "create";
        let mut name = "";
        for line in input.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("action:") {
                action = if rest.trim() == "check" { "check" } else { "create" };
            }
            if let Some(rest) = line.strip_prefix("name:") {
                name = rest.trim();
            }
        }
        match action {
            "check" => format!("Checking department '{name}'"),
            "message" => format!("Messaging department '{name}'"),
            _ => format!("Creating department '{name}'"),
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
            } else if let Some(rest) = line.strip_prefix("message:") {
                message = rest.trim().to_string();
            }
        }

        if name.is_empty() {
            return "[error] Department name is required".into();
        }

        let name = name.replace(' ', "-").to_lowercase();

        match action.as_str() {
            "check" => self.check_department(&name),
            "message" => self.message_department(&name, &message),
            _ => self.create_department(&name, &role_key, &task),
        }
    }
}

impl DepartmentTool {
    fn create_department(&self, name: &str, role_key: &str, task: &str) -> String {
        if task.is_empty() {
            return "[error] Department task is required".into();
        }

        let role = match roles::find_role(role_key) {
            Some(r) => r,
            None => return format!("[error] Unknown role: {role_key}"),
        };

        let llm_config = match self.llm_config.lock() {
            Ok(c) => c.clone(),
            Err(_) => return "[error] Could not read LLM config".into(),
        };

        let manager = match DepartmentManager::new() {
            Some(m) => m,
            None => return "[error] Cannot determine config directory".into(),
        };

        if let Err(e) = manager.create(name, &role, task, &llm_config, 0) {
            return format!("[error] {e}");
        }

        let workspace = manager.dept_dir(name).join("workspace");
        let ws_display = workspace.to_string_lossy();

        match manager.spawn(name) {
            Ok(pid) => format!(
                "[Department '{name}' created — role: {}, PID: {pid}, status: working]\n\
                 Workspace: {ws_display}\n\
                 Note: The department can READ files anywhere, but can only WRITE within its workspace.\n\
                 To check on it later, use: <department>\naction: check\nname: {name}\n</department>",
                role.name
            ),
            Err(e) => format!(
                "[Department '{name}' created but failed to start: {e}]"
            ),
        }
    }

    fn check_department(&self, name: &str) -> String {
        let manager = match DepartmentManager::new() {
            Some(m) => m,
            None => return "[error] Cannot determine config directory".into(),
        };

        let meta = match manager.load_meta(name) {
            Some(m) => m,
            None => return format!("[error] Department '{name}' not found"),
        };

        let status = meta.status.as_str();
        let progress = if meta.progress.is_empty() { "—".to_string() } else { meta.progress };

        // Read output if available
        let dir = manager.dept_dir(name);
        let output_path = dir.join("workspace/output.md");
        let output = std::fs::read_to_string(&output_path)
            .unwrap_or_default();
        let output = output.trim();

        let mut result = format!("[Department '{name}' — status: {status}, progress: {progress}]");

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

    fn message_department(&self, name: &str, message: &str) -> String {
        if message.is_empty() {
            return "[error] Message is required".into();
        }

        let manager = match DepartmentManager::new() {
            Some(m) => m,
            None => return "[error] Cannot determine config directory".into(),
        };

        let meta = match manager.load_meta(name) {
            Some(m) => m,
            None => return format!("[error] Department '{name}' not found"),
        };

        // Inject the message into history with clear framing so the agent acts on it
        let history_path = manager.dept_dir(name).join("history.json");
        let msg_content = if meta.status == DeptStatus::Done || meta.status == DeptStatus::Failed {
            // Completed department — frame as a new follow-up task
            format!("[{}] [System] You have a new follow-up task. Act on this and update ./output.md and ./progress.txt with your new results:\n{message}", crate::timestamp())
        } else {
            format!("[{}] {message}", crate::timestamp())
        };
        if let Ok(content) = std::fs::read_to_string(&history_path) {
            if let Ok(mut entries) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
                entries.push(serde_json::json!({
                    "role": "user",
                    "content": msg_content
                }));
                if let Ok(json) = serde_json::to_string_pretty(&entries) {
                    let _ = atomic_write(&history_path, &json);
                }
            }
        }

        // If already running, message is queued — the agent will see it on next history load
        let already_running = meta.status == DeptStatus::Working
            && meta.pid > 0
            && manager.is_alive(meta.pid);

        if already_running {
            return format!(
                "[Message queued for department '{name}' (PID {}, currently working). \
                 It will see the message on its next round.]",
                meta.pid
            );
        }

        // Not running — auto-resume with the injected message
        match manager.spawn(name) {
            Ok(pid) => format!(
                "[Department '{name}' messaged and resumed — PID: {pid}, status: working]\n\
                 To check on it later, use: <department>\naction: check\nname: {name}\n</department>"
            ),
            Err(e) => format!("[error] Failed to resume '{name}': {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Headless department runner
// ---------------------------------------------------------------------------

/// History entry for JSON serialization.
#[derive(Serialize, Deserialize)]
struct HistoryEntry {
    role: String,
    content: String,
}

/// Run a department in headless mode (no TUI, no events).
/// Called via: stray --department <name>
pub fn run_headless(name: &str) {
    use crate::{call_llm, formats, Message, Role as MsgRole, tools};

    // Resolve the departments base dir
    let manager = match DepartmentManager::new() {
        Some(m) => m,
        None => {
            eprintln!("[dept] Cannot determine config directory");
            std::process::exit(1);
        }
    };

    let dir = manager.dept_dir(name);
    if !dir.exists() {
        eprintln!("[dept] Department '{}' not found", name);
        std::process::exit(1);
    }

    // Load department.toml
    let toml_path = dir.join("department.toml");
    let toml_content = match std::fs::read_to_string(&toml_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[dept] Cannot read department.toml: {e}");
            std::process::exit(1);
        }
    };
    let dept_toml: DeptToml = match toml::from_str(&toml_content) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[dept] Invalid department.toml: {e}");
            std::process::exit(1);
        }
    };

    // Build LLM config from frozen department config
    let llm_config = LlmConfig {
        api_url: dept_toml.llm.api_url,
        api_key: dept_toml.llm.api_key,
        model: dept_toml.llm.model,
        max_tokens: dept_toml.llm.max_tokens,
        vision: dept_toml.llm.vision,
    };

    // Load role
    let role = match roles::find_role(&dept_toml.role) {
        Some(r) => r,
        None => {
            eprintln!("[dept] Unknown role: {}", dept_toml.role);
            std::process::exit(1);
        }
    };

    // Build filtered tool registry (only role's tools)
    let vision_flag = std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(llm_config.vision)
    );
    let registry = {
        let mut r = tools::ToolRegistry::new();
        for tool_name in &role.tools {
            let name: &str = tool_name;
            match name {
                "bash" => r.add(Box::new(tools::BashTool)),
                "read" => r.add(Box::new(tools::ReadTool::new(vision_flag.clone()))),
                "write" => r.add(Box::new(tools::WriteTool)),
                "edit" => r.add(Box::new(tools::EditTool)),
                _ => eprintln!("[dept] Unknown tool in role: {tool_name}"),
            }
        }
        r
    };

    // Build format + tools JSON
    let format = formats::format_for_model(&llm_config.model, &registry);
    let tools_json = format.format_tools(&registry);
    let tags: Vec<&str> = registry.tags();

    // Change to workspace directory
    let workspace = dir.join("workspace");
    if let Err(e) = std::env::set_current_dir(&workspace) {
        eprintln!("[dept] Cannot chdir to workspace: {e}");
        std::process::exit(1);
    }
    let cwd = workspace.to_string_lossy().to_string();

    // Build system prompt — output instructions BEFORE tool docs so they don't get buried
    let system_prompt = format!(
        "{}\n\n\
         You are working in a sandboxed department workspace.\n\
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
        dept_toml.task,
        format.system_prompt_suffix(&registry)
    );

    // Load or initialize message history
    let history_path = dir.join("history.json");
    let mut messages = load_history(&history_path, &system_prompt);

    // If resuming (history has more than just system message), log it
    if messages.len() > 1 {
        eprintln!("[dept] Resuming '{}' with {} messages", name, messages.len());
    } else {
        eprintln!("[dept] Starting '{}' with role '{}'", name, role.name);
        // Add initial heartbeat message
        messages.push(Message {
            role: MsgRole::User,
            content: format!("[{}] Begin your task.", crate::timestamp()),
        });
    }

    // Update status to working + write PID
    let pid = std::process::id();
    manager.update_status(name, DeptStatus::Working, pid);

    // Set up SIGTERM handler for graceful shutdown
    let dept_name = name.to_string();
    let dept_dir = dir.clone();
    unsafe {
        HEADLESS_STATE = Some(HeadlessState {
            name: dept_name,
            dir: dept_dir,
        });
        libc::signal(libc::SIGTERM, headless_signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, headless_signal_handler as *const () as libc::sighandler_t);
    }

    let max_rounds = role.max_rounds;
    let compact_at = dept_toml.llm.compact_at;
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
            eprintln!("[dept] Pause requested, saving state");
            save_history(&history_path, &messages);
            manager.update_status(name, DeptStatus::Paused, 0);
            let _ = std::fs::remove_file(&pause_path);
            return;
        }

        // Call LLM (headless: no AppState, no event_rx)
        let resp = match call_llm(
            &llm_config, &messages, &tools_json, None, None, &tags, &mut Vec::new()
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[dept] LLM error: {e}");
                save_history(&history_path, &messages);
                manager.update_status(name, DeptStatus::Failed, 0);
                return;
            }
        };

        // Parse tool calls
        let (calls, _) = format.parse_response(&resp.content);
        messages.push(Message { role: MsgRole::Assistant, content: resp.content });

        if calls.is_empty() {
            // No tool calls — agent considers itself done
            // Finalisation: ensure progress + output are written
            if !has_progress(&progress_path) {
                eprintln!("[dept] No progress on finish, requesting finalisation");
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
            eprintln!("[dept] No tool calls, finishing");
            break;
        }

        // Execute tools synchronously
        let mut results: Vec<(String, String, String)> = Vec::new();
        for call in &calls {
            let tool = registry.tools().iter().find(|t| t.name() == call.tool);
            match tool {
                Some(t) => {
                    eprintln!("[dept] {} → {}", t.name(), tools::truncate_middle(&call.input, 60));
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
            eprintln!("[dept] Context at ~{token_count} tokens, compacting...");
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
                    eprintln!("[dept] Compacted to ~{new_tokens} tokens ({:.0}% reduction)",
                        (1.0 - new_tokens as f64 / token_count as f64) * 100.0);
                    save_history(&history_path, &messages);
                }
                Err(e) => {
                    eprintln!("[dept] Compaction failed: {e}");
                    messages.pop(); // remove the compact prompt
                }
            }
        }

        if max_rounds > 0 && round >= max_rounds {
            eprintln!("[dept] Max rounds ({max_rounds}) reached");
            break;
        }
    }

    // Finished — save final state
    save_history(&history_path, &messages);
    manager.update_status(name, DeptStatus::Done, 0);
    eprintln!("[dept] Department '{}' completed", name);
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

// ---------------------------------------------------------------------------
// Signal handling for headless mode
// ---------------------------------------------------------------------------

static mut HEADLESS_STATE: Option<HeadlessState> = None;

struct HeadlessState {
    name: String,
    dir: PathBuf,
}

extern "C" fn headless_signal_handler(_sig: libc::c_int) {
    // Save status as paused and exit gracefully
    // Note: we can't do full history save from a signal handler (not async-signal-safe),
    // but we can update the status file which is small
    unsafe {
        if let Some(ref state) = HEADLESS_STATE {
            // Write paused status — minimal I/O in signal handler
            let status_msg = format!("status = \"paused\"\npid = 0\n");
            let flag = state.dir.join(".status_paused");
            // Best-effort write
            let _ = std::fs::write(&flag, &status_msg);
        }
    }
    unsafe { libc::_exit(0); }
}
