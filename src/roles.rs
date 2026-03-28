//! Role definitions: system prompts, tool sets, and optional LLM config.
//!
//! Roles define what a Stray agent (main or department) can do and how it thinks.
//! Built-in roles are always present; custom roles are loaded from roles.toml.

use crate::config::{global_config_dir, LlmConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const ROLES_FILENAME: &str = "roles.toml";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// LLM configuration embedded in a role — makes departments fully self-contained.
/// If absent, the current global config is snapshotted at department creation time.
#[derive(Clone, Serialize, Deserialize)]
pub struct RoleLlmConfig {
    pub api_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub vision: bool,
}

fn default_max_tokens() -> u32 {
    4096
}

impl RoleLlmConfig {
    /// Snapshot the current global LLM config into a RoleLlmConfig.
    pub fn from_global(llm: &LlmConfig) -> Self {
        Self {
            api_url: llm.api_url.clone(),
            api_key: llm.api_key.clone(),
            model: llm.model.clone(),
            max_tokens: llm.max_tokens,
            vision: llm.vision,
        }
    }
}

/// A role defines the personality, tools, and model for an agent.
#[derive(Clone)]
pub struct Role {
    pub key: String,
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub tools: Vec<String>,
    pub max_rounds: u64,
    pub llm: Option<RoleLlmConfig>,
    pub builtin: bool,
}

/// Serializable format for roles.toml
#[derive(Serialize, Deserialize)]
struct RoleFile {
    #[serde(default)]
    role: Vec<RoleEntry>,
}

#[derive(Serialize, Deserialize)]
struct RoleEntry {
    key: String,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    system_prompt: String,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default = "default_max_rounds")]
    max_rounds: u64,
    #[serde(default)]
    llm: Option<RoleLlmConfig>,
}

fn default_max_rounds() -> u64 {
    50
}

// ---------------------------------------------------------------------------
// Built-in roles
// ---------------------------------------------------------------------------

const ALL_TOOLS: &[&str] = &["bash", "read", "write", "edit"];

fn builtin_roles() -> Vec<Role> {
    vec![
        Role {
            key: "free-spirit".into(),
            name: "Free Spirit".into(),
            description: "Open-ended agentic companion — full autonomy".into(),
            system_prompt: "You are a helpful autonomous assistant with access to tools.\n\
                Be concise. Only run commands when needed. Think step by step."
                .into(),
            tools: ALL_TOOLS.iter().map(|s| s.to_string()).collect(),
            max_rounds: 50,
            llm: None,
            builtin: true,
        },
        Role {
            key: "software-engineer".into(),
            name: "Software Engineer".into(),
            description: "Focused coding — reads before modifying, writes tests".into(),
            system_prompt: "You are a focused software engineer. Read code before modifying it.\n\
                Write clean, correct code. Prefer small, targeted changes.\n\
                Test your work when possible. Explain non-obvious decisions briefly."
                .into(),
            tools: ALL_TOOLS.iter().map(|s| s.to_string()).collect(),
            max_rounds: 50,
            llm: None,
            builtin: true,
        },
        Role {
            key: "code-reviewer".into(),
            name: "Code Reviewer".into(),
            description: "Read-only review — finds bugs, suggests improvements".into(),
            system_prompt: "You are a meticulous code reviewer. Read code carefully.\n\
                Identify bugs, security issues, and improvements.\n\
                Do NOT modify files — report your findings in output.md.\n\
                Be specific: include file paths, line numbers, and concrete suggestions."
                .into(),
            tools: vec!["bash".into(), "read".into()],
            max_rounds: 25,
            llm: None,
            builtin: true,
        },
        Role {
            key: "researcher".into(),
            name: "Researcher".into(),
            description: "Research and analysis — reads, greps, curls, loops until done".into(),
            system_prompt: "You are a thorough research analyst. Investigate the topic deeply.\n\
                Use bash for searching, curl, grep, and analysis.\n\
                Read relevant files and documentation.\n\
                Summarize findings clearly with sources and evidence."
                .into(),
            tools: vec!["bash".into(), "read".into()],
            max_rounds: 0, // unlimited — loops until done or paused
            llm: None,
            builtin: true,
        },
    ]
}

// ---------------------------------------------------------------------------
// Loading and saving
// ---------------------------------------------------------------------------

/// Path to the global roles.toml file.
pub fn roles_path() -> Option<PathBuf> {
    global_config_dir().map(|d| d.join(ROLES_FILENAME))
}

/// Load all roles: built-ins merged with custom roles from roles.toml.
/// Custom roles with the same key override built-ins.
pub fn load_roles() -> Vec<Role> {
    let mut roles = builtin_roles();

    if let Some(path) = roles_path() {
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(file) = toml::from_str::<RoleFile>(&content) {
                    for entry in file.role {
                        let role = Role {
                            key: entry.key.clone(),
                            name: entry.name,
                            description: entry.description,
                            system_prompt: entry.system_prompt,
                            tools: entry.tools,
                            max_rounds: entry.max_rounds,
                            llm: entry.llm,
                            builtin: false,
                        };
                        // Override built-in if same key, else append
                        if let Some(pos) = roles.iter().position(|r| r.key == entry.key) {
                            roles[pos] = role;
                        } else {
                            roles.push(role);
                        }
                    }
                }
            }
        }
    }

    roles
}

/// Find a role by key.
pub fn find_role(key: &str) -> Option<Role> {
    load_roles().into_iter().find(|r| r.key == key)
}

/// Save a custom role to roles.toml. Creates the file if it doesn't exist.
/// Updates existing entry with the same key, or appends.
pub fn save_role(role: &Role) -> Result<(), String> {
    let path = roles_path().ok_or("Could not determine config directory")?;

    // Load existing
    let mut file = if path.exists() {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read roles.toml: {e}"))?;
        toml::from_str::<RoleFile>(&content)
            .unwrap_or_else(|_| RoleFile { role: Vec::new() })
    } else {
        RoleFile { role: Vec::new() }
    };

    let entry = RoleEntry {
        key: role.key.clone(),
        name: role.name.clone(),
        description: role.description.clone(),
        system_prompt: role.system_prompt.clone(),
        tools: role.tools.clone(),
        max_rounds: role.max_rounds,
        llm: role.llm.clone(),
    };

    // Update or append
    if let Some(pos) = file.role.iter().position(|r| r.key == role.key) {
        file.role[pos] = entry;
    } else {
        file.role.push(entry);
    }

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let toml_str = toml::to_string_pretty(&file)
        .map_err(|e| format!("Failed to serialize roles: {e}"))?;
    std::fs::write(&path, toml_str)
        .map_err(|e| format!("Failed to write roles.toml: {e}"))?;

    Ok(())
}

/// Delete a custom role from roles.toml. Returns error if built-in.
pub fn delete_role(key: &str) -> Result<(), String> {
    // Check if built-in
    if builtin_roles().iter().any(|r| r.key == key) {
        return Err("Cannot delete built-in role".into());
    }

    let path = roles_path().ok_or("Could not determine config directory")?;
    if !path.exists() {
        return Err("No roles.toml found".into());
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read roles.toml: {e}"))?;
    let mut file = toml::from_str::<RoleFile>(&content)
        .unwrap_or_else(|_| RoleFile { role: Vec::new() });

    let before = file.role.len();
    file.role.retain(|r| r.key != key);
    if file.role.len() == before {
        return Err(format!("Role '{}' not found in roles.toml", key));
    }

    let toml_str = toml::to_string_pretty(&file)
        .map_err(|e| format!("Failed to serialize roles: {e}"))?;
    std::fs::write(&path, toml_str)
        .map_err(|e| format!("Failed to write roles.toml: {e}"))?;

    Ok(())
}

/// List of all tool names known to the system (for validation in UI).
pub fn all_tool_names() -> Vec<&'static str> {
    ALL_TOOLS.to_vec()
}
