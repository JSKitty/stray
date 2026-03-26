//! Configuration loading, resolution, and first-run setup wizard.

use crate::term::*;
use serde::Deserialize;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

const APP_ID: &str = "cat.jskitty.stray";
const CONFIG_FILENAME: &str = "stray.toml";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Model info (rich metadata from LMStudio native API)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ModelInfo {
    pub key: String,
    pub display_name: String,
    pub context_length: usize,
    pub vision: bool,
    pub tool_use: bool,
}

impl ModelInfo {
    /// Format context length for display: 4096 → "4k", 131072 → "128k", 1048576 → "1M"
    pub fn ctx_display(&self) -> String {
        if self.context_length >= 1_000_000 {
            format!("{}M", self.context_length / 1_000_000)
        } else if self.context_length >= 1_000 {
            format!("{}k", self.context_length / 1_000)
        } else {
            format!("{}", self.context_length)
        }
    }

    /// Compact capability tags: "vision · tools", "vision", "tools", or ""
    pub fn caps_display(&self) -> String {
        let mut caps = Vec::new();
        if self.vision { caps.push("vision"); }
        if self.tool_use { caps.push("tools"); }
        caps.join(" · ")
    }
}

/// Fetch models — tries LMStudio native API first, falls back to OpenAI compat
pub fn fetch_models(api_url: &str, api_key: &str) -> Vec<ModelInfo> {
    // Derive base URL from chat/completions URL
    let base = api_url
        .replace("/v1/chat/completions", "")
        .replace("/chat/completions", "");

    // Try LMStudio native API first: /api/v1/models
    let lms_url = format!("{}/api/v1/models", base);
    if let Ok(resp) = ureq::get(&lms_url)
        .set("Authorization", &format!("Bearer {}", api_key))
        .timeout(Duration::from_secs(3))
        .call()
    {
        if let Ok(body) = resp.into_string() {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&body) {
                if let Some(models) = parsed["models"].as_array() {
                    let mut result: Vec<ModelInfo> = models
                        .iter()
                        .filter(|m| m["type"].as_str() != Some("embedding"))
                        .map(|m| {
                            let caps = &m["capabilities"];
                            let native_ctx = m["max_context_length"].as_u64().unwrap_or(4096) as usize;
                            // Prefer loaded instance's configured context (user may limit for vRAM)
                            let loaded_ctx = m["loaded_instances"]
                                .as_array()
                                .and_then(|instances| instances.first())
                                .and_then(|inst| inst["config"]["context_length"].as_u64())
                                .map(|c| c as usize);
                            ModelInfo {
                                key: m["key"].as_str().unwrap_or("").to_string(),
                                display_name: m["display_name"].as_str().unwrap_or("").to_string(),
                                context_length: loaded_ctx.unwrap_or(native_ctx),
                                vision: caps["vision"].as_bool().unwrap_or(false),
                                tool_use: caps["trained_for_tool_use"].as_bool().unwrap_or(false),
                            }
                        })
                        .filter(|m| !m.key.is_empty())
                        .collect();
                    result.sort_by(|a, b| a.display_name.cmp(&b.display_name));
                    if !result.is_empty() {
                        return result;
                    }
                }
            }
        }
    }

    // Fallback: OpenAI-compatible /v1/models
    let oai_url = format!("{}/v1/models", base);
    let resp = ureq::get(&oai_url)
        .set("Authorization", &format!("Bearer {}", api_key))
        .timeout(Duration::from_secs(5))
        .call();
    match resp {
        Ok(r) => {
            let body = r.into_string().unwrap_or_default();
            let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
            let mut result: Vec<ModelInfo> = parsed["data"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| {
                            let id = m["id"].as_str()?.to_string();
                            Some(ModelInfo {
                                display_name: id.clone(),
                                key: id,
                                context_length: 4096,
                                vision: false,
                                tool_use: false,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            result.sort_by(|a, b| a.key.cmp(&b.key));
            result
        }
        Err(_) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct Config {
    pub agent: AgentConfig,
    pub llm: LlmConfig,
}

#[derive(Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub system_prompt: String,
    #[serde(alias = "ping_interval")]
    pub heartbeat: u64,
    #[serde(default = "default_compact_at")]
    pub compact_at: usize,
}

#[derive(Deserialize)]
pub struct LlmConfig {
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

fn default_compact_at() -> usize {
    100_000
}

// ---------------------------------------------------------------------------
// Config resolution
// ---------------------------------------------------------------------------

pub enum ConfigSource {
    Cli,
    Local,
    Global,
    Wizard,
}

pub struct LoadedConfig {
    pub config: Config,
    pub source: ConfigSource,
    pub path: PathBuf,
}

/// Load config: CLI arg → ./stray.toml → global → wizard
pub fn load() -> LoadedConfig {
    // 1. Explicit CLI argument
    if let Some(path) = std::env::args().nth(1) {
        let config = load_from_file(&path);
        return LoadedConfig {
            config,
            source: ConfigSource::Cli,
            path: path.into(),
        };
    }

    // 2. Local: ./stray.toml
    if std::path::Path::new("stray.toml").exists() {
        let config = load_from_file("stray.toml");
        return LoadedConfig {
            config,
            source: ConfigSource::Local,
            path: "stray.toml".into(),
        };
    }

    // 3. Global config
    if let Some(dir) = global_config_dir() {
        let global_path = dir.join(CONFIG_FILENAME);
        if global_path.exists() {
            let path_str = global_path.to_string_lossy().to_string();
            let config = load_from_file(&path_str);
            return LoadedConfig {
                config,
                source: ConfigSource::Global,
                path: global_path,
            };
        }
    }

    // 4. Nothing found — run the setup wizard
    run_setup_wizard()
}

fn load_from_file(path: &str) -> Config {
    let config_str = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", path, e);
        std::process::exit(1);
    });
    let config_str = expand_env_vars(&config_str);
    toml::from_str(&config_str).unwrap_or_else(|e| {
        eprintln!("Failed to parse {}: {}", path, e);
        std::process::exit(1);
    })
}

pub fn expand_env_vars(input: &str) -> String {
    let mut result = input.to_string();
    let mut scan_from = 0;
    while let Some(rel) = result[scan_from..].find("${") {
        let start = scan_from + rel;
        if let Some(end) = result[start..].find('}') {
            let var_name = &result[start + 2..start + end];
            let value = std::env::var(var_name).unwrap_or_default();
            let after = start + end + 1;
            result = format!("{}{}{}", &result[..start], value, &result[after..]);
            scan_from = start + value.len();
        } else {
            break;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Platform data directories
// ---------------------------------------------------------------------------

pub fn global_config_dir() -> Option<PathBuf> {
    global_data_base().map(|base| base.join(APP_ID))
}

#[cfg(target_os = "macos")]
fn global_data_base() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join("Library").join("Application Support"))
}

#[cfg(target_os = "linux")]
fn global_data_base() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg));
        }
    }
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".local").join("share"))
}

#[cfg(target_os = "windows")]
fn global_data_base() -> Option<PathBuf> {
    std::env::var("APPDATA").ok().map(PathBuf::from)
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn global_data_base() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".stray"))
}

// ---------------------------------------------------------------------------
// Setup wizard
// ---------------------------------------------------------------------------

fn probe_lmstudio() -> bool {
    ureq::get("http://127.0.0.1:1234/v1/models")
        .timeout(Duration::from_secs(2))
        .call()
        .is_ok()
}


/// Read a line with a default value. Returns default on empty Enter.
fn wizard_prompt(label: &str, default: &str) -> String {
    let prefix = if default.is_empty() {
        format!("  {label}: ")
    } else {
        format!("  {label} {DIM}[{default}]{RESET}: ")
    };
    print!("{prefix}");
    let _ = io::stdout().flush();

    let mut buf: Vec<char> = Vec::new();
    let mut cursor: usize = 0;

    loop {
        match read_key() {
            Some(Key::Enter) => {
                println!();
                let input: String = buf.iter().collect();
                let trimmed = input.trim().to_string();
                return if trimmed.is_empty() { default.to_string() } else { trimmed };
            }
            Some(Key::Char(ch)) => {
                buf.insert(cursor, ch);
                cursor += 1;
            }
            Some(Key::Backspace) => {
                if cursor > 0 {
                    buf.remove(cursor - 1);
                    cursor -= 1;
                }
            }
            Some(Key::Left) => { if cursor > 0 { cursor -= 1; } }
            Some(Key::Right) => { if cursor < buf.len() { cursor += 1; } }
            Some(Key::Home) => cursor = 0,
            Some(Key::End) => cursor = buf.len(),
            Some(Key::CtrlC) => {
                restore_terminal();
                println!();
                std::process::exit(1);
            }
            _ => continue,
        }
        // Re-render input
        let text: String = buf.iter().collect();
        // prefix contains ANSI codes — compute visible length for cursor positioning
        let vis_prefix = strip_ansi(&prefix);
        let vis_col = vis_prefix.len() + cursor + 1;
        print!("\r{prefix}{text}\x1b[K\x1b[{vis_col}G");
        let _ = io::stdout().flush();
    }
}

/// Strip ANSI escape codes from a string (for measuring visible length)
fn strip_ansi(s: &str) -> String {
    let mut result = String::new();
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            if ch.is_ascii_alphabetic() || ch == 'm' {
                in_escape = false;
            }
        } else if ch == '\x1b' {
            in_escape = true;
        } else {
            result.push(ch);
        }
    }
    result
}

fn run_setup_wizard() -> LoadedConfig {
    let _orig = enable_raw_mode();

    // Splash
    print!("\x1b[2J\x1b[H");
    let _ = io::stdout().flush();
    println!();
    println!("  {CYAN}  /\\_/\\    {BOLD}╔═╗╔╦╗╦═╗╔═╗╦ ╦{RESET}");
    println!("  {CYAN} ( o.o )   {BOLD}╚═╗ ║ ╠╦╝╠═╣╚╦╝{RESET}");
    println!("  {CYAN}  > ^ <    {BOLD}╚═╝ ╩ ╩╚═╩ ╩ ╩{RESET}");
    println!("  {DIM}           v{VERSION} · first run setup{RESET}");
    println!();

    // Probe LMStudio
    print!("  Checking for LMStudio...");
    let _ = io::stdout().flush();
    let lmstudio = probe_lmstudio();
    if lmstudio {
        println!(" {CYAN}found!{RESET}");
    } else {
        println!(" {DIM}not found{RESET}");
    }
    println!();

    // API URL
    let default_url = if lmstudio {
        "http://127.0.0.1:1234/v1/chat/completions"
    } else {
        ""
    };
    let api_url = wizard_prompt("API URL", default_url);

    // API Key
    let default_key = if lmstudio && api_url.contains("127.0.0.1:1234") {
        "lm-studio"
    } else {
        ""
    };
    let api_key = wizard_prompt("API Key", default_key);

    // Fetch models (uses LMStudio native API if available)
    print!("\n  {DIM}Fetching models...{RESET}");
    let _ = io::stdout().flush();
    let models = fetch_models(&api_url, &api_key);
    print!("\r\x1b[2K");
    let _ = io::stdout().flush();

    let (model, vision, context_length) = if models.is_empty() {
        println!("  {YELLOW}Could not fetch models — enter manually{RESET}");
        let m = wizard_prompt("Model", "");
        let v_str = wizard_prompt("Vision support", "false");
        (m, v_str == "true" || v_str == "yes", 100_000)
    } else {
        // Show model picker with rich info
        println!();
        println!("  {DIM}Available models:{RESET}");
        let name_w = models.iter().map(|m| m.display_name.len()).max().unwrap_or(10);
        for (i, m) in models.iter().enumerate() {
            let ctx = m.ctx_display();
            let caps = m.caps_display();
            let caps_str = if caps.is_empty() {
                String::new()
            } else {
                format!("  {DIM}{caps}{RESET}")
            };
            if i == 0 {
                println!("   {CYAN}>{RESET} {DIM}{}){RESET} {BOLD}{:<name_w$}{RESET}  {DIM}{:>5} ctx{RESET}{caps_str}",
                    i + 1, m.display_name, ctx);
            } else {
                println!("     {DIM}{}){RESET} {:<name_w$}  {DIM}{:>5} ctx{RESET}{caps_str}",
                    i + 1, m.display_name, ctx);
            }
        }
        println!();
        let choice = wizard_prompt("Pick", "1");
        let idx = choice.parse::<usize>().unwrap_or(1).saturating_sub(1).min(models.len().saturating_sub(1));
        let selected = &models[idx];

        println!("  {DIM}Auto-detected:{RESET} vision={}{}, context={}",
            if selected.vision { format!("{CYAN}true{RESET}") } else { format!("{DIM}false{RESET}") },
            if selected.tool_use { format!(", tools={CYAN}true{RESET}") } else { String::new() },
            selected.ctx_display());

        (selected.key.clone(), selected.vision, selected.context_length)
    };

    // Compact at ~80% of context length
    let compact_at = (context_length as f64 * 0.8) as usize;

    // Agent name
    let name = wizard_prompt("Agent name", "Stray");

    // Build TOML (escape special chars to prevent injection)
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let toml_content = format!(
        r#"[agent]
name = "{}"
heartbeat = 300
compact_at = {compact_at}
system_prompt = """
You are a helpful autonomous assistant with access to bash and file reading tools.
Be concise. Only run commands when needed.
"""

[llm]
api_url = "{}"
api_key = "{}"
model = "{}"
max_tokens = 4096
vision = {vision}
"#,
        esc(&name), esc(&api_url), esc(&api_key), esc(&model)
    );

    // Write to global config dir
    let config_dir = global_config_dir().unwrap_or_else(|| {
        eprintln!("\n  {RED}Could not determine config directory{RESET}");
        restore_terminal();
        std::process::exit(1);
    });
    let config_path = config_dir.join(CONFIG_FILENAME);

    if let Err(e) = std::fs::create_dir_all(&config_dir) {
        eprintln!("\n  {RED}Failed to create {}: {e}{RESET}", config_dir.display());
        restore_terminal();
        std::process::exit(1);
    }

    print!("\n  Writing to {}...", config_path.display());
    let _ = io::stdout().flush();

    if let Err(e) = std::fs::write(&config_path, &toml_content) {
        eprintln!(" {RED}failed: {e}{RESET}");
        restore_terminal();
        std::process::exit(1);
    }
    println!(" {CYAN}done!{RESET}");
    println!();

    restore_terminal();

    let config: Config = match toml::from_str(&toml_content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  {RED}Generated config is invalid: {e}{RESET}");
            eprintln!("  Please edit {} manually.", config_path.display());
            std::process::exit(1);
        }
    };
    LoadedConfig {
        config,
        source: ConfigSource::Wizard,
        path: config_path,
    }
}
