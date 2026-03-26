use super::Tool;
use std::fs;
use std::path::Path;

pub struct WriteTool;

impl Tool for WriteTool {
    fn name(&self) -> &str { "write" }

    fn description(&self) -> &str {
        "Create or overwrite a file. Format: first line is the file path, second line is ---, the rest is the file content."
    }

    fn tag(&self) -> &str { "write" }

    fn usage_hint(&self) -> &str {
        "/path/to/file\n---\nfile content here"
    }

    fn display_action(&self, input: &str) -> String {
        let path = input.lines().next().unwrap_or("").trim();
        super::truncate_middle(path, 60)
    }

    fn execute(&self, input: &str) -> String {
        let Some((path_str, content)) = input.split_once("\n---\n").or_else(|| input.split_once("\n---")) else {
            return "[error] Invalid format. Expected: filepath\\n---\\ncontent".into();
        };
        let path_str = path_str.trim();

        // Block sensitive paths
        const BLOCKED_PATHS: &[&str] = &[
            "/seed", "/secret", "/mnemonic", "wallet.json", "secret.key",
            "/etc/shadow", ".ssh/", "id_rsa", "id_ed25519",
        ];
        let path_lower = path_str.to_lowercase();
        for pattern in BLOCKED_PATHS {
            if path_lower.contains(pattern) {
                return format!("[BLOCKED] Writing to this path is restricted: {pattern}");
            }
        }

        let path = Path::new(path_str);

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return format!("[error] Failed to create directories: {e}");
                }
            }
        }

        match fs::write(path, content) {
            Ok(()) => {
                let lines = content.lines().count();
                let bytes = content.len();
                format!("[wrote {lines} lines, {bytes} bytes to {path_str}]")
            }
            Err(e) => format!("[error] Failed to write: {e}"),
        }
    }
}
