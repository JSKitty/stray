use super::Tool;
use std::fs;
use std::path::Path;

const DELIMITER: &str = "\n----------\n";

pub struct WriteTool;

impl Tool for WriteTool {
    fn name(&self) -> &str { "write" }

    fn description(&self) -> &str {
        "Create or overwrite a file. Format: first line is the file path, second line is ---------- (10 dashes), the rest is the file content."
    }

    fn tag(&self) -> &str { "write" }

    fn usage_hint(&self) -> &str {
        "/path/to/file\n----------\nfile content here"
    }

    fn display_action(&self, input: &str) -> String {
        let path = input.lines().next().unwrap_or("").trim();
        super::truncate_middle(path, 60)
    }

    fn execute(&self, input: &str) -> String {
        let Some((path_str, content)) = input.split_once(DELIMITER)
            .or_else(|| input.split_once("\n----------"))
        else {
            return "[error] Invalid format. Expected: filepath\\n----------\\ncontent".into();
        };
        let path_str = path_str.trim();

        if content.is_empty() {
            return "[error] No content provided after ---------- separator.".into();
        }

        // Resolve symlinks + normalize before checking blocklist
        let raw_path = Path::new(path_str);
        let resolved = if raw_path.exists() {
            raw_path.canonicalize().unwrap_or_else(|_| raw_path.to_path_buf())
        } else {
            // File doesn't exist yet — canonicalize parent if possible
            raw_path.parent()
                .and_then(|p| p.canonicalize().ok())
                .map(|p| p.join(raw_path.file_name().unwrap_or_default()))
                .unwrap_or_else(|| raw_path.to_path_buf())
        };
        let path_lower = resolved.to_string_lossy().to_lowercase();

        const BLOCKED_PATHS: &[&str] = &[
            "/seed", "/secret", "/mnemonic", "wallet.json", "secret.key",
            "/etc/shadow", ".ssh/", "id_rsa", "id_ed25519",
        ];
        for pattern in BLOCKED_PATHS {
            if path_lower.contains(pattern) {
                return format!("[BLOCKED] Writing to this path is restricted: {pattern}");
            }
        }

        // Create parent directories if needed
        if let Some(parent) = resolved.parent() {
            if !parent.exists() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return format!("[error] Failed to create directories: {e}");
                }
            }
        }

        match fs::write(&resolved, content) {
            Ok(()) => {
                let lines = content.lines().count();
                let bytes = content.len();
                format!("[wrote {lines} lines, {bytes} bytes to {path_str}]")
            }
            Err(e) => format!("[error] Failed to write: {e}"),
        }
    }
}
