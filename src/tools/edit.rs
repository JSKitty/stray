use super::Tool;
use std::fs;
use std::path::Path;

pub struct EditTool;

impl Tool for EditTool {
    fn name(&self) -> &str { "edit" }

    fn description(&self) -> &str {
        "Edit a file by replacing text. Format: first line is the file path, then ---, then the exact text to find, then ---, then the replacement text. The search text must be unique in the file (appear exactly once)."
    }

    fn tag(&self) -> &str { "edit" }

    fn usage_hint(&self) -> &str {
        "/path/to/file\n---\ntext to find\n---\nreplacement text"
    }

    fn display_action(&self, input: &str) -> String {
        let path = input.lines().next().unwrap_or("").trim();
        super::truncate_middle(path, 60)
    }

    fn execute(&self, input: &str) -> String {
        // Parse: filepath\n---\nold\n---\nnew
        let Some((path_str, rest)) = input.split_once("\n---\n") else {
            return "[error] Invalid format. Expected: filepath\\n---\\nold_text\\n---\\nnew_text".into();
        };
        let path_str = path_str.trim();

        let Some((old_text, new_text)) = rest.split_once("\n---\n").or_else(|| rest.split_once("\n---")) else {
            return "[error] Invalid format. Missing second --- delimiter between old and new text.".into();
        };

        // Block sensitive paths
        const BLOCKED_PATHS: &[&str] = &[
            "/seed", "/secret", "/mnemonic", "wallet.json", "secret.key",
            "/etc/shadow", ".ssh/", "id_rsa", "id_ed25519",
        ];
        let path_lower = path_str.to_lowercase();
        for pattern in BLOCKED_PATHS {
            if path_lower.contains(pattern) {
                return format!("[BLOCKED] Editing this path is restricted: {pattern}");
            }
        }

        let path = Path::new(path_str);
        if !path.exists() {
            return format!("[error] File not found: {path_str}");
        }

        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return format!("[error] Failed to read: {e}"),
        };

        // Check uniqueness
        let count = content.matches(old_text).count();
        if count == 0 {
            return "[error] Search text not found in file. Make sure it matches exactly (including whitespace and indentation).".into();
        }
        if count > 1 {
            return format!("[error] Search text found {count} times — must be unique. Add more surrounding context to disambiguate.");
        }

        // Perform replacement
        let new_content = content.replacen(old_text, new_text, 1);

        match fs::write(path, &new_content) {
            Ok(()) => {
                let old_lines = old_text.lines().count();
                let new_lines = new_text.lines().count();
                format!("[edited {path_str}: replaced {old_lines} lines with {new_lines} lines]")
            }
            Err(e) => format!("[error] Failed to write: {e}"),
        }
    }
}
