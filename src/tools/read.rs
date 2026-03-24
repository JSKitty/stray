use super::Tool;
use std::fs;
use std::path::Path;

const MAX_TEXT_SIZE: usize = 32768;

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"];

/// Marker prefix for image results — the harness detects this and converts
/// to a vision message before sending to the LLM.
pub const IMAGE_MARKER: &str = "[IMAGE:";

pub struct ReadTool {
    vision_enabled: bool,
}

impl ReadTool {
    pub fn new(vision_enabled: bool) -> Self {
        Self { vision_enabled }
    }
}

impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        if self.vision_enabled {
            "Read a text file or view an image. Supports text files (txt, json, log, csv, toml, rs, py, etc.) and images (png, jpg, gif, webp)."
        } else {
            "Read a text file. Supports txt, json, log, csv, toml, rs, py, and other text formats."
        }
    }

    fn tag(&self) -> &str {
        "read"
    }

    fn usage_hint(&self) -> &str {
        "/path/to/file"
    }

    fn display_action(&self, input: &str) -> String {
        let path = input.trim();
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let is_image = IMAGE_EXTENSIONS.contains(&ext.as_str());
        let truncated = super::truncate_middle(path, 50);
        if is_image {
            format!("Viewing {}", truncated)
        } else {
            format!("Reading {}", truncated)
        }
    }

    fn execute(&self, input: &str) -> String {
        let path_str = input.trim().to_lowercase();

        // Block sensitive paths (same spirit as bash blocklist)
        const BLOCKED_PATHS: &[&str] = &[
            "/seed", "/secret", "/mnemonic", "wallet.json", "secret.key",
            "/etc/shadow", ".ssh/", "id_rsa", "id_ed25519",
            ".env",
        ];
        for pattern in BLOCKED_PATHS {
            if path_str.contains(pattern) {
                return format!("[BLOCKED] Reading this path is restricted: {}", pattern);
            }
        }

        let path = Path::new(input.trim());

        if !path.exists() {
            return format!("[error] File not found: {}", input);
        }

        if !path.is_file() {
            return format!("[error] Not a file: {}", input);
        }

        // Check if it's an image
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
            if !self.vision_enabled {
                return "[error] Image viewing is not supported by the current model.".into();
            }

            // Read and base64 encode
            let bytes = match fs::read(path) {
                Ok(b) => b,
                Err(e) => return format!("[error] Failed to read file: {}", e),
            };

            // Cap image size (10MB)
            if bytes.len() > 10_485_760 {
                return "[error] Image too large (max 10MB)".into();
            }

            let mime = match ext.as_str() {
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                "gif" => "image/gif",
                "webp" => "image/webp",
                "bmp" => "image/bmp",
                "svg" => "image/svg+xml",
                _ => "application/octet-stream",
            };

            // Use base64 encoding (no extra dep — simple table-based encoder)
            let b64 = base64_encode(&bytes);

            // Return special marker that the harness will convert to a vision message
            return format!("{}data:{};base64,{}]", IMAGE_MARKER, mime, b64);
        }

        // Text file
        match fs::read_to_string(path) {
            Ok(content) => {
                if content.len() > MAX_TEXT_SIZE {
                    // Find a safe char boundary at or before MAX_TEXT_SIZE
                    let mut end = MAX_TEXT_SIZE;
                    while end > 0 && !content.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!(
                        "{}\n[truncated at {} bytes, file is {} bytes total]",
                        &content[..end],
                        end,
                        content.len()
                    )
                } else {
                    content
                }
            }
            Err(_) => "[error] File is not valid UTF-8 text (binary file?)".into(),
        }
    }
}

/// Simple base64 encoder — no deps needed
fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    let chunks = data.chunks(3);

    for chunk in chunks {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };

        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        result.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);

        if chunk.len() > 1 {
            result.push(TABLE[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(TABLE[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }

    result
}
