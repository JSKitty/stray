use super::Tool;
use std::process::Command;
use std::time::{Duration, Instant};

const MAX_OUTPUT: usize = 8192;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// Patterns that are blocked outright — matched against the normalized command
const BLOCKED_PATTERNS: &[&str] = &[
    // Destructive filesystem operations
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/",
    "rm -rf .",
    "rm -rf ..",
    "rm -rf $",      // variable expansion into rm -rf
    "rm -fr /",
    "rm -fr /*",
    "rmdir /",
    // Disk/filesystem destruction
    "mkfs",
    "dd if=",
    "format c:",
    "> /dev/sd",
    // Fork bombs and resource exhaustion
    ":(){ :",        // bash fork bomb
    "fork bomb",
    // System shutdown/reboot
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "halt",
    "poweroff",
    // Private key exfiltration
    "cat /etc/shadow",
    "cat /etc/passwd",
    // Network exfiltration of secrets
    "/seed",         // prevent cat wallet seed + curl
    "/secret",
    "/mnemonic",
    "wallet.json",   // prevent reading wallet file directly
    "secret.key",
];

/// Commands/prefixes that are always suspicious
const BLOCKED_PREFIXES: &[&str] = &[
    "chmod 777",
    "chown root",
    "iptables -F",   // flush firewall
    "ufw disable",
];

pub struct BashTool;

impl BashTool {
    fn check_blocked(input: &str) -> Option<String> {
        // Normalize: lowercase, collapse whitespace, normalize semicolons/pipes
        let normalized = input
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        // Check each command in a chain (split by ;, &&, ||, |)
        let sub_commands: Vec<&str> = normalized
            .split(|c| c == ';' || c == '|' || c == '&')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        // Check the full command and each sub-command in chains
        for cmd in std::iter::once(normalized.as_str()).chain(sub_commands.into_iter()) {
            for pattern in BLOCKED_PATTERNS {
                if cmd.contains(&pattern.to_lowercase()) {
                    return Some(format!(
                        "[BLOCKED] Command rejected — matches dangerous pattern: {}",
                        pattern
                    ));
                }
            }

            for prefix in BLOCKED_PREFIXES {
                if cmd.starts_with(&prefix.to_lowercase()) {
                    return Some(format!(
                        "[BLOCKED] Command rejected — matches blocked prefix: {}",
                        prefix
                    ));
                }
            }
        }

        None
    }
}

impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Run a shell command on this machine. Returns stdout, stderr, and exit code. Destructive commands are blocked."
    }

    fn tag(&self) -> &str {
        "bash"
    }

    fn usage_hint(&self) -> &str {
        "your command here"
    }

    fn display_action(&self, input: &str) -> String {
        let cmd = super::truncate_middle(input.trim(), 60);
        format!("Running {}", cmd)
    }

    fn execute(&self, input: &str) -> String {
        // Check blocklist before execution
        if let Some(reason) = Self::check_blocked(input) {
            return reason;
        }

        let mut child = match Command::new("bash")
            .arg("-c")
            .arg(input)
            .stdin(std::process::Stdio::null()) // no interactive input
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return format!("[error] Failed to spawn: {}", e),
        };

        let start = Instant::now();
        let output = loop {
            match child.try_wait() {
                Ok(Some(_)) => break child.wait_with_output(),
                Ok(None) => {
                    if start.elapsed() >= COMMAND_TIMEOUT {
                        let _ = child.kill();
                        return format!(
                            "[TIMEOUT] Command killed after {}s",
                            COMMAND_TIMEOUT.as_secs()
                        );
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return format!("[error] {}", e),
            }
        };

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let mut result = String::new();

                if !stdout.is_empty() {
                    result.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str("[stderr] ");
                    result.push_str(&stderr);
                }
                if result.is_empty() {
                    return format!("[exit code {}]", out.status.code().unwrap_or(-1));
                }

                if result.len() > MAX_OUTPUT {
                    let mut end = MAX_OUTPUT;
                    while end > 0 && !result.is_char_boundary(end) {
                        end -= 1;
                    }
                    result.truncate(end);
                    result.push_str("\n[output truncated]");
                }
                result
            }
            Err(e) => format!("[error] {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_rm_rf_root() {
        assert!(BashTool::check_blocked("rm -rf /").is_some());
        assert!(BashTool::check_blocked("rm  -rf  /").is_some());
        assert!(BashTool::check_blocked("RM -RF /").is_some());
        assert!(BashTool::check_blocked("rm -rf /*").is_some());
    }

    #[test]
    fn blocks_fork_bomb() {
        assert!(BashTool::check_blocked(":(){ :|:& };:").is_some());
    }

    #[test]
    fn blocks_wallet_read() {
        assert!(BashTool::check_blocked("cat wallet.json").is_some());
        assert!(BashTool::check_blocked("cat /some/path/wallet.json").is_some());
    }

    #[test]
    fn blocks_chained_dangerous() {
        assert!(BashTool::check_blocked("echo hi && rm -rf /").is_some());
        assert!(BashTool::check_blocked("ls; rm -rf /").is_some());
        assert!(BashTool::check_blocked("cat file | rm -rf /").is_some());
        assert!(BashTool::check_blocked("echo test; shutdown").is_some());
    }

    #[test]
    fn allows_safe_commands() {
        assert!(BashTool::check_blocked("ls -la").is_none());
        assert!(BashTool::check_blocked("pivx-agent-kit balance").is_none());
        assert!(BashTool::check_blocked("uname -a").is_none());
        assert!(BashTool::check_blocked("rm temp_file.txt").is_none());
    }
}
