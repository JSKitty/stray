mod bash;
pub mod read;

pub use bash::BashTool;
pub use read::ReadTool;

/// A tool the agent can invoke via XML tags in its output.
pub trait Tool {
    /// Tool name shown to the agent and in logs
    fn name(&self) -> &str;
    /// Short description of what the tool does
    fn description(&self) -> &str;
    /// XML tag used to invoke this tool (e.g. "bash" for <bash>...</bash>)
    fn tag(&self) -> &str;
    /// Example shown in the system prompt (what goes between the tags)
    fn usage_hint(&self) -> &str;
    /// Execute the tool with the given input, return output string
    fn execute(&self, input: &str) -> String;
    /// Human-readable action description for the TUI (e.g. "Running ls -la...")
    fn display_action(&self, input: &str) -> String {
        format!("{}", input)
    }
    /// Spawn as a background process for non-blocking execution with cancel support.
    /// Returns None for tools that complete synchronously (default).
    fn spawn(&self, _input: &str) -> Option<Result<std::process::Child, String>> { None }
    /// Format the output of a completed child process.
    fn format_output(&self, output: &std::process::Output) -> String {
        String::from_utf8_lossy(&output.stdout).to_string()
    }
    /// Timeout for spawned processes (default 30s).
    fn timeout(&self) -> std::time::Duration { std::time::Duration::from_secs(30) }
}

/// Truncate a string in the middle with "..." if it exceeds max_len (char-safe)
pub fn truncate_middle(s: &str, max_len: usize) -> String {
    let char_count: usize = s.chars().count();
    if char_count <= max_len {
        return s.to_string();
    }
    if max_len < 5 {
        return s.chars().take(max_len).collect();
    }
    let left_count = (max_len - 3) / 2;
    let right_count = max_len - 3 - left_count;
    let left: String = s.chars().take(left_count).collect();
    let right: String = s.chars().rev().take(right_count).collect::<Vec<_>>().into_iter().rev().collect();
    format!("{left}...{right}")
}

/// Registry of available tools
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn add(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn tools(&self) -> &[Box<dyn Tool>] {
        &self.tools
    }

    /// Get all tool tags (used by thinking stripper to know what NOT to execute from thoughts)
    pub fn tags(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.tag()).collect()
    }
}
