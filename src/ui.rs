//! Full-screen TUI renderer — AppState, layout, render.

use crate::config::{self, ModelInfo, VERSION};
use crate::markdown::MarkdownRenderer;
use crate::term::*;
use std::fmt::Write;
use std::io::{self, Write as IoWrite};
use std::sync::Mutex;
use std::time::Instant;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ---------------------------------------------------------------------------
// Chat line types
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum ChatLineKind {
    UserMessage,
    AgentText,
    AgentStreaming,
    ToolAction,
    SystemInfo,
    QueuedMessage,
    Error,
}

#[derive(Clone)]
pub struct ChatLine {
    pub kind: ChatLineKind,
    pub content: String,
}

// ---------------------------------------------------------------------------
// Slash commands
// ---------------------------------------------------------------------------

pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "List available commands"),
    ("/model", "Show or switch model"),
    ("/tokens", "Show current token count"),
    ("/compact", "Force context compaction"),
    ("/clear", "Clear conversation history"),
    ("/exit", "Exit stray"),
];

pub static AVAILABLE_MODELS: Mutex<Vec<ModelInfo>> = Mutex::new(Vec::new());

pub fn update_model_cache(api_url: &str, api_key: &str) {
    let models = config::fetch_models(api_url, api_key);
    if let Ok(mut guard) = AVAILABLE_MODELS.lock() {
        *guard = models;
    }
}

pub fn get_model_info(key: &str) -> Option<ModelInfo> {
    AVAILABLE_MODELS.lock().ok()?.iter().find(|m| m.key == key).cloned()
}

pub fn slash_suggestion(buf: &[char]) -> String {
    if buf.is_empty() || buf[0] != '/' || buf.len() < 2 {
        return String::new();
    }
    let text: String = buf.iter().collect();
    if let Some(space_pos) = text.find(' ') {
        let cmd = &text[..space_pos];
        let param = &text[space_pos + 1..];
        return param_suggestion(cmd, param);
    }
    for (cmd, _) in SLASH_COMMANDS {
        if cmd.starts_with(&text) && cmd.len() > text.len() {
            return cmd[text.len()..].to_string();
        }
    }
    String::new()
}

fn param_suggestion(cmd: &str, partial: &str) -> String {
    if partial.is_empty() {
        return String::new();
    }
    match cmd {
        "/model" => {
            if let Ok(models) = AVAILABLE_MODELS.lock() {
                for m in models.iter() {
                    if m.key.starts_with(partial) && m.key.len() > partial.len() {
                        return m.key[partial.len()..].to_string();
                    }
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// App State
// ---------------------------------------------------------------------------

pub struct SpinnerState {
    pub active: bool,
    pub frame: usize,
    pub label: String,
}

pub struct InputState {
    pub buf: Vec<char>,
    pub cursor: usize,
    pub active: bool,
}

pub struct AppState {
    pub width: usize,
    pub height: usize,
    pub header_lines: Vec<String>,
    pub chat: Vec<ChatLine>,
    pub spinner: SpinnerState,
    pub input: InputState,
    pub last_render: Instant,
    render_buf: String,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            width: term_width(),
            height: term_height(),
            header_lines: Vec::new(),
            chat: Vec::new(),
            spinner: SpinnerState { active: false, frame: 0, label: String::new() },
            input: InputState { buf: Vec::new(), cursor: 0, active: false },
            last_render: Instant::now(),
            render_buf: String::with_capacity(16384),
        }
    }

    pub fn update_dimensions(&mut self) {
        self.width = term_width();
        self.height = term_height();
    }

    // -- Chat mutations --

    pub fn push_chat(&mut self, line: ChatLine) {
        self.chat.push(line);
    }

    pub fn begin_streaming(&mut self) {
        self.chat.push(ChatLine {
            kind: ChatLineKind::AgentStreaming,
            content: String::new(),
        });
    }

    pub fn append_streaming(&mut self, delta: &str) {
        if let Some(last) = self.chat.last_mut() {
            if matches!(last.kind, ChatLineKind::AgentStreaming) {
                last.content.push_str(delta);
            }
        }
    }

    pub fn end_streaming(&mut self) {
        if let Some(last) = self.chat.last_mut() {
            if matches!(last.kind, ChatLineKind::AgentStreaming) {
                last.kind = ChatLineKind::AgentText;
            }
        }
    }

    pub fn start_spinner(&mut self, label: &str) {
        self.spinner.active = true;
        self.spinner.frame = 0;
        self.spinner.label = label.to_string();
    }

    pub fn stop_spinner(&mut self) {
        self.spinner.active = false;
    }

    // -- Header --

    pub fn build_header(&mut self, config: &crate::config::Config, tools_str: &str, fmt_str: &str,
                        config_path: &std::path::Path, config_source: &str) {
        self.header_lines.clear();
        self.header_lines.push(String::new());
        self.header_lines.push(format!("  {CYAN}  /\\_/\\    {BOLD}╔═╗╔╦╗╦═╗╔═╗╦ ╦{RESET}"));
        self.header_lines.push(format!("  {CYAN} ( o.o )   {BOLD}╚═╗ ║ ╠╦╝╠═╣╚╦╝{RESET}"));
        self.header_lines.push(format!("  {CYAN}  > ^ <    {BOLD}╚═╝ ╩ ╩╚═╩ ╩ ╩{RESET}"));
        self.header_lines.push(format!("  {DIM}           v{VERSION} · no leash required{RESET}"));
        self.header_lines.push(String::new());
        self.header_lines.push(format!("  {DIM}name{RESET}    {BOLD}{}{RESET}", config.agent.name));
        self.header_lines.push(format!("  {DIM}model{RESET}   {}", config.llm.model));
        self.header_lines.push(format!("  {DIM}tools{RESET}   {tools_str} {DIM}({fmt_str}){RESET}"));
        self.header_lines.push(format!("  {DIM}heart{RESET}   every {}s {DIM}· compact at ~{}k tokens{RESET}",
            config.agent.heartbeat, config.agent.compact_at / 1000));
        self.header_lines.push(format!("  {DIM}config{RESET}  {} {DIM}({config_source}){RESET}", config_path.display()));
        self.header_lines.push(String::new());
        self.header_lines.push(format!("  {DIM}Type a message — or /help for commands{RESET}"));
        self.header_lines.push(String::new());
    }

    // -- Render --

    /// Throttled render — call frequently, only repaints if enough time has passed
    pub fn maybe_render(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_render) >= std::time::Duration::from_millis(16) {
            self.render();
        }
    }

    pub fn render(&mut self) {
        let buf = &mut self.render_buf;
        buf.clear();

        let w = self.width;
        let h = self.height;
        if w == 0 || h == 0 { return; }

        // Hide cursor + clear screen + home (single write_all = no flicker)
        buf.push_str("\x1b[?25l\x1b[2J\x1b[H");

        // --- HEADER ---
        let header_h = self.header_lines.len();
        for line in &self.header_lines {
            write_padded_line(buf, line, w);
        }

        // --- INPUT BOX dimensions ---
        let input_h = if self.input.active {
            input_box_height(&self.input.buf, self.input.cursor, w)
        } else {
            0
        };

        // --- CHAT ZONE ---
        let chat_h = h.saturating_sub(header_h + input_h);

        // Build all chat rows (wrap each ChatLine to terminal width)
        let mut all_rows: Vec<String> = Vec::new();
        let mut md = MarkdownRenderer::new();
        let mut md_out = String::new();
        for line in &self.chat {
            let rows = wrap_chat_line(line, w, &mut md, &mut md_out);
            // Filter empty rows that can sneak in from trailing newlines
            for row in rows {
                if !row.trim().is_empty() || strip_ansi_width(&row) > 0 {
                    all_rows.push(row);
                }
            }
        }

        // Spinner row
        if self.spinner.active {
            let frame = SPINNER_FRAMES[self.spinner.frame % SPINNER_FRAMES.len()];
            all_rows.push(format!("{DIM}  {frame} {}{RESET}", self.spinner.label));
        }

        // Auto-scroll: show last chat_h rows
        let start = all_rows.len().saturating_sub(chat_h);
        let visible = &all_rows[start..];

        for row in visible.iter().take(chat_h) {
            write_padded_line(buf, row, w);
        }
        // Blank-fill remaining
        for _ in visible.len()..chat_h {
            write_padded_line(buf, "", w);
        }

        // Clear any remnants below chat zone (before input box)
        buf.push_str("\x1b[J");

        // --- INPUT BOX ---
        if self.input.active {
            render_input_box(buf, &self.input, w, input_h);
        }

        // Position cursor in input box
        if self.input.active {
            let indent = 2;
            let inner = w.saturating_sub(indent);
            // Calculate cursor line and column using display width
            let cursor_chars = &self.input.buf[..self.input.cursor.min(self.input.buf.len())];
            let cursor_display_pos = chars_display_width(cursor_chars);
            let cursor_line = if inner > 0 { cursor_display_pos / inner } else { 0 };
            let cursor_col = if inner > 0 { cursor_display_pos % inner } else { 0 } + indent + 1;
            let input_start_row = header_h + chat_h + 1; // +1 for top border
            let cursor_row = input_start_row + cursor_line + 1; // +1 for 1-indexed
            let _ = write!(buf, "\x1b[{cursor_row};{cursor_col}H");
            buf.push_str("\x1b[?25h"); // show cursor
        }

        // Single write
        let stdout = io::stdout();
        let mut lock = stdout.lock();
        let _ = lock.write_all(buf.as_bytes());
        let _ = lock.flush();

        self.last_render = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

fn input_box_height(buf: &[char], cursor: usize, width: usize) -> usize {
    let inner = width.saturating_sub(2);
    if inner == 0 { return 3; }
    // Use display width (emoji = 2 cols) for accurate line count
    let buf_width = chars_display_width(buf);
    let cursor_width = chars_display_width(&buf[..cursor.min(buf.len())]) + 1; // +1 for cursor itself
    let effective_width = buf_width.max(cursor_width);
    let content_lines = (effective_width + inner - 1) / inner;
    content_lines.max(1) + 2
}

fn render_input_box(out: &mut String, input: &InputState, width: usize, expected_height: usize) {
    let indent = 2;
    let inner = width.saturating_sub(indent);
    let suggestion = slash_suggestion(&input.buf);
    let sugg_chars: Vec<char> = suggestion.chars().collect();
    let all_chars: Vec<char> = input.buf.iter().chain(sugg_chars.iter()).cloned().collect();
    let buf_len = input.buf.len();

    // Wrap into visual lines by display width (not char count)
    let mut lines: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0;
    while pos < all_chars.len() {
        let mut line_width = 0;
        let mut end = pos;
        while end < all_chars.len() {
            let cw = char_display_width(all_chars[end]);
            if line_width + cw > inner { break; }
            line_width += cw;
            end += 1;
        }
        if end == pos && pos < all_chars.len() { end = pos + 1; } // ensure progress
        lines.push((pos, end));
        pos = end;
    }
    if lines.is_empty() {
        lines.push((0, 0));
    }

    // Top border
    let _ = write!(out, "{DIM}");
    for _ in 0..width { out.push('─'); }
    let _ = write!(out, "{RESET}\r\n");

    // Content lines
    for (i, &(start, end)) in lines.iter().enumerate() {
        let prefix = if i == 0 { format!("{BOLD}{CYAN}❯{RESET} ") } else { "  ".to_string() };
        if start >= buf_len {
            let s: String = all_chars[start..end].iter().collect();
            let _ = write!(out, "{prefix}{DIM}{s}{RESET}");
        } else if end > buf_len {
            let real: String = all_chars[start..buf_len].iter().collect();
            let sugg: String = all_chars[buf_len..end].iter().collect();
            let _ = write!(out, "{prefix}{real}{DIM}{sugg}{RESET}");
        } else {
            let s: String = all_chars[start..end].iter().collect();
            let _ = write!(out, "{prefix}{s}");
        }
        // Pad to width using display width and newline
        let vis_len = indent + chars_display_width(&all_chars[start..end]);
        for _ in vis_len..width { out.push(' '); }
        out.push_str("\r\n");
    }

    // Fill empty content lines if cursor extends past content (cursor on new empty line)
    let expected_content = expected_height.saturating_sub(2); // minus borders
    for _ in lines.len()..expected_content {
        let prefix = "  "; // continuation indent
        let _ = write!(out, "{prefix}");
        for _ in indent..width { out.push(' '); }
        out.push_str("\r\n");
    }

    // Bottom border
    let _ = write!(out, "{DIM}");
    for _ in 0..width { out.push('─'); }
    let _ = write!(out, "{RESET}");
}

fn wrap_chat_line(line: &ChatLine, width: usize, md: &mut MarkdownRenderer, md_out: &mut String) -> Vec<String> {
    match line.kind {
        ChatLineKind::UserMessage => {
            let prefix = format!("  {BOLD}{CYAN}❯{RESET} {CYAN}");
            wrap_styled_with_prefix(&prefix, "    ", &format!("{}{RESET}", line.content), width)
        }
        ChatLineKind::AgentText | ChatLineKind::AgentStreaming => {
            // Render markdown into styled text
            *md = MarkdownRenderer::new();
            md_out.clear();
            // Strip leading/trailing whitespace from raw content before rendering
            let clean = line.content.trim();
            if clean.is_empty() {
                return vec![format!("{BOLD}●{RESET} ")];
            }
            md.feed(clean, md_out);
            md.flush(md_out);
            let trimmed = md_out.trim();

            let prefix = format!("  {BOLD}●{RESET} ");
            wrap_styled_with_prefix(&prefix, "    ", trimmed, width)
        }
        ChatLineKind::ToolAction => {
            vec![format!("  {DIM}{}{RESET}", line.content)]
        }
        ChatLineKind::SystemInfo => {
            // SystemInfo content may already contain ANSI codes (e.g. from /help, /model)
            let mut rows = Vec::new();
            for l in line.content.lines() {
                rows.push(format!("  {DIM}{l}{RESET}"));
            }
            if rows.is_empty() {
                rows.push(format!("  {DIM}{}{RESET}", line.content));
            }
            rows
        }
        ChatLineKind::QueuedMessage => {
            let prefix = format!("  {DIM}❯ ");
            wrap_styled_with_prefix(&prefix, "    ", &format!("{}{RESET}", line.content), width)
        }
        ChatLineKind::Error => {
            vec![format!("  {RED}{}{RESET}", line.content)]
        }
    }
}

/// Wrap ANSI-styled text — can't split mid-escape-sequence
fn wrap_styled_with_prefix(first_prefix: &str, cont_prefix: &str, styled: &str, width: usize) -> Vec<String> {
    let fp_vis = strip_ansi_width(first_prefix);
    let _cp_vis = strip_ansi_width(cont_prefix);
    let inner = width.saturating_sub(fp_vis);
    if inner == 0 { return vec![format!("{first_prefix}{styled}")]; }

    // Simple approach: split by visible character count, preserving ANSI sequences
    let mut rows: Vec<String> = Vec::new();
    let mut current_row = String::new();
    let mut vis_count = 0;
    let mut in_escape = false;
    let max = inner; // first line width

    for ch in styled.chars() {
        if in_escape {
            current_row.push(ch);
            if ch.is_ascii_alphabetic() { in_escape = false; }
            continue;
        }
        if ch == '\x1b' {
            current_row.push(ch);
            in_escape = true;
            continue;
        }
        if ch == '\n' {
            rows.push(current_row);
            current_row = String::new();
            vis_count = 0;
            continue;
        }
        if vis_count >= max {
            rows.push(current_row);
            current_row = String::new();
            vis_count = 0;
        }
        current_row.push(ch);
        vis_count += 1;
    }
    if !current_row.is_empty() || rows.is_empty() {
        rows.push(current_row);
    }

    // Prepend prefixes
    for (i, row) in rows.iter_mut().enumerate() {
        let prefix = if i == 0 { first_prefix } else { cont_prefix };
        *row = format!("{prefix}{row}");
    }

    rows
}

fn write_padded_line(buf: &mut String, content: &str, width: usize) {
    // Strip any embedded newlines — each padded line must be exactly 1 terminal row
    for ch in content.chars() {
        if ch != '\n' && ch != '\r' {
            buf.push(ch);
        }
    }
    let vis = strip_ansi_width(content);
    if vis < width {
        for _ in 0..(width - vis) { buf.push(' '); }
    }
    buf.push_str("\r\n");
}

/// Display width of a single character (1 for most, 2 for wide/emoji)
pub fn char_display_width(ch: char) -> usize {
    let cp = ch as u32;
    // Wide characters: CJK, emoji, fullwidth forms
    if (0x1100..=0x115F).contains(&cp)     // Hangul Jamo
        || (0x2E80..=0x303E).contains(&cp) // CJK Radicals
        || (0x3040..=0x33BF).contains(&cp) // Hiragana, Katakana, CJK
        || (0x3400..=0x4DBF).contains(&cp) // CJK Extension A
        || (0x4E00..=0x9FFF).contains(&cp) // CJK Unified
        || (0xA000..=0xA4CF).contains(&cp) // Yi
        || (0xAC00..=0xD7AF).contains(&cp) // Hangul Syllables
        || (0xF900..=0xFAFF).contains(&cp) // CJK Compatibility
        || (0xFE30..=0xFE6F).contains(&cp) // CJK Forms
        || (0xFF01..=0xFF60).contains(&cp) // Fullwidth Forms
        || (0xFFE0..=0xFFE6).contains(&cp) // Fullwidth Signs
        || (0x1F000..=0x1FBFF).contains(&cp) // Emoji & symbols
        || (0x20000..=0x2FA1F).contains(&cp) // CJK Extension B+
    {
        2
    } else {
        1
    }
}

/// Display width of a slice of chars
pub fn chars_display_width(chars: &[char]) -> usize {
    chars.iter().map(|&c| char_display_width(c)).sum()
}

/// Count the visible display width of a string (strip ANSI escape codes, handle wide chars)
pub fn strip_ansi_width(s: &str) -> usize {
    let mut count = 0;
    let mut in_escape = false;
    for ch in s.chars() {
        if in_escape {
            if ch.is_ascii_alphabetic() { in_escape = false; }
        } else if ch == '\x1b' {
            in_escape = true;
        } else if ch >= ' ' {
            count += char_display_width(ch);
        }
    }
    count
}
