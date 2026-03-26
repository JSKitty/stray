//! Full-screen TUI renderer — AppState, layout, render.

use crate::config::{self, ModelInfo, VERSION};
use crate::highlight;
use crate::markdown::MarkdownRenderer;
use crate::term::*;
use std::fmt::Write;
use std::io::{self, Write as IoWrite};
use std::sync::Mutex;
use std::time::Instant;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ---------------------------------------------------------------------------
// Cat animation state machine
// ---------------------------------------------------------------------------

const TAGLINES: &[&str] = &[
    "no leash required",
    "born to wander",
    "comes and goes as he pleases",
    "will purr for tokens",
    "catnip not included",
];

pub enum CatAnimKind {
    Boot,
    Squint,
    Talking,
    Idle,
}

pub struct CatAnim {
    pub kind: CatAnimKind,
    pub frame: u8,
}

impl CatAnim {
    pub fn new() -> Self {
        Self { kind: CatAnimKind::Boot, frame: 0 }
    }

    /// Get the current face for this animation state
    pub fn face(&self) -> &'static str {
        match self.kind {
            CatAnimKind::Boot => match self.frame {
                0..=7   => "( -.- )",
                8..=9   => "( o.- )",
                10..=11 => "( -.- )",
                12..=14 => "( o.- )",
                15..=16 => "( o.o )",
                17..=18 => "( o.- )",
                _       => "( o.o )",
            },
            CatAnimKind::Squint => match self.frame {
                0..=8  => "( -.-')",
                9..=10 => "( o.- )",
                _      => "( o.o )",
            },
            CatAnimKind::Talking => match self.frame % 4 {
                0 => "( o.o )",
                1 => "( o_o )",
                2 => "( o.o )",
                _ => "( oᴗo )",
            },
            CatAnimKind::Idle => "( o.o )",
        }
    }

    /// Advance one tick. Returns true if the face changed.
    pub fn tick(&mut self) -> bool {
        match self.kind {
            CatAnimKind::Boot => {
                if self.frame >= 24 {
                    self.kind = CatAnimKind::Idle;
                    return false;
                }
                self.frame += 1;
                true
            }
            CatAnimKind::Talking => {
                self.frame = self.frame.wrapping_add(1);
                true // loops until stopped
            }
            CatAnimKind::Squint => {
                if self.frame >= 12 {
                    self.kind = CatAnimKind::Idle;
                    self.frame = 0;
                    return true; // face changes back to normal
                }
                self.frame += 1;
                true
            }
            CatAnimKind::Idle => false,
        }
    }

    /// Start a new animation, overriding any current one
    pub fn play(&mut self, kind: CatAnimKind) {
        self.kind = kind;
        self.frame = 0;
    }

    /// Whether any animation is active
    pub fn is_active(&self) -> bool {
        !matches!(self.kind, CatAnimKind::Idle)
    }
}

/// Gray shades for streaming fade-in (lightest → darkest, 256-color mode, 1 shade per char)
const FADE_SHADES: &[&str] = &[
    "\x1b[38;5;255m", "\x1b[38;5;254m", "\x1b[38;5;253m",
    "\x1b[38;5;252m", "\x1b[38;5;251m", "\x1b[38;5;250m",
    "\x1b[38;5;249m", "\x1b[38;5;248m", "\x1b[38;5;247m",
    "\x1b[38;5;246m", "\x1b[38;5;245m", "\x1b[38;5;244m",
    "\x1b[38;5;243m", "\x1b[38;5;242m", "\x1b[38;5;241m",
    "\x1b[38;5;240m", "\x1b[38;5;239m", "\x1b[38;5;238m",
];

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
    ("/context", "Show context usage"),
    ("/compact", "Force context compaction"),
    ("/copy", "Copy last response to clipboard"),
    ("/config", "Edit configuration"),
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
    pub frame: u8,
    pub label: String,
}

/// Gap-buffer input — O(1) insert/delete at cursor, O(n) only for full materialization.
pub struct InputState {
    before: Vec<char>,  // chars before cursor
    after: Vec<char>,   // chars after cursor (reversed)
    pub active: bool,
}

impl InputState {
    pub fn new() -> Self {
        Self { before: Vec::new(), after: Vec::new(), active: false }
    }

    pub fn insert(&mut self, ch: char) { self.before.push(ch); }

    pub fn insert_str(&mut self, s: &str) {
        for ch in s.chars() { self.before.push(ch); }
    }

    pub fn backspace(&mut self) -> bool { self.before.pop().is_some() }

    pub fn delete(&mut self) -> bool { self.after.pop().is_some() }

    pub fn move_left(&mut self) -> bool {
        self.before.pop().map(|c| self.after.push(c)).is_some()
    }

    pub fn move_right(&mut self) -> bool {
        self.after.pop().map(|c| self.before.push(c)).is_some()
    }

    pub fn move_home(&mut self) {
        while let Some(c) = self.before.pop() { self.after.push(c); }
    }

    pub fn move_end(&mut self) {
        while let Some(c) = self.after.pop() { self.before.push(c); }
    }

    pub fn cursor(&self) -> usize { self.before.len() }

    pub fn len(&self) -> usize { self.before.len() + self.after.len() }

    pub fn is_empty(&self) -> bool { self.before.is_empty() && self.after.is_empty() }

    pub fn clear(&mut self) { self.before.clear(); self.after.clear(); }

    pub fn drain_all(&mut self) -> String {
        let s: String = self.before.drain(..).chain(self.after.drain(..).rev()).collect();
        s
    }

    /// Chars before cursor — O(1) slice, useful for display-width calculations
    pub fn before_cursor(&self) -> &[char] { &self.before }

    /// Materialize full buffer — O(n), use sparingly (rendering)
    pub fn to_chars(&self) -> Vec<char> {
        self.before.iter().chain(self.after.iter().rev()).cloned().collect()
    }

    /// Move cursor to absolute position — O(k) where k = distance moved
    pub fn set_cursor(&mut self, pos: usize) {
        let pos = pos.min(self.len());
        while self.before.len() > pos {
            if let Some(c) = self.before.pop() { self.after.push(c); } else { break; }
        }
        while self.before.len() < pos {
            if let Some(c) = self.after.pop() { self.before.push(c); } else { break; }
        }
    }
}

// ---------------------------------------------------------------------------
// Below-input selector (modular picker for /model, etc.)
// ---------------------------------------------------------------------------

pub struct SelectorItem {
    pub label: String,
    pub value: String,
}

pub struct Selector {
    pub id: String,
    pub items: Vec<SelectorItem>,
    pub selected: usize,
    pub scroll_offset: usize,
    pub max_visible: usize,
}

impl Selector {
    pub fn new(id: &str, items: Vec<SelectorItem>, max_visible: usize) -> Self {
        let max_visible = max_visible.min(items.len()).max(1);
        Self {
            id: id.to_string(),
            items,
            selected: 0,
            scroll_offset: 0,
            max_visible,
        }
    }

    pub fn move_down(&mut self) {
        if self.selected + 1 < self.items.len() {
            self.selected += 1;
            if self.selected >= self.scroll_offset + self.max_visible {
                self.scroll_offset = self.selected - self.max_visible + 1;
            }
        }
    }

    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            if self.selected < self.scroll_offset {
                self.scroll_offset = self.selected;
            }
        }
    }

    pub fn selected_value(&self) -> Option<&str> {
        self.items.get(self.selected).map(|i| i.value.as_str())
    }

    pub fn height(&self) -> usize {
        self.max_visible.min(self.items.len())
    }
}

pub struct AppState {
    pub width: usize,
    pub height: usize,
    pub header_lines: Vec<String>,
    pub chat: Vec<ChatLine>,
    pub spinner: SpinnerState,
    pub input: InputState,
    pub selector: Option<Selector>,
    pub config_edit: Option<(String, String)>, // (scope, field) — active config text input
    pub input_label: String,                     // shown in input box top border
    pub input_hint: String,                      // description shown below border
    pub ctrlc_hint: bool,                        // whether the Ctrl+C exit hint is showing
    pub fade_chars: Vec<u8>,                      // per-char fade countdown (0 = white)
    pub cat_anim: CatAnim,                       // unified cat animation state
    pub tagline: &'static str,                   // random boot tagline
    pub last_render: Instant,
    render_buf: String,
    // Wrapped row cache — avoids O(n) re-wrapping every frame
    cached_rows: Vec<String>,
    cache_chat_len: usize,
    cache_last_content_len: usize,
    cache_width: usize,
    cache_dirty: bool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            width: term_width(),
            height: term_height(),
            header_lines: Vec::new(),
            chat: Vec::new(),
            spinner: SpinnerState { active: false, frame: 0, label: String::new() },
            input: InputState::new(),
            selector: None,
            config_edit: None,
            input_label: String::new(),
            input_hint: String::new(),
            ctrlc_hint: false,
            fade_chars: Vec::new(),
            cat_anim: CatAnim::new(),
            tagline: TAGLINES[std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default()
                .as_millis() as usize % TAGLINES.len()],
            last_render: Instant::now(),
            render_buf: String::with_capacity(16384),
            cached_rows: Vec::new(),
            cache_chat_len: 0,
            cache_last_content_len: 0,
            cache_width: 0,
            cache_dirty: false,
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
        self.fade_chars.clear();
        self.cat_anim.play(CatAnimKind::Talking);
        self.chat.push(ChatLine {
            kind: ChatLineKind::AgentStreaming,
            content: String::new(),
        });
    }

    pub fn append_streaming(&mut self, delta: &str) {
        for line in self.chat.iter_mut().rev() {
            if matches!(line.kind, ChatLineKind::AgentStreaming) {
                line.content.push_str(delta);
                for _ in delta.chars() {
                    self.fade_chars.push(FADE_SHADES.len() as u8);
                }
                self.cache_dirty = true;
                break;
            }
        }
    }

    pub fn end_streaming(&mut self) {
        // Stop talking, return to idle
        if matches!(self.cat_anim.kind, CatAnimKind::Talking) {
            self.cat_anim.play(CatAnimKind::Idle);
        }
        // Don't clear fade_chars — let remaining chars finish fading naturally
        for line in self.chat.iter_mut().rev() {
            if matches!(line.kind, ChatLineKind::AgentStreaming) {
                line.kind = ChatLineKind::AgentText;
                self.cache_dirty = true;
                break;
            }
        }
    }

    pub fn start_spinner(&mut self, label: &str) {
        self.spinner.active = true;
        self.spinner.frame = 0;
        self.spinner.label = label.to_string();
    }

    /// Advance the streaming fade-in toward fully settled
    /// Tick all per-char fade countdowns. Returns true if anything changed.
    pub fn tick_fade(&mut self) -> bool {
        if self.fade_chars.is_empty() { return false; }
        let mut any_active = false;
        for f in self.fade_chars.iter_mut() {
            if *f > 0 {
                *f = f.saturating_sub(3); // ~480ms full fade (18 shades / 3 per tick / 80ms)
                any_active = true;
            }
        }
        if any_active {
            self.cache_dirty = true;
        } else if !self.chat.iter().any(|l| matches!(l.kind, ChatLineKind::AgentStreaming)) {
            self.fade_chars.clear(); // all settled + no streaming → done
        }
        any_active
    }

    /// Advance the cat animation state machine. Returns true if anything changed.
    pub fn advance_cat_anim(&mut self) -> bool {
        if !self.cat_anim.is_active() { return false; }
        let was_boot = matches!(self.cat_anim.kind, CatAnimKind::Boot);
        let changed = self.cat_anim.tick();
        if changed && self.header_lines.len() > 2 {
            let face = self.cat_anim.face();
            self.header_lines[2] = format!("  {CYAN} {face}   {BOLD}╚═╗ ║ ╠╦╝╠═╣╚╦╝{RESET}");
        }
        // Fade in tagline during boot (starts at frame 10)
        if was_boot && self.cat_anim.frame >= 10 {
            if self.header_lines.len() > 4 {
                let full_tag = format!(" · {}", self.tagline);
                let reveal = (self.cat_anim.frame - 10) as usize * 3;
                let mut tag_display = String::new();
                let mut in_fade = false;
                for (i, ch) in full_tag.chars().enumerate() {
                    if i >= reveal { break; }
                    let dist = reveal.saturating_sub(1) - i;
                    if dist < FADE_SHADES.len() {
                        let shade_idx = FADE_SHADES.len() - 1 - dist;
                        tag_display.push_str(FADE_SHADES[shade_idx]);
                        in_fade = true;
                    } else if in_fade {
                        tag_display.push_str(RESET);
                        tag_display.push_str(DIM);
                        in_fade = false;
                    }
                    tag_display.push(ch);
                }
                self.header_lines[4] = format!("  {DIM}           v{VERSION}{tag_display}{RESET}");
            }
        }
        true
    }

    pub fn stop_spinner(&mut self) {
        self.spinner.active = false;
    }

    // -- Header --

    pub fn build_header(&mut self, config: &crate::config::Config, tools_str: &str, fmt_str: &str,
                        config_path: &std::path::Path, config_source: &str) {
        self.header_lines.clear();
        self.header_lines.push(String::new());
        let face = self.cat_anim.face();
        self.header_lines.push(format!("  {CYAN}  /\\_/\\    {BOLD}╔═╗╔╦╗╦═╗╔═╗╦ ╦{RESET}"));
        self.header_lines.push(format!("  {CYAN} {face}   {BOLD}╚═╗ ║ ╠╦╝╠═╣╚╦╝{RESET}"));
        self.header_lines.push(format!("  {CYAN}  >   <    {BOLD}╚═╝ ╩ ╩╚═╩ ╩ ╩{RESET}"));
        // Tagline — blank during boot animation, faded in by advance_cat_anim
        if self.cat_anim.is_active() {
            self.header_lines.push(format!("  {DIM}           v{VERSION}{RESET}"));
        } else {
            self.header_lines.push(format!("  {DIM}           v{VERSION} · {}{RESET}", self.tagline));
        }
        self.header_lines.push(String::new());
        self.header_lines.push(format!("  {DIM}name{RESET}    {BOLD}{}{RESET}", config.agent.name));
        self.header_lines.push(format!("  {DIM}model{RESET}   {}", config.llm.model));
        self.header_lines.push(format!("  {DIM}tools{RESET}   {tools_str} {DIM}({fmt_str}){RESET}"));
        if config.agent.heartbeat > 0 {
            self.header_lines.push(format!("  {DIM}mode{RESET}    autonomous {DIM}· every {}s · compact at ~{}k tokens{RESET}",
                config.agent.heartbeat, config.agent.compact_at / 1000));
        } else {
            self.header_lines.push(format!("  {DIM}mode{RESET}    interactive {DIM}· compact at ~{}k tokens{RESET}",
                config.agent.compact_at / 1000));
        }
        self.header_lines.push(format!("  {DIM}config{RESET}  {} {DIM}({config_source}){RESET}", config_path.display()));
        self.header_lines.push(String::new());
        self.header_lines.push(format!("  {DIM}Type a message — {RESET}{CYAN}/help{RESET}{DIM} for commands · {RESET}{CYAN}/exit{RESET}{DIM} to quit{RESET}"));
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

        // Hide cursor + home (padded lines overwrite old content, no full-screen clear needed)
        buf.push_str("\x1b[?25l\x1b[H");

        // --- HEADER ---
        let header_h = self.header_lines.len();
        for line in &self.header_lines {
            write_padded_line(buf, line, w);
        }

        // --- INPUT BOX dimensions ---
        // Selector-only mode (no text input): collapse to just the labeled border
        let selector_only = self.selector.is_some() && self.config_edit.is_none();
        let input_h = if self.input.active {
            if selector_only { 1 } else { input_box_height(&self.input, w, !self.input_hint.is_empty()) }
        } else {
            0
        };

        // --- SELECTOR dimensions ---
        let selector_h = self.selector.as_ref().map(|s| s.height()).unwrap_or(0);

        // --- CHAT ZONE ---
        let chat_h = h.saturating_sub(header_h + input_h + selector_h);

        // Build wrapped chat rows (cached — only re-wrap on content/width change)
        let cur_chat_len = self.chat.len();
        let cur_last_len = self.chat.last().map(|l| l.content.len()).unwrap_or(0);
        if cur_chat_len != self.cache_chat_len
            || cur_last_len != self.cache_last_content_len
            || w != self.cache_width
            || self.cache_dirty
        {
            self.cache_dirty = false;
            self.cached_rows.clear();
            let mut md = MarkdownRenderer::new();
            let mut md_out = String::new();
            // Per-char fade: find which line to apply it to
            let fade_data: Option<&[u8]> = if !self.fade_chars.is_empty() { Some(&self.fade_chars) } else { None };
            let fade_line_idx = if fade_data.is_some() {
                self.chat.iter().rposition(|l| matches!(l.kind, ChatLineKind::AgentStreaming))
                    .or_else(|| self.chat.iter().rposition(|l| matches!(l.kind, ChatLineKind::AgentText)))
            } else { None };
            for (idx, line) in self.chat.iter().enumerate() {
                let line_fade = if Some(idx) == fade_line_idx { fade_data } else { None };
                let rows = wrap_chat_line(line, w, &mut md, &mut md_out, line_fade);
                for row in rows {
                    if !row.trim().is_empty() || strip_ansi_width(&row) > 0 {
                        self.cached_rows.push(row);
                    }
                }
            }
            self.cache_chat_len = cur_chat_len;
            self.cache_last_content_len = cur_last_len;
            self.cache_width = w;
        }

        // Spinner row (appended dynamically, not cached)
        let spinner_row = if self.spinner.active {
            let frame = SPINNER_FRAMES[self.spinner.frame as usize];
            Some(format!("{DIM}  {frame} {}{RESET}", self.spinner.label))
        } else {
            None
        };
        let total_rows = self.cached_rows.len() + if spinner_row.is_some() { 1 } else { 0 };

        // Auto-scroll: show last chat_h rows
        let start = total_rows.saturating_sub(chat_h);

        // Render visible rows from cache + optional spinner
        let mut rendered = 0;
        for i in start..total_rows {
            if rendered >= chat_h { break; }
            if i < self.cached_rows.len() {
                write_padded_line(buf, &self.cached_rows[i], w);
            } else if let Some(ref sr) = spinner_row {
                write_padded_line(buf, sr, w);
            }
            rendered += 1;
        }
        // Blank-fill remaining
        for _ in rendered..chat_h {
            write_padded_line(buf, "", w);
        }

        // Clear any remnants below chat zone (before input box)
        buf.push_str("\x1b[J");

        // --- INPUT BOX ---
        if self.input.active {
            if selector_only {
                // Just the labeled top border, no prompt or bottom border
                if self.input_label.is_empty() {
                    let _ = write!(buf, "{DIM}");
                    for _ in 0..w { buf.push('─'); }
                    let _ = write!(buf, "{RESET}\x1b[K\r\n");
                } else {
                    let label_vis = 3 + strip_ansi_width(&self.input_label) + 1;
                    let _ = write!(buf, "{DIM}── {RESET}{CYAN}{}{RESET}{DIM} ", self.input_label);
                    for _ in label_vis..w { buf.push('─'); }
                    let _ = write!(buf, "{RESET}\x1b[K\r\n");
                }
            } else {
                render_input_box(buf, &self.input, w, input_h, selector_h > 0, &self.input_label, &self.input_hint);
            }
        }

        // --- SELECTOR ---
        if let Some(ref selector) = self.selector {
            render_selector(buf, selector, w);
        }

        // Position cursor in input box (hidden when selector is active)
        if self.input.active && self.selector.is_none() {
            let indent = 2;
            let inner = w.saturating_sub(indent);
            // Find cursor's visual line and column using line-break-aware layout
            let mut chars = self.input.to_chars();
            let cursor_pos = self.input.before_cursor().len();
            if cursor_pos >= chars.len() { chars.push(' '); } // ensure cursor has a position
            let lines = visual_line_breaks(&chars, inner);
            let cur_line_idx = lines.iter().position(|(s, e)| cursor_pos >= *s && cursor_pos < *e)
                .unwrap_or(lines.len().saturating_sub(1));
            let line_start = lines.get(cur_line_idx).map(|l| l.0).unwrap_or(0);
            let col_chars = &chars[line_start..cursor_pos];
            let col_width: usize = col_chars.iter().filter(|&&c| c != '\n').map(|&c| char_display_width(c)).sum();
            let cursor_line = cur_line_idx;
            let cursor_col = col_width + indent + 1;
            let hint_lines = if self.input_hint.is_empty() { 0 } else { 1 };
            let input_start_row = header_h + chat_h + 1 + hint_lines; // +1 for top border + hint
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

/// Compute visual line breaks for a char slice, handling both wrapping and \n.
pub fn visual_line_breaks(chars: &[char], inner: usize) -> Vec<(usize, usize)> {
    if inner == 0 { return vec![(0, chars.len())]; }
    let mut lines = Vec::new();
    let mut pos = 0;
    while pos < chars.len() {
        let mut line_width = 0;
        let mut end = pos;
        while end < chars.len() {
            if chars[end] == '\n' { end += 1; break; } // newline forces line break
            let cw = char_display_width(chars[end]);
            if line_width + cw > inner { break; }
            line_width += cw;
            end += 1;
        }
        if end == pos && pos < chars.len() { end = pos + 1; }
        lines.push((pos, end));
        pos = end;
    }
    // Trailing newline implies a new empty line after it
    if !chars.is_empty() && *chars.last().unwrap() == '\n' {
        lines.push((chars.len(), chars.len()));
    }
    if lines.is_empty() { lines.push((0, 0)); }
    lines
}

fn input_box_height(input: &InputState, width: usize, has_hint: bool) -> usize {
    let inner = width.saturating_sub(2);
    if inner == 0 { return 3; }
    let mut chars = input.to_chars();
    let cursor_pos = input.before_cursor().len();
    if cursor_pos >= chars.len() { chars.push(' '); }
    let lines = visual_line_breaks(&chars, inner);
    let hint_lines = if has_hint { 1 } else { 0 };
    lines.len().max(1) + 2 + hint_lines
}

fn render_input_box(out: &mut String, input: &InputState, width: usize, expected_height: usize, more_below: bool, label: &str, hint: &str) {
    let indent = 2;
    let inner = width.saturating_sub(indent);
    let buf_chars = input.to_chars();
    let suggestion = slash_suggestion(&buf_chars);
    let sugg_chars: Vec<char> = suggestion.chars().collect();
    let all_chars: Vec<char> = buf_chars.iter().chain(sugg_chars.iter()).cloned().collect();
    let buf_len = buf_chars.len();

    // Wrap into visual lines (handles both width wrapping and \n)
    let lines = visual_line_breaks(&all_chars, inner);

    // Top border (with optional label embedded)
    if label.is_empty() {
        let _ = write!(out, "{DIM}");
        for _ in 0..width { out.push('─'); }
        let _ = write!(out, "{RESET}\r\n");
    } else {
        let label_vis = 3 + strip_ansi_width(label) + 1; // "── " + label + " "
        let _ = write!(out, "{DIM}── {RESET}{CYAN}{label}{RESET}{DIM} ");
        for _ in label_vis..width { out.push('─'); }
        let _ = write!(out, "{RESET}\r\n");
    }

    // Hint line (description of the field being edited)
    if !hint.is_empty() {
        let _ = write!(out, "  {DIM}{hint}{RESET}\x1b[K\r\n");
    }

    // Content lines (filter \n from output, it's only a line-break marker)
    for (i, &(start, end)) in lines.iter().enumerate() {
        let prefix = if i == 0 { format!("{BOLD}{CYAN}❯{RESET} ") } else { "  ".to_string() };
        if start >= buf_len {
            let s: String = all_chars[start..end].iter().filter(|&&c| c != '\n').collect();
            let _ = write!(out, "{prefix}{DIM}{s}{RESET}");
        } else if end > buf_len {
            let real: String = all_chars[start..buf_len].iter().filter(|&&c| c != '\n').collect();
            let sugg: String = all_chars[buf_len..end].iter().filter(|&&c| c != '\n').collect();
            let _ = write!(out, "{prefix}{real}{DIM}{sugg}{RESET}");
        } else {
            let s: String = all_chars[start..end].iter().filter(|&&c| c != '\n').collect();
            let _ = write!(out, "{prefix}{s}");
        }
        // Clear to end of line + newline
        out.push_str("\x1b[K\r\n");
    }

    // Fill empty content lines if cursor extends past content (cursor on new empty line)
    let hint_h = if hint.is_empty() { 0 } else { 1 };
    let expected_content = expected_height.saturating_sub(2 + hint_h); // minus borders + hint
    for _ in lines.len()..expected_content {
        let _ = write!(out, "  \x1b[K\r\n");
    }

    // Bottom border
    let _ = write!(out, "{DIM}");
    for _ in 0..width { out.push('─'); }
    let _ = write!(out, "{RESET}");
    if more_below { out.push_str("\r\n"); }
}

fn render_selector(out: &mut String, selector: &Selector, width: usize) {
    let end = (selector.scroll_offset + selector.max_visible).min(selector.items.len());
    if end <= selector.scroll_offset { return; }
    let can_scroll_up = selector.scroll_offset > 0;
    let can_scroll_down = end < selector.items.len();
    for i in selector.scroll_offset..end {
        let item = &selector.items[i];
        let is_selected = i == selector.selected;
        let is_first = i == selector.scroll_offset;
        let is_last = i + 1 == end;

        let line = if is_selected {
            format!("  {CYAN}{BOLD}▸ {}{RESET}", item.label)
        } else {
            format!("    {DIM}{}{RESET}", item.label)
        };

        // Add scroll indicators at edge rows
        let indicator = if is_first && can_scroll_up { format!(" {DIM}▲{RESET}") }
            else if is_last && can_scroll_down { format!(" {DIM}▼{RESET}") }
            else { String::new() };
        let full = format!("{line}{indicator}");

        if is_last {
            // Last visible item: clear to end, no \r\n (we're at the screen bottom)
            for ch in full.chars() {
                if ch != '\n' && ch != '\r' { out.push(ch); }
            }
            out.push_str("\x1b[K");
        } else {
            write_padded_line(out, &full, width);
        }
    }
}

/// Render content with code block support: text → markdown, code → syntect highlighting.
fn render_content_with_code(content: &str, md: &mut MarkdownRenderer, md_out: &mut String, is_streaming: bool) -> String {
    let segments = highlight::split_code_blocks(content);
    // If no code blocks, fast path
    if segments.len() == 1 {
        if let highlight::Segment::Text(t) = &segments[0] {
            *md = MarkdownRenderer::new();
            md_out.clear();
            md.feed(t, md_out);
            if is_streaming { md.flush_streaming(md_out); } else { md.flush(md_out); }
            return md_out.clone();
        }
    }
    let mut out = String::new();
    for seg in &segments {
        match seg {
            highlight::Segment::Text(text) => {
                *md = MarkdownRenderer::new();
                md_out.clear();
                md.feed(text, md_out);
                if is_streaming { md.flush_streaming(md_out); } else { md.flush(md_out); }
                out.push_str(md_out);
            }
            highlight::Segment::Code { lang, code } => {
                // Ensure code block starts on its own line
                if !out.is_empty() && !out.ends_with('\n') { out.push('\n'); }
                let highlighted = highlight::highlight_code(lang, code);
                out.push_str(&highlighted);
                // Ensure text after code block starts on its own line
                if !out.ends_with('\n') { out.push('\n'); }
            }
        }
    }
    out
}

fn wrap_chat_line(line: &ChatLine, width: usize, md: &mut MarkdownRenderer, md_out: &mut String, fade: Option<&[u8]>) -> Vec<String> {
    match line.kind {
        ChatLineKind::UserMessage => {
            *md = MarkdownRenderer::with_base_style(CYAN);
            md_out.clear();
            let clean = line.content.trim();
            md.feed(clean, md_out);
            md.flush(md_out);
            let styled = md_out.trim();
            let prefix = format!("  {BOLD}{CYAN}❯{RESET} ");
            wrap_styled_with_prefix(&prefix, "    ", &format!("{CYAN}{styled}{RESET}"), width)
        }
        ChatLineKind::AgentText | ChatLineKind::AgentStreaming => {
            *md = MarkdownRenderer::new();
            md_out.clear();
            let is_streaming = matches!(line.kind, ChatLineKind::AgentStreaming);
            // Streaming: preserve trailing newlines so line breaks appear immediately
            // Collapse runs of 3+ newlines to 2 (one blank line max)
            let trimmed = if is_streaming { line.content.trim_start() } else { line.content.trim() };
            let mut clean_buf = String::with_capacity(trimmed.len());
            let mut consecutive_nl = 0u8;
            for ch in trimmed.chars() {
                if ch == '\n' {
                    consecutive_nl += 1;
                    if consecutive_nl <= 2 { clean_buf.push(ch); }
                } else {
                    consecutive_nl = 0;
                    clean_buf.push(ch);
                }
            }
            let clean = clean_buf.as_str();
            if clean.is_empty() {
                return vec![format!("{BOLD}●{RESET} ")];
            }

            // Helper: flush markdown (streaming preserves trailing newlines)
            let md_flush = |md: &mut MarkdownRenderer, out: &mut String| {
                if is_streaming { md.flush_streaming(out); } else { md.flush(out); }
            };

            if let Some(fades) = fade {
                let settled_count = fades.iter().position(|&f| f > 0).unwrap_or(fades.len());
                let char_count = clean.chars().count();

                if settled_count >= char_count {
                    let rendered = render_content_with_code(clean, md, md_out, is_streaming);
                    let prefix = format!("  {BOLD}●{RESET} ");
                    wrap_styled_with_prefix(&prefix, "    ", rendered.trim_start(), width)
                } else {
                    // Settled portion: render with code block support
                    let settled: String = clean.chars().take(settled_count).collect();
                    let rendered = render_content_with_code(&settled, md, md_out, is_streaming);
                    let mut combined = rendered.trim_start().to_string();
                    for (i, ch) in clean.chars().skip(settled_count).enumerate() {
                        let fade_val = fades.get(settled_count + i).copied().unwrap_or(0);
                        if fade_val > 0 {
                            let shade_idx = (fade_val as usize - 1).min(FADE_SHADES.len() - 1);
                            combined.push_str(FADE_SHADES[shade_idx]);
                        }
                        combined.push(ch);
                    }
                    combined.push_str(RESET);
                    let prefix = format!("  {BOLD}●{RESET} ");
                    wrap_styled_with_prefix(&prefix, "    ", &combined, width)
                }
            } else {
                // No fade: full rendering with code block support
                let rendered = render_content_with_code(clean, md, md_out, is_streaming);
                let prefix = format!("  {BOLD}●{RESET} ");
                wrap_styled_with_prefix(&prefix, "    ", rendered.trim_start(), width)
            }
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
        let cw = char_display_width(ch);
        if vis_count + cw > max {
            rows.push(current_row);
            current_row = String::new();
            vis_count = 0;
        }
        current_row.push(ch);
        vis_count += cw;
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

fn write_padded_line(buf: &mut String, content: &str, _width: usize) {
    // Strip any embedded newlines — each padded line must be exactly 1 terminal row
    for ch in content.chars() {
        if ch != '\n' && ch != '\r' {
            buf.push(ch);
        }
    }
    // Clear to end of line — handles emoji width miscalculations without full-screen clear
    buf.push_str("\x1b[K\r\n");
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
