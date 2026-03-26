//! Streaming markdown renderer — bold/italic via ANSI, glob-safe.
//! Writes to a &mut String buffer instead of directly to stdout.

use crate::term::*;
use std::fmt::Write;

pub struct MarkdownRenderer {
    bold: bool,
    italic: bool,
    code: bool,
    pending_star: bool,
    pending_newlines: usize,
    last_printed: char,
    base_style: &'static str,
}

impl MarkdownRenderer {
    pub fn new() -> Self {
        Self { bold: false, italic: false, code: false, pending_star: false, pending_newlines: 0, last_printed: ' ', base_style: "" }
    }

    pub fn with_base_style(style: &'static str) -> Self {
        Self { base_style: style, ..Self::new() }
    }

    fn apply_style(&self, out: &mut String) {
        let _ = write!(out, "{RESET}{}", self.base_style);
        if self.code { let _ = write!(out, "{YELLOW}"); }
        if self.bold { let _ = write!(out, "{BOLD}"); }
        if self.italic { let _ = write!(out, "{ITALIC}"); }
    }

    fn emit_newlines(&mut self, out: &mut String) {
        for _ in 0..self.pending_newlines {
            out.push('\n');
        }
        self.pending_newlines = 0;
    }

    fn print_char(&mut self, ch: char, out: &mut String) {
        if ch == '\n' {
            self.pending_newlines += 1;
        } else {
            self.emit_newlines(out);
            out.push(ch);
            self.last_printed = ch;
        }
    }

    fn is_italic_marker(&self, next_ch: char) -> bool {
        if self.italic {
            !self.last_printed.is_whitespace() // closing: any non-whitespace before *
        } else {
            next_ch.is_alphanumeric()          // opening: alphanumeric after * (glob-safe)
        }
    }

    pub fn feed(&mut self, text: &str, out: &mut String) {
        for ch in text.chars() {
            // Inside inline code: only backtick is special
            if self.code {
                if ch == '`' {
                    self.code = false;
                    self.apply_style(out);
                } else {
                    self.print_char(ch, out);
                }
                continue;
            }
            if self.pending_star {
                self.pending_star = false;
                if ch == '*' {
                    self.bold = !self.bold;
                    self.apply_style(out);
                } else if self.is_italic_marker(ch) {
                    self.italic = !self.italic;
                    self.apply_style(out);
                    self.print_char(ch, out);
                } else {
                    self.print_char('*', out);
                    self.print_char(ch, out);
                }
            } else if ch == '*' {
                self.pending_star = true;
            } else if ch == '`' {
                self.code = true;
                self.apply_style(out);
            } else {
                self.print_char(ch, out);
            }
        }
    }

    pub fn flush(&mut self, out: &mut String) {
        if self.pending_star {
            self.pending_star = false;
            if self.italic {
                self.italic = false;
                self.apply_style(out);
            } else if self.bold {
                self.bold = false;
                self.apply_style(out);
            } else {
                out.push('*');
            }
        }
        self.code = false;
        self.pending_newlines = 0;
        let _ = write!(out, "{RESET}");
    }

    /// Flush preserving trailing newlines (for streaming — so line breaks appear immediately)
    pub fn flush_streaming(&mut self, out: &mut String) {
        if self.pending_star {
            self.pending_star = false;
            if self.italic {
                self.italic = false;
                self.apply_style(out);
            } else if self.bold {
                self.bold = false;
                self.apply_style(out);
            } else {
                out.push('*');
            }
        }
        self.emit_newlines(out); // keep trailing newlines
        let _ = write!(out, "{RESET}");
    }
}
