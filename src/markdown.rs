//! Streaming markdown renderer — bold/italic via ANSI, glob-safe.
//! Writes to a &mut String buffer instead of directly to stdout.

use crate::term::*;
use std::fmt::Write;

pub struct MarkdownRenderer {
    bold: bool,
    italic: bool,
    pending_star: bool,
    pending_newlines: usize,
    last_printed: char,
}

impl MarkdownRenderer {
    pub fn new() -> Self {
        Self { bold: false, italic: false, pending_star: false, pending_newlines: 0, last_printed: ' ' }
    }

    fn apply_style(&self, out: &mut String) {
        let _ = write!(out, "{RESET}");
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
            self.last_printed.is_alphanumeric()
        } else {
            next_ch.is_alphanumeric()
        }
    }

    pub fn feed(&mut self, text: &str, out: &mut String) {
        for ch in text.chars() {
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
            } else {
                self.print_char(ch, out);
            }
        }
    }

    pub fn flush(&mut self, out: &mut String) {
        if self.pending_star {
            self.pending_star = false;
            out.push('*');
        }
        self.pending_newlines = 0;
        let _ = write!(out, "{RESET}");
    }
}
