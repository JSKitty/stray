//! Code block detection and syntax highlighting via inkjet (tree-sitter).

use crate::term::*;
use inkjet::{Highlighter, Language};
use inkjet::formatter::Formatter;
use inkjet::constants::HIGHLIGHT_NAMES;
use inkjet::tree_sitter_highlight::HighlightEvent;
use std::fmt::Write;

/// A segment of content — either plain text or a fenced code block.
pub enum Segment<'a> {
    Text(&'a str),
    Code { lang: &'a str, code: &'a str },
}

/// Split content into text and ``` code blocks.
pub fn split_code_blocks(content: &str) -> Vec<Segment<'_>> {
    let mut segments = Vec::new();
    let mut rest = content;

    loop {
        let Some(open_pos) = rest.find("```") else {
            if !rest.is_empty() { segments.push(Segment::Text(rest)); }
            break;
        };

        if open_pos > 0 {
            segments.push(Segment::Text(&rest[..open_pos]));
        }

        let after_backticks = &rest[open_pos + 3..];
        let lang_end = after_backticks.find('\n').unwrap_or(after_backticks.len());
        let lang = after_backticks[..lang_end].trim();
        let code_start = if lang_end < after_backticks.len() { lang_end + 1 } else { lang_end };
        let code_rest = &after_backticks[code_start..];

        if let Some(close_pos) = code_rest.find("```") {
            let code = &code_rest[..close_pos];
            let code = code.strip_suffix('\n').unwrap_or(code);
            segments.push(Segment::Code { lang, code });
            rest = &code_rest[close_pos + 3..];
            if rest.starts_with('\n') { rest = &rest[1..]; }
        } else {
            segments.push(Segment::Code { lang, code: code_rest });
            break;
        }
    }

    segments
}

/// Map a language hint string to an inkjet Language.
fn lang_from_str(s: &str) -> Language {
    match s.to_lowercase().as_str() {
        "rust" | "rs" => Language::Rust,
        "python" | "py" => Language::Python,
        "javascript" | "js" => Language::Javascript,
        "typescript" | "ts" => Language::Typescript,
        "bash" | "sh" | "shell" | "zsh" => Language::Bash,
        "go" | "golang" => Language::Go,
        "c" => Language::C,
        "cpp" | "c++" | "cxx" => Language::Cpp,
        "json" => Language::Json,
        "toml" => Language::Toml,
        "html" => Language::Html,
        "css" => Language::Css,
        "yaml" | "yml" => Language::Yaml,
        "sql" => Language::Sql,
        "dockerfile" | "docker" => Language::Dockerfile,
        _ => Language::Plaintext,
    }
}

/// Map a highlight name to an ANSI color code.
fn highlight_color(name: &str) -> &'static str {
    match name {
        n if n.starts_with("keyword") => "\x1b[36m",              // cyan
        n if n.starts_with("string") => "\x1b[32m",               // green
        n if n.starts_with("comment") => "\x1b[2;37m",            // dim white
        "function" | "function.method" | "function.builtin"
            | "function.macro" => "\x1b[33m",                      // yellow
        "type" | "type.builtin" | "constructor" => "\x1b[36;1m",  // bold cyan
        "constant" | "constant.builtin" | "number"
            | "boolean" | "float" => "\x1b[33m",                   // yellow
        "operator" => "\x1b[37m",                                   // white
        n if n.starts_with("punctuation") => "\x1b[2m",           // dim
        "label" | "tag" => "\x1b[36m",                             // cyan
        "namespace" | "module" => "\x1b[36;2m",                    // dim cyan
        "escape" => "\x1b[35m",                                    // magenta
        _ => "",
    }
}

/// Custom ANSI formatter for inkjet.
struct AnsiFormatter;

impl Formatter for AnsiFormatter {
    fn write<W>(&self, source: &str, writer: &mut W, event: HighlightEvent) -> inkjet::Result<()>
    where W: std::fmt::Write,
    {
        match event {
            HighlightEvent::HighlightStart(h) => {
                if let Some(name) = HIGHLIGHT_NAMES.get(h.0) {
                    let color = highlight_color(name);
                    if !color.is_empty() {
                        writer.write_str(color)?;
                    }
                }
            }
            HighlightEvent::HighlightEnd => {
                writer.write_str(RESET)?;
            }
            HighlightEvent::Source { start, end } => {
                if end <= source.len() {
                    writer.write_str(&source[start..end])?;
                }
            }
        }
        Ok(())
    }
}

/// Syntax-highlight a code block, returning ANSI-colored string.
pub fn highlight_code(lang: &str, code: &str) -> String {
    let language = lang_from_str(lang);
    let mut highlighter = Highlighter::new();
    let formatter = AnsiFormatter;

    match highlighter.highlight_to_string(language, &formatter, code) {
        Ok(mut s) => {
            let _ = write!(s, "{RESET}");
            s
        }
        Err(_) => {
            // Fallback: plain code
            format!("{code}{RESET}")
        }
    }
}
