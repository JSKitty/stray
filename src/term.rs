//! Shared terminal primitives — ANSI styling, raw mode, key reading.

use std::io::{self, Write};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

// ANSI styling
pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const CYAN: &str = "\x1b[36m";
pub const ITALIC: &str = "\x1b[3m";
pub const YELLOW: &str = "\x1b[33m";
pub const RED: &str = "\x1b[31m";
pub const CURSOR_BAR: &str = "\x1b[6 q";
pub const CURSOR_DEFAULT: &str = "\x1b[0 q";

// ---------------------------------------------------------------------------
// Raw terminal mode
// ---------------------------------------------------------------------------

pub static ORIG_TERMIOS: Mutex<Option<libc::termios>> = Mutex::new(None);

/// Restore terminal to its original state (safe to call from panic hooks)
pub fn restore_terminal() {
    print!("{CURSOR_DEFAULT}\x1b[?2004l"); // restore cursor + disable bracketed paste
    let _ = io::stdout().flush();
    let guard = ORIG_TERMIOS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ref orig) = *guard {
        unsafe { libc::tcsetattr(0, libc::TCSAFLUSH, orig); }
    }
}

pub fn enable_raw_mode() -> libc::termios {
    unsafe {
        let mut orig: libc::termios = std::mem::zeroed();
        libc::tcgetattr(0, &mut orig);
        if let Ok(mut guard) = ORIG_TERMIOS.lock() {
            *guard = Some(orig);
        }
        let mut raw = orig;
        raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
        raw.c_iflag &= !(libc::ICRNL); // don't translate CR→NL (lets us distinguish Enter from Shift+Enter)
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        libc::tcsetattr(0, libc::TCSAFLUSH, &raw);
        // Enable bracketed paste mode
        print!("\x1b[?2004h");
        let _ = io::stdout().flush();
        orig
    }
}

pub fn is_tty() -> bool {
    unsafe { libc::isatty(0) == 1 }
}

// ---------------------------------------------------------------------------
// Low-level input
// ---------------------------------------------------------------------------

pub fn read_byte_raw() -> Option<u8> {
    let mut buf = [0u8; 1];
    let n = unsafe { libc::read(0, buf.as_mut_ptr() as *mut libc::c_void, 1) };
    if n == 1 { Some(buf[0]) } else { None }
}

pub fn stdin_ready(timeout_ms: i32) -> bool {
    unsafe {
        let mut pfd = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
        libc::poll(&mut pfd, 1, timeout_ms) > 0
    }
}

pub fn now_millis() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

pub fn term_height() -> usize {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_row > 0 {
            ws.ws_row as usize
        } else {
            24
        }
    }
}

pub fn term_width() -> usize {
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
            ws.ws_col as usize
        } else {
            80
        }
    }
}

// ---------------------------------------------------------------------------
// Key reading
// ---------------------------------------------------------------------------

pub enum Key {
    Char(char),
    Paste(String),
    Enter,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    ShiftEnter,
    PageUp,
    PageDown,
    ShiftUp,
    ShiftDown,
    AltUp,
    AltDown,
    Tab,
    Escape,
    CtrlC,
}

/// Read pasted content between bracketed paste markers (after \x1b[200~ has been consumed)
fn read_paste() -> Key {
    let mut paste = String::new();
    const MAX_PASTE: usize = 64 * 1024; // 64KB safety limit

    loop {
        if paste.len() >= MAX_PASTE { break; }
        if !stdin_ready(500) { break; } // 500ms timeout between bytes
        let b = match read_byte_raw() {
            Some(b) => b,
            None => break,
        };

        if b == b'\x1b' {
            // Might be end-of-paste: \x1b[201~
            if stdin_ready(50) {
                if let Some(b2) = read_byte_raw() {
                    if b2 == b'[' {
                        // Read parameter + final byte (with timeout + length limit)
                        let mut seq = Vec::new();
                        while seq.len() < 16 {
                            if !stdin_ready(100) { break; }
                            if let Some(sb) = read_byte_raw() {
                                seq.push(sb);
                                if sb >= b'@' { break; } // final byte of CSI
                            } else { break; }
                        }
                        if seq == b"201~" {
                            break; // end of paste
                        }
                        // Not end-of-paste — discard this escape sequence
                        continue;
                    }
                }
            }
            continue;
        }

        // Handle UTF-8 multi-byte
        if b >= 128 {
            let len = if b & 0xE0 == 0xC0 { 2 }
                 else if b & 0xF0 == 0xE0 { 3 }
                 else if b & 0xF8 == 0xF0 { 4 }
                 else { continue };
            let mut bytes = vec![b];
            for _ in 1..len {
                if let Some(nb) = read_byte_raw() { bytes.push(nb); }
            }
            if let Ok(s) = std::str::from_utf8(&bytes) {
                paste.push_str(s);
            }
        } else if b == b'\r' || b == b'\n' {
            paste.push('\n');
        } else if b >= 32 || b == b'\t' {
            paste.push(b as char);
        }
    }

    Key::Paste(paste)
}

pub fn read_key() -> Option<Key> {
    let b = read_byte_raw()?;
    match b {
        b'\r' => Some(Key::Enter),
        b'\n' => Some(Key::ShiftEnter), // Ctrl+J → insert newline
        b'\t' => Some(Key::Tab),
        127 | 8 => Some(Key::Backspace),
        1 => Some(Key::Home),     // Ctrl+A
        3 => Some(Key::CtrlC),
        5 => Some(Key::End),      // Ctrl+E
        b'\x1b' => {
            if !stdin_ready(50) { return Some(Key::Escape); }
            let b2 = read_byte_raw()?;
            if b2 == b'\r' || b2 == b'\n' { return Some(Key::ShiftEnter); } // Alt+Enter → newline
            if b2 != b'[' { return Some(Key::Escape); }

            // Read CSI parameters (digits, semicolons) then final byte
            let mut params = Vec::new();
            loop {
                if !stdin_ready(100) { return None; } // timeout: malformed sequence
                let b3 = read_byte_raw()?;
                if (b3 >= b'0' && b3 <= b'9') || b3 == b';' {
                    if params.len() >= 16 { return None; } // sanity limit
                    params.push(b3);
                } else {
                    // b3 is the final byte
                    let param: String = params.iter().map(|&c| c as char).collect();
                    return match (param.as_str(), b3) {
                        ("", b'A') => Some(Key::Up),
                        ("", b'B') => Some(Key::Down),
                        ("", b'C') => Some(Key::Right),
                        ("", b'D') => Some(Key::Left),
                        ("", b'H') => Some(Key::Home),
                        ("", b'F') => Some(Key::End),
                        ("1", b'~') => Some(Key::Home),
                        ("3", b'~') => Some(Key::Delete),
                        ("4", b'~') => Some(Key::End),
                        ("5", b'~') => Some(Key::PageUp),
                        ("6", b'~') => Some(Key::PageDown),
                        ("1;2", b'A') => Some(Key::ShiftUp),
                        ("1;2", b'B') => Some(Key::ShiftDown),
                        ("1;3", b'A') => Some(Key::AltUp),    // Alt/Option+Up
                        ("1;3", b'B') => Some(Key::AltDown),  // Alt/Option+Down
                        ("1;5", b'A') => Some(Key::AltUp),    // Ctrl+Up (same action)
                        ("1;5", b'B') => Some(Key::AltDown),  // Ctrl+Down (same action)
                        ("13;2", b'u') => Some(Key::ShiftEnter), // Shift+Enter (CSI u)
                        ("200", b'~') => Some(read_paste()),
                        _ => None,
                    };
                }
            }
        }
        b if b >= 32 && b < 128 => Some(Key::Char(b as char)),
        b if b >= 128 => {
            let len = if b & 0xE0 == 0xC0 { 2 }
                 else if b & 0xF0 == 0xE0 { 3 }
                 else if b & 0xF8 == 0xF0 { 4 }
                 else { return None };
            let mut bytes = vec![b];
            for _ in 1..len {
                bytes.push(read_byte_raw()?);
            }
            std::str::from_utf8(&bytes).ok()?.chars().next().map(Key::Char)
        }
        _ => None,
    }
}
