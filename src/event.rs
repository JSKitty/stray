//! Event system — channels, input thread, tick thread, resize watcher.

use crate::term::{read_key, Key};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;

pub enum Event {
    Key(Key),
    Resize,
    Tick,
}

/// Spawn all event-producing threads. Returns the event receiver.
pub fn setup_event_channels() -> Receiver<Event> {
    let (tx, rx) = mpsc::channel();

    // Input thread (raw terminal, sends Key events — never touches stdout)
    let tx_input = tx.clone();
    let is_tty = crate::term::is_tty();
    if is_tty {
        thread::spawn(move || {
            loop {
                if let Some(key) = read_key() {
                    if tx_input.send(Event::Key(key)).is_err() {
                        break;
                    }
                }
            }
        });
    } else {
        // Piped input: read lines, send as Key::Paste
        thread::spawn(move || {
            use std::io::BufRead;
            let stdin = std::io::stdin();
            let reader = stdin.lock();
            for line in reader.lines().flatten() {
                let trimmed = line.trim().to_string();
                if !trimmed.is_empty() {
                    if tx_input.send(Event::Key(Key::Paste(trimmed))).is_err() {
                        break;
                    }
                    // Simulate Enter after paste
                    if tx_input.send(Event::Key(Key::Enter)).is_err() {
                        break;
                    }
                }
            }
        });
    }

    // Tick thread (80ms interval for spinner animation)
    let tx_tick = tx.clone();
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(80));
            if tx_tick.send(Event::Tick).is_err() {
                break;
            }
        }
    });

    // Resize watcher (SIGWINCH → pipe → event)
    let tx_resize = tx;
    setup_resize_watcher(tx_resize);

    rx
}

fn setup_resize_watcher(tx: Sender<Event>) {
    // Create a pipe for signal-safe communication
    let mut fds = [0i32; 2];
    unsafe {
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return; // pipe failed, resize won't work
        }
        // Store write end in a global for the signal handler
        RESIZE_PIPE_WRITE.store(fds[1], std::sync::atomic::Ordering::Relaxed);
        libc::signal(libc::SIGWINCH, sigwinch_handler as *const () as libc::sighandler_t);
    }

    // Watcher thread: reads from pipe, sends Event::Resize
    let read_fd = fds[0];
    thread::spawn(move || {
        let mut buf = [0u8; 1];
        loop {
            let n = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, 1) };
            if n <= 0 { break; }
            let _ = tx.send(Event::Resize);
        }
    });
}

/// Install signal handlers as a safety net — restores terminal on SIGINT/SIGTERM
/// even if the event loop is stuck.
pub fn setup_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGINT, fatal_signal_handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, fatal_signal_handler as *const () as libc::sighandler_t);
    }
}

static RESIZE_PIPE_WRITE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

extern "C" fn sigwinch_handler(_: libc::c_int) {
    let fd = RESIZE_PIPE_WRITE.load(std::sync::atomic::Ordering::Relaxed);
    if fd >= 0 {
        unsafe {
            let buf: [u8; 1] = [1];
            let _ = libc::write(fd, buf.as_ptr() as *const libc::c_void, 1);
        }
    }
}

extern "C" fn fatal_signal_handler(sig: libc::c_int) {
    // Restore terminal and exit — last resort for external kill signals.
    // Note: keyboard Ctrl+C goes through the event loop (ISIG is disabled),
    // so this handler only fires from external `kill -2` / `kill -15`.
    unsafe {
        // Leave alternate screen + restore cursor + disable bracketed paste
        let msg = b"\x1b[?1049l\x1b[0 q\x1b[?2004l";
        let _ = libc::write(1, msg.as_ptr() as *const libc::c_void, msg.len());
        // Restore termios — use try_lock to avoid deadlock if mutex is held
        if let Ok(guard) = crate::term::ORIG_TERMIOS.try_lock() {
            if let Some(ref orig) = *guard {
                libc::tcsetattr(0, libc::TCSAFLUSH, orig);
            }
        }
        libc::_exit(128 + sig);
    }
}
