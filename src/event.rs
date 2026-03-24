//! Event system — channels, input thread, tick thread, resize watcher.

use crate::term::{enable_raw_mode, read_key, Key};
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
            let _orig = enable_raw_mode();
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
