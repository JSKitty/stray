//! Headless `serve` mode — Stray as a 24/7 daemon with no TUI/TTY.
//!
//! Runs the same agent (tools, trust, heartbeat) as the interactive TUI, but
//! driven by a local Unix-socket control interface instead of a terminal:
//!
//!   stray serve            # the daemon (systemd ExecStart)
//!   stray send "<msg>"     # submit a message, stream progress, print the reply
//!   stray status           # snapshot of the running daemon
//!
//! Architecture: exactly ONE thread (this loop) ever calls the LLM or executes
//! tools, so message history needs no lock — turns serialize by construction.
//! Socket connections and (later) Link peers funnel requests through an mpsc
//! channel; `status` is served lock-free from a snapshot so it never blocks
//! behind a long turn. The loop wakes at least once a second to poll the
//! shutdown flag and the heartbeat clock.
//!
//! Trust: the local socket runs turns at the operator's configured trust. Remote
//! Stray Link peers are intentionally NOT wired to run turns here yet — inbound
//! Link messages are only observed/logged (peer identities are persisted). A
//! later step runs Link-originated turns at a reduced, separate trust.

use crate::config::LlmConfig;
use crate::formats::ModelFormat;
use crate::tools::ToolRegistry;
use crate::{Message, Role};

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Set by the SIGTERM/SIGINT handler; polled by the loop for a clean shutdown.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

const MAX_TURN_ROUNDS: u64 = 40; // tool rounds before a turn is force-ended
const MAX_TURN_SECS: u64 = 600; // wall-clock guard for a single turn
const LOOP_TICK: Duration = Duration::from_secs(1); // shutdown/heartbeat poll granularity

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// The control socket both the daemon and the `send`/`status` clients use.
/// Lives in the config dir (NOT $XDG_RUNTIME_DIR, which is unset over plain ssh)
/// so the path resolves identically for systemd and for interactive clients.
pub fn control_socket_path() -> Option<PathBuf> {
    crate::config::global_config_dir().map(|d| d.join("control.sock"))
}

fn history_path() -> Option<PathBuf> {
    crate::config::global_config_dir().map(|d| d.join("serve-history.json"))
}

// ---------------------------------------------------------------------------
// Control protocol (newline-delimited JSON, one request → one+ response lines)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct Request {
    cmd: String,
    #[serde(default)]
    content: String,
}

/// Frames the agent loop streams back for a `send`, in order.
enum ReplyFrame {
    Progress(String),
    Done(String),
    Error(String),
}

/// Work funneled into the single agent loop.
enum ServeMsg {
    Control {
        content: String,
        reply: Sender<ReplyFrame>,
    },
}

// ---------------------------------------------------------------------------
// Status snapshot (read lock-free by `status`, updated by the agent loop)
// ---------------------------------------------------------------------------

struct ServeStatus {
    started: Instant,
    model: String,
    trust: String,
    heartbeat_secs: u64,
    endpoint_id: String,
    busy: bool,
    busy_since: Option<Instant>,
    turns: u64,
    history_len: usize,
}

impl ServeStatus {
    fn snapshot_json(&self) -> serde_json::Value {
        serde_json::json!({
            "uptime_secs": self.started.elapsed().as_secs(),
            "model": self.model,
            "trust": self.trust,
            "heartbeat_secs": self.heartbeat_secs,
            "endpoint_id": self.endpoint_id,
            "busy": self.busy,
            "busy_secs": self.busy_since.map(|t| t.elapsed().as_secs()),
            "turns": self.turns,
            "history_len": self.history_len,
        })
    }
}

fn set_busy(status: &Arc<Mutex<ServeStatus>>, busy: bool) {
    if let Ok(mut s) = status.lock() {
        s.busy = busy;
        s.busy_since = if busy { Some(Instant::now()) } else { None };
    }
}

fn finish_turn(status: &Arc<Mutex<ServeStatus>>, hist_len: usize) {
    if let Ok(mut s) = status.lock() {
        s.busy = false;
        s.busy_since = None;
        s.turns += 1;
        s.history_len = hist_len;
    }
}

// ---------------------------------------------------------------------------
// History persistence (own copy — tasks::{load,save}_history are private)
// ---------------------------------------------------------------------------

/// On-disk history row. `Message`/`Role` aren't serde-derivable (Role is a plain
/// enum), so persistence goes through this string-tagged shape — same as tasks.rs.
#[derive(serde::Serialize, serde::Deserialize)]
struct HistoryEntry {
    role: String,
    content: String,
}

fn role_from_str(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        _ => Role::User,
    }
}

fn load_history() -> Vec<Message> {
    let Some(path) = history_path() else {
        return Vec::new();
    };
    let Ok(data) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(entries) = serde_json::from_str::<Vec<HistoryEntry>>(&data) else {
        return Vec::new();
    };
    entries
        .into_iter()
        .map(|e| Message { role: role_from_str(&e.role), content: e.content })
        .collect()
}

fn save_history(messages: &[Message]) {
    let Some(path) = history_path() else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let entries: Vec<HistoryEntry> = messages
        .iter()
        .map(|m| HistoryEntry { role: m.role.as_str().to_string(), content: m.content.clone() })
        .collect();
    if let Ok(data) = serde_json::to_string(&entries) {
        let _ = std::fs::write(&path, data);
    }
}

// ---------------------------------------------------------------------------
// Signals — async-signal-safe: the handler only flips a flag.
// ---------------------------------------------------------------------------

extern "C" fn on_signal(_sig: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
    }
}

// ---------------------------------------------------------------------------
// The agent turn (mirrors tasks::run_headless's inner loop, minus task ceremony)
// ---------------------------------------------------------------------------

/// Run one full agent turn to completion: append `input`, then loop
/// call_llm → parse → execute tools until the model stops calling tools (or a
/// guard trips). Returns the final assistant text. `on_progress` is invoked with
/// a short line per tool call so callers can stream liveness. Never panics out —
/// LLM/tool errors become returned strings so the daemon survives.
fn run_turn(
    input: &str,
    messages: &mut Vec<Message>,
    registry: &ToolRegistry,
    tools_json: &Option<serde_json::Value>,
    format: &dyn ModelFormat,
    llm: &LlmConfig,
    compact_at: usize,
    on_progress: &mut dyn FnMut(&str),
) -> String {
    messages.push(Message {
        role: Role::User,
        content: input.to_string(),
    });

    let deadline = Instant::now() + Duration::from_secs(MAX_TURN_SECS);
    let mut round: u64 = 0;

    loop {
        // Bound shutdown latency: don't start another LLM call once asked to stop.
        if SHUTDOWN.load(Ordering::SeqCst) {
            return "[turn aborted: shutting down]".to_string();
        }
        if Instant::now() > deadline {
            return "[turn ended: exceeded time budget]".to_string();
        }

        let resp = match crate::call_llm(llm, messages, tools_json, None, None, &[], &mut Vec::new()) {
            Ok(r) => r,
            Err(e) => return format!("[llm error] {e}"),
        };

        // In headless mode call_llm reports a transport failure IN-BAND — empty
        // content with `disturbed` set — rather than as Err. Surface it instead
        // of returning a silent "" (a dead endpoint must not look like success).
        if resp.disturbed {
            return "[llm error] request failed — check the API endpoint, key, or network".to_string();
        }

        let (calls, _) = format.parse_response(&resp.content);
        messages.push(Message {
            role: Role::Assistant,
            content: resp.content.clone(),
        });

        if calls.is_empty() {
            maybe_compact(messages, llm, compact_at);
            return resp.content;
        }

        let mut results: Vec<(String, String, String)> = Vec::new();
        for call in &calls {
            on_progress(&format!(
                "{} {}",
                call.tool,
                crate::tools::truncate_middle(&call.input, 60)
            ));
            let output = match registry.tools().iter().find(|t| t.name() == call.tool) {
                Some(t) => {
                    if let Some(spawn_result) = t.spawn(&call.input) {
                        match spawn_result {
                            Ok(child) => match child.wait_with_output() {
                                Ok(out) => t.format_output(&out),
                                Err(e) => format!("[error] {e}"),
                            },
                            Err(e) => e,
                        }
                    } else {
                        t.execute(&call.input)
                    }
                }
                None => format!("[error] Unknown tool: {}", call.tool),
            };
            results.push((call.tool.clone(), call.input.clone(), output));
        }

        messages.push(Message {
            role: Role::User,
            content: format.format_results(&results),
        });

        round += 1;
        maybe_compact(messages, llm, compact_at);
        if round >= MAX_TURN_ROUNDS {
            return format!("[turn ended: reached {MAX_TURN_ROUNDS} tool rounds]");
        }
    }
}

/// Inline context compaction — the headless equivalent of the TUI's
/// `compact_context` (which is AppState-coupled). Mirrors tasks.rs.
fn maybe_compact(messages: &mut Vec<Message>, llm: &LlmConfig, compact_at: usize) {
    let tokens = crate::estimate_tokens(messages);
    if compact_at == 0 || tokens < compact_at {
        return;
    }
    eprintln!("[serve] context ~{tokens} tokens — compacting");
    messages.push(Message {
        role: Role::User,
        content: crate::COMPACT_PROMPT.into(),
    });
    match crate::call_llm(llm, messages, &None, None, None, &[], &mut Vec::new()) {
        Ok(resp) => {
            let system = messages.first().cloned().unwrap_or(Message {
                role: Role::System,
                content: String::new(),
            });
            messages.clear();
            messages.push(system);
            messages.push(Message {
                role: Role::Assistant,
                content: format!("[Context compacted from ~{tokens} tokens]\n\n{}", resp.content),
            });
        }
        Err(e) => {
            eprintln!("[serve] compaction failed: {e}");
            messages.pop(); // drop the compact prompt; try again later
        }
    }
}

/// Run a turn for a `send`, streaming progress frames and a terminal frame to
/// the client. Panic-isolated so a misbehaving tool can't take down the daemon.
#[allow(clippy::too_many_arguments)]
fn run_turn_streaming(
    content: &str,
    messages: &mut Vec<Message>,
    registry: &ToolRegistry,
    tools_json: &Option<serde_json::Value>,
    format: &dyn ModelFormat,
    llm: &LlmConfig,
    compact_at: usize,
    status: &Arc<Mutex<ServeStatus>>,
    reply: &Sender<ReplyFrame>,
) {
    set_busy(status, true);
    let outcome = {
        let mut cb = |m: &str| {
            let _ = reply.send(ReplyFrame::Progress(m.to_string()));
        };
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_turn(content, messages, registry, tools_json, format, llm, compact_at, &mut cb)
        }))
    };
    match outcome {
        Ok(text) => {
            let _ = reply.send(ReplyFrame::Done(text));
        }
        Err(_) => {
            let _ = reply.send(ReplyFrame::Error("turn aborted (internal panic)".into()));
        }
    }
    save_history(messages);
    finish_turn(status, messages.len());
}

/// Run a heartbeat turn, logging progress + a preview to stderr (journald).
#[allow(clippy::too_many_arguments)]
fn run_turn_logged(
    tag: &str,
    content: &str,
    messages: &mut Vec<Message>,
    registry: &ToolRegistry,
    tools_json: &Option<serde_json::Value>,
    format: &dyn ModelFormat,
    llm: &LlmConfig,
    compact_at: usize,
    status: &Arc<Mutex<ServeStatus>>,
) {
    set_busy(status, true);
    let outcome = {
        let mut cb = |m: &str| eprintln!("[serve/{tag}] · {m}");
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_turn(content, messages, registry, tools_json, format, llm, compact_at, &mut cb)
        }))
    };
    match outcome {
        Ok(text) => eprintln!("[serve/{tag}] {}", crate::tools::truncate_middle(text.trim(), 200)),
        Err(_) => eprintln!("[serve/{tag}] turn aborted (internal panic)"),
    }
    save_history(messages);
    finish_turn(status, messages.len());
}

// ---------------------------------------------------------------------------
// Control socket — listener + per-connection handlers
// ---------------------------------------------------------------------------

fn spawn_socket_listener(
    sock_path: &Path,
    serve_tx: Sender<ServeMsg>,
    status: Arc<Mutex<ServeStatus>>,
) {
    // Restrict the socket to owner-only. umask before bind closes the window
    // between node creation and the explicit chmod below; restore it right after
    // so it doesn't silently 0600 every other file the daemon later writes.
    let old_umask = unsafe { libc::umask(0o077) };
    let listener = match UnixListener::bind(sock_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[serve] cannot bind control socket {}: {e}", sock_path.display());
            std::process::exit(1);
        }
    };
    let _ = std::fs::set_permissions(sock_path, std::fs::Permissions::from_mode(0o600));
    unsafe {
        libc::umask(old_umask);
    }

    std::thread::spawn(move || {
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    // One handler thread per connection so a long `send` never
                    // blocks a concurrent `status` (which is snapshot-only).
                    let tx = serve_tx.clone();
                    let st = status.clone();
                    std::thread::spawn(move || handle_conn(stream, tx, st));
                }
                Err(_) => continue,
            }
        }
    });
}

fn handle_conn(stream: UnixStream, serve_tx: Sender<ServeMsg>, status: Arc<Mutex<ServeStatus>>) {
    // Don't let a client that connects and never sends a newline pin a handler
    // thread forever, or stream an unbounded line to exhaust memory.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let mut writer = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut reader = BufReader::new(stream).take(64 * 1024);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
        return;
    }

    let req: Request = match serde_json::from_str(line.trim()) {
        Ok(r) => r,
        Err(e) => {
            let _ = writeln!(writer, "{}", serde_json::json!({"ok": false, "error": format!("bad request: {e}")}));
            return;
        }
    };

    match req.cmd.as_str() {
        "status" => {
            let snap = status
                .lock()
                .map(|s| s.snapshot_json())
                .unwrap_or_else(|e| e.into_inner().snapshot_json());
            let _ = writeln!(writer, "{}", serde_json::json!({"ok": true, "status": snap}));
        }
        "send" => {
            if req.content.trim().is_empty() {
                let _ = writeln!(writer, "{}", serde_json::json!({"ok": false, "error": "empty message"}));
                return;
            }
            let (reply_tx, reply_rx) = mpsc::channel::<ReplyFrame>();
            if serve_tx
                .send(ServeMsg::Control { content: req.content, reply: reply_tx })
                .is_err()
            {
                let _ = writeln!(writer, "{}", serde_json::json!({"ok": false, "error": "daemon shutting down"}));
                return;
            }
            // Stream frames until a terminal one. If the sender is dropped
            // (agent-loop panic before Done), recv() errors → report it.
            loop {
                match reply_rx.recv() {
                    Ok(ReplyFrame::Progress(m)) => {
                        if writeln!(writer, "{}", serde_json::json!({"ok": true, "partial": m})).is_err() {
                            break; // client hung up; the turn keeps running server-side
                        }
                    }
                    Ok(ReplyFrame::Done(text)) => {
                        let _ = writeln!(writer, "{}", serde_json::json!({"ok": true, "reply": text}));
                        break;
                    }
                    Ok(ReplyFrame::Error(e)) => {
                        let _ = writeln!(writer, "{}", serde_json::json!({"ok": false, "error": e}));
                        break;
                    }
                    Err(_) => {
                        let _ = writeln!(writer, "{}", serde_json::json!({"ok": false, "error": "turn aborted"}));
                        break;
                    }
                }
            }
        }
        other => {
            let _ = writeln!(writer, "{}", serde_json::json!({"ok": false, "error": format!("unknown cmd: {other}")}));
        }
    }
}

/// Probe an existing socket: true if a live daemon answers `status`.
fn ping_existing(sock_path: &Path) -> bool {
    let Ok(stream) = UnixStream::connect(sock_path) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let Ok(mut wr) = stream.try_clone() else {
        return false;
    };
    if writeln!(wr, "{}", serde_json::json!({"cmd": "status"})).is_err() {
        return false;
    }
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map(|n| n > 0).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Link forwarder (feature-gated) — 3a: observe only, persist peer identities.
// ---------------------------------------------------------------------------

#[cfg(feature = "link")]
fn spawn_link_forwarder(event_rx: mpsc::Receiver<crate::event::Event>) {
    std::thread::spawn(move || {
        while let Ok(ev) = event_rx.recv() {
            let crate::event::Event::Link(le) = ev else {
                continue;
            };
            match le {
                crate::link::LinkEvent::IncomingMessage { from_name, content, is_result, .. } => {
                    // Observed but NOT executed here — running Link-originated
                    // turns (at a reduced, separate trust) is a later step.
                    eprintln!(
                        "[serve/link] inbound from '{from_name}' (result={is_result}): {}",
                        crate::tools::truncate_middle(&content, 100)
                    );
                }
                crate::link::LinkEvent::PeerIdentified { endpoint_id, name, addr, .. } => {
                    let mut peers = crate::link::load_peers();
                    let addr_str = addr.unwrap_or_default();
                    if let Some(existing) = peers.iter_mut().find(|p| p.endpoint_id == endpoint_id) {
                        existing.name = name;
                        if !addr_str.is_empty() {
                            existing.addr = addr_str;
                        }
                    } else {
                        peers.push(crate::link::PeerEntry {
                            name,
                            endpoint_id,
                            trusted: true,
                            last_seen: 0,
                            addr: addr_str,
                        });
                    }
                    crate::link::save_peers(&peers);
                }
                crate::link::LinkEvent::Ready => eprintln!("[serve/link] endpoint online"),
                crate::link::LinkEvent::Error(e) => eprintln!("[serve/link] error: {e}"),
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Daemon entrypoint
// ---------------------------------------------------------------------------

pub fn run() {
    // 1. Config, non-interactively (never the TTY wizard).
    let Some(loaded) = crate::config::load_existing() else {
        eprintln!(
            "[serve] no config found (looked for ./stray.toml and the global config).\n\
             Run `stray` once to set it up, or place a stray.toml, then start the service."
        );
        std::process::exit(1);
    };
    let config = loaded.config;

    let Some(sock_path) = control_socket_path() else {
        eprintln!("[serve] cannot resolve the config directory for the control socket");
        std::process::exit(1);
    };

    // 2. Single-instance guard: don't steal a live socket.
    if sock_path.exists() {
        if ping_existing(&sock_path) {
            eprintln!("[serve] another stray daemon is already running ({})", sock_path.display());
            std::process::exit(1);
        }
        let _ = std::fs::remove_file(&sock_path); // stale socket — reclaim it
    }

    // 3. Build the agent, identical to the TUI main agent (operator trust).
    let vision_flag = Arc::new(AtomicBool::new(config.llm.vision));
    let shared_llm = Arc::new(Mutex::new(config.llm.clone()));
    let shared_trust = Arc::new(Mutex::new(config.agent.trust));

    #[cfg(feature = "link")]
    let (link_cmd_tx, link_cmd_rx) = mpsc::channel::<crate::link::LinkCommand>();
    #[cfg(feature = "link")]
    let link_endpoint_id = crate::link::get_endpoint_id();
    #[cfg(not(feature = "link"))]
    let link_endpoint_id = String::from("(link disabled)");

    let registry = {
        let base = ["bash", "read", "write", "edit"].map(String::from);
        let mut r = crate::tools::build_registry(&base, shared_trust.clone(), true, vision_flag.clone());
        r.add(Box::new(crate::tasks::TaskTool::new(shared_llm.clone())));
        #[cfg(feature = "link")]
        r.add(Box::new(crate::link::LinkTool::new(
            link_cmd_tx.clone(),
            link_endpoint_id.clone(),
            config.agent.name.clone(),
        )));
        r
    };

    let format = crate::formats::format_for_model(&config.llm.model, &registry);
    let tools_json = format.format_tools(&registry);
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".into());

    // 4. History: resume across reboots/sleep; always refresh the system prompt.
    let system = crate::build_system_prompt(&config, &*format, &registry, &cwd);
    let mut messages = load_history();
    if messages.is_empty() {
        messages.push(Message { role: Role::System, content: system });
    } else if let Some(first) = messages.first_mut() {
        first.role = Role::System;
        first.content = system;
    }

    // 5. Sandbox-posture guard: a boxed trust with no OS backend = silent unconfined.
    let trust_now = config.agent.trust;
    let backend = crate::sandbox::backend();
    if trust_now.is_boxed() && backend == "none" {
        eprintln!(
            "[serve] ⚠ trust '{}' expects an OS sandbox but none is available — bash tool \
             calls will run UNCONFINED. Install bubblewrap (Linux) or lower the risk.",
            trust_now.as_str()
        );
    }
    eprintln!(
        "[serve] up · trust={} · sandbox={} · model={} · heartbeat={}s · endpoint={}",
        trust_now.as_str(),
        backend,
        config.llm.model,
        config.agent.heartbeat,
        link_endpoint_id
    );

    // 6. Shutdown signals.
    install_signal_handlers();

    // 7. Status snapshot.
    let status = Arc::new(Mutex::new(ServeStatus {
        started: Instant::now(),
        model: config.llm.model.clone(),
        trust: trust_now.as_str().to_string(),
        heartbeat_secs: config.agent.heartbeat,
        endpoint_id: link_endpoint_id.clone(),
        busy: false,
        busy_since: None,
        turns: 0,
        history_len: messages.len(),
    }));

    // 8. Control socket.
    let (serve_tx, serve_rx) = mpsc::channel::<ServeMsg>();
    spawn_socket_listener(&sock_path, serve_tx.clone(), status.clone());

    // 9. Link: live endpoint + peer persistence (inbound turns come later).
    #[cfg(feature = "link")]
    {
        let (event_tx, event_rx) = mpsc::channel::<crate::event::Event>();
        // The iroh endpoint runs in the thread LinkManager::start spawns, and the
        // command channel stays open via link_cmd_tx + the registry's LinkTool
        // clone — so the returned handle can simply drop here.
        let _mgr = crate::link::LinkManager::start(
            link_cmd_rx,
            link_cmd_tx.clone(),
            event_tx,
            config.agent.name.clone(),
        );
        spawn_link_forwarder(event_rx);
    }

    // 10. The single-writer agent loop.
    let compact_at = config.agent.compact_at;
    let heartbeat = config.agent.heartbeat;
    let mut next_hb = Instant::now() + Duration::from_secs(heartbeat.max(1));

    loop {
        if SHUTDOWN.load(Ordering::SeqCst) {
            eprintln!("[serve] shutdown requested — saving state");
            save_history(&messages);
            let _ = std::fs::remove_file(&sock_path);
            eprintln!("[serve] bye");
            std::process::exit(0);
        }

        if heartbeat > 0 && Instant::now() >= next_hb {
            let prompt = format!("[{}] Heartbeat. Check in and do your tasks.", crate::timestamp());
            run_turn_logged(
                "heartbeat", &prompt, &mut messages, &registry, &tools_json, &*format,
                &config.llm, compact_at, &status,
            );
            next_hb = Instant::now() + Duration::from_secs(heartbeat);
        }

        match serve_rx.recv_timeout(LOOP_TICK) {
            Ok(ServeMsg::Control { content, reply }) => {
                run_turn_streaming(
                    &content, &mut messages, &registry, &tools_json, &*format,
                    &config.llm, compact_at, &status, &reply,
                );
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                eprintln!("[serve] control channel closed — exiting");
                break;
            }
        }
    }

    // Defensive fallback only: the normal exit is process::exit(0) in the
    // SHUTDOWN branch above (which already saves + unlinks). We reach here only
    // if the control channel ever disconnects, which can't happen while `run`
    // holds the original serve_tx — kept so a future refactor stays correct.
    save_history(&messages);
    let _ = std::fs::remove_file(&sock_path);
}

// ---------------------------------------------------------------------------
// Clients: `stray send` and `stray status`
// ---------------------------------------------------------------------------

/// `stray send "<msg>"` — submit a message; stream progress to stderr, print the
/// final reply to stdout. Returns a process exit code.
pub fn send_cli(msg: &str) -> i32 {
    let content = if msg.trim().is_empty() || msg.trim() == "-" {
        let mut s = String::new();
        if std::io::stdin().read_to_string(&mut s).is_err() {
            eprintln!("stray send: failed to read stdin");
            return 2;
        }
        s
    } else {
        msg.to_string()
    };
    if content.trim().is_empty() {
        eprintln!("stray send: empty message");
        return 2;
    }

    let Some(path) = control_socket_path() else {
        eprintln!("stray send: cannot resolve the socket path");
        return 2;
    };
    let stream = match UnixStream::connect(&path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("stray send: no daemon at {} (is `stray serve` running?)", path.display());
            return 3;
        }
    };
    let mut wr = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return 2,
    };
    if writeln!(wr, "{}", serde_json::json!({"cmd": "send", "content": content})).is_err() {
        eprintln!("stray send: write failed");
        return 2;
    }

    let reader = BufReader::new(stream);
    let mut exit = 0;
    let mut got_terminal = false;
    for line in reader.lines() {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if let Some(p) = v.get("partial").and_then(|x| x.as_str()) {
            eprintln!("… {p}");
        } else if let Some(reply) = v.get("reply").and_then(|x| x.as_str()) {
            println!("{reply}");
            got_terminal = true;
        } else if v.get("ok").and_then(|x| x.as_bool()) == Some(false) {
            eprintln!("stray send: {}", v.get("error").and_then(|x| x.as_str()).unwrap_or("unknown error"));
            exit = 1;
            got_terminal = true;
        }
    }
    // A stream that closed before any reply/error frame = the turn never
    // completed (handler died mid-dispatch); don't report that as success.
    if !got_terminal {
        eprintln!("stray send: connection closed before a reply (daemon may have crashed)");
        return 3;
    }
    exit
}

/// `stray status` — print the running daemon's status snapshot.
pub fn status_cli() -> i32 {
    let Some(path) = control_socket_path() else {
        eprintln!("stray status: cannot resolve the socket path");
        return 2;
    };
    let stream = match UnixStream::connect(&path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("stray status: no daemon running (socket {})", path.display());
            return 3;
        }
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let mut wr = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return 2,
    };
    if writeln!(wr, "{}", serde_json::json!({"cmd": "status"})).is_err() {
        return 2;
    }
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
        eprintln!("stray status: no response");
        return 3;
    }
    match serde_json::from_str::<serde_json::Value>(line.trim()) {
        Ok(v) => {
            let body = v.get("status").cloned().unwrap_or(v);
            println!("{}", serde_json::to_string_pretty(&body).unwrap_or(line));
            0
        }
        Err(_) => {
            print!("{line}");
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_str_roundtrip() {
        // Persisted role tags must survive a save/load cycle unchanged, or a
        // restart would silently rewrite assistant/system turns as user turns.
        assert_eq!(role_from_str(Role::System.as_str()).as_str(), "system");
        assert_eq!(role_from_str(Role::User.as_str()).as_str(), "user");
        assert_eq!(role_from_str(Role::Assistant.as_str()).as_str(), "assistant");
        // Unknown tags fall back to User.
        assert_eq!(role_from_str("garbage").as_str(), "user");
    }

    #[test]
    fn history_entries_serialize_roundtrip() {
        let entries = vec![
            HistoryEntry { role: "system".into(), content: "sys".into() },
            HistoryEntry { role: "user".into(), content: "hi".into() },
            HistoryEntry { role: "assistant".into(), content: "yo".into() },
        ];
        let json = serde_json::to_string(&entries).unwrap();
        let back: Vec<HistoryEntry> = serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back[2].role, "assistant");
        assert_eq!(back[2].content, "yo");
    }

    #[test]
    fn request_parses_send_and_status() {
        let r: Request = serde_json::from_str(r#"{"cmd":"send","content":"hello"}"#).unwrap();
        assert_eq!(r.cmd, "send");
        assert_eq!(r.content, "hello");
        // `content` defaults so `status` needs no body.
        let s: Request = serde_json::from_str(r#"{"cmd":"status"}"#).unwrap();
        assert_eq!(s.cmd, "status");
        assert_eq!(s.content, "");
        // Malformed input is a parse error (→ handler replies with an error frame).
        assert!(serde_json::from_str::<Request>("not json").is_err());
    }
}
