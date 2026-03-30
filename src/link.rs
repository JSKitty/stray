//! Stray Link — P2P communication between Stray instances via Iroh.
//!
//! Enables cross-machine agent orchestration: send tasks to remote Strays,
//! receive results as push notifications, manage trusted peers.

use crate::config;
use crate::event::Event;
use crate::tools::Tool;
use iroh::endpoint::Connection;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::mpsc::Sender;

/// ALPN protocol identifier for Stray Link connections
const STRAY_ALPN: &[u8] = b"stray-link/1";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Events sent from the Iroh thread to the main event loop.
pub enum LinkEvent {
    /// A remote peer sent us a message
    IncomingMessage {
        from_id: String,
        from_name: String,
        content: String,
        is_result: bool,
    },
    /// A peer responded to our ping (pairing handshake)
    PeerIdentified {
        endpoint_id: String,
        name: String,
        version: String,
        addr: Option<String>, // serialized EndpointAddr for reconnection
    },
    /// Error from the Iroh thread
    Error(String),
    /// Link is ready (endpoint is online)
    Ready,
}

/// Commands sent from the main thread to the Iroh thread.
pub enum LinkCommand {
    /// Connect to a peer by endpoint address string
    Connect(String),
    /// Send a message to a connected peer
    Send { endpoint_id: String, addr: Option<String>, message: WireMessage },
    /// Shut down the endpoint
    Shutdown,
}

/// Wire protocol messages (JSON, length-prefixed over QUIC bidi streams).
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum WireMessage {
    #[serde(rename = "ping")]
    Ping { name: String, version: String },
    #[serde(rename = "pong")]
    Pong { name: String, version: String },
    #[serde(rename = "task")]
    Task { from_name: String, content: String },
    #[serde(rename = "result")]
    Result { from_name: String, content: String },
}

// ---------------------------------------------------------------------------
// Peer trust list (peers.toml)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
pub struct PeerEntry {
    pub name: String,
    pub endpoint_id: String,
    pub trusted: bool,
    #[serde(default)]
    pub last_seen: u64,
    /// Serialized EndpointAddr JSON for reconnection (includes relay + direct addrs)
    #[serde(default)]
    pub addr: String,
}

#[derive(Serialize, Deserialize, Default)]
struct PeersFile {
    #[serde(default)]
    peer: Vec<PeerEntry>,
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn link_dir() -> Option<PathBuf> {
    config::global_config_dir().map(|d| d.join("link"))
}

fn keypair_path() -> Option<PathBuf> {
    link_dir().map(|d| d.join("keypair"))
}

fn peers_path() -> Option<PathBuf> {
    link_dir().map(|d| d.join("peers.toml"))
}

// ---------------------------------------------------------------------------
// Keypair persistence
// ---------------------------------------------------------------------------

fn load_or_generate_keypair() -> iroh::SecretKey {
    if let Some(path) = keypair_path() {
        if path.exists() {
            if let Ok(bytes) = std::fs::read(&path) {
                if bytes.len() == 32 {
                    if let Ok(key) = iroh::SecretKey::try_from(&bytes[..]) {
                        return key;
                    }
                }
            }
        }
        // Generate new key and persist
        let key = iroh::SecretKey::generate(&mut &mut rand::rng());
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&path, key.to_bytes());
        key
    } else {
        iroh::SecretKey::generate(&mut &mut rand::rng())
    }
}

pub fn load_peers() -> Vec<PeerEntry> {
    peers_path()
        .and_then(|p| std::fs::read_to_string(&p).ok())
        .and_then(|s| toml::from_str::<PeersFile>(&s).ok())
        .map(|f| f.peer)
        .unwrap_or_default()
}

pub fn save_peers(peers: &[PeerEntry]) {
    if let Some(path) = peers_path() {
        let file = PeersFile { peer: peers.to_vec() };
        if let Ok(s) = toml::to_string_pretty(&file) {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&path, s);
        }
    }
}

/// Get the endpoint ID (public key) without starting the full link system.
pub fn get_endpoint_id() -> String {
    let key = load_or_generate_keypair();
    key.public().to_string()
}

// ---------------------------------------------------------------------------
// LinkManager — main-thread handle to the Iroh bridge
// ---------------------------------------------------------------------------

pub struct LinkManager {
    cmd_tx: Sender<LinkCommand>,
    endpoint_id: String,
}

impl LinkManager {
    /// Spawn the Iroh bridge thread with a pre-created command channel.
    pub fn start(
        cmd_rx: std::sync::mpsc::Receiver<LinkCommand>,
        cmd_tx: Sender<LinkCommand>,
        event_tx: Sender<Event>,
        agent_name: String,
    ) -> Self {
        let secret_key = load_or_generate_keypair();
        let endpoint_id = secret_key.public().to_string();

        std::thread::spawn(move || {
            iroh_thread(secret_key, cmd_rx, event_tx, agent_name);
        });

        LinkManager { cmd_tx, endpoint_id }
    }

    pub fn endpoint_id(&self) -> &str {
        &self.endpoint_id
    }

    pub fn send_command(&self, cmd: LinkCommand) {
        let _ = self.cmd_tx.send(cmd);
    }

    pub fn cmd_sender(&self) -> Sender<LinkCommand> {
        self.cmd_tx.clone()
    }
}

// ---------------------------------------------------------------------------
// Iroh bridge thread (runs tokio internally)
// ---------------------------------------------------------------------------

fn iroh_thread(
    secret_key: iroh::SecretKey,
    cmd_rx: std::sync::mpsc::Receiver<LinkCommand>,
    event_tx: Sender<Event>,
    agent_name: String,
) {
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(2)
        .thread_name("stray-iroh")
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Failed to create tokio runtime: {e}")
            )));
            return;
        }
    };

    rt.block_on(async move {
        iroh_main(secret_key, cmd_rx, event_tx, agent_name).await;
    });
}

async fn iroh_main(
    secret_key: iroh::SecretKey,
    cmd_rx: std::sync::mpsc::Receiver<LinkCommand>,
    event_tx: Sender<Event>,
    agent_name: String,
) {
    use iroh::endpoint::presets;

    // Build endpoint with persistent identity
    let endpoint = match iroh::Endpoint::builder(presets::N0)
        .secret_key(secret_key)
        .alpns(vec![STRAY_ALPN.to_vec()])
        .bind()
        .await
    {
        Ok(ep) => ep,
        Err(e) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Iroh bind failed: {e}")
            )));
            return;
        }
    };

    // Wait for the endpoint to be online (connected to relay)
    endpoint.online().await;
    let _ = event_tx.send(Event::Link(LinkEvent::Ready));

    // Spawn accept loop for incoming connections
    let ep_accept = endpoint.clone();
    let evt_accept = event_tx.clone();
    let name_accept = agent_name.clone();
    tokio::spawn(async move {
        accept_loop(ep_accept, evt_accept, name_accept).await;
    });

    // Process commands from main thread
    loop {
        match cmd_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(LinkCommand::Connect(addr_str)) => {
                let ep = endpoint.clone();
                let evt = event_tx.clone();
                let name = agent_name.clone();
                tokio::spawn(async move {
                    handle_connect(ep, &addr_str, evt, name).await;
                });
            }
            Ok(LinkCommand::Send { endpoint_id, addr, message }) => {
                let ep = endpoint.clone();
                let evt = event_tx.clone();
                tokio::spawn(async move {
                    handle_send(ep, &endpoint_id, addr, message, evt).await;
                });
            }
            Ok(LinkCommand::Shutdown) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    endpoint.close().await;
}

// ---------------------------------------------------------------------------
// Accept loop — handles incoming connections using raw accept
// ---------------------------------------------------------------------------

async fn accept_loop(
    endpoint: iroh::Endpoint,
    event_tx: Sender<Event>,
    agent_name: String,
) {
    loop {
        let incoming = match endpoint.accept().await {
            Some(incoming) => incoming,
            None => break,
        };

        let evt = event_tx.clone();
        let name = agent_name.clone();
        tokio::spawn(async move {
            match incoming.accept() {
                Ok(connecting) => {
                    if let Ok(conn) = connecting.await {
                        handle_incoming(conn, evt, name).await;
                    }
                }
                Err(_) => {}
            }
        });
    }
}

async fn handle_incoming(
    conn: Connection,
    event_tx: Sender<Event>,
    agent_name: String,
) {
    let remote_id = conn.remote_id().to_string();

    // Accept streams from this connection
    loop {
        let (send, recv) = match conn.accept_bi().await {
            Ok(s) => s,
            Err(_) => break,
        };

        let msg = match read_message(recv).await {
            Ok(m) => m,
            Err(_) => break,
        };

        match &msg {
            WireMessage::Ping { name, version } => {
                let pong = WireMessage::Pong {
                    name: agent_name.clone(),
                    version: config::VERSION.to_string(),
                };
                let _ = write_message(send, &pong).await;
                // Capture remote address for reconnection
                let remote_addr: Option<String> = None; // will be populated from remote_info on connect
                let _ = event_tx.send(Event::Link(LinkEvent::PeerIdentified {
                    endpoint_id: remote_id.clone(),
                    name: name.clone(),
                    version: version.clone(),
                    addr: remote_addr,
                }));
            }
            WireMessage::Task { from_name, content } => {
                let _ = event_tx.send(Event::Link(LinkEvent::IncomingMessage {
                    from_id: remote_id.clone(),
                    from_name: from_name.clone(),
                    content: content.clone(),
                    is_result: false,
                }));
            }
            WireMessage::Result { from_name, content } => {
                let _ = event_tx.send(Event::Link(LinkEvent::IncomingMessage {
                    from_id: remote_id.clone(),
                    from_name: from_name.clone(),
                    content: content.clone(),
                    is_result: true,
                }));
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Wire protocol — length-prefixed JSON over QUIC bidi streams
// ---------------------------------------------------------------------------

async fn read_message(mut recv: iroh::endpoint::RecvStream) -> std::result::Result<WireMessage, String> {
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await.map_err(|e| e.to_string())?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > 1_000_000 {
        return Err("Message too large".into());
    }
    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf).await.map_err(|e| e.to_string())?;
    serde_json::from_slice(&buf).map_err(|e| e.to_string())
}

async fn write_message(mut send: iroh::endpoint::SendStream, msg: &WireMessage) -> std::result::Result<(), String> {
    use tokio::io::AsyncWriteExt;
    let json = serde_json::to_vec(msg).map_err(|e| e.to_string())?;
    let len = (json.len() as u32).to_be_bytes();
    send.write_all(&len).await.map_err(|e| e.to_string())?;
    send.write_all(&json).await.map_err(|e| e.to_string())?;
    let _ = send.finish();
    Ok(())
}

// ---------------------------------------------------------------------------
// Outbound operations
// ---------------------------------------------------------------------------

async fn handle_connect(
    endpoint: iroh::Endpoint,
    addr_str: &str,
    event_tx: Sender<Event>,
    agent_name: String,
) {
    // Parse as EndpointId (PublicKey) and wrap in EndpointAddr
    let endpoint_id: iroh::EndpointId = match addr_str.parse() {
        Ok(id) => id,
        Err(e) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Invalid endpoint ID: {e}")
            )));
            return;
        }
    };
    let addr = iroh::EndpointAddr::new(endpoint_id);

    let conn = match endpoint.connect(addr, STRAY_ALPN).await {
        Ok(c) => c,
        Err(e) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Connect failed: {e}")
            )));
            return;
        }
    };

    // Send Ping
    let (send, recv) = match conn.open_bi().await {
        Ok(s) => s,
        Err(e) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Stream failed: {e}")
            )));
            return;
        }
    };

    let ping = WireMessage::Ping {
        name: agent_name,
        version: config::VERSION.to_string(),
    };
    if let Err(e) = write_message(send, &ping).await {
        let _ = event_tx.send(Event::Link(LinkEvent::Error(format!("Ping failed: {e}"))));
        return;
    }

    // Wait for Pong
    match read_message(recv).await {
        Ok(WireMessage::Pong { name, version }) => {
            let remote_addr: Option<String> = None; // populated via remote_info on next send
            let _ = event_tx.send(Event::Link(LinkEvent::PeerIdentified {
                endpoint_id: conn.remote_id().to_string(),
                name,
                version,
                addr: remote_addr,
            }));
        }
        Ok(_) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                "Unexpected response (expected Pong)".into()
            )));
        }
        Err(e) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Pong read failed: {e}")
            )));
        }
    }
}

async fn handle_send(
    endpoint: iroh::Endpoint,
    endpoint_id_str: &str,
    addr_json: Option<String>,
    message: WireMessage,
    event_tx: Sender<Event>,
) {
    let eid: iroh::EndpointId = match endpoint_id_str.parse() {
        Ok(id) => id,
        Err(e) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Invalid endpoint ID: {e}")
            )));
            return;
        }
    };

    // Try stored address first, then check Iroh's route cache, fall back to bare ID
    let addr = if let Some(ref json) = addr_json {
        serde_json::from_str::<iroh::EndpointAddr>(json)
            .unwrap_or_else(|_| iroh::EndpointAddr::new(eid))
    } else if let Some(info) = endpoint.remote_info(eid).await {
        // Build EndpointAddr from cached remote info
        let addrs: Vec<iroh::TransportAddr> = info.addrs()
            .map(|a| a.addr().clone())
            .collect();
        iroh::EndpointAddr::from_parts(eid, addrs)
    } else {
        iroh::EndpointAddr::new(eid)
    };

    let conn = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        endpoint.connect(addr, STRAY_ALPN)
    ).await {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Send failed: {e}")
            )));
            return;
        }
        Err(_) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                "Send failed: connection timed out (10s)".into()
            )));
            return;
        }
    };

    let (send, _recv) = match conn.open_bi().await {
        Ok(s) => s,
        Err(e) => {
            let _ = event_tx.send(Event::Link(LinkEvent::Error(
                format!("Stream open failed: {e}")
            )));
            return;
        }
    };

    if let Err(e) = write_message(send, &message).await {
        let _ = event_tx.send(Event::Link(LinkEvent::Error(
            format!("Message send failed: {e}")
        )));
    }
}

// ---------------------------------------------------------------------------
// LinkTool — lets the AI send tasks to remote peers
// ---------------------------------------------------------------------------

pub struct LinkTool {
    cmd_tx: Sender<LinkCommand>,
    endpoint_id: String,
    agent_name: String,
}

impl LinkTool {
    pub fn new(cmd_tx: Sender<LinkCommand>, endpoint_id: String, agent_name: String) -> Self {
        Self { cmd_tx, endpoint_id, agent_name }
    }
}

impl Tool for LinkTool {
    fn name(&self) -> &str { "link" }
    fn tag(&self) -> &str { "link" }

    fn description(&self) -> &str {
        "Send tasks to remote Stray instances over encrypted P2P.\n\
         You will be AUTOMATICALLY NOTIFIED of responses — no need to poll.\n\n\
         Actions:\n\
         - list: Show connected peers\n\
         - send: Send a task to a remote peer"
    }

    fn usage_hint(&self) -> &str {
        "action: list\n\naction: send\npeer: vps-stray\nmessage: check disk usage and report back"
    }

    fn display_action(&self, input: &str) -> String {
        let mut action = "list";
        let mut peer = "";
        for line in input.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("action:") { action = rest.trim(); }
            if let Some(rest) = line.strip_prefix("peer:") { peer = rest.trim(); }
        }
        match action {
            "send" => format!("Sending task to '{peer}'"),
            _ => "Listing link peers".into(),
        }
    }

    fn execute(&self, input: &str) -> String {
        let mut action = "list".to_string();
        let mut peer_name = String::new();
        let mut message = String::new();

        for line in input.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("action:") {
                action = rest.trim().to_lowercase();
            } else if let Some(rest) = line.strip_prefix("peer:") {
                peer_name = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("message:") {
                message = rest.trim().to_string();
            }
        }

        match action.as_str() {
            "list" => {
                let peers = load_peers();
                let id_short = if self.endpoint_id.len() > 16 { &self.endpoint_id[..16] } else { &self.endpoint_id };
                if peers.is_empty() {
                    format!("[Stray Link — ID: {id_short}...]\nNo peers connected. Use /link connect <endpoint-addr> to pair.")
                } else {
                    let mut result = format!("[Stray Link — ID: {id_short}...]\nPeers:\n");
                    for p in &peers {
                        let trust = if p.trusted { "trusted" } else { "untrusted" };
                        let id_short = if p.endpoint_id.len() > 16 { &p.endpoint_id[..16] } else { &p.endpoint_id };
                        result.push_str(&format!("  {} — {} ({id_short}...)\n", p.name, trust));
                    }
                    result
                }
            }
            "send" => {
                if peer_name.is_empty() {
                    return "[error] Peer name required".into();
                }
                if message.is_empty() {
                    return "[error] Message required".into();
                }
                let peers = load_peers();
                match peers.iter().find(|p| p.name == peer_name) {
                    Some(p) => {
                        let msg = WireMessage::Task {
                            from_name: self.agent_name.clone(),
                            content: message,
                        };
                        let addr = if p.addr.is_empty() { None } else { Some(p.addr.clone()) };
                        let _ = self.cmd_tx.send(LinkCommand::Send {
                            endpoint_id: p.endpoint_id.clone(),
                            addr,
                            message: msg,
                        });
                        format!("[Task sent to '{peer_name}' — you will be notified when they respond]")
                    }
                    None => format!("[error] Unknown peer: {peer_name}. Use /link to see connected peers."),
                }
            }
            _ => format!("[error] Unknown action: {action}. Use list or send."),
        }
    }
}
