//! Stray ⇄ Vector bridge.
//!
//! Relays Vector (Nostr NIP-17 private DMs) into Stray's file inbox and delivers
//! Stray's replies back out. Runs as its own process beside the headless daemon,
//! sharing only the filesystem — the `stray` core never links any Vector code.
//!
//!   Vector DM  ─►  inbox/vector/new/<id>.json   (from = sender npub, reply_to = sender npub)
//!   outbox/vector/<x>.json  ─►  Vector DM to reply_to
//!
//! Security: the bridge stamps `from`/`reply_to` ONLY from the cryptographically
//! authenticated gift-unwrap sender — never anything derived from message content
//! — and only ever delivers a reply to the `reply_to` npub in the outbox file
//! (which Stray's reply tool set to the original sender). So a stranger can never
//! make Stray message a third party.

use nostr_sdk::nips::nip59::UnwrappedGift;
use nostr_sdk::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use vector_sdk::VectorBot;

const MAX_CONTENT_BYTES: usize = 32 * 1024;
const SEEN_CAP: usize = 4000;
const OUTBOX_MAX_ATTEMPTS: u32 = 10;

#[derive(serde::Deserialize)]
struct OutboxReply {
    reply_to: String,
    content: String,
}

#[tokio::main]
async fn main() {
    let data = data_dir();
    let inbox_new = data.join("inbox").join("vector").join("new");
    let outbox = data.join("outbox").join("vector");
    let failed = outbox.join("failed");
    let _ = std::fs::create_dir_all(&inbox_new);
    let _ = std::fs::create_dir_all(&failed);

    let keys = load_or_generate_keys(&data);
    let npub = keys.public_key().to_bech32().unwrap_or_default();
    eprintln!("[vector-bridge] identity npub: {npub}");
    eprintln!("[vector-bridge] DM that npub from Vector; add it to Stray's [inbox.sources.vector] trusted list.");

    let bot = VectorBot::quick(keys.clone()).await;

    // Create the notifications receiver BEFORE subscribing, so gift-wraps the
    // relays replay right after the subscription aren't missed (a broadcast
    // receiver only sees events sent after it exists).
    let mut notifications = bot.client.notifications();

    let filter = Filter::new().pubkey(keys.public_key()).kind(Kind::GiftWrap);
    if let Err(e) = bot.client.subscribe(filter, None).await {
        eprintln!("[vector-bridge] subscribe failed: {e}");
    }
    eprintln!("[vector-bridge] listening for DMs");

    // Receive loop: unwrap incoming gift-wraps → write an inbox event.
    {
        let recv_keys = keys.clone();
        let inbox_dir = inbox_new.clone();
        let mut seen = SeenSet::load(data.join("vector-bridge").join("seen.log"));
        tokio::spawn(async move {
            loop {
                let notification = match notifications.recv().await {
                    Ok(n) => n,
                    // A lag (burst outran us) is fully recoverable — keep going,
                    // do NOT let the bridge silently go deaf.
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        eprintln!("[vector-bridge] notifications lagged by {n}, continuing");
                        continue;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        eprintln!("[vector-bridge] notification stream closed — exiting for restart");
                        std::process::exit(1); // let systemd restart us
                    }
                };
                let RelayPoolNotification::Event { event, .. } = notification else {
                    continue;
                };
                if event.kind != Kind::GiftWrap {
                    continue;
                }
                // The gift-wrap event id is stable per message → dedup relay
                // redeliveries (which would otherwise re-run a trusted action).
                let id = format!("vec-{}", event.id.to_hex());
                if seen.seen(&id) {
                    continue;
                }
                let unwrapped = match UnwrappedGift::from_gift_wrap(&recv_keys, &event).await {
                    Ok(u) => u,
                    Err(_) => continue, // not for us / undecryptable
                };
                if unwrapped.rumor.kind != Kind::PrivateDirectMessage {
                    continue;
                }
                let mut content = unwrapped.rumor.content.clone();
                if content.trim().is_empty() {
                    continue;
                }
                if content.len() > MAX_CONTENT_BYTES {
                    content.truncate(byte_boundary(&content, MAX_CONTENT_BYTES));
                }
                // Identity comes from the SIGNATURE-VERIFIED seal author, never
                // the spoofable rumor.pubkey.
                let from = match unwrapped.sender.to_bech32() {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                match write_inbox_event(&inbox_dir, &id, &from, &content) {
                    Ok(()) => {
                        seen.mark(&id);
                        eprintln!("[vector-bridge] inbound DM from {from} → inbox");
                    }
                    Err(e) => eprintln!("[vector-bridge] failed to write inbox event: {e}"),
                }
            }
        });
    }

    // Outbox loop: deliver Stray's replies, with bounded retries + dead-letter.
    let mut attempts: HashMap<String, u32> = HashMap::new();
    loop {
        process_outbox(&bot, &outbox, &failed, &mut attempts).await;
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

fn write_inbox_event(new_dir: &Path, id: &str, from_npub: &str, content: &str) -> std::io::Result<()> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let event = serde_json::json!({
        "id": id,
        "source": "vector",
        "kind": "message",
        "from": from_npub,
        "content": content,
        "reply_to": from_npub, // reply always goes back to the authenticated sender
        "ts": ts,
    });
    // Atomic: write a temp then rename in, so Stray's watcher never reads a partial file.
    let tmp = new_dir.join(format!(".tmp-{id}"));
    let final_path = new_dir.join(format!("{id}.json"));
    if let Err(e) = std::fs::write(&tmp, serde_json::to_string(&event)?) {
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, &final_path) {
        let _ = std::fs::remove_file(&tmp); // don't leave an orphan temp behind
        return Err(e);
    }
    Ok(())
}

async fn process_outbox(bot: &VectorBot, dir: &Path, failed: &Path, attempts: &mut HashMap<String, u32>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || !name.ends_with(".json") {
            continue; // skips the `failed/` subdir too
        }
        let Ok(data) = std::fs::read_to_string(&path) else {
            continue;
        };
        let reply: OutboxReply = match serde_json::from_str(&data) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[vector-bridge] bad outbox file {name}: {e} — dropping");
                let _ = std::fs::remove_file(&path);
                attempts.remove(&name);
                continue;
            }
        };
        // reply_to must be a valid npub (Stray set it to the original sender).
        let pk = match PublicKey::from_bech32(&reply.reply_to) {
            Ok(pk) => pk,
            Err(_) => {
                eprintln!("[vector-bridge] invalid reply_to '{}' — dropping", reply.reply_to);
                let _ = std::fs::remove_file(&path);
                attempts.remove(&name);
                continue;
            }
        };
        let channel = bot.get_chat(pk).await;
        if channel.send_private_message(&reply.content).await {
            let _ = std::fs::remove_file(&path);
            attempts.remove(&name);
            eprintln!("[vector-bridge] delivered reply to {}", reply.reply_to);
        } else {
            let n = attempts.entry(name.clone()).or_insert(0);
            *n += 1;
            if *n >= OUTBOX_MAX_ATTEMPTS {
                eprintln!("[vector-bridge] reply to {} failed {n} times — moving to failed/", reply.reply_to);
                let _ = std::fs::create_dir_all(failed);
                let _ = std::fs::rename(&path, failed.join(&name));
                attempts.remove(&name);
            } else {
                eprintln!("[vector-bridge] send to {} failed (attempt {n}/{OUTBOX_MAX_ATTEMPTS}) — will retry", reply.reply_to);
            }
        }
    }
    // Forget attempt counters for files that are gone (delivered or removed).
    attempts.retain(|k, _| dir.join(k).exists());
}

fn byte_boundary(s: &str, max: usize) -> usize {
    let mut end = max.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Persistent, bounded set of already-ingested gift-wrap ids, so relay
/// redelivery (which resends stored NIP-59 wraps on reconnect) can't make Stray
/// process — and re-run — the same message twice.
struct SeenSet {
    ids: HashSet<String>,
    path: PathBuf,
}

impl SeenSet {
    fn load(path: PathBuf) -> Self {
        let mut lines: Vec<String> = std::fs::read_to_string(&path)
            .map(|s| s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
            .unwrap_or_default();
        // Keep the file bounded across restarts.
        if lines.len() > SEEN_CAP {
            lines = lines.split_off(lines.len() - SEEN_CAP);
            let _ = std::fs::write(&path, lines.join("\n") + "\n");
        }
        SeenSet { ids: lines.into_iter().collect(), path }
    }
    fn seen(&self, id: &str) -> bool {
        self.ids.contains(id)
    }
    fn mark(&mut self, id: &str) {
        if self.ids.insert(id.to_string()) {
            if let Some(dir) = self.path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
                let _ = writeln!(f, "{id}");
            }
        }
    }
}

/// Stray's data dir. MUST stay in lockstep with `config::global_data_base`
/// (Linux: XDG_DATA_HOME else ~/.local/share; macOS: ~/Library/Application
/// Support) joined with `cat.jskitty.stray`, so the bridge writes the tree Stray
/// reads. `VECTOR_BRIDGE_DATA` is the explicit override (used as-is, no suffix).
fn data_dir() -> PathBuf {
    if let Ok(d) = std::env::var("VECTOR_BRIDGE_DATA") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    let base = if cfg!(target_os = "macos") {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library").join("Application Support"))
    } else {
        std::env::var("XDG_DATA_HOME")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".local").join("share")))
    };
    base.unwrap_or_else(|| PathBuf::from(".")).join("cat.jskitty.stray")
}

/// Load the bridge's persisted Nostr identity, or generate one on FIRST run.
/// Never clobbers an existing identity: if the key file exists but can't be
/// read/parsed, exit rather than overwrite it (that would silently rotate the
/// npub and break the operator's trust allowlist).
fn load_or_generate_keys(data: &Path) -> Keys {
    let key_path = data.join("vector-bridge").join("nsec");
    if key_path.exists() {
        let nsec = std::fs::read_to_string(&key_path).unwrap_or_else(|e| {
            eprintln!("[vector-bridge] FATAL: cannot read identity {}: {e}", key_path.display());
            std::process::exit(1);
        });
        return Keys::parse(nsec.trim()).unwrap_or_else(|e| {
            eprintln!(
                "[vector-bridge] FATAL: identity {} is corrupt ({e}); refusing to overwrite — fix or remove it manually",
                key_path.display()
            );
            std::process::exit(1);
        });
    }
    // First run: fresh identity.
    let keys = Keys::generate();
    if let Some(dir) = key_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let Ok(nsec) = keys.secret_key().to_bech32(); // infallible in nostr-sdk 0.42
    use std::os::unix::fs::OpenOptionsExt;
    match std::fs::OpenOptions::new().create(true).write(true).truncate(true).mode(0o600).open(&key_path) {
        Ok(mut f) => {
            let _ = f.write_all(nsec.as_bytes());
            eprintln!("[vector-bridge] generated a new identity, persisted 0600 at {}", key_path.display());
        }
        Err(e) => eprintln!("[vector-bridge] WARNING: could not persist identity ({e}) — it will change on restart"),
    }
    keys
}
