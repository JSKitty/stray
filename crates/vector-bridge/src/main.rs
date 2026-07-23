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
const MAX_FILE_BYTES: usize = 64 * 1024 * 1024; // cap a received attachment at 64 MB
const SEEN_CAP: usize = 4000;
const OUTBOX_MAX_ATTEMPTS: u32 = 10;

#[derive(serde::Deserialize)]
struct OutboxReply {
    reply_to: String,
    #[serde(default)]
    content: String,
    /// Optional local file path to send as an attachment (Stray's send_file tool).
    #[serde(default)]
    file: Option<String>,
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
        let files_dir = data.join("inbox").join("vector").join("files");
        cleanup_old_files(&files_dir, Duration::from_secs(7 * 24 * 3600));
        let trusted = load_trusted(&data);
        eprintln!("[vector-bridge] file downloads gated to {} trusted sender(s)", trusted.len());
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
                // Identity comes from the SIGNATURE-VERIFIED seal author, never
                // the spoofable rumor.pubkey.
                let from = match unwrapped.sender.to_bech32() {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                // kind 14 = text DM · kind 15 = file attachment (NIP-17).
                if unwrapped.rumor.kind == Kind::PrivateDirectMessage {
                    let mut content = unwrapped.rumor.content.clone();
                    if content.trim().is_empty() {
                        continue;
                    }
                    if content.len() > MAX_CONTENT_BYTES {
                        content.truncate(byte_boundary(&content, MAX_CONTENT_BYTES));
                    }
                    match write_inbox_event(&inbox_dir, &id, &from, &content) {
                        Ok(()) => {
                            seen.mark(&id);
                            eprintln!("[vector-bridge] inbound DM from {from} → inbox");
                        }
                        Err(e) => eprintln!("[vector-bridge] failed to write inbox event: {e}"),
                    }
                } else if unwrapped.rumor.kind == Kind::from_u16(15) {
                    // Only download files from a trusted sender. An unverified
                    // sender's file is NOT fetched (no SSRF, no disk write of
                    // attacker bytes) — just a neutral note that they offered one.
                    if !trusted.iter().any(|t| t == &from) {
                        let note = format!("[An unverified sender ({from}) offered a file over Vector. It was NOT downloaded.]");
                        if write_inbox_event(&inbox_dir, &id, &from, &note).is_ok() {
                            seen.mark(&id);
                            eprintln!("[vector-bridge] refused file from untrusted {from}");
                        }
                        continue;
                    }
                    match receive_file(&unwrapped.rumor, &id, &files_dir).await {
                        Ok((path, mime, n)) => {
                            // Neutral, provenance-accurate — no "operator", no directive.
                            let note = format!(
                                "[A file arrived over Vector from {from}; decrypted and saved on disk at {} ({}, {n} bytes).]",
                                path.display(), mime
                            );
                            match write_inbox_event(&inbox_dir, &id, &from, &note) {
                                Ok(()) => {
                                    seen.mark(&id);
                                    eprintln!("[vector-bridge] inbound FILE from {from} → {}", path.display());
                                }
                                Err(e) => eprintln!("[vector-bridge] failed to write inbox event: {e}"),
                            }
                        }
                        Err(e) => eprintln!("[vector-bridge] file receive from {from} failed: {e}"),
                    }
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
        // A file whose path is gone can never succeed — drop it (don't retry forever).
        if let Some(fp) = &reply.file {
            if !std::path::Path::new(fp).is_file() {
                eprintln!("[vector-bridge] outbox file '{fp}' missing — dropping");
                let _ = std::fs::remove_file(&path);
                attempts.remove(&name);
                continue;
            }
        }
        let channel = bot.get_chat(pk).await;
        // Send the attachment (if any) first, then the text (if any). Both must
        // succeed for the entry to be considered delivered.
        let mut ok = true;
        if let Some(fp) = &reply.file {
            match vector_sdk::AttachmentFile::from_path(fp) {
                Ok(af) => {
                    if !channel.send_private_file(Some(af)).await {
                        ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("[vector-bridge] cannot read outbox file '{fp}': {e} — dropping");
                    let _ = std::fs::remove_file(&path);
                    attempts.remove(&name);
                    continue;
                }
            }
        }
        if ok && !reply.content.trim().is_empty() {
            ok = channel.send_private_message(&reply.content).await;
        }
        if ok {
            let _ = std::fs::remove_file(&path);
            attempts.remove(&name);
            let what = if reply.file.is_some() { "file+reply" } else { "reply" };
            eprintln!("[vector-bridge] delivered {what} to {}", reply.reply_to);
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

/// Read a NIP-17 file rumor (kind 15): download the encrypted blob from the URL
/// in `content`, AES-256-GCM decrypt it with the key/nonce tags, save to
/// `<files_dir>/<id>.<ext>`. Returns (path, mime, plaintext_len).
async fn receive_file(
    rumor: &UnsignedEvent,
    id: &str,
    files_dir: &Path,
) -> Result<(PathBuf, String, usize), String> {
    let url = rumor.content.trim();
    // The url is attacker-suppliable (anyone who knows our npub can send a file),
    // so the download is a blind-SSRF surface. Require https (no plaintext /
    // internal http services), bound it with a timeout, and cap the size before
    // and after fetching. The response bytes are never returned to the sender.
    if !url.starts_with("https://") {
        return Err("attachment url is not https — refusing".into());
    }
    let key = tag_value(rumor, "decryption-key").ok_or("missing decryption-key")?;
    let nonce = tag_value(rumor, "decryption-nonce").ok_or("missing decryption-nonce")?;
    let mime = tag_value(rumor, "file-type").unwrap_or_else(|| "application/octet-stream".into());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        // No redirects: an https url that 302s to http://169.254.169.254/… would
        // otherwise defeat the https-only check and reach internal services.
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.get(url).send().await.map_err(|e| format!("download: {e}"))?;
    if let Some(len) = resp.content_length() {
        if len as usize > MAX_FILE_BYTES {
            return Err(format!("attachment too large ({len} bytes)"));
        }
    }
    let enc = resp.bytes().await.map_err(|e| format!("download body: {e}"))?;
    if enc.len() > MAX_FILE_BYTES {
        return Err(format!("attachment too large ({} bytes)", enc.len()));
    }
    let plain = decrypt_data(&enc, &key, &nonce)?;

    std::fs::create_dir_all(files_dir).map_err(|e| e.to_string())?;
    let path = files_dir.join(format!("{id}.{}", ext_from_mime(&mime)));
    std::fs::write(&path, &plain).map_err(|e| e.to_string())?;
    Ok((path, mime, plain.len()))
}

/// Mirror of vector_sdk::crypto::encrypt_data, inverted: AES-256-GCM with a
/// 16-byte nonce, the 16-byte auth tag appended to the ciphertext.
fn decrypt_data(enc: &[u8], key_hex: &str, nonce_hex: &str) -> Result<Vec<u8>, String> {
    use aes::Aes256;
    use aes_gcm::{AeadInPlace, AesGcm, KeyInit};
    use generic_array::{typenum::U16, GenericArray};

    let key = hex::decode(key_hex.trim()).map_err(|_| "bad key hex")?;
    let nonce = hex::decode(nonce_hex.trim()).map_err(|_| "bad nonce hex")?;
    if key.len() != 32 {
        return Err("key must be 32 bytes".into());
    }
    if nonce.len() != 16 {
        return Err("nonce must be 16 bytes".into());
    }
    if enc.len() < 16 {
        return Err("ciphertext shorter than the auth tag".into());
    }
    let (ct, tag) = enc.split_at(enc.len() - 16);
    let cipher = AesGcm::<Aes256, U16>::new(GenericArray::from_slice(&key));
    let mut buf = ct.to_vec();
    cipher
        .decrypt_in_place_detached(GenericArray::from_slice(&nonce), &[], &mut buf, GenericArray::from_slice(tag))
        .map_err(|_| "decryption failed (bad key/nonce/tag)".to_string())?;
    Ok(buf)
}

/// First value of a custom rumor tag, e.g. `["decryption-key", "<hex>"]`.
fn tag_value(rumor: &UnsignedEvent, name: &str) -> Option<String> {
    rumor.tags.iter().find_map(|t| {
        let s = t.as_slice();
        (s.len() >= 2 && s[0] == name).then(|| s[1].clone())
    })
}

/// A sensible file extension from a MIME type (common types + a subtype fallback).
fn ext_from_mime(mime: &str) -> &'static str {
    match mime.split(';').next().unwrap_or("").trim() {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/svg+xml" => "svg",
        "application/pdf" => "pdf",
        "application/zip" => "zip",
        "application/gzip" | "application/x-gzip" => "gz",
        "application/x-tar" => "tar",
        "application/json" => "json",
        "text/plain" => "txt",
        "text/markdown" => "md",
        "text/csv" => "csv",
        "audio/mpeg" => "mp3",
        "video/mp4" => "mp4",
        _ => "bin",
    }
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

/// Read the trusted-sender npubs for the Vector source from Stray's config, so
/// the bridge only ever downloads a file from a sender the operator has trusted.
/// (A stranger's file offer becomes a text note, never a fetch + disk write —
/// closing the SSRF / OOM / disk-fill surface at the source.)
fn load_trusted(data: &Path) -> Vec<String> {
    let Ok(s) = std::fs::read_to_string(data.join("stray.toml")) else {
        return Vec::new();
    };
    let Ok(v) = s.parse::<toml::Value>() else {
        return Vec::new();
    };
    v.get("inbox")
        .and_then(|i| i.get("sources"))
        .and_then(|s| s.get("vector"))
        .and_then(|vec| vec.get("trusted"))
        .and_then(|t| t.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

/// Best-effort retention: delete received files older than `max_age`.
fn cleanup_old_files(files_dir: &Path, max_age: Duration) {
    let Ok(entries) = std::fs::read_dir(files_dir) else {
        return;
    };
    for entry in entries.flatten() {
        if let Ok(meta) = entry.metadata() {
            if let Ok(modified) = meta.modified() {
                if modified.elapsed().map(|e| e > max_age).unwrap_or(false) {
                    let _ = std::fs::remove_file(entry.path());
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte-identical to vector_sdk::crypto::encrypt_data so the round-trip
    /// proves decrypt_data reverses real SDK-encrypted attachments.
    fn encrypt_mirror(data: &[u8], key: &[u8; 32], nonce: &[u8; 16]) -> Vec<u8> {
        use aes::Aes256;
        use aes_gcm::{AeadInPlace, AesGcm, KeyInit};
        use generic_array::{typenum::U16, GenericArray};
        let cipher = AesGcm::<Aes256, U16>::new(GenericArray::from_slice(key));
        let mut buf = data.to_vec();
        let tag = cipher
            .encrypt_in_place_detached(GenericArray::from_slice(nonce), &[], &mut buf)
            .unwrap();
        buf.extend_from_slice(tag.as_slice());
        buf
    }

    #[test]
    fn aes_gcm_roundtrip_matches_sdk_scheme() {
        let key = [7u8; 32];
        let nonce = [3u8; 16];
        let plain: &[u8] = b"quick brown fox \x00\xff binary too";
        let enc = encrypt_mirror(plain, &key, &nonce);
        let dec = decrypt_data(&enc, &hex::encode(key), &hex::encode(nonce)).unwrap();
        assert_eq!(dec, plain);
        // A wrong key must fail the GCM auth tag, not return garbage.
        assert!(decrypt_data(&enc, &hex::encode([9u8; 32]), &hex::encode(nonce)).is_err());
    }

    #[test]
    fn ext_from_mime_maps_common_types() {
        assert_eq!(ext_from_mime("application/zip"), "zip");
        assert_eq!(ext_from_mime("image/png"), "png");
        assert_eq!(ext_from_mime("text/plain; charset=utf-8"), "txt");
        assert_eq!(ext_from_mime("application/x-weird"), "bin");
    }
}
