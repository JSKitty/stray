//! Trust-driven OS sandboxing.
//!
//! Wraps an arbitrary command in a "closed box with named holes": file writes
//! are confined to one subtree, the network is gated, and only FreeRoam runs a
//! command raw. Used both for task sub-agents (`stray --task <name>`) and for
//! the main agent's `bash` tool calls.
//!
//! - macOS: `sandbox-exec` with an inline Seatbelt profile (no temp file).
//! - Linux: `bubblewrap` (bwrap) — root bound read-only, one subtree bound rw,
//!   a fresh /tmp, and an optional isolated network namespace.
//! - Fallback (neither available): run raw, with a warning.

use crate::trust::TrustLevel;
use std::path::Path;
use std::process::Command;

/// Which sandbox backend is active on this host (for display / diagnostics).
pub fn backend() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        if Path::new("/usr/bin/sandbox-exec").exists() {
            return "seatbelt";
        }
    }
    #[cfg(target_os = "linux")]
    {
        if find_bwrap().is_some() {
            return "bubblewrap";
        }
    }
    "none"
}

/// Wrap `argv` in an OS sandbox appropriate to `trust`.
///
/// - `writable_root`: the single directory subtree the payload may write to
///   (a task's dir, or the main agent's cwd) when `allow_writes` is set.
/// - `allow_writes`: may the payload write within `writable_root`?
/// - `allow_network`: may the payload use the network?
///
/// FreeRoam is never boxed — the command runs raw. If no OS sandbox backend is
/// available the command also runs raw (with a warning), so functionality never
/// silently breaks — it just isn't confined.
pub fn wrap(
    trust: TrustLevel,
    writable_root: &Path,
    argv: &[String],
    allow_writes: bool,
    allow_network: bool,
) -> Command {
    if argv.is_empty() {
        return Command::new("true");
    }
    if !trust.is_boxed() {
        return raw(argv);
    }

    #[cfg(target_os = "macos")]
    {
        if Path::new("/usr/bin/sandbox-exec").exists() {
            return wrap_macos(writable_root, argv, allow_writes, allow_network);
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(bwrap) = find_bwrap() {
            return wrap_linux(&bwrap, writable_root, argv, allow_writes, allow_network);
        }
    }

    eprintln!("[sandbox] no OS sandbox backend available — running '{}' unconfined", argv[0]);
    raw(argv)
}

fn raw(argv: &[String]) -> Command {
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]);
    cmd
}

// ---------------------------------------------------------------------------
// macOS — Seatbelt
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn wrap_macos(writable_root: &Path, argv: &[String], allow_writes: bool, allow_network: bool) -> Command {
    let profile = macos_profile(allow_writes, allow_network);
    let mut cmd = Command::new("/usr/bin/sandbox-exec");
    if allow_writes {
        // Pass the path as a named parameter, NOT interpolated into the profile
        // string — so a workspace path containing `"`, `(` or `)` can't break
        // out of / inject into the Seatbelt profile.
        cmd.arg("-D")
            .arg(format!("WRITABLE_ROOT={}", writable_root.to_string_lossy()));
    }
    cmd.arg("-p").arg(profile);
    cmd.args(argv);
    cmd
}

#[cfg(target_os = "macos")]
fn macos_profile(allow_writes: bool, allow_network: bool) -> String {
    // The writable root arrives via `-D WRITABLE_ROOT=…` and is referenced with
    // `(param …)`, so it's only present when allow_writes (else the param is
    // undefined and sandbox-exec would error).
    let root_write = if allow_writes {
        "(allow file-write* (subpath (param \"WRITABLE_ROOT\")))"
    } else {
        ""
    };
    let network = if allow_network {
        "(allow network*)\n(allow system-socket)"
    } else {
        ""
    };
    // Reads are broad (tools read project files, dylibs, etc.); writes are
    // default-deny with a few named holes: the writable root, scratch temp, /dev.
    format!(
        r#"(version 1)
(deny default)
(allow file-read*)
(allow process-fork)
(allow process-exec*)
(allow sysctl-read)
(allow signal (target self))
(allow mach*)
(allow file-write* (subpath "/dev"))
(allow file-write* (subpath "/private/tmp"))
(allow file-write* (subpath "/tmp"))
{root_write}
{network}
"#
    )
}

// ---------------------------------------------------------------------------
// Linux — bubblewrap
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn find_bwrap() -> Option<String> {
    for p in ["/usr/bin/bwrap", "/bin/bwrap", "/usr/local/bin/bwrap"] {
        if Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn wrap_linux(
    bwrap: &str,
    writable_root: &Path,
    argv: &[String],
    allow_writes: bool,
    allow_network: bool,
) -> Command {
    let root = writable_root.to_string_lossy().to_string();
    let mut cmd = Command::new(bwrap);
    // Whole filesystem read-only, then punch a rw hole for the writable root.
    cmd.arg("--ro-bind").arg("/").arg("/")
        .arg("--dev").arg("/dev")
        .arg("--proc").arg("/proc")
        .arg("--tmpfs").arg("/tmp")
        .arg("--die-with-parent");
    if allow_writes {
        cmd.arg("--bind").arg(&root).arg(&root);
    }
    if !allow_network {
        cmd.arg("--unshare-net");
    }
    // Start in the writable root if it exists (the payload may chdir further).
    cmd.arg("--chdir").arg(&root);
    cmd.arg("--").args(argv);
    cmd
}
