//! Trust levels — how much power an agent context (the main agent OR a task)
//! is granted over the system. The model is "a closed box with named holes",
//! not an on/off switch: an agent starts sealed and only the capabilities it
//! genuinely needs are cut open.
//!
//! The same ladder applies uniformly to the interactive agent and to task
//! sub-agents — isolation is a property of every context, not just tasks.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// How much power a context is trusted with, weakest → strongest.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrustLevel {
    /// Read broadly, but no writes outside the workspace and no network.
    /// Safe for untrusted work; the default for every context.
    Sandboxed,
    /// Writes allowed within the working-directory tree; still no network.
    Workspace,
    /// Workspace writes plus explicitly named holes (commands + paths) — the
    /// command-center level that administers a machine by least privilege.
    Admin,
    /// No box: full system access. The deliberate, opt-in escape hatch.
    FreeRoam,
}

impl Default for TrustLevel {
    // The everyday default: an isolated workspace (writes confined to the cwd
    // tree) but with network on, so a coding companion can still git/curl/install.
    // Drop to Sandboxed to also cut the network (for untrusted code).
    fn default() -> Self {
        TrustLevel::Workspace
    }
}

impl TrustLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            TrustLevel::Sandboxed => "sandboxed",
            TrustLevel::Workspace => "workspace",
            TrustLevel::Admin => "admin",
            TrustLevel::FreeRoam => "free-roam",
        }
    }

    /// Parse leniently from user input (accepts a few aliases and `_`/`-`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().replace('_', "-").as_str() {
            "sandboxed" | "sandbox" | "sealed" => Some(TrustLevel::Sandboxed),
            "workspace" | "work" => Some(TrustLevel::Workspace),
            "admin" | "caretaker" => Some(TrustLevel::Admin),
            "free-roam" | "freeroam" | "free" | "roam" | "off" => Some(TrustLevel::FreeRoam),
            _ => None,
        }
    }

    /// STRICT parse for an authoritative STORED value (e.g. a peer's granted
    /// trust in peers.toml). Only the exact canonical `as_str()` forms — NO
    /// lenient aliases — so a hand-edited or malformed `"off"`/`"free"` can never
    /// silently resolve to free-roam (remote root). Unknown → None (unauthorized).
    pub fn parse_canonical(s: &str) -> Option<Self> {
        match s {
            "sandboxed" => Some(TrustLevel::Sandboxed),
            "workspace" => Some(TrustLevel::Workspace),
            "admin" => Some(TrustLevel::Admin),
            "free-roam" => Some(TrustLevel::FreeRoam),
            _ => None,
        }
    }

    pub const ALL: [TrustLevel; 4] = [
        TrustLevel::Sandboxed,
        TrustLevel::Workspace,
        TrustLevel::Admin,
        TrustLevel::FreeRoam,
    ];

    /// One-line human description of what this level permits.
    pub fn describe(self) -> &'static str {
        match self {
            TrustLevel::Sandboxed => "reads anywhere · read-only · no network — for untrusted code",
            TrustLevel::Workspace => "writes within the workspace · network on — the default",
            TrustLevel::Admin => "workspace writes · network · plus explicitly named admin holes",
            TrustLevel::FreeRoam => "⚠ no sandbox — full system access",
        }
    }

    /// May a tool write to `path` (given the context's workspace root)?
    /// Enforced for in-process tools (write/edit) that can't be OS-sandboxed.
    /// Sandboxed is read-only; Workspace confines writes to the cwd tree;
    /// Admin/FreeRoam are unrestricted here (Admin is gated at the sandbox layer).
    pub fn allows_write(self, path: &Path, workspace: &Path) -> bool {
        match self {
            TrustLevel::Sandboxed => false,
            TrustLevel::Workspace => path_within(path, workspace),
            TrustLevel::Admin | TrustLevel::FreeRoam => true,
        }
    }

    /// Do tool subprocesses (bash) get an OS sandbox wrapper at all?
    /// FreeRoam runs raw; every other level is boxed.
    pub fn is_boxed(self) -> bool {
        self != TrustLevel::FreeRoam
    }

    /// May a MAIN-AGENT tool subprocess (e.g. bash) reach the network? On for
    /// every level except the strict Sandboxed one (so untrusted code can't
    /// phone home).
    ///
    /// NOTE: this does NOT apply to task sub-agents. A task is a whole `stray`
    /// process that makes its own LLM calls, so it *structurally* needs the
    /// network regardless of trust — a Sandboxed task is write-confined but not
    /// network-isolated. Its trust confines the blast radius (writes), not egress.
    pub fn allows_network(self) -> bool {
        self != TrustLevel::Sandboxed
    }
}

/// Best-effort check that `path` resolves inside `root`. Compares against the
/// nearest existing ancestor so not-yet-created files are still evaluated.
pub fn path_within(path: &Path, root: &Path) -> bool {
    let root = canonical_or_self(root);
    let target = canonical_or_self(path);
    target.starts_with(&root)
}

fn canonical_or_self(p: &Path) -> PathBuf {
    // Fully resolves symlinks + `..` when the whole path exists.
    if let Ok(c) = p.canonicalize() {
        return c;
    }
    // Absolutize (relative paths resolve against cwd), canonicalize the longest
    // existing prefix, then LEXICALLY normalize the remainder — so a `..` in a
    // not-yet-existing tail can't slip past a lexical starts_with and escape.
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    };
    let mut ancestors = abs.ancestors();
    let _ = ancestors.next();
    for anc in ancestors {
        if let Ok(c) = anc.canonicalize() {
            if let Ok(tail) = abs.strip_prefix(anc) {
                return normalize_lexical(&c.join(tail));
            }
        }
    }
    normalize_lexical(&abs)
}

/// Resolve `.` and `..` components lexically, without touching the filesystem.
fn normalize_lexical(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn parse_roundtrip_and_aliases() {
        for lvl in TrustLevel::ALL {
            assert_eq!(TrustLevel::parse(lvl.as_str()), Some(lvl));
        }
        assert_eq!(TrustLevel::parse("SANDBOX"), Some(TrustLevel::Sandboxed));
        assert_eq!(TrustLevel::parse("free_roam"), Some(TrustLevel::FreeRoam));
        assert_eq!(TrustLevel::parse("nonsense"), None);
        assert_eq!(TrustLevel::default(), TrustLevel::Workspace);
    }

    #[test]
    fn write_scoping() {
        let ws = Path::new("/tmp");
        // Sandboxed is read-only — even inside the workspace.
        assert!(!TrustLevel::Sandboxed.allows_write(Path::new("/tmp/x/y.txt"), ws));
        // Workspace allows writes within cwd but not outside.
        assert!(TrustLevel::Workspace.allows_write(Path::new("/tmp/x/y.txt"), ws));
        assert!(!TrustLevel::Workspace.allows_write(Path::new("/etc/passwd"), ws));
        // Admin/FreeRoam unrestricted at this (in-process) layer.
        assert!(TrustLevel::Admin.allows_write(Path::new("/etc/passwd"), ws));
        assert!(TrustLevel::FreeRoam.allows_write(Path::new("/etc/passwd"), ws));
    }

    #[test]
    fn write_scoping_blocks_dotdot_escape() {
        // Regression: a `..` in a NOT-YET-EXISTING path must not lexically slip
        // past the workspace root (the earlier bug: canonical_or_self left `..`
        // unresolved and starts_with was purely lexical).
        let ws = std::env::temp_dir();
        let ws = ws.canonicalize().unwrap_or(ws);
        let escape = ws.join("nope/../../../../../../etc/passwd");
        assert!(
            !TrustLevel::Workspace.allows_write(&escape, &ws),
            "Workspace must reject a `..` escape out of the workspace"
        );
        // A nested, not-yet-existing file inside the workspace is still allowed.
        let inside = ws.join("sub/dir/newfile.txt");
        assert!(
            TrustLevel::Workspace.allows_write(&inside, &ws),
            "Workspace must allow a nested new file inside the workspace"
        );
    }
}
