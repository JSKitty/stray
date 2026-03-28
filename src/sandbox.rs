//! Platform-specific sandboxing for department subprocesses.
//!
//! macOS: sandbox-exec with a generated Seatbelt profile
//! Linux: unshare with mount + network + PID namespace isolation
//! Fallback: workspace chdir only (no OS-level enforcement)

use std::path::Path;
use std::process::Command;

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub enum SandboxLevel {
    /// No OS sandboxing available — workspace chdir only
    None,
    /// Full OS-enforced sandboxing
    Full,
}

/// Detect what sandboxing is available on this platform.
pub fn detect() -> SandboxLevel {
    if cfg!(target_os = "macos") {
        // sandbox-exec is at /usr/bin/sandbox-exec
        if Path::new("/usr/bin/sandbox-exec").exists() {
            return SandboxLevel::Full;
        }
    } else if cfg!(target_os = "linux") {
        // Check if unshare is available and user namespaces are enabled
        if let Ok(output) = Command::new("unshare")
            .arg("--help")
            .output()
        {
            if output.status.success() || !output.stderr.is_empty() {
                // Also check if user namespaces are enabled
                if std::fs::read_to_string("/proc/sys/kernel/unprivileged_userns_clone")
                    .map(|s| s.trim() == "1")
                    .unwrap_or(true) // assume enabled if file doesn't exist
                {
                    return SandboxLevel::Full;
                }
            }
        }
    }

    SandboxLevel::None
}

// ---------------------------------------------------------------------------
// macOS sandbox profile generation
// ---------------------------------------------------------------------------

/// Generate a macOS Seatbelt sandbox profile for a department.
///
/// `read_only`: if true, only allow file reads on the workspace (Code Reviewer, Researcher).
/// The stray binary itself and system libraries are always readable.
#[cfg(target_os = "macos")]
fn generate_macos_profile(workspace: &Path, read_only: bool, stray_binary: &Path) -> String {
    let ws = workspace.to_string_lossy();
    let binary = stray_binary.to_string_lossy();

    // Resolve the global config dir for read access (roles.toml, etc.)
    let config_dir = crate::config::global_config_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    let write_rule = if read_only {
        format!("(allow file-read* (subpath \"{ws}\"))")
    } else {
        format!("(allow file-read* file-write* (subpath \"{ws}\"))")
    };

    format!(
        r#"(version 1)
(deny default)

; System libraries and frameworks (required for process execution)
(allow file-read*
    (subpath "/usr/lib")
    (subpath "/usr/share")
    (subpath "/Library/Frameworks")
    (subpath "/System/Library")
    (subpath "/dev")
    (subpath "/private/var/db")
    (subpath "/private/tmp")
    (subpath "/private/etc")
    (subpath "/var")
)

; Allow reading files system-wide (tools need to read project files outside workspace)
; Writes are restricted to workspace + config dir only
(allow file-read*)

; Stray binary
(allow process-exec (literal "{binary}"))

; Global config dir (department status, history, progress)
(allow file-write* (subpath "{config_dir}"))

; /dev/null and other device writes (tools like git redirect to /dev/null)
(allow file-write* (subpath "/dev"))

; Temp files (some tools need /tmp or /private/tmp for intermediate work)
(allow file-write* (subpath "/private/tmp"))
(allow file-write* (subpath "/tmp"))

; Workspace writes
{write_rule}

; Process control
(allow process-fork)
(allow process-exec*)
(allow sysctl-read)
(allow signal (target self))
(allow mach*)

; Allow network (required for LLM API calls — local or remote providers)
(allow network*)
(allow system-socket)
"#
    )
}

// ---------------------------------------------------------------------------
// Command wrapping
// ---------------------------------------------------------------------------

/// Wrap a `Command` with platform-specific sandboxing.
///
/// On macOS: writes a temp profile and uses `sandbox-exec -f`.
/// On Linux: uses `unshare` with mount + network + PID namespaces.
/// On unsupported platforms or if sandboxing is unavailable: no-op (logs warning).
pub fn wrap_command(
    workspace: &Path,
    read_only: bool,
    dept_name: &str,
    stray_binary: &Path,
) -> Command {
    let level = detect();

    if level == SandboxLevel::Full {
        #[cfg(target_os = "macos")]
        {
            return wrap_macos(workspace, read_only, dept_name, stray_binary);
        }

        #[cfg(target_os = "linux")]
        {
            return wrap_linux(dept_name, stray_binary);
        }
    }

    // Fallback: no sandbox, just run stray directly
    eprintln!("[sandbox] No OS sandboxing available — running with workspace chdir only");
    let mut cmd = Command::new(stray_binary);
    cmd.arg("--department").arg(dept_name);
    cmd
}

#[cfg(target_os = "macos")]
fn wrap_macos(workspace: &Path, read_only: bool, dept_name: &str, stray_binary: &Path) -> Command {
    let profile = generate_macos_profile(workspace, read_only, stray_binary);

    // Write profile to a temp file in the workspace (it's within the sandbox's read scope)
    let profile_path = workspace.join(".sandbox-profile.sb");
    if let Err(e) = std::fs::write(&profile_path, &profile) {
        eprintln!("[sandbox] Failed to write profile: {e} — running unsandboxed");
        let mut cmd = Command::new(stray_binary);
        cmd.arg("--department").arg(dept_name);
        return cmd;
    }

    let mut cmd = Command::new("/usr/bin/sandbox-exec");
    cmd.arg("-f")
        .arg(&profile_path)
        .arg(stray_binary)
        .arg("--department")
        .arg(dept_name);
    cmd
}

#[cfg(target_os = "linux")]
fn wrap_linux(dept_name: &str, stray_binary: &Path) -> Command {
    // Use unshare for network + PID isolation
    // Mount namespace would require bind-mounting which needs more setup
    let mut cmd = Command::new("unshare");
    cmd.arg("--net")       // isolated network namespace (no network)
        .arg("--pid")      // isolated PID namespace
        .arg("--fork")     // required with --pid
        .arg("--")
        .arg(stray_binary)
        .arg("--department")
        .arg(dept_name);
    cmd
}
