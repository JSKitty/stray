# Stray

```
  /\_/\    ╔═╗╔╦╗╦═╗╔═╗╦ ╦
 ( o.o )   ╚═╗ ║ ╠╦╝╠═╣╚╦╝
  >   <    ╚═╝ ╩ ╩╚═╩ ╩ ╩
```

A living AI companion for the terminal. Autonomous, sovereign, with a heartbeat. No leash required.

## Features

- **Extremely lightweight** — <10MB RAM, no bloat, raw ANSI + libc
- **Custom TUI engine** — high-performance full-screen renderer with animations, syntax highlighting, and streaming fade-in
- **Deep config system** — per-provider settings with hot-swap, local and global configs
- **Interactive or Autonomous modes** — use as a chat companion or an autonomous agent with heartbeat
- **Departments** — sandboxed sub-agents that work independently, with roles, persistent workspaces, and auto-notifications
- **Self-updating** — `/update` checks for and installs the latest version
- **LMStudio and PPQ support** — plus any OpenAI-compatible API

## Install

```bash
curl -sSf https://stray.jskitty.cat/install.sh | sh
```

Or build from source:

```bash
cargo install --git https://github.com/JSKitty/stray
```

## Quick Start

```bash
stray
```

On first run, a setup wizard guides you through connecting to your LLM provider (LMStudio, PPQ, or any OpenAI-compatible API).

Or create `stray.toml` in your project directory:

```toml
[agent]
name = "Stray"
heartbeat = 300
compact_at = 100000
system_prompt = "You are a helpful assistant."

[llm]
api_url = "http://127.0.0.1:1234/v1/chat/completions"
api_key = "lm-studio"
model = "your-model-name"
max_tokens = 4096
vision = false
```

Type `/help` for a list of commands.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

### Commercial Licensing

For commercial use without AGPL obligations, a separate commercial license is available.

**Contact:**
- Email: [mail@jskitty.cat](mailto:mail@jskitty.cat)
- Web: [jskitty.cat](https://jskitty.cat)
- Vector: [JSKitty on Vector](https://vectorapp.io/profile/npub16ye7evyevwnl0fc9hujsxf9zym72e063awn0pvde0huvpyec5nyq4dg4wn)

---

*Built with love and purrs.*
