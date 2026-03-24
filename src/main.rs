mod config;
mod event;
mod formats;
mod markdown;
mod term;
mod tools;
mod ui;

use config::{Config, ConfigSource, LlmConfig};
use event::Event;
use formats::ModelFormat;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{BufRead, BufReader};
use std::process::Command;
use std::sync::mpsc::Receiver;
use std::time::Duration;
use term::*;
use tools::read::IMAGE_MARKER;
use tools::ToolRegistry;
use ui::*;

const MAX_TOOL_ROUNDS: usize = 25;
const CHARS_PER_TOKEN: usize = 4;

// ---------------------------------------------------------------------------
// LLM client
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

fn message_to_json(msg: &Message, vision: bool) -> serde_json::Value {
    if vision && msg.content.contains(IMAGE_MARKER) {
        let mut parts: Vec<serde_json::Value> = Vec::new();
        let mut remaining = msg.content.as_str();
        while let Some(start) = remaining.find(IMAGE_MARKER) {
            let text_before = remaining[..start].trim();
            if !text_before.is_empty() {
                parts.push(json!({"type": "text", "text": text_before}));
            }
            let data_start = start + IMAGE_MARKER.len();
            if let Some(end) = remaining[data_start..].find(']') {
                let data_url = &remaining[data_start..data_start + end];
                parts.push(json!({"type": "image_url", "image_url": {"url": data_url}}));
                remaining = &remaining[data_start + end + 1..];
            } else {
                parts.push(json!({"type": "text", "text": &remaining[start..]}));
                remaining = "";
                break;
            }
        }
        let trailing = remaining.trim();
        if !trailing.is_empty() {
            parts.push(json!({"type": "text", "text": trailing}));
        }
        json!({"role": msg.role, "content": parts})
    } else {
        json!({"role": msg.role, "content": msg.content})
    }
}

struct LlmResponse {
    content: String,
    total_tokens: Option<usize>,
    disturbed: bool,
}

/// Call the LLM with SSE streaming. Updates AppState for display if provided.
fn call_llm(
    config: &LlmConfig,
    messages: &[Message],
    tools_json: &Option<serde_json::Value>,
    mut state: Option<&mut AppState>,
    event_rx: Option<&Receiver<Event>>,
    tool_tags: &[&str],
    queued_messages: &mut Vec<String>,
) -> Result<LlmResponse, String> {
    let messages_json: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| message_to_json(m, config.vision))
        .collect();

    let mut body = json!({
        "model": config.model,
        "messages": messages_json,
        "max_tokens": config.max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
    });

    if let Some(tools) = tools_json {
        body["tools"] = tools.clone();
    }

    // Move the ENTIRE HTTP request + SSE reading to a background thread
    // so the main thread never blocks on network I/O
    let body_str = body.to_string();
    let api_url = config.api_url.clone();
    let api_key = config.api_key.clone();

    let (sse_tx, sse_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    std::thread::spawn(move || {
        let resp = match ureq::post(&api_url)
            .set("Authorization", &format!("Bearer {}", api_key))
            .set("Content-Type", "application/json")
            .timeout(Duration::from_secs(120))
            .send_string(&body_str)
        {
            Ok(r) => r,
            Err(e) => {
                let _ = sse_tx.send(Err(format!("LLM request failed: {}", e)));
                return;
            }
        };
        let reader = BufReader::new(resp.into_reader());
        for line in reader.lines() {
            match line {
                Ok(l) => { if sse_tx.send(Ok(l)).is_err() { break; } }
                Err(_) => { let _ = sse_tx.send(Err("Stream read error".into())); break; }
            }
        }
        let _ = sse_tx.send(Ok("[SSE_DONE]".into()));
    });

    let mut content = String::new();
    let mut total_tokens: Option<usize> = None;
    let mut streaming_started = false;
    let mut was_disturbed = false;
    let mut tag_buf = String::new();
    let mut inside_tool_tag = false;

    let mut tc_names: Vec<String> = Vec::new();
    let mut tc_args: Vec<String> = Vec::new();

    loop {
        // Process pending UI events (tick, resize, disturb)
        if let Some(rx) = event_rx {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    Event::Key(Key::CtrlC) | Event::Key(Key::Escape) => {
                        was_disturbed = true;
                    }
                    Event::Resize => {
                        if let Some(ref mut s) = state {
                            s.update_dimensions();
                            s.render();
                        }
                    }
                    Event::Tick => {
                        if let Some(ref mut s) = state {
                            if s.spinner.active {
                                s.spinner.frame += 1;
                            }
                            s.render();
                        }
                    }
                    // Handle input during agent execution (queue messages)
                    Event::Key(key) => {
                        if let Some(ref mut s) = state {
                            match key {
                                Key::Char(ch) => { s.input.buf.insert(s.input.cursor, ch); s.input.cursor += 1; }
                                Key::Paste(text) => { for ch in text.chars() { s.input.buf.insert(s.input.cursor, ch); s.input.cursor += 1; } }
                                Key::Backspace => { if s.input.cursor > 0 { s.input.buf.remove(s.input.cursor - 1); s.input.cursor -= 1; } }
                                Key::Delete => { if s.input.cursor < s.input.buf.len() { s.input.buf.remove(s.input.cursor); } }
                                Key::Left => { if s.input.cursor > 0 { s.input.cursor -= 1; } }
                                Key::Right => { if s.input.cursor < s.input.buf.len() { s.input.cursor += 1; } }
                                Key::Home => { s.input.cursor = 0; }
                                Key::End => { s.input.cursor = s.input.buf.len(); }
                                Key::Enter => {
                                    let line: String = s.input.buf.drain(..).collect();
                                    s.input.cursor = 0;
                                    let trimmed = line.trim().to_string();
                                    if !trimmed.is_empty() {
                                        queued_messages.push(trimmed.clone());
                                        s.push_chat(ChatLine { kind: ChatLineKind::QueuedMessage, content: trimmed });
                                    }
                                }
                                Key::Tab => {
                                    let suggestion = slash_suggestion(&s.input.buf);
                                    for ch in suggestion.chars() { s.input.buf.push(ch); }
                                    s.input.cursor = s.input.buf.len();
                                }
                                _ => {}
                            }
                            s.render();
                        }
                    }
                }
            }
        }
        if was_disturbed { break; }

        // Poll SSE reader with short timeout (keeps UI responsive)
        let line = match sse_rx.recv_timeout(Duration::from_millis(16)) {
            Ok(Ok(l)) => {
                if l == "[SSE_DONE]" { break; }
                l
            }
            Ok(Err(e)) => {
                // HTTP or stream error from background thread
                if let Some(ref mut s) = state {
                    s.stop_spinner();
                    s.push_chat(ChatLine { kind: ChatLineKind::Error, content: e });
                    s.render();
                }
                return Ok(LlmResponse { content, total_tokens, disturbed: true });
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };

        let data = match line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) {
            Some(d) => d.trim_start(),
            None => continue,
        };

        if data == "[DONE]" { break; }

        let chunk: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Content delta
        if let Some(text) = chunk["choices"][0]["delta"]["content"].as_str() {
            if !text.is_empty() {
                content.push_str(text);

                if let Some(ref mut s) = state {
                    // Filter tool XML tags from display
                    let mut display_text = String::new();
                    for ch in text.chars() {
                        if inside_tool_tag {
                            tag_buf.push(ch);
                            for tag in tool_tags {
                                if tag_buf.ends_with(&format!("</{tag}>")) {
                                    inside_tool_tag = false;
                                    tag_buf.clear();
                                    break;
                                }
                            }
                            continue;
                        }
                        if ch == '<' || !tag_buf.is_empty() {
                            tag_buf.push(ch);
                            if ch == '>' {
                                if tool_tags.iter().any(|t| tag_buf == format!("<{t}>")) {
                                    inside_tool_tag = true;
                                    continue;
                                }
                                display_text.push_str(&tag_buf);
                                tag_buf.clear();
                            }
                            continue;
                        }
                        display_text.push(ch);
                    }

                    if !display_text.is_empty() {
                        if !streaming_started {
                            // Wait until we have non-whitespace content
                            let check = display_text.trim_start();
                            if !check.is_empty() {
                                s.stop_spinner();
                                s.begin_streaming();
                                s.append_streaming(check);
                                streaming_started = true;
                            }
                        } else {
                            s.append_streaming(&display_text);
                        }
                        s.maybe_render();
                    }
                }
            }
        }

        // Tool calls delta
        if let Some(tc_arr) = chunk["choices"][0]["delta"]["tool_calls"].as_array() {
            for tc in tc_arr {
                let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                while tc_names.len() <= idx {
                    tc_names.push(String::new());
                    tc_args.push(String::new());
                }
                if let Some(name) = tc["function"]["name"].as_str() {
                    tc_names[idx].push_str(name);
                }
                if let Some(args) = tc["function"]["arguments"].as_str() {
                    tc_args[idx].push_str(args);
                }
            }
        }

        if let Some(t) = chunk["usage"]["total_tokens"].as_u64() {
            total_tokens = Some(t as usize);
        }
    }

    // Finalize streaming
    if streaming_started {
        if was_disturbed {
            if let Some(ref mut s) = state {
                s.append_streaming(" [disturbed]");
            }
        }
        if let Some(ref mut s) = state {
            s.end_streaming();
            s.render();
        }
    } else if was_disturbed {
        if let Some(ref mut s) = state {
            s.stop_spinner();
            s.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: "  [disturbed]".into() });
            s.render();
        }
    }

    // Convert tool_calls to inline XML
    if was_disturbed { tc_names.clear(); }
    for i in 0..tc_names.len() {
        if !tc_names[i].is_empty() {
            let args: serde_json::Value = serde_json::from_str(&tc_args[i]).unwrap_or(json!({}));
            let command = args["command"].as_str().unwrap_or("");
            content.push_str(&format!("\n<{}>{}</{}>", tc_names[i], command, tc_names[i]));
        }
    }

    Ok(LlmResponse { content, total_tokens, disturbed: was_disturbed })
}

// ---------------------------------------------------------------------------
// Token estimation & compaction
// ---------------------------------------------------------------------------

const TOKENS_PER_IMAGE: usize = 1000;

fn estimate_tokens(messages: &[Message]) -> usize {
    messages.iter().map(|m| {
        let content = &m.content;
        let mut tokens = 4;
        let mut remaining = content.as_str();
        while let Some(start) = remaining.find(IMAGE_MARKER) {
            tokens += start / CHARS_PER_TOKEN;
            let data_start = start + IMAGE_MARKER.len();
            if let Some(end) = remaining[data_start..].find(']') {
                tokens += TOKENS_PER_IMAGE;
                remaining = &remaining[data_start + end + 1..];
            } else {
                remaining = &remaining[start..];
                break;
            }
        }
        tokens += remaining.len() / CHARS_PER_TOKEN;
        tokens
    }).sum()
}

const COMPACT_PROMPT: &str = "\
You are being asked to compact your conversation context. The conversation so far \
will be replaced by your summary. Preserve EVERYTHING important:\n\
1. **State**: What is the current state of your tasks, wallet, environment?\n\
2. **Pending**: What were you working on or planning to do next?\n\
3. **Learnings**: What have you discovered that affects future decisions?\n\
4. **Key data**: Any addresses, balances, transaction IDs, error messages, or other specifics you'll need.\n\n\
Be thorough but concise. This summary is your ONLY memory — anything you don't include is lost forever. \
Write in first person as if briefing your future self.";

fn compact_context(config: &LlmConfig, messages: &mut Vec<Message>, token_count: usize, state: &mut AppState) {
    state.push_chat(ChatLine {
        kind: ChatLineKind::SystemInfo,
        content: format!("[compact] Context at ~{token_count} tokens — compacting..."),
    });
    state.render();

    messages.push(Message { role: "user".into(), content: COMPACT_PROMPT.into() });

    let resp = match call_llm(config, messages, &None, None, None, &[], &mut Vec::new()) {
        Ok(r) => r,
        Err(e) => {
            messages.pop();
            state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("[compact] Failed: {e}") });
            state.render();
            return;
        }
    };

    let system = messages.first().cloned().unwrap_or(Message { role: "system".into(), content: String::new() });
    messages.clear();
    messages.push(system);
    messages.push(Message {
        role: "assistant".into(),
        content: format!("[Context compacted from ~{token_count} tokens]\n\n{}", resp.content),
    });

    let new_tokens = estimate_tokens(messages);
    state.push_chat(ChatLine {
        kind: ChatLineKind::SystemInfo,
        content: format!("[compact] Reduced to ~{new_tokens} tokens ({:.0}% reduction)",
            if token_count > 0 { (1.0 - new_tokens as f64 / token_count as f64) * 100.0 } else { 0.0 }),
    });
    state.render();
}

// ---------------------------------------------------------------------------
// Agent loop
// ---------------------------------------------------------------------------

fn run_heartbeat(
    config: &Config,
    registry: &ToolRegistry,
    format: &dyn ModelFormat,
    tools_json: &Option<serde_json::Value>,
    messages: &mut Vec<Message>,
    user_messages: Vec<String>,
    state: &mut AppState,
    event_rx: &Receiver<Event>,
    queued_messages: &mut Vec<String>,
) -> (Option<usize>, Vec<String>) {
    if user_messages.is_empty() {
        messages.push(Message {
            role: "user".into(),
            content: format!("[{}] Heartbeat. Check in and do your tasks.", timestamp()),
        });
    } else {
        let combined = user_messages.join("\n\n");
        messages.push(Message {
            role: "user".into(),
            content: format!("[{}] The operator says:\n{}", timestamp(), combined),
        });
    }

    state.start_spinner("thinking...");
    state.render();
    let mut last_tokens: Option<usize> = None;
    let tags = registry.tags();

    for round in 0..MAX_TOOL_ROUNDS {
        let resp = match call_llm(&config.llm, messages, tools_json, Some(state), Some(event_rx), &tags, queued_messages) {
            Ok(r) => r,
            Err(e) => {
                state.stop_spinner();
                state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("[error] {e}") });
                state.render();
                return (last_tokens, queued_messages.drain(..).collect());
            }
        };

        if let Some(t) = resp.total_tokens {
            last_tokens = Some(t);
        }

        if resp.disturbed {
            let partial = if resp.content.trim().is_empty() {
                "[no response generated]".to_string()
            } else {
                format!("{} [interrupted]", resp.content)
            };
            messages.push(Message { role: "assistant".into(), content: partial });
            messages.push(Message {
                role: "user".into(),
                content: format!("[{}] You were disturbed by the operator — your previous response was cut short.", timestamp()),
            });
            break;
        }

        let (calls, _) = format.parse_response(&resp.content);
        messages.push(Message { role: "assistant".into(), content: resp.content });

        if calls.is_empty() { break; }

        // Execute tool calls
        state.start_spinner("executing...");
        state.render();
        let mut results: Vec<(String, String, String)> = Vec::new();
        for call in &calls {
            let tool = registry.tools().iter().find(|t| t.name() == call.tool);
            match tool {
                Some(t) => {
                    let action = t.display_action(&call.input);
                    let title = t.name().chars().next().unwrap_or(' ').to_uppercase().to_string()
                        + &t.name()[1..];
                    state.stop_spinner();
                    state.push_chat(ChatLine {
                        kind: ChatLineKind::ToolAction,
                        content: format!("{title} → {action}"),
                    });
                    state.render();
                    let output = t.execute(&call.input);
                    results.push((call.tool.clone(), call.input.clone(), output));
                    state.start_spinner("thinking...");
                    state.render();
                }
                None => {
                    results.push((call.tool.clone(), call.input.clone(), format!("[error] Unknown tool: {}", call.tool)));
                }
            }
        }

        messages.push(Message { role: "user".into(), content: format.format_results(&results) });

        if round == MAX_TOOL_ROUNDS - 1 {
            state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Max tool rounds ({MAX_TOOL_ROUNDS}) reached") });
        }
    }

    state.stop_spinner();
    state.render();
    (last_tokens, queued_messages.drain(..).collect())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn build_system_prompt(config: &Config, format: &dyn ModelFormat, registry: &ToolRegistry, cwd: &str) -> String {
    format!(
        "{}\n\nToday is {}. Timestamps in messages are local time (HH:MM). Working directory: {}{}",
        config.agent.system_prompt, date_today(), cwd, format.system_prompt_suffix(registry)
    )
}

fn main() {
    // Panic hook: restore terminal
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Leave alternate screen + restore terminal
        print!("\x1b[?1049l");
        restore_terminal();
        default_hook(info);
    }));

    // Load config (wizard runs before TUI if needed)
    let loaded = config::load();
    let mut config = loaded.config;
    let config_path = loaded.path;
    let config_source = match loaded.source {
        ConfigSource::Cli => "cli",
        ConfigSource::Local => "local",
        ConfigSource::Global => "global",
        ConfigSource::Wizard => "new",
    };

    // Register tools
    let registry = {
        let mut r = ToolRegistry::new();
        r.add(Box::new(tools::BashTool));
        r.add(Box::new(tools::ReadTool::new(config.llm.vision)));
        r
    };

    let mut format = formats::format_for_model(&config.llm.model, &registry);
    let mut tools_json = format.format_tools(&registry);
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let system = build_system_prompt(&config, &*format, &registry, &cwd);
    let tools_str = registry.tools().iter().map(|t| t.name()).collect::<Vec<_>>().join(", ");
    let mut fmt_str = if tools_json.is_some() { "openai-api" } else { "prompt-xml" };

    // Fetch models for autocomplete
    update_model_cache(&config.llm.api_url, &config.llm.api_key);

    // Enter alternate screen
    print!("\x1b[?1049h");
    let _ = std::io::Write::flush(&mut std::io::stdout());

    // Build TUI state
    let mut state = AppState::new();
    state.build_header(&config, &tools_str, fmt_str, &config_path, config_source);
    state.render();

    let mut messages: Vec<Message> = vec![Message { role: "system".into(), content: system }];

    // Event channels (input thread + tick thread + resize watcher)
    let event_rx = event::setup_event_channels();

    // Ctrl+C double-tap state
    let mut last_ctrlc: u64 = 0;

    // --- Main event loop ---
    loop {
        // PROMPT PHASE
        state.input.active = true;
        state.render();

        let mut user_input: Option<String> = None;
        let deadline = std::time::Instant::now() + Duration::from_secs(config.agent.heartbeat);

        'prompt: loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break 'prompt; // heartbeat timeout
            }

            match event_rx.recv_timeout(remaining.min(Duration::from_millis(100))) {
                Ok(Event::Key(key)) => {
                    match key {
                        Key::Char(ch) => {
                            state.input.buf.insert(state.input.cursor, ch);
                            state.input.cursor += 1;
                            last_ctrlc = 0;
                        }
                        Key::Paste(text) => {
                            for ch in text.chars() {
                                state.input.buf.insert(state.input.cursor, ch);
                                state.input.cursor += 1;
                            }
                            last_ctrlc = 0;
                        }
                        Key::Enter => {
                            let line: String = state.input.buf.drain(..).collect();
                            state.input.cursor = 0;
                            let trimmed = line.trim().to_string();
                            if !trimmed.is_empty() {
                                user_input = Some(trimmed);
                                break 'prompt;
                            }
                            last_ctrlc = 0;
                        }
                        Key::Backspace => {
                            if state.input.cursor > 0 {
                                state.input.buf.remove(state.input.cursor - 1);
                                state.input.cursor -= 1;
                            }
                            last_ctrlc = 0;
                        }
                        Key::Delete => {
                            if state.input.cursor < state.input.buf.len() {
                                state.input.buf.remove(state.input.cursor);
                            }
                            last_ctrlc = 0;
                        }
                        Key::Left => { if state.input.cursor > 0 { state.input.cursor -= 1; } last_ctrlc = 0; }
                        Key::Right => { if state.input.cursor < state.input.buf.len() { state.input.cursor += 1; } last_ctrlc = 0; }
                        Key::Home => { state.input.cursor = 0; last_ctrlc = 0; }
                        Key::End => { state.input.cursor = state.input.buf.len(); last_ctrlc = 0; }
                        Key::Tab => {
                            let suggestion = slash_suggestion(&state.input.buf);
                            if !suggestion.is_empty() {
                                for ch in suggestion.chars() {
                                    state.input.buf.push(ch);
                                }
                                state.input.cursor = state.input.buf.len();
                            }
                            last_ctrlc = 0;
                        }
                        Key::Escape => {
                            state.input.buf.clear();
                            state.input.cursor = 0;
                        }
                        Key::CtrlC => {
                            let now = now_millis();
                            if last_ctrlc > 0 && now - last_ctrlc < 2000 {
                                // Double Ctrl+C → exit
                                print!("\x1b[?1049l");
                                restore_terminal();
                                std::process::exit(0);
                            }
                            last_ctrlc = now;
                            state.push_chat(ChatLine {
                                kind: ChatLineKind::SystemInfo,
                                content: "Press Ctrl+C again to exit".into(),
                            });
                        }
                    }
                    state.render();
                }
                Ok(Event::Resize) => {
                    state.update_dimensions();
                    state.render();
                }
                Ok(Event::Tick) => {
                    // No spinner during prompt, ignore
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // Check if deadline passed (heartbeat)
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    print!("\x1b[?1049l");
                    restore_terminal();
                    return;
                }
            }
        }

        // Remove double-Ctrl+C hint if it was shown
        if last_ctrlc > 0 {
            if let Some(last) = state.chat.last() {
                if matches!(last.kind, ChatLineKind::SystemInfo) && last.content.contains("Ctrl+C") {
                    state.chat.pop();
                }
            }
            last_ctrlc = 0;
        }

        // AGENT PHASE
        if let Some(text) = user_input {
            // Handle slash commands
            if text.starts_with('/') {
                let parts: Vec<&str> = text.splitn(2, ' ').collect();
                let cmd = parts[0];
                let args = parts.get(1).copied().unwrap_or("");

                match cmd {
                    "/help" => {
                        let mut help = String::new();
                        let name_w = SLASH_COMMANDS.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
                        for (name, desc) in SLASH_COMMANDS {
                            help.push_str(&format!("  {CYAN}{name:<name_w$}{RESET}  {DIM}{desc}{RESET}\n"));
                        }
                        state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: help.trim_end().to_string() });
                    }
                    "/model" => {
                        if args.is_empty() {
                            update_model_cache(&config.llm.api_url, &config.llm.api_key);
                            let models = AVAILABLE_MODELS.lock().map(|m| m.clone()).unwrap_or_default();
                            if models.is_empty() {
                                state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: format!("Model: {}", config.llm.model) });
                            } else {
                                let mut out = String::new();
                                for m in &models {
                                    let active = m.key == config.llm.model;
                                    let dot = if active { format!("{CYAN}●{RESET} ") } else { "  ".to_string() };
                                    let name = if active { format!("{BOLD}{}{RESET}", m.display_name) } else { m.display_name.clone() };
                                    let ctx = m.ctx_display();
                                    let caps = m.caps_display();
                                    let caps_str = if caps.is_empty() { String::new() } else { format!("  {caps}") };
                                    out.push_str(&format!("  {dot}{name}  {DIM}{ctx} ctx{RESET}{caps_str}\n"));
                                }
                                state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: out.trim_end().to_string() });
                            }
                        } else {
                            let new_model = args.trim().to_string();
                            let available = AVAILABLE_MODELS.lock().map(|m| m.clone()).unwrap_or_default();
                            let found = available.iter().find(|m| m.key == new_model);
                            if !available.is_empty() && found.is_none() {
                                state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Unknown model: {new_model}") });
                            } else {
                                config.llm.model = new_model.clone();
                                format = formats::format_for_model(&config.llm.model, &registry);
                                tools_json = format.format_tools(&registry);
                                fmt_str = if tools_json.is_some() { "openai-api" } else { "prompt-xml" };
                                if let Some(info) = found {
                                    config.llm.vision = info.vision;
                                    config.agent.compact_at = (info.context_length as f64 * 0.8) as usize;
                                }
                                if let Some(msg) = messages.first_mut() {
                                    msg.content = build_system_prompt(&config, &*format, &registry, &cwd);
                                }
                                state.push_chat(ChatLine {
                                    kind: ChatLineKind::SystemInfo,
                                    content: format!("Switched to {BOLD}{new_model}{RESET} ({fmt_str})"),
                                });
                                // Rebuild header with new model
                                state.build_header(&config, &tools_str, fmt_str, &config_path, config_source);
                            }
                        }
                    }
                    "/tokens" => {
                        let est = estimate_tokens(&messages);
                        state.push_chat(ChatLine {
                            kind: ChatLineKind::SystemInfo,
                            content: format!("Context: ~{est} tokens ({} messages, compact at ~{}k)", messages.len(), config.agent.compact_at / 1000),
                        });
                    }
                    "/compact" => {
                        let token_count = estimate_tokens(&messages);
                        compact_context(&config.llm, &mut messages, token_count, &mut state);
                        messages.push(Message { role: "user".into(), content: format!("[{}] Operator ran /compact", timestamp()) });
                    }
                    "/clear" => {
                        let sys = messages.first().cloned().unwrap_or(Message { role: "system".into(), content: String::new() });
                        messages.clear();
                        messages.push(sys);
                        state.chat.clear();
                        state.build_header(&config, &tools_str, fmt_str, &config_path, config_source);
                    }
                    "/exit" => {
                        print!("\x1b[?1049l");
                        restore_terminal();
                        std::process::exit(0);
                    }
                    _ => {
                        state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Unknown command: {cmd} (try /help)") });
                    }
                }
                state.render();
                continue;
            }

            // Regular message → send to agent
            state.push_chat(ChatLine { kind: ChatLineKind::UserMessage, content: text.clone() });
            state.render();

            let mut queued = Vec::new();
            let (api_tokens, new_queued) = run_heartbeat(
                &config, &registry, &*format, &tools_json,
                &mut messages, vec![text], &mut state, &event_rx, &mut queued,
            );
            queued.extend(new_queued);

            let token_count = api_tokens.unwrap_or_else(|| estimate_tokens(&messages));
            if token_count >= config.agent.compact_at {
                compact_context(&config.llm, &mut messages, token_count, &mut state);
            }

            // Process queued messages from during agent execution
            for q in queued {
                // Upgrade from QueuedMessage to UserMessage in chat
                if let Some(pos) = state.chat.iter().position(|l| matches!(l.kind, ChatLineKind::QueuedMessage) && l.content == q) {
                    state.chat[pos].kind = ChatLineKind::UserMessage;
                }
                state.render();
                let mut inner_queued = Vec::new();
                let (qt, _nq) = run_heartbeat(
                    &config, &registry, &*format, &tools_json,
                    &mut messages, vec![q], &mut state, &event_rx, &mut inner_queued,
                );
                // TODO: handle deeper queued messages if needed
                let tc = qt.unwrap_or_else(|| estimate_tokens(&messages));
                if tc >= config.agent.compact_at {
                    compact_context(&config.llm, &mut messages, tc, &mut state);
                }
            }
        } else {
            // Heartbeat (no user input, timeout expired)
            state.push_chat(ChatLine {
                kind: ChatLineKind::SystemInfo,
                content: format!("[heartbeat] {} (~{} tokens)", timestamp(), estimate_tokens(&messages)),
            });
            state.render();

            let mut queued = Vec::new();
            let (api_tokens, new_queued) = run_heartbeat(
                &config, &registry, &*format, &tools_json,
                &mut messages, vec![], &mut state, &event_rx, &mut queued,
            );
            queued.extend(new_queued);

            let token_count = api_tokens.unwrap_or_else(|| estimate_tokens(&messages));
            if token_count >= config.agent.compact_at {
                compact_context(&config.llm, &mut messages, token_count, &mut state);
            }

            // Process queued messages
            for q in queued {
                if let Some(pos) = state.chat.iter().position(|l| matches!(l.kind, ChatLineKind::QueuedMessage) && l.content == q) {
                    state.chat[pos].kind = ChatLineKind::UserMessage;
                }
                state.render();
                let mut inner_queued = Vec::new();
                let (qt, _) = run_heartbeat(
                    &config, &registry, &*format, &tools_json,
                    &mut messages, vec![q], &mut state, &event_rx, &mut inner_queued,
                );
                let tc = qt.unwrap_or_else(|| estimate_tokens(&messages));
                if tc >= config.agent.compact_at {
                    compact_context(&config.llm, &mut messages, tc, &mut state);
                }
            }
        }
    }
}

fn timestamp() -> String {
    Command::new("date").arg("+%H:%M").output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "???".into())
}

fn date_today() -> String {
    Command::new("date").arg("+%Y-%m-%d").output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "???".into())
}
