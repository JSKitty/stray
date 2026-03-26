mod config;
mod event;
mod formats;
mod highlight;
mod markdown;
mod term;
mod tools;
mod ui;

use config::{Config, ConfigSource, LlmConfig};
use event::Event;
use formats::ModelFormat;
use serde_json::json;
use std::io::{BufRead, BufReader};
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

#[derive(Clone, Copy)]
enum Role { System, User, Assistant }

impl Role {
    fn as_str(self) -> &'static str {
        match self { Role::System => "system", Role::User => "user", Role::Assistant => "assistant" }
    }
}

#[derive(Clone)]
struct Message {
    role: Role,
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
        json!({"role": msg.role.as_str(), "content": parts})
    } else {
        json!({"role": msg.role.as_str(), "content": msg.content})
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
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cancelled_bg = cancelled.clone();

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
            if cancelled_bg.load(std::sync::atomic::Ordering::Relaxed) { break; }
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
    let mut received_done = false;
    let mut tag_buf = String::new();
    let mut inside_tool_tag = false;
    let mut tool_tag_name = String::new();
    let mut tool_preview = String::new();
    let mut tool_preview_shown = false;

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
                            s.advance_cat_anim();
                            s.tick_fade();
                            if s.spinner.active {
                                s.spinner.frame = (s.spinner.frame + 1) % 10;
                            }
                            s.render();
                        }
                    }
                    // Handle input during agent execution (queue messages)
                    Event::Key(key) => {
                        if let Some(ref mut s) = state {
                            match key {
                                Key::Char(ch) => { s.input.insert(ch); }
                                Key::ShiftEnter => { s.input.insert('\n'); }
                                Key::Paste(text) => { s.input.insert_str(&text); }
                                Key::Backspace => { s.input.backspace(); }
                                Key::Delete => { s.input.delete(); }
                                Key::Left => { s.input.move_left(); }
                                Key::Right => { s.input.move_right(); }
                                Key::Home => { s.input.move_home(); }
                                Key::End => { s.input.move_end(); }
                                Key::Enter => {
                                    let line = s.input.drain_all();
                                    let trimmed = line.trim().to_string();
                                    if !trimmed.is_empty() {
                                        if trimmed.starts_with('/') {
                                            handle_inline_slash(&trimmed, s, messages);
                                        } else {
                                            queued_messages.push(trimmed.clone());
                                            s.push_chat(ChatLine { kind: ChatLineKind::QueuedMessage, content: trimmed });
                                        }
                                    }
                                }
                                Key::Tab => {
                                    let chars = s.input.to_chars();
                                    let suggestion = slash_suggestion(&chars);
                                    s.input.move_end();
                                    s.input.insert_str(&suggestion);
                                }
                                _ => {}
                            }
                            s.render();
                        }
                    }
                }
            }
        }
        if was_disturbed {
            cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
            break;
        }

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

        if data == "[DONE]" { received_done = true; break; }

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
                            // Update live tool preview (show first line = file path)
                            if ch != '<' {
                                tool_preview.push(ch);
                                let first_line = tool_preview.lines().next().unwrap_or("").trim();
                                if !first_line.is_empty() {
                                    let mut title_chars = tool_tag_name.chars();
                                    let title = match title_chars.next() {
                                        Some(c) => format!("{}{}", c.to_uppercase(), title_chars.as_str()),
                                        None => tool_tag_name.clone(),
                                    };
                                    if !tool_preview_shown {
                                        // First detection: switch from streaming to spinner
                                        if streaming_started {
                                            s.end_streaming();
                                        }
                                        s.start_spinner(&format!("{title} → {first_line}"));
                                        tool_preview_shown = true;
                                    } else {
                                        // Update spinner label as more content arrives
                                        s.spinner.label = format!("{title} → {first_line}");
                                    }
                                }
                            }
                            for tag in tool_tags {
                                if tag_buf.ends_with(&format!("</{tag}>")) {
                                    inside_tool_tag = false;
                                    tag_buf.clear();
                                    tool_preview.clear();
                                    tool_preview_shown = false;
                                    break;
                                }
                            }
                            continue;
                        }
                        if ch == '<' || !tag_buf.is_empty() {
                            tag_buf.push(ch);
                            if tag_buf.len() > 128 && !inside_tool_tag {
                                display_text.push_str(&tag_buf);
                                tag_buf.clear();
                                continue;
                            }
                            if ch == '>' {
                                if let Some(tag) = tool_tags.iter().find(|t| tag_buf == format!("<{t}>")) {
                                    inside_tool_tag = true;
                                    tool_tag_name = tag.to_string();
                                    tool_preview.clear();
                                    tool_preview_shown = false;
                                    tag_buf.clear();
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
                        s.tick_fade();
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

    // Flush any leftover tag buffer (partial tag at end of stream)
    // Note: content already has these chars from the streaming loop (line 256)
    if !tag_buf.is_empty() && !inside_tool_tag {
        if let Some(ref mut s) = state {
            if streaming_started {
                s.append_streaming(&tag_buf);
            }
        }
    }

    // Stream ended without data: [DONE] — server disconnected or thread died
    if !received_done && !was_disturbed && streaming_started {
        was_disturbed = true;
    }

    // Finalize streaming
    if streaming_started {
        if was_disturbed {
            if let Some(ref mut s) = state {
                s.append_streaming(&format!(" {DIM}[disturbed]{RESET}"));
            }
        }
        if let Some(ref mut s) = state {
            s.end_streaming();
            s.render();
        }
    } else if was_disturbed {
        if let Some(ref mut s) = state {
            s.stop_spinner();
            s.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: format!("{DIM}[disturbed]{RESET}") });
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

fn compact_context(
    config: &LlmConfig, messages: &mut Vec<Message>, token_count: usize,
    state: &mut AppState, event_rx: &Receiver<Event>,
) {
    state.push_chat(ChatLine {
        kind: ChatLineKind::SystemInfo,
        content: format!("[compact] Context at ~{token_count} tokens"),
    });
    state.start_spinner("compacting...");
    state.render();

    messages.push(Message { role: Role::User, content: COMPACT_PROMPT.into() });

    let resp = match call_llm(config, messages, &None, Some(state), Some(event_rx), &[], &mut Vec::new()) {
        Ok(r) => r,
        Err(e) => {
            messages.pop();
            state.stop_spinner();
            state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("[compact] Failed: {e}") });
            state.render();
            return;
        }
    };

    let system = messages.first().cloned().unwrap_or(Message { role: Role::System, content: String::new() });
    messages.clear();
    messages.push(system);
    messages.push(Message {
        role: Role::Assistant,
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
            role: Role::User,
            content: format!("[{}] Heartbeat. Check in and do your tasks.", timestamp()),
        });
    } else {
        let combined = user_messages.join("\n\n");
        messages.push(Message {
            role: Role::User,
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
            state.cat_anim.play(CatAnimKind::Squint);
            state.render();
            let partial = if resp.content.trim().is_empty() {
                "[no response generated]".to_string()
            } else {
                format!("{} [interrupted]", resp.content)
            };
            messages.push(Message { role: Role::Assistant, content: partial });
            messages.push(Message {
                role: Role::User,
                content: format!("[{}] You were disturbed by the operator — your previous response was cut short.", timestamp()),
            });
            break;
        }

        let (calls, _) = format.parse_response(&resp.content);
        messages.push(Message { role: Role::Assistant, content: resp.content });

        if calls.is_empty() { break; }

        // Execute tool calls (with spinner + ESC cancel)
        let mut results: Vec<(String, String, String)> = Vec::new();
        let mut tool_cancelled = false;
        for call in &calls {
            if tool_cancelled { break; }
            let tool = registry.tools().iter().find(|t| t.name() == call.tool);
            match tool {
                Some(t) => {
                    let action = t.display_action(&call.input);
                    let mut name_chars = t.name().chars();
                    let title = match name_chars.next() {
                        Some(c) => format!("{}{}", c.to_uppercase(), name_chars.as_str()),
                        None => String::new(),
                    };
                    state.stop_spinner();
                    state.push_chat(ChatLine {
                        kind: ChatLineKind::ToolAction,
                        content: format!("{title} → {action}"),
                    });

                    let output = if let Some(spawn_result) = t.spawn(&call.input) {
                        // Non-blocking: poll child with spinner + ESC cancel
                        match spawn_result {
                            Ok(mut child) => {
                                let timeout = t.timeout();
                                let start = std::time::Instant::now();
                                state.start_spinner(&format!("{title}..."));
                                state.render();
                                loop {
                                    while let Ok(ev) = event_rx.try_recv() {
                                        match ev {
                                            Event::Key(Key::CtrlC) | Event::Key(Key::Escape) => {
                                                let _ = child.kill();
                                                tool_cancelled = true;
                                            }
                                            Event::Tick => {
                                                state.advance_cat_anim();
                                                state.spinner.frame = (state.spinner.frame + 1) % 10;
                                                state.tick_fade();
                                                state.render();
                                            }
                                            Event::Resize => {
                                                state.update_dimensions();
                                                state.render();
                                            }
                                            Event::Key(key) => {
                                                match key {
                                                    Key::Char(ch) => { state.input.insert(ch); }
                                                    Key::ShiftEnter => { state.input.insert('\n'); }
                                                    Key::Paste(text) => { state.input.insert_str(&text); }
                                                    Key::Backspace => { state.input.backspace(); }
                                                    Key::Delete => { state.input.delete(); }
                                                    Key::Left => { state.input.move_left(); }
                                                    Key::Right => { state.input.move_right(); }
                                                    Key::Home => { state.input.move_home(); }
                                                    Key::End => { state.input.move_end(); }
                                                    Key::Enter => {
                                                        let line = state.input.drain_all();
                                                        let trimmed = line.trim().to_string();
                                                        if !trimmed.is_empty() {
                                                            if trimmed.starts_with('/') {
                                                                handle_inline_slash(&trimmed, state, messages);
                                                            } else {
                                                                queued_messages.push(trimmed.clone());
                                                                state.push_chat(ChatLine { kind: ChatLineKind::QueuedMessage, content: trimmed });
                                                            }
                                                        }
                                                    }
                                                    Key::Tab => {
                                                        let chars = state.input.to_chars();
                                                        let suggestion = slash_suggestion(&chars);
                                                        state.input.move_end();
                                                        state.input.insert_str(&suggestion);
                                                    }
                                                    _ => {}
                                                }
                                                state.render();
                                            }
                                        }
                                    }
                                    if tool_cancelled {
                                        let _ = child.kill();
                                        break "[cancelled by user]".to_string();
                                    }
                                    match child.try_wait() {
                                        Ok(Some(_)) => {
                                            break match child.wait_with_output() {
                                                Ok(out) => t.format_output(&out),
                                                Err(e) => format!("[error] {e}"),
                                            };
                                        }
                                        Ok(None) => {
                                            if start.elapsed() >= timeout {
                                                let _ = child.kill();
                                                break format!("[TIMEOUT] Command killed after {}s", timeout.as_secs());
                                            }
                                        }
                                        Err(e) => break format!("[error] {e}"),
                                    }
                                    std::thread::sleep(Duration::from_millis(16));
                                }
                            }
                            Err(e) => e,
                        }
                    } else {
                        // Synchronous for fast tools (read, etc.)
                        state.render();
                        t.execute(&call.input)
                    };

                    results.push((call.tool.clone(), call.input.clone(), output));
                    state.start_spinner("thinking...");
                    state.render();
                }
                None => {
                    results.push((call.tool.clone(), call.input.clone(), format!("[error] Unknown tool: {}", call.tool)));
                }
            }
        }

        if tool_cancelled {
            state.stop_spinner();
            state.cat_anim.play(CatAnimKind::Squint);
            state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: "[tool cancelled]".into() });
            messages.push(Message {
                role: Role::User,
                content: format!("[{}] Tool execution was cancelled by the operator.", timestamp()),
            });
            state.render();
            break;
        }

        messages.push(Message { role: Role::User, content: format.format_results(&results) });

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

// ---------------------------------------------------------------------------
// Config menu system — modular field definitions
// ---------------------------------------------------------------------------

/// Config fields: (key, label, description). Adding a new setting = adding one line here.
const CONFIG_FIELDS: &[(&str, &str, &str)] = &[
    ("provider", "Provider", "The LLM API provider to connect to"),
    ("api_key", "API Key", "Authentication key for the LLM provider"),
    ("name", "Agent Name", "Your companion's name, shown in the header and system prompt"),
    ("heartbeat", "Heartbeat", "Seconds between autonomous check-ins (0 to disable)"),
    ("system_prompt", "System Prompt", "Instructions that define the agent's personality and behavior"),
];

/// Known providers: (display_name, api_url, api_key)
const PROVIDERS: &[(&str, &str, &str)] = &[
    ("LMStudio", "http://127.0.0.1:1234/v1/chat/completions", "lm-studio"),
    ("PPQ", "https://api.ppq.ai/v1/chat/completions", ""),
];

fn detect_provider(api_url: &str) -> String {
    if api_url.contains("127.0.0.1:1234") || api_url.contains("localhost:1234") {
        "LMStudio".into()
    } else if api_url.contains("ppq.ai") {
        "PPQ".into()
    } else {
        "Custom".into()
    }
}

fn get_config_value(config: &Config, key: &str) -> String {
    match key {
        "provider" => detect_provider(&config.llm.api_url),
        "api_key" => {
            let k = &config.llm.api_key;
            if k.len() <= 8 { k.clone() }
            else { format!("{}...{}", &k[..4], &k[k.len()-4..]) } // mask middle
        }
        "name" => config.agent.name.clone(),
        "heartbeat" => format!("{}s", config.agent.heartbeat),
        "system_prompt" => {
            let s = config.agent.system_prompt.lines().next().unwrap_or("").trim();
            let truncated: String = s.chars().take(40).collect();
            if s.chars().count() > 40 { format!("{truncated}...") } else { truncated }
        }
        _ => String::new(),
    }
}

fn open_config_selector(state: &mut AppState, id: &str, config: &Config) {
    let parts: Vec<&str> = id.split(':').collect();
    match parts.as_slice() {
        ["config"] => {
            state.input_label = "Config".into();
            state.selector = Some(Selector::new("config", vec![
                SelectorItem { label: "Local (workspace)".into(), value: "local".into() },
                SelectorItem { label: "Global".into(), value: "global".into() },
            ], 2));
        }
        ["config", scope] => {
            let scope_label = if *scope == "local" { "Local" } else { "Global" };
            state.input_label = format!("Config · {scope_label}");
            let items: Vec<SelectorItem> = CONFIG_FIELDS.iter().map(|(key, label, _desc)| {
                let val = get_config_value(config, key);
                let display = if val.is_empty() { label.to_string() } else { format!("{label}: {val}") };
                SelectorItem { label: display, value: key.to_string() }
            }).collect();
            state.selector = Some(Selector::new(id, items, CONFIG_FIELDS.len().min(8)));
        }
        ["config", scope, sub] => {
            let scope_label = if *scope == "local" { "Local" } else { "Global" };
            let mut sub_label = sub.to_string();
            if let Some(c) = sub_label.get_mut(0..1) { c.make_ascii_uppercase(); }
            state.input_label = format!("Config · {scope_label} · {sub_label}");
            if *sub == "provider" {
                let current = detect_provider(&config.llm.api_url);
                let items: Vec<SelectorItem> = PROVIDERS.iter().map(|(name, _, _)| {
                    SelectorItem { label: name.to_string(), value: name.to_lowercase() }
                }).collect();
                let mut sel = Selector::new(id, items, PROVIDERS.len());
                if let Some(idx) = PROVIDERS.iter().position(|(n, _, _)| *n == current) {
                    sel.selected = idx;
                }
                state.selector = Some(sel);
            }
        }
        _ => {}
    }
}

fn save_config_field(scope: &str, field: &str, value: &str) -> Result<std::path::PathBuf, String> {
    let path = match scope {
        "local" => std::path::PathBuf::from("stray.toml"),
        "global" => {
            let dir = config::global_config_dir().ok_or("No global config directory")?;
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            dir.join("stray.toml")
        }
        _ => return Err("Unknown scope".into()),
    };

    let content = std::fs::read_to_string(&path).unwrap_or_default();
    let mut doc: toml::Value = content.parse()
        .unwrap_or_else(|_| toml::Value::Table(toml::map::Map::new()));
    let table = doc.as_table_mut().ok_or("Invalid TOML")?;

    match field {
        "provider" => {
            let (_, url, key) = PROVIDERS.iter()
                .find(|(n, _, _)| n.to_lowercase() == value)
                .ok_or("Unknown provider")?;
            let llm = table.entry("llm")
                .or_insert(toml::Value::Table(toml::map::Map::new()));
            if let Some(t) = llm.as_table_mut() {
                t.insert("api_url".into(), toml::Value::String(url.to_string()));
                if !key.is_empty() {
                    t.insert("api_key".into(), toml::Value::String(key.to_string()));
                }
            }
        }
        "heartbeat" => {
            let secs: u64 = value.trim().trim_end_matches('s').parse()
                .map_err(|_| "Invalid number".to_string())?;
            let sec = table.entry("agent")
                .or_insert(toml::Value::Table(toml::map::Map::new()));
            if let Some(t) = sec.as_table_mut() {
                t.insert("heartbeat".into(), toml::Value::Integer(secs as i64));
            }
        }
        "api_key" => {
            let llm = table.entry("llm")
                .or_insert(toml::Value::Table(toml::map::Map::new()));
            if let Some(t) = llm.as_table_mut() {
                t.insert("api_key".into(), toml::Value::String(value.into()));
            }
        }
        "name" | "system_prompt" => {
            let section = "agent";
            let toml_key = if field == "name" { "name" } else { "system_prompt" };
            let sec = table.entry(section)
                .or_insert(toml::Value::Table(toml::map::Map::new()));
            if let Some(t) = sec.as_table_mut() {
                t.insert(toml_key.into(), toml::Value::String(value.into()));
            }
        }
        _ => return Err(format!("Unknown field: {field}")),
    }

    let output = toml::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(&path, &output).map_err(|e| e.to_string())?;
    Ok(path)
}

fn save_model_to_config(path: &std::path::Path, model: &str) -> Result<(), String> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut doc: toml::Value = content.parse()
        .unwrap_or_else(|_| toml::Value::Table(toml::map::Map::new()));
    let table = doc.as_table_mut().ok_or("Invalid TOML")?;
    let llm = table.entry("llm")
        .or_insert(toml::Value::Table(toml::map::Map::new()));
    if let Some(t) = llm.as_table_mut() {
        t.insert("model".into(), toml::Value::String(model.into()));
    }
    let output = toml::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(path, output).map_err(|e| e.to_string())
}

fn copy_to_clipboard(text: &str) -> Result<(), String> {
    let cmd = if cfg!(target_os = "macos") { "pbcopy" }
        else if cfg!(target_os = "windows") { "clip" }
        else { "xclip -selection clipboard" };
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let mut child = std::process::Command::new(parts[0])
        .args(&parts[1..])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|_| "No clipboard tool found (pbcopy/xclip/clip)".to_string())?;
    if let Some(stdin) = child.stdin.as_mut() {
        std::io::Write::write_all(stdin, text.as_bytes()).map_err(|e| e.to_string())?;
    }
    child.wait().map_err(|e| e.to_string())?;
    Ok(())
}

/// Handle safe slash commands during agent execution. Returns true if handled.
fn handle_inline_slash(
    trimmed: &str, state: &mut AppState, messages: &[Message],
) -> bool {
    let cmd = trimmed.split_whitespace().next().unwrap_or("");
    state.push_chat(ChatLine { kind: ChatLineKind::UserMessage, content: trimmed.to_string() });
    match cmd {
        "/help" => {
            let cmds: Vec<_> = SLASH_COMMANDS.iter().filter(|(n, _)| *n != "/help").collect();
            let name_w = cmds.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
            let mut help = String::new();
            for (name, desc) in cmds {
                help.push_str(&format!("  {CYAN}{name:<name_w$}{RESET}  {DIM}{desc}{RESET}\n"));
            }
            state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: help.trim_end().to_string() });
        }
        "/copy" => {
            let last_agent = state.chat.iter().rev()
                .find(|l| matches!(l.kind, ChatLineKind::AgentText | ChatLineKind::AgentStreaming));
            if let Some(line) = last_agent {
                match copy_to_clipboard(line.content.trim()) {
                    Ok(()) => state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: "Copied to clipboard".into() }),
                    Err(e) => state.push_chat(ChatLine { kind: ChatLineKind::Error, content: e }),
                }
            } else {
                state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: "No agent response to copy".into() });
            }
        }
        "/context" => {
            let est = estimate_tokens(messages);
            state.push_chat(ChatLine {
                kind: ChatLineKind::SystemInfo,
                content: format!("Context: ~{est} tokens ({} messages)", messages.len()),
            });
        }
        "/exit" => {
            print!("\x1b[?1049l");
            restore_terminal();
            std::process::exit(0);
        }
        _ => {
            // Known but unsafe command, or unknown — don't queue as message
            if SLASH_COMMANDS.iter().any(|(n, _)| *n == cmd) {
                // Remove the UserMessage we just pushed (it's not a real message)
                state.chat.pop();
                state.push_chat(ChatLine {
                    kind: ChatLineKind::SystemInfo,
                    content: format!("{cmd} unavailable during processing"),
                });
            } else {
                state.chat.pop();
                state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Unknown command: {cmd}") });
            }
        }
    }
    true
}

fn fmt_tokens(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f64 / 1_000_000.0) }
    else if n >= 10_000 { format!("{}k", n / 1000) }
    else if n >= 1_000 { format!("{:.1}k", n as f64 / 1000.0) }
    else { format!("{n}") }
}

fn format_context_display(messages: &[Message], config: &Config) -> String {
    let total = estimate_tokens(messages);
    let system_tokens = if !messages.is_empty() { estimate_tokens(&messages[0..1]) } else { 0 };
    let msg_tokens = total.saturating_sub(system_tokens);
    let msg_count = messages.len().saturating_sub(1);
    let limit = config.agent.compact_at;
    let pct = if limit > 0 { (total as f64 / limit as f64 * 100.0).min(100.0) as usize } else { 0 };

    // Bar: 20 chars — system (yellow) + messages (cyan) + free (dim)
    let bar_len = 20usize;
    let sys_blocks = if limit > 0 {
        ((system_tokens as f64 / limit as f64) * bar_len as f64).ceil() as usize
    } else { 0 }.min(bar_len);
    let msg_blocks = if limit > 0 {
        ((msg_tokens as f64 / limit as f64) * bar_len as f64).ceil() as usize
    } else { 0 }.min(bar_len.saturating_sub(sys_blocks));
    let free_blocks = bar_len.saturating_sub(sys_blocks + msg_blocks);
    let bar_sys: String = std::iter::repeat('█').take(sys_blocks).collect();
    let bar_msg: String = std::iter::repeat('█').take(msg_blocks).collect();
    let bar_free: String = std::iter::repeat('░').take(free_blocks).collect();

    let mut out = String::new();
    out.push_str(&format!("{DIM}Context{RESET} · {BOLD}{}{RESET}\n", config.llm.model));
    out.push_str(&format!("\n  {YELLOW}{bar_sys}{RESET}{CYAN}{bar_msg}{RESET}{DIM}{bar_free}{RESET}  {} / {} tokens ({pct}%)\n", fmt_tokens(total), fmt_tokens(limit)));
    out.push_str(&format!("\n  {YELLOW}█{RESET} {DIM}System     ~{}{RESET}", fmt_tokens(system_tokens)));
    out.push_str(&format!("\n  {CYAN}█{RESET} {DIM}Messages   ~{} ({msg_count} msgs){RESET}", fmt_tokens(msg_tokens)));
    out.push_str(&format!("\n  {DIM}░ Free       ~{}{RESET}", fmt_tokens(limit.saturating_sub(total))));
    out
}

/// Switch provider with per-provider settings save/restore.
/// Saves current LLM settings under [providers.old_name], loads [providers.new_name] if it exists.
fn switch_provider(config: &mut Config, new_provider: &str, config_path: &std::path::Path) -> Result<bool, String> {
    let old_provider = detect_provider(&config.llm.api_url).to_lowercase();
    let path = config_path;

    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut doc: toml::Value = content.parse()
        .unwrap_or_else(|_| toml::Value::Table(toml::map::Map::new()));
    let table = doc.as_table_mut().ok_or("Invalid TOML")?;

    // Save current settings under [providers.old_name]
    let providers = table.entry("providers")
        .or_insert(toml::Value::Table(toml::map::Map::new()));
    if let Some(p) = providers.as_table_mut() {
        let mut saved = toml::map::Map::new();
        saved.insert("api_url".into(), toml::Value::String(config.llm.api_url.clone()));
        saved.insert("api_key".into(), toml::Value::String(config.llm.api_key.clone()));
        saved.insert("model".into(), toml::Value::String(config.llm.model.clone()));
        p.insert(old_provider, toml::Value::Table(saved));
    }

    // Check if [providers.new_name] exists with saved settings
    let restored = if let Some(saved) = table.get("providers")
        .and_then(|p| p.get(new_provider))
        .and_then(|s| s.as_table())
    {
        if let Some(url) = saved.get("api_url").and_then(|v| v.as_str()) {
            config.llm.api_url = url.to_string();
        }
        if let Some(key) = saved.get("api_key").and_then(|v| v.as_str()) {
            config.llm.api_key = key.to_string();
        }
        if let Some(model) = saved.get("model").and_then(|v| v.as_str()) {
            config.llm.model = model.to_string();
        }
        true
    } else {
        // No saved settings — use provider defaults
        if let Some((_, url, key)) = PROVIDERS.iter().find(|(n, _, _)| n.to_lowercase() == new_provider) {
            config.llm.api_url = url.to_string();
            if !key.is_empty() { config.llm.api_key = key.to_string(); }
        }
        false
    };

    // Write [llm] with new values
    let llm = table.entry("llm")
        .or_insert(toml::Value::Table(toml::map::Map::new()));
    if let Some(t) = llm.as_table_mut() {
        t.insert("api_url".into(), toml::Value::String(config.llm.api_url.clone()));
        t.insert("api_key".into(), toml::Value::String(config.llm.api_key.clone()));
        t.insert("model".into(), toml::Value::String(config.llm.model.clone()));
    }

    let output = toml::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    std::fs::write(path, output).map_err(|e| e.to_string())?;
    Ok(restored)
}

fn apply_config_change(config: &mut Config, field: &str, value: &str) {
    match field {
        "provider" => {
            if let Some((_, url, key)) = PROVIDERS.iter().find(|(n, _, _)| n.to_lowercase() == value) {
                config.llm.api_url = url.to_string();
                if !key.is_empty() { config.llm.api_key = key.to_string(); }
            }
        }
        "api_key" => config.llm.api_key = value.to_string(),
        "name" => config.agent.name = value.to_string(),
        "heartbeat" => {
            if let Ok(secs) = value.trim().trim_end_matches('s').parse::<u64>() {
                config.agent.heartbeat = secs;
            }
        }
        "system_prompt" => config.agent.system_prompt = value.to_string(),
        _ => {}
    }
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

    // Register tools (vision flag is shared so model switches update it)
    let vision_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(config.llm.vision));
    let registry = {
        let mut r = ToolRegistry::new();
        r.add(Box::new(tools::BashTool));
        r.add(Box::new(tools::ReadTool::new(vision_flag.clone())));
        r.add(Box::new(tools::WriteTool));
        r.add(Box::new(tools::EditTool));
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

    // Fetch models in background (non-blocking startup)
    {
        let url = config.llm.api_url.clone();
        let key = config.llm.api_key.clone();
        std::thread::spawn(move || { let _ = update_model_cache(&url, &key); });
    }

    // Enter raw mode + alternate screen
    enable_raw_mode();
    print!("\x1b[?1049h{CURSOR_BAR}");
    let _ = std::io::Write::flush(&mut std::io::stdout());

    // Build TUI state
    let mut state = AppState::new();
    state.build_header(&config, &tools_str, fmt_str, &config_path, config_source);
    state.render();

    let mut messages: Vec<Message> = vec![Message { role: Role::System, content: system }];

    // Event channels (input thread + tick thread + resize watcher)
    let event_rx = event::setup_event_channels();

    // Signal safety net — restores terminal on SIGINT/SIGTERM even if event loop is stuck
    event::setup_signal_handlers();

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
            if remaining.is_zero() && config.agent.heartbeat > 0 {
                state.selector = None;
                state.input_label.clear(); state.input_hint.clear();
                break 'prompt; // heartbeat timeout
            }

            let timeout = if config.agent.heartbeat == 0 { Duration::from_millis(100) } else { remaining.min(Duration::from_millis(100)) };
            match event_rx.recv_timeout(timeout) {
                Ok(Event::Key(key)) => {
                    // Selector navigation (model picker, etc.)
                    if state.selector.is_some() {
                        match key {
                            Key::Down => {
                                state.selector.as_mut().unwrap().move_down();
                                state.render();
                                continue 'prompt;
                            }
                            Key::Up => {
                                state.selector.as_mut().unwrap().move_up();
                                state.render();
                                continue 'prompt;
                            }
                            Key::Enter => {
                                let sel = state.selector.take().unwrap();
                                let value = sel.items[sel.selected].value.clone();
                                let sel_id = sel.id;

                                if sel_id == "model" {
                                    state.input_label.clear(); state.input_hint.clear();
                                    if value != config.llm.model {
                                        config.llm.model = value.clone();
                                        format = formats::format_for_model(&config.llm.model, &registry);
                                        tools_json = format.format_tools(&registry);
                                        fmt_str = if tools_json.is_some() { "openai-api" } else { "prompt-xml" };
                                        if let Some(info) = get_model_info(&value) {
                                            config.llm.vision = info.vision;
                                        vision_flag.store(info.vision, std::sync::atomic::Ordering::Relaxed);
                                            config.agent.compact_at = (info.context_length as f64 * 0.8) as usize;
                                        }
                                        if let Some(msg) = messages.first_mut() {
                                            msg.content = build_system_prompt(&config, &*format, &registry, &cwd);
                                        }
                                        state.push_chat(ChatLine {
                                            kind: ChatLineKind::SystemInfo,
                                            content: format!("Switched to {BOLD}{value}{RESET}"),
                                        });
                                        state.build_header(&config, &tools_str, fmt_str, &config_path, config_source);
                                        if let Err(e) = save_model_to_config(&config_path, &value) {
                                            state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Config save failed: {e}") });
                                        }
                                    }
                                } else {
                                    // Config menu routing (nested selectors)
                                    let parts: Vec<&str> = sel_id.split(':').collect();
                                    match parts.as_slice() {
                                        ["config"] => {
                                            open_config_selector(&mut state, &format!("config:{value}"), &config);
                                        }
                                        ["config", scope] => {
                                            let scope = scope.to_string();
                                            match value.as_str() {
                                                "provider" => {
                                                    open_config_selector(&mut state, &format!("config:{scope}:provider"), &config);
                                                }
                                                field => {
                                                    // Text input mode
                                                    let (label, desc) = CONFIG_FIELDS.iter()
                                                        .find(|f| f.0 == field)
                                                        .map(|f| (f.1, f.2))
                                                        .unwrap_or((field, ""));
                                                    let full_value = match field {
                                                        "name" => config.agent.name.clone(),
                                                        "api_key" => config.llm.api_key.clone(),
                                                        "heartbeat" => config.agent.heartbeat.to_string(),
                                                        "system_prompt" => config.agent.system_prompt.clone(),
                                                        _ => get_config_value(&config, field),
                                                    };
                                                    state.config_edit = Some((scope, field.to_string()));
                                                    state.input.clear();
                                                    state.input.insert_str(&full_value);
                                                    state.input_label = format!("{label} (ESC to cancel)");
                                                    state.input_hint = desc.to_string();
                                                }
                                            }
                                        }
                                        ["config", scope, "provider"] => {
                                            let scope = scope.to_string();
                                            match switch_provider(&mut config, &value, &config_path) {
                                                Ok(restored) => {
                                                    let prov = PROVIDERS.iter().find(|(n, _, _)| n.to_lowercase() == value);
                                                    let name = prov.map(|p| p.0).unwrap_or(&value);
                                                    let needs_key = prov.map(|p| p.2.is_empty()).unwrap_or(false);
                                                    let msg = if restored {
                                                        format!("Switched to {BOLD}{name}{RESET} {DIM}(settings restored){RESET}")
                                                    } else {
                                                        format!("Switched to {BOLD}{name}{RESET}")
                                                    };
                                                    state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: msg });
                                                    // Update format + header for new model
                                                    format = formats::format_for_model(&config.llm.model, &registry);
                                                    tools_json = format.format_tools(&registry);
                                                    fmt_str = if tools_json.is_some() { "openai-api" } else { "prompt-xml" };
                                                    if let Some(msg) = messages.first_mut() {
                                                        msg.content = build_system_prompt(&config, &*format, &registry, &cwd);
                                                    }
                                                    state.build_header(&config, &tools_str, fmt_str, &config_path, config_source);
                                                    if needs_key && !restored {
                                                        // New provider with no saved key — open key editor
                                                        state.config_edit = Some((scope.clone(), "api_key".to_string()));
                                                        state.input.clear();
                                                        state.input.insert_str(&config.llm.api_key);
                                                        state.input_label = "API Key (ESC to cancel)".into();
                                                        state.input_hint = "This provider requires an API key".into();
                                                    }
                                                }
                                                Err(e) => {
                                                    state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Failed: {e}") });
                                                }
                                            }
                                            if state.config_edit.is_none() {
                                                open_config_selector(&mut state, &format!("config:{scope}"), &config);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                state.render();
                                continue 'prompt;
                            }
                            Key::Escape => {
                                // Back-navigate: selectors with ':' in id have a parent
                                if let Some(sel) = state.selector.take() {
                                    if let Some((parent, _)) = sel.id.rsplit_once(':') {
                                        open_config_selector(&mut state, parent, &config);
                                    } else {
                                        state.input_label.clear(); state.input_hint.clear();
                                    }
                                }
                                state.render();
                                continue 'prompt;
                            }
                            _ => { state.selector = None; state.input_label.clear(); state.input_hint.clear(); } // close and fall through
                        }
                    }
                    match key {
                        Key::Char(ch) => {
                            state.input.insert(ch);
                            last_ctrlc = 0;
                        }
                        Key::ShiftEnter => {
                            state.input.insert('\n');
                            last_ctrlc = 0;
                        }
                        Key::Paste(text) => {
                            state.input.insert_str(&text);
                            last_ctrlc = 0;
                        }
                        Key::Enter => {
                            let line = state.input.drain_all();
                            let trimmed = line.trim().to_string();
                            if !trimmed.is_empty() {
                                user_input = Some(trimmed);
                                break 'prompt;
                            }
                            last_ctrlc = 0;
                        }
                        Key::Backspace => {
                            state.input.backspace();
                            last_ctrlc = 0;
                        }
                        Key::Delete => {
                            state.input.delete();
                            last_ctrlc = 0;
                        }
                        Key::Left => { state.input.move_left(); last_ctrlc = 0; }
                        Key::Right => { state.input.move_right(); last_ctrlc = 0; }
                        Key::Up | Key::ShiftUp => {
                            let is_shift = matches!(key, Key::ShiftUp);
                            let mut scrolled = false;
                            if is_shift {
                                state.scroll_offset += 1;
                                scrolled = true;
                            } else {
                                let inner = state.width.saturating_sub(2);
                                if inner > 0 {
                                    let chars = state.input.to_chars();
                                    let lines = visual_line_breaks(&chars, inner);
                                    let cursor = state.input.before_cursor().len();
                                    let cur_line = lines.iter().position(|(s, e)| cursor >= *s && cursor < *e)
                                        .unwrap_or(lines.len().saturating_sub(1));
                                    if cur_line > 0 {
                                        let col_width: usize = chars[lines[cur_line].0..cursor].iter()
                                            .filter(|&&c| c != '\n').map(|&c| char_display_width(c)).sum();
                                        let (ps, pe) = lines[cur_line - 1];
                                        let end = if pe > ps && chars.get(pe - 1) == Some(&'\n') { pe - 1 } else { pe };
                                        let mut w = 0;
                                        let mut target = ps;
                                        for i in ps..end {
                                            if chars[i] == '\n' { continue; }
                                            let cw = char_display_width(chars[i]);
                                            if w + cw > col_width { break; }
                                            w += cw;
                                            target = i + 1;
                                        }
                                        state.input.set_cursor(target);
                                    } else if state.input.before_cursor().is_empty() {
                                        // Already at very start — scroll up
                                        state.scroll_offset += 1;
                                        scrolled = true;
                                    } else {
                                        state.input.move_home();
                                    }
                                }
                            }
                            let _ = scrolled; // scroll_offset clamped in render
                            last_ctrlc = 0;
                        }
                        Key::Down | Key::ShiftDown => {
                            let is_shift = matches!(key, Key::ShiftDown);
                            if is_shift {
                                state.scroll_offset = state.scroll_offset.saturating_sub(1);
                            } else {
                                let inner = state.width.saturating_sub(2);
                                if inner > 0 {
                                    let chars = state.input.to_chars();
                                    let lines = visual_line_breaks(&chars, inner);
                                    let cursor = state.input.before_cursor().len();
                                    let cur_line = lines.iter().position(|(s, e)| cursor >= *s && cursor < *e)
                                        .unwrap_or(lines.len().saturating_sub(1));
                                    if cur_line + 1 < lines.len() {
                                        let col_width: usize = chars[lines[cur_line].0..cursor].iter()
                                            .filter(|&&c| c != '\n').map(|&c| char_display_width(c)).sum();
                                        let (ns, ne) = lines[cur_line + 1];
                                        let end = if ne > ns && chars.get(ne - 1) == Some(&'\n') { ne - 1 } else { ne };
                                        let mut w = 0;
                                        let mut target = ns;
                                        for i in ns..end {
                                            if chars[i] == '\n' { continue; }
                                            let cw = char_display_width(chars[i]);
                                            if w + cw > col_width { break; }
                                            w += cw;
                                            target = i + 1;
                                        }
                                        state.input.set_cursor(target);
                                    } else if state.scroll_offset > 0 {
                                        // Already at bottom of input — scroll down
                                        state.scroll_offset = state.scroll_offset.saturating_sub(1);
                                    } else {
                                        state.input.move_end();
                                    }
                                }
                            }
                            last_ctrlc = 0;
                        }
                        Key::PageUp => {
                            let chat_h = state.height.saturating_sub(state.header_lines.len() + 3);
                            state.scroll_offset += chat_h.max(1);
                            last_ctrlc = 0;
                        }
                        Key::PageDown => {
                            let chat_h = state.height.saturating_sub(state.header_lines.len() + 3);
                            state.scroll_offset = state.scroll_offset.saturating_sub(chat_h.max(1));
                            last_ctrlc = 0;
                        }
                        Key::Home => { state.input.move_home(); last_ctrlc = 0; }
                        Key::End => { state.input.move_end(); last_ctrlc = 0; }
                        Key::Tab => {
                            let chars = state.input.to_chars();
                            let suggestion = slash_suggestion(&chars);
                            if !suggestion.is_empty() {
                                state.input.move_end();
                                state.input.insert_str(&suggestion);
                            }
                            last_ctrlc = 0;
                        }
                        Key::Escape => {
                            if let Some((scope, _)) = state.config_edit.take() {
                                // Cancel config edit, return to options menu
                                open_config_selector(&mut state, &format!("config:{scope}"), &config);
                            }
                            state.input.clear();
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
                            if !state.ctrlc_hint {
                                state.push_chat(ChatLine {
                                    kind: ChatLineKind::SystemInfo,
                                    content: "Press Ctrl+C again to exit".into(),
                                });
                                state.ctrlc_hint = true;
                            }
                        }
                    }
                    // Clear Ctrl+C hint when another key was pressed
                    if last_ctrlc == 0 && state.ctrlc_hint {
                        if let Some(pos) = state.chat.iter().rposition(|l| matches!(l.kind, ChatLineKind::SystemInfo) && l.content == "Press Ctrl+C again to exit") {
                            state.chat.remove(pos);
                        }
                        state.ctrlc_hint = false;
                    }
                    state.render();
                }
                Ok(Event::Resize) => {
                    state.update_dimensions();
                    state.render();
                }
                Ok(Event::Tick) => {
                    // Boot animation
                    if state.advance_cat_anim() {
                        state.render();
                    }
                    // Continue fade-in animation after streaming ends
                    if state.tick_fade() {
                        state.render();
                    }
                    // Clear expired Ctrl+C hint
                    if last_ctrlc > 0 && now_millis() - last_ctrlc >= 2000 {
                        last_ctrlc = 0;
                        if state.ctrlc_hint {
                            state.chat.pop();
                            state.ctrlc_hint = false;
                            state.render();
                        }
                    }
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

        // Remove Ctrl+C hint if it was shown
        if state.ctrlc_hint {
            state.chat.pop();
            state.ctrlc_hint = false;
            last_ctrlc = 0;
        }

        // AGENT PHASE
        if let Some(text) = user_input {
            // Config edit mode — save the entered value
            if let Some((scope, field)) = state.config_edit.take() {
                state.input_label.clear(); state.input_hint.clear();
                // Sanitize single-line fields
                let value = match field.as_str() {
                    "name" | "api_key" => text.replace('\n', "").trim().to_string(),
                    "heartbeat" => text.trim().trim_end_matches('s').trim().to_string(),
                    _ => text.clone(),
                };
                match save_config_field(&scope, &field, &value) {
                    Ok(_) => {
                        apply_config_change(&mut config, &field, &value);
                        let label = CONFIG_FIELDS.iter().find(|f| f.0 == field).map(|f| f.1).unwrap_or(&field.as_str());
                        state.push_chat(ChatLine {
                            kind: ChatLineKind::SystemInfo,
                            content: format!("{label} updated"),
                        });
                        if field == "name" || field == "system_prompt" || field == "heartbeat" {
                            state.build_header(&config, &tools_str, fmt_str, &config_path, config_source);
                        }
                        if field == "system_prompt" {
                            if let Some(msg) = messages.first_mut() {
                                msg.content = build_system_prompt(&config, &*format, &registry, &cwd);
                            }
                        }
                    }
                    Err(e) => {
                        state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Failed: {e}") });
                    }
                }
                open_config_selector(&mut state, &format!("config:{scope}"), &config);
                state.render();
                continue;
            }

            // Handle slash commands
            if text.starts_with('/') {
                state.push_chat(ChatLine { kind: ChatLineKind::UserMessage, content: text.clone() });
                state.render();

                let parts: Vec<&str> = text.splitn(2, ' ').collect();
                let cmd = parts[0];
                let args = parts.get(1).copied().unwrap_or("");

                match cmd {
                    "/help" => {
                        let mut help = String::new();
                        let cmds: Vec<_> = SLASH_COMMANDS.iter().filter(|(n, _)| *n != "/help").collect();
                        let name_w = cmds.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
                        for (name, desc) in cmds {
                            help.push_str(&format!("  {CYAN}{name:<name_w$}{RESET}  {DIM}{desc}{RESET}\n"));
                        }
                        state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: help.trim_end().to_string() });
                    }
                    "/model" => {
                        if args.is_empty() {
                            // Fetch models in background with spinner
                            state.start_spinner("fetching models...");
                            state.render();
                            let fetch_url = config.llm.api_url.clone();
                            let fetch_key = config.llm.api_key.clone();
                            let (model_tx, model_rx) = std::sync::mpsc::channel();
                            std::thread::spawn(move || {
                                let _ = model_tx.send(config::fetch_models(&fetch_url, &fetch_key));
                            });
                            let result = loop {
                                while let Ok(ev) = event_rx.try_recv() {
                                    match ev {
                                        Event::Tick => {
                                            state.advance_cat_anim();
                                            state.spinner.frame = (state.spinner.frame + 1) % 10;
                                            state.render();
                                        }
                                        Event::Resize => { state.update_dimensions(); state.render(); }
                                        _ => {}
                                    }
                                }
                                match model_rx.recv_timeout(Duration::from_millis(16)) {
                                    Ok(r) => break r,
                                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                                    Err(_) => break Err("Fetch thread died".into()),
                                }
                            };
                            state.stop_spinner();
                            let models = match result {
                                Ok(m) => {
                                    if let Ok(mut guard) = AVAILABLE_MODELS.lock() { *guard = m.clone(); }
                                    m
                                }
                                Err(e) => {
                                    state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Model fetch failed: {e}") });
                                    state.render();
                                    continue;
                                }
                            };
                            state.render();
                            if models.is_empty() {
                                state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: format!("Model: {}", config.llm.model) });
                            } else {
                                let items: Vec<SelectorItem> = models.iter().map(|m| {
                                    let ctx = m.ctx_display();
                                    let caps = m.caps_display();
                                    let label = if caps.is_empty() {
                                        format!("{} · {} ctx", m.display_name, ctx)
                                    } else {
                                        format!("{} · {} ctx · {}", m.display_name, ctx, caps)
                                    };
                                    SelectorItem { label, value: m.key.clone() }
                                }).collect();
                                let current_idx = models.iter().position(|m| m.key == config.llm.model).unwrap_or(0);
                                let max_vis = 8.min(items.len());
                                let mut sel = Selector::new("model", items, max_vis);
                                sel.selected = current_idx;
                                if sel.selected >= sel.max_visible {
                                    sel.scroll_offset = sel.selected.saturating_sub(sel.max_visible / 2);
                                }
                                state.selector = Some(sel);
                                state.input_label = "Model".into();
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
                                        vision_flag.store(info.vision, std::sync::atomic::Ordering::Relaxed);
                                    config.agent.compact_at = (info.context_length as f64 * 0.8) as usize;
                                }
                                if let Some(msg) = messages.first_mut() {
                                    msg.content = build_system_prompt(&config, &*format, &registry, &cwd);
                                }
                                state.push_chat(ChatLine {
                                    kind: ChatLineKind::SystemInfo,
                                    content: format!("Switched to {BOLD}{new_model}{RESET}"),
                                });
                                // Rebuild header + persist to config file
                                state.build_header(&config, &tools_str, fmt_str, &config_path, config_source);
                                if let Err(e) = save_model_to_config(&config_path, &new_model) {
                                    state.push_chat(ChatLine { kind: ChatLineKind::Error, content: format!("Config save failed: {e}") });
                                }
                            }
                        }
                    }
                    "/copy" => {
                        let last_agent = state.chat.iter().rev()
                            .find(|l| matches!(l.kind, ChatLineKind::AgentText));
                        if let Some(line) = last_agent {
                            match copy_to_clipboard(line.content.trim()) {
                                Ok(()) => state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: "Copied to clipboard".into() }),
                                Err(e) => state.push_chat(ChatLine { kind: ChatLineKind::Error, content: e }),
                            }
                        } else {
                            state.push_chat(ChatLine { kind: ChatLineKind::SystemInfo, content: "No agent response to copy".into() });
                        }
                    }
                    "/context" => {
                        state.push_chat(ChatLine {
                            kind: ChatLineKind::SystemInfo,
                            content: format_context_display(&messages, &config),
                        });
                    }
                    "/compact" => {
                        let token_count = estimate_tokens(&messages);
                        if token_count < 1000 {
                            state.push_chat(ChatLine {
                                kind: ChatLineKind::SystemInfo,
                                content: format!("Context too small to compact (~{token_count} tokens)"),
                            });
                        } else {
                            compact_context(&config.llm, &mut messages, token_count, &mut state, &event_rx);
                        }
                    }
                    "/config" => {
                        open_config_selector(&mut state, "config", &config);
                    }
                    "/clear" => {
                        let sys = messages.first().cloned().unwrap_or(Message { role: Role::System, content: String::new() });
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
                compact_context(&config.llm, &mut messages, token_count, &mut state, &event_rx);
            }

            // Process queued messages — merge batch into one API call
            while !queued.is_empty() {
                let batch: Vec<String> = queued.drain(..).collect();
                for q in &batch {
                    if let Some(pos) = state.chat.iter().position(|l| matches!(l.kind, ChatLineKind::QueuedMessage) && l.content == *q) {
                        state.chat[pos].kind = ChatLineKind::UserMessage;
                    }
                }
                state.render();
                let (qt, new_queued) = run_heartbeat(
                    &config, &registry, &*format, &tools_json,
                    &mut messages, batch, &mut state, &event_rx, &mut queued,
                );
                queued.extend(new_queued);
                let tc = qt.unwrap_or_else(|| estimate_tokens(&messages));
                if tc >= config.agent.compact_at {
                    compact_context(&config.llm, &mut messages, tc, &mut state, &event_rx);
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
                compact_context(&config.llm, &mut messages, token_count, &mut state, &event_rx);
            }

            // Process queued messages — merge batch into one API call
            while !queued.is_empty() {
                let batch: Vec<String> = queued.drain(..).collect();
                for q in &batch {
                    if let Some(pos) = state.chat.iter().position(|l| matches!(l.kind, ChatLineKind::QueuedMessage) && l.content == *q) {
                        state.chat[pos].kind = ChatLineKind::UserMessage;
                    }
                }
                state.render();
                let (qt, new_queued) = run_heartbeat(
                    &config, &registry, &*format, &tools_json,
                    &mut messages, batch, &mut state, &event_rx, &mut queued,
                );
                queued.extend(new_queued);
                let tc = qt.unwrap_or_else(|| estimate_tokens(&messages));
                if tc >= config.agent.compact_at {
                    compact_context(&config.llm, &mut messages, tc, &mut state, &event_rx);
                }
            }
        }
    }
}

fn timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Convert to local time using libc
    let t = secs as libc::time_t;
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        format!("{:02}:{:02}", tm.tm_hour, tm.tm_min)
    }
}

fn date_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let t = secs as libc::time_t;
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        format!("{:04}-{:02}-{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday)
    }
}
