/// MiniMax format: uses <minimax:tool_call> XML blocks.
/// Translates to/from MiniMax's native format while keeping internal representation universal.
use super::{ModelFormat, ToolCall};
use crate::tools::ToolRegistry;
use serde_json::Value;

pub struct MiniMaxFormat;

impl ModelFormat for MiniMaxFormat {
    fn format_tools(&self, _registry: &ToolRegistry) -> Option<Value> {
        None // MiniMax uses XML in the prompt
    }

    fn system_prompt_suffix(&self, registry: &ToolRegistry) -> String {
        let mut s = String::from("\n\nYou have access to the following tools:\n\n");

        for tool in registry.tools() {
            s.push_str(&format!(
                "Tool: {}\nDescription: {}\n\
                 To use it, write:\n\
                 <minimax:tool_call>\n\
                 <invoke name=\"{}\">\n\
                 <parameter name=\"command\">{}</parameter>\n\
                 </invoke>\n\
                 </minimax:tool_call>\n\n",
                tool.name(),
                tool.description(),
                tool.name(),
                tool.usage_hint(),
            ));
        }

        s.push_str(
            "You can use multiple tool calls in a single response. After each round, \
             you'll see their output and can decide what to do next.\n\n\
             When you have nothing more to do, respond with just your thoughts — no tool calls.",
        );

        s
    }

    fn parse_response(&self, response: &str) -> (Vec<ToolCall>, String) {
        let mut calls = Vec::new();
        let mut remaining = response.to_string();
        let mut search_from = 0;

        let block_open = "<minimax:tool_call>";
        let block_close = "</minimax:tool_call>";

        // Also handle [TOOL_CALL] format (some MiniMax versions use this)
        let alt_open = "[TOOL_CALL]";
        let alt_close = "[/TOOL_CALL]";

        // Parse <minimax:tool_call> blocks (may contain multiple <invoke> elements)
        while let Some(start) = response[search_from..].find(block_open) {
            let abs_start = search_from + start + block_open.len();
            if let Some(end) = response[abs_start..].find(block_close) {
                let block = &response[abs_start..abs_start + end];
                calls.extend(parse_invoke_block(block));
                search_from = abs_start + end + block_close.len();
            } else {
                break;
            }
        }

        // Parse [TOOL_CALL] blocks as fallback
        if calls.is_empty() {
            search_from = 0;
            while let Some(start) = response[search_from..].find(alt_open) {
                let abs_start = search_from + start + alt_open.len();
                if let Some(end) = response[abs_start..].find(alt_close) {
                    let block = &response[abs_start..abs_start + end];
                    if let Some(call) = parse_tool_call_block(block) {
                        calls.push(call);
                    }
                    search_from = abs_start + end + alt_close.len();
                } else {
                    break;
                }
            }
        }

        // Extract text before first tool block
        let first_marker = [block_open, alt_open]
            .iter()
            .filter_map(|m| response.find(m))
            .min();
        if let Some(pos) = first_marker {
            remaining = response[..pos].trim().to_string();
        } else if calls.is_empty() {
            remaining = response.trim().to_string();
        }

        (calls, remaining)
    }

    fn format_results(&self, results: &[(String, String, String)]) -> String {
        let mut s = String::from("Tool output:\n\n");
        for (tool, input, output) in results {
            s.push_str(&format!("[{}] $ {}\n{}\n\n", tool, input, output));
        }
        s
    }

    fn display_filter_tags(&self) -> Vec<&'static str> {
        vec!["minimax:tool_call"]
    }
}

/// Parse all <invoke name="tool"><parameter name="command">...</parameter></invoke> blocks
fn parse_invoke_block(block: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;
    let invoke_open = "<invoke name=\"";
    let invoke_close = "</invoke>";

    while let Some(start) = block[search_from..].find(invoke_open) {
        let abs_start = search_from + start + invoke_open.len();
        if let Some(name_end) = block[abs_start..].find('"') {
            let tool = block[abs_start..abs_start + name_end].to_string();

            // Find the closing </invoke> for this invocation
            let invoke_body_start = abs_start + name_end;
            if let Some(close) = block[invoke_body_start..].find(invoke_close) {
                let invoke_body = &block[invoke_body_start..invoke_body_start + close];

                let param_tag = "<parameter name=\"command\">";
                if let Some(ps) = invoke_body.find(param_tag) {
                    let ps = ps + param_tag.len();
                    if let Some(pe) = invoke_body[ps..].find("</parameter>") {
                        let input = invoke_body[ps..ps + pe].trim().to_string();
                        calls.push(ToolCall { tool, input });
                    }
                }
                search_from = invoke_body_start + close + invoke_close.len();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    calls
}

/// Parse {tool => "name", args => {--command "..."}} format
fn parse_tool_call_block(block: &str) -> Option<ToolCall> {
    // Extract tool name
    let tool_start = block.find("tool => \"")? + "tool => \"".len();
    let tool_end = block[tool_start..].find('"')? + tool_start;
    let tool = block[tool_start..tool_end].to_string();

    // Extract command — find opening quote after --command, then match the LAST quote
    // before the closing brace to handle embedded quotes in the command
    let cmd_marker = "--command \"";
    let cmd_start = block.find(cmd_marker)? + cmd_marker.len();
    let rest = &block[cmd_start..];
    // Find last `"` in the remaining block (handles embedded quotes)
    let cmd_end = rest.rfind('"').unwrap_or(rest.len());
    let input = rest[..cmd_end].to_string();

    Some(ToolCall { tool, input })
}
