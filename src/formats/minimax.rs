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

        // Parse <minimax:tool_call> blocks
        while let Some(start) = response[search_from..].find(block_open) {
            let abs_start = search_from + start + block_open.len();
            if let Some(end) = response[abs_start..].find(block_close) {
                let block = &response[abs_start..abs_start + end];
                if let Some(call) = parse_invoke_block(block) {
                    calls.push(call);
                }
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
}

/// Parse <invoke name="tool"><parameter name="command">...</parameter></invoke>
fn parse_invoke_block(block: &str) -> Option<ToolCall> {
    let name_start = block.find("<invoke name=\"")? + "<invoke name=\"".len();
    let name_end = block[name_start..].find('"')? + name_start;
    let tool = block[name_start..name_end].to_string();

    let param_tag = "<parameter name=\"command\">";
    let param_start = block.find(param_tag)? + param_tag.len();
    let param_end = block[param_start..].find("</parameter>")? + param_start;
    let input = block[param_start..param_end].trim().to_string();

    Some(ToolCall { tool, input })
}

/// Parse {tool => "name", args => {--command "..."}} format
fn parse_tool_call_block(block: &str) -> Option<ToolCall> {
    // Extract tool name
    let tool_start = block.find("tool => \"")? + "tool => \"".len();
    let tool_end = block[tool_start..].find('"')? + tool_start;
    let tool = block[tool_start..tool_end].to_string();

    // Extract command from --command "..."
    let cmd_start = block.find("--command \"")? + "--command \"".len();
    let cmd_end = block[cmd_start..].find('"')? + cmd_start;
    let input = block[cmd_start..cmd_end].to_string();

    Some(ToolCall { tool, input })
}
