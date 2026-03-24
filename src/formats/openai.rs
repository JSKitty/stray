/// OpenAI-compatible format: tools passed via API `tools` field.
/// Works with PPQ.ai, OpenRouter, LMStudio (with tool support), and most major models.
use super::{ModelFormat, ToolCall};
use crate::tools::ToolRegistry;
use serde_json::{json, Value};

pub struct OpenAIFormat {
    pub tags: Vec<String>,
}

impl ModelFormat for OpenAIFormat {
    fn format_tools(&self, registry: &ToolRegistry) -> Option<Value> {
        let tools: Vec<Value> = registry
            .tools()
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": tool.usage_hint()
                                }
                            },
                            "required": ["command"]
                        }
                    }
                })
            })
            .collect();

        Some(json!(tools))
    }

    fn system_prompt_suffix(&self, _registry: &ToolRegistry) -> String {
        // Tools are passed via API, not in the prompt
        String::from(
            "\n\nYou have tools available. Use them when needed. \
             When you have nothing more to do, respond with just your thoughts.",
        )
    }

    fn parse_response(&self, response: &str) -> (Vec<ToolCall>, String) {
        // OpenAI tool calls come in the API response structure, not in content.
        // But when proxied through PPQ or other providers, they sometimes appear
        // as content. Try to parse both.

        // First: check if response contains our raw XML tags (some providers
        // convert tool_calls back to content text)
        let raw_parser = super::raw_xml::RawXmlFormat { tags: self.tags.clone() };
        let (raw_calls, text) = raw_parser.parse_response(response);
        if !raw_calls.is_empty() {
            return (raw_calls, text);
        }

        // No tool calls found in content — return as plain text
        (Vec::new(), response.trim().to_string())
    }

    fn format_results(&self, results: &[(String, String, String)]) -> String {
        let mut s = String::from("Tool output:\n\n");
        for (tool, input, output) in results {
            s.push_str(&format!("[{}] $ {}\n{}\n\n", tool, input, output));
        }
        s
    }
}
