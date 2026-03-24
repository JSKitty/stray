/// Fallback format: XML tags in the system prompt.
/// Works with any instruction-following model. No API-level tool support needed.
use super::{ModelFormat, ToolCall};
use crate::tools::ToolRegistry;
use serde_json::Value;

pub struct RawXmlFormat {
    pub tags: Vec<String>,
}

impl ModelFormat for RawXmlFormat {
    fn format_tools(&self, _registry: &ToolRegistry) -> Option<Value> {
        None // Tools go in the system prompt
    }

    fn system_prompt_suffix(&self, registry: &ToolRegistry) -> String {
        let mut s = String::from("\n\nYou have access to the following tools:\n\n");

        for tool in registry.tools() {
            s.push_str(&format!(
                "**{}** — {}\nUsage: <{}>{}</{}>\n\n",
                tool.name(),
                tool.description(),
                tool.tag(),
                tool.usage_hint(),
                tool.tag(),
            ));
        }

        s.push_str(
            "IMPORTANT: Use the EXACT XML tag format shown above to invoke tools. \
             Do NOT use any other format.\n\n\
             You can use multiple tools in a single response. After each round, \
             you'll see their output and can decide what to do next.\n\n\
             When you have nothing more to do, respond with just your thoughts — no tool tags.",
        );

        s
    }

    fn parse_response(&self, response: &str) -> (Vec<ToolCall>, String) {
        let mut calls = Vec::new();
        let mut remaining = response.to_string();

        // Extract all <tag>...</tag> blocks for known tool tags
        for tag in &self.tags {
            let tag = tag.as_str();
            let open = format!("<{}>", tag);
            let close = format!("</{}>", tag);
            let mut search_from = 0;

            while let Some(start) = remaining[search_from..].find(&open) {
                let abs_start = search_from + start + open.len();
                if let Some(end) = remaining[abs_start..].find(&close) {
                    let content = remaining[abs_start..abs_start + end].trim().to_string();
                    if !content.is_empty() {
                        calls.push(ToolCall {
                            tool: tag.to_string(),
                            input: content,
                        });
                    }
                    search_from = abs_start + end + close.len();
                } else {
                    break;
                }
            }
        }

        // Extract text reasoning (everything before the first tool tag)
        let first_tag_pos = self.tags.iter()
            .filter_map(|tag| remaining.find(&format!("<{}>", tag)))
            .min();
        if let Some(pos) = first_tag_pos {
            remaining = remaining[..pos].to_string();
        }

        (calls, remaining.trim().to_string())
    }

    fn format_results(&self, results: &[(String, String, String)]) -> String {
        let mut s = String::from("Tool output:\n\n");
        for (tool, input, output) in results {
            s.push_str(&format!("[{}] $ {}\n{}\n\n", tool, input, output));
        }
        s
    }
}
