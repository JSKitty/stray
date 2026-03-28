mod openai;
mod minimax;
mod raw_xml;

use crate::tools::ToolRegistry;
use serde_json::Value;

/// A parsed tool invocation (universal internal format)
pub struct ToolCall {
    pub tool: String,
    pub input: String,
}

/// Model-specific format for converting tools to/from API wire format
pub trait ModelFormat {
    /// Inject tool definitions into the request body (e.g., OpenAI `tools` field)
    /// Returns None if tools go in the system prompt instead.
    fn format_tools(&self, registry: &ToolRegistry) -> Option<Value>;

    /// Append tool instructions to the system prompt (for formats that need it)
    fn system_prompt_suffix(&self, registry: &ToolRegistry) -> String;

    /// Parse the LLM response into tool calls + any text output
    fn parse_response(&self, response: &str) -> (Vec<ToolCall>, String);

    /// Format tool results back into a message for the LLM
    fn format_results(&self, results: &[(String, String, String)]) -> String;

    /// Extra XML tags to filter from streaming display (format-specific wrappers).
    /// Default: none. MiniMax overrides to add its wrapper tags.
    fn display_filter_tags(&self) -> Vec<&'static str> { Vec::new() }
}

/// Select the right format based on model ID
pub fn format_for_model(model: &str, registry: &ToolRegistry) -> Box<dyn ModelFormat> {
    let model_lower = model.to_lowercase();
    let tags: Vec<String> = registry.tags().into_iter().map(|s| s.to_string()).collect();

    if model_lower.starts_with("minimax/") {
        Box::new(minimax::MiniMaxFormat)
    } else if model_lower.starts_with("z-ai/")
        || model_lower.starts_with("openai/")
        || model_lower.starts_with("anthropic/")
        || model_lower.starts_with("google/")
        || model_lower.starts_with("meta/")
        || model_lower.starts_with("mistral/")
        || model_lower.starts_with("deepseek/")
        || model_lower.starts_with("qwen/")
        || model_lower.contains("gpt")
        || model_lower.contains("claude")
        || model_lower.contains("gemini")
        || model_lower.contains("glm")
    {
        Box::new(openai::OpenAIFormat { tags })
    } else {
        // Local models, unknown providers — use raw XML in prompt
        Box::new(raw_xml::RawXmlFormat { tags })
    }
}
