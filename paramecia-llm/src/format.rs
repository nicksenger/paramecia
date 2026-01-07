//! Message formatting utilities for API communication.

use crate::streaming_parser::{StreamingToolCallParser, merge_tool_call_chunks};
use crate::types::{LlmMessage, ToolCall};
use crate::xml_tool_parser::{ToolCallStyle, XmlToolCallParser};

/// Handles formatting and parsing of API messages.
///
/// This handler provides utilities for processing API responses and merging
/// streaming tool call chunks. It supports multiple tool call formats:
///
/// - **Standard**: OpenAI-compatible JSON function calls (streaming)
/// - **QwenVL/Hybrid**: JSON-in-XML format with `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` (used by Qwen3-Next, Qwen-Coder, Qwen-VL)
/// - **QwenCoder (Legacy)**: Pure XML format with `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
/// - **GeneralInline**: Inline format with `[tool_call: name for 'args']`
#[derive(Debug, Default)]
pub struct ApiToolFormatHandler {
    /// Parser for accumulating streaming tool calls (standard format).
    parser: StreamingToolCallParser,
    /// Parser for XML-based tool call formats (Qwen models).
    xml_parser: XmlToolCallParser,
    /// The active tool call style.
    style: ToolCallStyle,
}

impl ApiToolFormatHandler {
    /// Create a new format handler with standard (OpenAI-compatible) style.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parser: StreamingToolCallParser::new(),
            xml_parser: XmlToolCallParser::new(ToolCallStyle::Standard),
            style: ToolCallStyle::Standard,
        }
    }

    /// Create a format handler for a specific model.
    ///
    /// Automatically detects the appropriate tool call style based on model name:
    /// - `qwen3-next`, `qwen*-coder`, `qwen*-vl`: Uses hybrid XML-JSON format
    /// - Others: Uses standard OpenAI-compatible JSON format
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        let style = ToolCallStyle::from_model_name(model);
        Self {
            parser: StreamingToolCallParser::new(),
            xml_parser: XmlToolCallParser::new(style),
            style,
        }
    }

    /// Get the current tool call style.
    #[must_use]
    pub fn style(&self) -> ToolCallStyle {
        self.style
    }

    /// Set the tool call style.
    pub fn set_style(&mut self, style: ToolCallStyle) {
        self.style = style;
        self.xml_parser.set_style(style);
    }

    /// Check if this handler uses XML-based tool call format.
    #[must_use]
    pub fn uses_xml_format(&self) -> bool {
        self.style.is_xml_based()
    }

    /// Process an API response message into our internal format.
    #[must_use]
    pub fn process_api_response_message(&self, message: LlmMessage) -> LlmMessage {
        // The message is already in our internal format, but we can do
        // any necessary normalization here
        message
    }

    /// Get the default tool choice directive.
    #[must_use]
    pub fn get_tool_choice(&self) -> Option<&'static str> {
        Some("auto")
    }

    /// Add a streaming tool call chunk and return parse result.
    ///
    /// Use this method when processing streaming responses to accumulate
    /// tool call arguments across multiple chunks (standard format).
    ///
    /// For XML-based formats, use `add_content` and `extract_xml_tool_calls` instead.
    ///
    /// # Arguments
    ///
    /// * `index` - Tool call index from the streaming response
    /// * `chunk` - The argument chunk to add
    /// * `id` - Optional tool call ID
    /// * `name` - Optional function name
    ///
    /// # Returns
    ///
    /// A `ToolCallParseResult` indicating whether parsing is complete
    pub fn add_tool_call_chunk(
        &mut self,
        index: u32,
        chunk: &str,
        id: Option<&str>,
        name: Option<&str>,
    ) -> crate::streaming_parser::ToolCallParseResult {
        self.parser.add_chunk(index, chunk, id, name)
    }

    /// Add content to the XML parser buffer.
    ///
    /// Use this for XML-based formats (QwenCoder, QwenVL) where tool calls
    /// are embedded in the text content rather than structured function calls.
    pub fn add_content(&mut self, content: &str) {
        self.xml_parser.add_content(content);
    }

    /// Check if the XML buffer contains complete tool calls.
    #[must_use]
    pub fn has_complete_xml_tool_calls(&self) -> bool {
        self.xml_parser.has_complete_tool_calls()
    }

    /// Check if the XML buffer contains a partial (incomplete) tool call.
    #[must_use]
    pub fn has_partial_xml_tool_call(&self) -> bool {
        self.xml_parser.has_partial_tool_call()
    }

    /// Extract tool calls from the XML buffer.
    ///
    /// Returns the tool calls and updates the buffer to contain only
    /// the non-tool-call content (text between/around tool calls).
    pub fn extract_xml_tool_calls(&mut self) -> Vec<ToolCall> {
        let parsed = self.xml_parser.extract_tool_calls();
        self.xml_parser.to_tool_calls(parsed)
    }

    /// Get the remaining text content after extracting XML tool calls.
    #[must_use]
    pub fn get_remaining_content(&self) -> &str {
        self.xml_parser.buffer()
    }

    /// Get all completed tool calls from the standard streaming parser.
    ///
    /// Call this after streaming is complete (when finish_reason is present)
    /// to get all accumulated tool calls with repaired JSON if needed.
    #[must_use]
    pub fn get_completed_tool_calls(&self) -> Vec<ToolCall> {
        self.parser.into_tool_calls()
    }

    /// Parse tool calls from text content.
    ///
    /// This is useful for parsing complete messages that may contain
    /// tool calls in any supported format. The parser will use the
    /// configured style to parse the content.
    #[must_use]
    pub fn parse_tool_calls_from_content(&self, content: &str) -> Vec<ToolCall> {
        let parsed = self.xml_parser.parse(content);
        let mut parser = XmlToolCallParser::new(self.style);
        parser.to_tool_calls(parsed)
    }

    /// Reset all internal parser state for a new stream.
    pub fn reset_parser(&mut self) {
        self.parser.reset();
        self.xml_parser.reset();
    }

    /// Check if there are incomplete tool calls being parsed (standard format).
    #[must_use]
    pub fn has_incomplete_calls(&self) -> bool {
        self.parser.has_incomplete_calls()
    }

    /// Merge streaming tool call chunks into complete tool calls.
    ///
    /// This is a convenience method that processes a batch of chunks at once.
    /// For streaming scenarios, prefer using `add_tool_call_chunk` followed
    /// by `get_completed_tool_calls`.
    ///
    /// # Features
    ///
    /// - Handles index collisions when same index is reused
    /// - Routes continuation chunks without IDs correctly
    /// - Repairs malformed JSON (unclosed strings, trailing commas)
    /// - Extracts valid JSON from surrounding garbage
    #[must_use]
    pub fn merge_tool_call_chunks(&self, chunks: &[ToolCall]) -> Vec<ToolCall> {
        merge_tool_call_chunks(chunks)
    }

    /// Get a hash of a tool call for deduplication or loop detection.
    #[must_use]
    pub fn hash_tool_call(name: &str, args: &serde_json::Value) -> String {
        StreamingToolCallParser::hash_tool_call(name, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionCall;

    #[test]
    fn test_merge_tool_call_chunks() {
        let handler = ApiToolFormatHandler::new();

        let chunks = vec![
            ToolCall {
                id: Some("call_1".to_string()),
                index: Some(0),
                function: FunctionCall {
                    name: Some("bash".to_string()),
                    arguments: Some(r#"{"com"#.to_string()),
                },
                r#type: "function".to_string(),
            },
            ToolCall {
                id: None,
                index: Some(0),
                function: FunctionCall {
                    name: None,
                    arguments: Some(r#"mand":"#.to_string()),
                },
                r#type: "function".to_string(),
            },
            ToolCall {
                id: None,
                index: Some(0),
                function: FunctionCall {
                    name: None,
                    arguments: Some(r#""ls"}"#.to_string()),
                },
                r#type: "function".to_string(),
            },
        ];

        let merged = handler.merge_tool_call_chunks(&chunks);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, Some("call_1".to_string()));
        assert_eq!(merged[0].function.name, Some("bash".to_string()));

        // Parse the merged arguments to verify correctness
        let args: serde_json::Value =
            serde_json::from_str(merged[0].function.arguments.as_ref().unwrap()).unwrap();
        assert_eq!(args.get("command").and_then(|v| v.as_str()), Some("ls"));
    }

    #[test]
    fn test_streaming_add_chunks() {
        let mut handler = ApiToolFormatHandler::new();

        // First chunk with ID and name
        let result = handler.add_tool_call_chunk(0, r#"{"com"#, Some("call_1"), Some("bash"));
        assert!(!result.complete);

        // Continuation
        let result = handler.add_tool_call_chunk(0, r#"mand": "ls"}"#, None, None);
        assert!(result.complete);

        let calls = handler.get_completed_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, Some("bash".to_string()));
    }

    #[test]
    fn test_hash_tool_call() {
        let hash1 =
            ApiToolFormatHandler::hash_tool_call("bash", &serde_json::json!({"command": "ls"}));
        let hash2 =
            ApiToolFormatHandler::hash_tool_call("bash", &serde_json::json!({"command": "ls"}));
        let hash3 =
            ApiToolFormatHandler::hash_tool_call("bash", &serde_json::json!({"command": "pwd"}));

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_reset_parser() {
        let mut handler = ApiToolFormatHandler::new();

        handler.add_tool_call_chunk(0, r#"{"command": "ls"}"#, Some("call_1"), Some("bash"));

        handler.reset_parser();

        assert!(!handler.has_incomplete_calls());
        assert!(handler.get_completed_tool_calls().is_empty());
    }

    #[test]
    fn test_multiple_parallel_tool_calls() {
        let handler = ApiToolFormatHandler::new();

        let chunks = vec![
            ToolCall {
                id: Some("call_1".to_string()),
                index: Some(0),
                function: FunctionCall {
                    name: Some("bash".to_string()),
                    arguments: Some(r#"{"command":"#.to_string()),
                },
                r#type: "function".to_string(),
            },
            ToolCall {
                id: Some("call_2".to_string()),
                index: Some(1),
                function: FunctionCall {
                    name: Some("read_file".to_string()),
                    arguments: Some(r#"{"path":"#.to_string()),
                },
                r#type: "function".to_string(),
            },
            ToolCall {
                id: None,
                index: Some(0),
                function: FunctionCall {
                    name: None,
                    arguments: Some(r#" "ls"}"#.to_string()),
                },
                r#type: "function".to_string(),
            },
            ToolCall {
                id: None,
                index: Some(1),
                function: FunctionCall {
                    name: None,
                    arguments: Some(r#" "/home"}"#.to_string()),
                },
                r#type: "function".to_string(),
            },
        ];

        let merged = handler.merge_tool_call_chunks(&chunks);
        assert_eq!(merged.len(), 2);

        // Find each call by ID
        let call_1 = merged
            .iter()
            .find(|c| c.id.as_deref() == Some("call_1"))
            .unwrap();
        let call_2 = merged
            .iter()
            .find(|c| c.id.as_deref() == Some("call_2"))
            .unwrap();

        let args_1: serde_json::Value =
            serde_json::from_str(call_1.function.arguments.as_ref().unwrap()).unwrap();
        let args_2: serde_json::Value =
            serde_json::from_str(call_2.function.arguments.as_ref().unwrap()).unwrap();

        assert_eq!(args_1.get("command").and_then(|v| v.as_str()), Some("ls"));
        assert_eq!(args_2.get("path").and_then(|v| v.as_str()), Some("/home"));
    }

    #[test]
    fn test_for_model_detection() {
        use crate::xml_tool_parser::ToolCallStyle;

        // Qwen3-Coder now uses hybrid XML-JSON format (QwenVl style)
        let handler = ApiToolFormatHandler::for_model("qwen3-coder");
        assert_eq!(handler.style(), ToolCallStyle::QwenVl);
        assert!(handler.uses_xml_format());

        // Qwen3-Next uses hybrid XML-JSON format
        let handler = ApiToolFormatHandler::for_model("qwen3-next");
        assert_eq!(handler.style(), ToolCallStyle::QwenVl);

        let handler = ApiToolFormatHandler::for_model("qwen2-vl");
        assert_eq!(handler.style(), ToolCallStyle::QwenVl);

        let handler = ApiToolFormatHandler::for_model("gpt-4");
        assert_eq!(handler.style(), ToolCallStyle::Standard);
        assert!(!handler.uses_xml_format());
    }

    #[test]
    fn test_xml_tool_call_extraction() {
        let mut handler = ApiToolFormatHandler::for_model("qwen3-coder");

        // Simulate streaming content with embedded tool call (hybrid XML-JSON format)
        handler.add_content("Let me check that file.\n");
        handler.add_content("<tool_call>\n");
        handler.add_content(r#"{"name": "read_file", "arguments": {"path": "/src/main.rs"}}"#);
        handler.add_content("\n</tool_call>\n");
        handler.add_content("Done checking.");

        assert!(handler.has_complete_xml_tool_calls());

        let calls = handler.extract_xml_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, Some("read_file".to_string()));

        // Remaining content should not have the tool call
        let remaining = handler.get_remaining_content();
        assert!(remaining.contains("Let me check that file."));
        assert!(remaining.contains("Done checking."));
        assert!(!remaining.contains("<tool_call>"));
    }

    #[test]
    fn test_parse_tool_calls_from_content() {
        let handler = ApiToolFormatHandler::for_model("qwen3-coder");

        // Use hybrid XML-JSON format
        let content = r#"
Here's what I'll do:
<tool_call>
{"name": "bash", "arguments": {"command": "cargo build"}}
</tool_call>
Building now...
"#;

        let calls = handler.parse_tool_calls_from_content(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, Some("bash".to_string()));

        let args: serde_json::Value =
            serde_json::from_str(calls[0].function.arguments.as_ref().unwrap()).unwrap();
        assert_eq!(
            args.get("command").and_then(|v| v.as_str()),
            Some("cargo build")
        );
    }

    #[test]
    fn test_qwen_vl_format_in_handler() {
        let handler = ApiToolFormatHandler::for_model("qwen2-vl");

        let content =
            r#"<tool_call>{"name": "bash", "arguments": {"command": "ls -la"}}</tool_call>"#;

        let calls = handler.parse_tool_calls_from_content(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, Some("bash".to_string()));
    }
}
