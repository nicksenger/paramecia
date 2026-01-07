//! Model-specific XML tool call parsing for Qwen models.
//!
//! This module handles parsing of various tool call formats used by Qwen models:
//!
//! ## Qwen3-Next Hybrid XML-JSON Format (Primary)
//! ```text
//! <tool_call>
//! {"name": "bash", "arguments": {"command": "ls -la"}}
//! </tool_call>
//! ```
//!
//! ## Legacy Qwen-Coder Pure XML Format
//! ```text
//! <tool_call>
//! <function=bash>
//! <parameter=command>
//! ls -la
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! ## General Inline Format
//! ```text
//! [tool_call: bash for 'ls -la']
//! ```

use crate::types::{FunctionCall, ToolCall};
use regex::Regex;
use std::sync::LazyLock;

/// Tool call format style detected or configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallStyle {
    /// Standard OpenAI-compatible JSON function calls (handled by streaming parser).
    #[default]
    Standard,
    /// Legacy Qwen-Coder pure XML format: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    QwenCoder,
    /// Hybrid XML-JSON format: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
    /// Used by Qwen3-Next and Qwen-VL models.
    QwenVl,
    /// General inline format: `[tool_call: name for 'args']`
    GeneralInline,
    /// Preserve original format: keep tool calls in their original format without reformatting
    Preserve,
}

impl ToolCallStyle {
    /// Detect the appropriate tool call style based on model name.
    #[must_use]
    pub fn from_model_name(model: &str) -> Self {
        let model_lower = model.to_lowercase();

        // Match qwen3-next (uses hybrid XML-JSON format)
        if model_lower.contains("qwen3-next") || model_lower.contains("qwen3next") {
            return Self::QwenVl;
        }

        // Match qwen*-coder patterns (e.g., qwen3-coder, qwen2.5-coder, qwen-coder)
        // Note: These may also use hybrid format in newer versions
        if model_lower.contains("qwen") && model_lower.contains("coder") {
            return Self::QwenVl;
        }

        // Match qwen*-vl patterns (e.g., qwen-vl, qwen2-vl, qwen3-vl)
        if model_lower.contains("qwen") && model_lower.contains("-vl") {
            return Self::QwenVl;
        }

        // Match coder-model pattern
        if model_lower.contains("coder-model") {
            return Self::QwenVl;
        }

        // Match vision-model pattern
        if model_lower.contains("vision-model") {
            return Self::QwenVl;
        }

        Self::Standard
    }

    /// Check if this style uses XML-based formatting.
    #[must_use]
    pub fn is_xml_based(&self) -> bool {
        matches!(self, Self::QwenCoder | Self::QwenVl | Self::Preserve)
    }

    /// Parse a tool call style from a config string.
    ///
    /// Accepted values:
    /// - `"xml"` → `QwenCoder` (pure XML format, default)
    /// - `"json_in_xml"` → `QwenVl` (JSON-in-XML format)
    /// - `"preserve"` → `Preserve` (keep original format)
    /// - `None` / unrecognized → `QwenCoder` (default)
    #[must_use]
    pub fn from_config(value: Option<&str>) -> Self {
        match value.map(|s| s.to_lowercase()).as_deref() {
            Some("json_in_xml") => Self::QwenVl,
            Some("preserve") => Self::Preserve,
            Some("xml") | None => Self::QwenCoder,
            Some(other) => {
                tracing::warn!("Unknown tool_call_format '{}', defaulting to 'xml'", other);
                Self::QwenCoder
            }
        }
    }
}

// Regex patterns for parsing (compiled once)
// Match complete or incomplete tool_call blocks (ref: vLLM/sglang Qwen3CoderToolParser)
static TOOL_CALL_BLOCK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$").unwrap());

// Match complete or incomplete function blocks
static QWEN_CODER_FUNCTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<function=([^>]+)>(.*?)</function>|<function=([^>]+)>(.*?)$").unwrap()
});

// Match parameter start tags to find parameter boundaries
static QWEN_CODER_PARAM_START_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<parameter=([^>]+)>").unwrap());

static INLINE_TOOL_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Match: [tool_call: name for 'args'] or [tool_call: name for "args"] or [tool_call: name for args]
    Regex::new(r#"\[tool_call:\s*(\w+)\s+for\s+(?:'([^']*)'|"([^"]*)"|([^\]]+))\]"#).unwrap()
});

/// A parsed tool call from XML or inline format.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    /// Function name.
    pub name: String,
    /// Arguments as a JSON object.
    pub arguments: serde_json::Value,
    /// Start position in the source text.
    pub start: usize,
    /// End position in the source text.
    pub end: usize,
    /// Original raw text of the tool call (for "preserve" mode).
    pub raw_text: Option<String>,
}

/// Parser for model-specific tool call formats.
#[derive(Debug, Default)]
pub struct XmlToolCallParser {
    /// The tool call style to use for parsing.
    style: ToolCallStyle,
    /// Buffer for accumulating streaming content.
    buffer: String,
    /// Counter for generating tool call IDs.
    id_counter: u32,
}

impl XmlToolCallParser {
    /// Create a new parser with the specified style.
    #[must_use]
    pub fn new(style: ToolCallStyle) -> Self {
        Self {
            style,
            buffer: String::new(),
            id_counter: 0,
        }
    }

    /// Create a parser based on model name detection.
    #[must_use]
    pub fn from_model(model: &str) -> Self {
        Self::new(ToolCallStyle::from_model_name(model))
    }

    /// Get the current style.
    #[must_use]
    pub fn style(&self) -> ToolCallStyle {
        self.style
    }

    /// Set the style.
    pub fn set_style(&mut self, style: ToolCallStyle) {
        self.style = style;
    }

    /// Add content to the buffer (for streaming).
    pub fn add_content(&mut self, content: &str) {
        self.buffer.push_str(content);
    }

    /// Get the current buffer content.
    #[must_use]
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Clear the buffer.
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }

    /// Reset the parser state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.id_counter = 0;
    }

    /// Parse all tool calls from the given text.
    #[must_use]
    pub fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        match self.style {
            ToolCallStyle::Standard => Vec::new(), // Handled by streaming parser
            ToolCallStyle::QwenCoder => self.parse_qwen_coder(text),
            ToolCallStyle::QwenVl => self.parse_qwen_vl(text),
            ToolCallStyle::GeneralInline => self.parse_inline(text),
            ToolCallStyle::Preserve => self.parse_any_xml(text), // Accept any format
        }
    }

    /// Parse tool calls from the buffer and return any complete ones.
    ///
    /// Uses `parse_any_xml` to accept both JSON-in-XML and pure XML formats.
    /// Returns a tuple of (tool_calls, remaining_text).
    /// The remaining text should be kept for display or further processing.
    #[must_use]
    pub fn parse_buffer(&self) -> (Vec<ParsedToolCall>, String) {
        let tool_calls = self.parse_any_xml(&self.buffer);

        if tool_calls.is_empty() {
            return (Vec::new(), self.buffer.clone());
        }

        // Remove parsed tool calls from the text, keeping non-tool-call content
        let mut remaining = self.buffer.clone();
        let mut offset = 0;

        for tc in &tool_calls {
            let start = tc.start.saturating_sub(offset);
            let end = tc.end.saturating_sub(offset);
            if end <= remaining.len() {
                remaining.replace_range(start..end, "");
                offset += tc.end - tc.start;
            }
        }

        (tool_calls, remaining.trim().to_string())
    }

    /// Extract and consume tool calls from the buffer.
    ///
    /// Returns the parsed tool calls and updates the buffer to contain
    /// only the non-tool-call content.
    pub fn extract_tool_calls(&mut self) -> Vec<ParsedToolCall> {
        let (tool_calls, remaining) = self.parse_buffer();
        self.buffer = remaining;
        tool_calls
    }

    /// Convert parsed tool calls to ToolCall type.
    #[must_use]
    pub fn to_tool_calls(&mut self, parsed: Vec<ParsedToolCall>) -> Vec<ToolCall> {
        parsed
            .into_iter()
            .map(|p| {
                let id = format!("xml_call_{}", self.id_counter);
                self.id_counter += 1;

                ToolCall {
                    id: Some(id),
                    index: Some(self.id_counter as usize - 1),
                    function: FunctionCall {
                        name: Some(p.name),
                        arguments: Some(p.arguments.to_string()),
                    },
                    r#type: "function".to_string(),
                    raw_text: p.raw_text,
                }
            })
            .collect()
    }

    /// Parse Qwen-Coder XML format (pure XML only).
    fn parse_qwen_coder(&self, text: &str) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();
        for cap in TOOL_CALL_BLOCK_RE.captures_iter(text) {
            let full_match = cap.get(0).unwrap();
            let inner = cap.get(1).or_else(|| cap.get(2)).unwrap().as_str();
            if let Some(parsed) = self.try_parse_xml_block(inner, &full_match, text) {
                results.push(parsed);
            }
        }
        results
    }

    /// Parse Qwen-VL JSON-in-XML format (JSON only).
    fn parse_qwen_vl(&self, text: &str) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();
        for cap in TOOL_CALL_BLOCK_RE.captures_iter(text) {
            let full_match = cap.get(0).unwrap();
            let inner = cap.get(1).or_else(|| cap.get(2)).unwrap().as_str();
            if let Some(parsed) = Self::try_parse_json_block(inner.trim(), &full_match, text) {
                results.push(parsed);
            }
        }
        results
    }

    /// Parse general inline format.
    fn parse_inline(&self, text: &str) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();

        for cap in INLINE_TOOL_CALL_RE.captures_iter(text) {
            let full_match = cap.get(0).unwrap();
            let func_name = cap.get(1).unwrap().as_str();

            // Get the argument from whichever capture group matched
            let arg_value = cap
                .get(2)
                .or_else(|| cap.get(3))
                .or_else(|| cap.get(4))
                .map(|m| m.as_str().trim())
                .unwrap_or("");

            // Try to parse argument as JSON, otherwise use as raw value
            let arguments = if arg_value.starts_with('{') {
                serde_json::from_str(arg_value).unwrap_or_else(|_| {
                    // If it looks like JSON but fails, create a simple args object
                    serde_json::json!({ "input": arg_value })
                })
            } else {
                // For simple string args, infer the parameter name from the function
                let param_name = infer_param_name(func_name);
                serde_json::json!({ param_name: arg_value })
            };

            let raw_text = text
                .get(full_match.start()..full_match.end())
                .map(|s| s.to_string());

            results.push(ParsedToolCall {
                name: func_name.to_string(),
                arguments,
                start: full_match.start(),
                end: full_match.end(),
                raw_text,
            });
        }

        results
    }

    /// Parse tool calls from text, accepting both JSON-in-XML and pure XML formats.
    ///
    /// For each `<tool_call>` block, tries JSON-in-XML first (with repair), then
    /// falls back to pure XML. This allows a single conversation to contain both
    /// formats — the model might switch between them across turns.
    #[must_use]
    pub fn parse_any_xml(&self, text: &str) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();

        for cap in TOOL_CALL_BLOCK_RE.captures_iter(text) {
            let full_match = cap.get(0).unwrap();
            let inner = cap.get(1).or_else(|| cap.get(2)).unwrap().as_str();
            let inner_trimmed = inner.trim();

            // Try JSON-in-XML first (more specific: content must start with '{')
            if inner_trimmed.starts_with('{')
                && let Some(parsed) = Self::try_parse_json_block(inner_trimmed, &full_match, text)
            {
                results.push(parsed);
                continue;
            }

            // Fall back to pure XML (<function=...> tags)
            if let Some(parsed) = self.try_parse_xml_block(inner, &full_match, text) {
                results.push(parsed);
            }
        }

        results
    }

    /// Try to parse a tool_call block as JSON-in-XML format.
    fn try_parse_json_block(
        inner: &str,
        full_match: &regex::Match<'_>,
        source_text: &str,
    ) -> Option<ParsedToolCall> {
        let json = serde_json::from_str::<serde_json::Value>(inner)
            .ok()
            .or_else(|| repair_json_in_xml(inner))?;

        let name = json
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if name.is_empty() {
            return None;
        }

        let arguments = json
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        let raw_text = source_text
            .get(full_match.start()..full_match.end())
            .map(|s| s.to_string());

        Some(ParsedToolCall {
            name,
            arguments,
            start: full_match.start(),
            end: full_match.end(),
            raw_text,
        })
    }

    /// Try to parse a tool_call block as pure XML format.
    fn try_parse_xml_block(
        &self,
        inner: &str,
        full_match: &regex::Match<'_>,
        source_text: &str,
    ) -> Option<ParsedToolCall> {
        let func_cap = QWEN_CODER_FUNCTION_RE.captures(inner)?;

        let func_name = func_cap.get(1).or_else(|| func_cap.get(3))?.as_str().trim();
        let func_body = func_cap.get(2).or_else(|| func_cap.get(4))?.as_str();

        let mut args = serde_json::Map::new();
        let param_starts: Vec<_> = QWEN_CODER_PARAM_START_RE
            .captures_iter(func_body)
            .map(|c| {
                let m = c.get(0).unwrap();
                let name = c.get(1).unwrap().as_str().trim().to_string();
                (name, m.end())
            })
            .collect();

        for (param_name, value_start) in &param_starts {
            let remaining = &func_body[*value_start..];
            let end_offset = [
                remaining.find("</parameter>"),
                remaining.find("<parameter="),
                remaining.find("</function>"),
            ]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(remaining.len());

            let raw_value = &remaining[..end_offset];
            let param_value = strip_bounding_newlines(raw_value);

            let value = serde_json::from_str(param_value)
                .unwrap_or_else(|_| serde_json::Value::String(param_value.to_string()));

            args.insert(param_name.to_string(), value);
        }

        let raw_text = source_text
            .get(full_match.start()..full_match.end())
            .map(|s| s.to_string());

        Some(ParsedToolCall {
            name: func_name.to_string(),
            arguments: serde_json::Value::Object(args),
            start: full_match.start(),
            end: full_match.end(),
            raw_text,
        })
    }

    /// Check if the buffer contains any complete tool calls.
    #[must_use]
    pub fn has_complete_tool_calls(&self) -> bool {
        match self.style {
            ToolCallStyle::Standard => false,
            ToolCallStyle::QwenCoder | ToolCallStyle::QwenVl | ToolCallStyle::Preserve => {
                self.buffer.contains("</tool_call>")
            }
            ToolCallStyle::GeneralInline => INLINE_TOOL_CALL_RE.is_match(&self.buffer),
        }
    }

    /// Check if the buffer contains a partial (incomplete) tool call.
    #[must_use]
    pub fn has_partial_tool_call(&self) -> bool {
        match self.style {
            ToolCallStyle::Standard => false,
            ToolCallStyle::QwenCoder | ToolCallStyle::QwenVl | ToolCallStyle::Preserve => {
                self.buffer.contains("<tool_call>") && !self.buffer.contains("</tool_call>")
            }
            ToolCallStyle::GeneralInline => {
                self.buffer.contains("[tool_call:") && !self.buffer.contains(']')
            }
        }
    }
}

/// Strip at most one leading newline and one trailing newline from a string.
///
/// This matches the reference vLLM/sglang behavior which is deliberately conservative,
/// only removing bounding newlines rather than all whitespace.
fn strip_bounding_newlines(s: &str) -> &str {
    let s = s.strip_prefix('\n').unwrap_or(s);

    (s.strip_suffix('\n').unwrap_or(s)) as _
}

/// Infer the primary parameter name based on function name.
fn infer_param_name(func_name: &str) -> &'static str {
    match func_name.to_lowercase().as_str() {
        "bash" | "shell" | "run_shell_command" => "command",
        "read_file" | "read" => "path",
        "write_file" | "write" => "path",
        "glob" | "find" => "pattern",
        "grep" | "search" => "pattern",
        "edit" | "str_replace" => "path",
        _ => "input",
    }
}

/// Attempt to repair malformed JSON inside a tool_call tag.
fn repair_json_in_xml(input: &str) -> Option<serde_json::Value> {
    let trimmed = input.trim();

    // Try standard parse first
    if let Ok(value) = serde_json::from_str(trimmed) {
        return Some(value);
    }

    // Try adding missing closing braces
    for suffix in &["}", "}}", "\"}", "\"}}", "]}", "\"]}"] {
        let repaired = format!("{}{}", trimmed, suffix);
        if let Ok(value) = serde_json::from_str(&repaired) {
            return Some(value);
        }
    }

    // Try to extract JSON object
    if let Some(start) = trimmed.find('{') {
        let mut depth = 0;
        let mut in_string = false;
        let mut escape = false;
        let mut end_pos = None;

        for (i, ch) in trimmed[start..].char_indices() {
            if !in_string {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end_pos = Some(start + i + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if ch == '"' && !escape {
                in_string = !in_string;
            }
            escape = ch == '\\' && !escape;
        }

        if let Some(end) = end_pos
            && let Ok(value) = serde_json::from_str(&trimmed[start..end])
        {
            return Some(value);
        }
    }

    None
}

/// Parse text for tool calls using any detected format.
///
/// This is a convenience function that tries all formats.
#[must_use]
pub fn parse_any_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    // Try QwenCoder first (most specific)
    let qwen_coder = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
    let results = qwen_coder.parse(text);
    if !results.is_empty() {
        return results;
    }

    // Try QwenVL next
    let qwen_vl = XmlToolCallParser::new(ToolCallStyle::QwenVl);
    let results = qwen_vl.parse(text);
    if !results.is_empty() {
        return results;
    }

    // Try inline format
    let inline = XmlToolCallParser::new(ToolCallStyle::GeneralInline);
    inline.parse(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_detection() {
        // Qwen3-Next uses hybrid XML-JSON format (QwenVl style)
        assert_eq!(
            ToolCallStyle::from_model_name("qwen3-next"),
            ToolCallStyle::QwenVl
        );
        assert_eq!(
            ToolCallStyle::from_model_name("qwen3next"),
            ToolCallStyle::QwenVl
        );
        // Qwen-Coder models also use hybrid format
        assert_eq!(
            ToolCallStyle::from_model_name("qwen3-coder"),
            ToolCallStyle::QwenVl
        );
        assert_eq!(
            ToolCallStyle::from_model_name("qwen2.5-coder-32b"),
            ToolCallStyle::QwenVl
        );
        // Qwen-VL models use hybrid format
        assert_eq!(
            ToolCallStyle::from_model_name("qwen-vl"),
            ToolCallStyle::QwenVl
        );
        assert_eq!(
            ToolCallStyle::from_model_name("qwen2-vl-72b"),
            ToolCallStyle::QwenVl
        );
        // Non-Qwen models use standard format
        assert_eq!(
            ToolCallStyle::from_model_name("gpt-4"),
            ToolCallStyle::Standard
        );
        assert_eq!(
            ToolCallStyle::from_model_name("claude-3"),
            ToolCallStyle::Standard
        );
    }

    #[test]
    fn test_style_from_config() {
        // Default (None) should be QwenCoder (XML)
        assert_eq!(ToolCallStyle::from_config(None), ToolCallStyle::QwenCoder);
        // Explicit "xml" should be QwenCoder
        assert_eq!(
            ToolCallStyle::from_config(Some("xml")),
            ToolCallStyle::QwenCoder
        );
        // "json_in_xml" should be QwenVl
        assert_eq!(
            ToolCallStyle::from_config(Some("json_in_xml")),
            ToolCallStyle::QwenVl
        );
        // Case-insensitive
        assert_eq!(
            ToolCallStyle::from_config(Some("JSON_IN_XML")),
            ToolCallStyle::QwenVl
        );
        // Unknown defaults to QwenCoder
        assert_eq!(
            ToolCallStyle::from_config(Some("unknown")),
            ToolCallStyle::QwenCoder
        );
    }

    #[test]
    fn test_qwen_coder_single_param() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"
<tool_call>
<function=bash>
<parameter=command>
ls -la
</parameter>
</function>
</tool_call>
"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
        assert_eq!(
            results[0].arguments.get("command").and_then(|v| v.as_str()),
            Some("ls -la")
        );
    }

    #[test]
    fn test_qwen_coder_multiple_params() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"
<tool_call>
<function=read_file>
<parameter=path>
/home/user/file.txt
</parameter>
<parameter=offset>
0
</parameter>
<parameter=limit>
100
</parameter>
</function>
</tool_call>
"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "read_file");
        assert_eq!(
            results[0].arguments.get("path").and_then(|v| v.as_str()),
            Some("/home/user/file.txt")
        );
        // Numbers parsed as strings are fine, they get converted during execution
        assert!(results[0].arguments.get("offset").is_some());
        assert!(results[0].arguments.get("limit").is_some());
    }

    #[test]
    fn test_qwen_coder_multiple_calls() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"
Let me check the file first.
<tool_call>
<function=read_file>
<parameter=path>
/src/main.rs
</parameter>
</function>
</tool_call>
Now I'll run the tests.
<tool_call>
<function=bash>
<parameter=command>
cargo test
</parameter>
</function>
</tool_call>
"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].name, "read_file");
        assert_eq!(results[1].name, "bash");
    }

    #[test]
    fn test_qwen_vl_format() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenVl);
        let text = r#"
<tool_call>
{"name": "bash", "arguments": {"command": "ls -la"}}
</tool_call>
"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
        assert_eq!(
            results[0].arguments.get("command").and_then(|v| v.as_str()),
            Some("ls -la")
        );
    }

    #[test]
    fn test_qwen_vl_complex() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenVl);
        let text = r#"
<tool_call>
{"name": "edit", "arguments": {"path": "src/lib.rs", "old_content": "fn old() {}", "new_content": "fn new() {}"}}
</tool_call>
"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "edit");
        assert_eq!(
            results[0].arguments.get("path").and_then(|v| v.as_str()),
            Some("src/lib.rs")
        );
    }

    #[test]
    fn test_inline_format() {
        let parser = XmlToolCallParser::new(ToolCallStyle::GeneralInline);
        let text = "[tool_call: bash for 'ls -la']";

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
        assert_eq!(
            results[0].arguments.get("command").and_then(|v| v.as_str()),
            Some("ls -la")
        );
    }

    #[test]
    fn test_inline_format_double_quotes() {
        let parser = XmlToolCallParser::new(ToolCallStyle::GeneralInline);
        let text = r#"[tool_call: read_file for "/path/to/file.txt"]"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "read_file");
        assert_eq!(
            results[0].arguments.get("path").and_then(|v| v.as_str()),
            Some("/path/to/file.txt")
        );
    }

    #[test]
    fn test_parse_any() {
        // Hybrid XML-JSON format (primary format for Qwen3-Next)
        let text1 = r#"<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>"#;
        let results1 = parse_any_tool_calls(text1);
        assert_eq!(results1.len(), 1);
        assert_eq!(results1[0].name, "bash");

        // Legacy Qwen-Coder pure XML format (still supported for backwards compatibility)
        let text2 =
            "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>";
        let results2 = parse_any_tool_calls(text2);
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].name, "bash");

        // Inline format
        let text3 = "[tool_call: bash for 'ls']";
        let results3 = parse_any_tool_calls(text3);
        assert_eq!(results3.len(), 1);
        assert_eq!(results3[0].name, "bash");
    }

    #[test]
    fn test_buffer_operations() {
        let mut parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);

        // Simulate streaming
        parser.add_content("Some text before\n");
        parser.add_content("<tool_call>\n");
        assert!(parser.has_partial_tool_call());
        assert!(!parser.has_complete_tool_calls());

        parser.add_content("<function=bash>\n");
        parser.add_content("<parameter=command>\n");
        parser.add_content("ls -la\n");
        parser.add_content("</parameter>\n");
        parser.add_content("</function>\n");
        parser.add_content("</tool_call>\n");
        assert!(parser.has_complete_tool_calls());

        parser.add_content("Some text after");

        let tool_calls = parser.extract_tool_calls();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "bash");

        // Buffer should only contain the non-tool-call text
        assert!(parser.buffer().contains("Some text before"));
        assert!(parser.buffer().contains("Some text after"));
        assert!(!parser.buffer().contains("<tool_call>"));
    }

    #[test]
    fn test_to_tool_calls_conversion() {
        let mut parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        parser.add_content(
            "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>",
        );

        let parsed = parser.extract_tool_calls();
        let tool_calls = parser.to_tool_calls(parsed);

        assert_eq!(tool_calls.len(), 1);
        assert!(tool_calls[0].id.is_some());
        assert_eq!(tool_calls[0].function.name, Some("bash".to_string()));
        assert!(tool_calls[0].function.arguments.is_some());
    }

    #[test]
    fn test_json_repair_in_xml() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenVl);

        // Missing closing brace
        let text = r#"<tool_call>{"name": "bash", "arguments": {"command": "ls"}</tool_call>"#;
        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
    }

    #[test]
    fn test_multiline_parameter_value() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"
<tool_call>
<function=write_file>
<parameter=path>
/src/main.rs
</parameter>
<parameter=content>
fn main() {
    println!("Hello, world!");
}
</parameter>
</function>
</tool_call>
"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "write_file");

        let content = results[0]
            .arguments
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(content.contains("fn main()"));
        assert!(content.contains("println!"));
    }

    #[test]
    fn test_param_without_closing_tag() {
        // Parameter terminated by next <parameter= instead of </parameter>
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"<tool_call>
<function=edit>
<parameter=path>/src/main.rs
<parameter=old_content>old code
<parameter=new_content>new code
</function>
</tool_call>"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "edit");
        assert_eq!(
            results[0].arguments.get("path").and_then(|v| v.as_str()),
            Some("/src/main.rs")
        );
        assert_eq!(
            results[0]
                .arguments
                .get("old_content")
                .and_then(|v| v.as_str()),
            Some("old code")
        );
        assert_eq!(
            results[0]
                .arguments
                .get("new_content")
                .and_then(|v| v.as_str()),
            Some("new code")
        );
    }

    #[test]
    fn test_param_terminated_by_function_close() {
        // Last parameter terminated by </function> instead of </parameter>
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"<tool_call>
<function=bash>
<parameter=command>ls -la
</function>
</tool_call>"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
        assert_eq!(
            results[0].arguments.get("command").and_then(|v| v.as_str()),
            Some("ls -la")
        );
    }

    #[test]
    fn test_incomplete_tool_call_block() {
        // No </tool_call> closing tag
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"<tool_call>
<function=bash>
<parameter=command>ls -la</parameter>
</function>"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
        assert_eq!(
            results[0].arguments.get("command").and_then(|v| v.as_str()),
            Some("ls -la")
        );
    }

    #[test]
    fn test_incomplete_function_block() {
        // No </function> closing tag
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"<tool_call>
<function=bash>
<parameter=command>ls -la</parameter>"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
        assert_eq!(
            results[0].arguments.get("command").and_then(|v| v.as_str()),
            Some("ls -la")
        );
    }

    #[test]
    fn test_whitespace_preservation() {
        // Only single leading/trailing newlines stripped, not all whitespace
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = "<tool_call>\n<function=write_file>\n<parameter=content>\n  indented line\n  another line  \n</parameter>\n</function>\n</tool_call>";

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        let content = results[0]
            .arguments
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap();
        // Leading spaces should be preserved
        assert!(content.starts_with("  indented line"));
        // Trailing spaces should be preserved
        assert!(content.ends_with("another line  "));
    }

    #[test]
    fn test_incomplete_qwen_vl_block() {
        // No </tool_call> closing tag for JSON-in-XML format
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenVl);
        let text = r#"<tool_call>
{"name": "bash", "arguments": {"command": "ls -la"}}"#;

        let results = parser.parse(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
        assert_eq!(
            results[0].arguments.get("command").and_then(|v| v.as_str()),
            Some("ls -la")
        );
    }

    #[test]
    fn test_parse_any_xml_mixed_formats() {
        // A response containing both JSON-in-XML and pure XML tool calls
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"Let me read the file first.
<tool_call>
{"name": "read_file", "arguments": {"path": "/src/main.rs"}}
</tool_call>
Now I'll run the build.
<tool_call>
<function=bash>
<parameter=command>cargo build</parameter>
</function>
</tool_call>"#;

        let results = parser.parse_any_xml(text);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].name, "read_file");
        assert_eq!(
            results[0].arguments.get("path").and_then(|v| v.as_str()),
            Some("/src/main.rs")
        );
        assert_eq!(results[1].name, "bash");
        assert_eq!(
            results[1].arguments.get("command").and_then(|v| v.as_str()),
            Some("cargo build")
        );
    }

    #[test]
    fn test_parse_any_xml_json_only() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        let text = r#"<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>"#;
        let results = parser.parse_any_xml(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
    }

    #[test]
    fn test_parse_any_xml_pure_xml_only() {
        let parser = XmlToolCallParser::new(ToolCallStyle::QwenVl);
        let text =
            "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>";
        let results = parser.parse_any_xml(text);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "bash");
    }

    #[test]
    fn test_parse_buffer_accepts_both_formats() {
        // Verify that parse_buffer (used in streaming) also accepts both formats
        let mut parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
        parser.add_content(
            r#"<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>"#,
        );
        let calls = parser.extract_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");

        // Now add pure XML format
        parser.add_content(
            "<tool_call><function=grep><parameter=pattern>TODO</parameter></function></tool_call>",
        );
        let calls = parser.extract_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
    }

    #[test]
    fn test_strip_bounding_newlines_helper() {
        assert_eq!(strip_bounding_newlines("\nhello\n"), "hello");
        assert_eq!(strip_bounding_newlines("\nhello"), "hello");
        assert_eq!(strip_bounding_newlines("hello\n"), "hello");
        assert_eq!(strip_bounding_newlines("hello"), "hello");
        // Only strips ONE newline from each end
        assert_eq!(strip_bounding_newlines("\n\nhello\n\n"), "\nhello\n");
        // Preserves other whitespace
        assert_eq!(strip_bounding_newlines("\n  hello  \n"), "  hello  ");
        assert_eq!(strip_bounding_newlines("  hello  "), "  hello  ");
    }
}
