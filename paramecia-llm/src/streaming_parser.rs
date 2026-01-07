//! Streaming tool call parser with JSON repair capabilities.
//!
//! This module handles the complexities of parsing tool calls from streaming LLM responses:
//! - Tool calls arrive with varying chunk shapes (empty strings, partial JSON, complete objects)
//! - Tool calls may lack IDs, names, or have inconsistent indices
//! - Multiple tool calls can be processed simultaneously with interleaved chunks
//! - Index collisions occur when the same index is reused for different tool calls
//! - JSON arguments are fragmented across multiple chunks and need reconstruction

use crate::types::{FunctionCall, ToolCall};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Result of parsing a JSON chunk in tool calls.
#[derive(Debug, Clone)]
pub struct ToolCallParseResult {
    /// Whether the JSON parsing is complete.
    pub complete: bool,
    /// The parsed JSON value (only present when complete is true).
    pub value: Option<serde_json::Value>,
    /// Error information if parsing failed.
    pub error: Option<String>,
    /// Whether the JSON was repaired (e.g., auto-closed unclosed strings).
    pub repaired: bool,
}

impl Default for ToolCallParseResult {
    fn default() -> Self {
        Self {
            complete: false,
            value: None,
            error: None,
            repaired: false,
        }
    }
}

impl ToolCallParseResult {
    /// Create an incomplete result (still accumulating chunks).
    #[must_use]
    pub fn incomplete() -> Self {
        Self::default()
    }

    /// Create a complete result with parsed value.
    #[must_use]
    pub fn complete(value: serde_json::Value) -> Self {
        Self {
            complete: true,
            value: Some(value),
            error: None,
            repaired: false,
        }
    }

    /// Create a complete result with repaired JSON.
    #[must_use]
    pub fn complete_repaired(value: serde_json::Value) -> Self {
        Self {
            complete: true,
            value: Some(value),
            error: None,
            repaired: true,
        }
    }

    /// Create a result with an error.
    #[must_use]
    pub fn with_error(error: impl Into<String>) -> Self {
        Self {
            complete: false,
            value: None,
            error: Some(error.into()),
            repaired: false,
        }
    }
}

/// Metadata for a tool call being parsed.
#[derive(Debug, Clone, Default)]
pub struct ToolCallMeta {
    /// Tool call ID.
    pub id: Option<String>,
    /// Function name.
    pub name: Option<String>,
}

/// State for tracking JSON parsing progress.
#[derive(Debug, Clone, Default)]
struct JsonParseState {
    /// Current nesting depth in JSON structure.
    depth: i32,
    /// Whether we're currently inside a string literal.
    in_string: bool,
    /// Whether the next character should be treated as escaped.
    escape: bool,
}

/// Streaming tool call parser that handles inconsistent chunk formats.
///
/// # Problems this parser addresses
///
/// - Tool calls arrive with varying chunk shapes (empty strings, partial JSON, complete objects)
/// - Tool calls may lack IDs, names, or have inconsistent indices
/// - Multiple tool calls can be processed simultaneously with interleaved chunks
/// - Index collisions occur when the same index is reused for different tool calls
/// - JSON arguments are fragmented across multiple chunks and need reconstruction
///
/// # Example
///
/// ```
/// use paramecia_llm::streaming_parser::StreamingToolCallParser;
///
/// let mut parser = StreamingToolCallParser::new();
///
/// // First chunk with ID and name
/// let result = parser.add_chunk(0, r#"{"com"#, Some("call_1"), Some("bash"));
/// assert!(!result.complete);
///
/// // Continuation chunk
/// let result = parser.add_chunk(0, r#"mand": "ls"}"#, None, None);
/// assert!(result.complete);
/// ```
#[derive(Debug, Default)]
pub struct StreamingToolCallParser {
    /// Accumulated buffer containing all received chunks for each tool call index.
    buffers: HashMap<u32, String>,
    /// JSON parsing state for each tool call index.
    parse_states: HashMap<u32, JsonParseState>,
    /// Metadata for each tool call index.
    tool_call_meta: HashMap<u32, ToolCallMeta>,
    /// Map from tool call ID to actual index used for storage.
    id_to_index_map: HashMap<String, u32>,
    /// Counter for generating new indices when collisions occur.
    next_available_index: u32,
}

impl StreamingToolCallParser {
    /// Create a new streaming tool call parser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Processes a new chunk of tool call data and attempts to parse complete JSON objects.
    ///
    /// Handles the core problems of streaming tool call parsing:
    /// - Resolves index collisions when the same index is reused for different tool calls
    /// - Routes chunks without IDs to the correct incomplete tool call
    /// - Tracks JSON parsing state (depth, string boundaries, escapes) per tool call
    /// - Attempts parsing only when JSON structure is complete (depth = 0)
    /// - Repairs common issues like unclosed strings
    ///
    /// # Arguments
    ///
    /// * `index` - Tool call index from streaming response (may collide with existing calls)
    /// * `chunk` - String chunk that may be empty, partial JSON, or complete data
    /// * `id` - Optional tool call ID for collision detection and chunk routing
    /// * `name` - Optional function name stored as metadata
    ///
    /// # Returns
    ///
    /// `ToolCallParseResult` with completion status, parsed value, and repair info
    pub fn add_chunk(
        &mut self,
        index: u32,
        chunk: &str,
        id: Option<&str>,
        name: Option<&str>,
    ) -> ToolCallParseResult {
        let actual_index = self.resolve_index(index, id);

        // Initialize state for the actual index if not exists
        self.ensure_index_initialized(actual_index);

        // Update metadata
        if let Some(meta) = self.tool_call_meta.get_mut(&actual_index) {
            if let Some(id) = id {
                meta.id = Some(id.to_string());
            }
            if let Some(name) = name {
                meta.name = Some(name.to_string());
            }
        }

        // Add chunk to buffer
        if let Some(buffer) = self.buffers.get_mut(&actual_index) {
            buffer.push_str(chunk);
        }

        // Update JSON parsing state
        self.update_parse_state(actual_index, chunk);

        // Attempt to parse if we might be complete
        self.try_parse(actual_index)
    }

    /// Resolve the actual index to use, handling collisions.
    fn resolve_index(&mut self, index: u32, id: Option<&str>) -> u32 {
        if let Some(id) = id {
            // This is a tool call with an ID
            if let Some(&existing_index) = self.id_to_index_map.get(id) {
                // We've seen this ID before, use the existing mapped index
                return existing_index;
            }

            // New tool call ID - check for collision
            let actual_index = if self.is_index_occupied_by_different_call(index, id) {
                self.find_next_available_index()
            } else {
                index
            };

            // Map this ID to the actual index
            self.id_to_index_map.insert(id.to_string(), actual_index);
            actual_index
        } else {
            // No ID provided - this is a continuation chunk
            self.find_index_for_continuation(index)
        }
    }

    /// Check if an index is occupied by a different completed tool call.
    fn is_index_occupied_by_different_call(&self, index: u32, new_id: &str) -> bool {
        if let Some(buffer) = self.buffers.get(&index) {
            let state = self.parse_states.get(&index);
            let meta = self.tool_call_meta.get(&index);

            // Check if we have a complete tool call with a different ID
            if !buffer.trim().is_empty()
                && state.map_or(false, |s| s.depth == 0)
                && meta.is_some_and(|m| m.id.as_deref().is_some_and(|id| id != new_id))
            {
                // Try to parse to confirm it's complete
                return serde_json::from_str::<serde_json::Value>(buffer).is_ok();
            }
        }
        false
    }

    /// Find an index for a continuation chunk (no ID provided).
    fn find_index_for_continuation(&mut self, index: u32) -> u32 {
        if let Some(buffer) = self.buffers.get(&index) {
            let state = self.parse_states.get(&index);

            // If there's an incomplete tool call at this index, continue with it
            if state.map_or(true, |s| s.depth > 0) || buffer.trim().is_empty() {
                return index;
            }

            // Check if the buffer at this index is complete
            if serde_json::from_str::<serde_json::Value>(buffer).is_ok() {
                // Buffer is complete, find the most recent incomplete tool call
                return self.find_most_recent_incomplete_index();
            }
        }

        index
    }

    /// Find the most recent incomplete tool call index.
    fn find_most_recent_incomplete_index(&self) -> u32 {
        let mut max_index: Option<u32> = None;

        for (&index, buffer) in &self.buffers {
            let state = self.parse_states.get(&index);
            let meta = self.tool_call_meta.get(&index);

            // Check if this tool call is incomplete
            let is_incomplete = if meta.is_some_and(|m| m.id.is_some()) {
                state.map_or(true, |s| s.depth > 0) || buffer.trim().is_empty()
            } else if !buffer.trim().is_empty() {
                serde_json::from_str::<serde_json::Value>(buffer).is_err()
            } else {
                false
            };

            if is_incomplete {
                max_index = Some(max_index.map_or(index, |m| m.max(index)));
            }
        }

        max_index.unwrap_or_else(|| self.next_available_index)
    }

    /// Find the next available index for a new tool call.
    fn find_next_available_index(&mut self) -> u32 {
        loop {
            let index = self.next_available_index;

            if let Some(buffer) = self.buffers.get(&index) {
                let state = self.parse_states.get(&index);
                let meta = self.tool_call_meta.get(&index);

                // If buffer is empty or incomplete, this index is available
                if buffer.trim().is_empty()
                    || state.map_or(true, |s| s.depth > 0)
                    || meta.map_or(true, |m| m.id.is_none())
                {
                    return index;
                }

                // Try to parse - if it fails, index is available
                if serde_json::from_str::<serde_json::Value>(buffer).is_err() {
                    return index;
                }

                // Index has complete tool call, try next
                self.next_available_index += 1;
            } else {
                // Index not in use
                self.next_available_index += 1;
                return index;
            }
        }
    }

    /// Ensure an index has initialized state.
    fn ensure_index_initialized(&mut self, index: u32) {
        self.buffers.entry(index).or_default();
        self.parse_states.entry(index).or_default();
        self.tool_call_meta.entry(index).or_default();
    }

    /// Update JSON parsing state for a chunk.
    fn update_parse_state(&mut self, index: u32, chunk: &str) {
        let state = self.parse_states.get_mut(&index).unwrap();

        for char in chunk.chars() {
            if !state.in_string {
                match char {
                    '{' | '[' => state.depth += 1,
                    '}' | ']' => state.depth -= 1,
                    _ => {}
                }
            }

            // Track string boundaries - toggle in_string state on unescaped quotes
            if char == '"' && !state.escape {
                state.in_string = !state.in_string;
            }

            // Track escape sequences - backslash followed by any character is escaped
            state.escape = char == '\\' && !state.escape;
        }
    }

    /// Try to parse the buffer at an index.
    fn try_parse(&self, index: u32) -> ToolCallParseResult {
        let buffer = match self.buffers.get(&index) {
            Some(b) => b,
            None => return ToolCallParseResult::incomplete(),
        };

        let state = match self.parse_states.get(&index) {
            Some(s) => s,
            None => return ToolCallParseResult::incomplete(),
        };

        // Only attempt parse when at root level (depth 0) and have data
        if state.depth != 0 || buffer.trim().is_empty() {
            return ToolCallParseResult::incomplete();
        }

        // Standard JSON parsing attempt
        match serde_json::from_str::<serde_json::Value>(buffer) {
            Ok(value) => ToolCallParseResult::complete(value),
            Err(e) => {
                // Try repair strategies
                if let Some(repaired) = self.try_repair_json(buffer, state) {
                    return repaired;
                }
                ToolCallParseResult::with_error(e.to_string())
            }
        }
    }

    /// Attempt to repair malformed JSON.
    fn try_repair_json(&self, buffer: &str, state: &JsonParseState) -> Option<ToolCallParseResult> {
        // Strategy 1: Auto-close unclosed strings
        if state.in_string {
            // Try closing just the string
            let repaired = format!("{}\"", buffer);
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&repaired) {
                return Some(ToolCallParseResult::complete_repaired(value));
            }

            // Try closing string + closing brace (common case: {"key": "value)
            let repaired = format!("{}\"}}", buffer);
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&repaired) {
                return Some(ToolCallParseResult::complete_repaired(value));
            }

            // Try closing string + closing array bracket
            let repaired = format!("{}\"]}}", buffer);
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&repaired) {
                return Some(ToolCallParseResult::complete_repaired(value));
            }
        }

        // Strategy 2: Try adding missing closing braces
        for suffix in &["}", "}}", "]}}", "]}", "\"}", "\"}}", "\"]}"] {
            let repaired = format!("{}{}", buffer, suffix);
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&repaired) {
                return Some(ToolCallParseResult::complete_repaired(value));
            }
        }

        // Strategy 3: Safe JSON parse with recovery
        if let Some(value) = safe_json_parse(buffer) {
            return Some(ToolCallParseResult::complete_repaired(value));
        }

        None
    }

    /// Gets the current tool call metadata for a specific index.
    #[must_use]
    pub fn get_tool_call_meta(&self, index: u32) -> Option<&ToolCallMeta> {
        self.tool_call_meta.get(&index)
    }

    /// Gets all completed tool calls that are ready to be emitted.
    ///
    /// Attempts to parse accumulated buffers using multiple strategies:
    /// 1. Standard JSON parse
    /// 2. Auto-close unclosed strings and retry
    /// 3. Fallback to safe JSON parse for malformed data
    ///
    /// Should be called when streaming is complete (finish_reason is present).
    #[must_use]
    pub fn get_completed_tool_calls(&self) -> Vec<CompletedToolCall> {
        let mut completed = Vec::new();

        for (&index, buffer) in &self.buffers {
            let meta = match self.tool_call_meta.get(&index) {
                Some(m) if m.name.is_some() && !buffer.trim().is_empty() => m,
                _ => continue,
            };

            let args = self.parse_buffer_with_fallback(buffer, index);

            completed.push(CompletedToolCall {
                id: meta.id.clone(),
                name: meta.name.clone(),
                args,
                index,
            });
        }

        // Sort by index for consistent ordering
        completed.sort_by_key(|c| c.index);
        completed
    }

    /// Parse a buffer with multiple fallback strategies.
    fn parse_buffer_with_fallback(&self, buffer: &str, index: u32) -> serde_json::Value {
        // Try standard parse
        if let Ok(value) = serde_json::from_str(buffer) {
            return value;
        }

        let state = self.parse_states.get(&index);

        // Strategy 1: Auto-close unclosed strings
        if state.is_some_and(|s| s.in_string) {
            // Try closing just the string
            let repaired = format!("{}\"", buffer);
            if let Ok(value) = serde_json::from_str(&repaired) {
                return value;
            }

            // Try closing string + closing brace
            let repaired = format!("{}\"}}", buffer);
            if let Ok(value) = serde_json::from_str(&repaired) {
                return value;
            }
        }

        // Strategy 2: Try adding missing closing braces/brackets
        for suffix in &["}", "}}", "]}}", "]}", "\"}", "\"}}", "\"]}"] {
            let repaired = format!("{}{}", buffer, suffix);
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&repaired) {
                return value;
            }
        }

        // Final fallback: safe JSON parse
        safe_json_parse(buffer).unwrap_or_else(|| serde_json::json!({}))
    }

    /// Convert completed tool calls to the ToolCall type.
    #[must_use]
    pub fn into_tool_calls(&self) -> Vec<ToolCall> {
        self.get_completed_tool_calls()
            .into_iter()
            .map(|c| ToolCall {
                id: c.id,
                index: Some(c.index as usize),
                function: FunctionCall {
                    name: c.name,
                    arguments: Some(c.args.to_string()),
                },
                r#type: "function".to_string(),
            })
            .collect()
    }

    /// Gets the current accumulated buffer content for a specific index.
    #[must_use]
    pub fn get_buffer(&self, index: u32) -> Option<&str> {
        self.buffers.get(&index).map(String::as_str)
    }

    /// Gets the current parsing state for a specific index.
    #[must_use]
    pub fn get_parse_state(&self, index: u32) -> Option<(i32, bool, bool)> {
        self.parse_states
            .get(&index)
            .map(|s| (s.depth, s.in_string, s.escape))
    }

    /// Resets the parser state for a specific tool call index.
    pub fn reset_index(&mut self, index: u32) {
        self.buffers.remove(&index);
        self.parse_states.remove(&index);
        self.tool_call_meta.remove(&index);
    }

    /// Resets the entire parser state for processing a new stream.
    pub fn reset(&mut self) {
        self.buffers.clear();
        self.parse_states.clear();
        self.tool_call_meta.clear();
        self.id_to_index_map.clear();
        self.next_available_index = 0;
    }

    /// Check if there are any incomplete tool calls.
    #[must_use]
    pub fn has_incomplete_calls(&self) -> bool {
        self.parse_states.values().any(|s| s.depth > 0)
    }

    /// Get a hash of a tool call for deduplication/loop detection.
    #[must_use]
    pub fn hash_tool_call(name: &str, args: &serde_json::Value) -> String {
        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        hasher.update(args.to_string().as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// A completed tool call ready for execution.
#[derive(Debug, Clone)]
pub struct CompletedToolCall {
    /// Tool call ID.
    pub id: Option<String>,
    /// Function name.
    pub name: Option<String>,
    /// Parsed arguments.
    pub args: serde_json::Value,
    /// Original index.
    pub index: u32,
}

/// Safely parse JSON with recovery from common errors.
///
/// Attempts multiple strategies to parse malformed JSON:
/// 1. Standard parse
/// 2. Trim whitespace and retry
/// 3. Fix common issues (trailing commas, missing quotes)
/// 4. Extract valid JSON object/array from surrounding garbage
fn safe_json_parse(input: &str) -> Option<serde_json::Value> {
    let trimmed = input.trim();

    // Try standard parse first
    if let Ok(value) = serde_json::from_str(trimmed) {
        return Some(value);
    }

    // Try to extract a JSON object
    if let Some(start) = trimmed.find('{') {
        // Find matching closing brace
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

        if let Some(end) = end_pos {
            let extracted = &trimmed[start..end];
            if let Ok(value) = serde_json::from_str(extracted) {
                return Some(value);
            }

            // Try fixing trailing comma
            let fixed = fix_trailing_commas(extracted);
            if let Ok(value) = serde_json::from_str(&fixed) {
                return Some(value);
            }
        }
    }

    // Last resort: try to build an empty object
    None
}

/// Fix trailing commas in JSON (common LLM error).
fn fix_trailing_commas(input: &str) -> String {
    // Simple regex-like replacement for trailing commas before } or ]
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == ',' {
            // Look ahead for } or ] (skipping whitespace)
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                // Skip this comma
                i += 1;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

/// Merge streaming tool call chunks into complete tool calls.
///
/// This is a convenience function that uses `StreamingToolCallParser` internally.
#[must_use]
pub fn merge_tool_call_chunks(chunks: &[ToolCall]) -> Vec<ToolCall> {
    let mut parser = StreamingToolCallParser::new();

    for chunk in chunks {
        let index = chunk.index.unwrap_or(0) as u32;
        let args = chunk.function.arguments.as_deref().unwrap_or("");
        let id = chunk.id.as_deref();
        let name = chunk.function.name.as_deref();

        parser.add_chunk(index, args, id, name);
    }

    parser.into_tool_calls()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_complete_json() {
        let mut parser = StreamingToolCallParser::new();

        let result = parser.add_chunk(0, r#"{"command": "ls -la"}"#, Some("call_1"), Some("bash"));

        assert!(result.complete);
        assert!(!result.repaired);
        assert_eq!(
            result.value.unwrap(),
            serde_json::json!({"command": "ls -la"})
        );
    }

    #[test]
    fn test_streaming_chunks() {
        let mut parser = StreamingToolCallParser::new();

        // First chunk with ID and name
        let result = parser.add_chunk(0, r#"{"com"#, Some("call_1"), Some("bash"));
        assert!(!result.complete);

        // Continuation chunks
        let result = parser.add_chunk(0, r#"mand":"#, None, None);
        assert!(!result.complete);

        let result = parser.add_chunk(0, r#" "ls"}"#, None, None);
        assert!(result.complete);
        assert_eq!(result.value.unwrap(), serde_json::json!({"command": "ls"}));
    }

    #[test]
    fn test_index_collision() {
        let mut parser = StreamingToolCallParser::new();

        // First tool call at index 0
        let result = parser.add_chunk(0, r#"{"command": "ls"}"#, Some("call_1"), Some("bash"));
        assert!(result.complete);

        // Second tool call also at index 0 but different ID
        let result = parser.add_chunk(0, r#"{"path": "/home"}"#, Some("call_2"), Some("read_file"));
        assert!(result.complete);

        // Should have 2 completed tool calls
        let completed = parser.get_completed_tool_calls();
        assert_eq!(completed.len(), 2);
    }

    #[test]
    fn test_unclosed_string_repair() {
        let mut parser = StreamingToolCallParser::new();

        // Add a chunk with unclosed string - this simulates a truncated response
        // where the LLM didn't finish the JSON properly
        parser.add_chunk(0, r#"{"command": "ls"#, Some("call_1"), Some("bash"));

        // Simulate the streaming being complete (depth forced to 0)
        // In real streaming, this would happen when finish_reason is received
        // but the JSON is malformed
        if let Some(state) = parser.parse_states.get_mut(&0) {
            state.depth = 0;
            state.in_string = true;
        }

        // Now when we get completed tool calls, it should repair the JSON
        // by auto-closing the string and adding the missing brace
        let completed = parser.get_completed_tool_calls();
        assert_eq!(completed.len(), 1);

        let args = &completed[0].args;
        // The repair should auto-close the string and add closing brace
        assert_eq!(args.get("command").and_then(|v| v.as_str()), Some("ls"));
    }

    #[test]
    fn test_unclosed_brace_repair() {
        let mut parser = StreamingToolCallParser::new();

        // Add a chunk with missing closing brace (string is closed properly)
        parser.add_chunk(0, r#"{"command": "ls""#, Some("call_1"), Some("bash"));

        // Force depth to 0 to trigger parse attempt
        if let Some(state) = parser.parse_states.get_mut(&0) {
            state.depth = 0;
        }

        let completed = parser.get_completed_tool_calls();
        assert_eq!(completed.len(), 1);

        let args = &completed[0].args;
        assert_eq!(args.get("command").and_then(|v| v.as_str()), Some("ls"));
    }

    #[test]
    fn test_repair_via_safe_parse() {
        // Test the safe_json_parse function directly for edge cases
        let result = super::safe_json_parse(r#"{"key": "value"}"#);
        assert!(result.is_some());

        // Trailing comma should be fixed
        let result = super::safe_json_parse(r#"{"key": "value",}"#);
        assert!(result.is_some());

        // Garbage prefix should be handled
        let result = super::safe_json_parse(r#"prefix garbage {"key": "value"}"#);
        assert!(result.is_some());
        assert_eq!(
            result.unwrap().get("key").and_then(|v| v.as_str()),
            Some("value")
        );
    }

    #[test]
    fn test_safe_json_parse() {
        // Valid JSON
        assert!(safe_json_parse(r#"{"key": "value"}"#).is_some());

        // JSON with garbage prefix
        assert!(safe_json_parse(r#"garbage{"key": "value"}"#).is_some());

        // JSON with trailing comma
        let result = safe_json_parse(r#"{"key": "value",}"#);
        assert!(result.is_some());

        // Empty string
        assert!(safe_json_parse("").is_none());
    }

    #[test]
    fn test_merge_tool_call_chunks() {
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

        let merged = merge_tool_call_chunks(&chunks);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].id, Some("call_1".to_string()));
        assert_eq!(merged[0].function.name, Some("bash".to_string()));

        let args: serde_json::Value =
            serde_json::from_str(merged[0].function.arguments.as_ref().unwrap()).unwrap();
        assert_eq!(args.get("command").and_then(|v| v.as_str()), Some("ls"));
    }

    #[test]
    fn test_parallel_tool_calls() {
        let mut parser = StreamingToolCallParser::new();

        // Interleaved chunks for two tool calls
        parser.add_chunk(0, r#"{"command":"#, Some("call_1"), Some("bash"));
        parser.add_chunk(1, r#"{"path":"#, Some("call_2"), Some("read_file"));
        parser.add_chunk(0, r#" "ls"}"#, None, None);
        parser.add_chunk(1, r#" "/home"}"#, None, None);

        let completed = parser.get_completed_tool_calls();
        assert_eq!(completed.len(), 2);

        // Verify correct routing
        let call_1 = completed.iter().find(|c| c.id.as_deref() == Some("call_1"));
        let call_2 = completed.iter().find(|c| c.id.as_deref() == Some("call_2"));

        assert!(call_1.is_some());
        assert!(call_2.is_some());

        assert_eq!(
            call_1.unwrap().args.get("command").and_then(|v| v.as_str()),
            Some("ls")
        );
        assert_eq!(
            call_2.unwrap().args.get("path").and_then(|v| v.as_str()),
            Some("/home")
        );
    }

    #[test]
    fn test_hash_tool_call() {
        let hash1 =
            StreamingToolCallParser::hash_tool_call("bash", &serde_json::json!({"command": "ls"}));
        let hash2 =
            StreamingToolCallParser::hash_tool_call("bash", &serde_json::json!({"command": "ls"}));
        let hash3 =
            StreamingToolCallParser::hash_tool_call("bash", &serde_json::json!({"command": "pwd"}));

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_reset() {
        let mut parser = StreamingToolCallParser::new();

        parser.add_chunk(0, r#"{"command": "ls"}"#, Some("call_1"), Some("bash"));

        parser.reset();

        assert!(parser.buffers.is_empty());
        assert!(parser.parse_states.is_empty());
        assert!(parser.tool_call_meta.is_empty());
        assert!(parser.id_to_index_map.is_empty());
    }

    #[test]
    fn test_nested_json() {
        let mut parser = StreamingToolCallParser::new();

        let nested = r#"{"config": {"nested": {"deep": "value"}, "array": [1, 2, 3]}}"#;
        let result = parser.add_chunk(0, nested, Some("call_1"), Some("complex_tool"));

        assert!(result.complete);
        let value = result.value.unwrap();
        assert_eq!(
            value
                .get("config")
                .and_then(|c| c.get("nested"))
                .and_then(|n| n.get("deep"))
                .and_then(|d| d.as_str()),
            Some("value")
        );
    }

    #[test]
    fn test_empty_object() {
        let mut parser = StreamingToolCallParser::new();

        let result = parser.add_chunk(0, r#"{}"#, Some("call_1"), Some("no_args"));

        assert!(result.complete);
        assert_eq!(result.value.unwrap(), serde_json::json!({}));
    }

    #[test]
    fn test_strings_with_special_chars() {
        let mut parser = StreamingToolCallParser::new();

        let json = r#"{"command": "echo \"hello\\nworld\""}"#;
        let result = parser.add_chunk(0, json, Some("call_1"), Some("bash"));

        assert!(result.complete);
        let value = result.value.unwrap();
        assert_eq!(
            value.get("command").and_then(|v| v.as_str()),
            Some(r#"echo "hello\nworld""#)
        );
    }
}
