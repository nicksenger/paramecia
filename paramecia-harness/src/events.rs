//! Agent events for streaming responses.

use paramecia_tools::types::ToolInfo;
use serde::{Deserialize, Serialize};

/// Events emitted by the agent during processing.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Assistant is generating text.
    Assistant(AssistantEvent),
    /// Assistant is calling a tool.
    ToolCall(ToolCallEvent),
    /// Tool execution completed.
    ToolResult(ToolResultEvent),
    /// Compaction started.
    CompactStart(CompactStartEvent),
    /// Compaction completed.
    CompactEnd(CompactEndEvent),
}

/// Event for assistant text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantEvent {
    /// The generated content.
    pub content: String,
    /// Whether the generation was stopped by middleware.
    #[serde(default)]
    pub stopped_by_middleware: bool,
}

/// Event for a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallEvent {
    /// Name of the tool being called.
    pub tool_name: String,
    /// Tool information.
    pub tool_info: Option<ToolInfo>,
    /// Arguments passed to the tool.
    pub args: serde_json::Value,
    /// Unique ID for this tool call.
    pub tool_call_id: String,
}

/// Event for a tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultEvent {
    /// Name of the tool that was called.
    pub tool_name: String,
    /// Result from the tool (if successful).
    pub result: Option<serde_json::Value>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Whether the tool was skipped.
    #[serde(default)]
    pub skipped: bool,
    /// Reason for skipping.
    pub skip_reason: Option<String>,
    /// Execution duration in seconds.
    pub duration: Option<f64>,
    /// Unique ID for this tool call.
    pub tool_call_id: String,
}

/// Event for compaction start.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactStartEvent {
    /// Current context tokens.
    pub current_context_tokens: u32,
    /// Threshold that triggered compaction.
    pub threshold: u32,
}

/// Event for compaction end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactEndEvent {
    /// Old context tokens before compaction.
    pub old_context_tokens: u32,
    /// New context tokens after compaction.
    pub new_context_tokens: u32,
    /// Length of the summary.
    pub summary_length: usize,
}
