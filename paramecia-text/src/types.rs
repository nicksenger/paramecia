//! Core types for LLM communication.

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::LlmResult;
use crate::tuning::LogitEntry;

/// Message role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message providing instructions to the model.
    System,
    /// User message from the human.
    User,
    /// Assistant message from the model.
    Assistant,
    /// Tool result message.
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

/// A function call request from the model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call.
    #[serde(default)]
    pub name: Option<String>,
    /// JSON-encoded arguments for the function.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// A tool call request from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    #[serde(default)]
    pub id: Option<String>,
    /// Index of this tool call in a batch (for streaming).
    #[serde(default)]
    pub index: Option<usize>,
    /// The function to call.
    #[serde(default)]
    pub function: FunctionCall,
    /// Type of tool (always "function" currently).
    #[serde(default = "default_function_type")]
    pub r#type: String,
    /// Original raw text of the tool call (for "preserve" mode). Not serialized to JSON.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub raw_text: Option<String>,
}

fn default_function_type() -> String {
    "function".to_string()
}

/// A message in the LLM conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    /// Role of the message sender.
    pub role: Role,
    /// Content of the message.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls requested by the assistant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Name of the tool (for tool responses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// ID of the tool call this message responds to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl LlmMessage {
    /// Create a new system message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a new user message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a new assistant message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a new assistant message with tool calls.
    #[must_use]
    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            tool_calls: Some(tool_calls),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a new tool response message.
    #[must_use]
    pub fn tool(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            name: Some(name.into()),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmUsage {
    /// Number of tokens in the prompt.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Number of tokens in the completion.
    #[serde(default)]
    pub completion_tokens: u32,
}

impl LlmUsage {
    /// Total tokens used.
    #[must_use]
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// A chunk of LLM response (used for streaming and non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmChunk {
    /// The message content.
    pub message: LlmMessage,
    /// Reason for finishing (if complete).
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Token usage (typically only in final chunk).
    #[serde(default)]
    pub usage: Option<LlmUsage>,
}

/// String-based tool choice values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StrToolChoice {
    /// Let the model decide whether to use tools.
    Auto,
    /// Don't use any tools.
    None,
    /// Use any available tool.
    Any,
    /// Require tool use.
    Required,
}

/// Description of an available function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableFunction {
    /// Name of the function.
    pub name: String,
    /// Description of what the function does.
    pub description: String,
    /// JSON Schema for the function parameters.
    pub parameters: serde_json::Value,
}

/// Description of an available tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailableTool {
    /// Type of tool (always "function" currently).
    #[serde(default = "default_function_type")]
    pub r#type: String,
    /// The function definition.
    pub function: AvailableFunction,
}

impl AvailableTool {
    /// Create a new function tool.
    #[must_use]
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            r#type: "function".to_string(),
            function: AvailableFunction {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// Tool choice - either a string directive or a specific tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// A string directive (auto, none, any, required).
    String(StrToolChoice),
    /// A specific tool to use.
    Tool(AvailableTool),
}

impl From<StrToolChoice> for ToolChoice {
    fn from(value: StrToolChoice) -> Self {
        Self::String(value)
    }
}

impl From<AvailableTool> for ToolChoice {
    fn from(value: AvailableTool) -> Self {
        Self::Tool(value)
    }
}

/// Optional tuning metadata attached to each generated token.
/// Carries top-k logits, tail samples, and expert routing data
/// so that consumers (e.g. `GenerationSavingStream`, `TuningWriter`)
/// can process tuning data without coupling to the backend internals.
#[derive(Debug, Clone)]
pub struct TokenTuningData {
    /// Top-k logit entries (sorted by probability, descending).
    pub top_k: Vec<LogitEntry>,
    /// Randomly sampled tail entries from outside top-k.
    pub tail: Vec<LogitEntry>,
    /// Total probability mass of the excluded tail.
    pub tail_mass: f32,
    /// Expert indices chosen by MoE routing for this token (flattened across layers).
    pub expert_indices: Option<Vec<u32>>,
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Context token usage update (used during prefill).
    ContextTokens {
        /// Current context tokens.
        context_tokens: u32,
    },
    /// A text delta from the assistant — render immediately in the UI.
    Token {
        /// Role of the message sender.
        role: Role,
        /// Text content delta.
        content: String,
        /// Context tokens in-flight for this turn (prompt + completion generated so far).
        context_tokens: u32,
        /// Tuning data for this token (present when tuning output is enabled).
        tuning: Option<TokenTuningData>,
    },
    /// A complete, parsed tool call ready for execution.
    ToolCall {
        /// Unique identifier for this tool call.
        id: String,
        /// Name of the tool/function to call.
        name: String,
        /// Parsed arguments as JSON.
        arguments: serde_json::Value,
        /// Original raw text of the tool call (for "preserve" mode).
        raw_text: Option<String>,
        /// Context tokens in-flight for this turn (prompt + completion generated so far).
        context_tokens: u32,
        /// Tuning data for all tokens that comprised this tool call
        /// (present when tuning output is enabled). One entry per generated token.
        tuning: Option<Vec<TokenTuningData>>,
    },
    /// Generation is finished.
    Done {
        /// Token usage statistics.
        usage: Option<LlmUsage>,
    },
}

/// Type alias for a boxed stream of `StreamEvent`s.
pub type EventStream = Pin<Box<dyn Stream<Item = LlmResult<StreamEvent>> + Send>>;
