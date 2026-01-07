//! LLM backend implementations for Paramecia CLI.
//!
//! This crate provides the local LLM backend for running quantized models.

pub mod backend;
pub mod chat_template;
pub mod error;
pub mod format;
pub mod streaming_parser;
pub mod types;
pub mod xml_tool_parser;

pub use backend::{Backend, BackendFactory, LocalBackend};
pub use chat_template::{ChatTemplate, ChatTemplateError, QWEN3_NEXT_CHAT_TEMPLATE};
pub use error::{LlmError, LlmResult};
pub use format::ApiToolFormatHandler;
pub use streaming_parser::{
    CompletedToolCall, StreamingToolCallParser, ToolCallMeta, ToolCallParseResult,
};
pub use types::{
    AvailableFunction, AvailableTool, FunctionCall, LlmChunk, LlmMessage, LlmUsage, Role,
    StrToolChoice, ToolCall, ToolChoice,
};
pub use xml_tool_parser::{ParsedToolCall, ToolCallStyle, XmlToolCallParser, parse_any_tool_calls};
