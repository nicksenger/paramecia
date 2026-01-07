//! Tool abstractions and builtin tools for Paramecia CLI.
//!
//! This crate provides the trait definitions for tools and implementations
//! of builtin tools like bash, read_file, write_file, search_replace, grep, and todo.

pub mod builtins;
pub mod error;
pub mod manager;
pub mod mcp_tool;
pub mod model_prompts;
pub mod tool_aliases;
pub mod types;

pub use builtins::{Bash, Grep, ReadFile, SearchReplace, Todo, WriteFile};
pub use error::{ToolError, ToolResult};
pub use manager::ToolManager;
pub use mcp_tool::McpTool;
pub use model_prompts::{
    ModelFamily, ModelPromptBuilder, ModelToolConfig, ToolCallExampleFormat, get_model_tool_prompt,
    select_prompt_for_model,
};
pub use tool_aliases::{ToolAliasConfig, ToolAliasManager, get_internal_tool_name, get_tool_alias};
pub use types::{Tool, ToolConfig, ToolInfo, ToolPermission, ToolState};
