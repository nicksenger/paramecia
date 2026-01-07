//! Error types for core operations.

use paramecia_llm::LlmError;
use paramecia_mcp::error::McpError;
use paramecia_tools::ToolError;
use thiserror::Error;

/// Errors that can occur in core operations.
#[derive(Debug, Error)]
pub enum VibeError {
    /// LLM error.
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// Tool error.
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Missing API key.
    #[error("Missing API key for {provider}. Set {env_var} environment variable.")]
    MissingApiKey {
        /// Provider name.
        provider: String,
        /// Environment variable name.
        env_var: String,
    },

    /// Missing prompt file.
    #[error(
        "Invalid system_prompt_id value: '{system_prompt_id}'. Must be one of the available prompts, or correspond to a .md file in {prompt_dir}"
    )]
    MissingPromptFile {
        /// System prompt ID.
        system_prompt_id: String,
        /// Prompt directory.
        prompt_dir: String,
    },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// TOML parsing error.
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    /// Session limit reached.
    #[error("Session limit reached: {0}")]
    SessionLimit(String),

    /// Agent state error.
    #[error("Agent state error: {0}")]
    AgentState(String),

    /// MCP error.
    #[error("MCP error: {0}")]
    Mcp(#[from] McpError),

    /// User cancelled operation.
    #[error("Operation cancelled by user")]
    Cancelled,
}

/// Result type for core operations.
pub type VibeResult<T> = Result<T, VibeError>;
