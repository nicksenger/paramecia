//! Error types for MCP operations.

use thiserror::Error;

/// Errors that can occur during MCP operations.
#[derive(Debug, Error)]
pub enum McpError {
    /// Failed to connect to MCP server.
    #[error("Failed to connect to MCP server: {0}")]
    ConnectionFailed(String),

    /// HTTP transport error.
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Protocol error.
    #[error("Protocol error: {0}")]
    ProtocolError(String),

    /// Tool invocation failed.
    #[error("Tool invocation failed: {0}")]
    ToolError(String),

    /// Server returned an error.
    #[error("Server error ({code}): {message}")]
    ServerError {
        /// Error code.
        code: i32,
        /// Error message.
        message: String,
    },

    /// Timeout waiting for response.
    #[error("Timeout waiting for MCP response")]
    Timeout,

    /// Process spawn error.
    #[error("Failed to spawn process: {0}")]
    ProcessSpawnError(String),
}

/// Result type for MCP operations.
pub type McpResult<T> = Result<T, McpError>;
