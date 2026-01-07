//! Error types for tool operations.

use thiserror::Error;

/// Errors that can occur during tool operations.
#[derive(Debug, Error)]
pub enum ToolError {
    /// Tool execution failed.
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),

    /// Invalid arguments provided.
    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    /// Permission denied.
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Tool not found.
    #[error("Tool not found: {0}")]
    NotFound(String),

    /// File operation error.
    #[error("File error: {0}")]
    FileError(String),

    /// Process execution error.
    #[error("Process error: {0}")]
    ProcessError(String),

    /// Timeout occurred.
    #[error("Timeout after {seconds} seconds")]
    Timeout {
        /// Timeout duration in seconds.
        seconds: u64,
    },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Regex error.
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),
}

/// Result type for tool operations.
pub type ToolResult<T> = Result<T, ToolError>;
