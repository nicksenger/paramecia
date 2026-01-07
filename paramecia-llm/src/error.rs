//! Error types for LLM operations.

use thiserror::Error;

/// Errors that can occur during LLM operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// Failed to parse response.
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Request timeout.
    #[error("Request timed out after {seconds} seconds")]
    Timeout {
        /// Timeout duration in seconds.
        seconds: u64,
    },

    /// Streaming error.
    #[error("Streaming error: {0}")]
    StreamError(String),

    /// Usage data missing.
    #[error("Usage data missing in response")]
    MissingUsage,

    /// Local model error.
    #[error("Local model error: {0}")]
    ModelError(String),
}

/// Result type for LLM operations.
pub type LlmResult<T> = Result<T, LlmError>;

impl From<candle::Error> for LlmError {
    fn from(value: candle::Error) -> Self {
        Self::ModelError(value.to_string())
    }
}

impl From<tokenizers::Error> for LlmError {
    fn from(value: tokenizers::Error) -> Self {
        Self::ModelError(value.to_string())
    }
}
