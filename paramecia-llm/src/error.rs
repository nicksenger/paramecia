//! Error types for LLM operations.

use thiserror::Error;

/// Errors that can occur during LLM operations.
#[derive(Debug, Error)]
pub enum LlmError {
    /// HTTP request failed.
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    /// Failed to parse response.
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// API returned an error.
    #[error("API error from {provider} ({status}): {message}")]
    ApiError {
        /// Name of the provider.
        provider: String,
        /// HTTP status code.
        status: u16,
        /// Error message from the API.
        message: String,
    },

    /// Rate limit exceeded.
    #[error("Rate limit exceeded for {provider}. Retry after: {retry_after:?}")]
    RateLimited {
        /// Name of the provider.
        provider: String,
        /// Seconds to wait before retrying.
        retry_after: Option<u64>,
    },

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Missing API key.
    #[error("Missing API key for {provider}. Set {env_var} environment variable.")]
    MissingApiKey {
        /// Name of the provider.
        provider: String,
        /// Environment variable name.
        env_var: String,
    },

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
