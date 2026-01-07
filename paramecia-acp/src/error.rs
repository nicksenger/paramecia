use thiserror::Error;

/// ACP-specific errors
#[derive(Error, Debug)]
pub enum AcpError {
    /// Protocol version mismatch
    #[error("Protocol version mismatch: expected {expected}, got {actual}")]
    ProtocolVersionMismatch { expected: String, actual: String },

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Session not found
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Invalid request parameters
    #[error("Invalid request parameters: {0}")]
    InvalidParameters(String),

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Internal server error
    #[error("Internal server error: {0}")]
    InternalError(String),

    /// Connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),
}
