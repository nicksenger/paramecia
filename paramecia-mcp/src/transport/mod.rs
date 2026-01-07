//! MCP transport implementations.

mod http;
mod stdio;

pub use http::HttpTransport;
pub use stdio::StdioTransport;

use crate::error::McpResult;
use async_trait::async_trait;

/// Trait for MCP transports.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a message and receive a response.
    async fn send(&self, message: &str) -> McpResult<String>;

    /// Close the transport.
    async fn close(&self) -> McpResult<()>;
}
