//! Model Context Protocol (MCP) client implementation.
//!
//! Provides the ability to connect to MCP servers and use their
//! tools as if they were builtin tools.

pub mod client;
pub mod error;
pub mod protocol;
pub mod transport;

pub use client::McpClient;
pub use error::{McpError, McpResult};
pub use protocol::{RemoteTool, ToolInput, ToolResult as McpToolResult};
pub use transport::{HttpTransport, StdioTransport, Transport};
