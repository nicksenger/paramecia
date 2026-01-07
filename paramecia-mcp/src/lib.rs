//! Model Context Protocol (MCP) client implementation for Paramecia CLI.
//!
//! This crate provides the ability to connect to MCP servers and use their
//! tools as if they were builtin tools.

pub mod client;
pub mod error;
pub mod protocol;
pub mod transport;

pub use client::McpClient;
pub use error::{McpError, McpResult};
pub use protocol::{RemoteTool, ToolInput, ToolResult};
pub use transport::{HttpTransport, StdioTransport, Transport};
