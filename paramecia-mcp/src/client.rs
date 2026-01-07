//! MCP client implementation.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{McpError, McpResult};
use crate::protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, InitializeResult, JsonRpcRequest,
    JsonRpcResponse, ListToolsResult, RemoteTool, ToolInput, ToolResult,
};
use crate::transport::Transport;

/// MCP client for communicating with MCP servers.
pub struct McpClient {
    transport: Arc<dyn Transport>,
    request_id: AtomicU64,
    initialized: bool,
}

impl McpClient {
    /// Create a new MCP client with the given transport.
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self {
            transport,
            request_id: AtomicU64::new(1),
            initialized: false,
        }
    }

    /// Get the next request ID.
    fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a JSON-RPC request and parse the response.
    async fn request<P: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: Option<P>,
    ) -> McpResult<R> {
        let request = JsonRpcRequest::new(self.next_id(), method, params);
        let request_json = serde_json::to_string(&request)?;

        let response_json = self.transport.send(&request_json).await?;
        let response: JsonRpcResponse<R> = serde_json::from_str(&response_json)?;

        if let Some(error) = response.error {
            return Err(McpError::ServerError {
                code: error.code,
                message: error.message,
            });
        }

        response
            .result
            .ok_or_else(|| McpError::ProtocolError("Response missing result".to_string()))
    }

    /// Initialize the connection with the server.
    pub async fn initialize(&mut self) -> McpResult<InitializeResult> {
        let params = InitializeParams {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ClientCapabilities {},
            client_info: ClientInfo {
                name: "paramecia".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        let result: InitializeResult = self.request("initialize", Some(params)).await?;

        // Send initialized notification
        let notification = JsonRpcRequest::<()>::new(0, "notifications/initialized", None);
        let notification_json = serde_json::to_string(&notification)?;
        let _ = self.transport.send(&notification_json).await;

        self.initialized = true;
        Ok(result)
    }

    /// List available tools from the server.
    pub async fn list_tools(&self) -> McpResult<Vec<RemoteTool>> {
        let result: ListToolsResult = self.request("tools/list", None::<()>).await?;
        Ok(result.tools)
    }

    /// Invoke a tool on the server.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> McpResult<ToolResult> {
        let params = ToolInput {
            name: name.to_string(),
            arguments,
        };

        self.request("tools/call", Some(params)).await
    }

    /// Close the client connection.
    pub async fn close(self) -> McpResult<()> {
        self.transport.close().await
    }
}

#[cfg(test)]
mod tests {
    // Tests would require a mock transport implementation
}
