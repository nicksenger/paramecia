//! MCP protocol types.

use serde::{Deserialize, Serialize};

/// JSON-RPC request.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcRequest<T> {
    /// JSON-RPC version.
    pub jsonrpc: &'static str,
    /// Request ID.
    pub id: u64,
    /// Method name.
    pub method: String,
    /// Parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<T>,
}

impl<T> JsonRpcRequest<T> {
    /// Create a new request.
    pub fn new(id: u64, method: impl Into<String>, params: Option<T>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC response.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse<T> {
    /// JSON-RPC version.
    pub jsonrpc: String,
    /// Request ID.
    pub id: Option<u64>,
    /// Result (mutually exclusive with error).
    pub result: Option<T>,
    /// Error (mutually exclusive with result).
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: i32,
    /// Error message.
    pub message: String,
    /// Additional data.
    pub data: Option<serde_json::Value>,
}

/// Server capabilities.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ServerCapabilities {
    /// Tool capabilities.
    #[serde(default)]
    pub tools: Option<ToolCapabilities>,
}

/// Tool capabilities.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ToolCapabilities {
    /// Whether the server supports tool listing changes.
    #[serde(default)]
    pub list_changed: bool,
}

/// Initialize request params.
#[derive(Debug, Clone, Serialize)]
pub struct InitializeParams {
    /// Protocol version.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Client capabilities.
    pub capabilities: ClientCapabilities,
    /// Client info.
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

/// Client capabilities.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ClientCapabilities {}

/// Client info.
#[derive(Debug, Clone, Serialize)]
pub struct ClientInfo {
    /// Client name.
    pub name: String,
    /// Client version.
    pub version: String,
}

/// Initialize response result.
#[derive(Debug, Clone, Deserialize)]
pub struct InitializeResult {
    /// Protocol version.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Server capabilities.
    pub capabilities: ServerCapabilities,
    /// Server info.
    #[serde(rename = "serverInfo")]
    pub server_info: Option<ServerInfo>,
}

/// Server info.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerInfo {
    /// Server name.
    pub name: String,
    /// Server version.
    pub version: Option<String>,
}

/// List tools response.
#[derive(Debug, Clone, Deserialize)]
pub struct ListToolsResult {
    /// List of available tools.
    pub tools: Vec<RemoteTool>,
}

/// A remote tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteTool {
    /// Tool name.
    pub name: String,
    /// Tool description.
    #[serde(default)]
    pub description: Option<String>,
    /// Input schema.
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// Tool input for invocation.
#[derive(Debug, Clone, Serialize)]
pub struct ToolInput {
    /// Tool name.
    pub name: String,
    /// Tool arguments.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Tool invocation result.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolResult {
    /// Content returned by the tool.
    pub content: Vec<ToolContent>,
    /// Whether the tool execution resulted in an error.
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

/// Content from a tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolContent {
    /// Text content.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
    },
    /// Image content.
    #[serde(rename = "image")]
    Image {
        /// Base64-encoded image data.
        data: String,
        /// MIME type.
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// Resource content.
    #[serde(rename = "resource")]
    Resource {
        /// Resource content.
        resource: ResourceContent,
    },
}

/// Resource content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    /// Resource URI.
    pub uri: String,
    /// MIME type.
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
    /// Text content.
    pub text: Option<String>,
    /// Binary content (base64).
    pub blob: Option<String>,
}

impl ToolResult {
    /// Get the text content from the result.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                ToolContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
