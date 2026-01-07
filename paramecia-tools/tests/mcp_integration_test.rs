//! Integration test for MCP tool registration

use paramecia_mcp::client::McpClient;
use paramecia_tools::manager::ToolManager;
use std::sync::Arc;

// Mock transport for testing
struct MockTransport;

impl MockTransport {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl paramecia_mcp::transport::Transport for MockTransport {
    async fn send(&self, _request: &str) -> paramecia_mcp::error::McpResult<String> {
        // Return mock responses for initialize and list_tools
        if _request.contains("initialize") {
            Ok(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock-server","version":"1.0.0"}}}"#.to_string())
        } else if _request.contains("tools/list") {
            Ok(r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"test_mcp_tool","description":"A test MCP tool","inputSchema":{"type":"object","properties":{"param1":{"type":"string"}}}}]}}"#.to_string())
        } else {
            Ok(r#"{"jsonrpc":"2.0","id":3,"result":{}}"#.to_string())
        }
    }

    async fn close(&self) -> paramecia_mcp::error::McpResult<()> {
        Ok(())
    }
}

#[tokio::test]
async fn test_mcp_tool_registration() {
    // Create a mock MCP client
    let mock_transport = Arc::new(MockTransport::new());
    let mut mock_client = McpClient::new(mock_transport);

    // Initialize the client
    mock_client.initialize().await.unwrap();

    // List tools
    let remote_tools = mock_client.list_tools().await.unwrap();
    assert_eq!(remote_tools.len(), 1);
    assert_eq!(remote_tools[0].name, "test_mcp_tool");
    assert_eq!(
        remote_tools[0].description,
        Some("A test MCP tool".to_string())
    );

    // Create tool manager and register MCP tools
    let mut tool_manager = ToolManager::new();
    let client_arc = Arc::new(mock_client);
    let registered = tool_manager.register_mcp_tools(client_arc, remote_tools);

    assert_eq!(registered, 1);

    // Verify the tool is available
    let available_tools = tool_manager.available_tools();
    assert!(available_tools.contains(&"test_mcp_tool".to_string()));

    // Verify we can get the tool
    let tool = tool_manager.get("test_mcp_tool").unwrap();
    let tool_read = tool.read();
    assert_eq!(tool_read.name(), "test_mcp_tool");
    assert_eq!(tool_read.description(), "A test MCP tool");
}

#[tokio::test]
async fn test_mcp_tool_info() {
    // Create a mock MCP client
    let mock_transport = Arc::new(MockTransport::new());
    let mut mock_client = McpClient::new(mock_transport);

    // Initialize and get tools
    mock_client.initialize().await.unwrap();
    let remote_tools = mock_client.list_tools().await.unwrap();

    // Create tool manager and register MCP tools
    let mut tool_manager = ToolManager::new();
    let client_arc = Arc::new(mock_client);
    tool_manager.register_mcp_tools(client_arc, remote_tools);

    // Get tool info
    let tool_infos = tool_manager.tool_infos();
    let mcp_tool_info = tool_infos.iter().find(|info| info.name == "test_mcp_tool");

    assert!(mcp_tool_info.is_some());
    let info = mcp_tool_info.unwrap();
    assert_eq!(info.name, "test_mcp_tool");
    assert_eq!(info.description, "A test MCP tool");

    // Verify the parameters schema is preserved
    let expected_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "param1": {"type": "string"}
        }
    });
    assert_eq!(info.parameters, expected_schema);
}
