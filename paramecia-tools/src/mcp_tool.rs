//! MCP tool wrapper that implements the Tool trait.

use crate::error::{ToolError, ToolResult};
use crate::types::{Tool, ToolConfig};
use async_trait::async_trait;
use paramecia_mcp::client::McpClient;
use paramecia_mcp::protocol::RemoteTool;
use std::sync::Arc;

/// Wrapper for MCP tools that implements the Tool trait.
pub struct McpTool {
    name: String,
    description: String,
    parameters_schema: serde_json::Value,
    client: Arc<McpClient>,
    config: ToolConfig,
}

impl McpTool {
    /// Create a new MCP tool wrapper.
    pub fn new(remote_tool: &RemoteTool, client: Arc<McpClient>) -> Self {
        Self {
            name: remote_tool.name.clone(),
            description: remote_tool.description.clone().unwrap_or_default(),
            parameters_schema: remote_tool.input_schema.clone(),
            client,
            config: ToolConfig::default(),
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.parameters_schema.clone()
    }

    fn config(&self) -> &ToolConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ToolConfig {
        &mut self.config
    }

    async fn execute(&mut self, args: serde_json::Value) -> ToolResult<serde_json::Value> {
        let result = self.client.call_tool(&self.name, args).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("MCP tool {} failed: {}", self.name, e))
        })?;

        // Convert MCP tool result to JSON value
        let text_content = result.text();
        Ok(serde_json::json!({ "result": text_content }))
    }
}
