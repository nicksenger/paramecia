//! MCP tool wrapper that implements the Tool trait.

use crate::error::{ToolError, ToolResult};
use crate::mcp::client::McpClient;
use crate::mcp::protocol::RemoteTool;
use crate::types::{Tool, ToolConfig};
use async_trait::async_trait;
use std::sync::Arc;

/// Wrapper for MCP tools that implements the Tool trait.
pub struct McpTool {
    name: String,
    description: String,
    parameters_schema: serde_json::Value,
    prompt: String,
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
            prompt: Self::build_prompt(remote_tool),
            client,
            config: ToolConfig::default(),
        }
    }

    fn build_prompt(remote_tool: &RemoteTool) -> String {
        let description = remote_tool
            .description
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("Remote MCP tool.");
        let parameters = serde_json::to_string_pretty(&remote_tool.input_schema)
            .unwrap_or_else(|_| remote_tool.input_schema.to_string());

        format!(
            "{description}\n\n**Parameters Schema:**\n```json\n{parameters}\n```"
        )
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

    fn prompt(&self) -> Option<&str> {
        Some(&self.prompt)
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
