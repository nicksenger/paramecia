//! Core types for the tool system.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::path::PathBuf;

use crate::error::ToolResult;

/// Permission level for tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolPermission {
    /// Always allow execution without prompting.
    Always,
    /// Never allow execution.
    Never,
    /// Ask user for permission (default).
    #[default]
    Ask,
}

impl std::str::FromStr for ToolPermission {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            "ask" => Ok(Self::Ask),
            _ => Err(format!(
                "Invalid permission: {s}. Must be 'always', 'never', or 'ask'"
            )),
        }
    }
}

/// Configuration for a tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Permission level for this tool.
    #[serde(default)]
    pub permission: ToolPermission,

    /// Working directory for the tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir: Option<PathBuf>,

    /// Patterns that automatically allow tool execution.
    #[serde(default)]
    pub allowlist: Vec<String>,

    /// Patterns that automatically deny tool execution.
    #[serde(default)]
    pub denylist: Vec<String>,

    /// Additional tool-specific configuration.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ToolConfig {
    /// Get the effective working directory.
    #[must_use]
    pub fn effective_workdir(&self) -> PathBuf {
        self.workdir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
    }

    /// Get a configuration value by key.
    #[must_use]
    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.extra
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Get a configuration value with a default.
    pub fn get_or<T: serde::de::DeserializeOwned>(&self, key: &str, default: T) -> T {
        self.get(key).unwrap_or(default)
    }
}

/// Information about a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    /// Name of the tool.
    pub name: String,
    /// Description of what the tool does.
    pub description: String,
    /// JSON Schema for the tool parameters.
    pub parameters: serde_json::Value,
}

/// State that a tool can maintain between invocations.
pub trait ToolState: Send + Sync + Any {
    /// Convert to Any for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Convert to mutable Any for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Clone the state into a boxed trait object.
    fn clone_box(&self) -> Box<dyn ToolState>;
}

impl Clone for Box<dyn ToolState> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Default implementation for tools that don't need state.
#[derive(Debug, Clone, Default)]
pub struct NoState;

impl ToolState for NoState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn ToolState> {
        Box::new(self.clone())
    }
}

/// Result of checking allowlist/denylist patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternCheckResult {
    /// Pattern matched allowlist - always allow.
    Allowed,
    /// Pattern matched denylist - always deny.
    Denied,
    /// No pattern matched - use normal permission check.
    NoMatch,
}

/// The main trait that all tools must implement.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the name of the tool.
    fn name(&self) -> &str;

    /// Get a description of what the tool does.
    fn description(&self) -> &str;

    /// Get the JSON Schema for the tool parameters.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Get information about this tool.
    fn info(&self) -> ToolInfo {
        ToolInfo {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters_schema(),
        }
    }

    /// Get the tool's prompt/additional context if available.
    fn prompt(&self) -> Option<&str> {
        None
    }

    /// Get the tool's configuration.
    fn config(&self) -> &ToolConfig;

    /// Get mutable access to the tool's configuration.
    fn config_mut(&mut self) -> &mut ToolConfig;

    /// Get the tool's state if it maintains any.
    fn state(&self) -> Option<&dyn ToolState> {
        None
    }

    /// Get mutable access to the tool's state.
    fn state_mut(&mut self) -> Option<&mut dyn ToolState> {
        None
    }

    /// Execute the tool with the given arguments.
    ///
    /// # Arguments
    ///
    /// * `args` - JSON value containing the tool arguments
    ///
    /// # Returns
    ///
    /// A JSON value containing the tool result.
    async fn execute(&mut self, args: serde_json::Value) -> ToolResult<serde_json::Value>;

    /// Check if the given arguments match allowlist/denylist patterns.
    ///
    /// The default implementation returns `NoMatch`. Tools should override
    /// this to implement pattern-based auto-approval/denial.
    fn check_patterns(&self, _args: &serde_json::Value) -> PatternCheckResult {
        PatternCheckResult::NoMatch
    }

    /// Reset the tool's state.
    fn reset(&mut self) {}
}

/// Extension trait for creating tools from configuration.
pub trait ToolFactory: Sized {
    /// Create a new tool instance with the given configuration.
    fn from_config(config: ToolConfig) -> Self;

    /// Create a new tool instance with default configuration.
    fn new() -> Self
    where
        Self: Default,
    {
        Self::default()
    }
}
