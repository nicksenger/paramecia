//! Todo list management tool.

use async_trait::async_trait;

/// Tool prompt loaded from markdown file.
const TODO_PROMPT: &str = include_str!("prompts/todo.md");
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{ToolError, ToolResult};
use crate::types::{Tool, ToolConfig, ToolPermission};

/// Status of a todo item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Not yet started.
    Pending,
    /// Currently being worked on.
    InProgress,
    /// Completed successfully.
    Completed,
    /// No longer needed.
    Cancelled,
}

impl std::fmt::Display for TodoStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::InProgress => write!(f, "in_progress"),
            Self::Completed => write!(f, "completed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Priority of a todo item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoPriority {
    /// Low priority.
    Low,
    /// Medium priority (default).
    #[default]
    Medium,
    /// High priority.
    High,
}

/// A single todo item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    /// Unique identifier.
    pub id: String,
    /// Content/description.
    pub content: String,
    /// Current status.
    #[serde(default)]
    pub status: TodoStatus,
    /// Priority level.
    #[serde(default)]
    pub priority: TodoPriority,
}

impl Default for TodoStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// Arguments for the todo tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoArgs {
    /// Action to perform: 'read' or 'write'.
    pub action: String,
    /// Complete list of todos when writing.
    #[serde(default)]
    pub todos: Option<Vec<TodoItem>>,
}

/// Result from todo operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoResult {
    /// Message describing what happened.
    pub message: String,
    /// Current list of all todos.
    pub todos: Vec<TodoItem>,
    /// Total count of todos.
    pub total_count: usize,
}

/// State for the todo tool.
#[derive(Debug, Clone, Default)]
pub struct TodoState {
    /// Current todo items.
    pub todos: Vec<TodoItem>,
}

impl crate::types::ToolState for TodoState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn crate::types::ToolState> {
        Box::new(self.clone())
    }
}

/// Todo list management tool.
pub struct Todo {
    config: ToolConfig,
    state: TodoState,
}

impl Default for Todo {
    fn default() -> Self {
        let mut config = ToolConfig {
            permission: ToolPermission::Always, // Safe operation
            ..Default::default()
        };
        // Max todos to store
        config.extra.insert("max_todos".to_string(), json!(100));

        Self {
            config,
            state: TodoState::default(),
        }
    }
}

impl Todo {
    fn max_todos(&self) -> usize {
        self.config.get_or("max_todos", 100)
    }

    fn read_todos(&self) -> TodoResult {
        TodoResult {
            message: format!("Retrieved {} todos", self.state.todos.len()),
            todos: self.state.todos.clone(),
            total_count: self.state.todos.len(),
        }
    }

    fn write_todos(&mut self, todos: Vec<TodoItem>) -> ToolResult<TodoResult> {
        let max = self.max_todos();
        if todos.len() > max {
            return Err(ToolError::InvalidArguments(format!(
                "Cannot store more than {} todos",
                max
            )));
        }

        // Check for duplicate IDs
        let ids: Vec<&str> = todos.iter().map(|t| t.id.as_str()).collect();
        let unique_ids: std::collections::HashSet<&str> = ids.iter().copied().collect();
        if ids.len() != unique_ids.len() {
            return Err(ToolError::InvalidArguments(
                "Todo IDs must be unique".to_string(),
            ));
        }

        self.state.todos = todos;

        Ok(TodoResult {
            message: format!("Updated {} todos", self.state.todos.len()),
            todos: self.state.todos.clone(),
            total_count: self.state.todos.len(),
        })
    }
}

#[async_trait]
impl Tool for Todo {
    fn name(&self) -> &str {
        "todo"
    }

    fn description(&self) -> &str {
        "Manage todos. Use action='read' to view, action='write' with complete list to update."
    }

    fn prompt(&self) -> Option<&str> {
        Some(TODO_PROMPT)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "Either 'read' or 'write'"
                },
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for the todo item"
                            },
                            "content": {
                                "type": "string",
                                "description": "Description of the todo item"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                                "description": "Current status of the todo item",
                                "default": "pending"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": "Priority level of the todo item",
                                "default": "medium"
                            }
                        },
                        "required": ["id", "content"]
                    },
                    "description": "Complete list of todos when writing"
                }
            },
            "required": ["action"]
        })
    }

    fn config(&self) -> &ToolConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ToolConfig {
        &mut self.config
    }

    fn state(&self) -> Option<&dyn crate::types::ToolState> {
        Some(&self.state)
    }

    fn state_mut(&mut self) -> Option<&mut dyn crate::types::ToolState> {
        Some(&mut self.state)
    }

    fn reset(&mut self) {
        self.state = TodoState::default();
    }

    async fn execute(&mut self, args: serde_json::Value) -> ToolResult<serde_json::Value> {
        let args: TodoArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let result = match args.action.as_str() {
            "read" => self.read_todos(),
            "write" => self.write_todos(args.todos.unwrap_or_default())?,
            _ => {
                return Err(ToolError::InvalidArguments(format!(
                    "Invalid action '{}'. Use 'read' or 'write'.",
                    args.action
                )));
            }
        };

        Ok(serde_json::to_value(result)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_todo_default_config() {
        let tool = Todo::default();
        assert_eq!(tool.name(), "todo");
        assert_eq!(tool.config().permission, ToolPermission::Always);
    }

    #[tokio::test]
    async fn test_todo_read_empty() {
        let mut tool = Todo::default();
        let result = tool
            .execute(json!({
                "action": "read"
            }))
            .await;

        assert!(result.is_ok());
        let result: TodoResult = serde_json::from_value(result.unwrap()).unwrap();
        assert_eq!(result.todos.len(), 0);
        assert_eq!(result.total_count, 0);
        assert!(result.message.contains("0 todos"));
    }

    #[tokio::test]
    async fn test_todo_write() {
        let mut tool = Todo::default();
        let result = tool
            .execute(json!({
                "action": "write",
                "todos": [
                    {"id": "1", "content": "First task", "status": "pending"},
                    {"id": "2", "content": "Second task", "status": "in_progress"}
                ]
            }))
            .await;

        assert!(result.is_ok());
        let result: TodoResult = serde_json::from_value(result.unwrap()).unwrap();
        assert_eq!(result.todos.len(), 2);
        assert_eq!(result.total_count, 2);
    }

    #[tokio::test]
    async fn test_todo_write_then_read() {
        let mut tool = Todo::default();

        // Write initial todos
        tool.execute(json!({
            "action": "write",
            "todos": [
                {"id": "1", "content": "First task", "status": "pending"}
            ]
        }))
        .await
        .unwrap();

        // Read back
        let result = tool
            .execute(json!({
                "action": "read"
            }))
            .await;

        assert!(result.is_ok());
        let result: TodoResult = serde_json::from_value(result.unwrap()).unwrap();
        assert_eq!(result.todos.len(), 1);
        assert_eq!(result.todos[0].id, "1");
        assert_eq!(result.todos[0].content, "First task");
    }

    #[tokio::test]
    async fn test_todo_duplicate_ids() {
        let mut tool = Todo::default();
        let result = tool
            .execute(json!({
                "action": "write",
                "todos": [
                    {"id": "1", "content": "First task", "status": "pending"},
                    {"id": "1", "content": "Duplicate ID", "status": "pending"}
                ]
            }))
            .await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("unique"));
        }
    }

    #[tokio::test]
    async fn test_todo_invalid_action() {
        let mut tool = Todo::default();
        let result = tool
            .execute(json!({
                "action": "invalid"
            }))
            .await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Invalid action"));
        }
    }
}
