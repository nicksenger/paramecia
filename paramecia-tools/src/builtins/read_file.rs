//! Read file tool.

use async_trait::async_trait;

/// Tool prompt loaded from markdown file.
const READ_FILE_PROMPT: &str = include_str!("prompts/read_file.md");
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::error::{ToolError, ToolResult};
use crate::types::{PatternCheckResult, Tool, ToolConfig, ToolPermission};

/// Arguments for the read_file tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadFileArgs {
    /// Path to the file to read.
    pub path: String,
    /// Line number to start reading from (0-indexed, inclusive).
    #[serde(default)]
    pub offset: usize,
    /// Maximum number of lines to read.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Result from reading a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadFileResult {
    /// Path that was read.
    pub path: String,
    /// Content of the file.
    pub content: String,
    /// Number of lines read.
    pub lines_read: usize,
    /// Whether reading was truncated due to byte limit.
    pub was_truncated: bool,
}

/// State for tracking recently read files.
#[derive(Debug, Clone, Default)]
pub struct ReadFileState {
    /// Recently read file paths.
    pub recently_read_files: Vec<String>,
}

impl crate::types::ToolState for ReadFileState {
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

/// Read file tool.
pub struct ReadFile {
    config: ToolConfig,
    state: ReadFileState,
}

impl Default for ReadFile {
    fn default() -> Self {
        let mut config = ToolConfig {
            permission: ToolPermission::Always, // Read is safe by default
            ..Default::default()
        };

        // Default max bytes to read
        config
            .extra
            .insert("max_read_bytes".to_string(), json!(64000));

        // Max state history
        config
            .extra
            .insert("max_state_history".to_string(), json!(10));

        Self {
            config,
            state: ReadFileState::default(),
        }
    }
}

impl ReadFile {
    /// Get max bytes to read.
    fn max_read_bytes(&self) -> usize {
        self.config.get_or("max_read_bytes", 64000)
    }

    /// Get max state history.
    fn max_state_history(&self) -> usize {
        self.config.get_or("max_state_history", 10)
    }

    /// Resolve a path relative to the working directory.
    fn resolve_path(&self, path: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path);
        let resolved = if path.is_absolute() {
            path
        } else {
            self.config.effective_workdir().join(path)
        };

        // Canonicalize to resolve any .. or symlinks
        resolved
            .canonicalize()
            .map_err(|e| ToolError::FileError(format!("Cannot resolve path: {e}")))
    }

    /// Update state with the recently read file.
    fn update_state(&mut self, path: &str) {
        self.state.recently_read_files.push(path.to_string());
        let max = self.max_state_history();
        if self.state.recently_read_files.len() > max {
            self.state.recently_read_files.remove(0);
        }
    }
}

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a UTF-8 file, returning content from a specific line range. \
         Reading is capped by a byte limit for safety."
    }

    fn prompt(&self) -> Option<&str> {
        Some(READ_FILE_PROMPT)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-indexed, inclusive)",
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["path"]
        })
    }

    fn config(&self) -> &ToolConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ToolConfig {
        &mut self.config
    }

    fn check_patterns(&self, args: &serde_json::Value) -> PatternCheckResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return PatternCheckResult::NoMatch,
        };

        let file_path = PathBuf::from(path);
        let resolved = if file_path.is_absolute() {
            file_path
        } else {
            self.config.effective_workdir().join(file_path)
        };
        let file_str = resolved.to_string_lossy();

        // Check denylist first
        for pattern in &self.config.denylist {
            if let Ok(glob_pattern) = glob::Pattern::new(pattern)
                && glob_pattern.matches(&file_str)
            {
                return PatternCheckResult::Denied;
            }
        }

        // Check allowlist
        for pattern in &self.config.allowlist {
            if let Ok(glob_pattern) = glob::Pattern::new(pattern)
                && glob_pattern.matches(&file_str)
            {
                return PatternCheckResult::Allowed;
            }
        }

        PatternCheckResult::NoMatch
    }

    fn state(&self) -> Option<&dyn crate::types::ToolState> {
        Some(&self.state)
    }

    fn state_mut(&mut self) -> Option<&mut dyn crate::types::ToolState> {
        Some(&mut self.state)
    }

    fn reset(&mut self) {
        self.state = ReadFileState::default();
    }

    async fn execute(&mut self, args: serde_json::Value) -> ToolResult<serde_json::Value> {
        let args: ReadFileArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        if args.path.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "Path cannot be empty".to_string(),
            ));
        }

        let file_path = self.resolve_path(&args.path)?;

        if !file_path.exists() {
            return Err(ToolError::FileError(format!(
                "File not found: {}",
                file_path.display()
            )));
        }

        if file_path.is_dir() {
            return Err(ToolError::FileError(format!(
                "Path is a directory, not a file: {}",
                file_path.display()
            )));
        }

        let max_bytes = self.max_read_bytes();
        let file = File::open(&file_path).await.map_err(|e| {
            ToolError::FileError(format!("Error opening {}: {e}", file_path.display()))
        })?;

        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut content = String::new();
        let mut line_index = 0;
        let mut lines_read = 0;
        let mut bytes_read = 0;
        let mut was_truncated = false;

        while let Some(line) = lines.next_line().await.map_err(|e| {
            ToolError::FileError(format!("Error reading {}: {e}", file_path.display()))
        })? {
            // Skip lines before offset
            if line_index < args.offset {
                line_index += 1;
                continue;
            }

            // Check limit
            if let Some(limit) = args.limit
                && lines_read >= limit
            {
                break;
            }

            // Check byte limit
            let line_bytes = line.len() + 1; // +1 for newline
            if bytes_read + line_bytes > max_bytes {
                was_truncated = true;
                break;
            }

            content.push_str(&line);
            content.push('\n');
            bytes_read += line_bytes;
            lines_read += 1;
            line_index += 1;
        }

        self.update_state(&file_path.to_string_lossy());

        let result = ReadFileResult {
            path: file_path.to_string_lossy().to_string(),
            content,
            lines_read,
            was_truncated,
        };

        Ok(serde_json::to_value(result)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_file_default_config() {
        let tool = ReadFile::default();
        assert_eq!(tool.name(), "read_file");
        assert_eq!(tool.config().permission, ToolPermission::Always);
    }

    #[tokio::test]
    async fn test_read_file_execute() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "line 1").unwrap();
        writeln!(file, "line 2").unwrap();
        writeln!(file, "line 3").unwrap();

        let mut tool = ReadFile::default();
        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let result: ReadFileResult = serde_json::from_value(result.unwrap()).unwrap();
        assert_eq!(result.lines_read, 3);
        assert!(result.content.contains("line 1"));
        assert!(result.content.contains("line 2"));
        assert!(result.content.contains("line 3"));
    }

    #[tokio::test]
    async fn test_read_file_with_offset_and_limit() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 1..=10 {
            writeln!(file, "line {i}").unwrap();
        }

        let mut tool = ReadFile::default();
        let result = tool
            .execute(json!({
                "path": file.path().to_str().unwrap(),
                "offset": 2,
                "limit": 3
            }))
            .await;

        assert!(result.is_ok());
        let result: ReadFileResult = serde_json::from_value(result.unwrap()).unwrap();
        assert_eq!(result.lines_read, 3);
        assert!(result.content.contains("line 3"));
        assert!(result.content.contains("line 4"));
        assert!(result.content.contains("line 5"));
        assert!(!result.content.contains("line 1"));
        assert!(!result.content.contains("line 6"));
    }
}
