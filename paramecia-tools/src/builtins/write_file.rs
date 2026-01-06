//! Write file tool.

use async_trait::async_trait;

/// Tool prompt loaded from markdown file.
const WRITE_FILE_PROMPT: &str = include_str!("prompts/write_file.md");
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use tokio::fs;

use crate::error::{ToolError, ToolResult};
use crate::types::{PatternCheckResult, Tool, ToolConfig, ToolPermission};

/// Arguments for the write_file tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteFileArgs {
    /// Path to the file to write.
    pub path: String,
    /// Content to write to the file.
    pub content: String,
    /// Must be set to true to overwrite an existing file.
    #[serde(default)]
    pub overwrite: bool,
}

/// Result from writing a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteFileResult {
    /// Path that was written.
    pub path: String,
    /// Number of bytes written.
    pub bytes_written: usize,
    /// Whether the file existed before writing.
    pub file_existed: bool,
    /// The content that was written.
    pub content: String,
}

/// Write file tool.
pub struct WriteFile {
    config: ToolConfig,
}

impl Default for WriteFile {
    fn default() -> Self {
        let mut config = ToolConfig {
            permission: ToolPermission::Ask,
            ..Default::default()
        };

        // Max bytes to write
        config
            .extra
            .insert("max_write_bytes".to_string(), json!(64000));

        // Whether to create parent directories
        config
            .extra
            .insert("create_parent_dirs".to_string(), json!(true));

        Self { config }
    }
}

impl WriteFile {
    /// Get max bytes to write.
    fn max_write_bytes(&self) -> usize {
        self.config.get_or("max_write_bytes", 64000)
    }

    /// Whether to create parent directories.
    fn create_parent_dirs(&self) -> bool {
        self.config.get_or("create_parent_dirs", true)
    }

    /// Resolve a path relative to the working directory.
    fn resolve_path(&self, path: &str) -> PathBuf {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            path
        } else {
            self.config.effective_workdir().join(path)
        }
    }
}

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create or overwrite a UTF-8 file. Fails if file exists unless 'overwrite=true'."
    }

    fn prompt(&self) -> Option<&str> {
        Some(WRITE_FILE_PROMPT)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Must be set to true to overwrite an existing file",
                    "default": false
                }
            },
            "required": ["path", "content"]
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

    async fn execute(&mut self, args: serde_json::Value) -> ToolResult<serde_json::Value> {
        let args: WriteFileArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        if args.path.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "Path cannot be empty".to_string(),
            ));
        }

        let content_bytes = args.content.len();
        if content_bytes > self.max_write_bytes() {
            return Err(ToolError::InvalidArguments(format!(
                "Content exceeds {} bytes limit",
                self.max_write_bytes()
            )));
        }

        let file_path = self.resolve_path(&args.path);

        // Security check: ensure file is within project directory
        // Find the closest existing ancestor of the path to canonicalize
        let workdir = self.config.effective_workdir();
        let workdir_canonical = workdir.canonicalize().unwrap_or_else(|_| workdir.clone());

        let closest_existing_ancestor = {
            let mut path = file_path.as_path();
            while let Some(parent) = path.parent() {
                if parent.exists() {
                    break;
                }
                path = parent;
            }
            path.parent().and_then(|p| p.canonicalize().ok())
        };

        let is_within_workdir = if let Some(ancestor) = closest_existing_ancestor {
            ancestor.starts_with(&workdir_canonical)
        } else if file_path.exists() {
            file_path
                .canonicalize()
                .map(|p| p.starts_with(&workdir_canonical))
                .unwrap_or(false)
        } else {
            // No existing ancestors, check if the non-canonical path starts with workdir
            file_path.starts_with(&workdir) || file_path.starts_with(&workdir_canonical)
        };

        if !is_within_workdir {
            return Err(ToolError::ExecutionFailed(format!(
                "Cannot write outside project directory: {}",
                file_path.display()
            )));
        }

        let file_existed = file_path.exists();

        // Check if file exists and overwrite is not set
        if file_existed && !args.overwrite {
            return Err(ToolError::ExecutionFailed(format!(
                "File '{}' exists. Set overwrite=true to replace.",
                file_path.display()
            )));
        }

        // Create parent directories if needed
        if self.create_parent_dirs() {
            if let Some(parent) = file_path.parent()
                && !parent.exists()
            {
                fs::create_dir_all(parent).await.map_err(|e| {
                    ToolError::FileError(format!(
                        "Failed to create directories for {}: {e}",
                        file_path.display()
                    ))
                })?;
            }
        } else if let Some(parent) = file_path.parent()
            && !parent.exists()
        {
            return Err(ToolError::FileError(format!(
                "Parent directory does not exist: {}",
                parent.display()
            )));
        }

        fs::write(&file_path, &args.content).await.map_err(|e| {
            ToolError::FileError(format!("Failed to write {}: {e}", file_path.display()))
        })?;

        let result = WriteFileResult {
            path: file_path.to_string_lossy().to_string(),
            bytes_written: content_bytes,
            file_existed,
            content: args.content,
        };

        Ok(serde_json::to_value(result)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_write_file_default_config() {
        let tool = WriteFile::default();
        assert_eq!(tool.name(), "write_file");
        assert_eq!(tool.config().permission, ToolPermission::Ask);
    }

    /// Create a WriteFile tool configured with workdir set to the temp directory.
    fn create_tool_with_workdir(workdir: &std::path::Path) -> WriteFile {
        let mut tool = WriteFile::default();
        tool.config.workdir = Some(workdir.to_path_buf());
        tool
    }

    #[tokio::test]
    async fn test_write_file_execute() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");

        let mut tool = create_tool_with_workdir(dir.path());
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": "Hello, World!"
            }))
            .await;

        assert!(result.is_ok());
        let result: WriteFileResult = serde_json::from_value(result.unwrap()).unwrap();
        assert!(!result.file_existed);
        assert_eq!(result.bytes_written, 13);

        // Verify content was written
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_write_file_overwrite_required() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("existing.txt");
        std::fs::write(&file_path, "original").unwrap();

        let mut tool = create_tool_with_workdir(dir.path());

        // Without overwrite flag, should fail
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": "new content"
            }))
            .await;
        assert!(result.is_err());

        // With overwrite flag, should succeed
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": "new content",
                "overwrite": true
            }))
            .await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "new content");
    }

    #[tokio::test]
    async fn test_write_file_creates_directories() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("subdir/nested/test.txt");

        let mut tool = create_tool_with_workdir(dir.path());
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": "Nested content"
            }))
            .await;

        assert!(result.is_ok());
        assert!(file_path.exists());
    }

    #[tokio::test]
    async fn test_write_file_outside_workdir_fails() {
        let dir = tempdir().unwrap();
        let other_dir = tempdir().unwrap();
        let file_path = other_dir.path().join("outside.txt");

        let mut tool = create_tool_with_workdir(dir.path());
        let result = tool
            .execute(json!({
                "path": file_path.to_str().unwrap(),
                "content": "Should fail"
            }))
            .await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string()
                    .contains("Cannot write outside project directory")
            );
        }
    }
}
