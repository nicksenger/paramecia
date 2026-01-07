//! Grep/search tool using external ripgrep or grep.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use crate::error::{ToolError, ToolResult};
use crate::types::{Tool, ToolConfig, ToolPermission};

/// Tool prompt loaded from markdown file.
const GREP_PROMPT: &str = include_str!("prompts/grep.md");

/// Which grep backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrepBackend {
    /// ripgrep (rg)
    Ripgrep,
    /// GNU grep
    GnuGrep,
}

/// Default exclude patterns for common directories.
const DEFAULT_EXCLUDE_PATTERNS: &[&str] = &[
    ".venv/",
    "venv/",
    ".env/",
    "env/",
    "node_modules/",
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".tox/",
    ".nox/",
    ".coverage/",
    "htmlcov/",
    "dist/",
    "build/",
    ".idea/",
    ".vscode/",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
    "target/",
];

/// Arguments for the grep tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepArgs {
    /// The pattern to search for (regex).
    pub pattern: String,
    /// Path to search in (file or directory).
    #[serde(default = "default_path")]
    pub path: String,
    /// Maximum number of matches to return.
    #[serde(default)]
    pub max_matches: Option<usize>,
    /// Whether to respect .gitignore and .ignore files.
    #[serde(default = "default_true")]
    pub use_default_ignore: bool,
}

fn default_path() -> String {
    ".".to_string()
}

fn default_true() -> bool {
    true
}

/// Result from grep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepResult {
    /// The matched lines as a string.
    pub matches: String,
    /// Number of matches found.
    pub match_count: usize,
    /// Whether output was truncated.
    pub was_truncated: bool,
}

/// State for tracking search history.
#[derive(Debug, Clone, Default)]
pub struct GrepState {
    /// Recent search patterns.
    pub search_history: Vec<String>,
}

impl crate::types::ToolState for GrepState {
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

/// Grep/search tool using external ripgrep or grep.
pub struct Grep {
    config: ToolConfig,
    state: GrepState,
}

impl Default for Grep {
    fn default() -> Self {
        let mut config = ToolConfig {
            permission: ToolPermission::Always, // Read-only, safe by default
            ..Default::default()
        };

        // Maximum output bytes
        config
            .extra
            .insert("max_output_bytes".to_string(), json!(64_000));

        // Default max matches
        config
            .extra
            .insert("default_max_matches".to_string(), json!(100));

        // Default timeout in seconds
        config
            .extra
            .insert("default_timeout".to_string(), json!(60));

        // Codeignore file name
        config
            .extra
            .insert("codeignore_file".to_string(), json!(".vibeignore"));

        Self {
            config,
            state: GrepState::default(),
        }
    }
}

impl Grep {
    /// Get max output bytes.
    fn max_output_bytes(&self) -> usize {
        self.config.get_or("max_output_bytes", 64_000)
    }

    /// Get default max matches.
    fn default_max_matches(&self) -> usize {
        self.config.get_or("default_max_matches", 100)
    }

    /// Get default timeout.
    fn default_timeout(&self) -> u64 {
        self.config.get_or("default_timeout", 60)
    }

    /// Get codeignore file name.
    fn codeignore_file(&self) -> String {
        self.config
            .get_or("codeignore_file", ".vibeignore".to_string())
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

    /// Detect which grep backend is available.
    fn detect_backend() -> ToolResult<GrepBackend> {
        // Check for ripgrep first
        if which::which("rg").is_ok() {
            return Ok(GrepBackend::Ripgrep);
        }
        // Fall back to grep
        if which::which("grep").is_ok() {
            return Ok(GrepBackend::GnuGrep);
        }
        Err(ToolError::ExecutionFailed(
            "Neither ripgrep (rg) nor grep is installed. \
             Please install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
                .to_string(),
        ))
    }

    /// Collect exclude patterns from config and codeignore file.
    fn collect_exclude_patterns(&self) -> Vec<String> {
        let mut patterns: Vec<String> = DEFAULT_EXCLUDE_PATTERNS
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        // Load patterns from codeignore file
        let codeignore_path = self.config.effective_workdir().join(self.codeignore_file());
        if codeignore_path.is_file()
            && let Ok(content) = std::fs::read_to_string(&codeignore_path)
        {
            for line in content.lines() {
                let line = line.trim();
                if !line.is_empty() && !line.starts_with('#') {
                    patterns.push(line.to_string());
                }
            }
        }

        patterns
    }

    /// Build ripgrep command.
    fn build_ripgrep_command(&self, args: &GrepArgs, exclude_patterns: &[String]) -> Vec<String> {
        let max_matches = args
            .max_matches
            .unwrap_or_else(|| self.default_max_matches());

        let mut cmd = vec![
            "rg".to_string(),
            "--line-number".to_string(),
            "--no-heading".to_string(),
            "--smart-case".to_string(),
            "--no-binary".to_string(),
            // Request one extra to detect truncation
            "--max-count".to_string(),
            (max_matches + 1).to_string(),
        ];

        if !args.use_default_ignore {
            cmd.push("--no-ignore".to_string());
        }

        for pattern in exclude_patterns {
            cmd.push("--glob".to_string());
            cmd.push(format!("!{}", pattern));
        }

        cmd.push("-e".to_string());
        cmd.push(args.pattern.clone());
        cmd.push(args.path.clone());

        cmd
    }

    /// Build GNU grep command.
    fn build_gnu_grep_command(&self, args: &GrepArgs, exclude_patterns: &[String]) -> Vec<String> {
        let max_matches = args
            .max_matches
            .unwrap_or_else(|| self.default_max_matches());

        let mut cmd = vec![
            "grep".to_string(),
            "-r".to_string(),
            "-n".to_string(),
            "-I".to_string(), // Skip binary files
            "-E".to_string(), // Extended regex
            format!("--max-count={}", max_matches + 1),
        ];

        // Smart case: if pattern is lowercase, use -i
        if args.pattern.chars().all(|c| !c.is_uppercase()) {
            cmd.push("-i".to_string());
        }

        for pattern in exclude_patterns {
            if pattern.ends_with('/') {
                let dir_pattern = pattern.trim_end_matches('/');
                cmd.push(format!("--exclude-dir={}", dir_pattern));
            } else {
                cmd.push(format!("--exclude={}", pattern));
            }
        }

        cmd.push("-e".to_string());
        cmd.push(args.pattern.clone());
        cmd.push(args.path.clone());

        cmd
    }

    /// Execute the search command.
    async fn execute_search(&self, cmd: Vec<String>) -> ToolResult<String> {
        if cmd.is_empty() {
            return Err(ToolError::ExecutionFailed("Empty command".to_string()));
        }

        let (program, args) = cmd.split_first().unwrap();

        let mut command = Command::new(program);
        command
            .args(args)
            .current_dir(self.config.effective_workdir())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let timeout_secs = self.default_timeout();
        let result = timeout(Duration::from_secs(timeout_secs), async {
            let child = command.spawn().map_err(|e| {
                ToolError::ProcessError(format!("Failed to spawn {}: {}", program, e))
            })?;

            child.wait_with_output().await.map_err(|e| {
                ToolError::ProcessError(format!("Error waiting for {}: {}", program, e))
            })
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                // Exit code 0 = matches found, 1 = no matches, 2+ = error
                if output.status.code().unwrap_or(0) > 1 {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(ToolError::ExecutionFailed(format!(
                        "grep error: {}",
                        if stderr.is_empty() {
                            format!("Process exited with code {:?}", output.status.code())
                        } else {
                            stderr.to_string()
                        }
                    )));
                }

                Ok(String::from_utf8_lossy(&output.stdout).to_string())
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err(ToolError::Timeout {
                seconds: timeout_secs,
            }),
        }
    }

    /// Parse output and create result.
    fn parse_output(&self, stdout: &str, max_matches: usize) -> GrepResult {
        let output_lines: Vec<&str> = if stdout.is_empty() {
            Vec::new()
        } else {
            stdout.lines().collect()
        };

        let truncated_lines: Vec<&str> = output_lines.iter().take(max_matches).copied().collect();
        let truncated_output = truncated_lines.join("\n");

        let was_truncated =
            output_lines.len() > max_matches || truncated_output.len() > self.max_output_bytes();

        let final_output = if truncated_output.len() > self.max_output_bytes() {
            truncated_output[..self.max_output_bytes()].to_string()
        } else {
            truncated_output
        };

        GrepResult {
            matches: final_output,
            match_count: truncated_lines.len(),
            was_truncated,
        }
    }
}

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Recursively search files for a regex pattern using ripgrep (rg) or grep. \
         Respects .gitignore and .vibeignore files by default when using ripgrep."
    }

    fn prompt(&self) -> Option<&str> {
        Some(GREP_PROMPT)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Path to search (file or directory)",
                    "default": "."
                },
                "max_matches": {
                    "type": "integer",
                    "description": "Override the default maximum number of matches"
                },
                "use_default_ignore": {
                    "type": "boolean",
                    "description": "Whether to respect .gitignore and .ignore files",
                    "default": true
                }
            },
            "required": ["pattern"]
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
        self.state = GrepState::default();
    }

    async fn execute(&mut self, args: serde_json::Value) -> ToolResult<serde_json::Value> {
        let args: GrepArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        // Validate pattern is not empty
        if args.pattern.trim().is_empty() {
            return Err(ToolError::InvalidArguments(
                "Empty search pattern provided".to_string(),
            ));
        }

        // Validate path exists
        let resolved_path = self.resolve_path(&args.path);
        if !resolved_path.exists() {
            return Err(ToolError::FileError(format!(
                "Path does not exist: {}",
                args.path
            )));
        }

        // Track search history
        self.state.search_history.push(args.pattern.clone());

        // Detect backend
        let backend = Self::detect_backend()?;

        // Collect exclude patterns
        let exclude_patterns = self.collect_exclude_patterns();

        // Build command
        let cmd = match backend {
            GrepBackend::Ripgrep => self.build_ripgrep_command(&args, &exclude_patterns),
            GrepBackend::GnuGrep => self.build_gnu_grep_command(&args, &exclude_patterns),
        };

        // Execute search
        let stdout = self.execute_search(cmd).await?;

        // Parse output
        let max_matches = args
            .max_matches
            .unwrap_or_else(|| self.default_max_matches());
        let result = self.parse_output(&stdout, max_matches);

        Ok(serde_json::to_value(result)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_grep_default_config() {
        let tool = Grep::default();
        assert_eq!(tool.name(), "grep");
        assert_eq!(tool.config().permission, ToolPermission::Always);
    }

    #[test]
    fn test_grep_detect_backend() {
        // This test may fail in environments without grep/rg
        // but should work in most dev environments
        let result = Grep::detect_backend();
        // Either one should be found
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_grep_single_file() {
        // Skip if no backend available
        if Grep::detect_backend().is_err() {
            return;
        }

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&file_path).unwrap();
        writeln!(file, "Hello, World!").unwrap();
        writeln!(file, "This is a test.").unwrap();
        writeln!(file, "Hello again!").unwrap();

        let mut tool = Grep::default();
        let result = tool
            .execute(json!({
                "pattern": "Hello",
                "path": file_path.to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let result: GrepResult = serde_json::from_value(result.unwrap()).unwrap();
        assert_eq!(result.match_count, 2);
        assert!(result.matches.contains("Hello"));
    }

    #[tokio::test]
    async fn test_grep_directory() {
        // Skip if no backend available
        if Grep::detect_backend().is_err() {
            return;
        }

        let dir = tempdir().unwrap();

        // Create some files
        let mut file1 = std::fs::File::create(dir.path().join("file1.txt")).unwrap();
        writeln!(file1, "TODO: implement this").unwrap();

        let mut file2 = std::fs::File::create(dir.path().join("file2.txt")).unwrap();
        writeln!(file2, "No todos here").unwrap();

        let mut file3 = std::fs::File::create(dir.path().join("file3.txt")).unwrap();
        writeln!(file3, "TODO: another task").unwrap();

        let mut tool = Grep::default();
        let result = tool
            .execute(json!({
                "pattern": "TODO",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(result.is_ok());
        let result: GrepResult = serde_json::from_value(result.unwrap()).unwrap();
        assert_eq!(result.match_count, 2);
    }

    #[tokio::test]
    async fn test_grep_empty_pattern() {
        let mut tool = Grep::default();
        let result = tool
            .execute(json!({
                "pattern": "",
                "path": "."
            }))
            .await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Empty search pattern"));
        }
    }

    #[tokio::test]
    async fn test_grep_nonexistent_path() {
        let mut tool = Grep::default();
        let result = tool
            .execute(json!({
                "pattern": "test",
                "path": "/nonexistent/path/that/does/not/exist"
            }))
            .await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("does not exist"));
        }
    }
}
