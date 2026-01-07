//! Bash command execution tool.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::error::{ToolError, ToolResult};
use crate::types::{PatternCheckResult, Tool, ToolConfig, ToolPermission};

/// Arguments for the bash tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashArgs {
    /// The command to execute.
    pub command: String,
    /// Whether to run the command in background.
    /// When true, the command is started and control returns immediately.
    #[serde(default)]
    pub is_background: bool,
    /// Optional brief description of what the command does.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional directory to run the command in.
    #[serde(default)]
    pub directory: Option<String>,
    /// Optional timeout override in seconds.
    #[serde(default)]
    pub timeout: Option<u64>,
}

/// Result from bash command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashResult {
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    pub stderr: String,
    /// Exit code.
    pub returncode: i32,
    /// Process ID if running in background.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_pid: Option<u32>,
    /// Whether this was a background command.
    #[serde(default)]
    pub is_background: bool,
}

/// Tool prompt loaded from markdown file.
const BASH_PROMPT: &str = include_str!("prompts/bash.md");

/// Bash command execution tool.
pub struct Bash {
    config: ToolConfig,
}

/// Get the default allowlist based on the platform.
fn get_default_allowlist() -> Vec<&'static str> {
    let common = vec![
        "echo",
        "find",
        "git diff",
        "git log",
        "git status",
        "tree",
        "whoami",
    ];

    if cfg!(target_os = "windows") {
        let mut list = common;
        list.extend(["dir", "findstr", "more", "type", "ver", "where"]);
        list
    } else {
        let mut list = common;
        list.extend([
            "cat", "file", "head", "ls", "pwd", "stat", "tail", "uname", "wc", "which",
        ]);
        list
    }
}

/// Get the default denylist based on the platform.
fn get_default_denylist() -> Vec<&'static str> {
    let common = vec!["gdb", "pdb", "passwd"];

    if cfg!(target_os = "windows") {
        let mut list = common;
        list.extend(["cmd /k", "powershell -NoExit", "pwsh -NoExit", "notepad"]);
        list
    } else {
        let mut list = common;
        list.extend([
            "nano", "vim", "vi", "emacs", "bash -i", "sh -i", "zsh -i", "fish -i", "dash -i",
            "screen", "tmux",
        ]);
        list
    }
}

/// Get the default standalone denylist based on the platform.
fn get_default_denylist_standalone() -> Vec<&'static str> {
    let common = vec!["python", "python3", "ipython"];

    if cfg!(target_os = "windows") {
        let mut list = common;
        list.extend(["cmd", "powershell", "pwsh", "notepad"]);
        list
    } else {
        let mut list = common;
        list.extend(["bash", "sh", "nohup", "vi", "vim", "emacs", "nano", "su"]);
        list
    }
}

impl Default for Bash {
    fn default() -> Self {
        let mut config = ToolConfig {
            permission: ToolPermission::Ask,
            ..Default::default()
        };

        // Set default allowlist (platform-specific)
        config
            .extra
            .insert("allowlist".to_string(), json!(get_default_allowlist()));

        // Set default denylist (platform-specific)
        config
            .extra
            .insert("denylist".to_string(), json!(get_default_denylist()));

        // Set default standalone denylist (platform-specific)
        config.extra.insert(
            "denylist_standalone".to_string(),
            json!(get_default_denylist_standalone()),
        );

        // Default timeout
        config
            .extra
            .insert("default_timeout".to_string(), json!(30));

        // Max output bytes
        config
            .extra
            .insert("max_output_bytes".to_string(), json!(16000));

        Self { config }
    }
}

impl Bash {
    /// Get the default timeout in seconds.
    fn default_timeout(&self) -> u64 {
        self.config.get_or("default_timeout", 30)
    }

    /// Get the maximum output bytes.
    fn max_output_bytes(&self) -> usize {
        self.config.get_or("max_output_bytes", 16000)
    }

    /// Get the allowlist patterns.
    fn allowlist(&self) -> Vec<String> {
        self.config.get_or("allowlist", Vec::new())
    }

    /// Get the denylist patterns.
    fn denylist(&self) -> Vec<String> {
        self.config.get_or("denylist", Vec::new())
    }

    /// Get the standalone denylist patterns.
    fn denylist_standalone(&self) -> Vec<String> {
        self.config.get_or("denylist_standalone", Vec::new())
    }

    /// Build environment variables for the subprocess.
    fn build_env(&self) -> HashMap<String, String> {
        let mut env: HashMap<String, String> = std::env::vars().collect();

        // Set non-interactive environment (common)
        env.insert("CI".to_string(), "true".to_string());
        env.insert("NONINTERACTIVE".to_string(), "1".to_string());
        env.insert("NO_TTY".to_string(), "1".to_string());
        env.insert("NO_COLOR".to_string(), "1".to_string());

        // Platform-specific environment
        if cfg!(target_os = "windows") {
            env.insert("GIT_PAGER".to_string(), "more".to_string());
            env.insert("PAGER".to_string(), "more".to_string());
        } else {
            env.insert("TERM".to_string(), "dumb".to_string());
            env.insert("DEBIAN_FRONTEND".to_string(), "noninteractive".to_string());
            env.insert("GIT_PAGER".to_string(), "cat".to_string());
            env.insert("PAGER".to_string(), "cat".to_string());
            env.insert("LESS".to_string(), "-FX".to_string());
            env.insert("LC_ALL".to_string(), "en_US.UTF-8".to_string());
        }

        env
    }
}

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Run a one-off bash command and capture its output."
    }

    fn prompt(&self) -> Option<&str> {
        Some(BASH_PROMPT)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "is_background": {
                    "type": "boolean",
                    "description": "Whether to run the command in background. Default is false. Set to true for long-running processes like development servers, watchers, or daemons that should continue running without blocking.",
                    "default": false
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what the command does for the user. Be specific and concise."
                },
                "directory": {
                    "type": "string",
                    "description": "The absolute path of the directory to run the command in. If not provided, uses the current working directory."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (default: 30). Ignored for background commands."
                }
            },
            "required": ["command"]
        })
    }

    fn config(&self) -> &ToolConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ToolConfig {
        &mut self.config
    }

    fn check_patterns(&self, args: &serde_json::Value) -> PatternCheckResult {
        let command = match args.get("command").and_then(|v| v.as_str()) {
            Some(cmd) => cmd,
            None => return PatternCheckResult::NoMatch,
        };

        // Split command by pipes and logical operators
        let parts: Vec<&str> = command
            .split(['|', ';'])
            .flat_map(|s| s.split("&&"))
            .flat_map(|s| s.split("||"))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if parts.is_empty() {
            return PatternCheckResult::NoMatch;
        }

        // Check denylist first
        for part in &parts {
            for pattern in self.denylist() {
                if part.starts_with(&pattern) {
                    return PatternCheckResult::Denied;
                }
            }

            // Check standalone denylist
            let words: Vec<&str> = part.split_whitespace().collect();
            if words.len() == 1 {
                let cmd_name = words[0].rsplit('/').next().unwrap_or(words[0]);
                for pattern in self.denylist_standalone() {
                    if cmd_name == pattern {
                        return PatternCheckResult::Denied;
                    }
                }
            }
        }

        // Check if all parts match allowlist
        let all_allowed = parts.iter().all(|part| {
            self.allowlist()
                .iter()
                .any(|pattern| part.starts_with(pattern))
        });

        if all_allowed {
            PatternCheckResult::Allowed
        } else {
            PatternCheckResult::NoMatch
        }
    }

    async fn execute(&mut self, args: serde_json::Value) -> ToolResult<serde_json::Value> {
        let args: BashArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let timeout_secs = args.timeout.unwrap_or_else(|| self.default_timeout());
        let max_bytes = self.max_output_bytes();
        let is_background = args.is_background;

        // Use specified directory or fall back to config workdir
        let workdir = args
            .directory
            .as_ref()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| self.config.effective_workdir());

        // Validate directory exists if specified
        if args.directory.is_some() && !workdir.exists() {
            return Err(ToolError::InvalidArguments(format!(
                "Directory does not exist: {}",
                workdir.display()
            )));
        }

        // Determine the shell to use
        // On Unix, asyncio.create_subprocess_shell uses 'sh' (POSIX shell)
        // On Windows, it uses COMSPEC or cmd.exe
        let shell = if cfg!(target_os = "windows") {
            std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string())
        } else {
            "sh".to_string()
        };

        let shell_arg = if cfg!(target_os = "windows") {
            "/C"
        } else {
            "-c"
        };

        // For background commands, append & if not already present (Unix only)
        let command = if is_background && !cfg!(target_os = "windows") {
            if args.command.trim().ends_with('&') {
                args.command.clone()
            } else {
                format!("{} &", args.command)
            }
        } else {
            args.command.clone()
        };

        let mut cmd = Command::new(&shell);
        cmd.arg(shell_arg)
            .arg(&command)
            .current_dir(&workdir)
            .envs(self.build_env())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // On Unix, start in a new process group
        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = cmd
            .spawn()
            .map_err(|e| ToolError::ProcessError(format!("Failed to spawn process: {e}")))?;

        let pid = child.id();

        // For background commands, don't wait for completion
        if is_background {
            // Give a brief moment to capture any immediate errors
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Try to get any immediate output without blocking
            let stdout = String::new();
            let stderr = String::new();

            let result = BashResult {
                stdout,
                stderr,
                returncode: 0,
                background_pid: pid,
                is_background: true,
            };
            return Ok(serde_json::to_value(result)?);
        }

        let stdout_handle = child.stdout.take();
        let stderr_handle = child.stderr.take();

        let result = timeout(Duration::from_secs(timeout_secs), async {
            let mut stdout_buf = Vec::new();
            let mut stderr_buf = Vec::new();

            if let Some(mut stdout) = stdout_handle {
                let _ = stdout.read_to_end(&mut stdout_buf).await;
            }
            if let Some(mut stderr) = stderr_handle {
                let _ = stderr.read_to_end(&mut stderr_buf).await;
            }

            let status = child.wait().await?;

            Ok::<_, std::io::Error>((stdout_buf, stderr_buf, status))
        })
        .await;

        match result {
            Ok(Ok((stdout_bytes, stderr_bytes, status))) => {
                let stdout =
                    String::from_utf8_lossy(&stdout_bytes[..stdout_bytes.len().min(max_bytes)])
                        .to_string();
                let stderr =
                    String::from_utf8_lossy(&stderr_bytes[..stderr_bytes.len().min(max_bytes)])
                        .to_string();
                let returncode = status.code().unwrap_or(-1);

                if returncode != 0 {
                    let mut error_msg = format!(
                        "Command failed: {:?}\nReturn code: {returncode}",
                        args.command
                    );
                    if !stderr.is_empty() {
                        error_msg.push_str(&format!("\nStderr: {stderr}"));
                    }
                    if !stdout.is_empty() {
                        error_msg.push_str(&format!("\nStdout: {stdout}"));
                    }
                    return Err(ToolError::ExecutionFailed(error_msg));
                }

                let result = BashResult {
                    stdout,
                    stderr,
                    returncode,
                    background_pid: None,
                    is_background: false,
                };
                Ok(serde_json::to_value(result)?)
            }
            Ok(Err(e)) => Err(ToolError::ProcessError(format!("Process error: {e}"))),
            Err(_) => {
                // Kill the process on timeout
                let _ = child.kill().await;
                Err(ToolError::Timeout {
                    seconds: timeout_secs,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bash_default_config() {
        let bash = Bash::default();
        assert_eq!(bash.name(), "bash");
        assert_eq!(bash.default_timeout(), 30);

        // Common commands should be in allowlist
        assert!(bash.allowlist().contains(&"echo".to_string()));
        assert!(bash.allowlist().contains(&"git status".to_string()));

        // Platform-specific allowlist checks
        #[cfg(not(target_os = "windows"))]
        {
            assert!(bash.allowlist().contains(&"ls".to_string()));
            assert!(bash.allowlist().contains(&"cat".to_string()));
        }

        #[cfg(target_os = "windows")]
        {
            assert!(bash.allowlist().contains(&"dir".to_string()));
            assert!(bash.allowlist().contains(&"type".to_string()));
        }

        // Common denylist
        assert!(bash.denylist().contains(&"gdb".to_string()));

        // Platform-specific denylist checks
        #[cfg(not(target_os = "windows"))]
        {
            assert!(bash.denylist().contains(&"vim".to_string()));
            assert!(bash.denylist().contains(&"nano".to_string()));
        }

        #[cfg(target_os = "windows")]
        {
            assert!(bash.denylist().contains(&"notepad".to_string()));
            assert!(bash.denylist().contains(&"cmd /k".to_string()));
        }
    }

    #[test]
    fn test_pattern_checking() {
        let bash = Bash::default();

        // Common allowed command
        let allowed = bash.check_patterns(&json!({"command": "echo hello"}));
        assert_eq!(allowed, PatternCheckResult::Allowed);

        // Platform-specific allowed commands
        #[cfg(not(target_os = "windows"))]
        {
            let allowed = bash.check_patterns(&json!({"command": "ls -la"}));
            assert_eq!(allowed, PatternCheckResult::Allowed);
        }

        #[cfg(target_os = "windows")]
        {
            let allowed = bash.check_patterns(&json!({"command": "dir"}));
            assert_eq!(allowed, PatternCheckResult::Allowed);
        }

        // Platform-specific denied commands
        #[cfg(not(target_os = "windows"))]
        {
            let denied = bash.check_patterns(&json!({"command": "vim file.txt"}));
            assert_eq!(denied, PatternCheckResult::Denied);
        }

        #[cfg(target_os = "windows")]
        {
            let denied = bash.check_patterns(&json!({"command": "notepad file.txt"}));
            assert_eq!(denied, PatternCheckResult::Denied);
        }

        // Unknown command
        let unknown = bash.check_patterns(&json!({"command": "custom_script.sh"}));
        assert_eq!(unknown, PatternCheckResult::NoMatch);
    }

    #[tokio::test]
    async fn test_bash_execute_echo() {
        let mut bash = Bash::default();
        let result = bash.execute(json!({"command": "echo hello"})).await;
        assert!(result.is_ok());
        let result: BashResult = serde_json::from_value(result.unwrap()).unwrap();
        assert!(result.stdout.trim() == "hello");
        assert_eq!(result.returncode, 0);
    }
}
