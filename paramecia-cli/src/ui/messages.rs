//! Message types for the chat interface.

use std::time::Instant;

/// Safely truncate a string to a maximum number of characters (not bytes).
/// This handles UTF-8 multi-byte characters correctly to avoid panics.
fn truncate_string(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars - 3).collect();
        format!("{}...", truncated)
    }
}

/// A message in the chat history.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Message {
    /// User message.
    User(UserMessage),
    /// Assistant message (streaming or complete).
    Assistant(AssistantMessage),
    /// Tool call being executed.
    ToolCall(ToolCallMessage),
    /// Tool execution result.
    ToolResult(ToolResultMessage),
    /// System/command message.
    System(SystemMessage),
    /// User command message (for /commands).
    UserCommand(UserCommandMessage),
    /// Error message.
    Error(ErrorMessage),
    /// Warning message.
    Warning(WarningMessage),
    /// Interrupt message.
    Interrupt,
    /// Compact message.
    Compact(CompactMessage),
    /// Bash command output (user ran !command).
    BashOutput(BashOutputMessage),
}

/// A user message.
#[derive(Debug, Clone)]
pub struct UserMessage {
    /// The message content.
    pub content: String,
    /// Whether the message is pending (waiting for agent init).
    pub pending: bool,
}

impl UserMessage {
    /// Create a new user message.
    #[must_use]
    pub fn new(content: String) -> Self {
        Self {
            content,
            pending: false,
        }
    }

    /// Create a pending user message.
    #[must_use]
    pub fn pending(content: String) -> Self {
        Self {
            content,
            pending: true,
        }
    }
}

/// An assistant message.
#[derive(Debug, Clone)]
pub struct AssistantMessage {
    /// The message content (may be partial during streaming).
    pub content: String,
    /// Whether the message is complete.
    pub complete: bool,
}

impl AssistantMessage {
    /// Create a new assistant message.
    #[must_use]
    pub fn new(content: String) -> Self {
        Self {
            content,
            complete: false,
        }
    }

    /// Create a complete assistant message.
    #[must_use]
    #[allow(dead_code)]
    pub fn complete(content: String) -> Self {
        Self {
            content,
            complete: true,
        }
    }

    /// Append content to the message.
    pub fn append(&mut self, content: &str) {
        self.content.push_str(content);
    }
}

/// A tool call message.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ToolCallMessage {
    /// Name of the tool being called.
    pub tool_name: String,
    /// Summary of what the tool is doing.
    pub summary: String,
    /// Whether the tool is currently executing.
    pub spinning: bool,
    /// Start time of the tool call.
    pub start_time: Option<Instant>,
}

impl ToolCallMessage {
    /// Create a new tool call message.
    #[must_use]
    pub fn new(tool_name: String, args: &serde_json::Value) -> Self {
        let summary = Self::generate_summary(&tool_name, args);
        Self {
            tool_name,
            summary,
            spinning: true,
            start_time: Some(Instant::now()),
        }
    }

    /// Generate a human-readable summary of the tool call.
    fn generate_summary(tool_name: &str, args: &serde_json::Value) -> String {
        match tool_name {
            "bash" => {
                if let Some(cmd) = args.get("command").and_then(|v| v.as_str()) {
                    let truncated = truncate_string(cmd, 60);
                    format!("bash: {truncated}")
                } else {
                    "bash".to_string()
                }
            }
            "read_file" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    format!("read_file: {path}")
                } else {
                    "read_file".to_string()
                }
            }
            "write_file" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    format!("write_file: {path}")
                } else {
                    "write_file".to_string()
                }
            }
            "search_replace" => {
                if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                    format!("search_replace: {path}")
                } else {
                    "search_replace".to_string()
                }
            }
            "grep" => {
                if let Some(pattern) = args.get("pattern").and_then(|v| v.as_str()) {
                    let truncated = truncate_string(pattern, 40);
                    format!("grep: {truncated}")
                } else {
                    "grep".to_string()
                }
            }
            "todo" => "todo".to_string(),
            _ => tool_name.to_string(),
        }
    }

    /// Stop the spinner.
    pub fn stop(&mut self) {
        self.spinning = false;
    }
}

/// A tool result message.
#[derive(Debug, Clone)]
pub struct ToolResultMessage {
    /// Name of the tool.
    pub tool_name: String,
    /// Result content (if successful).
    pub result: Option<serde_json::Value>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Whether the tool was skipped.
    pub skipped: bool,
    /// Skip reason.
    pub skip_reason: Option<String>,
    /// Execution duration.
    pub duration: Option<f64>,
    /// Whether to show collapsed.
    pub collapsed: bool,
}

impl ToolResultMessage {
    /// Create a success result.
    #[must_use]
    pub fn success(tool_name: String, result: serde_json::Value, duration: Option<f64>) -> Self {
        Self {
            tool_name,
            result: Some(result),
            error: None,
            skipped: false,
            skip_reason: None,
            duration,
            collapsed: true,
        }
    }

    /// Create an error result.
    #[must_use]
    pub fn error(tool_name: String, error: String, duration: Option<f64>) -> Self {
        Self {
            tool_name,
            result: None,
            error: Some(error),
            skipped: false,
            skip_reason: None,
            duration,
            collapsed: true,
        }
    }

    /// Create a skipped result.
    #[must_use]
    pub fn skipped(tool_name: String, reason: String) -> Self {
        Self {
            tool_name,
            result: None,
            error: None,
            skipped: true,
            skip_reason: Some(reason),
            duration: None,
            collapsed: true,
        }
    }

    /// Get a display summary.
    #[must_use]
    pub fn summary(&self) -> String {
        // Use different shortcut hint for todo tool
        let shortcut = if self.tool_name == "todo" {
            "ctrl+t"
        } else {
            "ctrl+o"
        };

        if let Some(error) = &self.error {
            if self.collapsed {
                format!("Error. ({shortcut} to expand)")
            } else {
                format!("Error: {error}")
            }
        } else if self.skipped {
            if self.collapsed {
                format!("Skipped. ({shortcut} to expand)")
            } else {
                format!(
                    "Skipped: {}",
                    self.skip_reason.as_deref().unwrap_or("User skipped")
                )
            }
        } else {
            let duration_str = self
                .duration
                .map(|d| format!(" ({:.1}s)", d))
                .unwrap_or_default();
            if self.collapsed {
                format!(
                    "{} completed{} ({shortcut} to expand)",
                    self.tool_name, duration_str
                )
            } else {
                self.format_result()
            }
        }
    }

    /// Format the result for display.
    fn format_result(&self) -> String {
        if let Some(result) = &self.result {
            let duration_str = self
                .duration
                .map(|d| format!(" ({:.1}s)", d))
                .unwrap_or_default();

            if let Some(obj) = result.as_object() {
                let mut lines = vec![format!("{} completed{}", self.tool_name, duration_str)];
                for (key, value) in obj {
                    let value_str = match value {
                        serde_json::Value::String(s) => {
                            // Truncate long strings (safely handle UTF-8)
                            truncate_string(s, 200)
                        }
                        serde_json::Value::Null => String::new(),
                        other => {
                            let s = other.to_string();
                            truncate_string(&s, 200)
                        }
                    };
                    if !value_str.is_empty() {
                        lines.push(format!("  {}: {}", key, value_str));
                    }
                }
                lines.join("\n")
            } else {
                format!("{} completed{}", self.tool_name, duration_str)
            }
        } else {
            format!("{} completed", self.tool_name)
        }
    }

    /// Toggle collapsed state.
    #[allow(dead_code)]
    pub fn toggle_collapsed(&mut self) {
        self.collapsed = !self.collapsed;
    }
}

/// A system/command response message.
#[derive(Debug, Clone)]
pub struct SystemMessage {
    /// The message content.
    pub content: String,
    /// Message type for styling.
    pub kind: SystemMessageKind,
}

/// Type of system message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemMessageKind {
    /// Informational message.
    Info,
    /// Warning message.
    Warning,
    /// Error message.
    Error,
}

impl SystemMessage {
    /// Create an info message.
    #[must_use]
    pub fn info(content: String) -> Self {
        Self {
            content,
            kind: SystemMessageKind::Info,
        }
    }

    /// Create a warning message.
    #[must_use]
    pub fn warning(content: String) -> Self {
        Self {
            content,
            kind: SystemMessageKind::Warning,
        }
    }

    /// Create an error message.
    #[must_use]
    pub fn error(content: String) -> Self {
        Self {
            content,
            kind: SystemMessageKind::Error,
        }
    }
}

/// A compact/summary message.
#[derive(Debug, Clone)]
pub struct CompactMessage {
    /// Whether compaction is in progress.
    pub in_progress: bool,
    /// Old token count (before compaction).
    pub old_tokens: Option<u32>,
    /// New token count (after compaction).
    pub new_tokens: Option<u32>,
    /// Error message if compaction failed.
    pub error: Option<String>,
}

impl Default for CompactMessage {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactMessage {
    /// Create a new in-progress compact message.
    #[must_use]
    pub fn new() -> Self {
        Self {
            in_progress: true,
            old_tokens: None,
            new_tokens: None,
            error: None,
        }
    }

    /// Mark compaction as complete.
    #[allow(dead_code)]
    pub fn complete(&mut self, old_tokens: u32, new_tokens: u32) {
        self.in_progress = false;
        self.old_tokens = Some(old_tokens);
        self.new_tokens = Some(new_tokens);
    }

    /// Mark compaction as failed.
    #[allow(dead_code)]
    pub fn fail(&mut self, error: String) {
        self.in_progress = false;
        self.error = Some(error);
    }
}

/// A bash command output message.
#[derive(Debug, Clone)]
pub struct BashOutputMessage {
    /// The command that was run.
    pub command: String,
    /// The working directory.
    pub cwd: String,
    /// The output of the command.
    pub output: String,
    /// The exit code.
    pub exit_code: i32,
}

impl BashOutputMessage {
    /// Create a new bash output message.
    #[must_use]
    pub fn new(command: String, cwd: String, output: String, exit_code: i32) -> Self {
        Self {
            command,
            cwd,
            output,
            exit_code,
        }
    }
}

/// A user command message (for /commands like /help, /status, etc.).
#[derive(Debug, Clone)]
pub struct UserCommandMessage {
    /// The command content.
    pub content: String,
}

/// An error message with better formatting.
#[derive(Debug, Clone)]
pub struct ErrorMessage {
    /// The error content.
    pub content: String,
}

/// A warning message with better formatting.
#[derive(Debug, Clone)]
pub struct WarningMessage {
    /// The warning content.
    pub content: String,
}
