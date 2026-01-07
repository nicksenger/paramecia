//! Stdio transport for MCP (local process communication).

use async_trait::async_trait;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use super::Transport;
use crate::error::{McpError, McpResult};

/// Stdio transport for local MCP server processes.
pub struct StdioTransport {
    child: Mutex<Child>,
    stdin: Mutex<tokio::process::ChildStdin>,
    stdout: Mutex<BufReader<tokio::process::ChildStdout>>,
    closed: AtomicBool,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning a process.
    ///
    /// # Arguments
    ///
    /// * `command` - The command and arguments to spawn
    pub async fn new(command: &[String]) -> McpResult<Self> {
        if command.is_empty() {
            return Err(McpError::ProcessSpawnError("Empty command".to_string()));
        }

        let mut cmd = Command::new(&command[0]);
        if command.len() > 1 {
            cmd.args(&command[1..]);
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd.spawn().map_err(|e| {
            McpError::ProcessSpawnError(format!("Failed to spawn '{}': {e}", command.join(" ")))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::ProcessSpawnError("Failed to capture stdin".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::ProcessSpawnError("Failed to capture stdout".to_string()))?;

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            closed: AtomicBool::new(false),
        })
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn send(&self, message: &str) -> McpResult<String> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(McpError::ConnectionFailed(
                "Transport is closed".to_string(),
            ));
        }

        // Write message with newline
        {
            let mut stdin = self.stdin.lock().await;
            stdin.write_all(message.as_bytes()).await?;
            stdin.write_all(b"\n").await?;
            stdin.flush().await?;
        }

        // Read response line
        let mut line = String::new();
        {
            let mut stdout = self.stdout.lock().await;
            stdout.read_line(&mut line).await?;
        }

        Ok(line.trim().to_string())
    }

    async fn close(&self) -> McpResult<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        // Close stdin to signal EOF
        {
            let mut stdin = self.stdin.lock().await;
            stdin.shutdown().await?;
        }

        // Wait for child to exit
        {
            let mut child = self.child.lock().await;
            let _ = child.wait().await;
        }

        Ok(())
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        // Mark as closed (best effort cleanup)
        self.closed.store(true, Ordering::SeqCst);
    }
}
