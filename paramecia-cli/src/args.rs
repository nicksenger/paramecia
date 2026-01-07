//! Command-line argument parsing.

use clap::Parser;
use paramecia_harness::modes::AgentMode;
use paramecia_harness::types::OutputFormat;
use std::io::{self, Read};

/// Version constant matching the Python CLI
const VERSION: &str = "1.2.1";

/// Paramecia CLI - Rust rewrite of Mistral's open-source CLI coding assistant
#[derive(Parser, Debug)]
#[command(name = "paramecia")]
#[command(author = "Devstral 2")]
#[command(version = VERSION)]
#[command(about = "Run the Paramecia interactive CLI")]
#[command(long_about = None)]
pub struct Args {
    /// Initial prompt to start the interactive session with
    #[arg(value_name = "PROMPT")]
    pub initial_prompt: Option<String>,

    /// Run in programmatic mode: send prompt, auto-approve all tools, output response, and exit
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    pub prompt: Option<String>,

    /// Start in auto-approve mode: never ask for approval before running tools
    #[arg(long, default_value = "false")]
    pub auto_approve: bool,

    /// Start in plan mode: read-only tools for exploration and planning
    #[arg(long, default_value = "false")]
    pub plan: bool,

    /// Maximum number of assistant turns (only applies in programmatic mode with -p)
    #[arg(long, value_name = "N")]
    pub max_turns: Option<u32>,

    /// Maximum cost in dollars (only applies in programmatic mode with -p).
    /// Session will be interrupted if cost exceeds this limit.
    #[arg(long, value_name = "DOLLARS")]
    pub max_price: Option<f64>,

    /// Enable specific tools. In programmatic mode (-p), this disables all other tools.
    /// Can use exact names, glob patterns (e.g., 'bash*'), or regex with 're:' prefix.
    /// Can be specified multiple times.
    #[arg(long = "enabled-tools", value_name = "TOOL")]
    pub enabled_tools: Vec<String>,

    /// Output format for programmatic mode (-p): 'text' for human-readable (default),
    /// 'json' for all messages at end, 'streaming' for newline-delimited JSON per message.
    #[arg(long, value_name = "FORMAT", default_value = "text")]
    pub output: String,

    /// Load agent configuration from ~/.vibe/agents/NAME.toml
    #[arg(long, value_name = "NAME")]
    pub agent: Option<String>,

    /// Setup API key and exit
    #[arg(long)]
    pub setup: bool,

    /// Continue from the most recent saved session
    #[arg(short = 'c', long = "continue")]
    pub continue_session: bool,

    /// Resume a specific session by its ID (supports partial matching)
    #[arg(long, value_name = "SESSION_ID")]
    pub resume: Option<String>,
}

impl Args {
    /// Get the agent mode based on arguments.
    #[must_use]
    pub fn mode(&self) -> AgentMode {
        if self.plan {
            AgentMode::Plan
        } else if self.auto_approve {
            AgentMode::AutoApprove
        } else {
            AgentMode::Default
        }
    }

    /// Get the output format.
    #[must_use]
    pub fn output_format(&self) -> OutputFormat {
        self.output.parse().unwrap_or_default()
    }

    /// Get the prompt from arguments or stdin.
    pub fn get_prompt(&self) -> anyhow::Result<String> {
        // Handle the case where --prompt is provided without a value (const="")
        // This matches Python's argparse behavior with nargs="?" and const=""
        if let Some(prompt) = &self.prompt {
            if !prompt.is_empty() {
                return Ok(prompt.clone());
            }
            // Empty prompt provided (--prompt without value), try stdin
            if let Some(stdin_prompt) = self.get_stdin_prompt() {
                return Ok(stdin_prompt);
            }
            // No stdin input available either
            anyhow::bail!("No prompt provided for programmatic mode")
        }

        // No --prompt flag provided, return empty string for interactive mode
        Ok(String::new())
    }

    /// Try to get prompt from stdin if available.
    pub fn get_stdin_prompt(&self) -> Option<String> {
        if atty::is(atty::Stream::Stdin) {
            return None;
        }

        let mut buffer = String::new();
        if io::stdin().read_to_string(&mut buffer).is_ok() && !buffer.trim().is_empty() {
            return Some(buffer.trim().to_string());
        }

        None
    }
}
