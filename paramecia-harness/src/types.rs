//! Core types for the agent.

use serde::{Deserialize, Serialize};

/// Statistics about an agent session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentStats {
    /// Number of conversation steps.
    pub steps: u32,
    /// Total prompt tokens used in the session.
    pub session_prompt_tokens: u32,
    /// Total completion tokens used in the session.
    pub session_completion_tokens: u32,
    /// Number of tool calls that were approved.
    pub tool_calls_agreed: u32,
    /// Number of tool calls that were rejected.
    pub tool_calls_rejected: u32,
    /// Number of tool calls that failed.
    pub tool_calls_failed: u32,
    /// Number of tool calls that succeeded.
    pub tool_calls_succeeded: u32,
    /// Current context tokens.
    pub context_tokens: u32,
    /// Prompt tokens from the last turn.
    pub last_turn_prompt_tokens: u32,
    /// Completion tokens from the last turn.
    pub last_turn_completion_tokens: u32,
    /// Duration of the last turn in seconds.
    pub last_turn_duration: f64,
    /// Tokens per second.
    pub tokens_per_second: f64,
    /// Price per million input tokens.
    pub input_price_per_million: f64,
    /// Price per million output tokens.
    pub output_price_per_million: f64,
}

impl AgentStats {
    /// Get total LLM tokens used in the session.
    #[must_use]
    pub fn session_total_llm_tokens(&self) -> u32 {
        self.session_prompt_tokens + self.session_completion_tokens
    }

    /// Get total tokens from the last turn.
    #[must_use]
    pub fn last_turn_total_tokens(&self) -> u32 {
        self.last_turn_prompt_tokens + self.last_turn_completion_tokens
    }

    /// Calculate the estimated session cost in dollars.
    #[must_use]
    pub fn session_cost(&self) -> f64 {
        let input_cost =
            (f64::from(self.session_prompt_tokens) / 1_000_000.0) * self.input_price_per_million;
        let output_cost = (f64::from(self.session_completion_tokens) / 1_000_000.0)
            * self.output_price_per_million;
        input_cost + output_cost
    }

    /// Update pricing information.
    pub fn update_pricing(&mut self, input_price: f64, output_price: f64) {
        self.input_price_per_million = input_price;
        self.output_price_per_million = output_price;
    }

    /// Reset context-related fields while preserving session totals.
    pub fn reset_context_state(&mut self) {
        self.context_tokens = 0;
        self.last_turn_prompt_tokens = 0;
        self.last_turn_completion_tokens = 0;
        self.last_turn_duration = 0.0;
        self.tokens_per_second = 0.0;
    }

    /// Add tokens from a message to the session totals.
    pub fn add_message_tokens(&mut self, msg: &paramecia_llm::LlmMessage) {
        // This is a simplified version - in a real implementation, you would
        // need to actually count the tokens in the message content
        if let Some(content) = &msg.content {
            let token_count = content.len() as u32 / 4; // Rough estimate
            match msg.role {
                paramecia_llm::Role::System | paramecia_llm::Role::User => {
                    // Count as prompt tokens
                    self.session_prompt_tokens += token_count;
                }
                paramecia_llm::Role::Assistant => {
                    // Count as completion tokens
                    self.session_completion_tokens += token_count;
                }
                paramecia_llm::Role::Tool => {
                    // Tool messages don't count toward LLM token usage
                }
            }
        }
    }

    /// Calculate context tokens from a list of messages.
    /// This represents the total size of the conversation context.
    pub fn calculate_context_tokens(messages: &[paramecia_llm::LlmMessage]) -> u32 {
        messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .map(|content| content.len() as u32 / 4) // Rough token estimate
            .sum()
    }
}

/// Response from approval callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalResponse {
    /// Approved.
    Yes,
    /// Denied.
    No,
    /// Always approve this tool.
    Always,
}

/// Session metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Session ID.
    pub session_id: String,
    /// Start time.
    pub start_time: String,
    /// End time.
    pub end_time: Option<String>,
    /// Git commit at session start.
    pub git_commit: Option<String>,
    /// Git branch at session start.
    pub git_branch: Option<String>,
    /// Whether auto-approve was enabled.
    pub auto_approve: bool,
    /// Username.
    pub username: String,
}

/// Output format for programmatic mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Plain text output.
    #[default]
    Text,
    /// JSON output.
    Json,
    /// Streaming JSON (newline-delimited).
    Streaming,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "streaming" => Ok(Self::Streaming),
            _ => Err(format!("Unknown output format: {s}")),
        }
    }
}
