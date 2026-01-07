//! System prompts for the agent.

use crate::paths::PROMPTS_DIR;
use std::path::PathBuf;

/// Built-in system prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemPrompt {
    /// CLI prompt for interactive coding assistant.
    Cli,
    /// Tests prompt for testing scenarios.
    Tests,
}

impl SystemPrompt {
    /// Get the prompt content.
    #[must_use]
    pub fn content(&self) -> &'static str {
        match self {
            Self::Cli => include_str!("prompts/cli.md"),
            Self::Tests => include_str!("prompts/tests.md"),
        }
    }
}

impl std::str::FromStr for SystemPrompt {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cli" => Ok(Self::Cli),
            "tests" => Ok(Self::Tests),
            _ => Err(format!("Unknown system prompt: {s}")),
        }
    }
}

/// Utility prompts used by the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtilityPrompt {
    /// Compact prompt for summarizing conversation history.
    Compact,
    /// Project context template.
    ProjectContext,
    /// Dangerous directory warning template.
    DangerousDirectory,
    /// Git repository context.
    Git,
}

impl UtilityPrompt {
    /// Get the prompt content.
    #[must_use]
    pub fn content(&self) -> &'static str {
        match self {
            Self::Compact => include_str!("prompts/compact.md"),
            Self::ProjectContext => include_str!("prompts/project_context.md"),
            Self::DangerousDirectory => include_str!("prompts/dangerous_directory.md"),
            Self::Git => include_str!("prompts/git.md"),
        }
    }

    /// Read the prompt content (alias for content for API compatibility).
    #[must_use]
    pub fn read(&self) -> &'static str {
        self.content()
    }
}

/// Load a system prompt by ID.
///
/// First tries built-in prompts, then looks in the prompts directory.
///
/// # Errors
///
/// Returns an error if the prompt cannot be found or read.
pub fn load_prompt(prompt_id: &str) -> Result<String, String> {
    // Try built-in prompts first
    if let Ok(builtin) = prompt_id.parse::<SystemPrompt>() {
        return Ok(builtin.content().to_string());
    }

    // Try custom prompt file
    let prompt_path = custom_prompt_path(prompt_id);
    if prompt_path.exists() {
        return std::fs::read_to_string(&prompt_path)
            .map_err(|e| format!("Failed to read prompt file {}: {e}", prompt_path.display()));
    }

    Err(format!(
        "Unknown prompt ID: '{prompt_id}'. Available: cli, tests, or create a custom prompt at {}",
        prompt_path.display()
    ))
}

/// Get the path for a custom prompt.
#[must_use]
pub fn custom_prompt_path(prompt_id: &str) -> PathBuf {
    PROMPTS_DIR.join(format!("{prompt_id}.md"))
}
