//! Core types, configuration, and agent logic for Paramecia CLI.
//!
//! This crate provides the central Agent implementation that coordinates
//! between the LLM backend and tools, as well as configuration management.

pub mod agent;
pub mod background_training;
pub mod config;
pub mod error;
pub mod events;
pub mod middleware;
pub mod modes;
pub mod multishot_examples;
pub mod paths;
pub mod project_context;
pub mod prompts;
pub mod session;
pub mod system_prompt;
pub mod trusted_folders;
pub mod types;
pub mod utils;

pub use agent::Agent;
pub use background_training::{
    BackgroundTrainingCommand, BackgroundTrainingEvent, BackgroundTrainingManager,
    BackgroundTrainingState, BackgroundTrainingStatus,
};
pub use config::VibeConfig;
pub use error::{VibeError, VibeResult};
pub use events::{AgentEvent, AssistantEvent, ToolCallEvent, ToolResultEvent};
pub use modes::AgentMode;
pub use multishot_examples::generate_multishot_examples;
pub use prompts::UtilityPrompt;
pub use system_prompt::{get_universal_system_prompt, get_universal_system_prompt_with_tools};
pub use types::AgentStats;
pub use utils::{
    CANCELLATION_TAG, CancellationReason, TOOL_ERROR_TAG, TaggedText, VIBE_STOP_EVENT_TAG,
    VIBE_WARNING_TAG, get_user_cancellation_message, is_user_cancellation_event,
};
