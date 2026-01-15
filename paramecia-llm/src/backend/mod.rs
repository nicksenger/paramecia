//! LLM backend implementations.

mod factory;
mod venture;
mod local;

pub use factory::{BackendFactory, BackendType};
pub use venture::VentureBackend;
pub use local::LocalBackend;

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::error::LlmResult;
use crate::types::{AvailableTool, LlmChunk, LlmMessage, ToolChoice};

/// Configuration for a model.
///
/// Sampling parameters can be configured via environment variables:
/// - `PARAMECIA_TEMPERATURE`: Temperature for generation (default: 0.7)
/// - `PARAMECIA_TOP_P`: Top-p (nucleus) sampling threshold (default: 0.8)
/// - `PARAMECIA_TOP_K`: Top-k sampling limit (default: 20)
/// - `PARAMECIA_REPEAT_PENALTY`: Repetition penalty (default: 1.0, disabled)
/// - `PARAMECIA_PRESENCE_PENALTY`: Presence penalty (default: 0.0, disabled)
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model name/identifier.
    pub name: String,
    /// Temperature for generation (default: 0.7).
    /// Can be set via `PARAMECIA_TEMPERATURE` env var.
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold (default: 0.8).
    /// Can be set via `PARAMECIA_TOP_P` env var.
    pub top_p: f32,
    /// Top-k sampling limit (default: 20).
    /// Can be set via `PARAMECIA_TOP_K` env var.
    pub top_k: usize,
    /// Min-p sampling threshold (default: 0.0, disabled).
    pub min_p: f32,
    /// Repetition penalty (multiplicative) (default: 1.0, disabled).
    /// Divides logits of previously seen tokens by this value.
    /// A value of 1.0 means no penalty (disabled).
    /// Can be set via `PARAMECIA_REPEAT_PENALTY` env var.
    pub repeat_penalty: f32,
    /// Presence penalty (additive/flat) (default: 0.0, disabled).
    /// Subtracts this value from logits of tokens that have appeared.
    /// A value of 0.0 means no penalty (disabled).
    /// Can be set via `PARAMECIA_PRESENCE_PENALTY` env var.
    pub presence_penalty: f32,
    /// Thinking budget in tokens (default: 500).
    /// After this many tokens are generated,  is injected to end thinking.
    /// Set to 0 to disable.
    pub thinking_budget: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            temperature: std::env::var("PARAMECIA_TEMPERATURE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7),
            top_p: std::env::var("PARAMECIA_TOP_P")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.8),
            top_k: std::env::var("PARAMECIA_TOP_K")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(20),
            min_p: 0.0,
            repeat_penalty: std::env::var("PARAMECIA_REPEAT_PENALTY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            presence_penalty: std::env::var("PARAMECIA_PRESENCE_PENALTY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            thinking_budget: 500,
        }
    }
}

impl ModelConfig {
    /// Create a new ModelConfig with the given name and default sampling parameters.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Create a new ModelConfig with custom temperature and default other parameters.
    #[must_use]
    pub fn with_temperature(name: impl Into<String>, temperature: f32) -> Self {
        Self {
            name: name.into(),
            temperature,
            ..Default::default()
        }
    }
}

/// Configuration for a provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Provider name.
    pub name: String,
    /// Base URL for the API (unused for local backend).
    pub api_base: String,
    /// Environment variable containing the API key (unused for local backend).
    pub api_key_env_var: String,
    /// Backend type to use.
    pub backend: BackendType,
    /// Path to a local quantized model (GGUF format).
    pub local_model_path: Option<String>,
    /// Path to a tokenizer.json.
    pub local_tokenizer_path: Option<String>,
    /// Maximum tokens to generate.
    pub local_max_tokens: Option<usize>,
    /// Preferred device hint: "cpu", "cuda", or "metal".
    pub local_device: Option<String>,
    /// Device offload mode: "none", "up", "updown", or "experts" (default: "experts").
    pub local_offload: Option<String>,
    /// Maximum context length for local inference.
    pub local_context_length: Option<usize>,
    /// KV cache quantization mode: "f16", "bf16", "q8", "q4" (default: "q8").
    /// F16/BF16 provide maximum accuracy, Q8/Q4 reduce memory at cost of some accuracy.
    pub local_kv_cache_quant: Option<String>,
}

/// Options for a completion request.
#[derive(Debug, Clone, Default)]
pub struct CompletionOptions {
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Tool choice directive.
    pub tool_choice: Option<ToolChoice>,
    /// Extra headers to include (unused for local backend).
    pub extra_headers: Option<std::collections::HashMap<String, String>>,
}

/// Type alias for boxed stream of chunks.
pub type ChunkStream = Pin<Box<dyn Stream<Item = LlmResult<LlmChunk>> + Send>>;

/// Trait for LLM backends.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Complete a chat conversation.
    async fn complete(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
        options: &CompletionOptions,
    ) -> LlmResult<LlmChunk>;

    /// Complete a chat conversation with streaming.
    async fn complete_streaming(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
        options: &CompletionOptions,
    ) -> LlmResult<ChunkStream>;

    /// Count tokens in a conversation.
    async fn count_tokens(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
    ) -> LlmResult<u32>;
}
