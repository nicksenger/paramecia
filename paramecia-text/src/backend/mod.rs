//! LLM backend implementations.

mod controller;
mod factory;
mod local;

pub use controller::ControllerBackend;
pub use factory::{BackendFactory, BackendType};
pub use local::LocalBackend;

use async_trait::async_trait;

use crate::error::LlmResult;
use crate::types::{AvailableTool, EventStream, LlmMessage, ToolChoice};

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
    /// Device offload mode: "none", "down", "updown", or "experts" (default: "experts").
    pub local_offload: Option<String>,
    /// Maximum context length for local inference (default: 262144 / 256K).
    /// Set to 1048576 to enable YARN for 1M token context extension.
    pub context_length: Option<usize>,
    /// KV cache quantization mode: "f16", "bf16", "q8", "q4" (default: "q8").
    /// F16/BF16 provide maximum accuracy, Q8/Q4 reduce memory at cost of some accuracy.
    pub local_kv_cache_quant: Option<String>,
    /// Layer split proportions for multi-GPU layer parallelism.
    /// Format: "3,1" = 75% GPU 0, 25% GPU 1.
    pub local_layer_split: Option<String>,
    /// Disable loading CONTEXT.txt from the current working directory.
    pub local_disable_context: Option<bool>,
    /// Tool call format: "xml" (pure XML, default) or "json_in_xml" (JSON-in-XML).
    /// "xml" uses `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    /// "json_in_xml" uses `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
    pub tool_call_format: Option<String>,
    /// Include full conversation (system prompt, tool descriptions, CONTEXT.txt) in
    /// generation output files. Default: false (user/tool messages only).
    pub full_generation_output: Option<bool>,
    /// Enable output of top-k logits for subsequent distillation/tuning.
    /// Can be set via `PARAMECIA_TUNING_OUTPUT` env var.
    pub tuning_output: Option<bool>,
    /// Number of top-k token predictions to store per assistant token (default: 100).
    /// Can be set via `PARAMECIA_TUNING_TOP_K` env var.
    pub tuning_top_k: Option<usize>,
    /// Number of randomly sampled tail tokens to store for importance sampling (default: 20).
    /// Can be set via `PARAMECIA_TUNING_TAIL_SAMPLES` env var.
    pub tuning_tail_samples: Option<usize>,
}

/// Options for a completion request.
#[derive(Debug, Clone, Default)]
pub struct CompletionOptions {
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Tool choice directive.
    pub tool_choice: Option<ToolChoice>,
}

/// Trait for LLM backends.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Stream a completion as granular `Token`/`ToolCall`/`Done` events.
    async fn complete_streaming(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
        options: &CompletionOptions,
    ) -> LlmResult<EventStream>;

    /// Count tokens in a conversation.
    async fn count_tokens(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
    ) -> LlmResult<u32>;
}
