//! Local backend that runs a quantized Qwen3-Next model directly.
//!
//! This backend delegates to `paramecia-engine`'s `ModelEngine` for inference,
//! penalty application, sampling, and snapshot management.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use paramecia_engine::Device;
use paramecia_engine::{
    DeviceOffloadMode, KvCacheQuantization, ModelEngine, ModelEngineBuilder, Snapshot,
    TokenOutputStream, YarnConfig,
};
use tokenizers::Tokenizer;

/// Number of recent tokens to consider for repeat/presence penalty.
const REPEAT_LAST_N: usize = 128;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;

use crate::backend::{Backend, CompletionOptions, ModelConfig, ProviderConfig};
use crate::chat_template::{ChatTemplate, QWEN3_NEXT_CHAT_TEMPLATE};
use crate::error::{LlmError, LlmResult};
use crate::types::{
    AvailableTool, EventStream, FunctionCall, LlmMessage, LlmUsage, Role, StreamEvent,
    TokenTuningData, ToolCall,
};

/// Default maximum context length for inference.
/// The Qwen3-Next model natively supports 256K context.
///
/// Context length can be configured via `context_length` config or
/// `PARAMECIA_CONTEXT_LENGTH` env var. Default: 262144 (256K).
///
/// **YARN Context Extension**: Set context_length to 1048576 to enable YARN
/// for 1M token context. YARN (Yet Another RoPE extensioN) uses frequency-based
/// RoPE interpolation to extend the context window beyond the native 256K limit.
///
/// KV cache quantization can be configured via `local_kv_cache_quant` config or
/// `PARAMECIA_KV_CACHE_QUANT` env var. Options: "f16", "bf16", "q8", "q4" (default: Q8_0).
///
/// Memory scaling (approximate, with expert offload for 30B MoE):
///   - F16 KV-cache (maximum accuracy):
///     - 32K tokens: ~22 GB peak
///     - 64K tokens: ~30 GB peak
///     - 128K tokens: ~46 GB peak
///   - Q8 KV-cache (~2x memory reduction):
///     - 32K tokens: ~18 GB peak
///     - 64K tokens: ~22 GB peak
///     - 128K tokens: ~28 GB peak
///   - Q4 KV-cache (~4x memory reduction):
///     - 32K tokens: ~16 GB peak
///     - 64K tokens: ~18 GB peak
///     - 128K tokens: ~22 GB peak
const DEFAULT_CONTEXT_TOKENS: usize = 262144;

/// Cached prefix snapshot for avoiding recomputation of shared conversation context.
struct CachedPrefix {
    snapshot: Snapshot,
    tokens: Vec<u32>,
}

/// Local backend for running a quantized model on-device.
#[derive(Clone)]
pub struct LocalBackend {
    executor: Arc<Mutex<ModelEngine>>,
    tokenizer: Tokenizer,
    device: Device,
    max_tokens: usize,
    /// Maximum context length the model supports (used for YARN determination).
    max_context: usize,
    eos_token: Option<u32>,
    /// Chat template for formatting messages.
    chat_template: ChatTemplate,
    /// Cached prefix snapshot for avoiding recomputation of shared conversation context.
    cached_prefix: Arc<Mutex<Option<CachedPrefix>>>,
    /// Unique conversation ID for logging conversations to disk.
    conversation_id: String,
    /// Persistent tuning writer that accumulates data across all turns.
    /// Shared via Arc<Mutex> so it persists across clones and async boundaries.
    tuning_writer: Arc<Mutex<Option<crate::tuning::TuningWriter>>>,
    /// Tool call format style (QwenCoder XML or QwenVl JSON-in-XML).
    tool_call_style: crate::xml_tool_parser::ToolCallStyle,
    /// Optional context from CONTEXT.txt in the current working directory.
    /// This is parsed as ChatML and inserted directly into the conversation.
    context_messages: Option<Vec<LlmMessage>>,
    /// If true, include full conversation (system prompt, tool descriptions, CONTEXT.txt) in generation output.
    full_generation_output: bool,
    /// If true, system context (CONTEXT.txt, tools, system prompt) has already been
    /// processed as part of a loaded snapshot. Skip re-sending on next completion.
    snapshot_mode: bool,
}

impl LocalBackend {
    /// Create a new local backend using the provided configuration.
    ///
    /// The provider configuration must set `local_model_path` or the environment
    /// variable `PARAMECIA_MODEL_PATH`. The tokenizer is automatically downloaded
    /// from HuggingFace if not specified.
    pub fn new(provider: ProviderConfig) -> LlmResult<Self> {
        let model_path = provider
            .local_model_path
            .as_ref()
            .map(PathBuf::from)
            .or_else(|| std::env::var("PARAMECIA_MODEL_PATH").ok().map(PathBuf::from))
            .ok_or_else(|| {
                LlmError::InvalidConfig(
                    "local_model_path is required for the local backend (or set PARAMECIA_MODEL_PATH)".to_string(),
                )
            })?;

        // Parse offload mode from config or environment
        let offload_str = provider
            .local_offload
            .clone()
            .or_else(|| std::env::var("PARAMECIA_OFFLOAD").ok());
        let offload_mode = Self::parse_offload_mode(offload_str.as_deref());

        // Parse KV cache quantization mode from config or environment
        let kv_cache_str = provider
            .local_kv_cache_quant
            .clone()
            .or_else(|| std::env::var("PARAMECIA_KV_CACHE_QUANT").ok());
        let kv_cache_quant = Self::parse_kv_cache_quant(kv_cache_str.as_deref());

        // Parse context length from config or environment variable
        let max_context = provider
            .context_length
            .or_else(|| {
                std::env::var("PARAMECIA_CONTEXT_LENGTH")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(DEFAULT_CONTEXT_TOKENS);

        // Parse layer split from config or environment
        let layer_split_str = provider
            .local_layer_split
            .clone()
            .or_else(|| std::env::var("PARAMECIA_LAYER_SPLIT").ok());

        tracing::info!("Loading Qwen3-Next model from {:?}", model_path);
        tracing::info!("Offload mode: {:?}", offload_mode);
        tracing::info!("KV-cache quantization: {:?}", kv_cache_quant);
        tracing::info!("Max context length: {} tokens", max_context);
        if let Some(ref split) = layer_split_str {
            tracing::info!("Layer split: {}", split);
        }

        // Extract chat template from GGUF before loading model
        let chat_template = Self::extract_chat_template_from_gguf(&model_path)?;
        tracing::info!(
            "Chat template: {} chars",
            chat_template.template_string().len()
        );

        // Enable YARN for 1M context if max_context exceeds native 256K
        const NATIVE_CONTEXT: usize = 262_144; // 256K
        let yarn_config = if max_context > NATIVE_CONTEXT {
            let yarn = YarnConfig::for_1m_context(NATIVE_CONTEXT);
            tracing::info!(
                "YARN enabled: {}K -> 1M tokens (scale factor: {:.1}x)",
                NATIVE_CONTEXT / 1024,
                yarn.scale_factor()
            );
            Some(yarn)
        } else {
            tracing::info!(
                "YARN disabled (context {} <= native {})",
                max_context,
                NATIVE_CONTEXT
            );
            None
        };

        // Parse sampling parameters from environment / ModelConfig defaults
        let temperature: f64 = std::env::var("PARAMECIA_TEMPERATURE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);
        let top_p: f64 = std::env::var("PARAMECIA_TOP_P")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.8);
        let repeat_penalty: f32 = std::env::var("PARAMECIA_REPEAT_PENALTY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let presence_penalty: f32 = std::env::var("PARAMECIA_PRESENCE_PENALTY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        // Parse full generation output configuration
        let full_generation_output = provider
            .full_generation_output
            .or_else(|| {
                std::env::var("PARAMECIA_FULL_GENERATION_OUTPUT")
                    .ok()
                    .map(|v| v == "1" || v.to_lowercase() == "true")
            })
            .unwrap_or(false);

        // Parse tuning configuration
        let tuning_enabled = provider
            .tuning_output
            .or_else(|| {
                std::env::var("PARAMECIA_TUNING_OUTPUT")
                    .ok()
                    .map(|v| v == "1" || v.to_lowercase() == "true")
            })
            .unwrap_or(false);

        let tuning_config = crate::tuning::TuningConfig::from_options(
            provider.tuning_top_k,
            provider.tuning_tail_samples,
        );

        // Build executor via ModelEngineBuilder
        let disable_prefetch = std::env::var("PARAMECIA_NO_PREFETCH")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let mut builder = ModelEngineBuilder::new(&model_path)
            .offload_mode(offload_mode)
            .kv_cache_quant(kv_cache_quant)
            .temperature(temperature)
            .top_p(top_p)
            .repeat_penalty(repeat_penalty)
            .presence_penalty(presence_penalty)
            .penalty_last_n(REPEAT_LAST_N)
            .prefetch(!disable_prefetch);

        if tuning_enabled {
            builder = builder
                .top_k(tuning_config.top_k)
                .tail_samples(tuning_config.tail_samples);
        }
        if let Some(yarn) = yarn_config {
            builder = builder.yarn_config(yarn);
        }
        if let Some(ref split) = layer_split_str {
            builder = builder.layer_split(split);
        }
        if let Some(ref tok) = provider.local_tokenizer_path {
            builder = builder.tokenizer_path(tok);
        }
        if let Some(ref hint) = provider.local_device {
            builder = builder.device(Self::select_device(Some(hint))?);
        }

        let executor = builder
            .build()
            .map_err(|e| LlmError::ModelError(e.to_string()))?;

        tracing::info!("Local Qwen3-Next model loaded");

        let tokenizer = executor.tokenizer().clone();
        let device = executor.device().clone();
        let eos_token = Self::detect_eos_token(&tokenizer);
        let vocab_size = tokenizer.get_vocab_size(true);

        // Create persistent tuning writer if enabled (with expert tracking)
        let tuning_writer = if tuning_enabled {
            let n_layers = executor.num_layers();
            let n_experts_per_tok = executor.num_experts_per_token();
            tracing::info!(
                "Tuning output enabled: top_k={}, tail_samples={}, layers={}, experts_per_tok={}",
                tuning_config.top_k,
                tuning_config.tail_samples,
                n_layers,
                n_experts_per_tok
            );
            Arc::new(Mutex::new(Some(
                crate::tuning::TuningWriter::new_with_experts(
                    tuning_config,
                    vocab_size,
                    n_layers,
                    n_experts_per_tok,
                ),
            )))
        } else {
            Arc::new(Mutex::new(None))
        };

        // Load CONTEXT.txt from current working directory if it exists
        let context_messages = if provider.local_disable_context.unwrap_or(false) {
            tracing::info!("Skipping CONTEXT.txt loading (local_disable_context=true)");
            None
        } else {
            Self::load_context_file()
        };

        // Parse tool call format from config
        let tool_call_style = crate::xml_tool_parser::ToolCallStyle::from_config(
            provider.tool_call_format.as_deref(),
        );
        tracing::info!("Tool call format: {:?}", tool_call_style);

        // Override chat template based on tool call format
        let chat_template = match tool_call_style {
            crate::xml_tool_parser::ToolCallStyle::QwenCoder => {
                tracing::info!("Using Qwen-Coder XML chat template for tool calls");
                ChatTemplate::new(crate::chat_template::QWEN_CODER_CHAT_TEMPLATE.to_string())
            }
            _ => chat_template,
        };

        Ok(Self {
            executor: Arc::new(Mutex::new(executor)),
            tokenizer,
            device,
            max_tokens: provider.local_max_tokens.unwrap_or(2048),
            max_context,
            eos_token,
            chat_template,
            cached_prefix: Arc::new(Mutex::new(None)),
            conversation_id: uuid::Uuid::new_v4().to_string(),
            tuning_writer,
            tool_call_style,
            context_messages,
            full_generation_output,
            snapshot_mode: false,
        })
    }

    /// Get the chat template used by this backend.
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Get the conversation ID for this backend instance.
    #[must_use]
    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    /// Get the generations directory path.
    fn generations_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".paramecia")
            .join("generations")
    }

    /// Get the generation file path for this session (text format).
    fn generation_file_path(&self) -> PathBuf {
        Self::generations_dir().join(format!("{}.txt", self.conversation_id))
    }

    /// Get the tuning output file path for this session (binary format).
    fn tuning_file_path(&self) -> PathBuf {
        Self::generations_dir().join(format!("{}.bin", self.conversation_id))
    }

    /// Parse ChatML format into a vector of LlmMessages.
    ///
    /// ChatML format is:
    /// ```text
    /// <|im_start|>role
    /// content
    /// <|im_end|>
    /// ```
    fn parse_chatml(content: &str) -> Result<Vec<LlmMessage>, String> {
        let mut messages = Vec::new();
        let trimmed = content.trim();

        if trimmed.is_empty() {
            return Ok(messages);
        }

        // Split by <|im_start|> to find message boundaries
        let parts: Vec<&str> = trimmed.split("<|im_start|>").collect();

        for part in parts {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            // Find the end marker
            let end_pos = part.find("<|im_end|>");
            let (message_content, _) = if let Some(pos) = end_pos {
                (&part[..pos], &part[pos..])
            } else {
                // No end marker found - treat entire part as the message
                (part, "")
            };

            // Split role from content (first line is role)
            let mut lines = message_content.lines();
            let role_line = match lines.next() {
                Some(line) => line.trim(),
                None => continue,
            };

            // Parse role
            let role = match role_line.to_lowercase().as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => {
                    return Err(format!("Invalid role in ChatML: '{}'", role_line));
                }
            };

            // Collect remaining lines as content
            let content: Vec<&str> = lines.collect();
            let content_str = content.join("\n").trim().to_string();

            messages.push(LlmMessage {
                role,
                content: if content_str.is_empty() {
                    None
                } else {
                    Some(content_str)
                },
                tool_calls: None,
                name: None,
                tool_call_id: None,
            });
        }

        Ok(messages)
    }

    /// Load CONTEXT.txt from the current working directory if it exists.
    /// The file must be in ChatML format.
    fn load_context_file() -> Option<Vec<LlmMessage>> {
        let context_path = std::env::current_dir().ok()?.join("CONTEXT.txt");

        if context_path.exists() {
            match std::fs::read_to_string(&context_path) {
                Ok(content) => {
                    let trimmed = content.trim();
                    if trimmed.is_empty() {
                        tracing::debug!("CONTEXT.txt exists but is empty, skipping");
                        return None;
                    }

                    match Self::parse_chatml(trimmed) {
                        Ok(messages) => {
                            if messages.is_empty() {
                                tracing::debug!(
                                    "CONTEXT.txt parsed but contains no messages, skipping"
                                );
                                None
                            } else {
                                tracing::info!(
                                    "Loaded CONTEXT.txt ({} messages, {} bytes) from {}",
                                    messages.len(),
                                    trimmed.len(),
                                    context_path.display()
                                );
                                Some(messages)
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to parse CONTEXT.txt as ChatML: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to read CONTEXT.txt: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }

    /// Insert context messages from CONTEXT.txt after the system prompt.
    /// The first message is assumed to be the system prompt; context is placed
    /// immediately after it, before any user/assistant turns.
    fn insert_context_messages(&self, messages: &[LlmMessage]) -> Vec<LlmMessage> {
        // Skip if we loaded from snapshot - context already in KV cache
        if self.snapshot_mode {
            return messages.to_vec();
        }

        if let Some(ref context_messages) = self.context_messages {
            let mut new_messages = Vec::with_capacity(messages.len() + context_messages.len());
            // System prompt first
            if let Some(system) = messages.first() {
                new_messages.push(system.clone());
            }
            // Then CONTEXT.txt content
            new_messages.extend_from_slice(context_messages);
            // Then remaining conversation messages
            if messages.len() > 1 {
                new_messages.extend_from_slice(&messages[1..]);
            }
            new_messages
        } else {
            messages.to_vec()
        }
    }

    /// Format a message in ChatML format.
    fn format_message_chatml(
        msg: &LlmMessage,
        tool_call_style: crate::xml_tool_parser::ToolCallStyle,
    ) -> String {
        let role = msg.role.to_string();
        let mut content = msg.content.clone().unwrap_or_default();

        // For assistant messages with tool calls, append the tool calls in the configured format
        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                match tool_call_style {
                    crate::xml_tool_parser::ToolCallStyle::Preserve => {
                        // Preserve original format: use raw_text if available, otherwise use JSON-in-XML
                        if let Some(ref raw_text) = tc.raw_text {
                            content.push('\n');
                            content.push_str(raw_text);
                        } else {
                            // Fallback to JSON-in-XML if no raw text is available
                            let name = tc.function.name.as_deref().unwrap_or("unknown");
                            let args = tc.function.arguments.as_deref().unwrap_or("{}");
                            content.push_str(&format!(
                                "\n<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"
                            ));
                        }
                    }
                    crate::xml_tool_parser::ToolCallStyle::QwenCoder => {
                        // Pure XML format
                        let name = tc.function.name.as_deref().unwrap_or("unknown");
                        let args = tc.function.arguments.as_deref().unwrap_or("{}");
                        content.push_str("\n<tool_call>\n<function=");
                        content.push_str(name);
                        content.push_str(">\n");
                        // Parse args JSON and render as individual parameters
                        if let Ok(args_obj) = serde_json::from_str::<serde_json::Value>(args)
                            && let Some(obj) = args_obj.as_object()
                        {
                            for (key, value) in obj {
                                content.push_str("<parameter=");
                                content.push_str(key);
                                content.push_str(">\n");
                                match value {
                                    serde_json::Value::String(s) => content.push_str(s),
                                    other => content.push_str(&other.to_string()),
                                }
                                content.push_str("\n</parameter>\n");
                            }
                        }
                        content.push_str("</function>\n</tool_call>");
                    }
                    _ => {
                        // JSON-in-XML format (default for QwenVl and others)
                        let name = tc.function.name.as_deref().unwrap_or("unknown");
                        let args = tc.function.arguments.as_deref().unwrap_or("{}");
                        content.push_str(&format!(
                            "\n<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"
                        ));
                    }
                }
            }
        }

        // For tool responses, include the tool name and call ID
        if msg.role == Role::Tool {
            let name = msg.name.as_deref().unwrap_or("unknown");
            let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
            format!("<|im_start|>{role}[{name}:{tool_call_id}]\n{content}<|im_end|>")
        } else {
            format!("<|im_start|>{role}\n{content}<|im_end|>")
        }
    }

    /// Format messages as a ChatML conversation. By default, excludes system messages.
    /// When `full_output` is true, includes system messages (system prompt, tool
    /// descriptions, CONTEXT.txt content, etc.).
    fn format_conversation_chatml(
        messages: &[LlmMessage],
        tool_call_style: crate::xml_tool_parser::ToolCallStyle,
        full_output: bool,
    ) -> String {
        messages
            .iter()
            .filter(|msg| full_output || msg.role != Role::System)
            .map(|msg| Self::format_message_chatml(msg, tool_call_style))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Build a filtered prompt for tuning output.
    ///
    /// This returns only the NON-ASSISTANT messages formatted for tokenization.
    /// Assistant messages are excluded because they are captured separately during generation
    /// with their logits. Including them here would cause token count mismatches since the
    /// formatted version includes ChatML wrappers that aren't in the generated tokens.
    ///
    /// When `full_output` is true, system messages (system prompt, tool descriptions) are
    /// included alongside user and tool messages.
    fn build_filtered_tuning_prompt(messages: &[LlmMessage], full_output: bool) -> String {
        let style = crate::xml_tool_parser::ToolCallStyle::QwenCoder;

        let filtered: Vec<String> = messages
            .iter()
            .filter(|msg| {
                if msg.role == Role::Assistant {
                    return false;
                }
                if !full_output && msg.role == Role::System {
                    return false;
                }
                true
            })
            .map(|msg| Self::format_message_chatml(msg, style))
            .collect();
        filtered.join("\n")
    }

    /// Clear the prefix cache.
    ///
    /// Call this when starting a new conversation to ensure the model doesn't
    /// try to reuse cached context from a previous conversation.
    pub async fn clear_prefix_cache(&self) {
        let mut cache = self.cached_prefix.lock().await;
        if let Some(old) = cache.take() {
            let exec = self.executor.lock().await;
            let _ = exec.drop_snapshot(old.snapshot.id).await;
        }
    }

    /// Get the current prefix cache length (number of cached tokens).
    ///
    /// Returns 0 if no prefix is cached.
    pub async fn prefix_cache_len(&self) -> usize {
        let cache = self.cached_prefix.lock().await;
        cache.as_ref().map(|c| c.tokens.len()).unwrap_or(0)
    }

    fn select_device(hint: Option<&str>) -> LlmResult<Device> {
        if let Some(requested) = hint {
            match requested {
                "cuda" => {
                    if let Ok(device) = Device::cuda_if_available(0) {
                        return Ok(device);
                    }
                }
                "vulkan" => {
                    if let Ok(device) = Device::new_vulkan(0) {
                        return Ok(device);
                    }
                }
                "metal" => {
                    if let Ok(device) = Device::new_metal(0) {
                        return Ok(device);
                    }
                }
                "cpu" => return Ok(Device::Cpu),
                _ => {}
            }
        }

        if let Ok(device) = Device::cuda_if_available(0)
            && matches!(device, Device::Cuda(_))
        {
            return Ok(device);
        }
        if let Ok(device) = Device::new_vulkan(0) {
            return Ok(device);
        }
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }

        Ok(Device::Cpu)
    }

    /// Parse offload mode from string. Defaults to "experts".
    fn parse_offload_mode(hint: Option<&str>) -> DeviceOffloadMode {
        match hint {
            Some("none") => DeviceOffloadMode::FullGpu,
            Some("down") => DeviceOffloadMode::DownProjectionsOnCpu,
            Some("updown") => DeviceOffloadMode::UpDownProjectionsOnCpu,
            Some("experts") | None => DeviceOffloadMode::ExpertsOnCpu,
            Some(other) => {
                tracing::warn!("Unknown offload mode '{}', using 'experts'", other);
                DeviceOffloadMode::ExpertsOnCpu
            }
        }
    }

    /// Parse KV cache quantization mode from string. Defaults to "q8_0".
    ///
    /// Supported values:
    /// - "f16" / "fp16": Store KV cache as f16 (maximum accuracy)
    /// - "bf16" / "bfloat16": Store KV cache as bf16
    /// - "q8" / "q8_0": Quantize to Q8_0 (8-bit, ~2x memory reduction)
    /// - "q4" / "q4k" / "q4_k": Quantize to Q4K (4-bit, ~4x memory reduction)
    fn parse_kv_cache_quant(hint: Option<&str>) -> KvCacheQuantization {
        match hint {
            None => KvCacheQuantization::Q8_0,
            Some(s) => KvCacheQuantization::from_str(s).unwrap_or_else(|| {
                tracing::warn!("Unknown KV cache quantization '{}', using 'q8_0'", s);
                KvCacheQuantization::Q8_0
            }),
        }
    }

    fn detect_eos_token(tokenizer: &Tokenizer) -> Option<u32> {
        let vocab = tokenizer.get_vocab(true);

        // Try common EOS token names in order of preference
        let candidates = [
            "<|im_end|>",    // Qwen3 chat template end token
            "<|endoftext|>", // Qwen EOS token
            "</s>",          // Common EOS token
            "",              // BOS/EOS control token
        ];

        for candidate in candidates {
            if let Some(&id) = vocab.get(candidate) {
                tracing::debug!("Using EOS token '{}' with id {}", candidate, id);
                return Some(id);
            }
        }

        // Fallback: Qwen3 hardcoded EOS token
        tracing::debug!("Using fallback Qwen3 EOS token id 151643");
        Some(151643)
    }

    fn extract_chat_template_from_gguf(model_path: &std::path::Path) -> LlmResult<ChatTemplate> {
        use paramecia_engine::gguf_file;

        let mut file = std::fs::File::open(model_path)
            .map_err(|e| LlmError::InvalidConfig(format!("Failed to open model file: {e}")))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| LlmError::InvalidConfig(format!("Failed to read GGUF content: {e}")))?;

        // Try to extract chat template from metadata
        let template_str = crate::chat_template::extract_chat_template_from_gguf(&content.metadata);

        match template_str {
            Some(template) => {
                tracing::info!("Using chat template from GGUF file");
                Ok(ChatTemplate::new(template))
            }
            None => {
                tracing::warn!(
                    "No chat template found in GGUF metadata, using default Qwen3-Next template"
                );
                Ok(ChatTemplate::new(QWEN3_NEXT_CHAT_TEMPLATE.to_string()))
            }
        }
    }

    /// Format a tool definition in Qwen3-Next native XML format.
    fn format_tool_definition(prompt: &mut String, tool: &AvailableTool) {
        prompt.push_str("## ");
        prompt.push_str(&tool.function.name);
        prompt.push_str("\n\n");
        prompt.push_str(&tool.function.description);
        prompt.push_str("\n\n");

        // Format parameters from JSON Schema
        let params = &tool.function.parameters;
        if let Some(properties) = params.get("properties").and_then(|p| p.as_object()) {
            let required: Vec<&str> = params
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();

            prompt.push_str("**Parameters:**\n\n");

            for (name, schema) in properties {
                let param_type = schema
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("string");
                let description = schema
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("");
                let is_required = required.contains(&name.as_str());

                prompt.push_str("- `");
                prompt.push_str(name);
                prompt.push_str("` (");
                prompt.push_str(param_type);
                if is_required {
                    prompt.push_str(", required");
                }
                prompt.push_str("): ");
                prompt.push_str(description);
                prompt.push('\n');
            }
            prompt.push('\n');
        }

        // Add example usage with hybrid XML-JSON format
        prompt.push_str("**Usage:**\n\n");
        prompt.push_str("<tool_call>\n{\"name\": \"");
        prompt.push_str(&tool.function.name);
        prompt.push_str("\", \"arguments\": {");

        // Add example parameters
        if let Some(properties) = params.get("properties").and_then(|p| p.as_object()) {
            let mut first = true;
            for (name, _) in properties.iter().take(2) {
                if !first {
                    prompt.push_str(", ");
                }
                prompt.push('"');
                prompt.push_str(name);
                prompt.push_str("\": \"value\"");
                first = false;
            }
        }

        prompt.push_str("}}\n</tool_call>\n\n");
    }

    /// Build prompt using the chat template.
    ///
    /// This applies the Jinja2 chat template from the GGUF file to format messages.
    fn build_prompt(
        chat_template: &ChatTemplate,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
    ) -> String {
        match chat_template.apply(messages, tools) {
            Ok(prompt) => prompt,
            Err(e) => {
                tracing::warn!("Failed to apply chat template: {}, using fallback", e);
                Self::build_prompt_fallback(messages, tools)
            }
        }
    }

    /// Fallback prompt builder when chat template fails.
    fn build_prompt_fallback(messages: &[LlmMessage], tools: Option<&[AvailableTool]>) -> String {
        // Build prompt using Qwen3-Next-Thinking chat format
        let mut prompt = String::new();

        let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);

        // Check if first message is system
        let first_is_system = messages
            .first()
            .map(|m| m.role == Role::System)
            .unwrap_or(false);

        if has_tools {
            // When tools are present, system message includes tool definitions
            prompt.push_str("<|im_start|>system\n");

            // Add system message content if present
            if first_is_system && let Some(content) = &messages[0].content {
                prompt.push_str(content);
                prompt.push_str("\n\n");
            }

            // Add tool definitions in Qwen3-Next native XML format
            prompt.push_str("# Tools\n\nYou have access to the following tools:\n\n");

            if let Some(tools) = tools {
                for tool in tools {
                    Self::format_tool_definition(&mut prompt, tool);
                }
            }

            prompt.push_str(
                "\n## Tool Call Format\n\nTo call a tool, use a JSON object within <tool_call></tool_call> tags:\n\n",
            );
            prompt.push_str("<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param_name\": \"value\"}}\n</tool_call>\n\n");
            prompt.push_str(
                "You may call multiple tools in sequence. Always wait for tool results before proceeding.\n",
            );
            prompt.push_str("<|im_end|>\n");
        } else if first_is_system {
            // No tools, but has system message
            prompt.push_str("<|im_start|>system\n");
            if let Some(content) = &messages[0].content {
                prompt.push_str(content);
            }
            prompt.push_str("<|im_end|>\n");
        }

        // Add conversation messages (skip first system message if already handled)
        let start_idx = if first_is_system { 1 } else { 0 };
        let msg_slice = &messages[start_idx..];

        let mut i = 0;
        while i < msg_slice.len() {
            let message = &msg_slice[i];
            match message.role {
                Role::System => {
                    // System message in middle of conversation
                    prompt.push_str("<|im_start|>system\n");
                    if let Some(content) = &message.content {
                        prompt.push_str(content);
                    }
                    prompt.push_str("<|im_end|>\n");
                    i += 1;
                }
                Role::User => {
                    prompt.push_str("<|im_start|>user\n");
                    if let Some(content) = &message.content {
                        prompt.push_str(content);
                    }
                    prompt.push_str("<|im_end|>\n");
                    i += 1;
                }
                Role::Assistant => {
                    prompt.push_str("<|im_start|>assistant\n");
                    if let Some(content) = &message.content {
                        prompt.push_str(content);
                    }
                    // Include tool calls if present
                    if let Some(tool_calls) = &message.tool_calls {
                        for (tc_idx, tc) in tool_calls.iter().enumerate() {
                            if (tc_idx == 0 && message.content.is_some()) || tc_idx > 0 {
                                prompt.push('\n');
                            }
                            prompt.push_str("<tool_call>\n{\"name\": \"");
                            if let Some(name) = &tc.function.name {
                                prompt.push_str(name);
                            }
                            prompt.push_str("\", \"arguments\": ");
                            if let Some(args) = &tc.function.arguments {
                                prompt.push_str(args);
                            } else {
                                prompt.push_str("{}");
                            }
                            prompt.push_str("}\n</tool_call>");
                        }
                    }
                    prompt.push_str("<|im_end|>\n");
                    i += 1;
                }
                Role::Tool => {
                    // Group consecutive tool messages together under one user block
                    prompt.push_str("<|im_start|>user");

                    // Process all consecutive tool messages
                    while i < msg_slice.len() && msg_slice[i].role == Role::Tool {
                        prompt.push_str("\n<tool_response>\n");
                        if let Some(content) = &msg_slice[i].content {
                            prompt.push_str(content);
                        }
                        prompt.push_str("\n</tool_response>");
                        i += 1;
                    }

                    prompt.push_str("<|im_end|>\n");
                }
            }
        }

        // Start assistant response
        prompt.push_str("<|im_start|>assistant\n");

        prompt
    }

    fn truncate_tokens(prompt_tokens: &[u32], max_new: usize, max_context: usize) -> Vec<u32> {
        // Ensure we keep at least half the context for the prompt
        let min_prompt_tokens = max_context / 2;

        // Calculate the effective max generation length we can support
        let effective_max_new = max_new.min(max_context.saturating_sub(min_prompt_tokens));

        let total_needed = prompt_tokens.len() + effective_max_new;
        if total_needed <= max_context {
            return prompt_tokens.to_vec();
        }

        // Calculate how many prompt tokens we can keep
        let max_prompt_tokens = max_context.saturating_sub(effective_max_new);
        let max_prompt_tokens = max_prompt_tokens.max(min_prompt_tokens);

        if prompt_tokens.len() <= max_prompt_tokens {
            return prompt_tokens.to_vec();
        }

        // Truncate from the beginning (keep the most recent context)
        let skip = prompt_tokens.len() - max_prompt_tokens;
        tracing::warn!(
            "Truncating prompt from {} to {} tokens to fit context window (max_context={})",
            prompt_tokens.len(),
            max_prompt_tokens,
            max_context
        );
        prompt_tokens.iter().skip(skip).copied().collect::<Vec<_>>()
    }

    /// Compute the length of the common prefix between two token sequences.
    fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    /// Expose the underlying device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Save current model state to snapshot file via executor.
    pub async fn save_snapshot<P: AsRef<std::path::Path>>(&self, path: P) -> LlmResult<PathBuf> {
        let exec = self.executor.lock().await;

        // Ensure snapshots directory exists
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|e| LlmError::ModelError(format!("Failed to get home directory: {}", e)))?;
        let snapshots_dir = std::path::PathBuf::from(home)
            .join(".paramecia")
            .join("snapshots");

        std::fs::create_dir_all(&snapshots_dir).map_err(|e| {
            LlmError::ModelError(format!("Failed to create snapshots directory: {}", e))
        })?;

        let snapshot = exec
            .take_snapshot()
            .await
            .map_err(|e| LlmError::ModelError(e.to_string()))?;
        let (_persisted, persisted_path) = exec
            .persist_snapshot(snapshot.id)
            .await
            .map_err(|e| LlmError::ModelError(e.to_string()))?;

        // Move to canonical location
        let file_name = path
            .as_ref()
            .file_name()
            .ok_or_else(|| LlmError::ModelError("Invalid snapshot filename".to_string()))?;
        let full_path = snapshots_dir.join(file_name);
        std::fs::rename(&persisted_path, &full_path)
            .map_err(|e| LlmError::ModelError(format!("Failed to move snapshot: {}", e)))?;

        Ok(full_path)
    }

    /// Load snapshot and restore model state.
    ///
    /// Restores KV cache, DeltaNet state, and returns token history.
    /// Sets snapshot_mode flag to skip system context/tools on next completion.
    pub async fn load_snapshot<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> LlmResult<(Vec<u32>, usize)> {
        // Resolve path (check if absolute, otherwise look in snapshots dir)
        let snapshot_path = if path.as_ref().is_absolute() {
            path.as_ref().to_path_buf()
        } else {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map_err(|e| {
                    LlmError::ModelError(format!("Failed to get home directory: {}", e))
                })?;
            std::path::PathBuf::from(home)
                .join(".paramecia")
                .join("snapshots")
                .join(path.as_ref())
        };

        let exec = self.executor.lock().await;
        exec.load_persisted_snapshot(snapshot_path.clone())
            .await
            .map_err(|e| LlmError::ModelError(format!("Failed to load snapshot: {}", e)))?;

        let tokens = exec
            .tokens()
            .await
            .map_err(|e| LlmError::ModelError(e.to_string()))?;
        let state_position = exec
            .state_position()
            .await
            .map_err(|e| LlmError::ModelError(e.to_string()))?;

        // Set snapshot mode to skip system context/tools on next completion
        self.snapshot_mode = true;

        Ok((tokens, state_position))
    }
}

/// A stream wrapper that saves the generation (`.txt` conversation + `.bin` tuning)
/// when the stream completes or is dropped.
struct GenerationSavingStream {
    inner: std::pin::Pin<Box<dyn futures::Stream<Item = LlmResult<StreamEvent>> + Send>>,
    input_messages: Vec<LlmMessage>,
    accumulated_content: std::sync::Arc<std::sync::Mutex<String>>,
    accumulated_tool_calls: std::sync::Arc<std::sync::Mutex<Vec<ToolCall>>>,
    generation_file_path: PathBuf,
    tuning_file_path: PathBuf,
    generations_dir: PathBuf,
    tool_call_style: crate::xml_tool_parser::ToolCallStyle,
    /// Whether to include full conversation (system prompt, etc.) in output.
    full_generation_output: bool,
    /// Shared tuning writer (from LocalBackend) — used to write the `.bin` on drop.
    tuning_writer: Arc<Mutex<Option<crate::tuning::TuningWriter>>>,
    saved: bool,
}

impl GenerationSavingStream {
    fn save_generation_sync(&self) {
        let content = self
            .accumulated_content
            .lock()
            .map(|c| c.clone())
            .unwrap_or_default();
        let tool_calls = self
            .accumulated_tool_calls
            .lock()
            .map(|t| t.clone())
            .ok()
            .filter(|t| !t.is_empty());

        tracing::info!(
            "GenerationSavingStream: saving generation with {} chars of content",
            content.len()
        );

        let response = LlmMessage {
            role: Role::Assistant,
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls,
            name: None,
            tool_call_id: None,
        };

        let mut all_messages = self.input_messages.clone();
        all_messages.push(response);

        let formatted = LocalBackend::format_conversation_chatml(
            &all_messages,
            self.tool_call_style,
            self.full_generation_output,
        );

        // Don't save empty generations
        if formatted.is_empty() {
            return;
        }

        // Use blocking IO since we're in Drop
        if let Err(e) = std::fs::create_dir_all(&self.generations_dir) {
            tracing::error!(
                "Failed to create generations directory {}: {e}",
                self.generations_dir.display()
            );
            return;
        }

        if let Err(e) = std::fs::write(&self.generation_file_path, &formatted) {
            tracing::error!(
                "Failed to write generation file {}: {e}",
                self.generation_file_path.display()
            );
        } else {
            tracing::info!(
                "Saved generation ({} bytes) to {}",
                formatted.len(),
                self.generation_file_path.display()
            );
        }

        // Write tuning data from the tuning writer (if enabled)
        // The tuning writer was already populated by generate_streaming_task.
        if let Ok(writer_guard) = self.tuning_writer.try_lock()
            && let Some(ref writer) = *writer_guard
            && writer.n_assistant_tokens() > 0
        {
            if let Some(parent) = self.tuning_file_path.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                tracing::error!(
                    "Failed to create tuning output directory {}: {e}",
                    parent.display()
                );
                return;
            }
            if let Err(e) = writer.write_to_file(&self.tuning_file_path) {
                tracing::error!(
                    "Failed to write tuning file {}: {e}",
                    self.tuning_file_path.display()
                );
            } else {
                tracing::info!(
                    "Saved tuning data ({} tokens, {} assistant) to {}",
                    writer.n_tokens(),
                    writer.n_assistant_tokens(),
                    self.tuning_file_path.display()
                );
            }
        }
    }
}

impl Drop for GenerationSavingStream {
    fn drop(&mut self) {
        tracing::info!("GenerationSavingStream::drop called, saved={}", self.saved);
        if !self.saved {
            self.save_generation_sync();
            self.saved = true;
        }
    }
}

impl futures::Stream for GenerationSavingStream {
    type Item = LlmResult<StreamEvent>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

#[async_trait]
impl Backend for LocalBackend {
    async fn complete_streaming(
        &self,
        _model_config: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
        _options: &CompletionOptions,
    ) -> LlmResult<EventStream> {
        // Prepend context from CONTEXT.txt if available
        let messages_with_context = self.insert_context_messages(messages);
        let prompt = Self::build_prompt(&self.chat_template, &messages_with_context, tools);
        let prompt_char_len = prompt.len();
        tracing::debug!(
            "LocalBackend::complete_streaming prompt ({} chars, {} messages, {} tools): {}",
            prompt_char_len,
            messages.len(),
            tools.map(|t| t.len()).unwrap_or(0),
            if prompt_char_len > 500 {
                &prompt[..500]
            } else {
                &prompt
            }
        );
        let encoded = self
            .tokenizer
            .encode(prompt, true)
            .map_err(LlmError::from)?;
        let mut tokens = encoded.get_ids().to_vec();
        tracing::info!(
            "LocalBackend: prompt={} chars, {} tokens, {} tools, max_context={}",
            prompt_char_len,
            tokens.len(),
            tools.map(|t| t.len()).unwrap_or(0),
            self.max_context
        );
        let prompt_tokens = tokens.len() as u32;
        let max_new_tokens = self.max_tokens;
        tokens = Self::truncate_tokens(&tokens, max_new_tokens, self.max_context);

        // Create a channel for streaming events
        let (tx, rx) = tokio::sync::mpsc::channel::<LlmResult<StreamEvent>>(32);

        // Clone what we need for the spawned task
        let executor = Arc::clone(&self.executor);
        let cached_prefix = Arc::clone(&self.cached_prefix);
        let tokenizer = self.tokenizer.clone();
        let eos_token = self.eos_token;
        let tool_call_style = self.tool_call_style;

        // Clone the shared tuning writer for the streaming task
        let tuning_writer = Arc::clone(&self.tuning_writer);

        // Build filtered prompt tokens for tuning output (matches text output filtering)
        let tuning_enabled = self.tuning_writer.lock().await.is_some();
        let tuning_prompt_tokens: Vec<u32> = if tuning_enabled {
            let tuning_source = if self.full_generation_output {
                &messages_with_context
            } else {
                messages
            };
            let filtered_prompt =
                Self::build_filtered_tuning_prompt(tuning_source, self.full_generation_output);
            if filtered_prompt.is_empty() {
                Vec::new()
            } else {
                self.tokenizer
                    .encode(filtered_prompt, true)
                    .map(|e| e.get_ids().to_vec())
                    .unwrap_or_default()
            }
        } else {
            Vec::new()
        };

        // Capture data needed for saving the conversation.
        // When full_generation_output is enabled, include the CONTEXT.txt messages.
        let input_messages = if self.full_generation_output {
            messages_with_context.clone()
        } else {
            messages.to_vec()
        };
        let generations_dir = Self::generations_dir();
        let generation_file_path = self.generation_file_path();

        // Track accumulated response content for saving
        let accumulated_content = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let accumulated_tool_calls =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::<ToolCall>::new()));

        let content_clone = accumulated_content.clone();
        let tool_calls_clone = accumulated_tool_calls.clone();

        // Spawn a task for generation
        tokio::spawn(async move {
            let result = LocalBackend::generate_streaming_task(
                executor,
                tokenizer,
                tokens,
                prompt_tokens,
                max_new_tokens,
                eos_token,
                tx.clone(),
                tuning_writer,
                tuning_prompt_tokens,
                tool_call_style,
                cached_prefix,
            )
            .await;

            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            }
        });

        // Wrap the stream to accumulate content for saving
        let inner_stream = ReceiverStream::new(rx);
        let accumulated_tuning =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::<TokenTuningData>::new()));
        let tuning_clone = accumulated_tuning.clone();
        let tuning_writer_for_saving = Arc::clone(&self.tuning_writer);
        let tuning_file_path_for_saving = self.tuning_file_path();

        let accumulating_stream = inner_stream.map(move |result| {
            if let Ok(ref event) = result {
                match event {
                    StreamEvent::ContextTokens { .. } => {}
                    StreamEvent::Token {
                        content, tuning, ..
                    } => {
                        if let Ok(mut acc) = content_clone.lock() {
                            acc.push_str(content);
                        }
                        if let Some(td) = tuning
                            && let Ok(mut acc) = tuning_clone.lock()
                        {
                            acc.push(td.clone());
                        }
                    }
                    StreamEvent::ToolCall {
                        id,
                        name,
                        arguments,
                        raw_text,
                        tuning,
                        ..
                    } => {
                        if let Ok(mut acc) = tool_calls_clone.lock() {
                            let idx = acc.len();
                            acc.push(ToolCall {
                                id: Some(id.clone()),
                                index: Some(idx),
                                function: FunctionCall {
                                    name: Some(name.clone()),
                                    arguments: Some(arguments.to_string()),
                                },
                                r#type: "function".to_string(),
                                raw_text: raw_text.clone(),
                            });
                        }
                        if let Some(tuning_vec) = tuning
                            && let Ok(mut acc) = tuning_clone.lock()
                        {
                            acc.extend(tuning_vec.iter().cloned());
                        }
                    }
                    StreamEvent::Done { .. } => {}
                }
            }
            result
        });

        // Wrap with GenerationSavingStream to save on completion
        let saving_stream = GenerationSavingStream {
            inner: Box::pin(accumulating_stream),
            input_messages,
            accumulated_content,
            accumulated_tool_calls,
            generation_file_path,
            tuning_file_path: tuning_file_path_for_saving,
            generations_dir,
            tool_call_style: self.tool_call_style,
            full_generation_output: self.full_generation_output,
            tuning_writer: tuning_writer_for_saving,
            saved: false,
        };

        Ok(Box::pin(saving_stream))
    }

    async fn count_tokens(
        &self,
        _model: &ModelConfig,
        messages: &[LlmMessage],
        _tools: Option<&[AvailableTool]>,
    ) -> LlmResult<u32> {
        // Prepend context from CONTEXT.txt if available
        let messages_with_context = self.insert_context_messages(messages);
        let prompt = Self::build_prompt(&self.chat_template, &messages_with_context, None);
        let encoded = self.tokenizer.encode(prompt, true)?;
        Ok(encoded.len() as u32)
    }
}

impl LocalBackend {
    /// Internal streaming generation task using ModelEngine.
    ///
    /// Delegates prefill, prediction, and penalty application to the executor.
    /// Performs real-time XML tool call detection for proper UI feedback.
    #[allow(clippy::too_many_arguments)]
    async fn generate_streaming_task(
        executor: Arc<Mutex<ModelEngine>>,
        tokenizer: Tokenizer,
        tokens: Vec<u32>,
        prompt_tokens: u32,
        max_new_tokens: usize,
        eos_token: Option<u32>,
        tx: tokio::sync::mpsc::Sender<LlmResult<StreamEvent>>,
        tuning_writer: Arc<Mutex<Option<crate::tuning::TuningWriter>>>,
        tuning_prompt_tokens: Vec<u32>,
        tool_call_style: crate::xml_tool_parser::ToolCallStyle,
        cached_prefix: Arc<Mutex<Option<CachedPrefix>>>,
    ) -> LlmResult<()> {
        use crate::xml_tool_parser::XmlToolCallParser;

        let mut token_stream = TokenOutputStream::new(tokenizer.clone());
        let mut generated_tokens = 0usize;

        // Add new non-assistant prompt tokens to the shared tuning writer
        {
            let mut writer_guard = tuning_writer.lock().await;
            if let Some(ref mut writer) = *writer_guard {
                let recorded_non_assistant = writer.n_non_assistant_tokens();
                if tuning_prompt_tokens.len() > recorded_non_assistant {
                    writer
                        .add_non_assistant_tokens(&tuning_prompt_tokens[recorded_non_assistant..]);
                }
            }
        }

        // Real-time tool call parser for detecting and extracting tool calls
        let mut xml_parser = XmlToolCallParser::new(tool_call_style);
        let mut pending_content = String::new();
        let mut tool_call_index = 0u32;

        let exec = executor.lock().await;
        let prompt_token_offset = prompt_tokens.saturating_sub(tokens.len() as u32);

        // Fill context while streaming progressive prefill token counts.
        async fn fill_context_with_progress(
            exec: &ModelEngine,
            new_tokens: &[u32],
            prefill_base_tokens: u32,
            prompt_token_offset: u32,
            tx: &tokio::sync::mpsc::Sender<LlmResult<StreamEvent>>,
        ) -> LlmResult<()> {
            let mut progress_rx = exec
                .fill_context_tokens(new_tokens)
                .await
                .map_err(|e| LlmError::ModelError(e.to_string()))?;

            while let Some(progress) = progress_rx.recv().await {
                let processed = progress.map_err(|e| LlmError::ModelError(e.to_string()))?;
                let context_tokens = prompt_token_offset + prefill_base_tokens + processed;
                let _ = tx
                    .send(Ok(StreamEvent::ContextTokens { context_tokens }))
                    .await;
            }
            Ok(())
        }

        // Prefix cache flow using executor snapshots
        {
            let mut prefix_guard = cached_prefix.lock().await;
            if let Some(ref cached) = *prefix_guard {
                let common_len = Self::common_prefix_len(&cached.tokens, &tokens);
                if common_len > 0 && common_len == cached.tokens.len() {
                    // Full match — restore from snapshot and process suffix
                    let snapshot_id = cached.snapshot.id;
                    match exec.restore_snapshot(snapshot_id).await {
                        Ok(_) => {
                            let suffix = &tokens[common_len..];
                            tracing::debug!(
                                "Prefix cache hit: restored {} tokens, processing {} new tokens",
                                common_len,
                                suffix.len()
                            );
                            if !suffix.is_empty() {
                                fill_context_with_progress(
                                    &exec,
                                    suffix,
                                    common_len as u32,
                                    prompt_token_offset,
                                    &tx,
                                )
                                .await?;
                            } else {
                                let context_tokens = prompt_token_offset + common_len as u32;
                                let _ = tx
                                    .send(Ok(StreamEvent::ContextTokens { context_tokens }))
                                    .await;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to restore prefix cache: {}", e);
                            exec.reset_state()
                                .await
                                .map_err(|e| LlmError::ModelError(e.to_string()))?;
                            fill_context_with_progress(&exec, &tokens, 0, prompt_token_offset, &tx)
                                .await?;
                        }
                    }
                } else {
                    // No/partial match — drop old snapshot, reset, fill from scratch
                    let old = prefix_guard.take();
                    if let Some(old_cached) = old {
                        let _ = exec.drop_snapshot(old_cached.snapshot.id).await;
                    }
                    exec.reset_state()
                        .await
                        .map_err(|e| LlmError::ModelError(e.to_string()))?;
                    fill_context_with_progress(&exec, &tokens, 0, prompt_token_offset, &tx).await?;
                }
            } else {
                // No cache — start fresh
                exec.reset_state()
                    .await
                    .map_err(|e| LlmError::ModelError(e.to_string()))?;
                fill_context_with_progress(&exec, &tokens, 0, prompt_token_offset, &tx).await?;
            }

            // Save new prefix cache snapshot
            if let Some(old_cached) = prefix_guard.take() {
                let _ = exec.drop_snapshot(old_cached.snapshot.id).await;
            }
            let snapshot = exec
                .take_snapshot()
                .await
                .map_err(|e| LlmError::ModelError(e.to_string()))?;
            *prefix_guard = Some(CachedPrefix {
                snapshot,
                tokens: tokens.clone(),
            });
        }

        // Prepare for generation (handles edge case where suffix is empty after snapshot restore)
        exec.prepare_for_generation()
            .await
            .map_err(|e| LlmError::ModelError(e.to_string()))?;

        /// Helper to send a Token event.
        async fn send_token_event(
            content: &str,
            tuning: Option<TokenTuningData>,
            context_tokens: u32,
            tx: &tokio::sync::mpsc::Sender<LlmResult<StreamEvent>>,
        ) {
            if !content.is_empty() {
                let event = StreamEvent::Token {
                    role: Role::Assistant,
                    content: content.to_string(),
                    context_tokens,
                    tuning,
                };
                let _ = tx.send(Ok(event)).await;
            }
        }

        /// Helper to send a ToolCall event.
        async fn send_tool_call_event(
            id: String,
            name: String,
            arguments: serde_json::Value,
            raw_text: Option<String>,
            context_tokens: u32,
            tuning: Option<Vec<TokenTuningData>>,
            tx: &tokio::sync::mpsc::Sender<LlmResult<StreamEvent>>,
        ) {
            let event = StreamEvent::ToolCall {
                id,
                name,
                arguments,
                raw_text,
                context_tokens,
                tuning,
            };
            let _ = tx.send(Ok(event)).await;
        }

        for _ in 0..max_new_tokens {
            let predicted = exec
                .predict_token()
                .await
                .map_err(|e| LlmError::ModelError(e.to_string()))?;

            let next_token = predicted.token_id;

            // Record to tuning writer using pre-computed distribution
            {
                let mut writer_guard = tuning_writer.lock().await;
                if let Some(ref mut writer) = *writer_guard {
                    let top_k: Vec<(u32, f32)> = predicted
                        .top_k
                        .iter()
                        .map(|e| (e.token_id, e.log_prob))
                        .collect();
                    let tail: Vec<(u32, f32)> = predicted
                        .tail
                        .iter()
                        .map(|e| (e.token_id, e.log_prob))
                        .collect();
                    let expert_indices = if predicted.expert_indices.is_empty() {
                        None
                    } else {
                        Some(predicted.expert_indices.as_slice())
                    };
                    writer.add_assistant_token_from_distribution(
                        next_token,
                        &top_k,
                        &tail,
                        predicted.tail_mass,
                        expert_indices,
                    );
                }
            }

            // Commit the token for next iteration
            exec.commit_token(next_token)
                .await
                .map_err(|e| LlmError::ModelError(e.to_string()))?;

            generated_tokens += 1;

            // Stream the token - decode and process through XML parser
            if let Ok(Some(delta)) = token_stream.next_token(next_token) {
                xml_parser.add_content(&delta);
                pending_content.push_str(&delta);

                if xml_parser.has_complete_tool_calls() {
                    let parsed_calls = xml_parser.extract_tool_calls();
                    let remaining = xml_parser.buffer().to_string();
                    let current_context_tokens = prompt_tokens + generated_tokens as u32;

                    pending_content.clear();

                    if !remaining.is_empty() {
                        send_token_event(&remaining, None, current_context_tokens, &tx).await;
                    }

                    for parsed in parsed_calls {
                        let tc_id = format!("local_call_{}", tool_call_index);
                        let args: serde_json::Value =
                            serde_json::from_str(&parsed.arguments.to_string())
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        tool_call_index += 1;

                        send_tool_call_event(
                            tc_id,
                            parsed.name,
                            args,
                            parsed.raw_text,
                            current_context_tokens,
                            None,
                            &tx,
                        )
                        .await;
                    }
                } else if !xml_parser.has_partial_tool_call() && !pending_content.is_empty() {
                    let current_context_tokens = prompt_tokens + generated_tokens as u32;
                    send_token_event(&pending_content, None, current_context_tokens, &tx).await;
                    pending_content.clear();
                    xml_parser.clear_buffer();
                }
            }

            if eos_token == Some(next_token) {
                break;
            }
        }

        // Flush any remaining text from the tokenizer
        if let Ok(Some(rest)) = token_stream.decode_rest() {
            xml_parser.add_content(&rest);
            pending_content.push_str(&rest);
        }

        // Process any remaining complete tool calls in the buffer
        if xml_parser.has_complete_tool_calls() {
            let parsed_calls = xml_parser.extract_tool_calls();
            let remaining = xml_parser.buffer().to_string();
            let current_context_tokens = prompt_tokens + generated_tokens as u32;

            pending_content.clear();

            if !remaining.is_empty() {
                send_token_event(&remaining, None, current_context_tokens, &tx).await;
            }

            for parsed in parsed_calls {
                let tc_id = format!("local_call_{}", tool_call_index);
                let args: serde_json::Value = serde_json::from_str(&parsed.arguments.to_string())
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                tool_call_index += 1;

                send_tool_call_event(
                    tc_id,
                    parsed.name,
                    args,
                    parsed.raw_text,
                    current_context_tokens,
                    None,
                    &tx,
                )
                .await;
            }
        } else if !pending_content.is_empty() {
            let current_context_tokens = prompt_tokens + generated_tokens as u32;
            send_token_event(&pending_content, None, current_context_tokens, &tx).await;
        }

        // Send Done event with usage stats
        let done = StreamEvent::Done {
            usage: Some(LlmUsage {
                prompt_tokens,
                completion_tokens: generated_tokens as u32,
            }),
        };
        let _ = tx.send(Ok(done)).await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_chatml_basic() {
        let input = r#"<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
<|im_start|>assistant
Hi there! How can I help you?
<|im_end|>"#;

        let messages = LocalBackend::parse_chatml(input).unwrap();
        assert_eq!(messages.len(), 3);

        assert_eq!(messages[0].role, Role::System);
        assert_eq!(
            messages[0].content.as_deref(),
            Some("You are a helpful assistant.")
        );

        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content.as_deref(), Some("Hello!"));

        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(
            messages[2].content.as_deref(),
            Some("Hi there! How can I help you?")
        );
    }

    #[test]
    fn test_parse_chatml_multiline() {
        let input = r#"<|im_start|>system
You are a helpful assistant.
You should be polite and professional.
<|im_end|>
<|im_start|>user
Can you help me with:
1. Task one
2. Task two
<|im_end|>"#;

        let messages = LocalBackend::parse_chatml(input).unwrap();
        assert_eq!(messages.len(), 2);

        assert_eq!(messages[0].role, Role::System);
        assert_eq!(
            messages[0].content.as_deref(),
            Some("You are a helpful assistant.\nYou should be polite and professional.")
        );

        assert_eq!(messages[1].role, Role::User);
        assert_eq!(
            messages[1].content.as_deref(),
            Some("Can you help me with:\n1. Task one\n2. Task two")
        );
    }

    #[test]
    fn test_parse_chatml_empty() {
        let messages = LocalBackend::parse_chatml("").unwrap();
        assert_eq!(messages.len(), 0);

        let messages = LocalBackend::parse_chatml("   ").unwrap();
        assert_eq!(messages.len(), 0);
    }

    #[test]
    fn test_parse_chatml_invalid_role() {
        let input = r#"<|im_start|>invalid_role
Content here
<|im_end|>"#;

        let result = LocalBackend::parse_chatml(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid role"));
    }

    #[test]
    fn test_parse_chatml_no_end_marker() {
        let input = r#"<|im_start|>system
You are a helpful assistant."#;

        let messages = LocalBackend::parse_chatml(input).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(
            messages[0].content.as_deref(),
            Some("You are a helpful assistant.")
        );
    }
}
