//! Local backend that runs a quantized Qwen3-Next model directly.
//!
//! This backend is intended for on-device training/evaluation loops such as the
//! Paramecia gym. It uses the `paramecia-model` quantized weights and a
//! tokenizer.json to perform autoregressive decoding and optionally emit tool
//! calls when the model produces a structured JSON payload.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use candle::{DType, Device, Tensor};
use paramecia_model::generation::{LogitsProcessor, Sampling};
use paramecia_model::qwen3_next::{
    DeviceOffloadMode, KvCacheQuantization, ModelWeights, PrefixCache,
};
use paramecia_model::token_output_stream::TokenOutputStream;
use paramecia_model::utils::{apply_presence_penalty, apply_repeat_penalty};
use tokenizers::Tokenizer;

/// Number of recent tokens to consider for repeat/presence penalty.
const REPEAT_LAST_N: usize = 128;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;

use crate::backend::{Backend, ChunkStream, CompletionOptions, ModelConfig, ProviderConfig};
use crate::chat_template::{ChatTemplate, QWEN3_NEXT_CHAT_TEMPLATE};
use crate::error::{LlmError, LlmResult};
use crate::types::{AvailableTool, FunctionCall, LlmChunk, LlmMessage, LlmUsage, Role, ToolCall};

/// Default maximum context length for inference.
/// The Qwen3-Next model supports up to 128K context.
///
/// Context length can be configured via `local_context_length` config or
/// `PARAMECIA_CONTEXT_LENGTH` env var. Default: 131072 (128K).
///
/// KV cache quantization can be configured via `local_kv_cache_quant` config or
/// `PARAMECIA_KV_CACHE_QUANT` env var. Options: "f16", "bf16", "q8", "q4" (default).
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
///   - Q4 KV-cache (default, ~4x memory reduction):
///     - 32K tokens: ~16 GB peak
///     - 64K tokens: ~18 GB peak
///     - 128K tokens: ~22 GB peak
const DEFAULT_CONTEXT_TOKENS: usize = 131072;

/// Local backend for running a quantized model on-device.
#[derive(Clone)]
pub struct LocalBackend {
    tokenizer: Tokenizer,
    model: Arc<Mutex<ModelWeights>>,
    device: Device,
    max_tokens: usize,
    max_context: usize,
    eos_token: Option<u32>,
    /// Token ID for `<think>` tag (start of thinking).
    think_start_token: Option<u32>,
    /// Token IDs for `</think>` sequence (end of thinking).
    think_end_tokens: Vec<u32>,
    /// Accumulated MTP statistics (total predictions, accepted predictions).
    /// These are accumulated during speculative generation and can be reset.
    mtp_stats: Arc<Mutex<MtpStats>>,
    /// Chat template for formatting messages.
    chat_template: ChatTemplate,
    /// Cached prefix state for avoiding recomputation of shared conversation context.
    /// When a new prompt starts with the same tokens as a previous prompt, we can
    /// restore the KV cache from the prefix and only process new tokens.
    prefix_cache: Arc<Mutex<Option<PrefixCache>>>,
}

/// Accumulated MTP statistics for tracking acceptance rate.
#[derive(Debug, Clone, Default)]
pub struct MtpStats {
    /// Total number of MTP predictions made.
    pub total_predictions: usize,
    /// Number of MTP predictions that were accepted.
    pub accepted_predictions: usize,
}

impl LocalBackend {
    /// Default HuggingFace repo for the Qwen3-Next tokenizer.
    const DEFAULT_TOKENIZER_REPO: &'static str = "Qwen/Qwen3-Next-80B-A3B-Instruct";

    /// Create a new local backend using the provided configuration.
    ///
    /// The provider configuration must set `local_model_path` or the environment
    /// variable `PARAMECIA_MODEL_PATH`. The tokenizer is automatically downloaded
    /// from HuggingFace if not specified.
    pub fn new(provider: ProviderConfig, _timeout: Duration) -> LlmResult<Self> {
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

        let tokenizer = Self::load_tokenizer(provider.local_tokenizer_path.as_deref())?;

        let device = Self::select_device(provider.local_device.as_deref())?;

        // Parse offload mode from config or environment
        let offload_str = provider
            .local_offload
            .clone()
            .or_else(|| std::env::var("PARAMECIA_OFFLOAD").ok());
        let offload_mode = Self::parse_offload_mode(offload_str.as_deref());

        // Parse KV cache quantization mode from config or environment
        // Defaults to Q4K for maximum memory efficiency
        let kv_cache_str = provider
            .local_kv_cache_quant
            .clone()
            .or_else(|| std::env::var("PARAMECIA_KV_CACHE_QUANT").ok());
        let kv_cache_quant = Self::parse_kv_cache_quant(kv_cache_str.as_deref());

        tracing::info!("Loading Qwen3-Next model from {:?}", model_path);
        tracing::info!("Offload mode: {:?}", offload_mode);
        tracing::info!("KV-cache quantization: {:?}", kv_cache_quant);

        // Extract chat template from GGUF before loading model
        let chat_template = Self::extract_chat_template_from_gguf(&model_path)?;
        tracing::info!(
            "Chat template: {} chars",
            chat_template.template_string().len()
        );

        let mut model = ModelWeights::from_gguf_with_offload_mode(
            &model_path,
            &device,
            offload_mode,
            kv_cache_quant,
        )?;

        // Enable prefetch pipeline for hiding CPU/GPU transfer latency
        // Can be disabled with PARAMECIA_NO_PREFETCH=1 for debugging/comparison
        let disable_prefetch = std::env::var("PARAMECIA_NO_PREFETCH")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        if disable_prefetch {
            tracing::info!("Prefetch pipeline disabled via PARAMECIA_NO_PREFETCH");
        } else if let Err(e) = model.enable_prefetch_pipeline() {
            tracing::warn!("Failed to enable prefetch pipeline: {}", e);
        } else {
            tracing::info!("Prefetch pipeline enabled");
        }

        tracing::info!("Local Qwen3-Next model loaded");

        let eos_token = Self::detect_eos_token(&tokenizer);
        let think_start_token = Self::detect_think_start_token(&tokenizer);
        let think_end_tokens = Self::get_think_end_tokens(&tokenizer);

        if think_start_token.is_some() {
            tracing::debug!(
                "Thinking budget support: start_token={:?}, end_tokens={:?}",
                think_start_token,
                think_end_tokens
            );
        }

        // Parse context length from config or environment variable
        let max_context = provider
            .local_context_length
            .or_else(|| {
                std::env::var("PARAMECIA_CONTEXT_LENGTH")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .unwrap_or(DEFAULT_CONTEXT_TOKENS);
        tracing::info!("Max context length: {} tokens", max_context);

        Ok(Self {
            tokenizer,
            model: Arc::new(Mutex::new(model)),
            device,
            max_tokens: provider.local_max_tokens.unwrap_or(2048),
            max_context,
            eos_token,
            think_start_token,
            think_end_tokens,
            mtp_stats: Arc::new(Mutex::new(MtpStats::default())),
            chat_template,
            prefix_cache: Arc::new(Mutex::new(None)),
        })
    }

    /// Get the chat template used by this backend.
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Clear the prefix cache.
    ///
    /// Call this when starting a new conversation to ensure the model doesn't
    /// try to reuse cached context from a previous conversation.
    pub async fn clear_prefix_cache(&self) {
        let mut cache = self.prefix_cache.lock().await;
        *cache = None;
    }

    /// Get the current prefix cache length (number of cached tokens).
    ///
    /// Returns 0 if no prefix is cached.
    pub async fn prefix_cache_len(&self) -> usize {
        let cache = self.prefix_cache.lock().await;
        cache.as_ref().map(|c| c.prefix_tokens.len()).unwrap_or(0)
    }

    /// Load tokenizer from path, env var, or download from HuggingFace.
    fn load_tokenizer(path_override: Option<&str>) -> LlmResult<Tokenizer> {
        // 1. Check explicit path override
        if let Some(path) = path_override {
            return Tokenizer::from_file(path).map_err(LlmError::from);
        }

        // 2. Check environment variable
        if let Ok(path) = std::env::var("PARAMECIA_TOKENIZER_PATH") {
            return Tokenizer::from_file(&path).map_err(LlmError::from);
        }

        // 3. Download from HuggingFace
        tracing::info!(
            "Downloading tokenizer from HuggingFace: {}",
            Self::DEFAULT_TOKENIZER_REPO
        );
        let api = hf_hub::api::sync::Api::new().map_err(|e| {
            LlmError::InvalidConfig(format!("Failed to create HuggingFace API: {e}"))
        })?;
        let repo = api.model(Self::DEFAULT_TOKENIZER_REPO.to_string());
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| LlmError::InvalidConfig(format!("Failed to download tokenizer: {e}")))?;
        Tokenizer::from_file(&tokenizer_path).map_err(LlmError::from)
    }

    fn select_device(hint: Option<&str>) -> LlmResult<Device> {
        if let Some(requested) = hint {
            match requested {
                "cuda" => {
                    if let Ok(device) = Device::cuda_if_available(0) {
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

        if let Ok(device) = Device::cuda_if_available(0) {
            if matches!(device, Device::Cuda(_)) {
                return Ok(device);
            }
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
            Some("up") => DeviceOffloadMode::UpProjectionsOnCpu,
            Some("updown") => DeviceOffloadMode::UpDownProjectionsOnCpu,
            Some("experts") | None => DeviceOffloadMode::ExpertsOnCpu,
            Some(other) => {
                tracing::warn!("Unknown offload mode '{}', using 'experts'", other);
                DeviceOffloadMode::ExpertsOnCpu
            }
        }
    }

    /// Parse KV cache quantization mode from string. Defaults to "q4k".
    ///
    /// Supported values:
    /// - "f16" / "fp16": Store KV cache as f16 (maximum accuracy)
    /// - "bf16" / "bfloat16": Store KV cache as bf16
    /// - "q8" / "q8_0": Quantize to Q8_0 (8-bit, ~2x memory reduction)
    /// - "q4" / "q4k" / "q4_k": Quantize to Q4K (4-bit, ~4x memory reduction, default)
    fn parse_kv_cache_quant(hint: Option<&str>) -> KvCacheQuantization {
        match hint {
            None => KvCacheQuantization::Q4K,
            Some(s) => KvCacheQuantization::from_str(s).unwrap_or_else(|| {
                tracing::warn!("Unknown KV cache quantization '{}', using 'q4k'", s);
                KvCacheQuantization::Q4K
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

    /// Detect the `<think>` start token.
    fn detect_think_start_token(tokenizer: &Tokenizer) -> Option<u32> {
        let vocab = tokenizer.get_vocab(true);
        vocab.get("<think>").copied()
    }

    /// Extract chat template from GGUF file metadata.
    fn extract_chat_template_from_gguf(model_path: &std::path::Path) -> LlmResult<ChatTemplate> {
        use candle::quantized::gguf_file;

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

    /// Get the token IDs for `</think>` sequence.
    fn get_think_end_tokens(tokenizer: &Tokenizer) -> Vec<u32> {
        // Try to encode "</think>" - it may be one or multiple tokens
        if let Ok(encoding) = tokenizer.encode("</think>", false) {
            encoding.get_ids().to_vec()
        } else {
            Vec::new()
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
                prompt.push_str("\"");
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
            if first_is_system {
                if let Some(content) = &messages[0].content {
                    prompt.push_str(content);
                    prompt.push_str("\n\n");
                }
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

    fn parse_tool_calls(raw: &str) -> Option<Vec<ToolCall>> {
        // Parse tool calls using the hybrid XML-JSON format (Qwen3-Next format)
        // Format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        use crate::xml_tool_parser::{ToolCallStyle, XmlToolCallParser};

        let mut parser = XmlToolCallParser::new(ToolCallStyle::QwenVl);
        let parsed_calls = parser.parse(raw);
        let mut parsed = parser.to_tool_calls(parsed_calls);

        // If no hybrid format calls found, try legacy Qwen-Coder XML format as fallback
        if parsed.is_empty() {
            let mut legacy_parser = XmlToolCallParser::new(ToolCallStyle::QwenCoder);
            let legacy_calls = legacy_parser.parse(raw);
            parsed = legacy_parser.to_tool_calls(legacy_calls);
        }

        // Also try the old JSON format for backwards compatibility
        if parsed.is_empty() {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
                if let Some(tool_calls) = value.get("tool_calls").and_then(|v| v.as_array()) {
                    for (i, tc) in tool_calls.iter().enumerate() {
                        let name = tc
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                            .or_else(|| {
                                tc.get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|v| v.as_str())
                                    .map(str::to_string)
                            });

                        let args_value = tc
                            .get("arguments")
                            .cloned()
                            .or_else(|| {
                                tc.get("function").and_then(|f| f.get("arguments").cloned())
                            })
                            .unwrap_or(serde_json::Value::Null);

                        let arguments =
                            serde_json::to_string(&args_value).unwrap_or_else(|_| "{}".to_string());

                        parsed.push(ToolCall {
                            id: tc
                                .get("id")
                                .and_then(|v| v.as_str())
                                .map(str::to_string)
                                .or_else(|| Some(format!("local-tool-{i}"))),
                            index: Some(i),
                            function: FunctionCall {
                                name,
                                arguments: Some(arguments),
                            },
                            r#type: "function".to_string(),
                        });
                    }
                }
            }
        }

        if parsed.is_empty() {
            None
        } else {
            Some(parsed)
        }
    }

    fn sampling_for_model(model: &ModelConfig) -> Sampling {
        if model.temperature <= 0.0 {
            tracing::debug!(
                "LocalBackend sampling: ArgMax (greedy, temperature={})",
                model.temperature
            );
            Sampling::ArgMax
        } else {
            // Use top-k then top-p sampling with the configured parameters.
            // Default recommended settings: temperature=0.7, top_p=0.8, top_k=20, min_p=0.0
            tracing::debug!(
                "LocalBackend sampling: TopKThenTopP(k={}, p={}, temperature={})",
                model.top_k,
                model.top_p,
                model.temperature
            );
            Sampling::TopKThenTopP {
                k: model.top_k,
                p: model.top_p as f64,
                temperature: model.temperature as f64,
            }
        }
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

    fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> String {
        tokenizer
            .decode(tokens, true)
            .unwrap_or_else(|_| String::new())
    }

    /// Compute the length of the common prefix between two token sequences.
    fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    async fn generate(
        &self,
        model_config: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
    ) -> LlmResult<(String, Option<Vec<ToolCall>>)> {
        let prompt = Self::build_prompt(&self.chat_template, messages, tools);
        tracing::debug!(
            "LocalBackend::generate prompt ({} chars, {} messages, {} tools): {}",
            prompt.len(),
            messages.len(),
            tools.map(|t| t.len()).unwrap_or(0),
            if prompt.len() > 500 {
                &prompt[..500]
            } else {
                &prompt
            }
        );
        let encoded = self.tokenizer.encode(prompt, true)?;
        let mut tokens = encoded.get_ids().to_vec();
        tracing::debug!("LocalBackend::generate encoded to {} tokens", tokens.len());
        let max_new_tokens = self.max_tokens;
        tokens = Self::truncate_tokens(&tokens, max_new_tokens, self.max_context);
        let prompt_len = tokens.len();

        let sampling = Self::sampling_for_model(model_config);
        let mut processor = LogitsProcessor::from_sampling(42, sampling.clone());

        let mut generated = Vec::new();

        // Thinking budget tracking
        let thinking_budget = model_config.thinking_budget;
        let mut in_thinking = false;
        let mut thinking_tokens = 0usize;
        let mut thinking_budget_exhausted = false;

        let mut model = self.model.lock().await;

        // Try to use prefix cache
        let mut prefix_cache_guard = self.prefix_cache.lock().await;
        let (start_offset, tokens_to_process) = if let Some(ref cache) = *prefix_cache_guard {
            let common_len = Self::common_prefix_len(&cache.prefix_tokens, &tokens);
            if common_len > 0 && common_len == cache.prefix_tokens.len() {
                // The new prompt starts with the entire cached prefix
                // Restore the cache and only process new tokens
                match model.restore_prefix_cache(cache) {
                    Ok(_) => {
                        tracing::debug!(
                            "Prefix cache hit: restored {} tokens, processing {} new tokens",
                            common_len,
                            tokens.len() - common_len
                        );
                        (common_len, &tokens[common_len..])
                    }
                    Err(e) => {
                        tracing::warn!("Failed to restore prefix cache: {}", e);
                        model.clear_kv_cache();
                        (0, tokens.as_slice())
                    }
                }
            } else if common_len > 0 {
                // Partial match - could truncate cache but simpler to just recompute
                tracing::debug!(
                    "Prefix cache partial match: {} common tokens out of {} cached, recomputing",
                    common_len,
                    cache.prefix_tokens.len()
                );
                model.clear_kv_cache();
                (0, tokens.as_slice())
            } else {
                // No match - clear and recompute
                model.clear_kv_cache();
                (0, tokens.as_slice())
            }
        } else {
            // No cache - start fresh
            model.clear_kv_cache();
            (0, tokens.as_slice())
        };

        // Process tokens that weren't covered by the prefix cache and get initial logits
        let initial_logits = if !tokens_to_process.is_empty() {
            let input = Tensor::new(tokens_to_process, &self.device)?.unsqueeze(0)?;
            Some(model.forward(&input, start_offset)?)
        } else {
            // All tokens were cached - need to do a forward pass with just the last token
            // to get logits for the first generated token
            let last_token = tokens[tokens.len() - 1];
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            Some(model.forward(&input, tokens.len() - 1)?)
        };

        // Save the current state as the new prefix cache (prompt tokens only)
        let new_cache = model.save_prefix_cache(tokens[..prompt_len].to_vec());
        *prefix_cache_guard = Some(new_cache);
        drop(prefix_cache_guard);

        // Standard generation (no MTP)
        // Use the initial logits from prompt processing for the first token
        let mut pending_logits = initial_logits;

        for _ in 0..max_new_tokens {
            // Check if we need to inject </think> due to thinking budget
            if in_thinking
                && thinking_budget > 0
                && thinking_tokens >= thinking_budget
                && !thinking_budget_exhausted
            {
                tracing::debug!(
                    "Thinking budget exhausted ({} tokens), injecting </think>",
                    thinking_tokens
                );
                // Inject </think> tokens
                for &end_token in &self.think_end_tokens {
                    tokens.push(end_token);
                    generated.push(end_token);
                }
                thinking_budget_exhausted = true;
                in_thinking = false;
                // Need new logits after injecting tokens
                pending_logits = None;
            }

            // Use pending logits if available (first iteration or after thinking injection)
            // Otherwise compute from last token only
            let logits = if let Some(logits) = pending_logits.take() {
                logits.squeeze(0)?
            } else {
                let start_pos = tokens.len() - 1;
                let input = Tensor::new(&[tokens[start_pos]], &self.device)?.unsqueeze(0)?;
                model.forward(&input, start_pos)?.squeeze(0)?
            };
            let (logits, sanitized) = Self::sanitize_logits(&logits, &self.device)?;
            if sanitized {
                tracing::warn!(
                    "Sanitized non-finite logits at position {} during local inference",
                    tokens.len()
                );
            }

            // Apply repeat penalty (frequency-based) and presence penalty (binary)
            let penalty_start = tokens.len().saturating_sub(REPEAT_LAST_N);
            let penalty_context = &tokens[penalty_start..];
            let logits =
                apply_repeat_penalty(&logits, model_config.repeat_penalty, penalty_context)?;
            let logits =
                apply_presence_penalty(&logits, model_config.presence_penalty, penalty_context)?;

            let next_token = Self::sample_next_token(&mut processor, &logits)?;
            tokens.push(next_token);
            generated.push(next_token);

            // Track thinking state
            if let Some(start_token) = self.think_start_token {
                if next_token == start_token {
                    in_thinking = true;
                    thinking_tokens = 0;
                }
            }
            if in_thinking {
                thinking_tokens += 1;
                // Check if we just generated </think>
                if !self.think_end_tokens.is_empty() {
                    let end_len = self.think_end_tokens.len();
                    if generated.len() >= end_len
                        && generated[generated.len() - end_len..] == self.think_end_tokens
                    {
                        in_thinking = false;
                    }
                }
            }

            if let Some(eos) = self.eos_token {
                if next_token == eos {
                    break;
                }
            }
        }

        let text = Self::decode_tokens(&self.tokenizer, &generated);
        let tool_calls = Self::parse_tool_calls(&text);

        Ok((text, tool_calls))
    }

    fn sanitize_logits(logits: &Tensor, device: &Device) -> LlmResult<(Tensor, bool)> {
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let shape = logits_f32.dims().to_vec();
        let mut values = logits_f32.to_vec1::<f32>()?;

        let mut sanitized = false;
        for value in &mut values {
            if !value.is_finite() {
                *value = -1e4;
                sanitized = true;
            } else if *value > 1e4 {
                *value = 1e4;
                sanitized = true;
            } else if *value < -1e4 {
                *value = -1e4;
                sanitized = true;
            }
        }

        if sanitized {
            let tensor = Tensor::from_vec(values, shape, device)?;
            Ok((tensor, true))
        } else {
            Ok((logits_f32, false))
        }
    }

    fn sample_next_token(processor: &mut LogitsProcessor, logits: &Tensor) -> LlmResult<u32> {
        match processor.sample(logits) {
            Ok(token) => Ok(token),
            Err(e) => {
                let logits_f32 = logits.to_dtype(DType::F32)?;
                let logits_vec = logits_f32.to_vec1::<f32>()?;

                let mut best: Option<(usize, f32)> = None;
                for (idx, value) in logits_vec.iter().enumerate() {
                    if !value.is_finite() {
                        continue;
                    }

                    match best {
                        Some((_, best_val)) if value <= &best_val => {}
                        _ => best = Some((idx, *value)),
                    }
                }

                if let Some((idx, _)) = best {
                    tracing::warn!(
                        "Non-finite logits encountered during sampling ({}); falling back to argmax",
                        e
                    );
                    Ok(idx as u32)
                } else {
                    Err(LlmError::ModelError(format!(
                        "Sampling failed due to non-finite logits: {}",
                        e
                    )))
                }
            }
        }
    }

    /// Expose the underlying model for fine-tuning perturbations.
    pub async fn apply_perturbations(
        &self,
        scales: Option<std::collections::HashMap<String, Tensor>>,
        loras: Option<std::collections::HashMap<String, (Tensor, Tensor)>>,
        sigma: f64,
    ) -> LlmResult<()> {
        let mut model = self.model.lock().await;
        if let Some(scales_map) = scales {
            model.set_all_custom_scales(scales_map)?;
        }
        if let Some(lora_map) = loras {
            model.set_lora_adapters(&lora_map, sigma)?;
        }
        Ok(())
    }

    /// Clear any custom scales or adapters applied during evaluation.
    pub async fn clear_perturbations(&self) -> LlmResult<()> {
        let mut model = self.model.lock().await;
        model.clear_custom_scales();
        model.clear_lora_adapters();
        Ok(())
    }

    /// Get layer configurations suitable for EGGROLL.
    pub async fn eggroll_layer_configs(
        &self,
    ) -> LlmResult<Vec<(String, (usize, usize), Option<usize>)>> {
        let model = self.model.lock().await;
        Ok(model.eggroll_layer_configs())
    }

    /// Expose the underlying device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Generate tokens from a prompt (for Replicate training mode).
    ///
    /// # Arguments
    /// * `prompt_tokens` - The prompt token IDs
    /// * `max_new_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// * Vector of generated token IDs (including prompt)
    pub async fn generate_tokens(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> LlmResult<Vec<u32>> {
        let mut model = self.model.lock().await;
        model.clear_kv_cache();

        let mut generated = prompt_tokens.to_vec();
        let device = self.device.clone();

        for _ in 0..max_new_tokens {
            let input = Tensor::new(&generated[..], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;

            // Greedy sampling
            let next_token = logits.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];

            // Check for EOS (Qwen uses 151643)
            if next_token == 151643 {
                break;
            }

            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Get embeddings for a token sequence (for Replicate training mode).
    ///
    /// Uses the last token's hidden state as the embedding.
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to embed
    ///
    /// # Returns
    /// * Vector of embedding values
    pub async fn get_embeddings(&self, tokens: &[u32]) -> LlmResult<Vec<f32>> {
        let mut model = self.model.lock().await;
        model.clear_kv_cache();

        let device = self.device.clone();
        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;
        let embeddings = model.forward_embeddings_last(&input)?;

        Ok(embeddings.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// Evaluate MTP (Multi-Token Prediction) accuracy on a token sequence.
    ///
    /// This runs MTP predictions and verifies them against the ground truth
    /// tokens in the sequence, returning the percentage of correct predictions.
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to evaluate on (needs at least num_mtp_steps + 2 tokens)
    /// * `num_mtp_steps` - Number of MTP prediction steps to evaluate (typically 1-4)
    ///
    /// # Returns
    /// * (accuracy, total_predictions, correct_predictions)
    pub async fn evaluate_mtp_accuracy(
        &self,
        tokens: &[u32],
        num_mtp_steps: usize,
    ) -> LlmResult<(f32, usize, usize)> {
        let mut model = self.model.lock().await;
        let device = self.device.clone();
        let input = Tensor::new(tokens, &device)?.unsqueeze(0)?;

        let (accuracy, total, correct) = model.evaluate_mtp_accuracy(&input, num_mtp_steps)?;
        Ok((accuracy, total, correct))
    }

    /// Check if the model has MTP weights loaded.
    pub async fn has_mtp(&self) -> bool {
        let model = self.model.lock().await;
        model.has_mtp()
    }

    /// Reset accumulated MTP statistics.
    ///
    /// Call this before evaluating a population member to start fresh tracking.
    pub async fn reset_mtp_stats(&self) {
        let mut stats = self.mtp_stats.lock().await;
        stats.total_predictions = 0;
        stats.accepted_predictions = 0;
    }

    /// Get accumulated MTP statistics.
    ///
    /// Returns (total_predictions, accepted_predictions, acceptance_rate).
    /// The acceptance_rate is None if no predictions were made.
    pub async fn get_mtp_stats(&self) -> (usize, usize, Option<f32>) {
        let stats = self.mtp_stats.lock().await;
        let rate = if stats.total_predictions > 0 {
            Some(stats.accepted_predictions as f32 / stats.total_predictions as f32)
        } else {
            None
        };
        (stats.total_predictions, stats.accepted_predictions, rate)
    }

    /// Accumulate MTP statistics (internal helper).
    async fn accumulate_mtp_stats(&self, total: usize, accepted: usize) {
        let mut stats = self.mtp_stats.lock().await;
        stats.total_predictions += total;
        stats.accepted_predictions += accepted;
    }

    /// Generate tokens using speculative decoding with MTP.
    ///
    /// This method uses the MTP head for speculative decoding and tracks
    /// the acceptance rate, returning it along with the generated tokens.
    /// If MTP is not available, falls back to standard generation.
    ///
    /// # Arguments
    /// * `prompt_tokens` - Initial prompt tokens
    /// * `max_new_tokens` - Maximum tokens to generate
    /// * `num_speculative` - Number of tokens to speculate per step (typically 2-4)
    ///
    /// # Returns
    /// * `GenerationWithMtp` containing tokens and MTP statistics
    pub async fn generate_tokens_speculative(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        num_speculative: usize,
    ) -> LlmResult<GenerationWithMtp> {
        let mut model = self.model.lock().await;
        model.clear_kv_cache();

        let device = self.device.clone();
        let mut generated = prompt_tokens.to_vec();

        // Check if MTP is available
        if !model.has_mtp() {
            // Fallback to standard generation
            for _ in 0..max_new_tokens {
                let input = Tensor::new(&generated[..], &device)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;

                let next_token = logits.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];

                if next_token == 151643 {
                    break;
                }

                generated.push(next_token);
            }

            return Ok(GenerationWithMtp {
                tokens: generated,
                mtp_total_predictions: 0,
                mtp_accepted_predictions: 0,
                mtp_acceptance_rate: None,
            });
        }

        // Use speculative decoding with MTP
        let mut tokens_generated = 0;
        let mut mtp_total_predictions = 0usize;
        let mut mtp_accepted_predictions = 0usize;

        while tokens_generated < max_new_tokens {
            let input = Tensor::new(&generated[..], &device)?.unsqueeze(0)?;
            model.clear_kv_cache();

            // Greedy sampling function
            let sample_fn = |logits: &Tensor| -> candle::Result<u32> {
                logits.argmax(candle::D::Minus1)?.to_scalar::<u32>()
            };

            let (_logits, accepted_tokens, num_accepted) =
                model.forward_speculative(&input, 0, num_speculative, sample_fn)?;

            // Track MTP acceptance rate
            mtp_total_predictions += num_speculative;
            mtp_accepted_predictions += num_accepted;

            // No tokens accepted means generation should stop
            if accepted_tokens.is_empty() {
                break;
            }

            // Add accepted tokens
            let mut hit_eos = false;
            for token in accepted_tokens.iter() {
                if *token == 151643 {
                    hit_eos = true;
                    break;
                }
                generated.push(*token);
                tokens_generated += 1;
            }

            if hit_eos {
                break;
            }
        }

        // Accumulate stats for tracking across multiple generations
        self.accumulate_mtp_stats(mtp_total_predictions, mtp_accepted_predictions)
            .await;

        let mtp_acceptance_rate = if mtp_total_predictions > 0 {
            Some(mtp_accepted_predictions as f32 / mtp_total_predictions as f32)
        } else {
            None
        };

        Ok(GenerationWithMtp {
            tokens: generated,
            mtp_total_predictions,
            mtp_accepted_predictions,
            mtp_acceptance_rate,
        })
    }
}

/// Result of token generation with MTP statistics.
#[derive(Debug, Clone)]
pub struct GenerationWithMtp {
    /// Generated tokens (including prompt).
    pub tokens: Vec<u32>,
    /// Total number of MTP predictions made.
    pub mtp_total_predictions: usize,
    /// Number of MTP predictions that were accepted (matched main model).
    pub mtp_accepted_predictions: usize,
    /// MTP acceptance rate (0.0-1.0), None if MTP was not used.
    pub mtp_acceptance_rate: Option<f32>,
}

#[async_trait]
impl Backend for LocalBackend {
    async fn complete(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
        _options: &CompletionOptions,
    ) -> LlmResult<LlmChunk> {
        let (text, tool_calls) = self.generate(model, messages, tools).await?;

        let message = if let Some(tool_calls) = tool_calls {
            LlmMessage {
                role: Role::Assistant,
                content: None,
                tool_calls: Some(tool_calls),
                name: None,
                tool_call_id: None,
            }
        } else {
            LlmMessage::assistant(text)
        };

        Ok(LlmChunk {
            message,
            finish_reason: Some("stop".to_string()),
            usage: None,
        })
    }

    async fn complete_streaming(
        &self,
        model_config: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
        _options: &CompletionOptions,
    ) -> LlmResult<ChunkStream> {
        let prompt = Self::build_prompt(&self.chat_template, messages, tools);
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

        let sampling = Self::sampling_for_model(model_config);
        let processor = LogitsProcessor::from_sampling(42, sampling.clone());

        // Create a channel for streaming chunks
        let (tx, rx) = tokio::sync::mpsc::channel::<LlmResult<LlmChunk>>(32);

        // Clone what we need for the spawned task
        let model = Arc::clone(&self.model);
        let prefix_cache = Arc::clone(&self.prefix_cache);
        let device = self.device.clone();
        let tokenizer = self.tokenizer.clone();
        let eos_token = self.eos_token;
        let repeat_penalty = model_config.repeat_penalty;
        let presence_penalty = model_config.presence_penalty;
        let thinking_budget = model_config.thinking_budget;
        let think_start_token = self.think_start_token;
        let think_end_tokens = self.think_end_tokens.clone();

        // Spawn a task for generation
        tokio::spawn(async move {
            let result = LocalBackend::generate_streaming_task(
                model,
                prefix_cache,
                device,
                tokenizer,
                tokens,
                prompt_tokens,
                max_new_tokens,
                processor,
                eos_token,
                repeat_penalty,
                presence_penalty,
                thinking_budget,
                think_start_token,
                think_end_tokens,
                tx.clone(),
            )
            .await;

            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn count_tokens(
        &self,
        _model: &ModelConfig,
        messages: &[LlmMessage],
        _tools: Option<&[AvailableTool]>,
    ) -> LlmResult<u32> {
        let prompt = Self::build_prompt(&self.chat_template, messages, None);
        let encoded = self.tokenizer.encode(prompt, true)?;
        Ok(encoded.len() as u32)
    }
}

impl LocalBackend {
    /// Internal streaming generation task - standard generation only (no MTP)
    ///
    /// This task performs real-time XML tool call detection to provide proper UI feedback.
    /// Instead of streaming raw `<tool_call>` XML to the UI, it:
    /// 1. Buffers content when `<tool_call>` is detected
    /// 2. Emits proper `ToolCall` chunks when `</tool_call>` is complete
    /// 3. Only sends non-tool-call content as text chunks
    #[allow(clippy::too_many_arguments)]
    async fn generate_streaming_task(
        model: Arc<Mutex<ModelWeights>>,
        prefix_cache: Arc<Mutex<Option<PrefixCache>>>,
        device: Device,
        tokenizer: Tokenizer,
        mut tokens: Vec<u32>,
        prompt_tokens: u32,
        max_new_tokens: usize,
        mut processor: LogitsProcessor,
        eos_token: Option<u32>,
        repeat_penalty: f32,
        presence_penalty: f32,
        thinking_budget: usize,
        think_start_token: Option<u32>,
        think_end_tokens: Vec<u32>,
        tx: tokio::sync::mpsc::Sender<LlmResult<LlmChunk>>,
    ) -> LlmResult<()> {
        use crate::xml_tool_parser::{ToolCallStyle, XmlToolCallParser};

        let mut token_stream = TokenOutputStream::new(tokenizer.clone());
        let mut generated_tokens = 0usize;
        let mut generated_token_ids = Vec::new();

        // Real-time tool call parser for detecting and extracting hybrid XML-JSON tool calls
        let mut xml_parser = XmlToolCallParser::new(ToolCallStyle::QwenVl);
        // Buffer for content that hasn't been sent yet (used when inside a partial tool call)
        let mut pending_content = String::new();
        // Track tool call index for proper ordering
        let mut tool_call_index = 0u32;

        // Thinking budget tracking
        let mut in_thinking = false;
        let mut thinking_tokens = 0usize;
        let mut thinking_budget_exhausted = false;

        let mut model = model.lock().await;

        // Try to use prefix cache
        let mut prefix_cache_guard = prefix_cache.lock().await;
        let (start_offset, tokens_to_process) = if let Some(ref cache) = *prefix_cache_guard {
            let common_len = Self::common_prefix_len(&cache.prefix_tokens, &tokens);
            if common_len > 0 && common_len == cache.prefix_tokens.len() {
                // The new prompt starts with the entire cached prefix
                match model.restore_prefix_cache(cache) {
                    Ok(_) => {
                        tracing::debug!(
                            "Streaming prefix cache hit: restored {} tokens, processing {} new tokens",
                            common_len,
                            tokens.len() - common_len
                        );
                        (common_len, tokens[common_len..].to_vec())
                    }
                    Err(e) => {
                        tracing::warn!("Failed to restore prefix cache: {}", e);
                        model.clear_kv_cache();
                        (0, tokens.clone())
                    }
                }
            } else {
                model.clear_kv_cache();
                (0, tokens.clone())
            }
        } else {
            model.clear_kv_cache();
            (0, tokens.clone())
        };

        // Process tokens that weren't covered by the prefix cache and get initial logits
        let initial_logits = if !tokens_to_process.is_empty() {
            let input = Tensor::new(tokens_to_process.as_slice(), &device)?.unsqueeze(0)?;
            Some(model.forward(&input, start_offset)?)
        } else {
            // All tokens were cached - need to do a forward pass with just the last token
            let last_token = tokens[tokens.len() - 1];
            let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
            Some(model.forward(&input, tokens.len() - 1)?)
        };

        // Save the current state as the new prefix cache (prompt tokens only)
        let prompt_len = tokens.len();
        let new_cache = model.save_prefix_cache(tokens[..prompt_len].to_vec());
        *prefix_cache_guard = Some(new_cache);
        drop(prefix_cache_guard);

        // Use the initial logits from prompt processing for the first token
        let mut pending_logits = initial_logits;

        /// Helper to send content that is safe (not part of a tool call)
        async fn send_safe_content(
            content: &str,
            prompt_tokens: u32,
            generated_tokens: usize,
            tx: &tokio::sync::mpsc::Sender<LlmResult<LlmChunk>>,
        ) {
            if !content.is_empty() {
                let chunk = LlmChunk {
                    message: LlmMessage::assistant(content),
                    finish_reason: None,
                    usage: Some(LlmUsage {
                        prompt_tokens,
                        completion_tokens: generated_tokens as u32,
                    }),
                };
                let _ = tx.send(Ok(chunk)).await;
            }
        }

        /// Helper to send a tool call chunk
        async fn send_tool_call(
            tool_call: ToolCall,
            prompt_tokens: u32,
            generated_tokens: usize,
            tx: &tokio::sync::mpsc::Sender<LlmResult<LlmChunk>>,
        ) {
            let chunk = LlmChunk {
                message: LlmMessage {
                    role: Role::Assistant,
                    content: None,
                    tool_calls: Some(vec![tool_call]),
                    name: None,
                    tool_call_id: None,
                },
                finish_reason: None,
                usage: Some(LlmUsage {
                    prompt_tokens,
                    completion_tokens: generated_tokens as u32,
                }),
            };
            let _ = tx.send(Ok(chunk)).await;
        }

        for _ in 0..max_new_tokens {
            // Check if we need to inject </think> due to thinking budget
            if in_thinking
                && thinking_budget > 0
                && thinking_tokens >= thinking_budget
                && !thinking_budget_exhausted
            {
                tracing::debug!(
                    "Thinking budget exhausted ({} tokens), injecting </think>",
                    thinking_tokens
                );
                // Inject </think> tokens and stream them
                for &end_token in &think_end_tokens {
                    tokens.push(end_token);
                    generated_tokens += 1;
                    generated_token_ids.push(end_token);
                    if let Ok(Some(delta)) = token_stream.next_token(end_token) {
                        // Add to parser and pending content
                        xml_parser.add_content(&delta);
                        pending_content.push_str(&delta);
                    }
                }
                thinking_budget_exhausted = true;
                in_thinking = false;
                // Need new logits after injecting tokens
                pending_logits = None;
            }

            // Use pending logits if available (first iteration or after thinking injection)
            // Otherwise compute from last token only
            let logits = if let Some(logits) = pending_logits.take() {
                logits.squeeze(0)?
            } else {
                let start_pos = tokens.len() - 1;
                let input = Tensor::new(&[tokens[start_pos]], &device)?.unsqueeze(0)?;
                model.forward(&input, start_pos)?.squeeze(0)?
            };
            let (logits, _) = Self::sanitize_logits(&logits, &device)?;

            // Apply repeat penalty (frequency-based) and presence penalty (binary)
            let penalty_start = tokens.len().saturating_sub(REPEAT_LAST_N);
            let penalty_context = &tokens[penalty_start..];
            let logits = apply_repeat_penalty(&logits, repeat_penalty, penalty_context)?;
            let logits = apply_presence_penalty(&logits, presence_penalty, penalty_context)?;

            let next_token = Self::sample_next_token(&mut processor, &logits)?;

            tokens.push(next_token);
            generated_tokens += 1;
            generated_token_ids.push(next_token);

            // Track thinking state
            if let Some(start_token) = think_start_token {
                if next_token == start_token {
                    in_thinking = true;
                    thinking_tokens = 0;
                }
            }
            if in_thinking {
                thinking_tokens += 1;
                // Check if we just generated </think>
                if !think_end_tokens.is_empty() {
                    let end_len = think_end_tokens.len();
                    if generated_token_ids.len() >= end_len
                        && generated_token_ids[generated_token_ids.len() - end_len..]
                            == think_end_tokens
                    {
                        in_thinking = false;
                    }
                }
            }

            // Stream the token - decode and process through XML parser
            if let Ok(Some(delta)) = token_stream.next_token(next_token) {
                // Add delta to the XML parser's buffer
                xml_parser.add_content(&delta);
                pending_content.push_str(&delta);

                // Check for complete tool calls
                if xml_parser.has_complete_tool_calls() {
                    // Extract complete tool calls and get remaining content
                    let parsed_calls = xml_parser.extract_tool_calls();
                    let remaining = xml_parser.buffer().to_string();

                    // The content before the tool call(s) needs to be sent
                    // Find content that was before the first tool call
                    // The parser's buffer now contains only non-tool-call content
                    // But pending_content has everything - we need to figure out what to send

                    // Clear pending_content since we're handling it now
                    pending_content.clear();

                    // Send any remaining non-tool-call content that was extracted
                    if !remaining.is_empty() {
                        // This is content that came before or after tool calls
                        // For now, we'll send it as content
                        send_safe_content(&remaining, prompt_tokens, generated_tokens, &tx).await;
                    }

                    // Convert parsed tool calls to proper ToolCall structs and emit them
                    for parsed in parsed_calls {
                        let tool_call = ToolCall {
                            id: Some(format!("local_call_{}", tool_call_index)),
                            index: Some(tool_call_index as usize),
                            function: FunctionCall {
                                name: Some(parsed.name),
                                arguments: Some(parsed.arguments.to_string()),
                            },
                            r#type: "function".to_string(),
                        };
                        tool_call_index += 1;

                        send_tool_call(tool_call, prompt_tokens, generated_tokens, &tx).await;
                    }
                } else if !xml_parser.has_partial_tool_call() {
                    // No partial tool call in progress - safe to send content immediately
                    // But we should send from pending_content and clear it
                    if !pending_content.is_empty() {
                        send_safe_content(&pending_content, prompt_tokens, generated_tokens, &tx)
                            .await;
                        pending_content.clear();
                        // Keep parser buffer in sync
                        xml_parser.clear_buffer();
                    }
                }
                // If has_partial_tool_call() is true, we buffer and wait for completion
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

            pending_content.clear();

            if !remaining.is_empty() {
                send_safe_content(&remaining, prompt_tokens, generated_tokens, &tx).await;
            }

            for parsed in parsed_calls {
                let tool_call = ToolCall {
                    id: Some(format!("local_call_{}", tool_call_index)),
                    index: Some(tool_call_index as usize),
                    function: FunctionCall {
                        name: Some(parsed.name),
                        arguments: Some(parsed.arguments.to_string()),
                    },
                    r#type: "function".to_string(),
                };
                tool_call_index += 1;

                send_tool_call(tool_call, prompt_tokens, generated_tokens, &tx).await;
            }
        } else if !pending_content.is_empty() {
            // Send any remaining buffered content that wasn't part of a tool call
            // This handles incomplete tool calls at the end (they get sent as text)
            send_safe_content(&pending_content, prompt_tokens, generated_tokens, &tx).await;
        }

        // Send final chunk with finish_reason (signals completion)
        let final_chunk = LlmChunk {
            message: LlmMessage {
                role: Role::Assistant,
                content: None,
                tool_calls: None,
                name: None,
                tool_call_id: None,
            },
            finish_reason: Some("stop".to_string()),
            usage: Some(LlmUsage {
                prompt_tokens,
                completion_tokens: generated_tokens as u32,
            }),
        };

        let _ = tx.send(Ok(final_chunk)).await;

        Ok(())
    }
}
