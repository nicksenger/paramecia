//! Core agent implementation.

use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::config::VibeConfig;
use crate::error::{VibeError, VibeResult};
use crate::events::{
    AgentEvent, AssistantEvent, CompactEndEvent, CompactStartEvent, ToolCallEvent, ToolResultEvent,
};
use crate::middleware::{
    AutoCompactMiddleware, ContextWarningMiddleware, ConversationContext, MiddlewareAction,
    MiddlewarePipeline, PlanModeMiddleware, PriceLimitMiddleware, ResetReason, TurnLimitMiddleware,
};
use crate::modes::AgentMode;
use crate::multishot_examples::generate_multishot_examples;
use crate::prompts::UtilityPrompt;
use crate::session::SessionLogger;
use crate::system_prompt::get_universal_system_prompt_with_tools;
use crate::types::{AgentStats, ApprovalResponse};
use crate::utils::{
    CancellationReason, TOOL_ERROR_TAG, VIBE_STOP_EVENT_TAG, get_user_cancellation_message,
    is_user_cancellation_event,
};
use paramecia_llm::backend::{
    Backend, BackendFactory, CompletionOptions, ModelConfig as LlmModelConfig,
};
use paramecia_llm::format::ApiToolFormatHandler;
use paramecia_llm::{
    AvailableTool, LlmMessage, LlmUsage, Role, StrToolChoice, ToolCall, ToolChoice,
};
use paramecia_mcp::client::McpClient;
use paramecia_mcp::transport::{HttpTransport, StdioTransport};
use paramecia_tools::ToolManager;
use paramecia_tools::types::{PatternCheckResult, ToolPermission};

/// Callback type for tool approval (async).
/// Returns (response, optional feedback message).
pub type ApprovalCallback = Arc<
    dyn Fn(
            String,
            serde_json::Value,
            String,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = (ApprovalResponse, Option<String>)> + Send>,
        > + Send
        + Sync,
>;

/// Result of tool execution matching Python's behavior.
#[derive(Debug)]
enum ToolExecutionResult {
    /// Tool executed successfully with a result.
    Success(serde_json::Value),
    /// Tool was skipped by user with a reason.
    Skipped(String),
    /// Tool execution failed with an error.
    Failed(String),
}

/// The main agent that coordinates LLM and tools.
pub struct Agent {
    config: VibeConfig,
    mode: Arc<std::sync::atomic::AtomicU8>,
    mode_value: AgentMode, // Keep a local copy for convenience
    backend: Arc<dyn Backend>,
    tool_manager: ToolManager,
    #[allow(dead_code)]
    format_handler: ApiToolFormatHandler,
    messages: Vec<LlmMessage>,
    stats: AgentStats,
    middleware: MiddlewarePipeline,
    session_logger: SessionLogger,
    session_id: String,
    approval_callback: Option<ApprovalCallback>,
    enable_streaming: bool,
    max_turns: Option<u32>,
    max_price: Option<f64>,
    /// Tracks the finish_reason from the last LLM response (matches Python's _last_chunk.finish_reason).
    last_finish_reason: Option<String>,
    /// Tools approved for "always allow" during this session.
    session_approved_tools: std::collections::HashSet<String>,
    /// Cancellation flag for interrupting ongoing operations.
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl Agent {
    /// Create a new agent.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot be created.
    pub fn new(config: VibeConfig, mode: AgentMode) -> VibeResult<Self> {
        let active_model = config.get_active_model()?;
        let provider = config.get_provider_for_model(active_model)?;

        let backend = BackendFactory::create(
            &paramecia_llm::backend::ProviderConfig {
                name: provider.name.clone(),
                api_base: provider.api_base.clone(),
                api_key_env_var: provider.api_key_env_var.clone(),
                backend: provider.backend,
                local_model_path: provider.local_model_path.clone(),
                local_tokenizer_path: provider.local_tokenizer_path.clone(),
                local_max_tokens: provider.local_max_tokens,
                local_device: provider.local_device.clone(),
                local_offload: provider.local_offload.clone(),
                local_context_length: provider.local_context_length,
                local_kv_cache_quant: provider.local_kv_cache_quant.clone(),
            },
            Duration::from_secs_f64(config.api_timeout),
        )
        .map_err(VibeError::Config)?;

        Self::from_backend(config, mode, backend)
    }

    /// Create a new agent using a pre-constructed backend (useful for local training).
    pub fn from_backend(
        config: VibeConfig,
        mode: AgentMode,
        backend: Arc<dyn Backend>,
    ) -> VibeResult<Self> {
        let tool_manager = ToolManager::with_configs(config.tools.clone());
        let session_logger = SessionLogger::new(config.session_logging.clone());
        let session_id = session_logger.session_id().to_string();

        // Build system prompt using the universal system prompt builder
        let system_prompt = get_universal_system_prompt_with_tools(&tool_manager, &config);
        tracing::info!(
            "System prompt built: {} chars (~{} tokens)",
            system_prompt.len(),
            system_prompt.len() / 4 // Rough estimate: 4 chars per token
        );

        // Start with system prompt
        let mut messages = vec![LlmMessage::system(system_prompt)];

        // Add multishot examples to teach the model tool usage format
        let workdir = config.effective_workdir();
        let multishot_examples = generate_multishot_examples(&tool_manager, &workdir);
        tracing::info!(
            "Added {} multishot example messages for tool usage",
            multishot_examples.len()
        );
        messages.extend(multishot_examples);

        // Create atomic mode for sharing with middleware
        let mode_atomic = Arc::new(std::sync::atomic::AtomicU8::new(mode as u8));

        let mut agent = Self {
            config,
            mode: mode_atomic,
            mode_value: mode,
            backend,
            tool_manager,
            format_handler: ApiToolFormatHandler::new(),
            messages,
            stats: AgentStats::default(),
            middleware: MiddlewarePipeline::new(),
            session_logger,
            session_id,
            approval_callback: None,
            enable_streaming: false,
            max_turns: None,
            max_price: None,
            last_finish_reason: None,
            session_approved_tools: std::collections::HashSet::new(),
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };

        // Update pricing
        if let Ok(model) = agent.config.get_active_model() {
            agent
                .stats
                .update_pricing(model.input_price, model.output_price);
        }

        // Setup middleware
        agent.setup_middleware(None, None);

        Ok(agent)
    }

    /// Create a new agent and connect to MCP servers.
    pub async fn new_with_mcp(config: VibeConfig, mode: AgentMode) -> VibeResult<Self> {
        let mut agent = Self::new(config, mode)?;

        // Connect to MCP servers and register their tools
        if let Err(e) = agent.connect_mcp_servers().await {
            println!("Warning: Failed to connect to MCP servers: {}", e);
        }

        Ok(agent)
    }

    /// Create an agent with additional options.
    pub async fn with_options(
        config: VibeConfig,
        mode: AgentMode,
        max_turns: Option<u32>,
        max_price: Option<f64>,
        enable_streaming: bool,
    ) -> VibeResult<Self> {
        let mut agent = Self::new(config, mode)?;
        agent.enable_streaming = enable_streaming;
        agent.max_turns = max_turns;
        agent.max_price = max_price;
        agent.setup_middleware(max_turns, max_price);

        // Connect to MCP servers and register their tools
        if let Err(e) = agent.connect_mcp_servers().await {
            println!("Warning: Failed to connect to MCP servers: {}", e);
        }

        Ok(agent)
    }

    /// Create an agent with a custom backend and additional options.
    pub async fn with_backend_options(
        config: VibeConfig,
        mode: AgentMode,
        backend: Arc<dyn Backend>,
        max_turns: Option<u32>,
        max_price: Option<f64>,
        enable_streaming: bool,
    ) -> VibeResult<Self> {
        let mut agent = Self::from_backend(config, mode, backend)?;
        agent.enable_streaming = enable_streaming;
        agent.max_turns = max_turns;
        agent.max_price = max_price;
        agent.setup_middleware(max_turns, max_price);

        // Connect to MCP servers and register their tools
        if let Err(e) = agent.connect_mcp_servers().await {
            println!("Warning: Failed to connect to MCP servers: {}", e);
        }

        Ok(agent)
    }

    fn setup_middleware(&mut self, max_turns: Option<u32>, max_price: Option<f64>) {
        self.middleware.clear();

        if let Some(turns) = max_turns {
            self.middleware.add(TurnLimitMiddleware::new(turns));
        }

        if let Some(price) = max_price {
            self.middleware.add(PriceLimitMiddleware::new(price));
        }

        if self.config.auto_compact_threshold > 0 {
            self.middleware.add(AutoCompactMiddleware::new(
                self.config.auto_compact_threshold,
            ));

            // Add context warnings at 50% threshold
            if self.config.context_warnings {
                self.middleware.add(ContextWarningMiddleware::new(
                    0.5,
                    self.config.auto_compact_threshold,
                ));
            }
        }

        // Add plan mode middleware
        let mode_atomic = Arc::clone(&self.mode);
        self.middleware
            .add(PlanModeMiddleware::new(Arc::new(move || {
                let val = mode_atomic.load(std::sync::atomic::Ordering::Relaxed);
                // Convert u8 back to AgentMode
                match val {
                    0 => AgentMode::Default,
                    1 => AgentMode::Plan,
                    2 => AgentMode::AcceptEdits,
                    3 => AgentMode::AutoApprove,
                    _ => AgentMode::Default,
                }
            })));
    }

    /// Get the current mode.
    #[must_use]
    pub fn mode(&self) -> AgentMode {
        self.mode_value
    }

    /// Set the current mode.
    pub fn set_mode(&mut self, mode: AgentMode) {
        self.mode_value = mode;
        self.mode
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }

    /// Cancel the current operation.
    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check if cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if auto-approve is enabled.
    #[must_use]
    pub fn auto_approve(&self) -> bool {
        self.mode_value.auto_approve()
    }

    /// Get the session ID.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the log file path for the current session.
    #[must_use]
    pub fn log_file_path(&self) -> Option<std::path::PathBuf> {
        self.session_logger.filepath()
    }

    /// Get current stats.
    #[must_use]
    pub fn stats(&self) -> &AgentStats {
        &self.stats
    }

    /// Get the messages.
    #[must_use]
    pub fn messages(&self) -> &[LlmMessage] {
        &self.messages
    }

    /// Build the active model config.
    fn active_model_config(&self) -> VibeResult<LlmModelConfig> {
        let active_model = self.config.get_active_model()?;
        Ok(LlmModelConfig {
            name: active_model.name.clone(),
            temperature: active_model.temperature,
            top_p: active_model.top_p,
            top_k: active_model.top_k,
            min_p: active_model.min_p,
            repeat_penalty: active_model.repeat_penalty,
            presence_penalty: active_model.presence_penalty,
            thinking_budget: active_model.thinking_budget,
        })
    }

    /// Recalculate context tokens using the backend tokenizer.
    async fn refresh_context_tokens(&mut self) -> VibeResult<u32> {
        let model_config = self.active_model_config()?;
        let token_count = self
            .backend
            .count_tokens(&model_config, &self.messages, None)
            .await
            .map_err(VibeError::Llm)?;
        self.stats.context_tokens = token_count;
        Ok(token_count)
    }

    /// Set the approval callback.
    pub fn set_approval_callback(&mut self, callback: ApprovalCallback) {
        self.approval_callback = Some(callback);
    }

    /// Process a user message and stream events.
    ///
    /// Events are sent to the provided channel as they happen.
    /// The caller should spawn this in a separate task and listen on the receiver
    /// concurrently to receive events in real-time.
    pub async fn act(
        &mut self,
        user_message: &str,
        event_tx: mpsc::Sender<AgentEvent>,
    ) -> VibeResult<()> {
        // Reset cancellation flag
        self.cancelled
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.act_with_cancellation(user_message, event_tx, self.cancelled.clone())
            .await
    }

    /// Process a user message and stream events with external cancellation.
    ///
    /// Events are sent to the provided channel as they happen.
    /// The caller should spawn this in a separate task and listen on the receiver
    /// concurrently to receive events in real-time.
    pub async fn act_with_cancellation(
        &mut self,
        user_message: &str,
        event_tx: mpsc::Sender<AgentEvent>,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
    ) -> VibeResult<()> {
        // Reset the provided cancellation flag
        cancelled.store(false, std::sync::atomic::Ordering::Relaxed);
        // Clean message history before processing
        self.clean_message_history().await?;

        // Add user message
        self.messages.push(LlmMessage::user(user_message));
        self.stats.steps += 1;

        // Update context tokens to reflect the current conversation size
        self.refresh_context_tokens().await?;

        // Run conversation loop - events are sent to event_tx as they happen
        self.conversation_loop(event_tx, cancelled).await?;

        Ok(())
    }

    /// Clean message history to ensure valid state.
    async fn clean_message_history(&mut self) -> VibeResult<()> {
        if self.messages.len() < 2 {
            return Ok(());
        }

        self.fill_missing_tool_responses();
        self.ensure_assistant_after_tools().await?;
        Ok(())
    }

    /// Fill in missing tool responses.
    fn fill_missing_tool_responses(&mut self) {
        // Collect insertions to make
        let mut insertions: Vec<(usize, LlmMessage)> = Vec::new();

        let mut i = 1;
        while i < self.messages.len() {
            let is_assistant = self.messages[i].role == Role::Assistant;

            if is_assistant {
                // Collect tool call info
                let tool_calls_info: Vec<(String, String)> = self.messages[i]
                    .tool_calls
                    .as_ref()
                    .map(|tcs| {
                        tcs.iter()
                            .map(|tc| {
                                (
                                    tc.id.clone().unwrap_or_default(),
                                    tc.function.name.clone().unwrap_or_default(),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let expected = tool_calls_info.len();
                if expected > 0 {
                    let mut actual = 0;
                    let mut j = i + 1;
                    while j < self.messages.len() && self.messages[j].role == Role::Tool {
                        actual += 1;
                        j += 1;
                    }

                    if actual < expected {
                        let insertion_point = i + 1 + actual;
                        for (offset, (tool_call_id, tool_name)) in
                            tool_calls_info.into_iter().skip(actual).enumerate()
                        {
                            let cancel_msg = get_user_cancellation_message(
                                CancellationReason::ToolNoResponse,
                                None,
                            );
                            let empty_response =
                                LlmMessage::tool(&tool_call_id, &tool_name, cancel_msg.to_string());
                            insertions.push((insertion_point + offset, empty_response));
                        }
                        i = i + 1 + expected;
                        continue;
                    }
                }
            }
            i += 1;
        }

        // Apply insertions in reverse order to maintain correct indices
        for (idx, msg) in insertions.into_iter().rev() {
            self.messages.insert(idx, msg);
        }
    }

    /// Ensure there's an assistant message after tool responses.
    async fn ensure_assistant_after_tools(&mut self) -> VibeResult<()> {
        if self.messages.len() < 2 {
            return Ok(());
        }

        if let Some(last) = self.messages.last()
            && last.role == Role::Tool
        {
            self.messages.push(LlmMessage::assistant("Understood."));
            // Update context tokens to reflect the current conversation size
            self.refresh_context_tokens().await?;
        }
        Ok(())
    }

    async fn conversation_loop(
        &mut self,
        tx: mpsc::Sender<AgentEvent>,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
    ) -> VibeResult<()> {
        loop {
            // Run before-turn middleware
            let context = ConversationContext {
                messages: &self.messages,
                stats: &self.stats,
                config: &self.config,
            };
            let result = self.middleware.run_before_turn(&context);

            match result.action {
                MiddlewareAction::Stop => {
                    if let Some(reason) = result.reason {
                        let _ = tx
                            .send(AgentEvent::Assistant(AssistantEvent {
                                content: format!(
                                    "<{VIBE_STOP_EVENT_TAG}>{reason}</{VIBE_STOP_EVENT_TAG}>"
                                ),
                                stopped_by_middleware: true,
                            }))
                            .await;
                    }
                    break;
                }
                MiddlewareAction::Compact => {
                    let old_tokens = self.stats.context_tokens;
                    let _ = tx
                        .send(AgentEvent::CompactStart(CompactStartEvent {
                            current_context_tokens: old_tokens,
                            threshold: self.config.auto_compact_threshold,
                        }))
                        .await;

                    let summary = self.compact().await?;

                    let _ = tx
                        .send(AgentEvent::CompactEnd(CompactEndEvent {
                            old_context_tokens: old_tokens,
                            new_context_tokens: self.stats.context_tokens,
                            summary_length: summary.len(),
                        }))
                        .await;
                }
                MiddlewareAction::InjectMessage => {
                    // Inject message into the last message's content
                    if let Some(msg) = result.message
                        && let Some(last_msg) = self.messages.last_mut()
                    {
                        if let Some(content) = &mut last_msg.content {
                            content.push_str("\n\n");
                            content.push_str(&msg);
                        } else {
                            last_msg.content = Some(msg);
                        }
                    }
                }
                MiddlewareAction::Continue => {}
            }

            // Perform LLM turn
            self.stats.steps += 1;
            let (should_continue, user_cancelled) =
                self.perform_llm_turn(&tx, Arc::clone(&cancelled)).await?;

            if user_cancelled {
                // User cancelled during tool execution, stop the loop
                break;
            }

            if !should_continue {
                break;
            }

            // Run after-turn middleware
            let context = ConversationContext {
                messages: &self.messages,
                stats: &self.stats,
                config: &self.config,
            };
            let result = self.middleware.run_after_turn(&context);

            if result.action == MiddlewareAction::Stop {
                break;
            }
        }

        // Save session
        let _ = self
            .session_logger
            .save(&self.messages, &self.stats, self.auto_approve())
            .await;

        Ok(())
    }

    /// Performs an LLM turn. Returns (should_continue, user_cancelled).
    async fn perform_llm_turn(
        &mut self,
        tx: &mpsc::Sender<AgentEvent>,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
    ) -> VibeResult<(bool, bool)> {
        let active_model = self.config.get_active_model()?;
        let model_config = LlmModelConfig {
            name: active_model.name.clone(),
            temperature: active_model.temperature,
            top_p: active_model.top_p,
            top_k: active_model.top_k,
            min_p: active_model.min_p,
            repeat_penalty: active_model.repeat_penalty,
            presence_penalty: active_model.presence_penalty,
            thinking_budget: active_model.thinking_budget,
        };

        // Get available tools
        let tools = self.get_available_tools();

        let options = CompletionOptions {
            tool_choice: Some(ToolChoice::String(StrToolChoice::Auto)),
            ..Default::default()
        };

        let start_time = std::time::Instant::now();

        if self.enable_streaming {
            // Use streaming API when enabled
            let mut stream = self
                .backend
                .complete_streaming(&model_config, &self.messages, Some(&tools), &options)
                .await
                .map_err(VibeError::Llm)?;

            let mut full_content = String::new();
            // Use a map indexed by tool call index to properly accumulate arguments
            // (matching Python's OrderedDict approach)
            let mut tool_calls_map: std::collections::BTreeMap<u32, ToolCall> =
                std::collections::BTreeMap::new();
            let mut finish_reason: Option<String> = None;
            let mut usage: Option<LlmUsage> = None;

            // Send content chunks immediately for smooth streaming
            const BATCH_SIZE: usize = 5;
            let mut content_buffer = String::new();
            let mut chunks_with_content = 0;
            let mut sent_tool_call_ids = std::collections::HashSet::new();

            // Process streaming chunks
            while let Some(chunk) = stream.next().await {
                // Check for cancellation periodically
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok((false, false));
                }

                let chunk = chunk.map_err(VibeError::Llm)?;

                // Update finish reason and usage from the last chunk
                if chunk.finish_reason.is_some() {
                    finish_reason = chunk.finish_reason.clone();
                }
                if chunk.usage.is_some() {
                    usage = chunk.usage.clone();
                }

                // Handle tool calls - flush content buffer when tool calls arrive
                // Accumulate tool calls by index (matching Python's OrderedDict approach)
                if let Some(chunk_tool_calls) = &chunk.message.tool_calls {
                    // Flush any pending content before tool calls
                    if !content_buffer.is_empty() {
                        let _ = tx
                            .send(AgentEvent::Assistant(AssistantEvent {
                                content: content_buffer.clone(),
                                stopped_by_middleware: false,
                            }))
                            .await;
                        content_buffer.clear();
                        chunks_with_content = 0;
                    }

                    for tc in chunk_tool_calls {
                        let idx = tc.index.unwrap_or(0) as u32;

                        if let Some(existing) = tool_calls_map.get_mut(&idx) {
                            // Accumulate arguments for existing tool call
                            if let Some(new_args) = &tc.function.arguments {
                                let current_args =
                                    existing.function.arguments.get_or_insert_with(String::new);
                                current_args.push_str(new_args);
                            }
                        } else {
                            // New tool call - insert it
                            tool_calls_map.insert(idx, tc.clone());

                            // Send tool call event for new tool calls
                            let tool_call_id =
                                tc.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
                            if !sent_tool_call_ids.contains(&tool_call_id) {
                                sent_tool_call_ids.insert(tool_call_id.clone());

                                let tool_name = tc.function.name.as_deref().unwrap_or("unknown");
                                // Note: args might be incomplete at this point, but we show the spinner
                                let args: serde_json::Value = tc
                                    .function
                                    .arguments
                                    .as_ref()
                                    .and_then(|a| serde_json::from_str(a).ok())
                                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                                let _ = tx
                                    .send(AgentEvent::ToolCall(ToolCallEvent {
                                        tool_name: tool_name.to_string(),
                                        tool_info: None,
                                        args: args.clone(),
                                        tool_call_id: tool_call_id.clone(),
                                    }))
                                    .await;
                            }
                        }
                    }
                    continue;
                }

                // Accumulate content with batching
                if let Some(content) = &chunk.message.content
                    && !content.is_empty()
                {
                    full_content.push_str(content);
                    content_buffer.push_str(content);
                    chunks_with_content += 1;

                    // Send batched content when we hit BATCH_SIZE
                    if chunks_with_content >= BATCH_SIZE {
                        let _ = tx
                            .send(AgentEvent::Assistant(AssistantEvent {
                                content: content_buffer.clone(),
                                stopped_by_middleware: false,
                            }))
                            .await;
                        content_buffer.clear();
                        chunks_with_content = 0;
                    }
                }
            }

            // Flush any remaining content in buffer
            if !content_buffer.is_empty() {
                let _ = tx
                    .send(AgentEvent::Assistant(AssistantEvent {
                        content: content_buffer,
                        stopped_by_middleware: false,
                    }))
                    .await;
            }

            // Convert accumulated tool calls map to vector (matching Python's behavior)
            let tool_calls: Option<Vec<ToolCall>> = if tool_calls_map.is_empty() {
                None
            } else {
                Some(tool_calls_map.into_values().collect())
            };

            // Build the final message - only if we have content or tool_calls
            let has_content = !full_content.is_empty();
            let has_tool_calls = tool_calls.is_some();

            let final_message = if has_content || has_tool_calls {
                Some(LlmMessage {
                    role: Role::Assistant,
                    content: if full_content.is_empty() {
                        None
                    } else {
                        Some(full_content)
                    },
                    tool_calls,
                    name: None,
                    tool_call_id: None,
                })
            } else {
                None // Don't create empty assistant messages
            };

            let duration = start_time.elapsed().as_secs_f64();

            // Track finish_reason like Python's _last_chunk.finish_reason
            self.last_finish_reason = finish_reason.clone();

            // Update stats
            if let Some(usage) = &usage {
                self.stats.last_turn_prompt_tokens = usage.prompt_tokens;
                self.stats.last_turn_completion_tokens = usage.completion_tokens;
                self.stats.session_prompt_tokens += usage.prompt_tokens;
                self.stats.session_completion_tokens += usage.completion_tokens;
                self.stats.last_turn_duration = duration;
                if duration > 0.0 {
                    self.stats.tokens_per_second = f64::from(usage.completion_tokens) / duration;
                }
            }

            // Add assistant message only if we have one
            if let Some(msg) = final_message.clone() {
                self.messages.push(msg);
            }

            // Update context tokens to reflect the current conversation size
            self.refresh_context_tokens().await?;

            // Handle tool calls
            let mut user_cancelled = false;
            if has_tool_calls {
                if let Some(ref msg) = final_message
                    && let Some(tool_calls) = &msg.tool_calls
                {
                    for tool_call in tool_calls {
                        let tool_name = tool_call.function.name.as_deref().unwrap_or("unknown");
                        let tool_call_id = tool_call
                            .id
                            .clone()
                            .unwrap_or_else(|| Uuid::new_v4().to_string());

                        // Parse arguments
                        let args: serde_json::Value = tool_call
                            .function
                            .arguments
                            .as_ref()
                            .and_then(|a| serde_json::from_str(a).ok())
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                        // Execute tool
                        let tool_result =
                            self.execute_tool(tool_name, &args, &tool_call_id, tx).await;

                        // Add tool response message
                        // Format tool result based on execution outcome to match Python behavior
                        let response_content = match &tool_result {
                            ToolExecutionResult::Success(value) => {
                                // Format as "key: value" pairs like Python does
                                if let serde_json::Value::Object(map) = value {
                                    map.iter()
                                        .map(|(k, v)| {
                                            let v_str = match v {
                                                serde_json::Value::String(s) => s.clone(),
                                                serde_json::Value::Null => String::new(),
                                                other => other.to_string(),
                                            };
                                            format!("{}: {}", k, v_str)
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                } else {
                                    serde_json::to_string(value).unwrap_or_default()
                                }
                            }
                            ToolExecutionResult::Skipped(skip_reason) => {
                                // Check if this was a user cancellation
                                if is_user_cancellation_event(Some(skip_reason)) {
                                    user_cancelled = true;
                                }
                                // Skipped tools use the skip reason directly (no tool_error tags)
                                skip_reason.clone()
                            }
                            ToolExecutionResult::Failed(error_msg) => {
                                // Errors use the tool_error tag format
                                format!(
                                    "<{TOOL_ERROR_TAG}>{} failed: {}</{TOOL_ERROR_TAG}>",
                                    tool_name, error_msg
                                )
                            }
                        };

                        self.messages.push(LlmMessage::tool(
                            &tool_call_id,
                            tool_name,
                            &response_content,
                        ));
                        // Update context tokens to reflect the current conversation size
                        self.refresh_context_tokens().await?;
                    }
                }

                // Continue the loop if we had tool calls (unless user cancelled)
                return Ok((true, user_cancelled));
            }

            // Check if we should break the loop - matching Python's behavior:
            // should_break_loop = (last_message.role != Role.tool and
            //                      self._last_chunk is not None and
            //                      self._last_chunk.finish_reason is not None)
            // Since we just added the assistant message, last_message.role is Assistant (not Tool),
            // and we have the finish_reason tracked. Only break if finish_reason is set.
            //
            // Note: We must rely ONLY on finish_reason, not content presence.
            // The LLM might send content first then tool calls, so having content
            // doesn't mean we should stop. This matches Python's behavior exactly.
            let should_break = finish_reason.is_some();
            Ok((!should_break, false))
        } else {
            // Use non-streaming API (original implementation)
            let result = self
                .backend
                .complete(&model_config, &self.messages, Some(&tools), &options)
                .await
                .map_err(VibeError::Llm)?;

            let duration = start_time.elapsed().as_secs_f64();

            // Track finish_reason like Python's _last_chunk.finish_reason
            self.last_finish_reason = result.finish_reason.clone();

            // Update stats
            if let Some(usage) = &result.usage {
                self.stats.last_turn_prompt_tokens = usage.prompt_tokens;
                self.stats.last_turn_completion_tokens = usage.completion_tokens;
                self.stats.session_prompt_tokens += usage.prompt_tokens;
                self.stats.session_completion_tokens += usage.completion_tokens;
                self.stats.last_turn_duration = duration;
                if duration > 0.0 {
                    self.stats.tokens_per_second = f64::from(usage.completion_tokens) / duration;
                }
            }

            // Add assistant message
            self.messages.push(result.message.clone());

            // Update context tokens to reflect the current conversation size
            self.refresh_context_tokens().await?;

            // Send assistant event
            if let Some(content) = &result.message.content
                && !content.is_empty()
            {
                let _ = tx
                    .send(AgentEvent::Assistant(AssistantEvent {
                        content: content.clone(),
                        stopped_by_middleware: false,
                    }))
                    .await;
            }

            // Handle tool calls
            let mut user_cancelled = false;
            if let Some(tool_calls) = &result.message.tool_calls {
                for tool_call in tool_calls {
                    let tool_name = tool_call.function.name.as_deref().unwrap_or("unknown");
                    let tool_call_id = tool_call
                        .id
                        .clone()
                        .unwrap_or_else(|| Uuid::new_v4().to_string());

                    // Parse arguments
                    let args: serde_json::Value = tool_call
                        .function
                        .arguments
                        .as_ref()
                        .and_then(|a| serde_json::from_str(a).ok())
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    // Send tool call event
                    let _ = tx
                        .send(AgentEvent::ToolCall(ToolCallEvent {
                            tool_name: tool_name.to_string(),
                            tool_info: None,
                            args: args.clone(),
                            tool_call_id: tool_call_id.clone(),
                        }))
                        .await;

                    // Execute tool
                    let tool_result = self.execute_tool(tool_name, &args, &tool_call_id, tx).await;

                    // Add tool response message
                    // Format tool result based on execution outcome to match Python behavior
                    let response_content = match &tool_result {
                        ToolExecutionResult::Success(value) => {
                            // Format as "key: value" pairs like Python does
                            if let serde_json::Value::Object(map) = value {
                                map.iter()
                                    .map(|(k, v)| {
                                        let v_str = match v {
                                            serde_json::Value::String(s) => s.clone(),
                                            serde_json::Value::Null => String::new(),
                                            other => other.to_string(),
                                        };
                                        format!("{}: {}", k, v_str)
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            } else {
                                serde_json::to_string(value).unwrap_or_default()
                            }
                        }
                        ToolExecutionResult::Skipped(skip_reason) => {
                            // Check if this was a user cancellation
                            if is_user_cancellation_event(Some(skip_reason)) {
                                user_cancelled = true;
                            }
                            // Skipped tools use the skip reason directly (no tool_error tags)
                            skip_reason.clone()
                        }
                        ToolExecutionResult::Failed(error_msg) => {
                            // Errors use the tool_error tag format
                            format!(
                                "<{TOOL_ERROR_TAG}>{} failed: {}</{TOOL_ERROR_TAG}>",
                                tool_name, error_msg
                            )
                        }
                    };

                    self.messages.push(LlmMessage::tool(
                        &tool_call_id,
                        tool_name,
                        &response_content,
                    ));
                    // Update context tokens to reflect the current conversation size
                    self.refresh_context_tokens().await?;
                }

                // Continue the loop if we had tool calls (unless user cancelled)
                return Ok((true, user_cancelled));
            }

            // Check if we should break the loop - matching Python's behavior:
            // should_break_loop = (last_message.role != Role.tool and
            //                      self._last_chunk is not None and
            //                      self._last_chunk.finish_reason is not None)
            // Since we just added the assistant message, last_message.role is Assistant (not Tool),
            // and we have the finish_reason tracked. Only break if finish_reason is set.
            let should_break = self.last_finish_reason.is_some();
            Ok((!should_break, false))
        }
    }

    async fn execute_tool(
        &mut self,
        tool_name: &str,
        args: &serde_json::Value,
        tool_call_id: &str,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> ToolExecutionResult {
        let start_time = std::time::Instant::now();

        // Get tool instance
        let tool_arc = match self.tool_manager.get(tool_name) {
            Ok(tool) => tool,
            Err(e) => {
                let error_msg = format!("Error getting tool '{}': {}", tool_name, e);
                let _ = tx
                    .send(AgentEvent::ToolResult(ToolResultEvent {
                        tool_name: tool_name.to_string(),
                        result: None,
                        error: Some(error_msg.clone()),
                        skipped: false,
                        skip_reason: None,
                        duration: None,
                        tool_call_id: tool_call_id.to_string(),
                    }))
                    .await;
                self.stats.tool_calls_failed += 1;
                return ToolExecutionResult::Failed(error_msg);
            }
        };

        // Check if this tool was session-approved (via "Always allow for this session")
        if self.session_approved_tools.contains(tool_name) {
            // Tool was already approved for this session, skip the dialog
        } else {
            // Check if we should execute (async for approval dialog)
            let (pattern_result, permission, tool_label) = match tool_arc
                .inspect(|tool| {
                    (
                        tool.check_patterns(args),
                        tool.config().permission,
                        tool.name().to_string(),
                    )
                })
                .await
            {
                Ok(values) => values,
                Err(error) => {
                    let error_msg = format!("Error checking tool '{}': {}", tool_name, error);
                    let _ = tx
                        .send(AgentEvent::ToolResult(ToolResultEvent {
                            tool_name: tool_name.to_string(),
                            result: None,
                            error: Some(error_msg.clone()),
                            skipped: false,
                            skip_reason: None,
                            duration: None,
                            tool_call_id: tool_call_id.to_string(),
                        }))
                        .await;
                    self.stats.tool_calls_failed += 1;
                    return ToolExecutionResult::Failed(error_msg);
                }
            };

            let (should_execute, skip_feedback, add_to_session) = match self
                .should_execute_tool(&tool_label, pattern_result, permission, args, tool_call_id)
                .await
            {
                Ok((should, feedback, session_approve)) => (should, feedback, session_approve),
                Err(_) => (false, None, false),
            };

            // Add to session-approved tools if requested
            if add_to_session {
                self.session_approved_tools.insert(tool_name.to_string());
            }

            if !should_execute {
                self.stats.tool_calls_rejected += 1;
                let skip_reason = skip_feedback.unwrap_or_else(|| {
                    get_user_cancellation_message(CancellationReason::ToolSkipped, Some(tool_name))
                        .to_string()
                });
                let _ = tx
                    .send(AgentEvent::ToolResult(ToolResultEvent {
                        tool_name: tool_name.to_string(),
                        result: None,
                        error: None,
                        skipped: true,
                        skip_reason: Some(skip_reason.clone()),
                        duration: None,
                        tool_call_id: tool_call_id.to_string(),
                    }))
                    .await;
                return ToolExecutionResult::Skipped(skip_reason);
            }
        }

        self.stats.tool_calls_agreed += 1;

        // Execute the tool
        let result = tool_arc.execute(args.clone()).await;

        let duration = start_time.elapsed().as_secs_f64();

        match result {
            Ok(value) => {
                self.stats.tool_calls_succeeded += 1;
                let _ = tx
                    .send(AgentEvent::ToolResult(ToolResultEvent {
                        tool_name: tool_name.to_string(),
                        result: Some(value.clone()),
                        error: None,
                        skipped: false,
                        skip_reason: None,
                        duration: Some(duration),
                        tool_call_id: tool_call_id.to_string(),
                    }))
                    .await;
                ToolExecutionResult::Success(value)
            }
            Err(e) => {
                self.stats.tool_calls_failed += 1;
                let error_msg = e.to_string();
                let _ = tx
                    .send(AgentEvent::ToolResult(ToolResultEvent {
                        tool_name: tool_name.to_string(),
                        result: None,
                        error: Some(error_msg.clone()),
                        skipped: false,
                        skip_reason: None,
                        duration: Some(duration),
                        tool_call_id: tool_call_id.to_string(),
                    }))
                    .await;
                ToolExecutionResult::Failed(error_msg)
            }
        }
    }

    /// Returns (should_execute, feedback, add_to_session_approved)
    async fn should_execute_tool(
        &self,
        tool_name: &str,
        pattern_result: PatternCheckResult,
        permission: ToolPermission,
        args: &serde_json::Value,
        tool_call_id: &str,
    ) -> VibeResult<(bool, Option<String>, bool)> {
        // Check mode
        if self.auto_approve() {
            return Ok((true, None, false));
        }

        // Check pattern-based auto-approval
        match pattern_result {
            PatternCheckResult::Allowed => return Ok((true, None, false)),
            PatternCheckResult::Denied => {
                let reason = format!("Tool '{tool_name}' blocked by denylist");
                return Ok((false, Some(reason), false));
            }
            PatternCheckResult::NoMatch => {}
        }

        // Check tool permission
        match permission {
            ToolPermission::Always => return Ok((true, None, false)),
            ToolPermission::Never => {
                let reason = format!("Tool '{tool_name}' is permanently disabled");
                return Ok((false, Some(reason), false));
            }
            ToolPermission::Ask => {}
        }

        // Ask user via async callback
        if let Some(callback) = &self.approval_callback {
            let (response, feedback) = callback(
                tool_name.to_string(),
                args.clone(),
                tool_call_id.to_string(),
            )
            .await;

            match response {
                ApprovalResponse::Yes => {
                    return Ok((true, feedback, false));
                }
                ApprovalResponse::Always => {
                    // Approve and mark for session-wide approval
                    return Ok((true, feedback, true));
                }
                ApprovalResponse::No => {
                    let reason = feedback.unwrap_or_else(|| {
                        get_user_cancellation_message(CancellationReason::OperationCancelled, None)
                            .to_string()
                    });
                    return Ok((false, Some(reason), false));
                }
            }
        }

        // No callback, deny by default
        let reason = "Tool execution not permitted - no approval callback".to_string();
        Ok((false, Some(reason), false))
    }

    fn get_available_tools(&self) -> Vec<AvailableTool> {
        self.tool_manager
            .tool_infos()
            .into_iter()
            .map(|info| AvailableTool::function(info.name, info.description, info.parameters))
            .collect()
    }

    #[allow(dead_code)]
    fn get_context(&self) -> ConversationContext<'_> {
        ConversationContext {
            messages: &self.messages,
            stats: &self.stats,
            config: &self.config,
        }
    }

    /// Compact the conversation history by asking the LLM for a summary.
    pub async fn compact(&mut self) -> VibeResult<String> {
        // Clean message history first
        self.clean_message_history().await?;

        // Save current session
        let _ = self
            .session_logger
            .save(&self.messages, &self.stats, self.auto_approve())
            .await;

        // Find the last user message
        let last_user_message = self
            .messages
            .iter()
            .rev()
            .find(|msg| msg.role == Role::User)
            .and_then(|msg| msg.content.clone());

        // Request a summary from the LLM
        let summary_request = UtilityPrompt::Compact.read();
        self.messages.push(LlmMessage::user(summary_request));
        self.stats.steps += 1;

        // Update context tokens to reflect the current conversation size
        self.refresh_context_tokens().await?;

        // Get summary from LLM
        let active_model = self.config.get_active_model()?;
        let model_config = LlmModelConfig {
            name: active_model.name.clone(),
            temperature: active_model.temperature,
            top_p: active_model.top_p,
            top_k: active_model.top_k,
            min_p: active_model.min_p,
            repeat_penalty: active_model.repeat_penalty,
            presence_penalty: active_model.presence_penalty,
            thinking_budget: active_model.thinking_budget,
        };

        let options = CompletionOptions::default();
        let result = self
            .backend
            .complete(&model_config, &self.messages, None, &options)
            .await
            .map_err(VibeError::Llm)?;

        // Update stats
        if let Some(usage) = &result.usage {
            self.stats.session_prompt_tokens += usage.prompt_tokens;
            self.stats.session_completion_tokens += usage.completion_tokens;
        }

        let mut summary_content = result.message.content.clone().unwrap_or_default();

        // Append last user message context
        if let Some(last_msg) = last_user_message {
            summary_content.push_str(&format!("\n\nLast request from user was: {}", last_msg));
        }

        // Reset messages to system + multishot examples + summary
        let system_message = self.messages.first().cloned();
        let summary_message = LlmMessage::user(&summary_content);

        self.messages = match system_message {
            Some(sys) => {
                let mut msgs = vec![sys];
                // Re-add multishot examples after system prompt
                let workdir = self.config.effective_workdir();
                msgs.extend(generate_multishot_examples(&self.tool_manager, &workdir));
                msgs.push(summary_message);
                msgs
            }
            None => vec![summary_message],
        };

        // Update context tokens to reflect the current conversation size
        let new_context = self.refresh_context_tokens().await?;

        // Ensure we don't set tokens too close to threshold to prevent infinite loops
        let threshold = self.config.auto_compact_threshold;
        if new_context >= threshold {
            // If compaction didn't reduce below threshold, reduce by 20% to prevent looping
            self.stats.context_tokens = (new_context as f64 * 0.8) as u32;
        } else {
            self.stats.context_tokens = new_context;
        }

        // Reset session
        self.reset_session();

        // Save the new session state
        let _ = self
            .session_logger
            .save(&self.messages, &self.stats, self.auto_approve())
            .await;

        // Reset middleware with compact reason
        self.middleware.reset_with_reason(ResetReason::Compact);

        Ok(summary_content)
    }

    /// Reset session ID.
    fn reset_session(&mut self) {
        self.session_logger.reset_session();
        self.session_id = self.session_logger.session_id().to_string();
    }

    /// Clear conversation history.
    pub async fn clear_history(&mut self) -> VibeResult<()> {
        // Save current session
        let _ = self
            .session_logger
            .save(&self.messages, &self.stats, self.auto_approve())
            .await;

        // Keep system message and multishot examples
        if let Some(system_msg) = self.messages.first().cloned() {
            self.messages = vec![system_msg];
            // Re-add multishot examples after system prompt
            let workdir = self.config.effective_workdir();
            self.messages
                .extend(generate_multishot_examples(&self.tool_manager, &workdir));
        }

        self.stats = AgentStats::default();
        if let Ok(model) = self.config.get_active_model() {
            self.stats
                .update_pricing(model.input_price, model.output_price);
        }

        self.middleware.reset();
        self.tool_manager.reset_all();
        self.reset_session();

        Ok(())
    }

    /// Load conversation history from previous messages.
    pub fn load_history(&mut self, messages: Vec<LlmMessage>) -> VibeResult<()> {
        // Keep system message, multishot examples, and append loaded messages
        let system_message = self.messages.first().cloned();
        self.messages = match system_message {
            Some(sys_msg) => {
                let mut new_messages = vec![sys_msg];
                // Add multishot examples first
                let workdir = self.config.effective_workdir();
                new_messages.extend(generate_multishot_examples(&self.tool_manager, &workdir));
                // Then the loaded history
                new_messages.extend(messages);
                new_messages
            }
            None => messages,
        };

        // Update stats from loaded messages (skip system + multishot examples)
        let workdir = self.config.effective_workdir();
        let multishot_count = generate_multishot_examples(&self.tool_manager, &workdir).len();
        let skip_count = 1 + multishot_count; // system + multishot
        for msg in self.messages.iter().skip(skip_count) {
            self.stats.add_message_tokens(msg);
        }

        Ok(())
    }

    /// Connect to MCP servers and register their tools.
    async fn connect_mcp_servers(&mut self) -> VibeResult<()> {
        for server_config in &self.config.mcp_servers {
            let transport: Arc<dyn paramecia_mcp::transport::Transport> = match server_config
                .transport
            {
                crate::config::McpTransport::Http => {
                    let transport = HttpTransport::new(
                        server_config.url.clone().unwrap_or_default(),
                        Some(server_config.headers.clone()),
                        Duration::from_secs(30), // Default timeout
                    )?;
                    Arc::new(transport)
                }
                crate::config::McpTransport::StreamableHttp => {
                    let transport = HttpTransport::new(
                        server_config.url.clone().unwrap_or_default(),
                        Some(server_config.headers.clone()),
                        Duration::from_secs(30), // Default timeout
                    )?;
                    Arc::new(transport)
                }
                crate::config::McpTransport::Stdio => {
                    // Convert command and args to the expected format
                    let mut command_parts = vec![server_config.command.clone().unwrap_or_default()];
                    command_parts.extend(server_config.args.clone());
                    let transport = StdioTransport::new(&command_parts).await?;
                    Arc::new(transport)
                }
            };

            let mut client = McpClient::new(transport);

            // Initialize the client
            if let Err(e) = client.initialize().await {
                println!(
                    "Warning: Failed to initialize MCP server {}: {}",
                    server_config.name, e
                );
                continue;
            }

            // List available tools
            match client.list_tools().await {
                Ok(remote_tools) => {
                    let client_arc = Arc::new(client);
                    let registered = self
                        .tool_manager
                        .register_mcp_tools(client_arc, remote_tools);
                    println!(
                        "Info: Registered {} tools from MCP server {}",
                        registered, server_config.name
                    );
                }
                Err(e) => {
                    println!(
                        "Warning: Failed to list tools from MCP server {}: {}",
                        server_config.name, e
                    );
                }
            }
        }

        Ok(())
    }
}
