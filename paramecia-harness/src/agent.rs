//! Core agent implementation.

use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::config::ParameciaConfig;
use crate::error::{ParameciaError, ParameciaResult};
use crate::events::{
    AgentEvent, AssistantEvent, CompactEndEvent, CompactStartEvent, ContextTokensEvent,
    ToolCallEvent, ToolResultEvent,
};
use crate::middleware::{
    AutoCompactMiddleware, ContextWarningMiddleware, ConversationContext, MiddlewareAction,
    MiddlewarePipeline, PlanModeMiddleware, PriceLimitMiddleware, ResetReason, TurnLimitMiddleware,
};
use crate::modes::AgentMode;
use crate::prompts::UtilityPrompt;
use crate::session::SessionLogger;
use crate::system_prompt::get_universal_system_prompt_with_tools;
use crate::types::{AgentStats, ApprovalResponse};
use crate::utils::{
    CancellationReason, TOOL_ERROR_TAG, VIBE_STOP_EVENT_TAG, get_user_cancellation_message,
    is_user_cancellation_event,
};
use paramecia_text::backend::{
    Backend, BackendFactory, CompletionOptions, ModelConfig as LlmModelConfig,
};
use paramecia_text::format::ApiToolFormatHandler;
use paramecia_text::{
    AvailableTool, FunctionCall, LlmMessage, LlmUsage, Role, StrToolChoice, StreamEvent, ToolCall,
    ToolChoice,
};
use paramecia_tools::ToolManager;
use paramecia_tools::mcp::McpClient;
use paramecia_tools::mcp::transport::{HttpTransport, StdioTransport};
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
    config: ParameciaConfig,
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
    pub fn new(config: ParameciaConfig, mode: AgentMode) -> ParameciaResult<Self> {
        let active_model = config.get_active_model()?;
        let provider = config.get_provider_for_model(active_model)?;

        // Use top-level context_length, falling back to provider's setting
        let context_length = Some(config.context_length as usize)
            .filter(|&v| v > 0)
            .or(provider.context_length);

        let backend = BackendFactory::create(&paramecia_text::backend::ProviderConfig {
            name: provider.name.clone(),
            backend: provider.backend,
            local_model_path: provider.local_model_path.clone(),
            local_tokenizer_path: provider.local_tokenizer_path.clone(),
            local_max_tokens: provider.local_max_tokens,
            local_device: provider.local_device.clone(),
            local_offload: provider.local_offload.clone(),
            context_length,
            local_kv_cache_quant: provider.local_kv_cache_quant.clone(),
            local_layer_split: provider.local_layer_split.clone(),
            local_disable_context: provider.local_disable_context,
            tool_call_format: provider.tool_call_format.clone(),
            full_generation_output: provider.full_generation_output,
            tuning_output: provider.tuning_output,
            tuning_top_k: provider.tuning_top_k,
            tuning_tail_samples: provider.tuning_tail_samples,
        })
        .map_err(ParameciaError::Config)?;

        Self::from_backend(config, mode, backend)
    }

    /// Create a new agent using a pre-constructed backend (useful for local training).
    pub fn from_backend(
        config: ParameciaConfig,
        mode: AgentMode,
        backend: Arc<dyn Backend>,
    ) -> ParameciaResult<Self> {
        Self::from_backend_internal(config, mode, backend, true)
    }

    fn from_backend_internal(
        config: ParameciaConfig,
        mode: AgentMode,
        backend: Arc<dyn Backend>,
        initialize_system_prompt: bool,
    ) -> ParameciaResult<Self> {
        let tool_manager = ToolManager::with_configs_and_builtin_filter(
            config.tools.clone(),
            &config.builtin_tools,
            config.no_builtin_tools,
        );
        let session_logger = SessionLogger::new(config.session_logging.clone());
        let session_id = session_logger.session_id().to_string();

        // Create atomic mode for sharing with middleware
        let mode_atomic = Arc::new(std::sync::atomic::AtomicU8::new(mode as u8));

        let mut agent = Self {
            config,
            mode: mode_atomic,
            mode_value: mode,
            backend,
            tool_manager,
            format_handler: ApiToolFormatHandler::new(),
            messages: Vec::new(),
            stats: AgentStats::default(),
            middleware: MiddlewarePipeline::new(),
            session_logger,
            session_id,
            approval_callback: None,
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

        if initialize_system_prompt {
            agent.rebuild_system_prompt();
        }

        Ok(agent)
    }

    /// Create a new agent and connect to MCP servers.
    pub async fn new_with_mcp(config: ParameciaConfig, mode: AgentMode) -> ParameciaResult<Self> {
        let active_model = config.get_active_model()?;
        let provider = config.get_provider_for_model(active_model)?;

        let context_length = Some(config.context_length as usize)
            .filter(|&v| v > 0)
            .or(provider.context_length);

        let backend = BackendFactory::create(&paramecia_text::backend::ProviderConfig {
            name: provider.name.clone(),
            backend: provider.backend,
            local_model_path: provider.local_model_path.clone(),
            local_tokenizer_path: provider.local_tokenizer_path.clone(),
            local_max_tokens: provider.local_max_tokens,
            local_device: provider.local_device.clone(),
            local_offload: provider.local_offload.clone(),
            context_length,
            local_kv_cache_quant: provider.local_kv_cache_quant.clone(),
            local_layer_split: provider.local_layer_split.clone(),
            local_disable_context: provider.local_disable_context,
            tool_call_format: provider.tool_call_format.clone(),
            full_generation_output: provider.full_generation_output,
            tuning_output: provider.tuning_output,
            tuning_top_k: provider.tuning_top_k,
            tuning_tail_samples: provider.tuning_tail_samples,
        })
        .map_err(ParameciaError::Config)?;

        let mut agent = Self::from_backend_internal(config, mode, backend, false)?;

        // Connect to MCP servers and register their tools
        if let Err(e) = agent.connect_mcp_servers().await {
            println!("Warning: Failed to connect to MCP servers: {}", e);
        }

        agent.rebuild_system_prompt();

        Ok(agent)
    }

    /// Create an agent with additional options.
    pub async fn with_options(
        config: ParameciaConfig,
        mode: AgentMode,
        max_turns: Option<u32>,
        max_price: Option<f64>,
    ) -> ParameciaResult<Self> {
        let active_model = config.get_active_model()?;
        let provider = config.get_provider_for_model(active_model)?;

        let context_length = Some(config.context_length as usize)
            .filter(|&v| v > 0)
            .or(provider.context_length);

        let backend = BackendFactory::create(&paramecia_text::backend::ProviderConfig {
            name: provider.name.clone(),
            backend: provider.backend,
            local_model_path: provider.local_model_path.clone(),
            local_tokenizer_path: provider.local_tokenizer_path.clone(),
            local_max_tokens: provider.local_max_tokens,
            local_device: provider.local_device.clone(),
            local_offload: provider.local_offload.clone(),
            context_length,
            local_kv_cache_quant: provider.local_kv_cache_quant.clone(),
            local_layer_split: provider.local_layer_split.clone(),
            local_disable_context: provider.local_disable_context,
            tool_call_format: provider.tool_call_format.clone(),
            full_generation_output: provider.full_generation_output,
            tuning_output: provider.tuning_output,
            tuning_top_k: provider.tuning_top_k,
            tuning_tail_samples: provider.tuning_tail_samples,
        })
        .map_err(ParameciaError::Config)?;

        let mut agent = Self::from_backend_internal(config, mode, backend, false)?;
        agent.max_turns = max_turns;
        agent.max_price = max_price;
        agent.setup_middleware(max_turns, max_price);

        // Connect to MCP servers and register their tools
        if let Err(e) = agent.connect_mcp_servers().await {
            println!("Warning: Failed to connect to MCP servers: {}", e);
        }

        agent.rebuild_system_prompt();

        Ok(agent)
    }

    /// Create an agent with a custom backend and additional options.
    pub async fn with_backend_options(
        config: ParameciaConfig,
        mode: AgentMode,
        backend: Arc<dyn Backend>,
        max_turns: Option<u32>,
        max_price: Option<f64>,
    ) -> ParameciaResult<Self> {
        let mut agent = Self::from_backend_internal(config, mode, backend, false)?;
        agent.max_turns = max_turns;
        agent.max_price = max_price;
        agent.setup_middleware(max_turns, max_price);

        // Connect to MCP servers and register their tools
        if let Err(e) = agent.connect_mcp_servers().await {
            println!("Warning: Failed to connect to MCP servers: {}", e);
        }

        agent.rebuild_system_prompt();

        Ok(agent)
    }

    fn rebuild_system_prompt(&mut self) {
        let system_prompt =
            get_universal_system_prompt_with_tools(&self.tool_manager, &self.config);
        tracing::info!(
            "System prompt built: {} chars (~{} tokens)",
            system_prompt.len(),
            system_prompt.len() / 4
        );
        Self::set_system_prompt_message(&mut self.messages, system_prompt);
    }

    fn set_system_prompt_message(messages: &mut Vec<LlmMessage>, system_prompt: String) {
        let has_system = messages
            .first()
            .map(|message| message.role == Role::System)
            .unwrap_or(false);

        if system_prompt.trim().is_empty() {
            if has_system {
                messages.remove(0);
            }
            return;
        }

        let system_message = LlmMessage::system(system_prompt);
        if has_system {
            messages[0] = system_message;
        } else {
            messages.insert(0, system_message);
        }
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
    fn active_model_config(&self) -> ParameciaResult<LlmModelConfig> {
        let active_model = self.config.get_active_model()?;
        Ok(LlmModelConfig {
            name: active_model.name.clone(),
            temperature: active_model.temperature,
            top_p: active_model.top_p,
            top_k: active_model.top_k,
            min_p: active_model.min_p,
            repeat_penalty: active_model.repeat_penalty,
            presence_penalty: active_model.presence_penalty,
        })
    }

    /// Recalculate context tokens using the backend tokenizer.
    async fn refresh_context_tokens(&mut self) -> ParameciaResult<u32> {
        let model_config = self.active_model_config()?;
        let token_count = self
            .backend
            .count_tokens(&model_config, &self.messages, None)
            .await
            .map_err(ParameciaError::Llm)?;
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
    ) -> ParameciaResult<()> {
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
    ) -> ParameciaResult<()> {
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
    async fn clean_message_history(&mut self) -> ParameciaResult<()> {
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
    async fn ensure_assistant_after_tools(&mut self) -> ParameciaResult<()> {
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
    ) -> ParameciaResult<()> {
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
                                context_tokens: Some(self.stats.context_tokens),
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
    ) -> ParameciaResult<(bool, bool)> {
        let turn_start_context_tokens = self.stats.context_tokens;
        let active_model = self.config.get_active_model()?;
        let model_config = LlmModelConfig {
            name: active_model.name.clone(),
            temperature: active_model.temperature,
            top_p: active_model.top_p,
            top_k: active_model.top_k,
            min_p: active_model.min_p,
            repeat_penalty: active_model.repeat_penalty,
            presence_penalty: active_model.presence_penalty,
        };

        // Get available tools
        let tools = self.get_available_tools();

        let tool_choice = if tools.is_empty() {
            None
        } else {
            Some(ToolChoice::String(StrToolChoice::Auto))
        };
        let options = CompletionOptions {
            tool_choice,
            ..Default::default()
        };
        let tool_slice = if tools.is_empty() {
            None
        } else {
            Some(&tools[..])
        };

        let start_time = std::time::Instant::now();

        let mut stream = self
            .backend
            .complete_streaming(&model_config, &self.messages, tool_slice, &options)
            .await
            .map_err(ParameciaError::Llm)?;

        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut usage: Option<LlmUsage> = None;
        let mut got_done = false;

        while let Some(event) = stream.next().await {
            // Check for cancellation periodically
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                return Ok((false, false));
            }

            let event = event.map_err(ParameciaError::Llm)?;

            match event {
                StreamEvent::ContextTokens { context_tokens } => {
                    let clamped_context_tokens = context_tokens.max(self.stats.context_tokens);
                    self.stats.context_tokens = clamped_context_tokens;
                    let _ = tx
                        .send(AgentEvent::ContextTokens(ContextTokensEvent {
                            context_tokens: clamped_context_tokens,
                        }))
                        .await;
                }
                StreamEvent::Token {
                    content,
                    context_tokens,
                    ..
                } => {
                    full_content.push_str(&content);
                    let clamped_context_tokens = context_tokens.max(self.stats.context_tokens);
                    self.stats.context_tokens = clamped_context_tokens;
                    // Forward immediately to UI — no batching
                    let _ = tx
                        .send(AgentEvent::Assistant(AssistantEvent {
                            content,
                            context_tokens: Some(clamped_context_tokens),
                            stopped_by_middleware: false,
                        }))
                        .await;
                }
                StreamEvent::ToolCall {
                    id,
                    name,
                    arguments,
                    raw_text,
                    context_tokens,
                    ..
                } => {
                    let clamped_context_tokens = context_tokens.max(self.stats.context_tokens);
                    self.stats.context_tokens = clamped_context_tokens;
                    // Send tool call event to UI
                    let _ = tx
                        .send(AgentEvent::ToolCall(ToolCallEvent {
                            tool_name: name.clone(),
                            tool_info: None,
                            args: arguments.clone(),
                            tool_call_id: id.clone(),
                            context_tokens: Some(clamped_context_tokens),
                        }))
                        .await;

                    // Collect into tool_calls vec for message history
                    let idx = tool_calls.len();
                    tool_calls.push(ToolCall {
                        id: Some(id),
                        index: Some(idx),
                        function: FunctionCall {
                            name: Some(name),
                            arguments: Some(serde_json::to_string(&arguments).unwrap_or_default()),
                        },
                        r#type: "function".to_string(),
                        raw_text,
                    });
                }
                StreamEvent::Done { usage: u } => {
                    usage = u;
                    got_done = true;
                }
            }
        }

        // Build the final message
        let has_content = !full_content.is_empty();
        let has_tool_calls = !tool_calls.is_empty();

        let final_message = if has_content || has_tool_calls {
            Some(LlmMessage {
                role: Role::Assistant,
                content: if full_content.is_empty() {
                    None
                } else {
                    Some(full_content)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                name: None,
                tool_call_id: None,
            })
        } else {
            None
        };

        let duration = start_time.elapsed().as_secs_f64();

        // Track finish_reason — Done event means generation completed
        self.last_finish_reason = if got_done {
            Some("stop".to_string())
        } else {
            None
        };

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
        let context_tokens = self
            .refresh_context_tokens()
            .await?
            .max(turn_start_context_tokens);
        self.stats.context_tokens = context_tokens;
        let _ = tx
            .send(AgentEvent::ContextTokens(ContextTokensEvent {
                context_tokens,
            }))
            .await;

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
                    let tool_result = self.execute_tool(tool_name, &args, &tool_call_id, tx).await;

                    // Format tool result based on execution outcome
                    let response_content = match &tool_result {
                        ToolExecutionResult::Success(value) => {
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
                            if is_user_cancellation_event(Some(skip_reason)) {
                                user_cancelled = true;
                            }
                            skip_reason.clone()
                        }
                        ToolExecutionResult::Failed(error_msg) => {
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
                    let context_tokens = self
                        .refresh_context_tokens()
                        .await?
                        .max(self.stats.context_tokens);
                    self.stats.context_tokens = context_tokens;
                    let _ = tx
                        .send(AgentEvent::ContextTokens(ContextTokensEvent {
                            context_tokens,
                        }))
                        .await;
                }
            }

            // Continue the loop if we had tool calls (unless user cancelled)
            return Ok((true, user_cancelled));
        }

        // Break if we received Done (generation completed)
        let should_break = self.last_finish_reason.is_some();
        Ok((!should_break, false))
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
                        context_tokens: None,
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
                            context_tokens: None,
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
                        context_tokens: None,
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
                        context_tokens: None,
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
                        context_tokens: None,
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
    ) -> ParameciaResult<(bool, Option<String>, bool)> {
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
        let disabled_tools = self
            .config
            .disabled_tools
            .iter()
            .map(|name| name.as_str())
            .collect::<std::collections::HashSet<_>>();
        if disabled_tools.contains("*") {
            return Vec::new();
        }

        let enabled_tools = self
            .config
            .enabled_tools
            .iter()
            .map(|name| name.as_str())
            .collect::<std::collections::HashSet<_>>();

        self.tool_manager
            .tool_infos()
            .into_iter()
            .filter(|info| {
                if !enabled_tools.is_empty() && !enabled_tools.contains(info.name.as_str()) {
                    return false;
                }
                !disabled_tools.contains(info.name.as_str())
            })
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
    pub async fn compact(&mut self) -> ParameciaResult<String> {
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
        };

        let options = CompletionOptions::default();
        let mut stream = self
            .backend
            .complete_streaming(&model_config, &self.messages, None, &options)
            .await
            .map_err(ParameciaError::Llm)?;

        let mut summary_content = String::new();
        while let Some(event) = stream.next().await {
            match event.map_err(ParameciaError::Llm)? {
                StreamEvent::Token { content, .. } => {
                    summary_content.push_str(&content);
                }
                StreamEvent::ContextTokens { .. } => {}
                StreamEvent::Done { usage } => {
                    if let Some(usage) = &usage {
                        self.stats.session_prompt_tokens += usage.prompt_tokens;
                        self.stats.session_completion_tokens += usage.completion_tokens;
                    }
                }
                StreamEvent::ToolCall { .. } => {}
            }
        }

        // Append last user message context
        if let Some(last_msg) = last_user_message {
            summary_content.push_str(&format!("\n\nLast request from user was: {}", last_msg));
        }

        // Reset messages to system prompt + summary
        let system_message = self.messages.first().cloned();
        let summary_message = LlmMessage::user(&summary_content);

        self.messages = match system_message {
            Some(sys) => vec![sys, summary_message],
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
    pub async fn clear_history(&mut self) -> ParameciaResult<()> {
        // Save current session
        let _ = self
            .session_logger
            .save(&self.messages, &self.stats, self.auto_approve())
            .await;

        // Keep system message
        if let Some(system_msg) = self.messages.first().cloned() {
            self.messages = vec![system_msg];
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

    /// Save current model state to snapshot file.
    ///
    /// Only supported for LocalBackend. Returns an error for other backends.
    ///
    /// # Arguments
    /// * `snapshot_name` - Name of the snapshot file (will be saved in ~/.paramecia/snapshots/)
    ///
    /// # Returns
    /// * Path to the saved snapshot file
    pub async fn save_snapshot(&self, snapshot_name: &str) -> ParameciaResult<std::path::PathBuf> {
        // Downcast backend to LocalBackend
        use paramecia_text::backend::LocalBackend;
        let backend_any = &self.backend as &dyn std::any::Any;

        if let Some(local_backend) = backend_any.downcast_ref::<LocalBackend>() {
            let path = local_backend
                .save_snapshot(snapshot_name)
                .await
                .map_err(|e| {
                    crate::error::ParameciaError::AgentState(format!(
                        "Failed to save snapshot: {}",
                        e
                    ))
                })?;

            tracing::info!("Snapshot saved: {}", path.display());
            Ok(path)
        } else {
            Err(crate::error::ParameciaError::AgentState(
                "Snapshotting is only supported for LocalBackend".to_string(),
            ))
        }
    }

    /// Load conversation history from previous messages.
    pub fn load_history(&mut self, messages: Vec<LlmMessage>) -> ParameciaResult<()> {
        // Keep system message and append loaded messages
        let system_message = self.messages.first().cloned();
        self.messages = match system_message {
            Some(sys_msg) => {
                let mut new_messages = vec![sys_msg];
                new_messages.extend(messages);
                new_messages
            }
            None => messages,
        };

        // Update stats from loaded messages (skip system prompt)
        for msg in self.messages.iter().skip(1) {
            self.stats.add_message_tokens(msg);
        }

        Ok(())
    }

    /// Connect to MCP servers and register their tools.
    async fn connect_mcp_servers(&mut self) -> ParameciaResult<()> {
        for server_config in &self.config.mcp_servers {
            let transport: Arc<dyn paramecia_tools::mcp::transport::Transport> = match server_config
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
                    let _registered = self
                        .tool_manager
                        .register_mcp_tools(client_arc, remote_tools);
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

#[cfg(test)]
mod tests {
    use super::Agent;
    use paramecia_text::{LlmMessage, Role};

    #[test]
    fn set_system_prompt_replaces_existing_system_message() {
        let mut messages = vec![LlmMessage::system("old"), LlmMessage::user("hello")];

        Agent::set_system_prompt_message(&mut messages, "new".to_string());

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content.as_deref(), Some("new"));
        assert_eq!(messages[1].role, Role::User);
    }

    #[test]
    fn set_system_prompt_inserts_when_missing() {
        let mut messages = vec![LlmMessage::user("hello")];

        Agent::set_system_prompt_message(&mut messages, "new".to_string());

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content.as_deref(), Some("new"));
        assert_eq!(messages[1].role, Role::User);
    }
}
