//! Middleware for processing agent conversation turns.

use crate::config::VibeConfig;
use crate::modes::AgentMode;
use crate::types::AgentStats;
use crate::utils::VIBE_WARNING_TAG;
use paramecia_llm::LlmMessage;
use std::sync::Arc;

/// Context passed to middleware.
pub struct ConversationContext<'a> {
    /// Current messages in the conversation.
    pub messages: &'a [LlmMessage],
    /// Current agent stats.
    pub stats: &'a AgentStats,
    /// Agent configuration.
    pub config: &'a VibeConfig,
}

/// Reason for resetting middleware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetReason {
    /// Stop requested.
    Stop,
    /// Compact performed.
    Compact,
}

/// Action returned by middleware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MiddlewareAction {
    /// Continue with normal processing.
    Continue,
    /// Stop the conversation.
    Stop,
    /// Inject a message into the conversation.
    InjectMessage,
    /// Trigger conversation compaction.
    Compact,
}

/// Result from middleware processing.
#[derive(Debug, Clone)]
pub struct MiddlewareResult {
    /// Action to take.
    pub action: MiddlewareAction,
    /// Optional reason for the action.
    pub reason: Option<String>,
    /// Optional message to inject.
    pub message: Option<String>,
    /// Optional metadata.
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for MiddlewareResult {
    fn default() -> Self {
        Self {
            action: MiddlewareAction::Continue,
            reason: None,
            message: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

impl MiddlewareResult {
    /// Create a continue result.
    #[must_use]
    pub fn continue_() -> Self {
        Self::default()
    }

    /// Create a stop result.
    #[must_use]
    pub fn stop(reason: impl Into<String>) -> Self {
        Self {
            action: MiddlewareAction::Stop,
            reason: Some(reason.into()),
            ..Default::default()
        }
    }

    /// Create a compact result.
    #[must_use]
    pub fn compact() -> Self {
        Self {
            action: MiddlewareAction::Compact,
            ..Default::default()
        }
    }

    /// Create an inject message result.
    #[must_use]
    pub fn inject_message(message: impl Into<String>) -> Self {
        Self {
            action: MiddlewareAction::InjectMessage,
            message: Some(message.into()),
            ..Default::default()
        }
    }
}

/// Trait for middleware implementations.
pub trait Middleware: Send + Sync {
    /// Called before each turn.
    fn before_turn(&mut self, context: &ConversationContext) -> MiddlewareResult {
        let _ = context;
        MiddlewareResult::continue_()
    }

    /// Called after each turn.
    fn after_turn(&mut self, context: &ConversationContext) -> MiddlewareResult {
        let _ = context;
        MiddlewareResult::continue_()
    }

    /// Reset the middleware state.
    fn reset(&mut self, _reason: ResetReason) {}
}

/// Middleware that limits the number of turns.
pub struct TurnLimitMiddleware {
    max_turns: u32,
}

impl TurnLimitMiddleware {
    /// Create a new turn limit middleware.
    #[must_use]
    pub fn new(max_turns: u32) -> Self {
        Self { max_turns }
    }
}

impl Middleware for TurnLimitMiddleware {
    fn before_turn(&mut self, context: &ConversationContext) -> MiddlewareResult {
        // Use steps from stats, subtracting 1 to match Python behavior
        // (steps is incremented before this check in Python)
        if context.stats.steps.saturating_sub(1) >= self.max_turns {
            MiddlewareResult::stop(format!("Turn limit of {} reached", self.max_turns))
        } else {
            MiddlewareResult::continue_()
        }
    }

    fn reset(&mut self, _reason: ResetReason) {
        // No state to reset - uses stats.steps
    }
}

/// Middleware that limits the session cost.
pub struct PriceLimitMiddleware {
    max_price: f64,
}

impl PriceLimitMiddleware {
    /// Create a new price limit middleware.
    #[must_use]
    pub fn new(max_price: f64) -> Self {
        Self { max_price }
    }
}

impl Middleware for PriceLimitMiddleware {
    fn before_turn(&mut self, context: &ConversationContext) -> MiddlewareResult {
        let cost = context.stats.session_cost();
        if cost > self.max_price {
            MiddlewareResult::stop(format!(
                "Price limit exceeded: ${:.4} > ${:.2}",
                cost, self.max_price
            ))
        } else {
            MiddlewareResult::continue_()
        }
    }

    fn reset(&mut self, _reason: ResetReason) {}
}

/// Middleware that triggers auto-compaction.
pub struct AutoCompactMiddleware {
    threshold: u32,
}

impl AutoCompactMiddleware {
    /// Create a new auto-compact middleware.
    #[must_use]
    pub fn new(threshold: u32) -> Self {
        Self { threshold }
    }
}

impl Middleware for AutoCompactMiddleware {
    fn before_turn(&mut self, context: &ConversationContext) -> MiddlewareResult {
        if context.stats.context_tokens >= self.threshold {
            let mut result = MiddlewareResult::compact();
            result.metadata.insert(
                "old_tokens".to_string(),
                serde_json::Value::Number(context.stats.context_tokens.into()),
            );
            result.metadata.insert(
                "threshold".to_string(),
                serde_json::Value::Number(self.threshold.into()),
            );
            result
        } else {
            MiddlewareResult::continue_()
        }
    }

    fn reset(&mut self, _reason: ResetReason) {
        // Prevent immediate re-triggering after compaction by adding a small buffer
        // This helps avoid infinite loops where compaction doesn't reduce tokens enough
    }
}

/// Middleware that warns when context usage is high.
pub struct ContextWarningMiddleware {
    threshold_percent: f64,
    max_context: Option<u32>,
    has_warned: bool,
}

impl ContextWarningMiddleware {
    /// Create a new context warning middleware.
    #[must_use]
    pub fn new(threshold_percent: f64, max_context: u32) -> Self {
        Self {
            threshold_percent,
            max_context: Some(max_context),
            has_warned: false,
        }
    }
}

impl Middleware for ContextWarningMiddleware {
    fn before_turn(&mut self, context: &ConversationContext) -> MiddlewareResult {
        if self.has_warned {
            return MiddlewareResult::continue_();
        }

        let Some(max_context) = self.max_context else {
            return MiddlewareResult::continue_();
        };

        let threshold_tokens = (max_context as f64 * self.threshold_percent) as u32;
        if context.stats.context_tokens >= threshold_tokens {
            self.has_warned = true;

            let percentage_used =
                (context.stats.context_tokens as f64 / max_context as f64) * 100.0;
            let warning_msg = format!(
                "<{}>You have used {:.0}% of your total context ({}/{} tokens)</{}>",
                VIBE_WARNING_TAG,
                percentage_used,
                context.stats.context_tokens,
                max_context,
                VIBE_WARNING_TAG
            );

            MiddlewareResult::inject_message(warning_msg)
        } else {
            MiddlewareResult::continue_()
        }
    }

    fn reset(&mut self, _reason: ResetReason) {
        self.has_warned = false;
    }
}

/// Plan mode reminder message.
pub const PLAN_MODE_REMINDER: &str = concat!(
    "<vibe_warning>Plan mode is active. The user indicated that they do not want you to execute yet -- ",
    "you MUST NOT make any edits, run any non-readonly tools (including changing configs or making commits), ",
    "or otherwise make any changes to the system. This supersedes any other instructions you have received ",
    "(for example, to make edits). Instead, you should:\n",
    "1. Answer the user's query comprehensively\n",
    "2. When you're done researching, present your plan by giving the full plan and not doing further tool calls ",
    "to return input to the user. Do NOT make any file changes or run any tools that modify the system state ",
    "in any way until the user has confirmed the plan.</vibe_warning>"
);

/// Middleware that injects plan mode reminders.
pub struct PlanModeMiddleware {
    mode_getter: Arc<dyn Fn() -> AgentMode + Send + Sync>,
    reminder: String,
}

impl PlanModeMiddleware {
    /// Create a new plan mode middleware.
    #[must_use]
    pub fn new(mode_getter: Arc<dyn Fn() -> AgentMode + Send + Sync>) -> Self {
        Self {
            mode_getter,
            reminder: PLAN_MODE_REMINDER.to_string(),
        }
    }

    /// Create with a custom reminder message.
    #[must_use]
    pub fn with_reminder(
        mode_getter: Arc<dyn Fn() -> AgentMode + Send + Sync>,
        reminder: impl Into<String>,
    ) -> Self {
        Self {
            mode_getter,
            reminder: reminder.into(),
        }
    }

    fn is_plan_mode(&self) -> bool {
        (self.mode_getter)() == AgentMode::Plan
    }
}

impl Middleware for PlanModeMiddleware {
    fn before_turn(&mut self, _context: &ConversationContext) -> MiddlewareResult {
        if self.is_plan_mode() {
            MiddlewareResult::inject_message(self.reminder.clone())
        } else {
            MiddlewareResult::continue_()
        }
    }

    fn reset(&mut self, _reason: ResetReason) {}
}

/// Pipeline for running multiple middleware.
#[derive(Default)]
pub struct MiddlewarePipeline {
    middleware: Vec<Box<dyn Middleware>>,
}

impl MiddlewarePipeline {
    /// Create a new empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add middleware to the pipeline.
    pub fn add(&mut self, middleware: impl Middleware + 'static) {
        self.middleware.push(Box::new(middleware));
    }

    /// Clear all middleware.
    pub fn clear(&mut self) {
        self.middleware.clear();
    }

    /// Run before_turn on all middleware, collecting inject messages.
    pub fn run_before_turn(&mut self, context: &ConversationContext) -> MiddlewareResult {
        let mut messages_to_inject = Vec::new();

        for m in &mut self.middleware {
            let result = m.before_turn(context);
            match result.action {
                MiddlewareAction::InjectMessage => {
                    if let Some(msg) = result.message {
                        messages_to_inject.push(msg);
                    }
                }
                MiddlewareAction::Stop | MiddlewareAction::Compact => {
                    return result;
                }
                MiddlewareAction::Continue => {}
            }
        }

        if !messages_to_inject.is_empty() {
            let combined_message = messages_to_inject.join("\n\n");
            MiddlewareResult::inject_message(combined_message)
        } else {
            MiddlewareResult::continue_()
        }
    }

    /// Run after_turn on all middleware.
    pub fn run_after_turn(&mut self, context: &ConversationContext) -> MiddlewareResult {
        for m in &mut self.middleware {
            let result = m.after_turn(context);
            match result.action {
                MiddlewareAction::InjectMessage => {
                    // InjectMessage not allowed in after_turn
                    panic!("InjectMessage not allowed in after_turn");
                }
                MiddlewareAction::Stop | MiddlewareAction::Compact => {
                    return result;
                }
                MiddlewareAction::Continue => {}
            }
        }
        MiddlewareResult::continue_()
    }

    /// Reset all middleware.
    pub fn reset(&mut self) {
        self.reset_with_reason(ResetReason::Stop);
    }

    /// Reset all middleware with a specific reason.
    pub fn reset_with_reason(&mut self, reason: ResetReason) {
        for m in &mut self.middleware {
            m.reset(reason);
        }
    }
}
