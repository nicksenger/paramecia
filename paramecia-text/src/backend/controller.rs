//! Controller backend — bridges a WASM controller module to the agent system.
//!
//! Instead of running local inference, this backend forwards messages to a WASM
//! guest controller via bridge channels and streams events back.
//!
//! Architecture: a long-lived `bridge_relay` task solely owns the bridge
//! channels (`user_tx`, `agent_rx`).  `ControllerBackend` communicates with it
//! via a `BridgeRequest` channel, eliminating all shared mutable state.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use tokio_stream::wrappers::ReceiverStream;

use paramecia_bridge::{ControllerAgentEvent, ControllerBridge, ControllerUserEvent};

use super::{Backend, CompletionOptions, ModelConfig};
use crate::error::LlmResult;
use crate::types::{AvailableTool, EventStream, LlmMessage, Role, StreamEvent};

/// Internal request sent from `ControllerBackend` to the relay task.
enum BridgeRequest {
    StartStreaming {
        user_event: Option<ControllerUserEvent>,
        stream_tx: tokio::sync::mpsc::Sender<LlmResult<StreamEvent>>,
    },
}

/// Long-lived task that solely owns the bridge channels.
///
/// Runs in two phases:
/// 1. **Init**: waits for `Initialized` or `Error` from the guest, signalling
///    the result via `ready_tx`.
/// 2. **Request loop**: for each `StartStreaming` request, drains stale events,
///    sends the user event, then forwards agent events until `Done` or error.
async fn bridge_relay(
    user_tx: tokio::sync::mpsc::Sender<ControllerUserEvent>,
    mut agent_rx: tokio::sync::mpsc::Receiver<ControllerAgentEvent>,
    mut request_rx: tokio::sync::mpsc::Receiver<BridgeRequest>,
    last_context_tokens: Arc<std::sync::atomic::AtomicU32>,
    ready_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
) {
    // Phase 1 — wait for guest initialization
    let initialized = loop {
        match agent_rx.recv().await {
            Some(ControllerAgentEvent::Initialized) => {
                let _ = ready_tx.send(Ok(()));
                break true;
            }
            Some(ControllerAgentEvent::Error(e)) => {
                let _ = ready_tx.send(Err(e));
                break false;
            }
            Some(_) => continue, // ignore unexpected events during init
            None => {
                let _ = ready_tx.send(Err(
                    "Controller disconnected before initialization".to_string()
                ));
                break false;
            }
        }
    };
    if !initialized {
        return;
    }

    // Phase 2 — request/response loop
    while let Some(req) = request_rx.recv().await {
        let BridgeRequest::StartStreaming {
            user_event,
            stream_tx,
        } = req;

        // Drain events that arrived between turns.
        // Accumulate ContextTokens (e.g. from system prompt prefill) instead of
        // discarding them; ignore Initialized (already handled).
        let mut startup_error: Option<String> = None;
        while let Ok(event) = agent_rx.try_recv() {
            match event {
                ControllerAgentEvent::Error(e) => {
                    startup_error = Some(e);
                    break;
                }
                ControllerAgentEvent::ContextTokens { context_tokens } => {
                    last_context_tokens.fetch_add(context_tokens, Ordering::Relaxed);
                }
                _ => {}
            }
        }
        if let Some(e) = startup_error {
            let _ = stream_tx
                .send(Err(crate::error::LlmError::StreamError(e)))
                .await;
            continue;
        }

        // Send the user event — the guest is idle and waiting
        if let Some(event) = user_event {
            let _ = user_tx.send(event).await;
        }

        // Forward agent events until Done or error.
        // The guest sends delta token counts; we accumulate into last_context_tokens.
        // fetch_add returns the previous value, so + delta gives the new total.
        let mut finished = false;
        while let Some(event) = agent_rx.recv().await {
            let is_done = matches!(event, ControllerAgentEvent::Done);
            let stream_event = match event {
                ControllerAgentEvent::Initialized => continue, // ignore late arrival
                ControllerAgentEvent::ContextTokens { context_tokens } => {
                    let new_total = last_context_tokens
                        .fetch_add(context_tokens, Ordering::Relaxed)
                        + context_tokens;
                    StreamEvent::ContextTokens {
                        context_tokens: new_total,
                    }
                }
                ControllerAgentEvent::Token { content } => {
                    let new_total = last_context_tokens.fetch_add(1, Ordering::Relaxed) + 1;
                    StreamEvent::Token {
                        role: Role::Assistant,
                        content,
                        context_tokens: new_total,
                        tuning: None,
                    }
                }
                ControllerAgentEvent::ToolCall {
                    id,
                    name,
                    arguments,
                    n_tokens,
                } => {
                    let new_total =
                        last_context_tokens.fetch_add(n_tokens, Ordering::Relaxed) + n_tokens;
                    let args: serde_json::Value =
                        serde_json::from_str(&arguments).unwrap_or(serde_json::Value::Null);
                    StreamEvent::ToolCall {
                        id: format_uuid_tuple(id),
                        name,
                        arguments: args,
                        raw_text: None,
                        context_tokens: new_total,
                        tuning: None,
                    }
                }
                ControllerAgentEvent::Done => StreamEvent::Done { usage: None },
                ControllerAgentEvent::Error(e) => {
                    let _ = stream_tx
                        .send(Err(crate::error::LlmError::StreamError(e)))
                        .await;
                    finished = true;
                    break;
                }
            };
            if stream_tx.send(Ok(stream_event)).await.is_err() {
                break;
            }
            if is_done {
                finished = true;
                break;
            }
        }
        if !finished {
            let _ = stream_tx
                .send(Err(crate::error::LlmError::StreamError(
                    "Controller stream closed unexpectedly".to_string(),
                )))
                .await;
        }
    }
}

/// A backend that communicates with a WASM controller module via bridge channels.
pub struct ControllerBackend {
    request_tx: tokio::sync::mpsc::Sender<BridgeRequest>,
    last_context_tokens: Arc<std::sync::atomic::AtomicU32>,
}

impl ControllerBackend {
    /// Create a new ControllerBackend from a bridge endpoint.
    ///
    /// Spawns a relay task that owns the bridge channels.  Returns the backend
    /// and a oneshot receiver that resolves when the guest sends `Initialized`
    /// (Ok) or `Error` (Err).
    pub fn new(
        bridge: ControllerBridge,
    ) -> (Self, tokio::sync::oneshot::Receiver<Result<(), String>>) {
        let last_context_tokens = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let (request_tx, request_rx) = tokio::sync::mpsc::channel::<BridgeRequest>(4);
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();

        tokio::spawn(bridge_relay(
            bridge.user_tx,
            bridge.agent_rx,
            request_rx,
            Arc::clone(&last_context_tokens),
            ready_tx,
        ));

        (
            Self {
                request_tx,
                last_context_tokens,
            },
            ready_rx,
        )
    }
}

#[async_trait]
impl Backend for ControllerBackend {
    async fn complete_streaming(
        &self,
        _model: &ModelConfig,
        messages: &[LlmMessage],
        _tools: Option<&[AvailableTool]>,
        _options: &CompletionOptions,
    ) -> LlmResult<EventStream> {
        // Determine what to send based on the last message
        let user_event = messages.last().map(|last_msg| match last_msg.role {
            Role::User => {
                let content = last_msg.content.clone().unwrap_or_default();
                ControllerUserEvent::Prompt(content)
            }
            Role::Tool => {
                let id_str = last_msg.tool_call_id.clone().unwrap_or_default();
                let id = parse_uuid_tuple(&id_str);
                let name = last_msg.name.clone().unwrap_or_default();
                let content = last_msg.content.clone().unwrap_or_default();
                ControllerUserEvent::ToolResult { id, name, content }
            }
            _ => {
                let content = last_msg.content.clone().unwrap_or_default();
                ControllerUserEvent::Prefill(content)
            }
        });

        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel::<LlmResult<StreamEvent>>(32);

        // Send request to the relay — it will drive the streaming
        let _ = self
            .request_tx
            .send(BridgeRequest::StartStreaming {
                user_event,
                stream_tx,
            })
            .await;

        let stream = ReceiverStream::new(stream_rx);
        Ok(Box::pin(stream))
    }

    async fn count_tokens(
        &self,
        _model: &ModelConfig,
        _messages: &[LlmMessage],
        _tools: Option<&[AvailableTool]>,
    ) -> LlmResult<u32> {
        // Return the last context token count reported by the guest controller
        Ok(self.last_context_tokens.load(Ordering::Relaxed))
    }
}

/// Format a UUID tuple as "hi:lo" string for tool call IDs.
fn format_uuid_tuple(id: paramecia_bridge::Uuid) -> String {
    format!("{}:{}", id.0, id.1)
}

/// Parse a "hi:lo" UUID string back into a tuple.
fn parse_uuid_tuple(s: &str) -> paramecia_bridge::Uuid {
    if let Some((hi, lo)) = s.split_once(':') {
        let hi = hi.parse::<u64>().unwrap_or(0);
        let lo = lo.parse::<u64>().unwrap_or(0);
        (hi, lo)
    } else {
        (0, 0)
    }
}
