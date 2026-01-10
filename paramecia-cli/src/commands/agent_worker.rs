//! Background agent worker with message-passing concurrency.
//!
//! The agent runs in a dedicated thread with its own tokio runtime,
//! communicating with the UI thread via channels. This keeps the UI
//! responsive even during heavy inference operations.

use std::sync::Arc;
use std::thread;

use paramecia_harness::events::AgentEvent;
use paramecia_harness::modes::AgentMode;
use paramecia_harness::types::ApprovalResponse;
use paramecia_harness::{Agent, VibeConfig};
use tokio::sync::{Mutex, mpsc, oneshot};

/// Deliver an approval response to the pending callback, if any.
async fn send_approval_response(
    pending_approval: &PendingApproval,
    response: ApprovalResponse,
    feedback: Option<String>,
) {
    let mut guard = pending_approval.lock().await;
    if let Some(tx) = guard.take() {
        let _ = tx.send((response, feedback));
    }
}

/// Commands sent from UI to agent worker.
#[derive(Debug)]
pub enum AgentCommand {
    /// Process a user message.
    Act { prompt: String },
    /// Clear conversation history.
    Clear,
    /// Compact the conversation.
    Compact,
    /// Respond to an approval request.
    ApprovalResponse {
        response: ApprovalResponse,
        feedback: Option<String>,
    },
    /// Interrupt current operation.
    Interrupt,
    /// Change the agent mode.
    SetMode(AgentMode),
    /// Shutdown the worker.
    Shutdown,
}

/// Results/events sent from agent worker to UI.
#[derive(Debug)]
pub enum AgentResult {
    /// Agent is ready (initial loading complete).
    Ready { session_id: String },
    /// Agent failed to initialize.
    InitError(String),
    /// An agent event (streaming content, tool calls, etc.).
    Event(AgentEvent),
    /// Agent task completed successfully.
    Done { context_tokens: u32 },
    /// Agent task failed.
    Error(String),
    /// Approval requested from user.
    ApprovalNeeded {
        tool_name: String,
        args: serde_json::Value,
        tool_call_id: String,
    },
    /// Clear completed.
    Cleared,
    /// Compact completed.
    Compacted { old_tokens: u32, new_tokens: u32 },
    /// Compact failed.
    CompactError(String),
    /// Worker is shutting down.
    ShutdownAck,
}

/// Handle to communicate with the agent worker.
pub struct AgentHandle {
    cmd_tx: mpsc::Sender<AgentCommand>,
    result_rx: mpsc::Receiver<AgentResult>,
    worker_thread: Option<thread::JoinHandle<()>>,
}

impl AgentHandle {
    /// Spawn a new agent worker in a background thread.
    pub fn spawn(config: VibeConfig, mode: AgentMode) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<AgentCommand>(32);
        let (result_tx, result_rx) = mpsc::channel::<AgentResult>(256);

        let worker_thread = thread::spawn(move || {
            // Create a dedicated runtime for this thread
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create worker runtime");

            rt.block_on(async move {
                run_worker(config, mode, cmd_rx, result_tx).await;
            });
        });

        Self {
            cmd_tx,
            result_rx,
            worker_thread: Some(worker_thread),
        }
    }

    /// Send a command to the worker.
    pub async fn send(
        &self,
        cmd: AgentCommand,
    ) -> Result<(), mpsc::error::SendError<AgentCommand>> {
        self.cmd_tx.send(cmd).await
    }

    /// Try to receive a result without blocking.
    pub fn try_recv(&mut self) -> Option<AgentResult> {
        self.result_rx.try_recv().ok()
    }

    /// Shutdown the worker and wait for it to finish.
    pub async fn shutdown(mut self) {
        let _ = self.cmd_tx.send(AgentCommand::Shutdown).await;
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
}

// Shared state for pending approval response
type PendingApproval = Arc<Mutex<Option<oneshot::Sender<(ApprovalResponse, Option<String>)>>>>;

/// The actual worker loop that runs in the background thread.
async fn run_worker(
    config: VibeConfig,
    mode: AgentMode,
    mut cmd_rx: mpsc::Receiver<AgentCommand>,
    result_tx: mpsc::Sender<AgentResult>,
) {
    // Create the agent (this is the slow part - model loading)
    let mut agent = match Agent::with_options(config, mode, None, None, true).await {
        Ok(agent) => {
            let session_id = agent.session_id().to_string();
            let _ = result_tx.send(AgentResult::Ready { session_id }).await;
            agent
        }
        Err(e) => {
            let _ = result_tx.send(AgentResult::InitError(e.to_string())).await;
            return;
        }
    };

    // Shared state for the pending approval
    let pending_approval: PendingApproval = Arc::new(Mutex::new(None));

    // Set up approval callback
    {
        let result_tx = result_tx.clone();
        let pending_approval = Arc::clone(&pending_approval);

        agent.set_approval_callback(Arc::new(move |tool_name, args, tool_call_id| {
            let result_tx = result_tx.clone();
            let pending_approval = Arc::clone(&pending_approval);

            Box::pin(async move {
                // Create a oneshot channel for this specific approval
                let (resp_tx, resp_rx) = oneshot::channel();

                // Store the sender so the main loop can respond
                {
                    let mut guard = pending_approval.lock().await;
                    *guard = Some(resp_tx);
                }

                // Send approval request to UI
                let _ = result_tx
                    .send(AgentResult::ApprovalNeeded {
                        tool_name,
                        args,
                        tool_call_id,
                    })
                    .await;

                // Wait for response (with timeout)
                match tokio::time::timeout(std::time::Duration::from_secs(300), resp_rx).await {
                    Ok(Ok((response, feedback))) => (response, feedback),
                    _ => (
                        ApprovalResponse::No,
                        Some("Approval timeout or cancelled".to_string()),
                    ),
                }
            })
        }));
    }

    loop {
        // Wait for next command
        let Some(cmd) = cmd_rx.recv().await else {
            break;
        };

        match cmd {
            AgentCommand::Act { prompt } => {
                // Create event channel for this action
                let (event_tx, mut event_rx) = mpsc::channel::<AgentEvent>(64);
                let result_tx_clone = result_tx.clone();

                // Create a cancellation flag that can be accessed without borrowing agent
                let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let cancelled_for_agent = Arc::clone(&cancelled);

                // Spawn event forwarder
                let forward_handle = tokio::spawn(async move {
                    while let Some(event) = event_rx.recv().await {
                        if result_tx_clone
                            .send(AgentResult::Event(event))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                });

                // Run the agent (this is where inference happens)
                let mut act_future =
                    Box::pin(agent.act_with_cancellation(&prompt, event_tx, cancelled_for_agent));
                let mut act_result: Option<Result<(), paramecia_harness::error::VibeError>> = None;
                let mut shutdown_requested = false;
                let mut interrupted = false;

                loop {
                    tokio::select! {
                        result = &mut act_future => {
                            // Action finished, handle completion below
                            act_result = Some(result);
                            break;
                        }
                        Some(inner_cmd) = cmd_rx.recv() => {
                            match inner_cmd {
                                AgentCommand::ApprovalResponse { response, feedback } => {
                                    send_approval_response(&pending_approval, response, feedback)
                                        .await;
                                }
                                AgentCommand::Interrupt => {
                                    // Cancel the ongoing agent action
                                    interrupted = true;
                                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                AgentCommand::Shutdown => {
                                    shutdown_requested = true;
                                }
                                _ => {
                                    // Drop any other command while the agent is busy
                                }
                            }
                        }
                        else => {
                            // Channel closed; abort waiting
                            break;
                        }
                    }
                }

                // Wait for event forwarding to complete
                let _ = forward_handle.await;

                // Drop the future to release the mutable borrow on agent before reading stats
                drop(act_future);

                match act_result {
                    Some(Ok(())) => {
                        if interrupted {
                            let _ = result_tx
                                .send(AgentResult::Error(
                                    "Agent action was interrupted by user".to_string(),
                                ))
                                .await;
                        } else {
                            let _ = result_tx
                                .send(AgentResult::Done {
                                    context_tokens: agent.stats().context_tokens,
                                })
                                .await;
                        }
                    }
                    Some(Err(e)) => {
                        let _ = result_tx.send(AgentResult::Error(e.to_string())).await;
                    }
                    None => {
                        let _ = result_tx
                            .send(AgentResult::Error("Agent action was cancelled".to_string()))
                            .await;
                    }
                }

                if shutdown_requested {
                    let _ = result_tx.send(AgentResult::ShutdownAck).await;
                    break;
                }
            }

            AgentCommand::Clear => {
                if let Err(e) = agent.clear_history().await {
                    let _ = result_tx.send(AgentResult::Error(e.to_string())).await;
                } else {
                    let _ = result_tx.send(AgentResult::Cleared).await;
                }
            }

            AgentCommand::Compact => {
                let old_tokens = agent.stats().context_tokens;
                match agent.compact().await {
                    Ok(_) => {
                        let new_tokens = agent.stats().context_tokens;
                        let _ = result_tx
                            .send(AgentResult::Compacted {
                                old_tokens,
                                new_tokens,
                            })
                            .await;
                    }
                    Err(e) => {
                        let _ = result_tx
                            .send(AgentResult::CompactError(e.to_string()))
                            .await;
                    }
                }
            }

            AgentCommand::ApprovalResponse { response, feedback } => {
                // Send response to the waiting approval callback
                let mut guard = pending_approval.lock().await;
                if let Some(tx) = guard.take() {
                    let _ = tx.send((response, feedback));
                }
            }

            AgentCommand::Interrupt => {
                // Interrupt is handled during Act commands - no-op in other states
                // The UI will show the interrupt message immediately
            }

            AgentCommand::SetMode(new_mode) => {
                agent.set_mode(new_mode);
            }

            AgentCommand::Shutdown => {
                let _ = result_tx.send(AgentResult::ShutdownAck).await;
                break;
            }
        }
    }
}
