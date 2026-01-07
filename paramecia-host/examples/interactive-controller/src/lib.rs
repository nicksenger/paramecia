//! Interactive WASM controller — minimal backend for the agentic TUI.
//!
//! Demonstrates the interactive interface using an actor pattern:
//! 1. Calls interactive::connect() to establish bidirectional communication
//! 2. Loads the model
//! 3. Runs three concurrent actors via futures::join!:
//!    - Writer: drains internal event channel → agent output stream
//!    - User reader: reads user events → dispatches commands / sets cancel flag
//!    - Inference runner: processes fill-context / predict-completion, checks cancel
//!
//! Every fill-context progress update is forwarded to the host as a
//! context-tokens event. The host maintains the running tally of total
//! context tokens — the guest only reports tokens added per operation.
//!
//! Cancellation is cooperative: the user reader sets a shared flag that the
//! inference runner checks between loop iterations (at each await point).
//! This is safe because all actors run on the same wasm thread.

mod bindings {
    include!(concat!(env!("OUT_DIR"), "/paramecia_bindgen.rs"));
}

use core::cell::Cell;

use bindings::exports::wasi::cli::run::Guest;
use bindings::paramecia::controller::inference;
use bindings::paramecia::controller::interactive;
use bindings::paramecia::controller::structure;
use bindings::paramecia::controller::types::{AgentEvent, ModelInput, UserEvent, Weights};

struct InteractiveController;

/// Commands sent from the user reader to the inference runner.
enum InferenceCmd {
    /// Fill context with the given text, optionally followed by generation.
    Fill { text: String, generate: bool },
}

impl Guest for InteractiveController {
    async fn run() -> Result<(), ()> {
        // Connect first so the host can relay events while the model loads
        let (mut agent_tx, agent_rx) = bindings::wit_stream::new::<AgentEvent>();
        let mut user_stream = match interactive::connect(agent_rx) {
            Ok(stream) => stream,
            Err(_e) => {
                drop(agent_tx);
                return Err(());
            }
        };

        // Load the model — signal Initialized on success, Error on failure
        let model = match structure::load_model(Weights::HostDefault).await {
            Ok(m) => m,
            Err(e) => {
                let _ = agent_tx
                    .write_one(AgentEvent::Error(format!("{e:?}")))
                    .await;
                drop(agent_tx);
                return Err(());
            }
        };
        let _ = agent_tx.write_one(AgentEvent::Initialized).await;

        // Internal channels
        let (event_tx, event_rx) = async_channel::unbounded::<AgentEvent>();
        let (cmd_tx, cmd_rx) = async_channel::unbounded::<InferenceCmd>();

        // Cooperative cancel flag — safe because all actors share one wasm thread
        let cancelled = Cell::new(false);

        futures::join!(
            // Actor 1 — Writer: drain internal event channel → WIT agent stream
            async {
                let mut agent_tx = agent_tx;
                while let Ok(event) = event_rx.recv().await {
                    if agent_tx.write_one(event).await.is_some() {
                        break;
                    }
                }
                drop(agent_tx);
            },
            // Actor 2 — User reader: user events → commands / cancel flag
            async {
                let _ = cmd_tx.send(InferenceCmd::Fill {
                    text: "<|im_start|>system\nYou are an agent of the artificially intelligent kind.\n<|im_end|>\n".to_string(),
                    generate: false
                }).await;
                while let Some(user_event) = user_stream.next().await {
                    match user_event {
                        Ok(UserEvent::Prompt(text)) => {
                            let formatted = format!(
                                "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
                            );
                            let _ = cmd_tx
                                .send(InferenceCmd::Fill {
                                    text: formatted,
                                    generate: true,
                                })
                                .await;
                        }
                        Ok(UserEvent::Prefill(text)) => {
                            let _ = cmd_tx
                                .send(InferenceCmd::Fill {
                                    text,
                                    generate: false,
                                })
                                .await;
                        }
                        Ok(UserEvent::Cancel) => {
                            cancelled.set(true);
                        }
                        // Ignore tool results and compaction in this simple example
                        _ => {}
                    }
                }
                cmd_tx.close();
            },
            // Actor 3 — Inference runner: fill-context / predict-completion
            async {
                while let Ok(cmd) = cmd_rx.recv().await {
                    let InferenceCmd::Fill { text, generate } = cmd;

                    // Pre-emptive cancel check (e.g. Cancel arrived before this Fill)
                    if cancelled.get() {
                        cancelled.set(false);
                        if event_tx.send(AgentEvent::Done).await.is_err() {
                            return;
                        }
                        continue;
                    }

                    // Fill context, forwarding every progress update as a delta.
                    // fill-context returns cumulative progress; convert to deltas.
                    let mut fill_stream =
                        inference::fill_context(model.clone(), &[ModelInput::Text(text)]);
                    let mut was_cancelled = false;
                    let mut last_progress = 0u32;
                    while let Some(progress) = fill_stream.next().await {
                        if cancelled.get() {
                            was_cancelled = true;
                            break;
                        }
                        match progress {
                            Ok(n) => {
                                let delta = n.saturating_sub(last_progress);
                                last_progress = n;
                                if delta > 0 {
                                    if event_tx
                                        .send(AgentEvent::ContextTokens(delta))
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                            Err(_) => break,
                        }
                    }

                    // Generate tokens (if requested and not cancelled)
                    if generate && !was_cancelled {
                        let mut completion = inference::predict_completion(model.clone());
                        while let Some(result) = completion.next().await {
                            if cancelled.get() {
                                let _ = inference::cancel_prediction(model.clone()).await;
                                break;
                            }
                            match result {
                                Ok(predicted) => {
                                    if let Some(text) = predicted.text {
                                        if text != "<|im_end|>" {
                                            if event_tx.send(AgentEvent::Token(text)).await.is_err()
                                            {
                                                let _ = inference::cancel_prediction(model.clone())
                                                    .await;
                                                return;
                                            }
                                        }
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                    }

                    // Signal operation complete and reset cancel flag
                    cancelled.set(false);
                    if event_tx.send(AgentEvent::Done).await.is_err() {
                        return;
                    }
                }
            }
        );

        let _ = structure::unload_model(model).await;
        Ok(())
    }
}

bindings::export!(InteractiveController with_types_in bindings);
