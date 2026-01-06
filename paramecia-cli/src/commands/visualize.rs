//! Visualize subcommand — launch the visualizer GUI with in-process channels.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use paramecia_engine::tensor_trace_agg::TensorOpAggregationHandle;
use paramecia_harness::ParameciaConfig;
use paramecia_harness::config::{ModelConfig as HarnessModelConfig, ProviderConfig};
use paramecia_harness::events::AgentEvent;
use paramecia_harness::modes::AgentMode;
use paramecia_text::backend::BackendType;
use paramecia_visualizer::{
    ChatOverlayCommand, ChatOverlayEvent, TensorOpSnapshot, TensorOpSnapshotRow, VisualizerChannels,
};

use super::{AgentCommand, AgentHandle, AgentResult};

const DEFAULT_TRACE_TOP_K: usize = 4096;
const DEFAULT_TRACE_POLL_MS: u64 = 200;
const AGENT_POLL_MS: u64 = 25;

pub fn run(trace_handle: TensorOpAggregationHandle) -> Result<()> {
    let mut config = ParameciaConfig::load(None)?;
    apply_visualizer_agent_overrides(&mut config)?;

    let (trace_tx, trace_rx) = mpsc::channel::<TensorOpSnapshot>();
    let (chat_event_tx, chat_event_rx) = mpsc::channel::<ChatOverlayEvent>();
    let (chat_cmd_tx, chat_cmd_rx) = mpsc::channel::<ChatOverlayCommand>();
    let stop = Arc::new(AtomicBool::new(false));

    let trace_thread = spawn_trace_publisher(
        trace_handle,
        trace_tx,
        Arc::clone(&stop),
        trace_top_k_from_env(),
        trace_interval_ms_from_env(),
    );
    let agent_thread = spawn_agent_worker(
        config,
        chat_cmd_rx,
        chat_event_tx,
        Arc::clone(&stop),
        AgentMode::AutoApprove,
    );

    let app_result = paramecia_visualizer::go_with_channels(VisualizerChannels {
        trace_rx,
        chat_event_rx,
        chat_command_tx: chat_cmd_tx,
    })
    .map_err(|err| anyhow!("{err}"));

    stop.store(true, Ordering::Relaxed);
    join_thread(trace_thread);
    join_thread(agent_thread);

    app_result
}

fn spawn_trace_publisher(
    handle: TensorOpAggregationHandle,
    tx: Sender<TensorOpSnapshot>,
    stop: Arc<AtomicBool>,
    top_k: usize,
    interval_ms: u64,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut sequence: u64 = 1;
        let interval = Duration::from_millis(interval_ms.max(1));
        while !stop.load(Ordering::Relaxed) {
            let rows = handle
                .snapshot_top(top_k.max(1))
                .into_iter()
                .map(|row| TensorOpSnapshotRow {
                    key: Some(row.key),
                    node_stable_id: row.node_stable_id,
                    count: row.count,
                    total_ns: row.total_ns,
                })
                .collect::<Vec<_>>();
            let snapshot = TensorOpSnapshot {
                sequence,
                emitted_at_unix_ms: now_unix_ms(),
                rows,
            };
            if tx.send(snapshot).is_err() {
                break;
            }
            sequence = sequence.saturating_add(1);
            thread::sleep(interval);
        }
    })
}

fn spawn_agent_worker(
    config: ParameciaConfig,
    chat_cmd_rx: Receiver<ChatOverlayCommand>,
    chat_event_tx: Sender<ChatOverlayEvent>,
    stop: Arc<AtomicBool>,
    mode: AgentMode,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let outer_chat_event_tx = chat_event_tx.clone();
        let panic_result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || -> Result<()> {
                let chat_event_tx = outer_chat_event_tx;
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|err| anyhow!("Failed to start async runtime: {err}"))?;

                runtime.block_on(async move {
                    let (mut handle, worker) = AgentHandle::new(config, mode, None);
                    tokio::spawn(worker);

                    let mut shutdown_requested = false;
                    let mut ticker = tokio::time::interval(Duration::from_millis(AGENT_POLL_MS));

                    loop {
                        if stop.load(Ordering::Relaxed) && !shutdown_requested {
                            let _ = handle.send(AgentCommand::Shutdown).await;
                            shutdown_requested = true;
                        }

                        loop {
                            match chat_cmd_rx.try_recv() {
                                Ok(ChatOverlayCommand::SendPrompt(prompt)) => {
                                    if handle.send(AgentCommand::Act { prompt }).await.is_err() {
                                        let _ = chat_event_tx.send(ChatOverlayEvent::Error(
                                            "Failed to send prompt to agent worker".to_string(),
                                        ));
                                    }
                                }
                                Ok(ChatOverlayCommand::Interrupt) => {
                                    let _ = handle.send(AgentCommand::Interrupt).await;
                                }
                                Err(TryRecvError::Empty) => break,
                                Err(TryRecvError::Disconnected) => {
                                    if !shutdown_requested {
                                        let _ = handle.send(AgentCommand::Shutdown).await;
                                        shutdown_requested = true;
                                    }
                                    break;
                                }
                            }
                        }

                        while let Some(result) = handle.try_recv() {
                            if handle_agent_result(&chat_event_tx, result) {
                                return;
                            }
                        }

                        if shutdown_requested {
                            // Allow one extra cycle to process shutdown ack.
                            ticker.tick().await;
                            while let Some(result) = handle.try_recv() {
                                if handle_agent_result(&chat_event_tx, result) {
                                    return;
                                }
                            }
                            return;
                        }

                        ticker.tick().await;
                    }
                });

                Ok(())
            }));

        match panic_result {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                let _ = chat_event_tx.send(ChatOverlayEvent::Error(err.to_string()));
            }
            Err(payload) => {
                let _ = chat_event_tx.send(ChatOverlayEvent::Error(format!(
                    "Chat backend panicked: {}",
                    panic_payload_to_string(payload)
                )));
            }
        }
    })
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}

fn handle_agent_result(chat_event_tx: &Sender<ChatOverlayEvent>, result: AgentResult) -> bool {
    match result {
        AgentResult::Ready { session_id } => {
            let short = session_id.chars().take(8).collect::<String>();
            let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                "Agent ready (session {short})"
            )));
        }
        AgentResult::InitError(message) => {
            let _ = chat_event_tx.send(ChatOverlayEvent::Error(message));
        }
        AgentResult::Event(event) => match event {
            AgentEvent::Assistant(data) => {
                let _ = chat_event_tx.send(ChatOverlayEvent::AssistantDelta(data.content));
            }
            AgentEvent::ToolCall(data) => {
                let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                    "Tool call: {}",
                    data.tool_name
                )));
            }
            AgentEvent::ToolResult(data) => {
                let status = if data.error.is_some() {
                    "failed"
                } else if data.skipped {
                    "skipped"
                } else {
                    "ok"
                };
                let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                    "Tool result: {} ({status})",
                    data.tool_name
                )));
            }
            AgentEvent::ContextTokens(data) => {
                let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                    "Context: {} tokens",
                    data.context_tokens
                )));
            }
            AgentEvent::CompactStart(data) => {
                let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                    "Compacting context at {} tokens",
                    data.current_context_tokens
                )));
            }
            AgentEvent::CompactEnd(data) => {
                let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                    "Compacted: {} -> {} tokens",
                    data.old_context_tokens, data.new_context_tokens
                )));
            }
        },
        AgentResult::Done { context_tokens } => {
            let _ = chat_event_tx.send(ChatOverlayEvent::AssistantDone);
            let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                "Done (context {context_tokens} tokens)"
            )));
        }
        AgentResult::Error(message)
        | AgentResult::CompactError(message)
        | AgentResult::SnapshotError(message) => {
            let _ = chat_event_tx.send(ChatOverlayEvent::Error(message));
        }
        AgentResult::ApprovalNeeded { tool_name, .. } => {
            let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                "Approval requested for {tool_name}"
            )));
        }
        AgentResult::Cleared => {
            let _ =
                chat_event_tx.send(ChatOverlayEvent::Status("Conversation cleared".to_string()));
        }
        AgentResult::Compacted {
            old_tokens,
            new_tokens,
        } => {
            let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!(
                "Compacted: {old_tokens} -> {new_tokens} tokens"
            )));
        }
        AgentResult::SnapshotSaved { path } => {
            let _ = chat_event_tx.send(ChatOverlayEvent::Status(format!("Snapshot saved: {path}")));
        }
        AgentResult::ShutdownAck => {
            let _ = chat_event_tx.send(ChatOverlayEvent::Status("Agent stopped".to_string()));
            return true;
        }
    }
    false
}

fn trace_top_k_from_env() -> usize {
    std::env::var("PARAMECIA_VIS_TRACE_TOPK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_TRACE_TOP_K)
}

fn trace_interval_ms_from_env() -> u64 {
    std::env::var("PARAMECIA_VIS_TRACE_INTERVAL_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_TRACE_POLL_MS)
}

fn now_unix_ms() -> u128 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis(),
        Err(_) => 0,
    }
}

fn join_thread(join: JoinHandle<()>) {
    let _ = join.join();
}

fn apply_visualizer_agent_overrides(config: &mut ParameciaConfig) -> Result<()> {
    let model_path = std::env::var("PARAMECIA_MODEL_PATH")
        .map_err(|_| anyhow!("PARAMECIA_MODEL_PATH must be set for `paramecia visualize`"))?;
    let tokenizer_path = std::env::var("PARAMECIA_TOKENIZER_PATH").ok();

    config.instructions.clear();
    config.workdir = None;
    config.use_minimal_system_prompt = true;
    config.system_prompt_id = "__none__".to_string();
    config.include_prompt_detail = false;
    config.include_project_context = false;
    config.include_model_info = false;
    config.include_commit_signature = false;
    config.enable_update_checks = false;
    config.mcp_servers.clear();
    config.tools.clear();
    config.enabled_tools.clear();
    config.disabled_tools = vec!["*".to_string()];

    let provider_name = "local";
    if let Some(provider) = config
        .providers
        .iter_mut()
        .find(|p| p.name == provider_name)
    {
        provider.backend = BackendType::Local;
        provider.local_model_path = Some(model_path.clone());
        provider.local_tokenizer_path = tokenizer_path.clone();
        provider.local_disable_context = Some(true);
    } else {
        config.providers.push(ProviderConfig {
            name: provider_name.to_string(),
            api_base: String::new(),
            api_key_env_var: String::new(),
            backend: BackendType::Local,
            local_model_path: Some(model_path.clone()),
            local_tokenizer_path: tokenizer_path,
            local_max_tokens: Some(4096),
            local_device: None,
            local_offload: None,
            context_length: None,
            local_kv_cache_quant: None,
            local_layer_split: None,
            local_disable_context: Some(true),
            tool_call_format: None,
            full_generation_output: None,
            tuning_output: None,
            tuning_top_k: None,
            tuning_tail_samples: None,
        });
    }

    let model_alias = "visualizer-local";
    if let Some(model) = config
        .models
        .iter_mut()
        .find(|m| m.alias.as_deref() == Some(model_alias) || m.name == model_alias)
    {
        model.provider = provider_name.to_string();
        model.alias = Some(model_alias.to_string());
    } else {
        config.models.push(HarnessModelConfig {
            name: model_alias.to_string(),
            provider: provider_name.to_string(),
            alias: Some(model_alias.to_string()),
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.0,
            repeat_penalty: 1.1,
            presence_penalty: 1.0,
            input_price: 0.0,
            output_price: 0.0,
        });
    }
    config.active_model = model_alias.to_string();

    Ok(())
}
