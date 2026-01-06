//! Programmatic (non-interactive) mode.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{Result, anyhow};
use paramecia_harness::events::AgentEvent;
use paramecia_harness::modes::AgentMode;
use paramecia_harness::types::OutputFormat;
use paramecia_harness::{Agent, ParameciaConfig};
use tokio::sync::mpsc;
use tokio::task::LocalSet;

use crate::args::TuiArgs;

type SharedAgent = Rc<RefCell<Option<Agent>>>;

fn with_agent_mut<R, F>(agent: &SharedAgent, op: F) -> Result<R>
where
    F: FnOnce(&mut Agent) -> R,
{
    let mut slot = agent.borrow_mut();
    let mut agent_value = slot.take().ok_or_else(|| anyhow!("Agent is unavailable"))?;
    let result = op(&mut agent_value);
    *slot = Some(agent_value);
    Ok(result)
}

async fn with_agent_async<R, F, Fut>(agent: &SharedAgent, op: F) -> Result<R>
where
    F: FnOnce(Agent) -> Fut,
    Fut: std::future::Future<Output = (Agent, R)>,
{
    let agent_value = {
        let mut slot = agent.borrow_mut();
        slot.take().ok_or_else(|| anyhow!("Agent is unavailable"))?
    };

    let (agent_value, result) = op(agent_value).await;

    let mut slot = agent.borrow_mut();
    *slot = Some(agent_value);

    Ok(result)
}

/// Run in programmatic mode.
pub async fn run(
    config: ParameciaConfig,
    mode: AgentMode,
    prompt: &str,
    tui_args: &TuiArgs,
    loaded_messages: Option<Vec<paramecia_text::LlmMessage>>,
) -> Result<()> {
    // In programmatic mode, if enabled_tools is specified, disable all other tools
    let mut config = config;
    if !tui_args.enabled_tools.is_empty() && matches!(mode, AgentMode::AutoApprove) {
        config.enabled_tools = tui_args.enabled_tools.clone();
        // Clear disabled_tools when enabled_tools is set
        config.disabled_tools.clear();
    }

    let agent: SharedAgent = Rc::new(RefCell::new(Some(
        Agent::with_options(config, mode, tui_args.max_turns, None).await?,
    )));

    // Load previous messages if provided
    if let Some(messages) = loaded_messages {
        let history_result = with_agent_mut(&agent, |agent| agent.load_history(messages))?;
        history_result?;
    }

    let output_format = tui_args.output_format();

    // Use LocalSet since Agent is not Send
    let local = LocalSet::new();

    let all_events = Rc::new(RefCell::new(Vec::new()));
    let final_response = Rc::new(RefCell::new(String::new()));

    local
        .run_until(async {
            // Create a channel for receiving events
            let (tx, mut rx) = mpsc::channel::<AgentEvent>(100);

            let agent_clone = Rc::clone(&agent);
            let prompt_owned = prompt.to_string();

            // Spawn agent task locally
            tokio::task::spawn_local(async move {
                match with_agent_async(&agent_clone, |mut agent| async move {
                    let result = agent.act(&prompt_owned, tx).await;
                    (agent, result)
                })
                .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => eprintln!("Agent error: {e}"),
                    Err(e) => eprintln!("Agent unavailable: {e}"),
                }
            });

            // Process events as they arrive
            while let Some(event) = rx.recv().await {
                match output_format {
                    OutputFormat::Streaming => {
                        // Output each event as JSON
                        if let Ok(json) = event_to_json(&event) {
                            println!("{json}");
                        }
                    }
                    OutputFormat::Text => {
                        // Collect text for final output
                        if let AgentEvent::Assistant(assistant_event) = &event
                            && !assistant_event.stopped_by_middleware
                        {
                            final_response
                                .borrow_mut()
                                .push_str(&assistant_event.content);
                        }
                    }
                    OutputFormat::Json => {
                        // Collect all events
                        all_events.borrow_mut().push(event.clone());
                    }
                }
            }
        })
        .await;

    match output_format {
        OutputFormat::Text => {
            let response = final_response.borrow();
            if !response.is_empty() {
                println!("{response}");
            }
        }
        OutputFormat::Json => {
            let events = all_events.borrow();
            let json = events_to_json(&events)?;
            println!("{json}");
        }
        OutputFormat::Streaming => {
            // Already output
        }
    }

    Ok(())
}

fn event_to_json(event: &AgentEvent) -> Result<String> {
    let value = match event {
        AgentEvent::Assistant(e) => serde_json::json!({
            "type": "assistant",
            "content": e.content,
            "context_tokens": e.context_tokens,
            "stopped_by_middleware": e.stopped_by_middleware,
        }),
        AgentEvent::ToolCall(e) => serde_json::json!({
            "type": "tool_call",
            "tool_name": e.tool_name,
            "args": e.args,
            "tool_call_id": e.tool_call_id,
            "context_tokens": e.context_tokens,
        }),
        AgentEvent::ToolResult(e) => serde_json::json!({
            "type": "tool_result",
            "tool_name": e.tool_name,
            "result": e.result,
            "error": e.error,
            "skipped": e.skipped,
            "duration": e.duration,
            "tool_call_id": e.tool_call_id,
            "context_tokens": e.context_tokens,
        }),
        AgentEvent::ContextTokens(e) => serde_json::json!({
            "type": "context_tokens",
            "context_tokens": e.context_tokens,
        }),
        AgentEvent::CompactStart(e) => serde_json::json!({
            "type": "compact_start",
            "current_context_tokens": e.current_context_tokens,
            "threshold": e.threshold,
        }),
        AgentEvent::CompactEnd(e) => serde_json::json!({
            "type": "compact_end",
            "old_context_tokens": e.old_context_tokens,
            "new_context_tokens": e.new_context_tokens,
            "summary_length": e.summary_length,
        }),
    };

    Ok(serde_json::to_string(&value)?)
}

fn events_to_json(events: &[AgentEvent]) -> Result<String> {
    let values: Vec<serde_json::Value> = events
        .iter()
        .filter_map(|e| serde_json::from_str(&event_to_json(e).ok()?).ok())
        .collect();

    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "messages": values
    }))?)
}
