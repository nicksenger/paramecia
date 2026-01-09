//! Interactive mode with TUI.
//!
//! Uses message-passing concurrency: the agent runs in a dedicated background
//! thread with its own runtime, while the UI runs on the main thread. All
//! communication happens via channels, keeping the UI responsive even during
//! heavy inference operations.

use std::time::Duration;

use anyhow::{Result, anyhow};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use tokio::time;

use paramecia_harness::VibeConfig;
use paramecia_harness::modes::AgentMode;
use paramecia_harness::types::ApprovalResponse;

use crate::args::Args;
use crate::ui::messages::SystemMessageKind;
use crate::ui::{App, AppAction};

use super::{AgentCommand, AgentHandle, AgentResult};

/// Apply TUI-specific config overrides to align with local_agent example behavior.
fn apply_tui_overrides(config: &mut VibeConfig) {
    config.include_project_context = false;
}

/// Run interactive mode.
pub async fn run(
    config: VibeConfig,
    mode: AgentMode,
    initial_prompt: Option<String>,
    args: &Args,
    _loaded_messages: Option<Vec<paramecia_llm::LlmMessage>>,
) -> Result<()> {
    // Handle enabled_tools override
    let mut config = config.clone();
    if !args.enabled_tools.is_empty() {
        config.enabled_tools = args.enabled_tools.clone();
        config.disabled_tools.clear();
    }

    apply_tui_overrides(&mut config);

    // Create the TUI app
    let mut app = App::new(config.clone(), mode);

    // Initialize version update checker
    if config.enable_update_checks {
        let cache_dir = paramecia_harness::paths::CONFIG_FILE.parent().unwrap();
        app.version_update_checker =
            Some(crate::ui::version_update::VersionUpdateChecker::new(cache_dir).await);
    }

    // Set compact threshold for token display
    if config.auto_compact_threshold > 0 {
        app.update_tokens(0, config.auto_compact_threshold);
    }

    // Check for version updates
    if let Err(e) = app.check_version_updates().await {
        tracing::warn!("Failed to check for version updates: {}", e);
    }

    // Show dangerous directory warning
    app.show_dangerous_directory_warning();

    // Create terminal wrapper
    let mut terminal = match crate::ui::app::TerminalWrapper::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to initialize terminal: {e}");
            return Err(e.into());
        }
    };

    // Show loading state and spawn agent in background thread
    app.set_model_loading(true);

    // Spawn agent worker in a dedicated thread with its own runtime
    let mut agent_handle = AgentHandle::spawn(config.clone(), mode);

    // Animate loading screen while waiting for agent to initialize
    let session_id = loop {
        // Tick animations
        app.tick();

        // Draw the loading screen
        if terminal.draw(&mut app).is_err() {
            return Err(anyhow!("Failed to render loading state"));
        }

        // Check for agent ready (non-blocking)
        if let Some(result) = agent_handle.try_recv() {
            match result {
                AgentResult::Ready { session_id } => {
                    app.set_model_loading(false);
                    break session_id;
                }
                AgentResult::InitError(e) => {
                    app.set_model_loading(false);
                    app.add_system_message(
                        format!("Failed to create agent: {}", e),
                        SystemMessageKind::Error,
                    );
                    let _ = terminal.draw(&mut app);
                    time::sleep(Duration::from_secs(2)).await;
                    return Err(anyhow!("Failed to create agent: {}", e));
                }
                _ => {} // Ignore other messages during init
            }
        }

        // Check for Ctrl+C during loading
        if event::poll(Duration::from_millis(0))?
            && let Event::Key(key) = event::read()?
            && key.code == KeyCode::Char('c')
            && key.modifiers.contains(KeyModifiers::CONTROL)
        {
            agent_handle.shutdown().await;
            return Err(anyhow!("Interrupted by user"));
        }

        // Animation frame rate (~30 FPS)
        time::sleep(Duration::from_millis(33)).await;
    };

    app.set_session_id(session_id);

    // Process initial prompt if provided
    if let Some(prompt) = initial_prompt {
        app.add_user_message(prompt.clone(), true);
        app.start_loading();
        let _ = agent_handle.send(AgentCommand::Act { prompt }).await;
    }

    // Main event loop
    loop {
        // Tick animations
        app.tick();

        // Draw
        if terminal.draw(&mut app).is_err() {
            break;
        }

        // Process terminal events (non-blocking)
        while event::poll(Duration::from_millis(0))? {
            match event::read()? {
                Event::Key(key) => {
                    let action = app.handle_key(key);

                    match handle_app_action(action, &mut app, &agent_handle).await {
                        LoopControl::Continue => {}
                        LoopControl::Break => {
                            agent_handle.shutdown().await;
                            drop(terminal);
                            if let Some(session_id) = &app.session_id {
                                println!();
                                println!("To continue this session, run: paramecia --continue");
                                println!(
                                    "Or: paramecia --resume {}",
                                    &session_id[..8.min(session_id.len())]
                                );
                            }
                            return Ok(());
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    app.handle_mouse(mouse);
                }
                _ => {}
            }
        }

        // Process agent events (non-blocking)
        // Limit streaming content events per frame to avoid chunky rendering
        let mut streaming_events_this_frame = 0;
        const MAX_STREAMING_EVENTS_PER_FRAME: usize = 1;

        while let Some(result) = agent_handle.try_recv() {
            // Check if this is a streaming content event
            let is_streaming_content = matches!(
                &result,
                AgentResult::Event(paramecia_harness::events::AgentEvent::Assistant(_))
            );

            if is_streaming_content {
                streaming_events_this_frame += 1;
                if streaming_events_this_frame > MAX_STREAMING_EVENTS_PER_FRAME {
                    // Re-queue this event for next frame by breaking out
                    // Note: we can't re-queue, so just process it but break after
                    handle_agent_result(&mut app, result);
                    break;
                }
            }

            handle_agent_result(&mut app, result);
        }

        // Short sleep for animation updates
        time::sleep(Duration::from_millis(16)).await; // ~60 FPS
    }

    agent_handle.shutdown().await;
    drop(terminal);

    if let Some(session_id) = &app.session_id {
        println!();
        println!("To continue this session, run: paramecia --continue");
        println!(
            "Or: paramecia --resume {}",
            &session_id[..8.min(session_id.len())]
        );
    }

    Ok(())
}

/// Control flow for the main loop.
enum LoopControl {
    Continue,
    Break,
}

/// Handle an agent result from the worker.
fn handle_agent_result(app: &mut App, result: AgentResult) {
    match result {
        AgentResult::Ready { .. } => {
            // Already handled during init
        }
        AgentResult::InitError(_) => {
            // Already handled during init
        }
        AgentResult::Event(event) => {
            use paramecia_harness::events::AgentEvent;
            match event {
                AgentEvent::Assistant(e) => {
                    if !e.content.is_empty() {
                        app.update_assistant_message(&e.content);
                    }
                }
                AgentEvent::ToolCall(e) => {
                    app.add_tool_call(e.tool_name, &e.args);
                }
                AgentEvent::ToolResult(e) => {
                    app.add_tool_result(&e);
                }
                AgentEvent::CompactStart(e) => {
                    app.add_system_message(
                        format!(
                            "Compacting conversation ({}k tokens > {}k threshold)...",
                            e.current_context_tokens / 1000,
                            e.threshold / 1000
                        ),
                        SystemMessageKind::Info,
                    );
                }
                AgentEvent::CompactEnd(e) => {
                    app.add_system_message(
                        format!(
                            "Compacted: {}k → {}k tokens",
                            e.old_context_tokens / 1000,
                            e.new_context_tokens / 1000
                        ),
                        SystemMessageKind::Info,
                    );
                    app.update_tokens(e.new_context_tokens, app.token_context.1);
                }
            }
        }
        AgentResult::Done { context_tokens } => {
            app.stop_loading();
            app.complete_assistant_message();
            app.update_tokens(context_tokens, app.token_context.1);
        }
        AgentResult::Error(e) => {
            app.stop_loading();
            app.add_system_message(format!("Error: {e}"), SystemMessageKind::Error);
        }
        AgentResult::ApprovalNeeded {
            tool_name,
            args,
            tool_call_id,
        } => {
            app.show_approval(tool_name, args, tool_call_id);
        }
        AgentResult::Cleared => {
            app.messages.clear();
            app.add_system_message("Conversation cleared.".to_string(), SystemMessageKind::Info);
        }
        AgentResult::Compacted {
            old_tokens,
            new_tokens,
        } => {
            app.add_system_message(
                format!(
                    "Compacted: {}k → {}k tokens",
                    old_tokens / 1000,
                    new_tokens / 1000
                ),
                SystemMessageKind::Info,
            );
            app.update_tokens(new_tokens, app.token_context.1);
        }
        AgentResult::CompactError(e) => {
            app.add_system_message(format!("Compact failed: {e}"), SystemMessageKind::Error);
        }
        AgentResult::ShutdownAck => {
            // Worker acknowledged shutdown
        }
    }
}

/// Handle an app action from user input.
async fn handle_app_action(
    action: AppAction,
    app: &mut App,
    agent_handle: &AgentHandle,
) -> LoopControl {
    match action {
        AppAction::Continue => LoopControl::Continue,

        AppAction::Exit => {
            let _ = agent_handle.send(AgentCommand::Shutdown).await;
            app.should_exit = true;
            LoopControl::Break
        }

        AppAction::Submit(content) => {
            if !app.loading {
                app.add_user_message(content.clone(), false);
                app.start_loading();
                let _ = agent_handle
                    .send(AgentCommand::Act { prompt: content })
                    .await;
            }
            LoopControl::Continue
        }

        AppAction::Command(cmd) => {
            if !app.loading {
                handle_command(app, agent_handle, &cmd).await
            } else {
                LoopControl::Continue
            }
        }

        AppAction::CycleMode => {
            let new_mode = app.mode;
            let _ = agent_handle.send(AgentCommand::SetMode(new_mode)).await;
            app.add_system_message(
                format!("Switched to {} mode", new_mode.display_name()),
                SystemMessageKind::Info,
            );
            LoopControl::Continue
        }

        AppAction::ToggleToolExpand => {
            app.toggle_tool_expand();
            LoopControl::Continue
        }

        AppAction::ScrollUp => {
            app.scroll_up();
            LoopControl::Continue
        }

        AppAction::ScrollDown => {
            app.scroll_down();
            LoopControl::Continue
        }

        AppAction::Interrupt => {
            let _ = agent_handle.send(AgentCommand::Interrupt).await;
            app.stop_loading();
            app.add_interrupt();
            LoopControl::Continue
        }

        AppAction::Approve(response) => {
            if app.approval.take().is_some() {
                let feedback = if response == ApprovalResponse::No {
                    Some("User rejected the tool execution".to_string())
                } else {
                    None
                };
                let _ = agent_handle
                    .send(AgentCommand::ApprovalResponse { response, feedback })
                    .await;
            }
            app.hide_approval();
            LoopControl::Continue
        }

        AppAction::OpenConfig => {
            app.open_config_app();
            LoopControl::Continue
        }
    }
}

/// Handle a command. Returns LoopControl.
async fn handle_command(app: &mut App, agent_handle: &AgentHandle, cmd: &str) -> LoopControl {
    let cmd = cmd.trim();

    // Handle shell commands
    if let Some(shell_cmd) = cmd.strip_prefix('!') {
        if shell_cmd.is_empty() {
            app.add_system_message(
                "No command provided after '!'".to_string(),
                SystemMessageKind::Error,
            );
            return LoopControl::Continue;
        }

        let cwd = app.config.effective_workdir().to_string_lossy().to_string();

        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(shell_cmd)
            .current_dir(app.config.effective_workdir())
            .output()
            .await;

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let result = if !stdout.is_empty() {
                    stdout.to_string()
                } else if !stderr.is_empty() {
                    stderr.to_string()
                } else {
                    "(no output)".to_string()
                };
                let exit_code = output.status.code().unwrap_or(-1);
                app.add_bash_output(shell_cmd.to_string(), cwd, result, exit_code);
            }
            Err(e) => {
                app.add_system_message(format!("Command failed: {e}"), SystemMessageKind::Error);
            }
        }
        return LoopControl::Continue;
    }

    // Parse command
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let command = parts.first().map(|s| s.to_lowercase());

    match command.as_deref() {
        Some("/quit") | Some("/exit") | Some("/q") => {
            app.add_system_message("Bye!".to_string(), SystemMessageKind::Info);
            LoopControl::Break
        }

        Some("/clear") => {
            let _ = agent_handle.send(AgentCommand::Clear).await;
            LoopControl::Continue
        }

        Some("/reset") => {
            let _ = agent_handle.send(AgentCommand::Clear).await;
            app.messages.clear();
            app.add_system_message(
                "Conversation reset. All history cleared.".to_string(),
                SystemMessageKind::Info,
            );
            LoopControl::Continue
        }

        Some("/compact") => {
            app.add_system_message(
                "Compacting conversation...".to_string(),
                SystemMessageKind::Info,
            );
            let _ = agent_handle.send(AgentCommand::Compact).await;
            LoopControl::Continue
        }

        Some("/help") | Some("/?") => {
            let help = app.commands.get_help_text();
            app.add_system_message(help, SystemMessageKind::Info);
            LoopControl::Continue
        }

        Some("/stats") | Some("/status") => {
            // Stats are tracked in the worker, we'd need to add a Stats command
            // For now, show what we have locally
            app.add_system_message(
                format!(
                    "## Session Info\n\n- **Context Tokens**: {}\n- **Mode**: {}",
                    app.token_context.0,
                    app.mode.display_name()
                ),
                SystemMessageKind::Info,
            );
            LoopControl::Continue
        }

        Some("/config") | Some("/theme") | Some("/model") => {
            app.open_config_app();
            LoopControl::Continue
        }

        Some("/reload") => {
            match VibeConfig::load(None) {
                Ok(new_config) => {
                    let mut new_config = new_config;
                    apply_tui_overrides(&mut new_config);
                    app.config = new_config;
                    app.add_system_message(
                        "Configuration reloaded successfully.".to_string(),
                        SystemMessageKind::Info,
                    );
                }
                Err(e) => {
                    app.add_system_message(
                        format!("Failed to reload configuration: {}", e),
                        SystemMessageKind::Error,
                    );
                }
            }
            LoopControl::Continue
        }

        Some("/session") => {
            let session_info = if let Some(session_id) = &app.session_id {
                format!(
                    "## Current Session\n\n- **Session ID**: {}\n- **Mode**: {}\n- **Working Directory**: {}",
                    &session_id[..8.min(session_id.len())],
                    app.mode.display_name(),
                    app.config.effective_workdir().display()
                )
            } else {
                "No active session".to_string()
            };
            app.add_system_message(session_info, SystemMessageKind::Info);
            LoopControl::Continue
        }

        Some("/terminal-setup") => {
            let terminal = std::env::var("TERM_PROGRAM").unwrap_or_default();
            let message = match terminal.as_str() {
                "iTerm.app" => {
                    "## iTerm2 Setup\n\nTo enable Shift+Enter for multiline input:\n\n1. Open iTerm2 Preferences (⌘,)\n2. Go to Keys → Key Bindings\n3. Click + to add a new binding\n4. Set Keyboard Shortcut to Shift+Enter\n5. Set Action to \"Send Text with vim Special Chars\"\n6. Enter: \\n\n\nRestart iTerm2 for changes to take effect."
                }
                "Apple_Terminal" => {
                    "## Terminal.app Setup\n\nMacOS Terminal has limited key binding support.\nConsider using iTerm2 or another terminal for better experience."
                }
                _ => {
                    "## Terminal Setup\n\nFor Shift+Enter support, check your terminal's documentation.\nMost modern terminals support custom key bindings.\n\nAlternatively, use Ctrl+J to insert newlines."
                }
            };
            app.add_system_message(message.to_string(), SystemMessageKind::Info);
            LoopControl::Continue
        }

        Some("/update") => {
            if let Err(e) = app.check_version_updates().await {
                app.add_system_message(
                    format!("Failed to check for updates: {}", e),
                    SystemMessageKind::Error,
                );
            } else {
                app.add_system_message(
                    "Checked for updates. No new version available.".to_string(),
                    SystemMessageKind::Info,
                );
            }
            LoopControl::Continue
        }

        Some(unknown) => {
            app.add_system_message(
                format!(
                    "Unknown command: {}. Type /help for available commands.",
                    unknown
                ),
                SystemMessageKind::Warning,
            );
            LoopControl::Continue
        }

        None => LoopControl::Continue,
    }
}
