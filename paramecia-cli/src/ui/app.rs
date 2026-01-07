//! Main TUI application.

use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crossterm::{
    cursor::{DisableBlinking, EnableBlinking},
    event::{
        self, DisableMouseCapture, EnableMouseCapture, KeyCode, KeyModifiers, MouseEvent,
        MouseEventKind,
    },
    execute,
    terminal::{
        Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
        enable_raw_mode,
    },
};
use paramecia_harness::VibeConfig;
use paramecia_harness::events::ToolResultEvent;
use paramecia_harness::modes::AgentMode;
use paramecia_harness::paths::HISTORY_FILE;
use paramecia_harness::types::ApprovalResponse;
use ratatui::{
    Terminal,
    prelude::*,
    widgets::{
        Block, Borders, Clear as ClearWidget, Paragraph, Scrollbar, ScrollbarOrientation,
        ScrollbarState,
    },
};
use textwrap::Options;
use unicode_width::UnicodeWidthStr;

use super::config_app::ConfigAction;
use super::input::InputState;
use super::messages::*;

use super::spinner::{Spinner, apply_easter_egg, get_loading_message, get_tool_status_text};

use super::widgets::{self, colors};
use crate::commands::CommandRegistry;

/// Actions the app can take after processing input.
#[derive(Debug, Clone, PartialEq)]
pub enum AppAction {
    /// Continue running.
    Continue,
    /// Submit user input.
    Submit(String),
    /// Exit the application.
    Exit,
    /// Execute a command.
    Command(String),
    /// Cycle to next mode.
    CycleMode,
    /// Toggle tool output expansion.
    ToggleToolExpand,

    /// Scroll chat up.
    ScrollUp,
    /// Scroll chat down.
    ScrollDown,
    /// Interrupt current operation.
    Interrupt,
    /// Approve tool execution.
    Approve(ApprovalResponse),
    /// Open config app.
    OpenConfig,
}

/// State for the approval dialog.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ApprovalState {
    /// Tool name.
    pub tool_name: String,
    /// Tool arguments.
    pub args: serde_json::Value,
    /// Tool call ID.
    pub tool_call_id: String,
    /// Currently selected option (0=yes, 1=no, 2=always).
    pub selected: usize,
}

/// The main TUI application state.
pub struct App {
    /// Configuration.
    pub config: VibeConfig,
    /// Current mode.
    pub mode: AgentMode,
    /// Chat messages.
    pub messages: Vec<Message>,
    /// Input state.
    pub input: InputState,
    /// Spinner for loading animations.
    pub spinner: Spinner,
    /// Whether the agent is currently processing.
    pub loading: bool,
    /// Loading start time.
    pub loading_start: Option<Instant>,
    /// Loading status message.
    pub loading_status: String,
    /// Current gradient color index for animations.
    pub color_index: usize,
    /// Chat scroll offset.
    pub scroll_offset: u16,
    /// Maximum scroll offset.
    pub max_scroll: u16,
    /// Auto-scroll enabled.
    pub auto_scroll: bool,
    /// Current tool call (for spinner).
    pub current_tool_call_idx: Option<usize>,
    /// Approval dialog state.
    pub approval: Option<ApprovalState>,
    /// Token context (current, max).
    pub token_context: (u32, u32),
    /// Whether to expand tool outputs.
    pub tools_expanded: bool,

    /// Current todo list for bottom todo area.
    pub current_todos: Option<Vec<paramecia_tools::builtins::todo::TodoItem>>,
    /// Command registry.
    pub commands: CommandRegistry,
    /// Should exit.
    pub should_exit: bool,
    /// Last render time for animations.
    pub last_render: Instant,
    /// Session ID for resume message.
    pub session_id: Option<String>,
    /// Welcome banner.
    pub welcome_banner: crate::ui::welcome::WelcomeBanner,
    /// Mode indicator.
    pub mode_indicator: crate::ui::mode_indicator::ModeIndicator,
    /// Path display.
    pub path_display: crate::ui::path_display::PathDisplay,
    /// Version update checker.
    pub version_update_checker: Option<crate::ui::version_update::VersionUpdateChecker>,
    /// Config app state.
    pub config_app: Option<crate::ui::config_app::ConfigApp>,
    /// Context progress indicator.
    pub context_progress: crate::ui::context_progress::ContextProgress,
    /// Whether the model is currently loading.
    pub model_loading: bool,
    /// Model loading start time.
    pub model_loading_start: Option<Instant>,
    /// Game of Life animation for loading screen.
    pub game_of_life: super::game_of_life::GameOfLife,
    /// Last time the Game of Life animation advanced.
    last_gol_step: Instant,
}

impl App {
    /// Create a new app.
    pub fn new(config: VibeConfig, mode: AgentMode) -> Self {
        let commands = crate::commands::create_default_registry();

        // Create input state with commands for completion
        let mut input = InputState::with_history(HISTORY_FILE.clone());
        input.set_commands(commands.get_completion_items());

        let config_clone = config.clone();

        // Initialize version update checker
        let version_update_checker = if config.enable_update_checks {
            // We'll create the checker later in an async context
            None
        } else {
            None
        };

        Self {
            config,
            mode,
            messages: Vec::new(),
            input,
            spinner: Spinner::new(),
            loading: false,
            loading_start: None,
            loading_status: get_loading_message().to_string(),
            color_index: 0,
            scroll_offset: 0,
            max_scroll: 0,
            auto_scroll: true,
            current_tool_call_idx: None,
            approval: None,
            token_context: (0, 0),
            tools_expanded: false,
            current_todos: None,
            commands,
            should_exit: false,
            last_render: Instant::now(),
            session_id: None,
            welcome_banner: crate::ui::welcome::WelcomeBanner::new(config_clone.clone()),
            mode_indicator: crate::ui::mode_indicator::ModeIndicator::new(mode),
            path_display: crate::ui::path_display::PathDisplay::new(
                config_clone.clone().effective_workdir().to_path_buf(),
            ),
            version_update_checker,
            config_app: None,
            context_progress: crate::ui::context_progress::ContextProgress::new(),
            model_loading: false,
            model_loading_start: None,
            // Initialize with a reasonable default size; will be resized on first render
            game_of_life: super::game_of_life::GameOfLife::new(80, 20),
            last_gol_step: Instant::now(),
        }
    }

    /// Set the session ID for resume message.
    pub fn set_session_id(&mut self, session_id: String) {
        self.session_id = Some(session_id);
    }

    /// Cycle to the next agent mode.
    pub fn cycle_mode(&mut self) -> AgentMode {
        let new_mode = self.mode_indicator.cycle_mode();
        self.mode = new_mode;
        new_mode
    }

    /// Set model loading state (for initial model load).
    pub fn set_model_loading(&mut self, loading: bool) {
        self.model_loading = loading;
        if loading {
            self.model_loading_start = Some(Instant::now());
        } else {
            self.model_loading_start = None;
        }
    }

    /// Start loading state.
    pub fn start_loading(&mut self) {
        self.loading = true;
        self.loading_start = Some(Instant::now());
        self.loading_status = get_loading_message().to_string();
    }

    /// Stop loading state.
    pub fn stop_loading(&mut self) {
        self.loading = false;
        self.loading_start = None;
    }

    /// Set loading status with easter egg logic (matching Mistral Vibe's set_status).
    /// Has a 10% chance to replace the status with a French easter egg.
    pub fn set_loading_status(&mut self, status: &str) {
        self.loading_status = apply_easter_egg(status);
    }

    /// Add a user message.
    pub fn add_user_message(&mut self, content: String, pending: bool) {
        let msg = if pending {
            UserMessage::pending(content)
        } else {
            UserMessage::new(content)
        };
        self.messages.push(Message::User(msg));
        if self.auto_scroll {
            self.anchor_scroll();
        }
    }

    /// Add or update assistant message (for streaming).
    pub fn update_assistant_message(&mut self, content: &str) {
        // Find the last assistant message and append, or create new
        if let Some(Message::Assistant(msg)) = self.messages.last_mut()
            && !msg.complete
        {
            msg.append(content);
            return;
        }

        // Create new assistant message
        self.messages.push(Message::Assistant(AssistantMessage::new(
            content.to_string(),
        )));
        if self.auto_scroll {
            self.anchor_scroll();
        }
    }

    /// Complete the current assistant message.
    pub fn complete_assistant_message(&mut self) {
        if let Some(Message::Assistant(msg)) = self.messages.last_mut() {
            msg.complete = true;
        }
    }

    /// Add a tool call message.
    pub fn add_tool_call(&mut self, tool_name: String, args: &serde_json::Value) {
        // Update loading status with tool-specific text (matching Mistral Vibe)
        // This has a 10% chance to show an easter egg instead
        let status_text = get_tool_status_text(&tool_name);
        self.set_loading_status(status_text);

        let msg = ToolCallMessage::new(tool_name, args);
        self.messages.push(Message::ToolCall(msg));
        self.current_tool_call_idx = Some(self.messages.len() - 1);
        if self.auto_scroll {
            self.anchor_scroll();
        }
    }

    /// Add a tool result message.
    pub fn add_tool_result(&mut self, event: &ToolResultEvent) {
        // Stop the current tool call spinner
        if let Some(idx) = self.current_tool_call_idx
            && let Some(Message::ToolCall(msg)) = self.messages.get_mut(idx)
        {
            msg.stop();
        }
        self.current_tool_call_idx = None;

        let msg = if event.error.is_some() {
            ToolResultMessage::error(
                event.tool_name.clone(),
                event.error.clone().unwrap_or_default(),
                event.duration,
            )
        } else if event.skipped {
            ToolResultMessage::skipped(
                event.tool_name.clone(),
                event.skip_reason.clone().unwrap_or_default(),
            )
        } else {
            ToolResultMessage::success(
                event.tool_name.clone(),
                event.result.clone().unwrap_or(serde_json::Value::Null),
                event.duration,
            )
        };
        self.messages.push(Message::ToolResult(msg));

        // Update current todos if this is a todo tool result
        if event.tool_name == "todo"
            && let Some(result_value) = &event.result
            && let Ok(todo_result) = serde_json::from_value::<
                paramecia_tools::builtins::todo::TodoResult,
            >(result_value.clone())
        {
            self.current_todos = Some(todo_result.todos);
        }
    }

    /// Add a system message.
    pub fn add_system_message(&mut self, content: String, kind: SystemMessageKind) {
        let msg = match kind {
            SystemMessageKind::Info => SystemMessage::info(content),
            SystemMessageKind::Warning => SystemMessage::warning(content),
            SystemMessageKind::Error => SystemMessage::error(content),
        };
        self.messages.push(Message::System(msg));
        if self.auto_scroll {
            self.anchor_scroll();
        }
    }

    /// Check for version updates and show notification if available.
    pub async fn check_version_updates(&mut self) -> anyhow::Result<()> {
        if let Some(checker) = &mut self.version_update_checker
            && checker.should_check()
        {
            match checker.check_for_updates(env!("CARGO_PKG_VERSION")).await {
                Ok(Some(update_msg)) => {
                    self.add_system_message(format!("📦 {}", update_msg), SystemMessageKind::Info);
                }
                Ok(None) => {
                    // No update available, but update was successful
                    tracing::debug!("No updates available");
                }
                Err(e) => {
                    tracing::warn!("Failed to check for updates: {}", e);
                    // Don't show error to user for version check failures - it's not critical
                }
            }
        }
        Ok(())
    }

    /// Show dangerous directory warning if applicable.
    pub fn show_dangerous_directory_warning(&mut self) {
        use paramecia_harness::project_context::is_dangerous_directory;

        let workdir = self.config.effective_workdir();
        let (is_dangerous, reason) = is_dangerous_directory(&workdir);
        if is_dangerous {
            self.add_system_message(
                format!(
                    "⚠️  You are in a sensitive directory: {}

{}",
                    workdir.display(),
                    reason
                ),
                SystemMessageKind::Warning,
            );
        }
    }

    /// Add an interrupt message.
    pub fn add_interrupt(&mut self) {
        self.messages.push(Message::Interrupt);
        if self.auto_scroll {
            self.anchor_scroll();
        }
    }

    /// Add a bash output message.
    pub fn add_bash_output(
        &mut self,
        command: String,
        cwd: String,
        output: String,
        exit_code: i32,
    ) {
        self.messages
            .push(Message::BashOutput(BashOutputMessage::new(
                command, cwd, output, exit_code,
            )));
        if self.auto_scroll {
            self.anchor_scroll();
        }
    }

    /// Show approval dialog.
    #[allow(dead_code)]
    pub fn show_approval(
        &mut self,
        tool_name: String,
        args: serde_json::Value,
        tool_call_id: String,
    ) {
        self.approval = Some(ApprovalState {
            tool_name,
            args,
            tool_call_id,
            selected: 0,
        });
    }

    /// Hide approval dialog.
    pub fn hide_approval(&mut self) {
        self.approval = None;
    }

    /// Open the config app.
    pub fn open_config_app(&mut self) {
        self.config_app = Some(crate::ui::config_app::ConfigApp::new(self.config.clone()));
    }

    /// Close the config app.
    pub fn close_config_app(&mut self) {
        self.config_app = None;
    }

    /// Apply a setting change from the config app.
    pub fn apply_setting_change(&mut self, key: &str) {
        if let Some(config_app) = &self.config_app
            && let Some(value) = config_app.changes.get(key)
            && key == "active_model"
        {
            // Update the active model in config
            self.config.active_model = value.clone();
        }
    }

    /// Handle config app action.
    pub fn handle_config_action(&mut self, action: ConfigAction) {
        match action {
            ConfigAction::Close => {
                self.close_config_app();
            }
            ConfigAction::SettingChanged(key) => {
                // Apply the setting change
                self.apply_setting_change(&key);
            }
        }
    }

    /// Update token context.
    pub fn update_tokens(&mut self, current: u32, max: u32) {
        self.token_context = (current, max);
        // Also update the context_progress component
        self.context_progress
            .update_tokens(super::context_progress::TokenState::new(max, current));
    }

    /// Anchor scroll position to bottom when auto_scroll is enabled.
    /// The actual scroll position is set during render when we know the content height.
    pub fn anchor_scroll(&mut self) {
        // Just ensure auto_scroll stays enabled - actual positioning happens in render
    }

    /// Toggle tool expansion.
    pub fn toggle_tool_expand(&mut self) {
        self.tools_expanded = !self.tools_expanded;
        for msg in &mut self.messages {
            if let Message::ToolResult(result) = msg {
                result.collapsed = !self.tools_expanded;
            }
        }
    }

    /// Handle a key event.
    pub fn handle_key(&mut self, key: event::KeyEvent) -> AppAction {
        // Handle config app input
        if let Some(config_app) = &mut self.config_app {
            if let Some(action) = config_app.handle_input(key) {
                self.handle_config_action(action);
                return AppAction::Continue;
            }
            return AppAction::Continue;
        }

        // Handle approval dialog input next
        if let Some(approval) = &mut self.approval {
            return match key.code {
                KeyCode::Up => {
                    if approval.selected > 0 {
                        approval.selected -= 1;
                    }
                    AppAction::Continue
                }
                KeyCode::Down => {
                    if approval.selected < 2 {
                        approval.selected += 1;
                    }
                    AppAction::Continue
                }
                KeyCode::Enter => {
                    // Options order: 0=Yes, 1=Yes+Always, 2=No
                    let response = match approval.selected {
                        0 => ApprovalResponse::Yes,
                        1 => ApprovalResponse::Always,
                        2 => ApprovalResponse::No,
                        _ => ApprovalResponse::No,
                    };
                    AppAction::Approve(response)
                }
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    AppAction::Approve(ApprovalResponse::Yes)
                }
                KeyCode::Char('n') | KeyCode::Char('N') => AppAction::Approve(ApprovalResponse::No),
                KeyCode::Char('a') | KeyCode::Char('A') => {
                    AppAction::Approve(ApprovalResponse::Always)
                }
                KeyCode::Esc => AppAction::Approve(ApprovalResponse::No),
                _ => AppAction::Continue,
            };
        }

        // Handle global shortcuts
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+C: Clear input if not empty, otherwise quit
                if !self.input.is_empty() {
                    self.input.clear();
                    self.add_system_message(
                        "Input cleared. Press Ctrl+C again to quit.".to_string(),
                        SystemMessageKind::Info,
                    );
                    AppAction::Continue
                } else {
                    AppAction::Exit
                }
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+D: Force quit
                AppAction::Exit
            }
            KeyCode::Esc => {
                if self.loading {
                    self.add_system_message(
                        "Interrupting current operation... Press Esc again to force quit."
                            .to_string(),
                        SystemMessageKind::Warning,
                    );
                    AppAction::Interrupt
                } else if !self.input.is_empty() {
                    self.input.clear();
                    self.add_system_message("Input cleared.".to_string(), SystemMessageKind::Info);
                    AppAction::Continue
                } else {
                    AppAction::Continue
                }
            }
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                AppAction::ToggleToolExpand
            }

            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+J inserts a newline (matching Python's behavior)
                // Unlike the toggle mode, this just directly inserts a newline
                self.input.insert_newline();
                AppAction::Continue
            }
            KeyCode::BackTab => {
                self.cycle_mode();
                AppAction::CycleMode
            }
            KeyCode::Up if key.modifiers.contains(KeyModifiers::SHIFT) => AppAction::ScrollUp,
            KeyCode::Down if key.modifiers.contains(KeyModifiers::SHIFT) => AppAction::ScrollDown,
            KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+S for config/settings
                AppAction::OpenConfig
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+K to clear from cursor to end
                self.input.clear_to_end();
                AppAction::Continue
            }
            KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+L to clear screen (like bash)
                // This would require terminal clearing functionality
                AppAction::Continue
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+U to clear line from cursor to beginning
                self.input.clear_to_beginning();
                AppAction::Continue
            }
            KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+W to delete word
                self.input.delete_word();
                AppAction::Continue
            }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+A to move cursor to beginning
                self.input.move_cursor_to_beginning();
                AppAction::Continue
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+E to move cursor to end
                self.input.move_cursor_to_end();
                AppAction::Continue
            }
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+P for previous history (alternative to up arrow)
                self.input.navigate_history_up();
                AppAction::Continue
            }
            KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+N for next history (alternative to down arrow)
                self.input.navigate_history_down();
                AppAction::Continue
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+F for forward search (would need implementation)
                AppAction::Continue
            }
            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+R for reverse search (would need implementation)
                AppAction::Continue
            }
            _ => {
                // Handle input
                if self.input.handle_key(key) {
                    let content = self.input.submit();
                    if !content.trim().is_empty() {
                        if content.starts_with('/') || content.starts_with('!') {
                            AppAction::Command(content)
                        } else {
                            AppAction::Submit(content)
                        }
                    } else {
                        AppAction::Continue
                    }
                } else {
                    AppAction::Continue
                }
            }
        }
    }

    /// Tick for animations.
    pub fn tick(&mut self) {
        let now = Instant::now();

        if now.duration_since(self.last_render) >= Duration::from_millis(100) {
            self.spinner.next_frame();
            self.color_index = (self.color_index + 1) % colors::GRADIENT.len();

            // Note: loading_status is set once in start_loading() and preserved
            // until loading stops (matching Mistral Vibe behavior where easter eggs
            // persist for the duration of the loading state)

            self.last_render = now;
        }
    }

    /// Advance the Game of Life animation at a slower cadence to reduce motion.
    fn step_game_of_life(&mut self) {
        let now = Instant::now();
        let step_interval = Duration::from_millis(200);

        if now.duration_since(self.last_gol_step) >= step_interval {
            self.game_of_life.step();
            self.last_gol_step = now;
        }
    }

    /// Render the UI.
    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();

        // If model is loading, show a dedicated loading screen
        if self.model_loading {
            self.render_model_loading(frame, area);
            return;
        }

        // Calculate layout matching Mistral Vibe's app.tcss more precisely:
        // - #chat: height 1fr (takes remaining space)
        // - #loading-area: height auto, padding 1 0 0 0 (1 line top padding + content)
        // - #todo-area: height auto (for todo checklist)
        // - #bottom-app-container: height auto (input box)
        // - #bottom-bar: height auto, padding 0 0 1 0 (1 line bottom padding)
        // Calculate dynamic input height based on content
        let input_height = self.calculate_input_height(area.width);
        let bottom_bar_height: u16 = 2; // Bottom bar (1 line content + 1 line bottom padding)
        let loading_area_height: u16 = 2; // Loading area (1 line top padding + 1 line content)

        // Calculate reserved height dynamically
        let reserved_height = loading_area_height + input_height + bottom_bar_height;
        let min_chat_height: u16 = 1;
        let max_todo_height = area
            .height
            .saturating_sub(reserved_height)
            .saturating_sub(min_chat_height);

        let todo_area_height = self
            .current_todos
            .as_ref()
            .filter(|todos| !todos.is_empty())
            .map(|todos| {
                let required_height = todos.len().saturating_add(1).min(u16::MAX as usize) as u16; // +1 for top border
                if max_todo_height == 0 {
                    0
                } else {
                    required_height.min(max_todo_height)
                }
            })
            .unwrap_or(0);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),                      // Chat area (fills remaining space)
                Constraint::Length(loading_area_height), // Loading area
                Constraint::Length(todo_area_height),    // Todo area (variable height)
                Constraint::Length(input_height), // Input area (dynamic height based on content)
                Constraint::Length(bottom_bar_height), // Bottom bar
            ])
            .margin(0) // No margin to match CSS
            .split(area);

        self.render_chat(frame, chunks[0]);

        // Render loading in bottom portion of its area (to add top padding effect)
        let loading_area = Rect {
            x: chunks[1].x,
            y: chunks[1].y + 1, // 1 line top padding
            width: chunks[1].width,
            height: 1,
        };
        self.render_loading(frame, loading_area);

        // Render todo area if there are todos and it's expanded
        if todo_area_height > 0 {
            let todo_area = Rect {
                x: chunks[2].x,
                y: chunks[2].y,
                width: chunks[2].width,
                height: todo_area_height,
            };
            self.render_todo_area(frame, todo_area);
        }

        // Render config app, approval dialog, or input
        // chunks[3] is the input area (after adding todo area at chunks[2])
        if let Some(config_app) = &self.config_app {
            // Overlay config app
            let dialog_height = 10.min(area.height.saturating_sub(6));
            let dialog_area = Rect {
                x: 0,
                y: chunks[3].y.saturating_sub(dialog_height),
                width: area.width,
                height: dialog_height + 2,
            };
            config_app.render(frame, dialog_area);
        } else if let Some(approval) = &self.approval {
            // Overlay approval dialog (matching Mistral Vibe's #approval-app max-height: 16)
            let dialog_height = 14.min(area.height.saturating_sub(6));
            let dialog_area = Rect {
                x: 0,
                y: chunks[3].y.saturating_sub(dialog_height),
                width: area.width,
                height: dialog_height + 3,
            };
            widgets::render_approval_dialog(
                frame,
                dialog_area,
                &approval.tool_name,
                &approval.args,
                approval.selected,
            );
        } else {
            self.render_input(frame, chunks[3]);
        }

        // Render bottom bar in top portion of its area (bottom padding leaves empty space)
        // chunks[4] is the bottom bar area
        let bottom_bar_area = Rect {
            x: chunks[4].x,
            y: chunks[4].y,
            width: chunks[4].width,
            height: 1,
        };

        // Split bottom bar area for context progress and bottom bar
        let bottom_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(70), // Bottom bar
                Constraint::Percentage(30), // Context progress
            ])
            .split(bottom_bar_area);

        self.render_bottom_bar(frame, bottom_chunks[0]);
        self.context_progress.render(frame, bottom_chunks[1]);
    }

    /// Render the model loading screen with Conway's Game of Life animation.
    fn render_model_loading(&mut self, frame: &mut Frame, area: Rect) {
        // Tick animations and advance Game of Life
        self.tick();
        self.step_game_of_life();

        // Resize Game of Life grid if needed
        let gol_width = ((area.width / 2).max(1)) as usize;
        let gol_height = (area.height as usize).max(1);
        self.game_of_life.resize(gol_width, gol_height);

        // Render Game of Life as background
        let gol_cells = self
            .game_of_life
            .render(self.color_index, &colors::GRADIENT);
        for (y, row) in gol_cells.iter().enumerate() {
            for (x, &(ch, color)) in row.iter().enumerate() {
                if ch == ' ' {
                    continue;
                }

                let cell_x = area.x + (x as u16 * 2);
                let cell_y = area.y + y as u16;

                if cell_x < area.right() && cell_y < area.bottom() {
                    if let Some(cell) = frame.buffer_mut().cell_mut((cell_x, cell_y)) {
                        cell.set_char(ch);
                        cell.set_fg(color);
                    }
                }

                let second_x = cell_x.saturating_add(1);
                if second_x < area.right() && cell_y < area.bottom() {
                    if let Some(cell) = frame.buffer_mut().cell_mut((second_x, cell_y)) {
                        cell.set_char(ch);
                        cell.set_fg(color);
                    }
                }
            }
        }

        let elapsed = self
            .model_loading_start
            .map(|s| s.elapsed().as_secs())
            .unwrap_or(0);

        // Build content for the loading box
        let status_text = format!("Loading model... ({}s)", elapsed);
        let hint_text = "This may take a while";
        let max_box_width = area.width.saturating_sub(2).max(1);
        let title_lines = self.paramecia_title_lines(max_box_width.saturating_sub(4));
        let title_width = title_lines
            .iter()
            .map(|line| line.trim_end().len() as u16)
            .max()
            .unwrap_or(0);
        let content_width = title_width
            .max(status_text.len() as u16)
            .max(hint_text.len() as u16);
        let desired_box_width = content_width.saturating_add(4);
        let mut box_width = desired_box_width
            .min(max_box_width)
            .max(content_width.saturating_add(2).min(max_box_width));
        box_width = box_width.max(3.min(area.width));
        let content_lines = title_lines.len() + if title_lines.len() > 1 { 1 } else { 0 } + 2; // blank + status + hint
        let box_height = (content_lines as u16 + 2)
            .min(area.height)
            .max(3.min(area.height));
        let box_x = area.x + (area.width.saturating_sub(box_width)) / 2;
        let box_y = area.y + (area.height.saturating_sub(box_height)) / 2;

        let box_area = Rect {
            x: box_x,
            y: box_y,
            width: box_width,
            height: box_height,
        };

        // Clear any background animation inside the box so dots do not show through
        frame.render_widget(ClearWidget, box_area);

        // Draw a semi-transparent box background
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(colors::ACCENT))
            .border_type(ratatui::widgets::BorderType::Rounded)
            .style(Style::default().bg(Color::Black));

        frame.render_widget(block, box_area);

        // Render loading text inside the box
        let inner_area = Rect {
            x: box_x + 1,
            y: box_y + 1,
            width: box_width.saturating_sub(2),
            height: box_height.saturating_sub(2),
        };

        let mut lines: Vec<Line> = Vec::new();

        for line_text in &title_lines {
            lines.push(Line::from(vec![Span::styled(
                line_text.clone(),
                Style::default()
                    .fg(colors::ACCENT)
                    .add_modifier(ratatui::style::Modifier::BOLD),
            )]));
        }

        if title_lines.len() > 1 {
            lines.push(Line::from(""));
        }

        lines.push(Line::from(vec![Span::styled(
            status_text,
            Style::default()
                .fg(colors::ACCENT)
                .add_modifier(ratatui::style::Modifier::BOLD),
        )]));

        lines.push(Line::from(vec![Span::styled(
            hint_text,
            Style::default().fg(colors::MUTED),
        )]));

        let paragraph = Paragraph::new(lines).alignment(ratatui::layout::Alignment::Center);

        frame.render_widget(paragraph, inner_area);
    }

    /// Build ASCII-art title lines for the loader, falling back if space is tight.
    fn paramecia_title_lines(&self, max_width: u16) -> Vec<String> {
        if max_width == 0 {
            return vec![String::new()];
        }

        let mut title = "Paramecia".to_string();
        if title.len() as u16 > max_width {
            title.truncate(max_width as usize);
        }

        if title.is_empty() {
            title.push('P');
        }

        vec![title]
    }

    /// Render the chat area.
    fn render_chat(&mut self, frame: &mut Frame, area: Rect) {
        // Welcome banner (if no messages yet)
        if self.messages.is_empty() {
            let banner_height = 10;
            if area.height >= banner_height {
                let banner_area = Rect {
                    x: area.x,
                    y: area.y,
                    width: area.width,
                    height: banner_height,
                };
                self.welcome_banner.set_color_index(self.color_index);
                self.welcome_banner.render(banner_area, frame.buffer_mut());
            }
            return;
        }

        // Build all lines from messages
        // Start with margin-top: 1 matching Python's #messages CSS
        let mut all_lines: Vec<Line> = vec![Line::from("")];

        for msg in &self.messages {
            let lines = match msg {
                Message::User(m) => widgets::render_user_message(m, area.width),
                Message::Assistant(m) => widgets::render_assistant_message(m, area.width),
                Message::ToolCall(m) => {
                    widgets::render_tool_call(m, &self.spinner, self.color_index)
                }
                Message::ToolResult(m) => widgets::render_tool_result(m),
                Message::System(m) => widgets::render_system_message(m),
                Message::UserCommand(m) => widgets::render_user_command_message(m),
                Message::Error(m) => widgets::render_error_message(m),
                Message::Warning(m) => widgets::render_warning_message(m),
                Message::Interrupt => widgets::render_interrupt(),
                Message::BashOutput(m) => widgets::render_bash_output(m, area.width),
                Message::Compact(m) => widgets::render_compact(m, &self.spinner, self.color_index),
            };
            all_lines.extend(lines);
        }

        let viewport_height = area.height as usize;
        let total_lines = all_lines.len();

        // Calculate max scroll
        self.max_scroll = total_lines.saturating_sub(viewport_height) as u16;

        // Auto-scroll to bottom when enabled
        if self.auto_scroll {
            self.scroll_offset = self.max_scroll;
        }

        // Clamp scroll offset
        self.scroll_offset = self.scroll_offset.min(self.max_scroll);

        // Use Paragraph with scroll() - this scrolls by line within the paragraph
        let paragraph = Paragraph::new(all_lines).scroll((self.scroll_offset, 0));
        frame.render_widget(paragraph, area);

        // Render scrollbar only if content exceeds viewport
        if total_lines > viewport_height {
            // Create a dedicated scrollbar area on the right edge
            let scrollbar_area = Rect {
                x: area.x + area.width.saturating_sub(1),
                y: area.y,
                width: 1,
                height: area.height,
            };

            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"))
                .thumb_symbol("┃")
                .track_style(Style::default().fg(colors::MUTED));

            let mut scrollbar_state = ScrollbarState::new(total_lines)
                .viewport_content_length(viewport_height)
                .position(self.scroll_offset as usize);

            frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
        }
    }

    /// Render the loading area (matching Mistral Vibe: loading content + mode indicator).
    fn render_loading(&self, frame: &mut Frame, area: Rect) {
        // Split area into loading content and mode indicator
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(1),     // Loading content
                Constraint::Length(40), // Mode indicator
            ])
            .split(area);

        // Render loading spinner if loading
        if self.loading {
            let elapsed = self
                .loading_start
                .map(|s| s.elapsed().as_secs())
                .unwrap_or(0);

            let line = widgets::render_loading(
                elapsed,
                &self.loading_status,
                &self.spinner,
                self.color_index,
            );
            let paragraph = Paragraph::new(line);
            frame.render_widget(paragraph, chunks[0]);
        }

        // Always render mode indicator
        let mode_area = Rect {
            x: chunks[1].x,
            y: chunks[1].y,
            width: chunks[1].width,
            height: chunks[1].height,
        };
        self.mode_indicator.render(mode_area, frame.buffer_mut());
    }

    /// Render the todo area (matching Mistral Vibe's todo checklist at bottom).
    fn render_todo_area(&self, frame: &mut Frame, area: Rect) {
        if let Some(todos) = &self.current_todos {
            if todos.is_empty() {
                return;
            }

            // Create a block for the todo area
            let block = Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(colors::MUTED))
                .title(" Todos ")
                .title_style(Style::default().fg(colors::ACCENT));

            let _inner_area = block.inner(area);

            // Create lines for each todo
            let mut lines = Vec::new();

            for todo in todos {
                let status_symbol = match todo.status {
                    paramecia_tools::builtins::todo::TodoStatus::Pending => "○ ",
                    paramecia_tools::builtins::todo::TodoStatus::InProgress => "● ",
                    paramecia_tools::builtins::todo::TodoStatus::Completed => "✓ ",
                    paramecia_tools::builtins::todo::TodoStatus::Cancelled => "✗ ",
                };

                let priority_color = match todo.priority {
                    paramecia_tools::builtins::todo::TodoPriority::High => colors::ERROR,
                    paramecia_tools::builtins::todo::TodoPriority::Medium => colors::WARNING,
                    paramecia_tools::builtins::todo::TodoPriority::Low => colors::MUTED,
                };

                let line = Line::from(vec![
                    Span::styled(status_symbol, Style::default().fg(priority_color)),
                    Span::styled(todo.content.clone(), Style::default()),
                ]);
                lines.push(line);
            }

            // Create a paragraph with the todo lines
            let paragraph = Paragraph::new(lines).block(block);

            frame.render_widget(paragraph, area);
        }
    }

    /// Calculate the required height for the input area based on content and available width.
    fn calculate_input_height(&self, available_width: u16) -> u16 {
        if self.input.content().is_empty() {
            // Default height when empty (top border + placeholder + bottom border)
            return 3;
        }

        let content_lines = self.displayed_line_count(available_width);
        content_lines as u16 + 2 // +2 for the top and bottom borders
    }

    /// Number of rendered lines for the current content given the available width.
    fn displayed_line_count(&self, available_width: u16) -> usize {
        self.wrap_content(available_width, None).0
    }

    /// Calculate the inner width available for input text (after padding).
    fn input_inner_width(&self, available_width: u16) -> usize {
        // Block uses left/right padding of 1 with no side borders.
        usize::from(available_width.saturating_sub(2)).max(1)
    }

    /// Calculate cursor position for multiline input.
    fn calculate_cursor_position(&self, area: Rect) -> (u16, u16) {
        let content = self.input.content();
        let mut cursor_byte = self.input.cursor().min(content.len());

        // Ensure cursor is on a char boundary to avoid slicing issues
        while cursor_byte > 0 && !content.is_char_boundary(cursor_byte) {
            cursor_byte -= 1;
        }

        // Empty content: position after prompt (and indicator if present)
        if content.is_empty() {
            let multiline = self.input.is_multiline();
            let prompt_width = UnicodeWidthStr::width("> ");
            let indicator_width = if multiline {
                UnicodeWidthStr::width("📝 ")
            } else {
                0
            };
            let inner_width = self.input_inner_width(area.width);
            let offset = prompt_width + if multiline { indicator_width } else { 0 };
            let cursor_x = area
                .x
                .saturating_add(1 + offset.min(inner_width) as u16)
                .min(area.right().saturating_sub(1));
            let cursor_y = area
                .y
                .saturating_add(1)
                .min(area.bottom().saturating_sub(1));
            return (cursor_x, cursor_y);
        }

        let (_, cursor_line_col) = self.wrap_content(area.width, Some(cursor_byte));

        if let Some((line_idx, col)) = cursor_line_col {
            let cursor_x = area
                .x
                .saturating_add(1 + col as u16)
                .min(area.right().saturating_sub(1));
            let cursor_y = area
                .y
                .saturating_add(1 + line_idx as u16)
                .min(area.bottom().saturating_sub(1));
            (cursor_x, cursor_y)
        } else {
            // Fallback: end of content
            let cursor_x = area.x.saturating_add(1).min(area.right().saturating_sub(1));
            let cursor_y = area
                .y
                .saturating_add(1 + self.displayed_line_count(area.width).saturating_sub(1) as u16)
                .min(area.bottom().saturating_sub(1));
            (cursor_x, cursor_y)
        }
    }

    /// Wrap the current content using word-aware wrapping to mirror ratatui's Paragraph.
    /// Returns (total rendered lines, optional (line_idx, column_width) for the cursor).
    fn wrap_content(
        &self,
        available_width: u16,
        cursor_byte: Option<usize>,
    ) -> (usize, Option<(usize, usize)>) {
        const CURSOR_SENTINEL: char = '\u{200b}'; // zero-width so it won't affect wrap width

        let width = self.input_inner_width(available_width);
        let content = self.input.content();
        let multiline = self.input.is_multiline() || content.contains('\n');
        let prompt_width = UnicodeWidthStr::width("> ");
        let indicator_width = if multiline {
            UnicodeWidthStr::width("📝 ")
        } else {
            0
        };
        let continuation_indent = if multiline {
            UnicodeWidthStr::width("  ")
        } else {
            0
        };

        let mut total_lines = 0usize;
        let mut cursor_location: Option<(usize, usize)> = None;
        let mut global_byte_offset = 0usize;

        for (line_idx, line) in content.split('\n').enumerate() {
            let line_start = global_byte_offset;
            let line_end = line_start + line.len();
            let contains_cursor = cursor_byte.map_or(false, |c| c >= line_start && c <= line_end);

            let mut line_content = line.to_string();
            if contains_cursor {
                if let Some(cursor) = cursor_byte {
                    let local_offset = cursor.saturating_sub(line_start);
                    let insert_at = line_content
                        .char_indices()
                        .nth(local_offset)
                        .map(|(idx, _)| idx)
                        .unwrap_or_else(|| line_content.len());
                    line_content.insert(insert_at, CURSOR_SENTINEL);
                }
            }

            let (first_prefix, subsequent_prefix) = match (multiline, line_idx) {
                (true, 0) => (indicator_width + prompt_width, continuation_indent),
                (true, _) => (continuation_indent, continuation_indent),
                (false, 0) => (prompt_width, 0),
                (false, _) => (0, 0),
            };

            let initial_indent = " ".repeat(first_prefix.min(width));
            let subsequent_indent = " ".repeat(subsequent_prefix.min(width));

            let options = Options::new(width)
                .initial_indent(initial_indent.as_str())
                .subsequent_indent(subsequent_indent.as_str())
                .break_words(true);

            let wrapped = textwrap::wrap(line_content.as_str(), &options);
            let wrapped_iter = if wrapped.is_empty() {
                vec![initial_indent.clone().into()]
            } else {
                wrapped
            };

            for wrapped_line in wrapped_iter {
                if contains_cursor && cursor_location.is_none() {
                    if let Some(pos) = wrapped_line.find(CURSOR_SENTINEL) {
                        let col = UnicodeWidthStr::width(&wrapped_line[..pos]);
                        cursor_location = Some((total_lines, col));
                    }
                }
                total_lines = total_lines.saturating_add(1);
            }

            global_byte_offset = line_end.saturating_add(1); // account for newline
        }

        (total_lines.max(1), cursor_location)
    }

    /// Render the input area.
    fn render_input(&self, frame: &mut Frame, area: Rect) {
        // Render completion popup above the input area if visible
        if self.input.completion().is_visible() {
            widgets::render_completion_popup(
                frame,
                area,
                self.input.completion().suggestions(),
                self.input.completion().selected_index(),
            );
        }

        let paragraph = widgets::render_input_box(
            self.input.content(),
            self.input.cursor(),
            self.mode,
            self.input.is_multiline(),
        );
        frame.render_widget(paragraph, area);

        // Render cursor with proper multiline positioning
        // Always show cursor at the correct position - let the terminal handle blinking
        let (cursor_x, cursor_y) = self.calculate_cursor_position(area);
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    /// Render the bottom bar (matching Mistral Vibe layout).
    fn render_bottom_bar(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(1),     // Path display (flexible width)
                Constraint::Min(1),     // Spacer
                Constraint::Length(25), // Reserved space for context progress (now handled by context_progress component)
            ])
            .split(area);

        // Path display
        self.path_display.render(chunks[0], frame.buffer_mut());

        // Spacer (empty)

        // Note: Context progress is now handled by the context_progress component
        // to avoid duplicate rendering and ensure proper alignment
    }

    /// Handle scroll up (matching Python's scroll_relative(y=-5)).
    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.auto_scroll = false;
            self.scroll_offset = self.scroll_offset.saturating_sub(5);
        }
    }

    /// Handle scroll down (matching Python's scroll_relative(y=5)).
    pub fn scroll_down(&mut self) {
        if self.scroll_offset < self.max_scroll {
            self.scroll_offset = (self.scroll_offset + 5).min(self.max_scroll);
        }
        // Re-enable auto-scroll when at bottom
        if self.scroll_offset >= self.max_scroll {
            self.auto_scroll = true;
        }
    }

    /// Check if scrolled to bottom.
    fn is_scrolled_to_bottom(&self) -> bool {
        self.scroll_offset >= self.max_scroll.saturating_sub(2)
    }

    /// Scroll to a specific position.
    #[allow(dead_code)]
    pub fn scroll_to(&mut self, position: u16) {
        self.auto_scroll = false;
        self.scroll_offset = position.min(self.max_scroll);
    }

    /// Handle mouse events (for scrolling, matching Python's scroll amount).
    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                // Scroll up by 5 lines (matching Python's scroll_relative)
                self.auto_scroll = false;
                self.scroll_offset = self.scroll_offset.saturating_sub(5);
            }
            MouseEventKind::ScrollDown => {
                // Scroll down by 5 lines
                self.scroll_offset = (self.scroll_offset + 5).min(self.max_scroll);
                if self.is_scrolled_to_bottom() {
                    self.auto_scroll = true;
                }
            }
            _ => {
                // Ignore other mouse events (click, drag, etc.)
            }
        }
    }
}

/// Terminal wrapper for the TUI.
pub struct TerminalWrapper {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalWrapper {
    /// Create and setup the terminal.
    pub fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(
            stdout,
            Clear(ClearType::All),
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBlinking
        )?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    /// Draw a frame.
    pub fn draw(&mut self, app: &mut App) -> io::Result<()> {
        self.terminal.draw(|frame| app.render(frame))?;
        Ok(())
    }

    /// Restore the terminal.
    pub fn restore(&mut self) -> io::Result<()> {
        disable_raw_mode()?;
        execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            DisableBlinking
        )?;
        self.terminal.show_cursor()?;
        Ok(())
    }
}

impl Drop for TerminalWrapper {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}
