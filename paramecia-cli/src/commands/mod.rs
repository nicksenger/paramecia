//! Command system for Paramecia CLI.

mod agent_worker;
pub mod interactive;
pub mod programmatic;
pub mod setup;

pub use agent_worker::{AgentCommand, AgentHandle, AgentResult};

use std::collections::HashMap;

/// A command that can be executed by the user.
#[derive(Debug, Clone)]
pub struct Command {
    /// The command name (e.g., "help", "status").
    pub name: String,
    /// The command description.
    pub description: String,
    /// Whether the command is hidden from help.
    pub hidden: bool,
    /// Command aliases (e.g., "/help" for "help").
    pub aliases: Vec<String>,
}

impl Command {
    /// Create a new command.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            hidden: false,
            aliases: vec![format!("/{}", name)],
        }
    }

    /// Create a new command with custom aliases.
    pub fn with_aliases(name: &str, description: &str, aliases: Vec<&str>) -> Self {
        let mut all_aliases: Vec<String> = aliases.iter().map(|a| a.to_string()).collect();
        all_aliases.push(format!("/{}", name));
        all_aliases.sort();
        all_aliases.dedup();

        Self {
            name: name.to_string(),
            description: description.to_string(),
            hidden: false,
            aliases: all_aliases,
        }
    }
}

/// Registry of available commands.
#[derive(Debug, Default)]
pub struct CommandRegistry {
    commands: HashMap<String, Command>,
}

impl CommandRegistry {
    /// Create a new command registry.
    pub fn new() -> Self {
        Self {
            commands: HashMap::new(),
        }
    }

    /// Register a command.
    pub fn register(&mut self, command: Command) {
        self.commands.insert(command.name.clone(), command);
    }

    /// Get all visible commands.
    pub fn visible_commands(&self) -> Vec<&Command> {
        self.commands.values().filter(|cmd| !cmd.hidden).collect()
    }

    /// Get help text for all commands.
    pub fn get_help_text(&self) -> String {
        let mut commands: Vec<_> = self.visible_commands();
        commands.sort_by(|a, b| a.name.cmp(&b.name));

        let mut help_text = String::new();
        help_text.push_str("## 🤖 Paramecia Help\n\n");
        help_text.push_str("### 🎹 Keyboard Shortcuts\n\n");
        help_text.push_str("- `Enter` Submit message\n");
        help_text.push_str("- `Ctrl+J` / `Shift+Enter` Insert newline\n");
        help_text.push_str("- `Escape` Interrupt agent or close dialogs\n");
        help_text.push_str("- `Ctrl+C` Quit (or clear input if text present)\n");
        help_text.push_str("- `Ctrl+D` Force quit\n");
        help_text.push_str("- `Ctrl+O` Toggle tool output view\n");

        help_text.push_str("- `Shift+Tab` Toggle auto-approve mode\n");
        help_text.push_str("- `Shift+Up` Scroll chat up\n");
        help_text.push_str("- `Shift+Down` Scroll chat down\n");
        help_text.push_str("- `Up/Down` Navigate command history\n");
        help_text.push_str("- `Tab` Autocomplete commands and paths\n");
        help_text.push_str("- `Ctrl+A` Move cursor to beginning of line\n");
        help_text.push_str("- `Ctrl+E` Move cursor to end of line\n");
        help_text.push_str("- `Ctrl+U` Clear line from cursor to beginning\n");
        help_text.push_str("- `Ctrl+W` Delete word before cursor\n");
        help_text.push_str("- `Ctrl+K` Clear line from cursor to end\n");
        help_text.push_str("- `Ctrl+P` Previous history (alternative to Up)\n");
        help_text.push_str("- `Ctrl+N` Next history (alternative to Down)\n");
        help_text.push('\n');
        help_text.push_str("### 📝 Special Features\n\n");
        help_text.push_str("- `!<command>` Execute bash command directly\n");
        help_text.push_str("- `@path/to/file/` Autocompletes file paths\n");
        help_text.push('\n');
        help_text.push_str("### 🎛️  Agent Modes\n\n");
        help_text.push_str("- **Default** - Asks for approval on modifications\n");
        help_text.push_str("- **Plan** - Planning mode, no actions executed\n");
        help_text.push_str("- **Accept Edits** - Auto-approves file changes\n");
        help_text.push_str("- **Auto Approve** - Auto-approves all actions (YOLO)\n");
        help_text.push('\n');
        help_text.push_str("### 📋 Commands\n\n");

        for cmd in commands {
            help_text.push_str(&format!("- `/{}`  {}\n", cmd.name, cmd.description));
        }

        help_text.push('\n');
        help_text.push_str("### 💡 Tips\n\n");
        help_text.push_str("- Use `/help` to show this message anytime\n");
        help_text.push_str("- Press `Ctrl+C` twice to force quit\n");
        help_text.push_str("- Use `!ls` or `!pwd` to check your working directory\n");
        help_text.push_str("- Type `/config` to customize your experience\n");

        help_text
    }

    /// Get completion items for all commands.
    /// Returns a list of (command_name, description) tuples.
    pub fn get_completion_items(&self) -> Vec<(String, String)> {
        let mut items: Vec<(String, String)> = Vec::new();

        for cmd in self.commands.values().filter(|cmd| !cmd.hidden) {
            let primary = format!("/{}", cmd.name);
            items.push((primary.clone(), cmd.description.clone()));

            for alias in &cmd.aliases {
                if alias != &primary {
                    items.push((alias.clone(), cmd.description.clone()));
                }
            }
        }

        items.sort_by(|a, b| a.0.cmp(&b.0));
        items.dedup_by(|a, b| a.0 == b.0);
        items
    }
}

/// Default command registry with all built-in commands.
pub fn create_default_registry() -> CommandRegistry {
    let mut registry = CommandRegistry::new();

    // Help command
    registry.register(Command::new("help", "Show help message"));

    // Config command with aliases
    registry.register(Command::with_aliases(
        "config",
        "Edit config settings",
        vec!["/config", "/theme", "/model"],
    ));

    // Reload command
    registry.register(Command::new("reload", "Reload configuration from disk"));

    // Clear command
    registry.register(Command::new("clear", "Clear conversation history"));

    // Log command (matching original naming)
    registry.register(Command::new(
        "log",
        "Show path to current interaction log file",
    ));

    // Compact command
    registry.register(Command::new(
        "compact",
        "Compact conversation history by summarizing",
    ));

    // Exit command
    registry.register(Command::new("exit", "Exit the application"));

    // Terminal setup command
    registry.register(Command::with_aliases(
        "terminal-setup",
        "Configure Shift+Enter for newlines",
        vec!["/terminal-setup"],
    ));

    // Theme command (already exists, but let's make sure it has proper aliases)
    registry.register(Command::with_aliases(
        "theme",
        "Change color theme",
        vec!["/theme"],
    ));

    // Model command (alias for config)
    registry.register(Command::with_aliases(
        "model",
        "Change LLM model",
        vec!["/model"],
    ));

    // Status command
    registry.register(Command::new("status", "Display agent statistics"));

    // Stats command (alias for status)
    registry.register(Command::with_aliases(
        "stats",
        "Display agent statistics",
        vec!["/stats"],
    ));

    // Reset command
    registry.register(Command::new(
        "reset",
        "Reset conversation and clear history",
    ));

    // Session command
    registry.register(Command::new("session", "Show current session information"));

    // Theme command
    registry.register(Command::new("theme", "Change color theme"));

    // Update command
    registry.register(Command::new("update", "Check for updates"));

    registry
}
