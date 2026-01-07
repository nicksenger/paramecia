//! Config/App Settings UI component.

use paramecia_harness::VibeConfig;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

/// Config/App Settings UI state.
#[derive(Debug, Clone)]
pub struct ConfigApp {
    /// Configuration.
    pub config: VibeConfig,
    /// Available models.
    pub models: Vec<String>,
    /// Available themes.
    pub themes: Vec<String>,
    /// Selected index.
    pub selected_index: usize,
    /// Changes made.
    pub changes: std::collections::HashMap<String, String>,
}

impl ConfigApp {
    /// Create a new ConfigApp.
    pub fn new(config: VibeConfig) -> Self {
        let models = config
            .models
            .iter()
            .map(|m| m.alias().to_string())
            .collect();

        let themes = vec![
            "default".to_string(),
            "dark".to_string(),
            "light".to_string(),
            "monokai".to_string(),
            "solarized".to_string(),
        ];

        Self {
            config,
            models,
            themes,
            selected_index: 0,
            changes: std::collections::HashMap::new(),
        }
    }

    /// Get the current settings as list items.
    pub fn get_settings_list(&self) -> Vec<ListItem<'_>> {
        let mut items = Vec::new();

        // Active Model
        let current_model = self
            .changes
            .get("active_model")
            .unwrap_or(&self.config.active_model);
        items.push(ListItem::new(format!("Model: {}", current_model)));

        // Theme
        let current_theme = self
            .changes
            .get("textual_theme")
            .unwrap_or(&self.config.textual_theme);
        items.push(ListItem::new(format!("Theme: {}", current_theme)));

        items
    }

    /// Handle key input.
    pub fn handle_input(&mut self, key: crossterm::event::KeyEvent) -> Option<ConfigAction> {
        match key.code {
            crossterm::event::KeyCode::Up => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                }
            }
            crossterm::event::KeyCode::Down => {
                if self.selected_index < self.get_settings_list().len() - 1 {
                    self.selected_index += 1;
                }
            }
            crossterm::event::KeyCode::Enter | crossterm::event::KeyCode::Char(' ') => {
                return self.toggle_setting();
            }
            crossterm::event::KeyCode::Esc => {
                return Some(ConfigAction::Close);
            }
            _ => {}
        }
        None
    }

    /// Toggle the current setting.
    fn toggle_setting(&mut self) -> Option<ConfigAction> {
        match self.selected_index {
            0 => self.cycle_model(),
            1 => self.cycle_theme(),
            _ => None,
        }
    }

    /// Cycle through available models.
    fn cycle_model(&mut self) -> Option<ConfigAction> {
        let current_model = self
            .changes
            .get("active_model")
            .unwrap_or(&self.config.active_model);

        if let Some(current_idx) = self.models.iter().position(|m| m == current_model) {
            let next_idx = (current_idx + 1) % self.models.len();
            let next_model = self.models[next_idx].clone();
            self.changes.insert("active_model".to_string(), next_model);
            Some(ConfigAction::SettingChanged("active_model".to_string()))
        } else {
            None
        }
    }

    /// Cycle through available themes.
    fn cycle_theme(&mut self) -> Option<ConfigAction> {
        let current_theme = self
            .changes
            .get("textual_theme")
            .unwrap_or(&self.config.textual_theme);

        if let Some(current_idx) = self.themes.iter().position(|t| t == current_theme) {
            let next_idx = (current_idx + 1) % self.themes.len();
            let next_theme = self.themes[next_idx].clone();
            self.changes.insert("textual_theme".to_string(), next_theme);
            Some(ConfigAction::SettingChanged("textual_theme".to_string()))
        } else {
            None
        }
    }

    /// Render the config app.
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default().title("Settings").borders(Borders::ALL);

        let inner_area = block.inner(area);
        frame.render_widget(block, area);

        let list = List::new(self.get_settings_list())
            .highlight_style(Style::default().fg(Color::Yellow))
            .highlight_symbol("> ");

        let mut state = ListState::default();
        state.select(Some(self.selected_index));

        frame.render_stateful_widget(list, inner_area, &mut state);

        // Help text
        let help_text = Paragraph::new("↑↓ Navigate  Space/Enter Toggle  ESC Exit")
            .style(Style::default().fg(Color::Gray));

        let help_area = Rect {
            x: inner_area.x,
            y: inner_area.bottom() - 1,
            width: inner_area.width,
            height: 1,
        };

        frame.render_widget(help_text, help_area);
    }
}

/// Actions that can be taken from the config app.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigAction {
    /// Close the config app.
    Close,
    /// A setting was changed.
    SettingChanged(String),
}
