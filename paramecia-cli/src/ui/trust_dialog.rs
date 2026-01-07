//! Trust folder dialog UI component.

use std::io::{self, Stdout};
use std::path::Path;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    prelude::*,
    widgets::{Block, BorderType, Borders, Padding, Paragraph},
};

use super::widgets::colors;

/// Result of the trust folder dialog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustDialogResult {
    /// User trusted the folder.
    Trusted,
    /// User did not trust the folder.
    Untrusted,
    /// User quit without making a selection.
    Quit,
}

/// Trust folder dialog UI.
pub struct TrustDialog {
    folder_path: String,
    selected_option: usize, // 0 = Yes, 1 = No
    should_quit: bool,
}

impl TrustDialog {
    /// Create a new trust dialog.
    pub fn new(folder_path: &Path) -> Self {
        Self {
            folder_path: folder_path.to_string_lossy().to_string(),
            selected_option: 1, // Default to "No"
            should_quit: false,
        }
    }

    /// Run the trust dialog and return the result.
    pub fn run(&mut self) -> io::Result<TrustDialogResult> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Run the dialog
        let result = self.run_dialog(&mut terminal);

        // Cleanup terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(result)
    }

    /// Run the dialog loop.
    fn run_dialog(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> TrustDialogResult {
        loop {
            if self.should_quit {
                return TrustDialogResult::Quit;
            }

            terminal.draw(|f| self.render(f)).expect("Failed to draw");

            if let Event::Key(key) = event::read().expect("Failed to read event") {
                match key.code {
                    KeyCode::Left | KeyCode::Char('h') => {
                        self.selected_option = (self.selected_option + 1) % 2;
                    }
                    KeyCode::Right | KeyCode::Char('l') => {
                        self.selected_option = (self.selected_option + 1) % 2;
                    }
                    KeyCode::Char('1') | KeyCode::Char('y') => {
                        return TrustDialogResult::Trusted;
                    }
                    KeyCode::Char('2') | KeyCode::Char('n') => {
                        return TrustDialogResult::Untrusted;
                    }
                    KeyCode::Enter => {
                        return match self.selected_option {
                            0 => TrustDialogResult::Trusted,
                            1 => TrustDialogResult::Untrusted,
                            _ => TrustDialogResult::Untrusted,
                        };
                    }
                    KeyCode::Char('q') | KeyCode::Esc => {
                        if key.modifiers.contains(KeyModifiers::CONTROL) {
                            self.should_quit = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Render the trust dialog.
    fn render(&self, frame: &mut Frame) {
        let area = frame.area();

        // Create main block with rounded borders and proper styling
        let block = Block::default()
            .title("⚠ Trust this folder?")
            .title_style(Style::default().fg(colors::WARNING).bold())
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(colors::MUTED))
            .style(colors::dialog_bg());

        let inner_area = block.inner(area);
        frame.render_widget(block, area);

        // Folder path - centered and styled like original
        let path_paragraph = Paragraph::new(self.folder_path.clone())
            .style(colors::path_text())
            .alignment(Alignment::Center)
            .block(Block::default().padding(Padding::horizontal(1)));
        frame.render_widget(path_paragraph, inner_area);

        // Message - centered and with proper spacing
        let message = "A .vibe/ directory was found here. Should Vibe load custom configuration and tools from it?";
        let message_paragraph = Paragraph::new(message)
            .style(colors::dialog_text())
            .alignment(Alignment::Center)
            .block(Block::default().padding(Padding::horizontal(1)));
        let message_area = Rect {
            x: inner_area.x,
            y: inner_area.y + 2,
            width: inner_area.width,
            height: 3,
        };
        frame.render_widget(message_paragraph, message_area);

        // Options - centered with proper spacing
        let options = ["Yes", "No"];
        let options_area = Rect {
            x: inner_area.x,
            y: inner_area.y + 6,
            width: inner_area.width,
            height: 1,
        };

        // Create a centered container for options
        let options_container = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(options_area);

        for (i, option) in options.iter().enumerate() {
            let cursor = if i == self.selected_option {
                ">> "
            } else {
                "  "
            };
            let option_text = format!("{} {}. {}", cursor, i + 1, option);
            let option_paragraph = Paragraph::new(option_text)
                .style(if i == self.selected_option {
                    colors::selected_option()
                } else {
                    colors::option_text()
                })
                .alignment(Alignment::Center);

            frame.render_widget(option_paragraph, options_container[i]);
        }

        // Help text - centered and styled
        let help_text = "← → navigate  Enter select";
        let help_paragraph = Paragraph::new(help_text)
            .style(colors::help_text())
            .alignment(Alignment::Center)
            .block(Block::default().padding(Padding::horizontal(1)));
        let help_area = Rect {
            x: inner_area.x,
            y: inner_area.y + 8,
            width: inner_area.width,
            height: 1,
        };
        frame.render_widget(help_paragraph, help_area);

        // Save info - centered and italic like original
        let save_info = format!(
            "Setting will be saved in: {}",
            paramecia_harness::paths::TRUSTED_FOLDERS_FILE.display()
        );
        let save_paragraph = Paragraph::new(save_info)
            .style(colors::save_info_text().add_modifier(Modifier::ITALIC))
            .alignment(Alignment::Center)
            .block(Block::default().padding(Padding::horizontal(1)));
        let save_area = Rect {
            x: inner_area.x,
            y: inner_area.y + 10,
            width: inner_area.width,
            height: 1,
        };
        frame.render_widget(save_paragraph, save_area);
    }
}
