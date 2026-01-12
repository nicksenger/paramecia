//! Context progress indicator component.

use crate::ui::widgets::colors;
use ratatui::{prelude::*, widgets::Paragraph};

/// Token state for context progress.
#[derive(Debug, Clone, Default)]
pub struct TokenState {
    /// Maximum tokens.
    pub max_tokens: u32,
    /// Current tokens.
    pub current_tokens: u32,
}

impl TokenState {
    /// Create a new TokenState.
    pub fn new(max_tokens: u32, current_tokens: u32) -> Self {
        Self {
            max_tokens,
            current_tokens,
        }
    }

    /// Get the progress percentage.
    pub fn percentage(&self) -> u8 {
        if self.max_tokens == 0 {
            return 0;
        }
        ((self.current_tokens as f32 / self.max_tokens as f32) * 100.0).min(100.0) as u8
    }

    /// Get the progress text (matching Python's "X% (current/max) tokens" format).
    pub fn progress_text(&self) -> String {
        if self.max_tokens == 0 {
            return String::new();
        }
        let percentage = self.percentage();
        format!(
            "{}% ({}/{}) tokens",
            percentage, self.current_tokens, self.max_tokens
        )
    }
}

/// Context progress indicator.
#[derive(Debug, Clone)]
pub struct ContextProgress {
    /// Token state.
    pub tokens: TokenState,
}

impl ContextProgress {
    /// Create a new ContextProgress.
    pub fn new() -> Self {
        Self {
            tokens: TokenState::default(),
        }
    }

    /// Update the token state.
    pub fn update_tokens(&mut self, tokens: TokenState) {
        self.tokens = tokens;
    }

    /// Render the context progress.
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let text = self.tokens.progress_text();
        if text.is_empty() {
            return;
        }

        let paragraph = Paragraph::new(text)
            .style(Style::default().fg(colors::ACCENT))
            .alignment(Alignment::Right);

        frame.render_widget(paragraph, area);
    }
}
