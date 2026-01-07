//! Context progress indicator component.

use crate::ui::widgets::colors;
use ratatui::{prelude::*, widgets::Paragraph};
use std::time::Instant;

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
    /// Smoothed token count shown in the UI.
    displayed_tokens: u32,
    /// Timestamp of the previous animation tick.
    last_tick: Instant,
}

impl ContextProgress {
    /// Create a new ContextProgress.
    pub fn new() -> Self {
        Self {
            tokens: TokenState::default(),
            displayed_tokens: 0,
            last_tick: Instant::now(),
        }
    }

    /// Update the token state.
    pub fn update_tokens(&mut self, tokens: TokenState) {
        // Context reductions (compaction/reset) should show immediately.
        if tokens.current_tokens < self.displayed_tokens {
            self.displayed_tokens = tokens.current_tokens;
        }
        self.tokens = tokens;
    }

    /// Advance UI interpolation toward the latest token count.
    pub fn tick(&mut self, loading: bool) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f64();
        self.last_tick = now;

        let target = self.tokens.current_tokens;
        if self.displayed_tokens >= target {
            self.displayed_tokens = target;
            return;
        }

        let gap = target - self.displayed_tokens;
        // UI-only interpolation. Keeps display responsive without changing backend behavior.
        let base_rate = if loading { 1800.0 } else { 3200.0 };
        let proportional_rate = (f64::from(gap) * 3.0).min(12000.0);
        let step = ((base_rate + proportional_rate) * dt).round().max(1.0) as u32;

        self.displayed_tokens = self.displayed_tokens.saturating_add(step).min(target);
    }

    /// Render the context progress.
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let state = TokenState::new(self.tokens.max_tokens, self.displayed_tokens);
        let text = state.progress_text();
        if text.is_empty() {
            return;
        }

        let paragraph = Paragraph::new(text)
            .style(Style::default().fg(colors::ACCENT))
            .alignment(Alignment::Right);

        frame.render_widget(paragraph, area);
    }
}
