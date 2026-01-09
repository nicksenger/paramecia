//! Path display widget.

use ratatui::prelude::*;
use std::path::PathBuf;

use super::widgets::colors;

/// Path display widget.
pub struct PathDisplay {
    /// Current working directory.
    pub path: PathBuf,
}

impl PathDisplay {
    /// Create a new path display.
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Get the display text, replacing home directory with ~.
    fn display_text(&self) -> String {
        let path_str = self.path.to_string_lossy().to_string();

        // Replace home directory with ~
        if let Ok(home) = std::env::var("HOME")
            && path_str.starts_with(&home)
        {
            return format!("~{}", &path_str[home.len()..]);
        }

        path_str
    }

    /// Render the path display.
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let path_text = self.display_text();
        let line = Line::from(vec![Span::styled(
            path_text,
            Style::default().fg(colors::PRIMARY),
        )]);
        buf.set_line(area.x, area.y, &line, area.width);
    }

    /// Update the path.
    #[allow(dead_code)]
    pub fn set_path(&mut self, path: PathBuf) {
        self.path = path;
    }
}
