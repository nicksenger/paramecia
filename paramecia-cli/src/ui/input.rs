//! Text input handling for the chat interface.
//!
//! Text input features:
//! - Up/down arrows only trigger history when cursor is on first/last line
//! - Proper multiline cursor navigation
//! - Prefix-based history filtering

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::completion::CompletionManager;
use super::history::HistoryManager;

/// State for text input.
#[derive(Debug)]
pub struct InputState {
    /// Current input text.
    content: String,
    /// Cursor position (byte offset).
    cursor: usize,
    /// Persistent history manager.
    history: Option<HistoryManager>,
    /// Current history prefix for filtering.
    history_prefix: Option<String>,
    /// Last cursor column used for history navigation.
    last_cursor_col: usize,
    /// Last used prefix for history (to detect changes).
    last_used_prefix: Option<String>,
    /// Multiline input mode.
    multiline: bool,
    /// Completion manager.
    completion: CompletionManager,
    /// Input mode (>, !, /) similar to Python version.
    input_mode: InputMode,
    /// Whether we're currently navigating history.
    navigating_history: bool,
    /// Original text before history navigation.
    original_text: String,
    /// Cursor position after loading from history.
    cursor_pos_after_load: Option<(usize, usize)>,
    /// Whether cursor has moved since loading from history.
    cursor_moved_since_load: bool,
}

/// Input mode enum matching Python version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    /// Normal input mode (default).
    Normal,
    /// Command mode (starts with /).
    Command,
    /// Bash mode (starts with !).
    Bash,
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

impl InputState {
    /// Create a new empty input state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            content: String::new(),
            cursor: 0,
            history: None,
            history_prefix: None,
            last_cursor_col: 0,
            last_used_prefix: None,
            multiline: false,
            completion: CompletionManager::default(),
            input_mode: InputMode::Normal,
            navigating_history: false,
            original_text: String::new(),
            cursor_pos_after_load: None,
            cursor_moved_since_load: false,
        }
    }

    /// Create a new input state with persistent history and commands.
    #[must_use]
    pub fn with_history(history_file: PathBuf) -> Self {
        Self {
            content: String::new(),
            cursor: 0,
            history: Some(HistoryManager::new(history_file, 100)),
            history_prefix: None,
            last_cursor_col: 0,
            last_used_prefix: None,
            multiline: false,
            completion: CompletionManager::default(),
            input_mode: InputMode::Normal,
            navigating_history: false,
            original_text: String::new(),
            cursor_pos_after_load: None,
            cursor_moved_since_load: false,
        }
    }

    /// Set the available commands for completion.
    pub fn set_commands(&mut self, commands: Vec<(String, String)>) {
        self.completion = CompletionManager::new(commands);
    }

    /// Get the completion manager.
    #[must_use]
    pub fn completion(&self) -> &CompletionManager {
        &self.completion
    }

    /// Get the current input content.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Set the current input content (for testing purposes).
    #[allow(dead_code)]
    pub fn set_content(&mut self, content: String) {
        self.multiline = content.contains('\n');
        self.cursor = content.len();
        self.content = content;
    }

    /// Get the cursor position.
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Get a safe cursor position that is guaranteed to be on a UTF-8 char boundary.
    /// This is a defensive method to prevent panics from invalid cursor positions.
    fn safe_cursor(&self) -> usize {
        if self.cursor <= self.content.len() && self.content.is_char_boundary(self.cursor) {
            self.cursor
        } else {
            // Find the nearest valid char boundary at or before cursor
            let mut pos = self.cursor.min(self.content.len());
            while pos > 0 && !self.content.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        }
    }

    /// Get cursor position as (row, col) in character coordinates.
    /// Row 0 is the first line.
    #[must_use]
    pub fn cursor_location(&self) -> (usize, usize) {
        if self.content.is_empty() {
            return (0, 0);
        }

        let safe_cursor = self.safe_cursor();
        let text_before_cursor = &self.content[..safe_cursor];
        let lines: Vec<&str> = text_before_cursor.split('\n').collect();
        let row = lines.len().saturating_sub(1);
        let col = lines.last().map(|l| l.chars().count()).unwrap_or(0);
        (row, col)
    }

    /// Get total number of lines in the content.
    #[must_use]
    pub fn line_count(&self) -> usize {
        if self.content.is_empty() {
            return 1;
        }
        self.content.split('\n').count()
    }

    /// Check if cursor is on the first line.
    #[must_use]
    fn is_cursor_on_first_line(&self) -> bool {
        self.cursor_location().0 == 0
    }

    /// Check if cursor is on the last line.
    #[must_use]
    fn is_cursor_on_last_line(&self) -> bool {
        self.cursor_location().0 == self.line_count().saturating_sub(1)
    }

    /// Get the prefix up to cursor position (for history filtering).
    /// Like Python's _get_prefix_up_to_cursor.
    fn get_prefix_up_to_cursor(&self) -> String {
        let (row, col) = self.cursor_location();
        let lines: Vec<&str> = self.content.split('\n').collect();

        if row < lines.len() {
            let line = lines[row];
            let prefix: String = line.chars().take(col).collect();

            // Include mode prefix on first line
            if row == 0 && self.input_mode != InputMode::Normal {
                match self.input_mode {
                    InputMode::Command => format!("/{}", prefix),
                    InputMode::Bash => format!("!{}", prefix),
                    InputMode::Normal => prefix,
                }
            } else {
                prefix
            }
        } else {
            String::new()
        }
    }

    /// Mark cursor as moved if position changed since history load.
    fn mark_cursor_moved_if_needed(&mut self) {
        if let Some(load_pos) = self.cursor_pos_after_load
            && !self.cursor_moved_since_load
            && self.cursor_location() != load_pos
        {
            self.cursor_moved_since_load = true;
            self.reset_prefix();
        }
    }

    /// Reset history prefix state.
    fn reset_prefix(&mut self) {
        self.history_prefix = None;
        self.last_used_prefix = None;
    }

    /// Check if input is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Clear the input.
    pub fn clear(&mut self) {
        self.content.clear();
        self.cursor = 0;
        self.reset_history_navigation();
        self.input_mode = InputMode::Normal;
        self.multiline = false;
    }

    /// Clear from cursor to beginning of line.
    pub fn clear_to_beginning(&mut self) {
        if self.cursor > 0 {
            self.content.drain(0..self.cursor);
            self.cursor = 0;
            self.on_text_changed();
        }
    }

    /// Delete word before cursor.
    pub fn delete_word(&mut self) {
        if self.cursor == 0 {
            return;
        }

        // Find the start of the current word
        let mut word_start = self.cursor;
        let chars: Vec<char> = self.content.chars().collect();

        while word_start > 0 {
            let prev_char = chars[word_start - 1];
            if prev_char.is_whitespace() {
                break;
            }
            word_start -= 1;
        }

        // Delete from word_start to cursor
        let byte_word_start = self
            .content
            .char_indices()
            .nth(word_start)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let byte_cursor = self
            .content
            .char_indices()
            .nth(self.cursor)
            .map(|(i, _)| i)
            .unwrap_or(self.content.len());

        self.content.replace_range(byte_word_start..byte_cursor, "");
        self.cursor = word_start;
        self.on_text_changed();
    }

    /// Clear from cursor to end of line.
    pub fn clear_to_end(&mut self) {
        if self.cursor < self.content.len() {
            self.content.truncate(self.cursor);
            self.on_text_changed();
        }
    }

    /// Move cursor to beginning of line.
    pub fn move_cursor_to_beginning(&mut self) {
        self.cursor = 0;
    }

    /// Move cursor to end of line.
    pub fn move_cursor_to_end(&mut self) {
        self.cursor = self.content.len();
    }

    /// Navigate history up (previous command).
    /// Like Python's _handle_history_up - only triggers on first line.
    pub fn navigate_history_up(&mut self) {
        let (row, col) = self.cursor_location();

        // Only trigger history navigation when on first line
        if row != 0 {
            return;
        }

        // Check if we have history
        if self.history.is_none() {
            return;
        }

        // Reset prefix if cursor column changed
        if self.history_prefix.is_some() && col != self.last_cursor_col {
            self.reset_prefix();
            self.last_cursor_col = 0;
        }

        // Initialize prefix on first navigation
        if self.history_prefix.is_none() {
            self.history_prefix = Some(self.get_prefix_up_to_cursor());
        }

        // Save original text on first navigation
        if !self.navigating_history {
            self.original_text = self.content.clone();
            self.navigating_history = true;
        }

        let prefix = self.history_prefix.clone().unwrap_or_default();
        let original = self.original_text.clone();

        // Now access history
        if let Some(history) = &mut self.history
            && let Some(prev) = history.get_previous(&original, &prefix)
        {
            self.load_history_entry(&prev, None);
        }
    }

    /// Navigate history down (next command).
    /// Like Python's _handle_history_down - only triggers on last line or when navigating on first line.
    pub fn navigate_history_down(&mut self) {
        let (row, col) = self.cursor_location();
        let total_lines = self.line_count();

        // Check conditions for intercepting down arrow (matching Python logic)
        let on_first_line_unmoved = row == 0 && !self.cursor_moved_since_load;
        let on_last_line = row == total_lines.saturating_sub(1);

        let should_intercept =
            (on_first_line_unmoved && self.history_prefix.is_some()) || on_last_line;

        if !should_intercept {
            return;
        }

        // Check if we have history
        if self.history.is_none() {
            return;
        }

        // Reset prefix if cursor column changed
        if self.history_prefix.is_some() && col != self.last_cursor_col {
            self.reset_prefix();
            self.last_cursor_col = 0;
        }

        // Initialize prefix if needed
        if self.history_prefix.is_none() {
            self.history_prefix = Some(self.get_prefix_up_to_cursor());
        }

        let prefix = self.history_prefix.clone().unwrap_or_default();

        // Get next entry and check navigation state
        let (next_entry, cursor_col) = {
            if let Some(history) = &mut self.history {
                let next = history.get_next(&prefix);
                let cursor_col = if !history.is_navigating() {
                    Some(prefix.len())
                } else {
                    None
                };
                (next, cursor_col)
            } else {
                (None, None)
            }
        };

        if let Some(next) = next_entry {
            self.load_history_entry(&next, cursor_col);
        }
    }

    /// Load a history entry into the input.
    /// Like Python's _load_history_entry.
    fn load_history_entry(&mut self, text: &str, cursor_col: Option<usize>) {
        // Parse mode from text
        let (mode, display_text) = if let Some(stripped) = text.strip_prefix('!') {
            (InputMode::Bash, stripped)
        } else if let Some(stripped) = text.strip_prefix('/') {
            (InputMode::Command, stripped)
        } else {
            (InputMode::Normal, text)
        };

        self.navigating_history = true;
        self.input_mode = mode;
        self.content = display_text.to_string();
        self.multiline = display_text.contains('\n');

        // Set cursor position
        let first_line = display_text.split('\n').next().unwrap_or("");
        let col = cursor_col.unwrap_or(first_line.chars().count());

        // Convert (0, col) to byte offset
        self.cursor = first_line.chars().take(col).map(|c| c.len_utf8()).sum();
        self.last_cursor_col = col;
        self.cursor_pos_after_load = Some((0, col));
        self.cursor_moved_since_load = false;
    }

    /// Submit the current input and add to history.
    pub fn submit(&mut self) -> String {
        let content = std::mem::take(&mut self.content);
        if let Some(history) = &mut self.history {
            history.add(&content);
        }
        self.cursor = 0;
        self.reset_history_navigation();
        self.input_mode = InputMode::Normal;
        self.multiline = false;
        content
    }

    /// Reset history navigation state.
    fn reset_history_navigation(&mut self) {
        self.reset_prefix();
        self.navigating_history = false;
        self.original_text.clear();
        self.cursor_pos_after_load = None;
        self.cursor_moved_since_load = false;
        self.last_cursor_col = 0;
        if let Some(history) = &mut self.history {
            history.reset_navigation();
        }
    }

    /// Reset history state when text changes.
    fn on_text_changed(&mut self) {
        self.multiline = self.content.contains('\n');
        if !self.navigating_history {
            self.reset_prefix();
            self.original_text.clear();
            self.cursor_pos_after_load = None;
            self.cursor_moved_since_load = false;
            if let Some(history) = &mut self.history {
                history.reset_navigation();
            }
        }
    }

    /// Handle a key event.
    /// Returns true if the input should be submitted.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        // Handle completion-related keys first when completion is visible
        if self.completion.is_visible() {
            match key.code {
                KeyCode::Tab | KeyCode::Enter => {
                    // Apply the selected completion
                    if let Some((new_text, new_cursor)) =
                        self.completion.apply_selected(&self.content)
                    {
                        self.content = new_text;
                        self.cursor = new_cursor;
                        self.update_completion();
                        return false;
                    }
                    // If Tab didn't apply, fall through to normal handling for Enter
                    if key.code == KeyCode::Tab {
                        return false;
                    }
                }
                KeyCode::Up => {
                    self.completion.select_previous();
                    return false;
                }
                KeyCode::Down => {
                    self.completion.select_next();
                    return false;
                }
                KeyCode::Esc => {
                    self.completion.reset();
                    return false;
                }
                _ => {
                    // Other keys fall through to normal handling
                }
            }
        }

        // Handle input mode changes first
        match key.code {
            KeyCode::Char('/') => {
                if self.cursor == 0 && !self.content.starts_with('/') {
                    // Switch to command mode
                    self.input_mode = InputMode::Command;
                    self.insert('/');
                    self.update_completion();
                    return false;
                }
            }
            KeyCode::Char('!') => {
                if self.cursor == 0 && !self.content.starts_with('!') {
                    // Switch to bash mode
                    self.input_mode = InputMode::Bash;
                    self.insert('!');
                    self.update_completion();
                    return false;
                }
            }
            _ => {}
        }

        // Track cursor moves for history navigation
        self.mark_cursor_moved_if_needed();

        match key.code {
            KeyCode::Enter => {
                // Shift+Enter or Ctrl+J for newline
                if key.modifiers.contains(KeyModifiers::SHIFT)
                    || (key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('j'))
                {
                    self.insert_newline();
                    self.update_completion();
                    false
                } else {
                    self.completion.reset();
                    true // Submit
                }
            }
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert_newline();
                self.update_completion();
                false
            }
            KeyCode::Char(c) => {
                self.insert(c);
                self.on_text_changed();
                self.update_completion();
                false
            }
            KeyCode::Backspace => {
                self.delete_backward();
                self.on_text_changed();
                self.update_completion();
                false
            }
            KeyCode::Delete => {
                self.delete_forward();
                self.on_text_changed();
                self.update_completion();
                false
            }
            KeyCode::Left => {
                self.move_cursor_left();
                self.mark_cursor_moved_if_needed();
                self.update_completion();
                false
            }
            KeyCode::Right => {
                self.move_cursor_right();
                self.mark_cursor_moved_if_needed();
                self.update_completion();
                false
            }
            KeyCode::Home => {
                self.move_to_line_start();
                self.mark_cursor_moved_if_needed();
                self.update_completion();
                false
            }
            KeyCode::End => {
                self.move_to_line_end();
                self.mark_cursor_moved_if_needed();
                self.update_completion();
                false
            }
            KeyCode::Tab => {
                // Tab without completion visible - trigger completion update
                self.update_completion();
                false
            }
            KeyCode::Up => {
                // Try history navigation first (only works on first line)
                let was_on_first_line = self.is_cursor_on_first_line();
                let prev_content = self.content.clone();

                self.navigate_history_up();

                // If history didn't change content and we're in multiline, move cursor up
                if self.content == prev_content && !was_on_first_line {
                    self.move_cursor_up();
                }
                self.mark_cursor_moved_if_needed();
                false
            }
            KeyCode::Down => {
                // Try history navigation first (only works on last line)
                let was_on_last_line = self.is_cursor_on_last_line();
                let prev_content = self.content.clone();

                self.navigate_history_down();

                // If history didn't change content and we're in multiline, move cursor down
                if self.content == prev_content && !was_on_last_line {
                    self.move_cursor_down();
                }
                self.mark_cursor_moved_if_needed();
                false
            }
            _ => false,
        }
    }

    /// Update the completion suggestions based on current input.
    fn update_completion(&mut self) {
        self.completion.update(&self.content, self.cursor);
    }

    /// Insert a character at the cursor.
    fn insert(&mut self, c: char) {
        // Ensure cursor is on a valid char boundary before inserting
        self.cursor = self.safe_cursor();
        self.content.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Delete the character before the cursor.
    fn delete_backward(&mut self) {
        if self.cursor > 0 {
            // Ensure cursor is on a valid char boundary
            let safe_cursor = self.safe_cursor();
            // Find the previous character boundary
            let prev = self.content[..safe_cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.content.remove(prev);
            self.cursor = prev;
        }
    }

    /// Delete the character after the cursor.
    fn delete_forward(&mut self) {
        // Ensure cursor is on a valid char boundary
        self.cursor = self.safe_cursor();
        if self.cursor < self.content.len() {
            self.content.remove(self.cursor);
        }
    }

    /// Move cursor left by one character.
    fn move_cursor_left(&mut self) {
        if self.cursor > 0 {
            // Ensure cursor is on a valid char boundary
            let safe_cursor = self.safe_cursor();
            self.cursor = self.content[..safe_cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    /// Move cursor right by one character.
    fn move_cursor_right(&mut self) {
        // Ensure cursor is on a valid char boundary
        self.cursor = self.safe_cursor();
        if self.cursor < self.content.len() {
            self.cursor = self.content[self.cursor..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor + i)
                .unwrap_or(self.content.len());
        }
    }

    /// Move cursor up one line (for multiline input).
    fn move_cursor_up(&mut self) {
        let (row, col) = self.cursor_location();
        if row == 0 {
            return; // Already on first line
        }

        // Get all lines
        let lines: Vec<&str> = self.content.split('\n').collect();

        // Calculate byte offset for the target position
        let target_row = row - 1;
        let target_line = lines.get(target_row).unwrap_or(&"");
        let target_col = col.min(target_line.chars().count());

        // Calculate byte offset
        let mut offset = 0;
        for (i, line) in lines.iter().enumerate() {
            if i == target_row {
                offset += line
                    .chars()
                    .take(target_col)
                    .map(|c| c.len_utf8())
                    .sum::<usize>();
                break;
            }
            offset += line.len() + 1; // +1 for newline
        }

        self.cursor = offset;
    }

    /// Move cursor down one line (for multiline input).
    fn move_cursor_down(&mut self) {
        let (row, col) = self.cursor_location();
        let line_count = self.line_count();
        if row >= line_count.saturating_sub(1) {
            return; // Already on last line
        }

        // Get all lines
        let lines: Vec<&str> = self.content.split('\n').collect();

        // Calculate byte offset for the target position
        let target_row = row + 1;
        let target_line = lines.get(target_row).unwrap_or(&"");
        let target_col = col.min(target_line.chars().count());

        // Calculate byte offset
        let mut offset = 0;
        for (i, line) in lines.iter().enumerate() {
            if i == target_row {
                offset += line
                    .chars()
                    .take(target_col)
                    .map(|c| c.len_utf8())
                    .sum::<usize>();
                break;
            }
            offset += line.len() + 1; // +1 for newline
        }

        self.cursor = offset;
    }

    /// Move cursor to start of current line.
    fn move_to_line_start(&mut self) {
        let (row, _col) = self.cursor_location();
        let lines: Vec<&str> = self.content.split('\n').collect();

        // Calculate byte offset for start of current line
        let mut offset = 0;
        for (i, line) in lines.iter().enumerate() {
            if i == row {
                break;
            }
            offset += line.len() + 1; // +1 for newline
        }

        self.cursor = offset;
    }

    /// Move cursor to end of current line.
    fn move_to_line_end(&mut self) {
        let (row, _col) = self.cursor_location();
        let lines: Vec<&str> = self.content.split('\n').collect();

        // Calculate byte offset for end of current line
        let mut offset = 0;
        for (i, line) in lines.iter().enumerate() {
            if i == row {
                offset += line.len();
                break;
            }
            offset += line.len() + 1; // +1 for newline
        }

        self.cursor = offset;
    }

    /// Insert a newline at current cursor position (Ctrl+J).
    /// This is the preferred way to insert newlines, matching Python's behavior.
    pub fn insert_newline(&mut self) {
        self.insert('\n');
        self.multiline = true; // Enable multiline mode when newline is inserted
        self.on_text_changed();
    }

    /// Check if multiline mode is active.
    pub fn is_multiline(&self) -> bool {
        self.multiline
    }
}
