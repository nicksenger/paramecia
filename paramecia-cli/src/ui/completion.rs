//! Completion system for path and command autocompletion.

use std::path::PathBuf;

/// Maximum number of suggestions to display.
const MAX_SUGGESTIONS: usize = 10;

/// Result of a completion suggestion.
#[derive(Debug, Clone)]
pub struct CompletionSuggestion {
    /// The text to insert.
    pub completion: String,
    /// Description/hint for this suggestion.
    pub description: String,
}

/// Completion state and logic.
#[derive(Debug, Default)]
pub struct CompletionManager {
    /// Current suggestions.
    suggestions: Vec<CompletionSuggestion>,
    /// Currently selected index.
    selected_index: usize,
    /// Whether completions are visible.
    visible: bool,
    /// Start position of the completion range.
    range_start: usize,
    /// End position of the completion range.
    range_end: usize,
    /// Available commands for completion.
    commands: Vec<(String, String)>,
}

impl CompletionManager {
    /// Create a new completion manager with available commands.
    #[must_use]
    pub fn new(commands: Vec<(String, String)>) -> Self {
        Self {
            suggestions: Vec::new(),
            selected_index: 0,
            visible: false,
            range_start: 0,
            range_end: 0,
            commands,
        }
    }

    /// Check if completions are visible.
    #[must_use]
    pub fn is_visible(&self) -> bool {
        self.visible && !self.suggestions.is_empty()
    }

    /// Get current suggestions.
    #[must_use]
    pub fn suggestions(&self) -> &[CompletionSuggestion] {
        &self.suggestions
    }

    /// Get selected index.
    #[must_use]
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    /// Reset/hide completions.
    pub fn reset(&mut self) {
        self.suggestions.clear();
        self.selected_index = 0;
        self.visible = false;
    }

    /// Update completions based on input text and cursor position.
    pub fn update(&mut self, text: &str, cursor: usize) {
        // Check for path completion (@path)
        if let Some((suggestions, start, end)) = self.get_path_completions(text, cursor) {
            self.suggestions = suggestions;
            self.range_start = start;
            self.range_end = end;
            self.selected_index = 0;
            self.visible = !self.suggestions.is_empty();
            return;
        }

        // Check for command completion (/command)
        if let Some((suggestions, start, end)) = self.get_command_completions(text, cursor) {
            self.suggestions = suggestions;
            self.range_start = start;
            self.range_end = end;
            self.selected_index = 0;
            self.visible = !self.suggestions.is_empty();
            return;
        }

        // No completion context
        self.reset();
    }

    /// Get path completions for @path syntax.
    fn get_path_completions(
        &self,
        text: &str,
        cursor: usize,
    ) -> Option<(Vec<CompletionSuggestion>, usize, usize)> {
        // Ensure we slice at a valid UTF-8 char boundary
        let safe_cursor = {
            let mut pos = cursor.min(text.len());
            while pos > 0 && !text.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        };
        let before_cursor = &text[..safe_cursor];

        // Find the @ symbol before cursor
        let at_pos = before_cursor.rfind('@')?;

        // Check if there's a space between @ and cursor (no completion)
        let fragment = &before_cursor[at_pos..];
        if fragment.contains(' ') {
            return None;
        }

        // Get the path fragment after @
        let path_fragment = &fragment[1..]; // Skip the @

        // Get completions
        let suggestions = self.complete_path(path_fragment);
        if suggestions.is_empty() {
            return None;
        }

        Some((suggestions, at_pos, cursor))
    }

    /// Complete a path fragment.
    fn complete_path(&self, fragment: &str) -> Vec<CompletionSuggestion> {
        let mut suggestions = Vec::new();

        // Determine base directory and partial name
        let (base_dir, partial) = if fragment.contains('/') {
            let last_slash = fragment.rfind('/').unwrap();
            let base = &fragment[..=last_slash];
            let partial = &fragment[last_slash + 1..];

            // Resolve the base directory
            let path = if base.starts_with('/') {
                PathBuf::from(base)
            } else if let Some(stripped_base) = base.strip_prefix("~/") {
                if let Some(home) = dirs::home_dir() {
                    home.join(stripped_base)
                } else {
                    PathBuf::from(base)
                }
            } else {
                match std::env::current_dir() {
                    Ok(cwd) => cwd.join(base),
                    Err(_) => return suggestions,
                }
            };
            (path, partial)
        } else {
            // No slash - complete from current directory
            let base = match std::env::current_dir() {
                Ok(cwd) => cwd,
                Err(_) => return suggestions,
            };
            (base, fragment)
        };

        // Read directory entries
        if let Ok(entries) = std::fs::read_dir(&base_dir) {
            for entry in entries.filter_map(Result::ok) {
                let name = entry.file_name().to_string_lossy().to_string();

                // Skip hidden files unless explicitly requested
                if name.starts_with('.') && !partial.starts_with('.') {
                    continue;
                }

                // Check if name starts with partial
                if !name.to_lowercase().starts_with(&partial.to_lowercase()) {
                    continue;
                }

                let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);

                // Build the completion text
                let completion = if fragment.contains('/') {
                    let prefix = &fragment[..fragment.rfind('/').unwrap() + 1];
                    if is_dir {
                        format!("@{prefix}{name}/")
                    } else {
                        format!("@{prefix}{name}")
                    }
                } else if is_dir {
                    format!("@{name}/")
                } else {
                    format!("@{name}")
                };

                let description = if is_dir { "directory" } else { "file" }.to_string();

                suggestions.push(CompletionSuggestion {
                    completion,
                    description,
                });
            }
        }

        // Sort and limit
        suggestions.sort_by(|a, b| a.completion.cmp(&b.completion));
        suggestions.truncate(MAX_SUGGESTIONS);

        suggestions
    }

    /// Get command completions for /command syntax.
    fn get_command_completions(
        &self,
        text: &str,
        cursor: usize,
    ) -> Option<(Vec<CompletionSuggestion>, usize, usize)> {
        // Ensure we slice at a valid UTF-8 char boundary
        let safe_cursor = {
            let mut pos = cursor.min(text.len());
            while pos > 0 && !text.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        };
        let before_cursor = &text[..safe_cursor];

        // Must start with /
        if !before_cursor.starts_with('/') {
            return None;
        }

        // Check if there's a space (command already complete)
        if before_cursor.contains(' ') {
            return None;
        }

        let fragment = &before_cursor[1..]; // Skip the /

        let mut suggestions = Vec::new();
        for (cmd, desc) in &self.commands {
            // Skip the leading / in cmd for matching
            let cmd_name = cmd.trim_start_matches('/');
            if cmd_name
                .to_lowercase()
                .starts_with(&fragment.to_lowercase())
            {
                suggestions.push(CompletionSuggestion {
                    completion: format!("/{cmd_name}"),
                    description: desc.clone(),
                });
            }
        }

        if suggestions.is_empty() {
            return None;
        }

        suggestions.truncate(MAX_SUGGESTIONS);
        Some((suggestions, 0, cursor))
    }

    /// Move selection up.
    pub fn select_previous(&mut self) {
        if !self.suggestions.is_empty() {
            if self.selected_index == 0 {
                self.selected_index = self.suggestions.len() - 1;
            } else {
                self.selected_index -= 1;
            }
        }
    }

    /// Move selection down.
    pub fn select_next(&mut self) {
        if !self.suggestions.is_empty() {
            self.selected_index = (self.selected_index + 1) % self.suggestions.len();
        }
    }

    /// Apply the selected completion.
    /// Returns the new text and cursor position.
    pub fn apply_selected(&mut self, text: &str) -> Option<(String, usize)> {
        if self.suggestions.is_empty() {
            return None;
        }

        let suggestion = &self.suggestions[self.selected_index];
        let completion = &suggestion.completion;

        // Ensure range positions are on valid UTF-8 char boundaries
        let safe_start = {
            let mut pos = self.range_start.min(text.len());
            while pos > 0 && !text.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        };
        let safe_end = {
            let mut pos = self.range_end.min(text.len());
            while pos > 0 && !text.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        };

        // Build new text: prefix + completion + suffix
        let prefix = &text[..safe_start];
        let suffix = &text[safe_end..];

        // Add space after completion (unless it ends with / for directories)
        let insertion = if completion.ends_with('/') {
            completion.clone()
        } else {
            format!("{completion} ")
        };

        let new_text = format!("{prefix}{insertion}{suffix}");
        let new_cursor = self.range_start + insertion.len();

        self.reset();
        Some((new_text, new_cursor))
    }
}
