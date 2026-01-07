//! Persistent command history management.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// Manages persistent command history.
#[derive(Debug)]
pub struct HistoryManager {
    /// Path to the history file.
    history_file: PathBuf,
    /// Maximum number of entries to keep.
    max_entries: usize,
    /// In-memory history entries.
    entries: Vec<String>,
    /// Current navigation index (-1 = not navigating).
    current_index: i32,
    /// Temporary input saved when navigating.
    temp_input: String,
}

impl HistoryManager {
    /// Create a new history manager.
    pub fn new(history_file: PathBuf, max_entries: usize) -> Self {
        let mut manager = Self {
            history_file,
            max_entries,
            entries: Vec::new(),
            current_index: -1,
            temp_input: String::new(),
        };
        manager.load_history();
        manager
    }

    /// Load history from file.
    fn load_history(&mut self) {
        if !self.history_file.exists() {
            return;
        }

        let file = match File::open(&self.history_file) {
            Ok(f) => f,
            Err(_) => return,
        };

        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l.trim().to_string(),
                Err(_) => continue,
            };

            if line.is_empty() {
                continue;
            }

            // Try to parse as JSON string, fall back to raw line
            let entry = if line.starts_with('"') {
                serde_json::from_str::<String>(&line).unwrap_or(line)
            } else {
                line
            };

            if !entry.is_empty() {
                entries.push(entry);
            }
        }

        // Keep only the last max_entries
        if entries.len() > self.max_entries {
            entries = entries.split_off(entries.len() - self.max_entries);
        }

        self.entries = entries;
    }

    /// Save history to file.
    fn save_history(&self) {
        // Ensure parent directory exists
        if let Some(parent) = self.history_file.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let file = match File::create(&self.history_file) {
            Ok(f) => f,
            Err(_) => return,
        };

        let mut writer = std::io::BufWriter::new(file);
        for entry in &self.entries {
            if let Ok(json) = serde_json::to_string(entry) {
                let _ = writeln!(writer, "{json}");
            }
        }
    }

    /// Add an entry to history.
    pub fn add(&mut self, text: &str) {
        let text = text.trim();

        // Skip empty entries and commands
        if text.is_empty() || text.starts_with('/') {
            return;
        }

        // Skip duplicates of the last entry
        if self.entries.last().map(|s| s.as_str()) == Some(text) {
            return;
        }

        self.entries.push(text.to_string());

        // Trim to max entries
        if self.entries.len() > self.max_entries {
            self.entries = self
                .entries
                .split_off(self.entries.len() - self.max_entries);
        }

        self.save_history();
        self.reset_navigation();
    }

    /// Get the previous history entry.
    ///
    /// Returns the previous entry that starts with the given prefix, if any.
    pub fn get_previous(&mut self, current_input: &str, prefix: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }

        // Start navigation if not already
        if self.current_index == -1 {
            self.temp_input = current_input.to_string();
            self.current_index = self.entries.len() as i32;
        }

        // Search backwards for matching entry
        for i in (0..self.current_index as usize).rev() {
            if self.entries[i].starts_with(prefix) {
                self.current_index = i as i32;
                return Some(self.entries[i].clone());
            }
        }

        None
    }

    /// Get the next history entry.
    ///
    /// Returns the next entry that starts with the given prefix, if any.
    pub fn get_next(&mut self, prefix: &str) -> Option<String> {
        if self.current_index == -1 {
            return None;
        }

        // Search forwards for matching entry
        for i in (self.current_index as usize + 1)..self.entries.len() {
            if self.entries[i].starts_with(prefix) {
                self.current_index = i as i32;
                return Some(self.entries[i].clone());
            }
        }

        // No more entries, return temp input
        let result = std::mem::take(&mut self.temp_input);
        self.reset_navigation();
        Some(result)
    }

    /// Reset navigation state.
    pub fn reset_navigation(&mut self) {
        self.current_index = -1;
        self.temp_input.clear();
    }

    /// Check if currently navigating history.
    pub fn is_navigating(&self) -> bool {
        self.current_index != -1
    }

    /// Get the number of entries.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if history is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_add_and_navigate() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("history");
        let mut manager = HistoryManager::new(path, 100);

        manager.add("first");
        manager.add("second");
        manager.add("third");

        // Navigate backwards
        assert_eq!(manager.get_previous("", ""), Some("third".to_string()));
        assert_eq!(manager.get_previous("", ""), Some("second".to_string()));
        assert_eq!(manager.get_previous("", ""), Some("first".to_string()));
        assert_eq!(manager.get_previous("", ""), None);

        // Navigate forwards
        assert_eq!(manager.get_next(""), Some("second".to_string()));
        assert_eq!(manager.get_next(""), Some("third".to_string()));
        assert_eq!(manager.get_next(""), Some("".to_string())); // temp input

        manager.reset_navigation();
    }

    #[test]
    fn test_prefix_filtering() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("history");
        let mut manager = HistoryManager::new(path, 100);

        manager.add("hello world");
        manager.add("help me");
        manager.add("goodbye");

        // Navigate with prefix
        assert_eq!(manager.get_previous("", "hel"), Some("help me".to_string()));
        assert_eq!(
            manager.get_previous("", "hel"),
            Some("hello world".to_string())
        );
        assert_eq!(manager.get_previous("", "hel"), None);
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("history");

        // Create and populate
        {
            let mut manager = HistoryManager::new(path.clone(), 100);
            manager.add("entry1");
            manager.add("entry2");
        }

        // Reload and verify
        {
            let mut manager = HistoryManager::new(path, 100);
            assert_eq!(manager.get_previous("", ""), Some("entry2".to_string()));
            assert_eq!(manager.get_previous("", ""), Some("entry1".to_string()));
        }
    }

    #[test]
    fn test_skip_duplicates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("history");
        let mut manager = HistoryManager::new(path, 100);

        manager.add("same");
        manager.add("same");
        manager.add("same");

        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_skip_commands() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("history");
        let mut manager = HistoryManager::new(path, 100);

        manager.add("/quit");
        manager.add("/help");
        manager.add("normal input");

        assert_eq!(manager.len(), 1);
    }
}
