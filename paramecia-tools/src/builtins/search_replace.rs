//! Search and replace tool using SEARCH/REPLACE blocks.

use async_trait::async_trait;

/// Tool prompt loaded from markdown file.
const SEARCH_REPLACE_PROMPT: &str = include_str!("prompts/search_replace.md");
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use tokio::fs;

use crate::error::{ToolError, ToolResult};
use crate::types::{Tool, ToolConfig, ToolPermission};

/// A parsed SEARCH/REPLACE block.
#[derive(Debug, Clone)]
struct SearchReplaceBlock {
    search: String,
    replace: String,
}

/// Result of applying blocks to content.
struct BlockApplyResult {
    content: String,
    applied: usize,
    errors: Vec<String>,
    warnings: Vec<String>,
}

/// Arguments for the search_replace tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchReplaceArgs {
    /// Path to the file to modify.
    pub file_path: String,
    /// Content containing SEARCH/REPLACE blocks.
    pub content: String,
}

/// Result from search and replace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchReplaceResult {
    /// Path that was modified.
    pub file: String,
    /// Number of blocks applied.
    pub blocks_applied: usize,
    /// Net lines changed (positive = added, negative = removed).
    pub lines_changed: i32,
    /// The original content with blocks.
    pub content: String,
    /// Any warnings encountered.
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Search and replace tool.
pub struct SearchReplace {
    config: ToolConfig,
}

impl Default for SearchReplace {
    fn default() -> Self {
        let mut config = ToolConfig {
            permission: ToolPermission::Ask,
            ..Default::default()
        };

        // Max content size
        config
            .extra
            .insert("max_content_size".to_string(), json!(100_000));

        // Fuzzy matching threshold
        config
            .extra
            .insert("fuzzy_threshold".to_string(), json!(0.9));

        Self { config }
    }
}

impl SearchReplace {
    /// Get max content size.
    fn max_content_size(&self) -> usize {
        self.config.get_or("max_content_size", 100_000)
    }

    /// Get fuzzy matching threshold.
    fn fuzzy_threshold(&self) -> f64 {
        self.config.get_or("fuzzy_threshold", 0.9)
    }

    /// Resolve a path relative to the working directory.
    fn resolve_path(&self, path: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path);
        let resolved = if path.is_absolute() {
            path
        } else {
            self.config.effective_workdir().join(path)
        };

        resolved
            .canonicalize()
            .map_err(|e| ToolError::FileError(format!("Cannot resolve path: {e}")))
    }

    /// Parse SEARCH/REPLACE blocks from content.
    fn parse_blocks(content: &str) -> Vec<SearchReplaceBlock> {
        // Pattern for blocks with optional code fences
        let block_with_fence = Regex::new(
            r"(?s)```[\s\S]*?\n<{5,} SEARCH\r?\n(.*?)\r?\n?={5,}\r?\n(.*?)\r?\n?>{5,} REPLACE\s*\n```"
        ).unwrap();

        // Pattern for blocks without fences
        let block_plain =
            Regex::new(r"(?s)<{5,} SEARCH\r?\n(.*?)\r?\n?={5,}\r?\n(.*?)\r?\n?>{5,} REPLACE")
                .unwrap();

        // Try fenced blocks first
        let mut blocks: Vec<SearchReplaceBlock> = block_with_fence
            .captures_iter(content)
            .map(|cap| SearchReplaceBlock {
                search: cap[1].trim_end_matches(['\r', '\n']).to_string(),
                replace: cap[2].trim_end_matches(['\r', '\n']).to_string(),
            })
            .collect();

        // If no fenced blocks found, try plain blocks
        if blocks.is_empty() {
            blocks = block_plain
                .captures_iter(content)
                .map(|cap| SearchReplaceBlock {
                    search: cap[1].trim_end_matches(['\r', '\n']).to_string(),
                    replace: cap[2].trim_end_matches(['\r', '\n']).to_string(),
                })
                .collect();
        }

        blocks
    }

    /// Apply blocks to content.
    fn apply_blocks(
        content: &str,
        blocks: &[SearchReplaceBlock],
        filepath: &Path,
        fuzzy_threshold: f64,
    ) -> BlockApplyResult {
        let mut current_content = content.to_string();
        let mut applied = 0;
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for (i, block) in blocks.iter().enumerate() {
            let block_num = i + 1;

            if !current_content.contains(&block.search) {
                // Try to find a fuzzy match and provide helpful error
                let context = Self::find_search_context(&current_content, &block.search);
                let fuzzy_context = Self::find_fuzzy_match_context(
                    &current_content,
                    &block.search,
                    fuzzy_threshold,
                );

                let mut error_msg = format!(
                    "SEARCH/REPLACE block {} failed: Search text not found in {}\n\
                     Search text was:\n{:?}\n\
                     Context analysis:\n{}",
                    block_num,
                    filepath.display(),
                    block.search,
                    context
                );

                if let Some(fuzzy) = fuzzy_context {
                    error_msg.push_str(&format!("\n{}", fuzzy));
                }

                error_msg.push_str(
                    "\nDebugging tips:\n\
                     1. Check for exact whitespace/indentation match\n\
                     2. Verify line endings match the file exactly (\\r\\n vs \\n)\n\
                     3. Ensure the search text hasn't been modified by previous blocks or user edits\n\
                     4. Check for typos or case sensitivity issues"
                );

                errors.push(error_msg);
                continue;
            }

            // Check for multiple occurrences
            let occurrences = current_content.matches(&block.search).count();
            if occurrences > 1 {
                warnings.push(format!(
                    "Search text in block {} appears {} times in the file. \
                     Only the first occurrence will be replaced. Consider making your \
                     search pattern more specific to avoid unintended changes.",
                    block_num, occurrences
                ));
            }

            // Apply the replacement (only first occurrence)
            current_content = current_content.replacen(&block.search, &block.replace, 1);
            applied += 1;
        }

        BlockApplyResult {
            content: current_content,
            applied,
            errors,
            warnings,
        }
    }

    /// Find context around potential matches.
    fn find_search_context(content: &str, search_text: &str) -> String {
        let lines: Vec<&str> = content.lines().collect();
        let search_lines: Vec<&str> = search_text.lines().collect();

        if search_lines.is_empty() {
            return "Search text is empty".to_string();
        }

        let first_search_line = search_lines[0].trim();
        if first_search_line.is_empty() {
            return "First line of search text is empty or whitespace only".to_string();
        }

        let mut matches = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line.contains(first_search_line) {
                matches.push(i);
            }
        }

        if matches.is_empty() {
            return format!(
                "First search line '{}' not found anywhere in file",
                first_search_line
            );
        }

        let mut context_lines = Vec::new();
        let max_context = 5;

        for &match_idx in matches.iter().take(3) {
            let start = match_idx.saturating_sub(max_context);
            let end = (match_idx + max_context + 1).min(lines.len());

            context_lines.push(format!(
                "\nPotential match area around line {}:",
                match_idx + 1
            ));
            for (i, line) in lines.iter().enumerate().take(end).skip(start) {
                let marker = if i == match_idx { ">>>" } else { "   " };
                context_lines.push(format!("{} {:3}: {}", marker, i + 1, line));
            }
        }

        context_lines.join("\n")
    }

    /// Find fuzzy match and create context.
    fn find_fuzzy_match_context(
        content: &str,
        search_text: &str,
        threshold: f64,
    ) -> Option<String> {
        let content_lines: Vec<&str> = content.lines().collect();
        let search_lines: Vec<&str> = search_text.lines().collect();
        let window_size = search_lines.len();

        if window_size == 0 || content_lines.len() < window_size {
            return None;
        }

        let mut best_similarity = 0.0;
        let mut best_start = 0;
        let mut best_text = String::new();

        // Slide a window over content
        for start in 0..=(content_lines.len() - window_size) {
            let window_text: String = content_lines[start..start + window_size].join("\n");
            let similarity = Self::calculate_similarity(search_text, &window_text);

            if similarity >= threshold && similarity > best_similarity {
                best_similarity = similarity;
                best_start = start;
                best_text = window_text;
            }
        }

        if best_similarity >= threshold {
            Some(format!(
                "Closest fuzzy match (similarity {:.1}%) at lines {}–{}:\n\
                 Diff between SEARCH and CLOSEST MATCH:\n{}",
                best_similarity * 100.0,
                best_start + 1,
                best_start + window_size,
                Self::create_simple_diff(search_text, &best_text)
            ))
        } else {
            None
        }
    }

    /// Calculate similarity between two strings (simple ratio).
    fn calculate_similarity(a: &str, b: &str) -> f64 {
        if a == b {
            return 1.0;
        }

        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();

        if a_chars.is_empty() && b_chars.is_empty() {
            return 1.0;
        }

        if a_chars.is_empty() || b_chars.is_empty() {
            return 0.0;
        }

        // Simple LCS-based similarity
        let lcs_len = Self::lcs_length(&a_chars, &b_chars);
        let _max_len = a_chars.len().max(b_chars.len());

        (2.0 * lcs_len as f64) / (a_chars.len() + b_chars.len()) as f64
    }

    /// Calculate longest common subsequence length.
    fn lcs_length(a: &[char], b: &[char]) -> usize {
        let m = a.len();
        let n = b.len();
        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }

    /// Create a simple diff output.
    fn create_simple_diff(expected: &str, actual: &str) -> String {
        let expected_lines: Vec<&str> = expected.lines().collect();
        let actual_lines: Vec<&str> = actual.lines().collect();

        let mut diff = Vec::new();
        let max_lines = expected_lines.len().max(actual_lines.len());

        for i in 0..max_lines {
            let exp = expected_lines.get(i).copied();
            let act = actual_lines.get(i).copied();

            match (exp, act) {
                (Some(e), Some(a)) if e == a => {
                    diff.push(format!("  {}", e));
                }
                (Some(e), Some(a)) => {
                    diff.push(format!("- {}", e));
                    diff.push(format!("+ {}", a));
                }
                (Some(e), None) => {
                    diff.push(format!("- {}", e));
                }
                (None, Some(a)) => {
                    diff.push(format!("+ {}", a));
                }
                (None, None) => {}
            }
        }

        let result = diff.join("\n");
        if result.len() > 2000 {
            format!("{}...(diff truncated)", &result[..2000])
        } else {
            result
        }
    }
}

#[async_trait]
impl Tool for SearchReplace {
    fn name(&self) -> &str {
        "search_replace"
    }

    fn description(&self) -> &str {
        "Replace sections of files using SEARCH/REPLACE blocks. \
         Supports fuzzy matching and detailed error reporting. \
         Format: <<<<<<< SEARCH\\n[text]\\n=======\\n[replacement]\\n>>>>>>> REPLACE"
    }

    fn prompt(&self) -> Option<&str> {
        Some(SEARCH_REPLACE_PROMPT)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to modify"
                },
                "content": {
                    "type": "string",
                    "description": "Content containing SEARCH/REPLACE blocks"
                }
            },
            "required": ["file_path", "content"]
        })
    }

    fn config(&self) -> &ToolConfig {
        &self.config
    }

    fn config_mut(&mut self) -> &mut ToolConfig {
        &mut self.config
    }

    async fn execute(&mut self, args: serde_json::Value) -> ToolResult<serde_json::Value> {
        let args: SearchReplaceArgs =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;

        let file_path_str = args.file_path.trim();
        let content = args.content.trim();

        if file_path_str.is_empty() {
            return Err(ToolError::InvalidArguments(
                "File path cannot be empty".to_string(),
            ));
        }

        if content.len() > self.max_content_size() {
            return Err(ToolError::InvalidArguments(format!(
                "Content size ({} bytes) exceeds max_content_size ({} bytes)",
                content.len(),
                self.max_content_size()
            )));
        }

        if content.is_empty() {
            return Err(ToolError::InvalidArguments(
                "Empty content provided".to_string(),
            ));
        }

        let file_path = self.resolve_path(file_path_str)?;

        if !file_path.exists() {
            return Err(ToolError::FileError(format!(
                "File does not exist: {}",
                file_path.display()
            )));
        }

        if !file_path.is_file() {
            return Err(ToolError::FileError(format!(
                "Path is not a file: {}",
                file_path.display()
            )));
        }

        // Parse blocks
        let blocks = Self::parse_blocks(content);
        if blocks.is_empty() {
            return Err(ToolError::InvalidArguments(
                "No valid SEARCH/REPLACE blocks found in content.\n\
                 Expected format:\n\
                 <<<<<<< SEARCH\n\
                 [exact content to find]\n\
                 =======\n\
                 [new content to replace with]\n\
                 >>>>>>> REPLACE"
                    .to_string(),
            ));
        }

        // Read original content
        let original_content = fs::read_to_string(&file_path).await.map_err(|e| {
            ToolError::FileError(format!("Error reading {}: {e}", file_path.display()))
        })?;

        // Apply blocks
        let result = Self::apply_blocks(
            &original_content,
            &blocks,
            &file_path,
            self.fuzzy_threshold(),
        );

        if !result.errors.is_empty() {
            let mut error_message = "SEARCH/REPLACE blocks failed:\n".to_string();
            error_message.push_str(&result.errors.join("\n\n"));
            if !result.warnings.is_empty() {
                error_message.push_str("\n\nWarnings encountered:\n");
                error_message.push_str(&result.warnings.join("\n"));
            }
            return Err(ToolError::ExecutionFailed(error_message));
        }

        // Calculate line changes
        let lines_changed = if result.content == original_content {
            0
        } else {
            let original_lines = original_content.lines().count() as i32;
            let new_lines = result.content.lines().count() as i32;

            // Write the modified content
            fs::write(&file_path, &result.content).await.map_err(|e| {
                ToolError::FileError(format!("Error writing {}: {e}", file_path.display()))
            })?;

            new_lines - original_lines
        };

        let output = SearchReplaceResult {
            file: file_path.to_string_lossy().to_string(),
            blocks_applied: result.applied,
            lines_changed,
            content: args.content,
            warnings: result.warnings,
        };

        Ok(serde_json::to_value(output)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_search_replace_default_config() {
        let tool = SearchReplace::default();
        assert_eq!(tool.name(), "search_replace");
        assert_eq!(tool.config().permission, ToolPermission::Ask);
    }

    #[test]
    fn test_parse_blocks() {
        let content = r#"Some text before

<<<<<<< SEARCH
old content
=======
new content
>>>>>>> REPLACE

Some text after"#;

        let blocks = SearchReplace::parse_blocks(content);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].search, "old content");
        assert_eq!(blocks[0].replace, "new content");
    }

    #[test]
    fn test_parse_multiple_blocks() {
        let content = r#"
<<<<<<< SEARCH
first old
=======
first new
>>>>>>> REPLACE

<<<<<<< SEARCH
second old
=======
second new
>>>>>>> REPLACE
"#;

        let blocks = SearchReplace::parse_blocks(content);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].search, "first old");
        assert_eq!(blocks[1].search, "second old");
    }

    #[tokio::test]
    async fn test_search_replace_execute() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Hello, World!").unwrap();
        writeln!(file, "This is a test.").unwrap();

        let mut tool = SearchReplace::default();
        let result = tool
            .execute(json!({
                "file_path": file.path().to_str().unwrap(),
                "content": r#"
<<<<<<< SEARCH
Hello, World!
=======
Goodbye, World!
>>>>>>> REPLACE
"#
            }))
            .await;

        assert!(result.is_ok());

        // Verify content was replaced
        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("Goodbye, World!"));
        assert!(!content.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_search_replace_not_found() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Hello, World!").unwrap();

        let mut tool = SearchReplace::default();
        let result = tool
            .execute(json!({
                "file_path": file.path().to_str().unwrap(),
                "content": r#"
<<<<<<< SEARCH
Not Found Content
=======
Replacement
>>>>>>> REPLACE
"#
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_replace_no_blocks() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Hello, World!").unwrap();

        let mut tool = SearchReplace::default();
        let result = tool
            .execute(json!({
                "file_path": file.path().to_str().unwrap(),
                "content": "Just some plain text without blocks"
            }))
            .await;

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("No valid SEARCH/REPLACE blocks"));
        }
    }
}
