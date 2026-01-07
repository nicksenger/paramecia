//! Project context provider for building system prompts.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use crate::config::ProjectContextConfig;
use crate::prompts::UtilityPrompt;

/// Default patterns to ignore when scanning directories.
const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    ".git",
    ".git/*",
    "*.pyc",
    "__pycache__",
    "node_modules",
    "node_modules/*",
    ".env",
    ".DS_Store",
    "*.log",
    ".vscode/settings.json",
    ".idea/*",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
    "coverage",
    ".nyc_output",
    "*.egg-info",
    ".pytest_cache",
    ".tox",
    "vendor",
    "third_party",
    "deps",
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.chunk.js",
    ".cache",
    "tmp",
    "temp",
    "logs",
];

/// Project documentation filenames to look for.
const PROJECT_DOC_FILENAMES: &[&str] = &["AGENTS.md", "VIBE.md", ".vibe.md"];

/// Provides project context information for the system prompt.
pub struct ProjectContextProvider {
    root_path: PathBuf,
    config: ProjectContextConfig,
    gitignore_patterns: Vec<String>,
    file_count: usize,
    start_time: Option<Instant>,
}

impl ProjectContextProvider {
    /// Create a new project context provider.
    pub fn new(config: ProjectContextConfig, root_path: impl AsRef<Path>) -> Self {
        let root = root_path.as_ref().to_path_buf();
        let gitignore_patterns = Self::load_gitignore_patterns(&root);

        Self {
            root_path: root,
            config,
            gitignore_patterns,
            file_count: 0,
            start_time: None,
        }
    }

    /// Load gitignore patterns from the project.
    fn load_gitignore_patterns(root: &Path) -> Vec<String> {
        let mut patterns: Vec<String> = DEFAULT_IGNORE_PATTERNS
            .iter()
            .map(|s| s.to_string())
            .collect();

        let gitignore_path = root.join(".gitignore");
        if gitignore_path.exists()
            && let Ok(content) = std::fs::read_to_string(&gitignore_path)
        {
            for line in content.lines() {
                let line = line.trim();
                if !line.is_empty() && !line.starts_with('#') {
                    patterns.push(line.to_string());
                }
            }
        }

        patterns
    }

    /// Check if a path should be ignored.
    fn is_ignored(&self, path: &Path) -> bool {
        let relative = match path.strip_prefix(&self.root_path) {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => return true,
        };

        for pattern in &self.gitignore_patterns {
            if pattern.ends_with('/') {
                if path.is_dir() && Self::glob_match(pattern.trim_end_matches('/'), &relative) {
                    return true;
                }
            } else if Self::glob_match(pattern, &relative) {
                return true;
            }
        }

        false
    }

    /// Simple glob matching.
    fn glob_match(pattern: &str, text: &str) -> bool {
        if pattern == text {
            return true;
        }

        // Handle simple wildcards
        if pattern.contains('*') {
            if pattern.starts_with('*') {
                let suffix = pattern.trim_start_matches('*');
                return text.ends_with(suffix);
            }
            if pattern.ends_with('*') {
                let prefix = pattern.trim_end_matches('*');
                return text.starts_with(prefix);
            }
        }

        // Check if pattern matches any component
        text.split(std::path::MAIN_SEPARATOR)
            .any(|component| component == pattern)
    }

    /// Check if we should stop scanning.
    fn should_stop(&self) -> bool {
        self.file_count >= self.config.max_files
            || self.start_time.is_some_and(|start| {
                start.elapsed() > Duration::from_secs_f64(self.config.timeout_seconds)
            })
    }

    /// Build the directory tree structure.
    pub fn get_directory_structure(&mut self) -> String {
        self.start_time = Some(Instant::now());
        self.file_count = 0;

        let mut lines = Vec::new();
        let header = format!(
            "Directory structure of {} (depth≤{}, max {} items):\n",
            self.root_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            self.config.max_depth,
            self.config.max_files
        );

        self.process_directory(&self.root_path.clone(), "", 0, &mut lines);

        let mut structure = header + &lines.join("\n");

        if self.file_count >= self.config.max_files {
            structure.push_str(&format!(
                "\n... (truncated at {} files limit)",
                self.config.max_files
            ));
        } else if self.start_time.is_some_and(|start| {
            start.elapsed() > Duration::from_secs_f64(self.config.timeout_seconds)
        }) {
            structure.push_str(&format!(
                "\n... (truncated due to {}s timeout)",
                self.config.timeout_seconds
            ));
        }

        structure
    }

    /// Process a directory and add its contents to the tree.
    fn process_directory(
        &mut self,
        path: &Path,
        prefix: &str,
        depth: usize,
        lines: &mut Vec<String>,
    ) {
        if depth > self.config.max_depth || self.should_stop() {
            return;
        }

        let entries: Vec<_> = match std::fs::read_dir(path) {
            Ok(entries) => entries
                .filter_map(|e| e.ok())
                .filter(|e| !self.is_ignored(&e.path()))
                .collect(),
            Err(_) => return,
        };

        let mut entries: Vec<_> = entries
            .into_iter()
            .map(|e| (e.path(), e.file_type().is_ok_and(|t| t.is_dir())))
            .collect();

        // Sort: directories first, then by name
        entries.sort_by(|a, b| match (a.1, b.1) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.0.file_name().cmp(&b.0.file_name()),
        });

        let max_per_level = self.config.max_dirs_per_level;
        let show_truncation = entries.len() > max_per_level;
        let entries: Vec<_> = entries.into_iter().take(max_per_level).collect();

        for (i, (entry_path, is_dir)) in entries.iter().enumerate() {
            if self.should_stop() {
                break;
            }

            let is_last = i == entries.len() - 1 && !show_truncation;
            let connector = if is_last { "└── " } else { "├── " };
            let name = entry_path.file_name().unwrap_or_default().to_string_lossy();
            let suffix = if *is_dir { "/" } else { "" };

            lines.push(format!("{}{}{}{}", prefix, connector, name, suffix));
            self.file_count += 1;

            if *is_dir && depth < self.config.max_depth {
                let child_prefix = format!("{}{}   ", prefix, if is_last { " " } else { "│" });
                self.process_directory(entry_path, &child_prefix, depth + 1, lines);
            }
        }

        if show_truncation && !self.should_stop() {
            lines.push(format!("{}└── ... (more items)", prefix));
        }
    }

    /// Get git status information.
    pub fn get_git_status(&self) -> String {
        let _timeout = Duration::from_secs_f64(self.config.timeout_seconds.min(10.0));

        // Get current branch
        let current_branch = match Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(&self.root_path)
            .output()
        {
            Ok(output) if output.status.success() => {
                String::from_utf8_lossy(&output.stdout).trim().to_string()
            }
            _ => return "Not a git repository or git not available".to_string(),
        };

        // Detect main branch
        let main_branch = match Command::new("git")
            .args(["branch", "-r"])
            .current_dir(&self.root_path)
            .output()
        {
            Ok(output) if output.status.success() => {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if output_str.contains("origin/master") {
                    "master"
                } else {
                    "main"
                }
            }
            _ => "main",
        };

        // Get status
        let status = match Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&self.root_path)
            .output()
        {
            Ok(output) if output.status.success() => {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let changes: Vec<_> = output_str.lines().collect();
                if changes.is_empty() {
                    "(clean)".to_string()
                } else if changes.len() > 50 {
                    format!("({} changes - use 'git status' for details)", changes.len())
                } else {
                    format!("({} changes)", changes.len())
                }
            }
            _ => "(unknown)".to_string(),
        };

        // Get recent commits
        let num_commits = self.config.default_commit_count;
        let recent_commits = match Command::new("git")
            .args([
                "log",
                "--oneline",
                &format!("-{}", num_commits),
                "--decorate",
            ])
            .current_dir(&self.root_path)
            .output()
        {
            Ok(output) if output.status.success() => {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .filter_map(|line| {
                        let line = line.trim();
                        if line.is_empty() {
                            return None;
                        }
                        // Strip decoration from commit message
                        if let Some((hash, rest)) = line.split_once(' ') {
                            let msg = if let Some(paren_idx) = rest.rfind('(') {
                                rest[..paren_idx].trim()
                            } else {
                                rest.trim()
                            };
                            Some(format!("{} {}", hash, msg))
                        } else {
                            Some(line.to_string())
                        }
                    })
                    .collect::<Vec<_>>()
            }
            _ => Vec::new(),
        };

        let mut parts = vec![
            format!("Current branch: {}", current_branch),
            format!(
                "Main branch (you will usually use this for PRs): {}",
                main_branch
            ),
            format!("Status: {}", status),
        ];

        if !recent_commits.is_empty() {
            parts.push("Recent commits:".to_string());
            parts.extend(recent_commits);
        }

        parts.join("\n")
    }

    /// Get the full project context.
    pub fn get_full_context(&mut self) -> String {
        let structure = self.get_directory_structure();
        let git_status = self.get_git_status();

        let large_repo_warning = if structure.len() >= (self.config.max_chars - 1000) {
            format!(
                " Large repository detected - showing summary view with depth limit {}. \
                 Use the LS tool (passing a specific path), Bash tool, and other tools to \
                 explore nested directories in detail.",
                self.config.max_depth
            )
        } else {
            String::new()
        };

        let template = UtilityPrompt::ProjectContext.read();
        template
            .replace("{large_repo_warning}", &large_repo_warning)
            .replace("{structure}", &structure)
            .replace("{abs_path}", &self.root_path.to_string_lossy())
            .replace("{git_status}", &git_status)
    }
}

/// Safely truncate a string to approximately max_bytes while respecting UTF-8 boundaries.
fn truncate_to_bytes(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    // Find a valid UTF-8 boundary at or before max_bytes
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Load project documentation file if present.
pub fn load_project_doc(workdir: &Path, max_bytes: usize) -> Option<String> {
    for name in PROJECT_DOC_FILENAMES {
        let path = workdir.join(name);
        if path.exists()
            && let Ok(content) = std::fs::read_to_string(&path)
        {
            let truncated = truncate_to_bytes(&content, max_bytes);
            return Some(truncated);
        }
    }
    None
}

/// Check if we're in a dangerous directory (home, root, etc).
pub fn is_dangerous_directory(path: &Path) -> (bool, String) {
    let resolved_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    // Get home directory
    let home_dir = dirs::home_dir();

    // Check home directory and common user folders
    if let Some(ref home) = home_dir {
        let dangerous_home_paths: Vec<(PathBuf, &str)> = vec![
            (home.clone(), "home directory"),
            (home.join("Documents"), "Documents folder"),
            (home.join("Desktop"), "Desktop folder"),
            (home.join("Downloads"), "Downloads folder"),
            (home.join("Pictures"), "Pictures folder"),
            (home.join("Movies"), "Movies folder"),
            (home.join("Music"), "Music folder"),
            (home.join("Library"), "Library folder"),
        ];

        for (dangerous_path, description) in dangerous_home_paths {
            if resolved_path == dangerous_path {
                return (true, format!("You are in the {}", description));
            }
        }
    }

    // Check system directories (macOS/Unix)
    let system_paths: &[(&str, &str)] = &[
        ("/Applications", "Applications folder"),
        ("/System", "System folder"),
        ("/Library", "System Library folder"),
        ("/usr", "System usr folder"),
        ("/private", "System private folder"),
    ];

    for (sys_path, description) in system_paths {
        let sys_path = Path::new(sys_path);
        if resolved_path == sys_path {
            return (true, format!("You are in the {}", description));
        }
    }

    (false, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_project_context_provider() {
        let dir = tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();

        let config = ProjectContextConfig::default();
        let mut provider = ProjectContextProvider::new(config, dir.path());

        let structure = provider.get_directory_structure();
        assert!(structure.contains("src/"));
        assert!(structure.contains("Cargo.toml"));
    }

    #[test]
    fn test_is_dangerous_directory() {
        // System directories
        let (is_dangerous, _) = is_dangerous_directory(Path::new("/usr"));
        assert!(is_dangerous);

        let (is_dangerous, _) = is_dangerous_directory(Path::new("/tmp/safe_project"));
        assert!(!is_dangerous);

        // Home directory
        if let Some(home) = dirs::home_dir() {
            let (is_dangerous, _) = is_dangerous_directory(&home);
            assert!(is_dangerous);

            // Documents folder
            let docs = home.join("Documents");
            if docs.exists() {
                let (is_dangerous, _) = is_dangerous_directory(&docs);
                assert!(is_dangerous);
            }
        }
    }
}
