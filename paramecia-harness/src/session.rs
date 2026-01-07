//! Session logging and management.

use chrono::{DateTime, Utc};
use paramecia_llm::LlmMessage;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

use crate::paths::LOGS_DIR;
use crate::types::{AgentStats, SessionMetadata};

/// Session logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLoggingConfig {
    /// Directory to save sessions.
    #[serde(default)]
    pub save_dir: String,
    /// Prefix for session files.
    #[serde(default = "default_session_prefix")]
    pub session_prefix: String,
    /// Whether logging is enabled.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_session_prefix() -> String {
    "session".to_string()
}

fn default_enabled() -> bool {
    true
}

impl Default for SessionLoggingConfig {
    fn default() -> Self {
        Self {
            save_dir: LOGS_DIR.to_string_lossy().to_string(),
            session_prefix: default_session_prefix(),
            enabled: default_enabled(),
        }
    }
}

impl SessionLoggingConfig {
    /// Get the effective save directory.
    #[must_use]
    pub fn save_dir(&self) -> PathBuf {
        if self.save_dir.is_empty() {
            LOGS_DIR.clone()
        } else {
            PathBuf::from(&self.save_dir)
        }
    }
}

/// A logged session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLog {
    /// Session metadata.
    pub metadata: SessionMetadata,
    /// Messages in the session.
    pub messages: Vec<LlmMessage>,
    /// Session statistics.
    pub stats: AgentStats,
}

/// Session logger.
pub struct SessionLogger {
    config: SessionLoggingConfig,
    session_id: String,
    start_time: DateTime<Utc>,
}

impl SessionLogger {
    /// Create a new session logger.
    #[must_use]
    pub fn new(config: SessionLoggingConfig) -> Self {
        Self {
            config,
            session_id: Uuid::new_v4().to_string(),
            start_time: Utc::now(),
        }
    }

    /// Get the current session ID.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Reset the session with a new ID.
    pub fn reset_session(&mut self) {
        self.session_id = Uuid::new_v4().to_string();
        self.start_time = Utc::now();
    }

    /// Get whether logging is enabled.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the log file path for the current session.
    #[must_use]
    pub fn filepath(&self) -> Option<PathBuf> {
        if !self.config.enabled {
            return None;
        }

        let save_dir = self.config.save_dir();
        let filename = format!("{}_{}.json", self.config.session_prefix, self.session_id);
        Some(save_dir.join(filename))
    }

    /// Save a session log.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be saved.
    pub async fn save(
        &self,
        messages: &[LlmMessage],
        stats: &AgentStats,
        auto_approve: bool,
    ) -> std::io::Result<PathBuf> {
        if !self.config.enabled {
            return Ok(PathBuf::new());
        }

        let save_dir = self.config.save_dir();
        tokio::fs::create_dir_all(&save_dir).await?;

        let filename = format!("{}_{}.json", self.config.session_prefix, self.session_id);
        let path = save_dir.join(&filename);

        // Get git info asynchronously to avoid blocking
        let (git_commit, git_branch) = tokio::join!(get_git_commit_async(), get_git_branch_async());

        let metadata = SessionMetadata {
            session_id: self.session_id.clone(),
            start_time: self.start_time.to_rfc3339(),
            end_time: Some(Utc::now().to_rfc3339()),
            git_commit,
            git_branch,
            auto_approve,
            username: whoami::username(),
        };

        let log = SessionLog {
            metadata,
            messages: messages.to_vec(),
            stats: stats.clone(),
        };

        let json = serde_json::to_string_pretty(&log)?;
        tokio::fs::write(&path, json).await?;

        Ok(path)
    }

    /// Find the latest session asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub async fn find_latest(config: &SessionLoggingConfig) -> std::io::Result<Option<PathBuf>> {
        let save_dir = config.save_dir();
        if !tokio::fs::try_exists(&save_dir).await.unwrap_or(false) {
            return Ok(None);
        }

        let mut read_dir = tokio::fs::read_dir(&save_dir).await?;
        let mut sessions: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();

        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                let modified = entry
                    .metadata()
                    .await
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                sessions.push((path, modified));
            }
        }

        sessions.sort_by_key(|(_, modified)| *modified);
        Ok(sessions.last().map(|(path, _)| path.clone()))
    }

    /// Load a session from a file asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if the session cannot be loaded.
    pub async fn load(path: &PathBuf) -> std::io::Result<SessionLog> {
        let content = tokio::fs::read_to_string(path).await?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Get the current git commit hash asynchronously.
async fn get_git_commit_async() -> Option<String> {
    tokio::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// Get the current git branch asynchronously.
async fn get_git_branch_async() -> Option<String> {
    tokio::process::Command::new("git")
        .args(["branch", "--show-current"])
        .output()
        .await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}
