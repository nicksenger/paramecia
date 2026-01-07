//! Version update checking functionality.

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use tracing::debug;

/// Version update configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionUpdateConfig {
    pub last_check: Option<SystemTime>,
    pub last_version: Option<String>,
    pub check_interval_hours: u64,
}

impl Default for VersionUpdateConfig {
    fn default() -> Self {
        Self {
            last_check: None,
            last_version: None,
            check_interval_hours: 24,
        }
    }
}

/// Version information from GitHub releases API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubRelease {
    pub tag_name: String,
    pub name: String,
    pub body: String,
    pub published_at: String,
    pub html_url: String,
}

/// Version update checker.
pub struct VersionUpdateChecker {
    client: Client,
    config: VersionUpdateConfig,
    cache_path: std::path::PathBuf,
}

impl VersionUpdateChecker {
    /// Create a new version update checker.
    pub async fn new(cache_dir: &std::path::Path) -> Self {
        let cache_path = cache_dir.join("version_update_cache.json");
        let config = Self::load_config(&cache_path).await.unwrap_or_default();

        Self {
            client: Client::new(),
            config,
            cache_path,
        }
    }

    /// Load configuration from cache file.
    async fn load_config(path: &std::path::PathBuf) -> Result<VersionUpdateConfig> {
        if path.exists() {
            let content = tokio::fs::read_to_string(path).await?;
            let config: VersionUpdateConfig = serde_json::from_str(&content)?;
            Ok(config)
        } else {
            Ok(VersionUpdateConfig::default())
        }
    }

    /// Save configuration to cache file.
    async fn save_config(&self) -> Result<()> {
        let content = serde_json::to_string(&self.config)?;
        if let Some(parent) = self.cache_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&self.cache_path, content).await?;
        Ok(())
    }

    /// Check if we should perform a version check.
    pub fn should_check(&self) -> bool {
        if let Some(last_check) = self.config.last_check
            && let Ok(duration) = last_check.elapsed()
        {
            return duration >= Duration::from_secs(self.config.check_interval_hours * 3600);
        }
        true // Always check if never checked before
    }

    /// Check for updates from GitHub releases.
    pub async fn check_for_updates(&mut self, current_version: &str) -> Result<Option<String>> {
        // Update last check time
        self.config.last_check = Some(SystemTime::now());
        self.save_config().await?;

        // Fetch latest release from GitHub
        let url = "https://api.github.com/repos/mistralai/vibe/releases/latest";
        let response = self
            .client
            .get(url)
            .header("User-Agent", "paramecia-version-check")
            .send()
            .await?;

        if !response.status().is_success() {
            debug!("Failed to fetch version info: {}", response.status());
            return Ok(None);
        }

        let release: GitHubRelease = response.json().await?;
        let latest_version = release.tag_name.trim_start_matches('v');

        // Compare versions
        if latest_version != current_version {
            self.config.last_version = Some(latest_version.to_string());
            self.save_config().await?;
            Ok(Some(format!(
                "Update available: {} → {}\n{}",
                current_version, latest_version, release.html_url
            )))
        } else {
            Ok(None)
        }
    }
}
