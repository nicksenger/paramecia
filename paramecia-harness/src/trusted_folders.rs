//! Trusted folders management for security.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::paths::CONFIG_DIR;

/// Trusted folders configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrustedFoldersConfig {
    /// List of trusted folder paths.
    #[serde(default)]
    pub trusted: Vec<String>,

    /// List of explicitly untrusted folder paths.
    #[serde(default)]
    pub untrusted: Vec<String>,
}

/// Trusted folders manager.
#[derive(Debug, Clone)]
pub struct TrustedFoldersManager {
    config: TrustedFoldersConfig,
}

impl Default for TrustedFoldersManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TrustedFoldersManager {
    /// Create a new trusted folders manager.
    pub fn new() -> Self {
        Self {
            config: TrustedFoldersConfig::default(),
        }
    }

    /// Load the trusted folders configuration.
    pub fn load() -> Result<Self> {
        let path = trusted_folders_file();
        if path.exists() {
            let content =
                fs::read_to_string(&path).context("Failed to read trusted folders file")?;
            let config: TrustedFoldersConfig =
                toml::from_str(&content).context("Failed to parse trusted folders config")?;
            Ok(Self { config })
        } else {
            Ok(Self::new())
        }
    }

    /// Save the trusted folders configuration.
    pub fn save(&self) -> Result<()> {
        let path = trusted_folders_file();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        let content = toml::to_string_pretty(&self.config)
            .context("Failed to serialize trusted folders config")?;
        fs::write(&path, content).context("Failed to write trusted folders file")?;
        Ok(())
    }

    /// Check if a folder is trusted.
    /// Returns Some(true) if trusted, Some(false) if untrusted, None if unknown.
    pub fn is_trusted(&self, path: &Path) -> Option<bool> {
        let canonical_path = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => return None,
        };
        let path_str = canonical_path.to_string_lossy();

        // Check if explicitly untrusted
        if self
            .config
            .untrusted
            .iter()
            .any(|p| path_str.starts_with(p))
        {
            return Some(false);
        }

        // Check if trusted
        if self.config.trusted.iter().any(|p| path_str.starts_with(p)) {
            return Some(true);
        }

        None
    }

    /// Add a folder to the trusted list.
    pub fn add_trusted(&mut self, path: &Path) -> Result<()> {
        let canonical_path = path.canonicalize().context("Failed to canonicalize path")?;
        let path_str = canonical_path.to_string_lossy().to_string();

        // Remove from untrusted if present
        self.config.untrusted.retain(|p| !path_str.starts_with(p));

        // Add to trusted if not already present
        if !self.config.trusted.iter().any(|p| path_str.starts_with(p)) {
            self.config.trusted.push(path_str);
        }

        self.save()
    }

    /// Add a folder to the untrusted list.
    pub fn add_untrusted(&mut self, path: &Path) -> Result<()> {
        let canonical_path = path.canonicalize().context("Failed to canonicalize path")?;
        let path_str = canonical_path.to_string_lossy().to_string();

        // Remove from trusted if present
        self.config.trusted.retain(|p| !path_str.starts_with(p));

        // Add to untrusted if not already present
        if !self
            .config
            .untrusted
            .iter()
            .any(|p| path_str.starts_with(p))
        {
            self.config.untrusted.push(path_str);
        }

        self.save()
    }
}

fn trusted_folders_file() -> PathBuf {
    CONFIG_DIR.join("trusted_folders.toml")
}
