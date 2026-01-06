//! Path utilities for configuration and data directories.

use once_cell::sync::Lazy;
use std::path::PathBuf;

/// Get the Paramecia home directory.
///
/// Uses `PARAMECIA_HOME` environment variable if set, otherwise `~/.paramecia`.
pub fn paramecia_home() -> PathBuf {
    std::env::var("PARAMECIA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".paramecia")
        })
}

/// Global Paramecia home directory.
pub static PARAMECIA_HOME: Lazy<PathBuf> = Lazy::new(paramecia_home);

/// Configuration directory.
pub static CONFIG_DIR: Lazy<PathBuf> = Lazy::new(|| PARAMECIA_HOME.clone());

/// Configuration file path.
pub static CONFIG_FILE: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("config.toml"));

/// Environment file for API keys.
pub static ENV_FILE: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join(".env"));

/// Agents directory for custom agent configurations.
pub static AGENTS_DIR: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("agents"));

/// Prompts directory for custom system prompts.
pub static PROMPTS_DIR: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("prompts"));

/// Tools directory for custom tools.
pub static TOOLS_DIR: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("tools"));

/// Session logs directory.
pub static LOGS_DIR: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("logs"));

/// History file for command history.
pub static HISTORY_FILE: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("history"));

/// Instructions file for user instructions.
pub static INSTRUCTIONS_FILE: Lazy<PathBuf> = Lazy::new(|| CONFIG_DIR.join("instructions.md"));

/// Trusted folders configuration file.
pub static TRUSTED_FOLDERS_FILE: Lazy<PathBuf> =
    Lazy::new(|| CONFIG_DIR.join("trusted_folders.toml"));

/// Local config directory name.
pub const LOCAL_CONFIG_DIR: &str = ".paramecia";

/// Get the local configuration directory for a given working directory.
#[must_use]
pub fn local_config_dir(workdir: &std::path::Path) -> PathBuf {
    workdir.join(LOCAL_CONFIG_DIR)
}

/// Get the local configuration file for a given working directory.
#[must_use]
pub fn local_config_file(workdir: &std::path::Path) -> PathBuf {
    local_config_dir(workdir).join("config.toml")
}

/// Ensure required directories exist.
///
/// # Errors
///
/// Returns an error if directories cannot be created.
pub fn ensure_dirs() -> std::io::Result<()> {
    std::fs::create_dir_all(&*CONFIG_DIR)?;
    std::fs::create_dir_all(&*AGENTS_DIR)?;
    std::fs::create_dir_all(&*PROMPTS_DIR)?;
    std::fs::create_dir_all(&*TOOLS_DIR)?;
    std::fs::create_dir_all(&*LOGS_DIR)?;
    Ok(())
}
