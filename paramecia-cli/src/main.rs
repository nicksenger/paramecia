//! Paramecia CLI

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

mod args;
mod commands;
mod ui;

use args::Args;
use paramecia_harness::paths::{CONFIG_FILE, HISTORY_FILE, INSTRUCTIONS_FILE};

/// Bootstrap configuration files, similar to Python's implementation.
async fn bootstrap_config_files() -> Result<()> {
    // Create config file if it doesn't exist
    if !CONFIG_FILE.exists()
        && let Err(e) = paramecia_harness::VibeConfig::default().save()
    {
        eprintln!("Could not create default config file: {e}");
    }

    // Create instructions file if it doesn't exist
    if !INSTRUCTIONS_FILE.exists() {
        if let Some(parent) = INSTRUCTIONS_FILE.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            eprintln!("Could not create instructions directory: {e}");
        }
        if let Err(e) = std::fs::write(&*INSTRUCTIONS_FILE, "") {
            eprintln!("Could not create instructions file: {e}");
        }
    }

    // Create history file if it doesn't exist
    if !HISTORY_FILE.exists() {
        if let Some(parent) = HISTORY_FILE.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            eprintln!("Could not create history directory: {e}");
        }
        if let Err(e) = std::fs::write(&*HISTORY_FILE, "Hello Vibe!\n") {
            eprintln!("Could not create history file: {e}");
        }
    }

    Ok(())
}

/// Check and resolve trusted folder status, similar to Python's implementation.
async fn check_and_resolve_trusted_folder() -> Result<()> {
    use paramecia_harness::trusted_folders::TrustedFoldersManager;

    let cwd = std::env::current_dir()?;
    let vibe_dir = cwd.join(".vibe");

    // Skip if not in a project directory or if in home directory
    if !vibe_dir.exists() || cwd == std::env::home_dir().unwrap_or_default() {
        return Ok(());
    }

    let trusted_folders_manager = TrustedFoldersManager::load()?;

    match trusted_folders_manager.is_trusted(&cwd) {
        Some(true) => Ok(()),  // Already trusted
        Some(false) => Ok(()), // Explicitly untrusted
        None => {
            // Folder is unknown, ask user with UI dialog
            let mut dialog = ui::trust_dialog::TrustDialog::new(&cwd);
            match dialog.run() {
                Ok(ui::trust_dialog::TrustDialogResult::Trusted) => {
                    let mut manager = trusted_folders_manager;
                    manager.add_trusted(&cwd)?;
                    println!("Folder trusted.");
                }
                Ok(ui::trust_dialog::TrustDialogResult::Untrusted) => {
                    let mut manager = trusted_folders_manager;
                    manager.add_untrusted(&cwd)?;
                    println!("Folder marked as untrusted.");
                }
                Ok(ui::trust_dialog::TrustDialogResult::Quit) | Err(_) => {
                    // User quit or error occurred, exit gracefully
                    std::process::exit(0);
                }
            }
            Ok(())
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Handle setup command
    if args.setup {
        commands::setup::run().await?;
        return Ok(());
    }

    // Determine if we're in interactive mode (TUI)
    let is_interactive = args.prompt.is_none();

    // Initialize logging
    // In TUI mode, suppress all logs to avoid interfering with the UI
    let env_filter = if is_interactive {
        // Allow user to override with RUST_LOG env var, but default to off for TUI
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("off"))
    } else {
        EnvFilter::from_default_env()
            .add_directive("paramecia=info".parse().unwrap())
            .add_directive("warn".parse().unwrap())
    };

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    // Bootstrap config files (create defaults if they don't exist)
    bootstrap_config_files().await?;

    // Check and resolve trusted folder (only for interactive mode)
    if is_interactive {
        check_and_resolve_trusted_folder().await?;
    }

    // Load configuration
    let config = paramecia_harness::VibeConfig::load(args.agent.as_deref()).map_err(|e| {
        match e {
            paramecia_harness::error::VibeError::MissingPromptFile {
                system_prompt_id,
                prompt_dir,
            } => {
                eprintln!("Invalid system prompt id: {}", system_prompt_id);
                eprintln!(
                    "Must be one of the available prompts, or correspond to a .md file in {}",
                    prompt_dir
                );
            }
            _ => {
                eprintln!("Configuration error: {e}");
            }
        }
        std::process::exit(1);
    })?;

    // Load session if requested
    let loaded_messages = if args.continue_session || args.resume.is_some() {
        if !config.session_logging.enabled {
            eprintln!(
                "Session logging is disabled. Enable it in config to use --continue or --resume"
            );
            std::process::exit(1);
        }

        let session_to_load = if args.continue_session {
            // Find latest session
            match paramecia_harness::session::SessionLogger::find_latest(&config.session_logging)
                .await
            {
                Ok(Some(path)) => Some(path),
                Ok(None) => {
                    eprintln!(
                        "No previous sessions found in {}",
                        config.session_logging.save_dir().display()
                    );
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Failed to find latest session: {e}");
                    std::process::exit(1);
                }
            }
        } else if let Some(session_id) = &args.resume {
            // Find session by ID (partial matching) - match Python behavior
            let save_dir = config.session_logging.save_dir();
            if !tokio::fs::try_exists(&save_dir).await.unwrap_or(false) {
                eprintln!("Session '{session_id}' not found in {}", save_dir.display());
                std::process::exit(1);
            }

            // Handle full UUID by extracting short form (first 8 chars), like Python
            let short_id = if session_id.contains('-') {
                session_id.split('-').next().unwrap_or(session_id)
            } else {
                session_id
            };

            // Try exact match first, then partial match (like Python)
            let session_prefix = config.session_logging.session_prefix.clone();
            let mut best_match: Option<(std::path::PathBuf, std::time::SystemTime)> = None;

            // First try exact match: {prefix}_{short_id}_*.json
            let mut read_dir = match tokio::fs::read_dir(&save_dir).await {
                Ok(rd) => rd,
                Err(e) => {
                    eprintln!("Failed to read session directory: {e}");
                    std::process::exit(1);
                }
            };

            while let Ok(Some(entry)) = read_dir.next_entry().await {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "json") {
                    let file_name_str = entry.file_name().to_string_lossy().into_owned();
                    let search_pattern = format!("{}_{}_", session_prefix, short_id);

                    // Check if filename matches the exact pattern
                    if file_name_str.contains(&search_pattern) {
                        let modified = entry
                            .metadata()
                            .await
                            .and_then(|m| m.modified())
                            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

                        // Keep the most recent match
                        if let Some((_, current_modified)) = &best_match {
                            if modified > *current_modified {
                                best_match = Some((path, modified));
                            }
                        } else {
                            best_match = Some((path, modified));
                        }
                    }
                }
            }

            // If no exact matches, try partial match: {prefix}_{short_id}*.json
            if best_match.is_none() {
                let mut read_dir = match tokio::fs::read_dir(&save_dir).await {
                    Ok(rd) => rd,
                    Err(e) => {
                        eprintln!("Failed to read session directory: {e}");
                        std::process::exit(1);
                    }
                };

                while let Ok(Some(entry)) = read_dir.next_entry().await {
                    let path = entry.path();
                    if path.extension().is_some_and(|ext| ext == "json") {
                        let file_name_str = entry.file_name().to_string_lossy().into_owned();
                        let search_pattern = format!("{}_{}", session_prefix, short_id);

                        // Check if filename contains the partial pattern
                        if file_name_str.contains(&search_pattern) {
                            let modified = entry
                                .metadata()
                                .await
                                .and_then(|m| m.modified())
                                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

                            // Keep the most recent match
                            if let Some((_, current_modified)) = &best_match {
                                if modified > *current_modified {
                                    best_match = Some((path, modified));
                                }
                            } else {
                                best_match = Some((path, modified));
                            }
                        }
                    }
                }
            }

            match best_match {
                Some((path, _)) => Some(path),
                None => {
                    eprintln!("Session '{session_id}' not found in {}", save_dir.display());
                    std::process::exit(1);
                }
            }
        } else {
            None
        };

        if let Some(session_path) = session_to_load {
            match paramecia_harness::session::SessionLogger::load(&session_path).await {
                Ok(session_log) => {
                    println!("Loaded session: {}", session_path.display());
                    Some(session_log.messages)
                }
                Err(e) => {
                    eprintln!("Failed to load session: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // Determine mode
    let mode = args.mode();

    // Check for programmatic mode
    if args.prompt.is_some() {
        let prompt = args.get_prompt()?;
        if !prompt.is_empty() {
            // In programmatic mode, use auto-approve mode
            let programmatic_mode = paramecia_harness::modes::AgentMode::AutoApprove;
            commands::programmatic::run(config, programmatic_mode, &prompt, &args, loaded_messages)
                .await?;
        } else {
            // Empty prompt in programmatic mode - try to get from stdin
            if let Some(stdin_prompt) = args.get_stdin_prompt() {
                let programmatic_mode = paramecia_harness::modes::AgentMode::AutoApprove;
                commands::programmatic::run(
                    config,
                    programmatic_mode,
                    &stdin_prompt,
                    &args,
                    loaded_messages,
                )
                .await?;
            } else {
                eprintln!("Error: No prompt provided for programmatic mode");
                std::process::exit(1);
            }
        }
    } else {
        // Interactive mode
        let initial_prompt = args
            .initial_prompt
            .clone()
            .or_else(|| args.get_stdin_prompt());
        commands::interactive::run(config, mode, initial_prompt, &args, loaded_messages).await?;
    }

    Ok(())
}
