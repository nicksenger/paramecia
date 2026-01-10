//! Setup command for initial configuration.

use anyhow::Result;
use console::style;
use dialoguer::{Password, theme::ColorfulTheme};
use paramecia_harness::paths::{ENV_FILE, ensure_dirs};
use std::fs;

/// Run the setup wizard.
pub async fn run() -> Result<()> {
    println!();
    println!("{}", style("Welcome to Paramecia!").bold().cyan());
    println!();
    println!("Let's set up your API key to get started.");
    println!();

    // Ensure directories exist
    ensure_dirs()?;

    // Get API key
    let api_key: String = Password::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter your API key")
        .with_confirmation("Confirm your API key", "API keys don't match, try again")
        .interact()?;

    if api_key.is_empty() {
        println!(
            "{}",
            style("No API key provided. Setup cancelled.").yellow()
        );
        return Ok(());
    }

    // Save to .env file
    let env_content = format!("API_KEY={api_key}\n");
    fs::write(&*ENV_FILE, env_content)?;

    println!();
    println!(
        "{} API key saved to {}",
        style("✓").green().bold(),
        ENV_FILE.display()
    );
    println!();
    println!(
        "You're all set! Run {} to start chatting.",
        style("paramecia").cyan()
    );
    println!();

    Ok(())
}
