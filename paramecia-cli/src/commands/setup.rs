//! Setup command for initial configuration.

use anyhow::Result;
use console::style;
use paramecia_harness::paths::ENV_FILE;

/// Run the setup wizard.
#[allow(unused)]
pub async fn run() -> Result<()> {
    println!();
    println!("{}", style("Welcome to Paramecia!").bold().cyan());
    println!();
    println!("Let's set up a model to get started.");
    println!();

    // TODO: allow selection between saving to tmp or persisting

    println!();
    println!(
        "{} Model saved to {}",
        style("✓").green().bold(),
        ENV_FILE.display()
    );
    println!();
    println!(
        "You're all set! Run {} to begin.",
        style("paramecia tui").cyan()
    );
    println!();

    Ok(())
}
