//! Prune transformer layers from a GGUF model.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example prune_layers --release -- \
//!     --input-model model.gguf \
//!     --output-model pruned.gguf \
//!     --layers-to-keep 12
//! ```

use anyhow::Result;
use clap::Parser;
use paramecia_opt::prune::PruneLayersOptions;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Prune transformer layers from a GGUF model")]
struct Args {
    /// Path to input GGUF model file
    #[arg(long)]
    input_model: PathBuf,

    /// Path for output GGUF with pruned layers
    #[arg(long)]
    output_model: PathBuf,

    /// Number of layers to keep (from the beginning)
    #[arg(long)]
    layers_to_keep: usize,

    /// Show verbose output
    #[arg(long, short)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let options = PruneLayersOptions {
        input_model: args.input_model,
        output_model: args.output_model,
        retained_layers: (0..args.layers_to_keep as u32).collect(),
        verbose: args.verbose,
    };

    paramecia_opt::prune::prune_layers(&options)
}
