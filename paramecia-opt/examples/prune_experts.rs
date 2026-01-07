//! Prune MoE experts based on usage statistics from tuning files.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example prune_experts --release -- \
//!     --input-model model.gguf \
//!     --tuning-data ./tuning_outputs/ \
//!     --output-model pruned.gguf \
//!     --experts-to-keep 32
//! ```

use anyhow::Result;
use clap::Parser;
use paramecia_opt::prune::{compute_expert_indices_from_tuning, PruneExpertsOptions};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Prune MoE experts based on usage statistics from tuning files"
)]
struct Args {
    /// Path to input GGUF model file
    #[arg(long)]
    input_model: PathBuf,

    /// Path to tuning data (.bin file or directory containing .bin files)
    #[arg(long)]
    tuning_data: PathBuf,

    /// Path for output GGUF with pruned experts
    #[arg(long)]
    output_model: PathBuf,

    /// Number of experts to keep per layer (rest will be pruned)
    #[arg(long)]
    experts_to_keep: usize,

    /// Show verbose output
    #[arg(long, short)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let retained_indices = compute_expert_indices_from_tuning(
        &args.input_model,
        &args.tuning_data,
        args.experts_to_keep,
        args.verbose,
    )?;

    let options = PruneExpertsOptions {
        input_model: args.input_model,
        output_model: args.output_model,
        retained_indices,
        verbose: args.verbose,
    };

    paramecia_opt::prune::prune_experts(&options)
}
