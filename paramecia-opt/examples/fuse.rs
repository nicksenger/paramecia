//! Task Arithmetic Fusion for GGUF models.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release -p paramecia-opt --example fuse -- \
//!     --base /path/to/qwen3-base.gguf \
//!     --model /path/to/os-tuned.gguf:0.8 \
//!     --model /path/to/compiler-tuned.gguf:0.4 \
//!     --output /path/to/merged.gguf
//! ```

use anyhow::Result;
use clap::Parser;
use paramecia_opt::fuse::{parse_model_spec, FuseOptions, QuantConflictStrategy};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Task Arithmetic Fusion for GGUF models")]
struct Args {
    /// Path to base GGUF model (reference point for computing deltas)
    #[arg(long)]
    base: PathBuf,

    /// Models to fuse with weights (format: path:weight, e.g., model.gguf:0.8)
    /// First model's quantization types are used for output
    #[arg(long, required = true)]
    model: Vec<String>,

    /// Output path for merged GGUF
    #[arg(long, short)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let models: Vec<(PathBuf, f32)> = args
        .model
        .iter()
        .map(|s| parse_model_spec(s))
        .collect::<Result<Vec<_>>>()?;

    let options = FuseOptions {
        base: args.base,
        models,
        output: args.output,
        quant_conflict_strategy: QuantConflictStrategy::Highest,
    };

    paramecia_opt::fuse::fuse_models(&options)
}
