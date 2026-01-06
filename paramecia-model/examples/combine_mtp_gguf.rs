//! Combine MTP head from one GGUF with remaining tensors from another

use anyhow::Result;
use clap::Parser;
use paramecia_model::graft::{graft, GraftOptions};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Combine MTP head from one GGUF with base tensors from another"
)]
struct Args {
    #[command(flatten)]
    options: GraftOptions,
}

fn main() -> Result<()> {
    let args = Args::parse();
    graft(&args.options)
}
