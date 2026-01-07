//! List all tensor names in a GGUF file

use anyhow::Result;
use clap::Parser;
use paramecia_model::inspect::{inspect_model, InspectOptions};

#[derive(Parser, Debug)]
#[command(author, version, about = "List tensor names in a GGUF file")]
struct Args {
    #[command(flatten)]
    options: InspectOptions,
}

fn main() -> Result<()> {
    let args = Args::parse();
    inspect_model(&args.options)
}
