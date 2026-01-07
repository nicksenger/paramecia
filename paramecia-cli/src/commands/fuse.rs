//! Fuse subcommand — task arithmetic fusion of GGUF models.

use anyhow::Result;

use crate::args::FuseArgs;

pub fn run(args: &FuseArgs) -> Result<()> {
    let models: Vec<(std::path::PathBuf, f32)> = args
        .model
        .iter()
        .map(|s| paramecia_engine::parse_model_spec(s))
        .collect::<Result<_>>()?;
    paramecia_engine::fuse_models(
        &args.base,
        &models,
        &args.output,
        paramecia_engine::QuantConflictStrategy::Reject,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
