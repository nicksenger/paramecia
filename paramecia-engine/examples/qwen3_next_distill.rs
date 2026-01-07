//! QuZO Knowledge Distillation Training for Qwen3-Next
//!
//! ## Usage
//!
//! ```bash
//! # Use --features=cuda, --features=metal, or --features=vulkan as appropriate
//! cargo run --features=cuda --release -p paramecia-engine --example qwen3_next_distill -- \
//!     --model-path /path/to/model.gguf \
//!     --tuning-data /path/to/tuning_outputs/ \
//!     --output-path /path/to/trained.gguf \
//!     --minibatch-size 2 \
//!     --n-grad-steps 2 \
//!     --chunk-size 2048 \
//!     --steps 1000
//! ```

use clap::Parser;
use paramecia_core::Result;
use paramecia_opt::tune::TuneOptions;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "QuZO knowledge distillation training for Qwen3-Next"
)]
struct Args {
    /// Path to input GGUF model file
    #[arg(long)]
    model_path: PathBuf,

    /// Path to tuning data (.bin file or directory containing .bin files)
    #[arg(long)]
    tuning_data: PathBuf,

    /// Path for output GGUF with trained weights
    #[arg(long)]
    output_path: PathBuf,

    /// Number of training steps
    #[arg(long, default_value = "1000")]
    steps: usize,

    /// Minibatch size (conversations processed in parallel)
    #[arg(long, default_value = "2")]
    minibatch_size: usize,

    /// Number of gradient accumulation steps
    #[arg(long, default_value = "2")]
    n_grad_steps: usize,

    /// Maximum sequence length (conversations longer than this will be truncated)
    #[arg(long)]
    max_sequence_length: Option<usize>,

    /// Chunk size for processing long sequences
    #[arg(long, default_value = "2048")]
    chunk_size: usize,

    /// QuZO learning rate
    #[arg(long, default_value = "0.0001")]
    lr: f64,

    /// QuZO perturbation magnitude (base epsilon)
    #[arg(long, default_value = "0.001")]
    epsilon: f64,

    /// Epsilon multiplier for embedding layer (token_embd)
    #[arg(long, default_value = "0.1")]
    eps_embedding: f64,

    /// Epsilon multiplier for attention Q/K/V/O projections
    #[arg(long, default_value = "1.0")]
    eps_attention: f64,

    /// Epsilon multiplier for MoE router/gating layers
    #[arg(long, default_value = "0.5")]
    eps_moe_gating: f64,

    /// Epsilon multiplier for MoE expert layers
    #[arg(long, default_value = "2.0")]
    eps_moe_experts: f64,

    /// Epsilon multiplier for MTP (Multi-Token Prediction) heads
    #[arg(long, default_value = "1.5")]
    eps_mtp: f64,

    /// Epsilon multiplier for other layers (SSM, norms, output)
    #[arg(long, default_value = "1.0")]
    eps_other: f64,

    /// Number of perturbation samples per step
    #[arg(long, default_value = "1")]
    num_samples: usize,

    /// Gradient clipping threshold.
    #[arg(long, default_value = "10.0")]
    clip_threshold: f64,

    /// Weight for Z-loss (router stability)
    #[arg(long = "z-loss", default_value = "0.001")]
    z_loss_weight: f64,

    /// Weight for load balance loss (expert utilization)
    #[arg(long = "lb-loss", default_value = "0.01")]
    lb_loss_weight: f64,

    /// Temperature for KL divergence (1.0 = no softening)
    #[arg(long, default_value = "1.0")]
    temperature: f64,

    /// Checkpoint every N steps (0 = no checkpointing)
    #[arg(long, default_value = "100")]
    checkpoint_every: usize,

    /// Random seed for reproducibility (random if not specified)
    #[arg(long)]
    seed: Option<u64>,

    /// Shuffle training data each epoch
    #[arg(long, default_value = "true")]
    shuffle: bool,

    /// Log progress every N steps
    #[arg(long, default_value = "1")]
    log_every: usize,

    /// Device offload mode: none, experts, up, updown
    #[arg(long, default_value = "experts")]
    offload: String,

    /// Which tensors to optimize: all, attention, qk
    #[arg(long, default_value = "all")]
    optimize: String,

    /// Dry run: train but don't save output model (useful for testing)
    #[arg(long)]
    dry_run: bool,

    /// Show verbose output
    #[arg(long, short)]
    verbose: bool,

    /// Number of speculative tokens for MTP loss (0 = disabled)
    #[arg(long, default_value = "0")]
    n_speculative: usize,

    /// MTP weight decay factor
    #[arg(long, default_value = "0.5")]
    mtp_decay: f64,

    /// Generate perturbation tensors lazily to reduce peak memory
    #[arg(long)]
    lazy_perturbations: bool,

    /// Multi-GPU layer split proportions (e.g., "3,1" = 75% GPU 0, 25% GPU 1)
    #[arg(long)]
    layer_split: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let options = TuneOptions {
        model_path: args.model_path,
        tuning_data: args.tuning_data,
        output_path: args.output_path,
        steps: args.steps,
        minibatch_size: args.minibatch_size,
        n_grad_steps: args.n_grad_steps,
        max_sequence_length: args.max_sequence_length,
        chunk_size: args.chunk_size,
        lr: args.lr,
        epsilon: args.epsilon,
        eps_embedding: args.eps_embedding,
        eps_attention: args.eps_attention,
        eps_moe_gating: args.eps_moe_gating,
        eps_moe_experts: args.eps_moe_experts,
        eps_mtp: args.eps_mtp,
        eps_other: args.eps_other,
        num_samples: args.num_samples,
        clip_threshold: args.clip_threshold,
        z_loss_weight: args.z_loss_weight,
        lb_loss_weight: args.lb_loss_weight,
        temperature: args.temperature,
        checkpoint_every: args.checkpoint_every,
        seed: args.seed,
        shuffle: args.shuffle,
        log_every: args.log_every,
        offload: args.offload,
        optimize: args.optimize,
        dry_run: args.dry_run,
        verbose: args.verbose,
        num_speculative: args.n_speculative,
        mtp_decay: args.mtp_decay,
        lazy_perturbations: args.lazy_perturbations,
        layer_split: args.layer_split,
    };

    paramecia_opt::tune::run_distillation(&options)
}
