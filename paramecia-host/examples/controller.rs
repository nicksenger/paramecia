//! Controller runner: loads a model and runs a WASM controller component.
//!
//! This is a thin CLI wrapper around `paramecia_host::run_host_cli`.

use anyhow::Result;
use clap::Parser;
use paramecia_host::HostOptions;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run a WASM controller component for inference"
)]
struct Args {
    /// Path to the compiled WASM controller component.
    #[arg(long)]
    controller: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Device offload mode for MoE expert weights.
    #[arg(long, default_value = "experts")]
    offload: String,

    /// Disable KV-cache quantization.
    #[arg(long)]
    no_kv_quant: bool,

    /// The model repository on HuggingFace Hub.
    #[arg(long, default_value = "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF")]
    model_id: String,

    /// The model file to download.
    #[arg(long, default_value = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf")]
    model_file: String,

    /// The tokenizer repository.
    #[arg(long, default_value = "Qwen/Qwen3-Next-80B-A3B-Instruct")]
    tokenizer_repo: String,

    /// Path to local model file (overrides HF download).
    #[arg(long)]
    model_path: Option<String>,

    /// Path to local tokenizer file (overrides HF download).
    #[arg(long)]
    tokenizer_path: Option<String>,

    /// Number of top-k logit entries to return in predictions.
    #[arg(long, default_value_t = 64)]
    top_k: usize,

    /// Number of stratified tail samples to return.
    #[arg(long, default_value_t = 16)]
    tail_samples: usize,

    /// Sampling temperature (0 = argmax/greedy).
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-p (nucleus) sampling threshold.
    #[arg(long)]
    top_p: Option<f64>,

    /// Sampling seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Directory for snapshot files.
    #[arg(long, default_value = "/tmp/paramecia-snapshots")]
    snapshot_dir: String,

    /// Multi-GPU layer split proportions (e.g., "3,1").
    #[arg(long)]
    layer_split: Option<String>,

    /// MTP speculation depth for inference (single-token decode only).
    #[arg(long = "n-speculative-inference")]
    n_speculative_inference: Option<usize>,

    /// MTP speculation depth for training loss.
    #[arg(long = "n-speculative-training")]
    n_speculative_training: Option<usize>,

    /// Allow overriding MTP speculative predictions at commit-time.
    #[arg(long = "commit-mtp-override")]
    commit_mtp_override: bool,

    /// Number of samples per minibatch for QuZO steps.
    #[arg(long, default_value_t = 2)]
    minibatch_size: usize,

    /// QuZO learning rate.
    #[arg(long, default_value = "0.0001")]
    training_lr: f64,

    /// QuZO perturbation epsilon.
    #[arg(long, default_value = "0.001")]
    training_epsilon: f64,

    /// Directory for training checkpoints.
    #[arg(long, default_value = "/tmp/paramecia-checkpoints")]
    checkpoint_dir: String,

    /// Number of gradient accumulation steps per training step.
    #[arg(long, default_value_t = 2)]
    n_grad_steps: usize,

    /// Which tensors to optimize: "all", "attention", "qk".
    #[arg(long, default_value = "all")]
    optimize_tensors: String,
}

impl From<Args> for HostOptions {
    fn from(args: Args) -> Self {
        HostOptions {
            controller: args.controller,
            cpu: args.cpu,
            offload: args.offload,
            no_kv_quant: args.no_kv_quant,
            model_id: args.model_id,
            model_file: args.model_file,
            tokenizer_repo: args.tokenizer_repo,
            model_path: args.model_path,
            tokenizer_path: args.tokenizer_path,
            top_k: args.top_k,
            tail_samples: args.tail_samples,
            temperature: args.temperature,
            top_p: args.top_p,
            seed: args.seed,
            snapshot_dir: args.snapshot_dir,
            layer_split: args.layer_split,
            n_speculative_inference: args.n_speculative_inference,
            n_speculative_training: args.n_speculative_training,
            commit_mtp_override: args.commit_mtp_override,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
            minibatch_size: args.minibatch_size,
            training_lr: args.training_lr,
            training_epsilon: args.training_epsilon,
            checkpoint_dir: args.checkpoint_dir,
            n_grad_steps: args.n_grad_steps,
            optimize_tensors: args.optimize_tensors,
            clip_threshold: 10.0,
            lb_loss: 0.01,
            z_loss: 0.001,
            interactive_endpoint: None,
            quiet: false,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    paramecia_host::run_host_cli(args.into())
}
