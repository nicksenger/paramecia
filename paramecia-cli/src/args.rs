//! Command-line argument parsing.

use clap::{Parser, Subcommand};
use paramecia_harness::modes::AgentMode;
use paramecia_harness::types::OutputFormat;
use std::io::{self, Read};
use std::path::PathBuf;

const VERSION: &str = "1.2.1";

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_CTL_MODEL_ID: &str = "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_CTL_MODEL_ID: &str = "unsloth/Qwen3.5-35B-A3B-GGUF";

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_CTL_MODEL_FILE: &str = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_CTL_MODEL_FILE: &str = "Qwen3.5-35B-A3B-Q4_K_M.gguf";

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_CTL_TOKENIZER_REPO: &str = "Qwen/Qwen3-Next-80B-A3B-Instruct";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_CTL_TOKENIZER_REPO: &str = "Qwen/Qwen3.5-35B-A3B";

/// Paramecia CLI
#[derive(Parser, Debug)]
#[command(name = "paramecia")]
#[command(author = "Nick Senger")]
#[command(version = VERSION)]
#[command(about = "Paramecia: a single-process agentic TUI and finetuning framework.")]
#[command(long_about = None)]
pub struct Args {
    /// Subcommand to run (`tui` for interactive mode)
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run the interactive TUI
    Tui(TuiArgs),
    /// Run a WASM controller component
    Ctl(Box<ControlArgs>),
    /// Inspect tensors in a GGUF model file
    Inspect(InspectArgs),
    /// Task arithmetic fusion of GGUF models
    Fuse(FuseArgs),
    /// Run the visualizer GUI
    Visualize,
}

/// Arguments for the `tui` subcommand.
#[derive(Parser, Debug)]
pub struct TuiArgs {
    /// Run in programmatic mode: send prompt, auto-approve all tools, output response, and exit
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    pub prompt: Option<String>,

    /// Maximum number of assistant turns (only applies in programmatic mode with -p)
    #[arg(long, value_name = "N")]
    pub max_turns: Option<u32>,

    /// Enable specific tools. In programmatic mode (-p), this disables all other tools.
    /// Can use exact names, glob patterns (e.g., 'bash*'), or regex with 're:' prefix.
    /// Can be specified multiple times.
    #[arg(long = "enabled-tools", value_name = "TOOL")]
    pub enabled_tools: Vec<String>,

    /// Output format for programmatic mode (-p): 'text' for human-readable (default),
    /// 'json' for all messages at end, 'streaming' for newline-delimited JSON per message.
    #[arg(long, value_name = "FORMAT", default_value = "text")]
    pub output: String,

    /// Start in auto-approve mode: never ask for approval before running tools
    #[arg(long, default_value = "false")]
    pub auto_approve: bool,

    /// Start in plan mode: read-only tools for exploration and planning
    #[arg(long, default_value = "false")]
    pub plan: bool,

    /// Load agent configuration from ~/.paramecia/agents/NAME.toml
    #[arg(long, value_name = "NAME")]
    pub agent: Option<String>,

    /// Continue from the most recent saved session
    #[arg(short = 'c', long = "continue")]
    pub continue_session: bool,

    /// Resume a specific session by its ID (supports partial matching)
    #[arg(long, value_name = "SESSION_ID")]
    pub resume: Option<String>,

    /// Initial prompt to start the interactive session with
    #[arg(value_name = "PROMPT")]
    pub initial_prompt: Option<String>,

    /// Path to a WASM controller component to use as backend
    #[arg(long, value_name = "PATH")]
    pub controller: Option<String>,
}

/// Arguments for the `ctl` subcommand.
#[derive(Parser, Debug)]
pub struct ControlArgs {
    /// Path to the compiled WASM controller component
    #[arg(long)]
    pub controller: String,

    /// Run on CPU rather than on GPU
    #[arg(long)]
    pub cpu: bool,

    /// Device offload mode for MoE expert weights
    #[arg(long, default_value = "experts")]
    pub offload: String,

    /// Disable KV-cache quantization
    #[arg(long)]
    pub no_kv_quant: bool,

    /// HuggingFace model repository
    #[arg(long, default_value = DEFAULT_CTL_MODEL_ID)]
    pub model_id: String,

    /// GGUF model filename
    #[arg(long, default_value = DEFAULT_CTL_MODEL_FILE)]
    pub model_file: String,

    /// HuggingFace tokenizer repository
    #[arg(long, default_value = DEFAULT_CTL_TOKENIZER_REPO)]
    pub tokenizer_repo: String,

    /// Local model path (overrides HF download)
    #[arg(long)]
    pub model_path: Option<String>,

    /// Local tokenizer path (overrides HF download)
    #[arg(long)]
    pub tokenizer_path: Option<String>,

    /// Directory for snapshot storage
    #[arg(long, default_value = "/tmp/paramecia-snapshots")]
    pub snapshot_dir: String,

    /// Top-k sampling limit
    #[arg(long, default_value_t = 64)]
    pub top_k: usize,

    /// Number of tail samples for importance sampling
    #[arg(long, default_value_t = 16)]
    pub tail_samples: usize,

    /// Temperature for generation (0.0 for greedy)
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,

    /// Top-p (nucleus) sampling threshold
    #[arg(long)]
    pub top_p: Option<f64>,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Multi-GPU layer split proportions
    #[arg(long)]
    pub layer_split: Option<String>,

    /// MTP speculation depth for inference (single-token decode only)
    #[arg(long = "n-speculative-inference")]
    pub n_speculative_inference: Option<usize>,

    /// MTP speculation depth for training loss
    #[arg(long = "n-speculative-training")]
    pub n_speculative_training: Option<usize>,

    /// Allow overriding MTP speculative predictions at commit-time
    #[arg(long = "commit-mtp-override")]
    pub commit_mtp_override: bool,

    /// Number of samples per minibatch for QuZO steps
    #[arg(long, default_value_t = 2)]
    pub minibatch_size: usize,

    /// QuZO learning rate
    #[arg(long, default_value = "0.0001")]
    pub training_lr: f64,

    /// QuZO perturbation epsilon
    #[arg(long, default_value = "0.001")]
    pub training_epsilon: f64,

    /// Directory for training checkpoints
    #[arg(long)]
    pub checkpoint_dir: Option<String>,

    /// Number of gradient accumulation steps per training step
    #[arg(long, default_value_t = 2)]
    pub n_grad_steps: usize,

    /// Which tensors to optimize: "all", "attention", "qk"
    #[arg(long, default_value = "all")]
    pub optimize_tensors: String,

    /// QuZO directional derivative clipping threshold
    #[arg(long, default_value_t = 10.0)]
    pub clip_threshold: f64,

    /// Error feedback mode: "none", "persistent", "replay:N"
    #[arg(long, default_value = "none")]
    pub error_feedback: String,

    /// Error feedback decay (gamma, 0-1)
    #[arg(long, default_value_t = 0.9)]
    pub error_decay: f64,

    /// Error feedback gain (0 = purely stochastic, 1 = full error feedback)
    #[arg(long, default_value_t = 1.0)]
    pub error_gain: f64,

    /// Load-balance loss weight for MoE routing
    #[arg(long, default_value_t = 0.01)]
    pub lb_loss: f64,

    /// Z-loss weight for MoE router stability
    #[arg(long, default_value_t = 0.001)]
    pub z_loss: f64,
}

/// Arguments for the `inspect` subcommand.
#[derive(Parser, Debug)]
pub struct InspectArgs {
    /// Path to GGUF file
    #[arg(long)]
    pub model_path: PathBuf,

    /// Filter tensors by name (case-insensitive substring match)
    #[arg(long)]
    pub filter: Option<String>,

    /// Show model metadata
    #[arg(long, default_value = "true")]
    pub show_metadata: bool,
}

/// Arguments for the `fuse` subcommand.
#[derive(Parser, Debug)]
pub struct FuseArgs {
    /// Path to base GGUF model (reference point for computing deltas)
    #[arg(long)]
    pub base: PathBuf,

    /// Models to fuse with weights (format: path:weight, e.g., model.gguf:0.8)
    #[arg(long, required = true)]
    pub model: Vec<String>,

    /// Output path for merged GGUF
    #[arg(long, short)]
    pub output: PathBuf,
}

impl TuiArgs {
    /// Get the agent mode based on arguments.
    #[must_use]
    pub fn mode(&self) -> AgentMode {
        if self.plan {
            AgentMode::Plan
        } else if self.auto_approve {
            AgentMode::AutoApprove
        } else {
            AgentMode::Default
        }
    }

    /// Get the output format.
    #[must_use]
    pub fn output_format(&self) -> OutputFormat {
        self.output.parse().unwrap_or_default()
    }

    /// Get the prompt from arguments or stdin.
    pub fn get_prompt(&self) -> anyhow::Result<String> {
        // Handle the case where --prompt is provided without a value (const="")
        // This matches Python's argparse behavior with nargs="?" and const=""
        if let Some(prompt) = &self.prompt {
            if !prompt.is_empty() {
                return Ok(prompt.clone());
            }
            // Empty prompt provided (--prompt without value), try stdin
            if let Some(stdin_prompt) = self.get_stdin_prompt() {
                return Ok(stdin_prompt);
            }
            // No stdin input available either
            anyhow::bail!("No prompt provided for programmatic mode")
        }

        // No --prompt flag provided; caller decides which mode to run.
        Ok(String::new())
    }

    /// Try to get prompt from stdin if available.
    pub fn get_stdin_prompt(&self) -> Option<String> {
        if atty::is(atty::Stream::Stdin) {
            return None;
        }

        let mut buffer = String::new();
        if io::stdin().read_to_string(&mut buffer).is_ok() && !buffer.trim().is_empty() {
            return Some(buffer.trim().to_string());
        }

        None
    }
}
