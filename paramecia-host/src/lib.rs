pub mod convert;
pub mod error;
pub mod host;
pub mod runtime;
pub mod stream_producer;

include!(concat!(env!("OUT_DIR"), "/paramecia_bindgen.rs"));

pub use error::HostError;
pub use host::HostState;
pub use paramecia_bridge::{ControllerBridge, ControllerHostEndpoint, create_bridge};
pub use runtime::run_host;

use anyhow::Result;
use host::ModelBuilderConfig;
use paramecia_engine::TrainingConfig;
use paramecia_model::DeviceOffloadMode;
use paramecia_model::models::qwen3_next::KvCacheQuantization;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use wasmtime::component::ResourceTable;
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtxBuilder};

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_MODEL_ID: &str = "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_MODEL_ID: &str = "unsloth/Qwen3.5-35B-A3B-GGUF";

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_MODEL_FILE: &str = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_MODEL_FILE: &str = "Qwen3.5-35B-A3B-Q4_K_M.gguf";

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3-Next-80B-A3B-Instruct";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3.5-35B-A3B";

/// Options for running a WASM controller component.
pub struct HostOptions {
    /// Path to the compiled WASM controller component.
    pub controller: String,

    /// Run on CPU rather than on GPU.
    pub cpu: bool,

    /// Device offload mode for MoE expert weights.
    ///
    /// Valid values: "none", "experts", "down", "updown".
    pub offload: String,

    /// Disable KV-cache quantization.
    pub no_kv_quant: bool,

    /// The model repository on HuggingFace Hub.
    pub model_id: String,

    /// The model file to download.
    pub model_file: String,

    /// The tokenizer repository.
    pub tokenizer_repo: String,

    /// Path to local model file (overrides HF download).
    pub model_path: Option<String>,

    /// Path to local tokenizer file (overrides HF download).
    pub tokenizer_path: Option<String>,

    /// Number of top-k logit entries to return in predictions.
    pub top_k: usize,

    /// Number of stratified tail samples to return.
    pub tail_samples: usize,

    /// Sampling temperature (0 = argmax/greedy).
    pub temperature: f64,

    /// Top-p (nucleus) sampling threshold.
    pub top_p: Option<f64>,

    /// Sampling seed.
    pub seed: u64,

    /// Directory for snapshot files.
    pub snapshot_dir: String,

    /// Multi-GPU layer split proportions (e.g., "3,1").
    pub layer_split: Option<String>,

    /// MTP speculation depth for inference (disabled when None or 0).
    pub n_speculative_inference: Option<usize>,

    /// MTP speculation depth for training loss (disabled when None or 0).
    pub n_speculative_training: Option<usize>,

    /// Allow overriding MTP speculative predictions at commit-time.
    pub commit_mtp_override: bool,

    /// Repeat penalty for sampling.
    pub repeat_penalty: f32,

    /// Presence penalty for sampling.
    pub presence_penalty: f32,

    /// Number of recent tokens for penalty window.
    pub penalty_last_n: usize,

    /// Number of samples per minibatch for QuZO steps.
    pub minibatch_size: usize,

    /// QuZO learning rate.
    pub training_lr: f64,

    /// QuZO perturbation epsilon.
    pub training_epsilon: f64,

    /// Directory for training checkpoints.
    pub checkpoint_dir: String,

    /// Number of gradient accumulation steps per training step.
    pub n_grad_steps: usize,

    /// Which tensors to optimize: "all", "attention", "qk".
    pub optimize_tensors: String,

    /// QuZO directional derivative clipping threshold.
    pub clip_threshold: f64,

    /// Load-balance loss weight for MoE routing.
    pub lb_loss: f64,

    /// Z-loss weight for MoE router stability.
    pub z_loss: f64,

    /// Interactive endpoint for connect() handler (enables interactive mode).
    pub interactive_endpoint: Option<ControllerHostEndpoint>,

    /// Suppress stdout logging (for embedding in a TUI).
    pub quiet: bool,
}

impl Default for HostOptions {
    fn default() -> Self {
        Self {
            controller: String::new(),
            cpu: false,
            offload: "experts".to_string(),
            no_kv_quant: false,
            model_id: DEFAULT_MODEL_ID.to_string(),
            model_file: DEFAULT_MODEL_FILE.to_string(),
            tokenizer_repo: DEFAULT_TOKENIZER_REPO.to_string(),
            model_path: None,
            tokenizer_path: None,
            top_k: 64,
            tail_samples: 16,
            temperature: 0.0,
            top_p: None,
            seed: 42,
            snapshot_dir: "/tmp/paramecia-snapshots".to_string(),
            layer_split: None,
            n_speculative_inference: None,
            n_speculative_training: None,
            commit_mtp_override: false,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
            minibatch_size: 2,
            training_lr: 0.0001,
            training_epsilon: 0.001,
            checkpoint_dir: "/tmp/paramecia-checkpoints".to_string(),
            n_grad_steps: 2,
            optimize_tensors: "all".to_string(),
            clip_threshold: 10.0,
            lb_loss: 0.01,
            z_loss: 0.001,
            interactive_endpoint: None,
            quiet: false,
        }
    }
}

/// Run a WASM controller component with the given options.
///
/// Models are loaded on-demand when the guest calls `load-model`.
/// This function sets up the device, resolves the default model path,
/// compiles the WASM controller, and runs it to completion.
pub fn run_host_cli(options: HostOptions) -> Result<()> {
    let quiet = options.quiet;
    macro_rules! log {
        ($($arg:tt)*) => { if !quiet { println!($($arg)*); } }
    }

    log!("Paramecia Host");
    log!("====================\n");

    let offload_mode = DeviceOffloadMode::parse(&options.offload, options.cpu);
    let kv_cache_quant = if options.no_kv_quant {
        KvCacheQuantization::F16
    } else {
        KvCacheQuantization::Q8_0
    };

    // Select device upfront
    let device = if options.cpu {
        paramecia_core::Device::Cpu
    } else {
        paramecia_model::select_best_device()
    };
    log!("Device: {:?}", device);

    // Resolve default model path (download from HF if needed)
    let default_model_path = if let Some(ref path) = options.model_path {
        let p = PathBuf::from(path);
        if p.exists() {
            p
        } else {
            anyhow::bail!("Model path does not exist: {path}");
        }
    } else {
        log!(
            "Downloading model from HF: {}/{}",
            options.model_id,
            options.model_file
        );
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            options.model_id.clone(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        repo.get(&options.model_file)?
    };
    log!("Default model: {}", default_model_path.display());

    // Snapshot/checkpoint directories
    let snapshot_dir = PathBuf::from(&options.snapshot_dir);
    let checkpoint_dir = PathBuf::from(&options.checkpoint_dir);
    std::fs::create_dir_all(&snapshot_dir)?;
    std::fs::create_dir_all(&checkpoint_dir)?;

    // Build ModelBuilderConfig — used by load-model to create ModelEngines
    let builder_config = ModelBuilderConfig {
        tokenizer_path: options.tokenizer_path.map(PathBuf::from),
        tokenizer_repo: options.tokenizer_repo.clone(),
        cpu: options.cpu,
        offload_mode,
        kv_cache_quant,
        layer_split: options.layer_split.clone(),
        prefetch: true,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        tail_samples: options.tail_samples,
        seed: options.seed,
        repeat_penalty: options.repeat_penalty,
        presence_penalty: options.presence_penalty,
        penalty_last_n: options.penalty_last_n,
        snapshot_dir: snapshot_dir.clone(),
        inference_mtp_speculation_depth: options.n_speculative_inference,
        training_mtp_speculation_depth: options.n_speculative_training,
        mtp_allow_inference_override: options.commit_mtp_override,
        training_config: TrainingConfig {
            minibatch_size: options.minibatch_size,
            n_grad_steps: options.n_grad_steps,
            lr: options.training_lr,
            epsilon: options.training_epsilon,
            optimize_tensors: options.optimize_tensors.clone(),
            mtp_speculation_depth: options.n_speculative_training,
            mtp_decay_factor: 0.5,
            checkpoint_dir: checkpoint_dir.clone(),
            clip_threshold: options.clip_threshold,
            lb_loss: options.lb_loss,
            z_loss: options.z_loss,
        },
    };

    // Build WASI context
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_env()
        .preopened_dir(".", ".", DirPerms::all(), FilePerms::all())
        .expect("failed to preopen current directory")
        .build();

    // Compile WASM controller component
    log!("\nCompiling WASM controller: {}", options.controller);
    let engine = runtime::create_engine()?;
    let component = runtime::compile_component(&engine, std::path::Path::new(&options.controller))?;
    let linker = runtime::create_linker(&engine)?;

    log!("Running host...\n");

    let state = HostState {
        models: HashMap::new(),
        checkpoints: HashMap::new(),
        snapshots: HashMap::new(),
        persisted_snapshots: HashMap::new(),
        prediction_cancels: Arc::new(Mutex::new(HashMap::new())),
        training_cancels: Arc::new(Mutex::new(HashMap::new())),
        default_model_path,
        snapshot_dir,
        checkpoint_dir,
        device,
        builder_config,
        next_uuid: 0,
        wasi,
        table: ResourceTable::new(),
        interactive_endpoint: options.interactive_endpoint,
    };

    // Run controller on a tokio runtime.
    // When called from within an existing runtime (e.g. via #[tokio::main]),
    // we cannot create a new Runtime on the same thread. Use the existing
    // runtime's Handle::block_on from a fresh thread instead.
    let run_async = || async { run_host(&engine, &linker, &component, state).await };

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        std::thread::scope(|s| {
            s.spawn(|| handle.block_on(run_async()))
                .join()
                .expect("host thread panicked")
        })?;
    } else {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_async())?;
    }

    log!("\nHost finished.");
    Ok(())
}
