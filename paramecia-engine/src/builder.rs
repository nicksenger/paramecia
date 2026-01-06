//! Builder for ModelEngine — consolidates model loading logic.

use crate::executor::{ModelEngine, ModelEngineInner};
use crate::model_actor::{spawn_model_actor, TrainingConfig};
use crate::types::Error;

use paramecia_core::Device;
use paramecia_model::generation::LogitsProcessor;
use paramecia_model::models::qwen3_next::{self, DeviceOffloadMode, KvCacheQuantization};
use paramecia_model::YarnConfig;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3-Next-80B-A3B-Instruct";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3.5-35B-A3B";

/// Builder for constructing a [`ModelEngine`].
pub struct ModelEngineBuilder {
    // Required
    model_path: PathBuf,
    // Optional with defaults
    tokenizer_path: Option<PathBuf>,
    device: Option<Device>,
    cpu: bool,
    offload_mode: DeviceOffloadMode,
    kv_cache_quant: KvCacheQuantization,
    yarn_config: Option<YarnConfig>,
    layer_split: Option<String>,
    prefetch: bool,
    // Sampling
    temperature: f64,
    top_p: Option<f64>,
    top_k: usize,
    tail_samples: usize,
    seed: u64,
    // Penalties
    repeat_penalty: f32,
    presence_penalty: f32,
    penalty_last_n: usize,
    // Directories
    snapshot_dir: PathBuf,
    initial_snapshot: Option<PathBuf>,
    // HF download settings
    model_id: Option<String>,
    model_file: Option<String>,
    tokenizer_repo: Option<String>,
    // MTP (Multi-Token Prediction)
    inference_mtp_speculation_depth: Option<usize>,
    training_mtp_speculation_depth: Option<usize>,
    mtp_allow_inference_override: bool,
    // Training
    training_config: TrainingConfig,
}

impl ModelEngineBuilder {
    /// Create a new builder with a model path.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: None,
            device: None,
            cpu: false,
            offload_mode: DeviceOffloadMode::ExpertsOnCpu,
            kv_cache_quant: KvCacheQuantization::Q8_0,
            yarn_config: None,
            layer_split: None,
            prefetch: true,
            temperature: 0.7,
            top_p: None,
            top_k: 64,
            tail_samples: 16,
            seed: 299792458,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            penalty_last_n: 128,
            snapshot_dir: PathBuf::from("/tmp/paramecia-snapshots"),
            initial_snapshot: None,
            model_id: None,
            model_file: None,
            tokenizer_repo: None,
            inference_mtp_speculation_depth: None,
            training_mtp_speculation_depth: None,
            mtp_allow_inference_override: false,
            training_config: TrainingConfig::default(),
        }
    }

    pub fn tokenizer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn cpu(mut self, cpu: bool) -> Self {
        self.cpu = cpu;
        self
    }

    pub fn offload_mode(mut self, mode: DeviceOffloadMode) -> Self {
        self.offload_mode = mode;
        self
    }

    pub fn kv_cache_quant(mut self, quant: KvCacheQuantization) -> Self {
        self.kv_cache_quant = quant;
        self
    }

    pub fn yarn_config(mut self, config: YarnConfig) -> Self {
        self.yarn_config = Some(config);
        self
    }

    pub fn layer_split(mut self, split: impl Into<String>) -> Self {
        self.layer_split = Some(split.into());
        self
    }

    pub fn prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn tail_samples(mut self, n: usize) -> Self {
        self.tail_samples = n;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = penalty;
        self
    }

    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }

    pub fn penalty_last_n(mut self, n: usize) -> Self {
        self.penalty_last_n = n;
        self
    }

    pub fn snapshot_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.snapshot_dir = dir.into();
        self
    }

    pub fn initial_snapshot(mut self, path: impl Into<PathBuf>) -> Self {
        self.initial_snapshot = Some(path.into());
        self
    }

    pub fn model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    pub fn model_file(mut self, file: impl Into<String>) -> Self {
        self.model_file = Some(file.into());
        self
    }

    pub fn tokenizer_repo(mut self, repo: impl Into<String>) -> Self {
        self.tokenizer_repo = Some(repo.into());
        self
    }

    /// Enable MTP speculative decoding for inference with a fixed speculation depth.
    ///
    /// Set `speculation_depth` to `0` to disable.
    pub fn enable_mtp_for_inference(mut self, speculation_depth: usize) -> Self {
        self.inference_mtp_speculation_depth = if speculation_depth > 0 {
            Some(speculation_depth)
        } else {
            None
        };
        self
    }

    /// Enable MTP loss for training with a fixed speculation depth.
    ///
    /// Set `speculation_depth` to `0` to disable.
    pub fn enable_mtp_for_training(mut self, speculation_depth: usize) -> Self {
        self.training_mtp_speculation_depth = if speculation_depth > 0 {
            Some(speculation_depth)
        } else {
            None
        };
        self
    }

    /// Allow overriding speculative MTP predictions via `commit_token`.
    ///
    /// When enabled, a mismatched commit during speculative decoding triggers
    /// a rollback+replay path to preserve correctness. Disabled by default.
    pub fn allow_mtp_inference_override(mut self, allow: bool) -> Self {
        self.mtp_allow_inference_override = allow;
        self
    }

    /// Set the training configuration for the unified actor.
    pub fn training_config(mut self, config: TrainingConfig) -> Self {
        self.training_config = config;
        self
    }

    /// Build the ModelEngine.
    pub fn build(self) -> Result<ModelEngine, Error> {
        // Select device
        let device = if let Some(d) = self.device {
            d
        } else if self.cpu {
            Device::Cpu
        } else {
            paramecia_model::select_best_device()
        };

        // Resolve model path (download from HF if needed)
        let model_path = if self.model_path.exists() {
            self.model_path.clone()
        } else if let (Some(model_id), Some(model_file)) = (&self.model_id, &self.model_file) {
            download_model(model_id, model_file)?
        } else {
            return Err(Error::ModelError(format!(
                "Model path does not exist: {:?}",
                self.model_path
            )));
        };

        // Load tokenizer
        let tokenizer = if let Some(path) = &self.tokenizer_path {
            Tokenizer::from_file(path).map_err(|e| Error::TokenizeError(e.to_string()))?
        } else if let Some(repo) = &self.tokenizer_repo {
            download_tokenizer(repo)?
        } else {
            // Try to find next to model
            let parent = model_path.parent().unwrap_or(std::path::Path::new("."));
            let tokenizer_path = parent.join("tokenizer.json");
            if tokenizer_path.exists() {
                Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| Error::TokenizeError(e.to_string()))?
            } else {
                // Default HF download
                download_tokenizer(DEFAULT_TOKENIZER_REPO)?
            }
        };

        // Load model weights
        let layer_split_str = self
            .layer_split
            .or_else(|| std::env::var("PARAMECIA_LAYER_SPLIT").ok());

        let offload_mode = if self.cpu {
            DeviceOffloadMode::FullGpu
        } else {
            self.offload_mode
        };

        let (mut model, device) = if let Some(ref split) = layer_split_str {
            let num_layers = read_num_layers(&model_path)?;
            let layer_device_map = qwen3_next::LayerDeviceMap::from_proportions(split, num_layers)
                .map_err(|e| Error::ModelError(e.to_string()))?;
            let primary = layer_device_map.primary_device().clone();
            let model = qwen3_next::ModelWeights::from_gguf_with_layer_split(
                &model_path,
                layer_device_map,
                offload_mode,
                self.kv_cache_quant,
                self.yarn_config,
            )
            .map_err(|e| Error::ModelError(e.to_string()))?;
            (model, primary)
        } else {
            let model = qwen3_next::ModelWeights::from_gguf_with_offload_and_yarn(
                &model_path,
                &device,
                offload_mode,
                self.kv_cache_quant,
                self.yarn_config,
            )
            .map_err(|e| Error::ModelError(e.to_string()))?;
            (model, device)
        };

        let tokenizer_max_id = tokenizer.get_vocab(true).values().copied().max();
        let model_vocab = model.vocab_size();
        if let Some(max_id) = tokenizer_max_id {
            if (max_id as usize) >= model_vocab {
                return Err(Error::TokenizeError(format!(
                    "Tokenizer token ID range mismatch: tokenizer max token id is {}, but model vocab size is {} (valid ids: 0..{}). \
Use a tokenizer that matches the loaded model.",
                    max_id,
                    model_vocab,
                    model_vocab.saturating_sub(1)
                )));
            }
        }

        // Enable prefetch pipeline
        if self.prefetch {
            model
                .enable_prefetch_pipeline()
                .map_err(|e| Error::ModelError(format!("Failed to enable prefetch: {e}")))?;
        }

        // Always capture expert indices (negligible overhead)
        model.set_capture_expert_indices(true);

        // Snapshot directory
        std::fs::create_dir_all(&self.snapshot_dir)
            .map_err(|e| Error::SnapshotError(format!("Failed to create snapshot dir: {e}")))?;

        // Load initial snapshot if specified
        let mut tokens = Vec::new();
        let mut state_position = 0;
        if let Some(ckpt_path) = &self.initial_snapshot {
            let snapshot = model
                .load_snapshot(ckpt_path)
                .map_err(|e| Error::SnapshotError(e.to_string()))?;
            tokens = snapshot.tokens;
            state_position = snapshot.state_position;
        }

        // Set up sampler
        let sampler = LogitsProcessor::new(
            self.seed,
            if self.temperature < 1e-7 {
                None
            } else {
                Some(self.temperature)
            },
            self.top_p,
        );

        let has_mtp = model.has_mtp();
        let num_layers = model.num_layers();
        let num_experts_per_token = model.num_experts_per_token();
        let handle_device = device.clone();
        let handle_model_path = model_path.clone();
        let handle_tokenizer = tokenizer.clone();

        let inner = ModelEngineInner {
            model,
            device,
            tokenizer,
            tokens,
            state_position,
            awaiting_commit: false,
            pending_logits: None,
            sampler,
            top_k: self.top_k,
            tail_samples: self.tail_samples,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            penalty_last_n: self.penalty_last_n,
            snapshot_dir: self.snapshot_dir,
            model_path,
            memory_snapshots: std::collections::HashMap::new(),
            next_snapshot_counter: 0,
            mtp_inference_speculation_depth: self.inference_mtp_speculation_depth,
            mtp_allow_inference_override: self.mtp_allow_inference_override,
            pending_speculative_predictions: std::collections::VecDeque::new(),
            pending_speculative_rollback: None,
            pending_speculative_commits: Vec::new(),
            pending_commit_expected_token: None,
            pending_commit_preappended: false,
            pending_metadata: Vec::new(),
        };

        // Set MTP speculation depth on the training config
        let mut training_config = self.training_config;
        if training_config.mtp_speculation_depth.is_none() {
            training_config.mtp_speculation_depth = self.training_mtp_speculation_depth;
        }

        let actor = std::sync::Arc::new(spawn_model_actor(inner, training_config));
        Ok(ModelEngine::new(
            actor,
            handle_device,
            handle_model_path,
            handle_tokenizer,
            has_mtp,
            num_layers,
            num_experts_per_token,
            self.inference_mtp_speculation_depth,
            self.training_mtp_speculation_depth,
            self.mtp_allow_inference_override,
        ))
    }
}

// --- Helper functions ---

fn download_model(model_id: &str, model_file: &str) -> Result<PathBuf, Error> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::ModelError(format!("HF API error: {e}")))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));
    repo.get(model_file)
        .map_err(|e| Error::ModelError(format!("Failed to download model: {e}")))
}

fn download_tokenizer(repo_id: &str) -> Result<Tokenizer, Error> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::TokenizeError(format!("HF API error: {e}")))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| Error::TokenizeError(format!("Failed to download tokenizer: {e}")))?;
    Tokenizer::from_file(tokenizer_path).map_err(|e| Error::TokenizeError(e.to_string()))
}

fn read_num_layers(model_path: &PathBuf) -> Result<usize, Error> {
    let mut file = std::fs::File::open(model_path)
        .map_err(|e| Error::ModelError(format!("Failed to open model: {e}")))?;
    let ct = paramecia_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| Error::ModelError(format!("Failed to read GGUF: {e}")))?;
    let md = &ct.metadata;
    md.get("qwen35moe.block_count")
        .or_else(|| md.get("qwen35.block_count"))
        .or_else(|| md.get("qwen3_5_moe.block_count"))
        .or_else(|| md.get("qwen3_5.block_count"))
        .or_else(|| md.get("qwen3next.block_count"))
        .or_else(|| md.get("qwen3moe.block_count"))
        .or_else(|| md.get("llama.block_count"))
        .and_then(|v| v.to_u32().ok())
        .map(|n| n as usize)
        .ok_or_else(|| Error::ModelError("Could not read num_layers from GGUF".into()))
}
