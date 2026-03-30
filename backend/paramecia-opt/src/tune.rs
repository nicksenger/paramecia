//! Knowledge distillation training entry point.
//!
//! Provides `TuneOptions` and `run_distillation()` — the high-level orchestrator
//! for offline distillation training from tuning data files.

use paramecia_core::Result;
use paramecia_model::models::qwen3_next::{
    select_best_device, DeviceOffloadMode, KvCacheQuantization, LayerDeviceMap, ModelWeights,
};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};

use crate::{
    save_trained_model, DistillationLossConfig, DistillationTrainer, DistillationTrainerConfig,
    EpsilonConfig, MtpLossConfig, OptimizeTensors, TuningDataset,
};

/// Options for knowledge distillation training.
pub struct TuneOptions {
    /// Path to input GGUF model file
    pub model_path: PathBuf,
    /// Path to tuning data (.bin file or directory containing .bin files)
    pub tuning_data: PathBuf,
    /// Path for output GGUF with trained weights
    pub output_path: PathBuf,
    /// Number of training steps
    pub steps: usize,
    /// Minibatch size (conversations processed in parallel)
    pub minibatch_size: usize,
    /// Number of gradient accumulation steps
    pub n_grad_steps: usize,
    /// Maximum sequence length (conversations longer than this will be truncated)
    pub max_sequence_length: Option<usize>,
    /// Chunk size for processing long sequences
    pub chunk_size: usize,
    /// QuZO learning rate
    pub lr: f64,
    /// QuZO perturbation magnitude (base epsilon)
    pub epsilon: f64,
    /// Epsilon multiplier for embedding layer (token_embd)
    pub eps_embedding: f64,
    /// Epsilon multiplier for attention Q/K/V/O projections
    pub eps_attention: f64,
    /// Epsilon multiplier for MoE router/gating layers
    pub eps_moe_gating: f64,
    /// Epsilon multiplier for MoE expert layers
    pub eps_moe_experts: f64,
    /// Epsilon multiplier for MTP (Multi-Token Prediction) heads
    pub eps_mtp: f64,
    /// Epsilon multiplier for other layers (SSM, norms, output)
    pub eps_other: f64,
    /// Number of perturbation samples per step
    pub num_samples: usize,
    /// Gradient clipping threshold
    pub clip_threshold: f64,
    /// Weight for Z-loss (router stability)
    pub z_loss_weight: f64,
    /// Weight for load balance loss (expert utilization)
    pub lb_loss_weight: f64,
    /// Temperature for KL divergence (1.0 = no softening)
    pub temperature: f64,
    /// Checkpoint every N steps (0 = no checkpointing)
    pub checkpoint_every: usize,
    /// Random seed for reproducibility (random if not specified)
    pub seed: Option<u64>,
    /// Shuffle training data each epoch
    pub shuffle: bool,
    /// Log progress every N steps
    pub log_every: usize,
    /// Device offload mode: "none", "experts", "down", "updown"
    pub offload: String,
    /// Which tensors to optimize: "all", "attention", "qk"
    pub optimize: String,
    /// Dry run: train but don't save output model (useful for testing)
    pub dry_run: bool,
    /// Show verbose output
    pub verbose: bool,
    /// Number of speculative tokens for MTP loss (0 = disabled)
    pub num_speculative: usize,
    /// MTP weight decay factor
    pub mtp_decay: f64,
    /// Generate perturbation tensors lazily to reduce peak memory
    pub lazy_perturbations: bool,
    /// Multi-GPU layer split proportions (e.g., "3,1" = 75% GPU 0, 25% GPU 1)
    pub layer_split: Option<String>,
}

fn parse_optimize_mode(mode: &str) -> OptimizeTensors {
    match mode.to_lowercase().as_str() {
        "all" => OptimizeTensors::All,
        "attention" => OptimizeTensors::AttentionOnly,
        "qk" => OptimizeTensors::AttentionQKOnly,
        _ => {
            warn!("Unknown optimize mode '{}', using 'all'", mode);
            OptimizeTensors::All
        }
    }
}

/// Run knowledge distillation training with the given options.
pub fn run_distillation(options: &TuneOptions) -> Result<()> {
    info!("QuZO Knowledge Distillation Training");
    info!("=====================================");

    // Get device
    let device = select_best_device();
    info!("Device: {:?}", device);

    let offload_mode = DeviceOffloadMode::parse(&options.offload, false);
    info!("Offload mode: {:?}", offload_mode);

    // Load model in training mode
    info!("Loading model from {:?}...", options.model_path);
    let load_start = Instant::now();

    let (mut model, device) = if let Some(ref split) = options.layer_split {
        // Multi-GPU: peek at GGUF to get layer count, build device map
        let mut peek_file = std::fs::File::open(&options.model_path)?;
        let peek_ct = paramecia_core::quantized::gguf_file::Content::read(&mut peek_file)?;
        let num_layers = peek_ct
            .metadata
            .get("qwen3next.block_count")
            .or_else(|| peek_ct.metadata.get("qwen3moe.block_count"))
            .or_else(|| peek_ct.metadata.get("llama.block_count"))
            .ok_or_else(|| {
                paramecia_core::Error::Msg("Cannot find block_count in GGUF metadata".into())
            })?
            .to_u32()? as usize;
        let layer_device_map = LayerDeviceMap::from_proportions(split, num_layers)?;
        info!(
            "Multi-GPU training: {} GPUs, {} layers",
            layer_device_map.num_gpus(),
            num_layers
        );
        let primary = layer_device_map.primary_device().clone();
        let model = ModelWeights::from_gguf_for_training_with_layer_split(
            &options.model_path,
            layer_device_map,
            offload_mode,
            KvCacheQuantization::Q8_0,
        )?;
        (model, primary)
    } else {
        let model = ModelWeights::from_gguf_for_training_with_offload(
            &options.model_path,
            &device,
            offload_mode,
            KvCacheQuantization::Q8_0,
        )?;
        (model, device)
    };

    info!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f32());

    // Enable prefetch pipeline
    model.enable_prefetch_pipeline()?;
    if model.has_prefetch_pipeline() {
        info!("Prefetch pipeline enabled for training");
    }

    // Generate random seed if not specified
    let seed = options.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    info!("Using seed: {}", seed);

    // Load tuning dataset
    info!("Loading tuning dataset from {:?}...", options.tuning_data);
    let mut dataset = TuningDataset::from_directory(&options.tuning_data, options.shuffle, seed)?;
    info!("Found {} tuning files", dataset.len());

    // Configure trainer
    let loss_config = DistillationLossConfig {
        distill_weight: 1.0,
        z_loss_weight: options.z_loss_weight,
        lb_loss_weight: options.lb_loss_weight,
        temperature: options.temperature,
        min_prob: 1e-10,
    };

    let optimize_tensors = parse_optimize_mode(&options.optimize);
    info!("Optimize tensors: {:?}", optimize_tensors);

    let epsilon_config = EpsilonConfig {
        embedding: options.eps_embedding,
        attention: options.eps_attention,
        moe_gating: options.eps_moe_gating,
        moe_experts: options.eps_moe_experts,
        moe_expert_banks: options.eps_moe_experts, // Use same default as moe_experts
        mtp: options.eps_mtp,
        norms: 0.5,
        ssm: 1.0,
        other: options.eps_other,
    };

    // Configure MTP
    let mtp_config = if options.num_speculative > 0 && model.has_mtp() {
        info!(
            "MTP enabled: {} speculative depths, decay={}",
            options.num_speculative, options.mtp_decay
        );
        Some(MtpLossConfig {
            num_depths: options.num_speculative,
            decay_factor: options.mtp_decay,
            normalize_weights: true,
        })
    } else {
        if options.num_speculative > 0 && !model.has_mtp() {
            info!("Note: --num-speculative set but model has no MTP support, using standard path");
        }
        None
    };

    let trainer_config = DistillationTrainerConfig {
        num_steps: options.steps,
        minibatch_size: options.minibatch_size,
        n_grad_steps: options.n_grad_steps,
        max_sequence_length: options.max_sequence_length,
        chunk_size: options.chunk_size,
        lr: options.lr,
        epsilon: options.epsilon,
        epsilon_config,
        num_samples: options.num_samples,
        clip_threshold: options.clip_threshold,
        seed,
        loss_config,
        mtp_config,
        checkpoint_every: options.checkpoint_every,
        checkpoint_dir: None,
        shuffle: options.shuffle,
        log_every: options.log_every,
        optimize_tensors,
        lazy_perturbations: options.lazy_perturbations,
    };

    let mut trainer = DistillationTrainer::new(trainer_config);

    if options.dry_run {
        warn!("[DRY RUN] Training will proceed but model will NOT be saved.");
    }

    info!("Starting training...");
    let train_start = Instant::now();

    let checkpoint = trainer.train(&mut model, &mut dataset, &device)?;

    let train_duration = train_start.elapsed();
    info!(
        "Training completed in {:.1}s ({:.2} steps/sec)",
        train_duration.as_secs_f32(),
        options.steps as f32 / train_duration.as_secs_f32()
    );

    let avg_loss = checkpoint.total_loss / checkpoint.step.max(1) as f64;
    let best_loss = checkpoint.best_loss;

    // Save trained model (unless dry run)
    if options.dry_run {
        info!("[DRY RUN] Skipping model save.");
    } else {
        info!("Saving trained model...");
        let save_start = Instant::now();

        save_trained_model(&model, &options.model_path, &options.output_path, None)?;

        info!("Model saved in {:.1}s", save_start.elapsed().as_secs_f32());
    }

    // Summary
    info!("=====================================");
    info!("Training Summary");
    info!("=====================================");
    info!("  Steps completed: {}", checkpoint.step);
    info!("  Epochs: {}", checkpoint.epoch);
    info!("  Initial loss: {:.4}", checkpoint.initial_loss);
    info!("  Final loss:   {:.4}", checkpoint.final_loss);
    info!("  Best loss:    {:.4}", best_loss);
    info!("  Average loss: {:.4}", avg_loss);

    let initial = checkpoint.initial_loss;
    let final_loss = checkpoint.final_loss;

    if !initial.is_nan() && !final_loss.is_nan() && initial > 0.0 {
        let change = final_loss - initial;
        let change_pct = (change / initial) * 100.0;

        if change < 0.0 {
            info!("  Change: {:.4} ({:.1}%)", change, change_pct);
            info!("  Status: IMPROVED");
        } else if change > 0.0 {
            info!("  Change: +{:.4} (+{:.1}%)", change, change_pct);
            info!("  Status: REGRESSED");
        } else {
            info!("  Change: 0.0 (0.0%)");
            info!("  Status: UNCHANGED");
        }
    }

    if options.dry_run {
        info!("  [DRY RUN] Model was NOT saved to disk.");
    } else {
        info!("  Output: {:?}", options.output_path);
    }

    Ok(())
}
