//! Main distillation trainer using QuZO optimization.
//!
//! Handles chunked processing of long sequences, batch accumulation across
//! conversations, and gradient estimation using central differences.

use paramecia_core::quantized::SharedQTensor;
use paramecia_core::{Device, Result, Tensor};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info};

use paramecia_model::ModelWeights;

use super::{
    DistillationLoss, DistillationLossConfig, MtpLossConfig, PaddedTuningData, TuningData,
    TuningDataset,
};
use crate::{ParamsQuZO, QuZO};

/// Which tensors to optimize during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizeTensors {
    /// Optimize all quantized tensors (slowest, ~301 for 48-layer MoE)
    #[default]
    All,
    /// Optimize only attention weights: Q, K, V, O projections (~192 for 48 layers)
    AttentionOnly,
    /// Optimize only attention Q and K projections (~96 for 48 layers)
    AttentionQKOnly,
}

/// Per-component epsilon multipliers for QuZO optimization.
///
/// Different model components benefit from different perturbation magnitudes:
/// - Embedding layers: small perturbations (sensitive to changes)
/// - Attention Q/K/V: standard perturbations
/// - MoE gating: smaller for stability (routers are sensitive)
/// - MoE experts: larger for impact (experts have more parameters)
/// - MTP heads: moderate perturbations
#[derive(Debug, Clone)]
pub struct EpsilonConfig {
    /// Multiplier for embedding layer (token_embd). Default: 0.1
    pub embedding: f64,
    /// Multiplier for attention Q/K/V/O projections. Default: 1.0
    pub attention: f64,
    /// Multiplier for MoE router/gating layers. Default: 0.5
    pub moe_gating: f64,
    /// Multiplier for MoE shared expert projections (ffn_*_shexp). Default: 2.0
    pub moe_experts: f64,
    /// Multiplier for MoE expert bank tensors (ffn_*_exps). Default: 2.0
    pub moe_expert_banks: f64,
    /// Multiplier for MTP (Multi-Token Prediction) heads. Default: 1.5
    pub mtp: f64,
    /// Multiplier for all norm weights (incl. MTP & SSM norms). Default: 0.5
    pub norms: f64,
    /// Multiplier for SSM parameters (ssm_a, ssm_conv1d, ssm_dt). Default: 1.0
    pub ssm: f64,
    /// Multiplier for all other layers (output, etc). Default: 1.0
    pub other: f64,
}

impl Default for EpsilonConfig {
    fn default() -> Self {
        Self {
            embedding: 0.1,
            attention: 1.0,
            moe_gating: 0.5,
            moe_experts: 2.0,
            moe_expert_banks: 2.0,
            mtp: 1.5,
            norms: 0.5,
            ssm: 1.0,
            other: 1.0,
        }
    }
}

impl EpsilonConfig {
    /// Get the epsilon multiplier for a tensor by its name.
    pub fn multiplier_for(&self, name: &str) -> f64 {
        // Embedding layer
        if name == "token_embd.weight" {
            return self.embedding;
        }

        // Norm weights (catches MTP norms, SSM norm, any other norms)
        // Must be checked before ssm_ so ssm_norm goes to norms, not ssm
        if name.contains("_norm") {
            return self.norms;
        }

        // SSM parameters (ssm_a, ssm_conv1d, ssm_dt, ssm_ba, ssm_out, ssm_in)
        if name.contains("ssm_") {
            return self.ssm;
        }

        // MTP heads (check before attention since mtp contains attn)
        if name.contains(".mtp.") || name.starts_with("mtp.") {
            return self.mtp;
        }

        // Attention Q/K/V/O projections
        if name.contains("attn_q")
            || name.contains("attn_k")
            || name.contains("attn_v")
            || name.contains("attn_output")
        {
            return self.attention;
        }

        // MoE gating (router)
        if name.contains("ffn_gate_inp") {
            return self.moe_gating;
        }

        // MoE expert banks (3D bank tensors: ffn_*_exps)
        if name.contains("ffn_gate_exps")
            || name.contains("ffn_up_exps")
            || name.contains("ffn_down_exps")
        {
            return self.moe_expert_banks;
        }

        // MoE shared expert projections (ffn_*_shexp)
        if name.contains("ffn_gate_shexp")
            || name.contains("ffn_up_shexp")
            || name.contains("ffn_down_shexp")
        {
            return self.moe_experts;
        }

        // Everything else (output, etc)
        self.other
    }
}

/// Configuration for the distillation trainer.
#[derive(Debug, Clone)]
pub struct DistillationTrainerConfig {
    /// Number of training steps
    pub num_steps: usize,
    /// Minibatch size (conversations processed in parallel, actual batch dim)
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
    /// Per-component epsilon multipliers
    pub epsilon_config: EpsilonConfig,
    /// Number of perturbation samples per step
    pub num_samples: usize,
    /// Gradient clipping threshold
    pub clip_threshold: f64,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Loss configuration
    pub loss_config: DistillationLossConfig,
    /// MTP (Multi-Token Prediction) loss configuration
    /// When Some and model has MTP, uses MTP for multi-token gradient signal
    pub mtp_config: Option<MtpLossConfig>,
    /// Checkpoint every N steps (0 = no checkpointing)
    pub checkpoint_every: usize,
    /// Path for checkpoint files
    pub checkpoint_dir: Option<PathBuf>,
    /// Whether to shuffle training data each epoch
    pub shuffle: bool,
    /// Log progress every N steps
    pub log_every: usize,
    /// Which tensors to optimize (affects training speed)
    pub optimize_tensors: OptimizeTensors,
    /// Generate perturbation tensors lazily to save memory.
    /// Recommended on Apple Silicon (unified memory). See ParamsQuZO::lazy_perturbations.
    pub lazy_perturbations: bool,
}

impl Default for DistillationTrainerConfig {
    fn default() -> Self {
        Self {
            num_steps: 1000,
            minibatch_size: 2,
            n_grad_steps: 2,
            max_sequence_length: None, // No limit by default
            chunk_size: 2048,
            lr: 1e-4,
            epsilon: 1e-3,
            epsilon_config: EpsilonConfig::default(),
            num_samples: 1,
            // clip_threshold should be scaled based on expected loss magnitude
            // For KL-divergence losses in the 10-50 range, 10.0 is a good starting point
            clip_threshold: 10.0,
            seed: 42,
            loss_config: DistillationLossConfig::default(),
            mtp_config: Some(MtpLossConfig::default()),
            checkpoint_every: 100,
            checkpoint_dir: None,
            shuffle: true,
            log_every: 1,
            optimize_tensors: OptimizeTensors::All,
            lazy_perturbations: false,
        }
    }
}

/// Training checkpoint for resume capability.
#[derive(Debug, Clone)]
pub struct TrainingCheckpoint {
    pub step: usize,
    pub epoch: usize,
    pub completed_files: HashSet<PathBuf>,
    pub total_loss: f64,
    pub best_loss: f64,
    /// Loss from the first training step (baseline for comparison)
    pub initial_loss: f64,
    /// Loss from the most recent step
    pub final_loss: f64,
}

impl TrainingCheckpoint {
    pub fn new() -> Self {
        Self {
            step: 0,
            epoch: 0,
            completed_files: HashSet::new(),
            total_loss: 0.0,
            best_loss: f64::INFINITY,
            initial_loss: f64::NAN,
            final_loss: f64::NAN,
        }
    }
}

impl Default for TrainingCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Training statistics for a single step.
#[derive(Debug, Clone)]
pub struct StepStats {
    pub step: usize,
    pub loss: f32,
    pub distill_loss: f32,
    pub z_loss: f32,
    pub lb_loss: f32,
    pub tokens_processed: usize,
    pub chunks_processed: usize,
    pub duration_secs: f32,
}

/// Main distillation trainer.
pub struct DistillationTrainer {
    config: DistillationTrainerConfig,
    loss_fn: DistillationLoss,
    checkpoint: TrainingCheckpoint,
}

impl DistillationTrainer {
    pub fn new(config: DistillationTrainerConfig) -> Self {
        let loss_fn = DistillationLoss::new(config.loss_config.clone());
        Self {
            config,
            loss_fn,
            checkpoint: TrainingCheckpoint::new(),
        }
    }

    /// Run the full training loop.
    ///
    /// # Arguments
    /// * `model` - Model loaded in training mode
    /// * `dataset` - Dataset of tuning files
    /// * `device` - Device for computation
    ///
    /// # Returns
    /// Final training statistics
    pub fn train(
        &mut self,
        model: &mut ModelWeights,
        dataset: &mut TuningDataset,
        device: &Device,
    ) -> Result<TrainingCheckpoint> {
        use paramecia_core::quantized::GgmlDType;

        // Get trainable tensors
        let quzo_tensors = model.quzo_qtensors();
        if quzo_tensors.is_empty() {
            return Err(paramecia_core::Error::Msg(
                "No trainable tensors found. Ensure model is loaded in training mode.".to_string(),
            ));
        }

        info!("Found {} trainable tensors", quzo_tensors.len());
        for (name, qt) in &quzo_tensors {
            let qt_read = qt.read().unwrap();
            debug!(
                "  [shared] {} : {:?} {:?}",
                name,
                qt_read.shape(),
                qt_read.dtype()
            );
        }

        // Filter to only include formats supported by QuZO
        // Supported: F32 (unquantized), BF16, Q2K, Q3K, Q4_0, Q4K, Q5K, Q6K, Q8_0, Q8K
        // NOT supported: F16
        let supported_dtypes = [
            GgmlDType::F32,
            GgmlDType::BF16,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
            GgmlDType::Q8K,
        ];

        // Filter by dtype first, also exclude pruning metadata tensors
        let dtype_filtered: Vec<_> = quzo_tensors
            .iter()
            .filter(|(name, qt)| {
                // Skip expert_mask and expert_remap tensors (pruning metadata, not trainable)
                if name.contains("expert_mask") || name.contains("expert_remap") {
                    debug!("  Skipping {} (pruning metadata)", name);
                    return false;
                }

                let qt_read = qt.read().unwrap();
                let dtype = qt_read.dtype();
                let supported = supported_dtypes.contains(&dtype);
                if !supported {
                    debug!("  Skipping {} ({:?} not supported by QuZO)", name, dtype);
                }
                supported
            })
            .collect();

        // Then filter by tensor type based on config, keeping names for epsilon multipliers
        let filtered_with_names: Vec<(String, SharedQTensor)> = dtype_filtered
            .into_iter()
            .filter(|(name, _)| {
                match self.config.optimize_tensors {
                    OptimizeTensors::All => true,
                    OptimizeTensors::AttentionOnly => {
                        // Match attention weight patterns: attn_q, attn_k, attn_v, attn_output
                        name.contains("attn_q")
                            || name.contains("attn_k")
                            || name.contains("attn_v")
                            || name.contains("attn_output")
                    }
                    OptimizeTensors::AttentionQKOnly => {
                        // Only Q and K projections
                        name.contains("attn_q") || name.contains("attn_k")
                    }
                }
            })
            .map(|(name, qt)| (name.clone(), qt.clone()))
            .collect();

        // Compute epsilon multipliers for each tensor
        let epsilon_multipliers: Vec<f64> = filtered_with_names
            .iter()
            .map(|(name, _)| self.config.epsilon_config.multiplier_for(name))
            .collect();

        // Extract just the tensors for the optimizer
        let optimizer_tensors: Vec<SharedQTensor> =
            filtered_with_names.into_iter().map(|(_, qt)| qt).collect();

        if optimizer_tensors.is_empty() {
            return Err(paramecia_core::Error::Msg(
                "No QuZO-compatible tensors found. Model may not have quantized weights."
                    .to_string(),
            ));
        }

        let tensor_type = match self.config.optimize_tensors {
            OptimizeTensors::All => "all quantized",
            OptimizeTensors::AttentionOnly => "attention Q/K/V/O",
            OptimizeTensors::AttentionQKOnly => "attention Q/K only",
        };
        info!(
            "Optimizing {} tensors ({})",
            optimizer_tensors.len(),
            tensor_type
        );

        // Create QuZO optimizer
        let params = ParamsQuZO {
            lr: self.config.lr,
            epsilon: self.config.epsilon,
            num_samples: self.config.num_samples,
            clip_threshold: self.config.clip_threshold,
            // Always use two-sided (central) differences for large MoE models
            // This provides stability for router/gate layers and DeltaNet states
            use_fused: true,
            perturbation_mode: crate::PerturbationMode::Weight,
            epsilon_multipliers: Some(epsilon_multipliers),
            lazy_perturbations: self.config.lazy_perturbations,
            error_feedback: None,
            error_decay: 0.9,
            error_gain: 1.0,
        };

        let mut optimizer = QuZO::new_with_seed(optimizer_tensors, params, self.config.seed)?;

        info!("Training Configuration:");
        info!("  Steps: {}", self.config.num_steps);
        info!(
            "  Minibatch size: {} conversations",
            self.config.minibatch_size
        );
        info!("  Gradient steps: {}", self.config.n_grad_steps);
        let effective_batch_size = self.config.minibatch_size * self.config.n_grad_steps;
        info!("  Effective batch: {} conversations", effective_batch_size);
        if let Some(max_len) = self.config.max_sequence_length {
            info!("  Max sequence length: {} tokens", max_len);
        } else {
            info!("  Max sequence length: unlimited");
        }
        info!("  Chunk size: {} tokens", self.config.chunk_size);
        info!("  Learning rate: {}", self.config.lr);
        info!("  Epsilon: {}", self.config.epsilon);
        info!("  Z-loss weight: {}", self.config.loss_config.z_loss_weight);
        info!(
            "  LB-loss weight: {}",
            self.config.loss_config.lb_loss_weight
        );
        info!("  Dataset: {} files", dataset.len());

        let total_batches_per_epoch = dataset.len().div_ceil(effective_batch_size);
        let estimated_epochs = self.config.num_steps.div_ceil(total_batches_per_epoch);
        info!(
            "  Estimated epochs: {} ({} batches/epoch)",
            estimated_epochs, total_batches_per_epoch
        );

        // Show detailed timing breakdown for first few steps to help diagnose performance
        info!("Note: Detailed timing shown for first 3 steps to diagnose performance.");
        info!(
            "{:>6} | {:>10} | {:>8} | {:>8}",
            "Step", "Loss", "Tokens", "Time"
        );
        info!("{}", "-".repeat(50));

        let training_start = Instant::now();

        for step in self.checkpoint.step..self.config.num_steps {
            let step_start = Instant::now();

            // Gradient accumulation loop
            let mut accumulated_loss = 0.0f32;
            let mut accumulated_positions = 0usize;
            let mut step_tokens = 0usize;

            for _grad_step in 0..self.config.n_grad_steps {
                // Load minibatch with dynamic length grouping
                let minibatch = dataset.next_batch_grouped(
                    self.config.minibatch_size,
                    0.5, // 50% length tolerance
                    self.config.max_sequence_length,
                )?;

                if minibatch.is_empty() {
                    continue;
                }

                // Count positions for proper loss weighting
                let positions_count: usize = minibatch.iter().map(|c| c.n_assistant_tokens()).sum();

                // QuZO step (does ±ε perturbation internally)
                let loss_val = optimizer.step(|| {
                    let (loss, _) = self.compute_minibatch_loss(model, &minibatch, device)?;
                    Ok(loss)
                })?;

                accumulated_loss += loss_val * positions_count as f32;
                accumulated_positions += positions_count;
                step_tokens += minibatch.iter().map(|c| c.seq_len()).sum::<usize>();
            }

            let final_loss = if accumulated_positions > 0 {
                accumulated_loss / accumulated_positions as f32
            } else {
                0.0f32
            };

            let step_duration = step_start.elapsed().as_secs_f32();

            self.checkpoint.step = step + 1;
            self.checkpoint.total_loss += final_loss as f64;
            self.checkpoint.final_loss = final_loss as f64;

            // Track initial loss from first step
            if step == 0 || self.checkpoint.initial_loss.is_nan() {
                self.checkpoint.initial_loss = final_loss as f64;
            }

            if final_loss < self.checkpoint.best_loss as f32 {
                self.checkpoint.best_loss = final_loss as f64;
            }

            // Log progress
            if (step + 1) % self.config.log_every == 0 || step == 0 {
                let tokens_per_sec = step_tokens as f32 / step_duration;
                info!(
                    "{:>6} | {:>10.4} | {:>8} | {:>7.1}s ({:.0} tok/s)",
                    step + 1,
                    final_loss,
                    step_tokens,
                    step_duration,
                    tokens_per_sec
                );
            }

            // Checkpoint
            if self.config.checkpoint_every > 0 && (step + 1) % self.config.checkpoint_every == 0 {
                let (files_done, total_files, epoch) = dataset.progress();
                self.checkpoint.epoch = epoch;
                info!(
                    "  [Checkpoint] Step {}, Epoch {}, Files {}/{}, Best loss: {:.4}",
                    step + 1,
                    epoch,
                    files_done,
                    total_files,
                    self.checkpoint.best_loss
                );
            }
        }

        let total_duration = training_start.elapsed();
        info!("Training complete!");
        info!("  Total time: {:.1}s", total_duration.as_secs_f32());
        info!("  Final epoch: {}", self.checkpoint.epoch);
        info!(
            "  Average loss: {:.4}",
            self.checkpoint.total_loss / self.config.num_steps as f64
        );
        info!("  Best loss: {:.4}", self.checkpoint.best_loss);

        Ok(self.checkpoint.clone())
    }

    /// Compute weighted target embeddings for MTP batching.
    /// Returns Vec of tensors, one per depth, each shaped [batch, chunk_len, hidden].
    fn compute_batched_mtp_embeddings(
        &self,
        model: &ModelWeights,
        padded_conversations: &[PaddedTuningData],
        chunk_start: usize,
        chunk_size: usize,
        num_depths: usize,
        device: &Device,
    ) -> Result<Vec<Tensor>> {
        let batch_size = padded_conversations.len();
        let embed_weights = model.embedding_weights();
        let hidden_size = embed_weights.dim(1)?;

        let mut weighted_embeds_per_depth = Vec::with_capacity(num_depths);

        for depth in 0..num_depths {
            let embed_shift = depth + 1;
            let mut batch_embeds = Vec::with_capacity(batch_size);

            for padded_conv in padded_conversations {
                let (chunk_tokens, _, chunk_padding) =
                    padded_conv.get_chunk(chunk_start, chunk_size);
                let chunk_len = chunk_tokens.len();

                // Build mapping from global position to assistant data index
                let mut global_to_data_idx = vec![None; padded_conv.original.seq_len()];
                let mut data_idx = 0;
                for (global_idx, &is_asst) in padded_conv.padded_assistant_mask.iter().enumerate() {
                    if is_asst && !padded_conv.padding_mask[global_idx] {
                        global_to_data_idx[global_idx] = Some(data_idx);
                        data_idx += 1;
                    }
                }

                // Build embeddings row by row
                let mut row_embeds = Vec::with_capacity(chunk_len);
                for (rel_pos, &_token_id) in chunk_tokens.iter().enumerate() {
                    let weighted_embed = if chunk_padding[rel_pos] {
                        // Padding position - use zeros
                        Tensor::zeros(hidden_size, paramecia_core::DType::F32, device)?
                    } else {
                        let global_pos = chunk_start + rel_pos;

                        if global_pos < padded_conv.original.seq_len() {
                            // Check if this position has assistant data
                            if let Some(data_idx) =
                                global_to_data_idx.get(global_pos).and_then(|&x| x)
                            {
                                let shifted_data_idx = data_idx + embed_shift;

                                if shifted_data_idx < padded_conv.original.n_assistant_tokens() {
                                    // Compute weighted embedding from teacher distribution
                                    padded_conv.original.compute_weighted_embedding(
                                        shifted_data_idx,
                                        embed_weights,
                                    )?
                                } else {
                                    Tensor::zeros(hidden_size, paramecia_core::DType::F32, device)?
                                }
                            } else {
                                Tensor::zeros(hidden_size, paramecia_core::DType::F32, device)?
                            }
                        } else {
                            Tensor::zeros(hidden_size, paramecia_core::DType::F32, device)?
                        }
                    };

                    row_embeds.push(weighted_embed);
                }

                // Stack rows into [chunk_len, hidden]
                let weighted_chunk = Tensor::stack(&row_embeds, 0)?;
                batch_embeds.push(weighted_chunk);
            }

            // Stack into [batch, chunk_len, hidden]
            let batched = Tensor::stack(&batch_embeds, 0)?;
            weighted_embeds_per_depth.push(batched);
        }

        Ok(weighted_embeds_per_depth)
    }

    /// Process a minibatch of conversations with true batching.
    /// Returns (loss_tensor, total_positions).
    fn compute_minibatch_loss(
        &self,
        model: &mut ModelWeights,
        conversations: &[TuningData],
        device: &Device,
    ) -> Result<(Tensor, usize)> {
        if conversations.is_empty() {
            return Ok((Tensor::new(0.0f32, device)?, 0));
        }

        model.clear_cache();

        // Determine max length (cap at max_sequence_length if set)
        let batch_max_len = conversations.iter().map(|c| c.seq_len()).max().unwrap_or(0);
        let max_len = if let Some(limit) = self.config.max_sequence_length {
            batch_max_len.min(limit)
        } else {
            batch_max_len
        };
        let pad_token = 0u32; // BOS/EOS token

        // Truncate conversations if needed
        let conversations: Vec<TuningData> = conversations
            .iter()
            .map(|c| {
                if c.seq_len() > max_len {
                    c.truncate(max_len)
                } else {
                    c.clone()
                }
            })
            .collect();

        let padded: Vec<PaddedTuningData> = conversations
            .iter()
            .map(|c| c.pad_to_length(max_len, pad_token))
            .collect();

        let batch_size = padded.len();
        let chunk_size = self.config.chunk_size;
        let use_mtp = self.config.mtp_config.is_some() && model.has_mtp();

        let mut total_loss = 0.0f32;
        let mut total_positions = 0usize;
        let mut offset = 0;

        while offset < max_len {
            let chunk_end = (offset + chunk_size).min(max_len);
            let chunk_len = chunk_end - offset;

            // Gather chunks for all conversations
            let mut batch_token_ids = Vec::with_capacity(batch_size * chunk_len);
            let mut _batch_padding_masks = Vec::with_capacity(batch_size);
            let mut batch_assistant_positions: Vec<Vec<(usize, usize)>> =
                Vec::with_capacity(batch_size);

            for padded_conv in &padded {
                let (chunk_tokens, assistant_pos, chunk_padding) =
                    padded_conv.get_chunk(offset, chunk_size);

                batch_token_ids.extend_from_slice(&chunk_tokens);
                _batch_padding_masks.push(chunk_padding);
                batch_assistant_positions.push(assistant_pos);
            }

            // Check if any conversation has assistant tokens in this chunk
            let has_assistant = batch_assistant_positions.iter().any(|pos| !pos.is_empty());

            // Create batched input [batch_size, chunk_len]
            let input_ids = Tensor::new(batch_token_ids.as_slice(), device)?
                .reshape((batch_size, chunk_len))?;

            if !has_assistant {
                // No loss computation needed, just update state
                model.forward_state_update_only(&input_ids, offset)?;
            } else if use_mtp {
                // Batched MTP forward pass
                let mtp_config = self.config.mtp_config.as_ref().unwrap();

                // Compute batched weighted embeddings for all MTP depths
                let weighted_embeds = self.compute_batched_mtp_embeddings(
                    model,
                    &padded,
                    offset,
                    chunk_size,
                    mtp_config.num_depths,
                    device,
                )?;

                // Batched forward pass with MTP
                let (logits, router_stats, mtp_logits) = model.forward_training_with_mtp_weighted(
                    &input_ids,
                    offset,
                    &weighted_embeds,
                    mtp_config.num_depths,
                )?;

                // Compute loss for each conversation in batch
                for (batch_idx, conv) in conversations.iter().enumerate() {
                    let positions = &batch_assistant_positions[batch_idx];

                    if positions.is_empty() {
                        continue;
                    }

                    // Extract logits for this batch element [1, chunk_len, vocab]
                    let batch_logits = logits.narrow(0, batch_idx, 1)?;

                    // Extract MTP logits for this batch element
                    let batch_mtp_logits: Vec<Tensor> = mtp_logits
                        .iter()
                        .map(|mtp_depth_logits| mtp_depth_logits.narrow(0, batch_idx, 1))
                        .collect::<Result<Vec<_>>>()?;

                    // Compute MTP loss
                    let chunk_loss = self.loss_fn.compute_mtp_loss(
                        &batch_logits,
                        &batch_mtp_logits,
                        &router_stats,
                        conv,
                        positions,
                        mtp_config,
                        device,
                    )?;

                    let n_positions = positions.len();
                    let chunk_loss_val = chunk_loss.to_vec0::<f32>()?;

                    total_loss += chunk_loss_val * n_positions as f32;
                    total_positions += n_positions;
                }
            } else {
                // Standard batched forward pass
                let (logits, router_stats) = model.forward_training(&input_ids, offset)?;

                // Compute loss for each conversation in batch
                for (batch_idx, conv) in conversations.iter().enumerate() {
                    let positions = &batch_assistant_positions[batch_idx];

                    if positions.is_empty() {
                        continue;
                    }

                    // Extract logits for this batch element [1, chunk_len, vocab]
                    let batch_logits = logits.narrow(0, batch_idx, 1)?;

                    // Compute loss
                    let chunk_loss = self.loss_fn.compute_total_loss(
                        &batch_logits,
                        &router_stats,
                        conv,
                        positions,
                        device,
                    )?;

                    let n_positions = positions.len();
                    let chunk_loss_val = chunk_loss.to_vec0::<f32>()?;

                    total_loss += chunk_loss_val * n_positions as f32;
                    total_positions += n_positions;
                }
            }

            offset = chunk_end;
        }

        let avg_loss = if total_positions > 0 {
            total_loss / total_positions as f32
        } else {
            0.0f32
        };

        Ok((Tensor::new(avg_loss, device)?, total_positions))
    }

    /// Get current checkpoint state.
    pub fn checkpoint(&self) -> &TrainingCheckpoint {
        &self.checkpoint
    }

    /// Resume from a checkpoint.
    pub fn resume_from(&mut self, checkpoint: TrainingCheckpoint) {
        self.checkpoint = checkpoint;
    }
}

/// Update metadata entries in an existing GGUF checkpoint file.
///
/// Rewrites the file in-place (via a temporary file + rename) with the
/// provided metadata entries merged into the existing metadata.
pub fn update_gguf_metadata(
    checkpoint_path: &Path,
    metadata_updates: &HashMap<String, paramecia_core::quantized::gguf_file::Value>,
) -> Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use paramecia_core::quantized::gguf_file::{self, Value, ValueType};
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

    // Read original GGUF
    let mut input_file =
        BufReader::new(File::open(checkpoint_path).map_err(|e| {
            paramecia_core::Error::Msg(format!("Failed to open checkpoint: {}", e))
        })?);
    let input_content = gguf_file::Content::read(&mut input_file)
        .map_err(|e| paramecia_core::Error::Msg(format!("Failed to read GGUF: {}", e)))?;

    // Merge metadata
    let mut merged_metadata = input_content.metadata.clone();
    for (k, v) in metadata_updates {
        merged_metadata
            .entry(k.clone())
            .and_modify(|existing| *existing = v.clone())
            .or_insert_with(|| v.clone());
    }
    let mut metadata: Vec<(&str, &Value)> = merged_metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    metadata.sort_by_key(|(k, _)| *k);

    // Write to a temp file in the same directory, then rename
    let parent = checkpoint_path.parent().unwrap_or_else(|| Path::new("."));
    let tmp_path = parent.join(format!(
        ".tmp-metadata-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let write_result = (|| -> Result<()> {
        let output_file = File::create(&tmp_path).map_err(|e| {
            paramecia_core::Error::Msg(format!("Failed to create temp file: {}", e))
        })?;
        let mut output = BufWriter::with_capacity(8 * 1024 * 1024, output_file);

        fn write_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
            let bytes = s.as_bytes();
            w.write_u64::<LittleEndian>(bytes.len() as u64)?;
            w.write_all(bytes)?;
            Ok(())
        }

        fn value_type_to_u32(vt: ValueType) -> u32 {
            match vt {
                ValueType::U8 => 0,
                ValueType::I8 => 1,
                ValueType::U16 => 2,
                ValueType::I16 => 3,
                ValueType::U32 => 4,
                ValueType::I32 => 5,
                ValueType::F32 => 6,
                ValueType::Bool => 7,
                ValueType::String => 8,
                ValueType::Array => 9,
                ValueType::U64 => 10,
                ValueType::I64 => 11,
                ValueType::F64 => 12,
            }
        }

        fn write_value<W: Write>(w: &mut W, value: &Value) -> std::io::Result<()> {
            match value {
                Value::U8(v) => w.write_u8(*v)?,
                Value::I8(v) => w.write_i8(*v)?,
                Value::U16(v) => w.write_u16::<LittleEndian>(*v)?,
                Value::I16(v) => w.write_i16::<LittleEndian>(*v)?,
                Value::U32(v) => w.write_u32::<LittleEndian>(*v)?,
                Value::I32(v) => w.write_i32::<LittleEndian>(*v)?,
                Value::U64(v) => w.write_u64::<LittleEndian>(*v)?,
                Value::I64(v) => w.write_i64::<LittleEndian>(*v)?,
                Value::F32(v) => w.write_f32::<LittleEndian>(*v)?,
                Value::F64(v) => w.write_f64::<LittleEndian>(*v)?,
                Value::Bool(v) => w.write_u8(u8::from(*v))?,
                Value::String(v) => write_string(w, v.as_str())?,
                Value::Array(arr) => {
                    let elem_type = if arr.is_empty() {
                        ValueType::U32
                    } else {
                        arr[0].value_type()
                    };
                    w.write_u32::<LittleEndian>(value_type_to_u32(elem_type))?;
                    w.write_u64::<LittleEndian>(arr.len() as u64)?;
                    for elem in arr {
                        write_value(w, elem)?;
                    }
                }
            }
            Ok(())
        }

        fn ggml_dtype_to_u32(dtype: paramecia_core::quantized::GgmlDType) -> u32 {
            use paramecia_core::quantized::GgmlDType;
            match dtype {
                GgmlDType::F32 => 0,
                GgmlDType::F16 => 1,
                GgmlDType::Q4_0 => 2,
                GgmlDType::Q4_1 => 3,
                GgmlDType::Q5_0 => 6,
                GgmlDType::Q5_1 => 7,
                GgmlDType::Q8_0 => 8,
                GgmlDType::Q8_1 => 9,
                GgmlDType::Q2K => 10,
                GgmlDType::Q3K => 11,
                GgmlDType::Q4K => 12,
                GgmlDType::Q5K => 13,
                GgmlDType::Q6K => 14,
                GgmlDType::Q8K => 15,
                GgmlDType::BF16 => 30,
            }
        }

        // Write header
        output
            .write_u32::<LittleEndian>(0x46554747)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        output
            .write_u32::<LittleEndian>(3)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        output
            .write_u64::<LittleEndian>(input_content.tensor_infos.len() as u64)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        output
            .write_u64::<LittleEndian>(metadata.len() as u64)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;

        // Write metadata
        for (key, value) in &metadata {
            write_string(&mut output, key)
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
            output
                .write_u32::<LittleEndian>(value_type_to_u32(value.value_type()))
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
            write_value(&mut output, value)
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        }

        // Sort tensor names for deterministic output
        let mut tensor_names: Vec<_> = input_content.tensor_infos.keys().cloned().collect();
        tensor_names.sort();

        fn tensor_size_in_bytes(info: &gguf_file::TensorInfo) -> usize {
            let elem_count = info.shape.elem_count();
            let block_size = info.ggml_dtype.block_size();
            let type_size = info.ggml_dtype.type_size();
            elem_count / block_size * type_size
        }

        // Write tensor headers
        let mut data_offset: usize = 0;
        for name in &tensor_names {
            let info = &input_content.tensor_infos[name];
            let dims = info.shape.dims();

            write_string(&mut output, name)
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
            output
                .write_u32::<LittleEndian>(dims.len() as u32)
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
            for &dim in dims.iter().rev() {
                output
                    .write_u64::<LittleEndian>(dim as u64)
                    .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
            }
            output
                .write_u32::<LittleEndian>(ggml_dtype_to_u32(info.ggml_dtype))
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
            output
                .write_u64::<LittleEndian>(data_offset as u64)
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;

            let size = tensor_size_in_bytes(info);
            let padding = 31 - (31 + size) % 32;
            data_offset += size + padding;
        }

        // Pad header to 32-byte alignment
        let pos = output
            .stream_position()
            .map_err(|e| paramecia_core::Error::Msg(format!("Stream error: {}", e)))?
            as usize;
        let padding = 31 - (31 + pos) % 32;
        output
            .write_all(&vec![0u8; padding])
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;

        // Copy all tensor data from original file
        const BUFFER_SIZE: usize = 8 * 1024 * 1024;
        let mut buffer = vec![0u8; BUFFER_SIZE];

        for name in &tensor_names {
            let info = &input_content.tensor_infos[name];
            let size = tensor_size_in_bytes(info);

            input_file
                .seek(SeekFrom::Start(
                    input_content.tensor_data_offset + info.offset,
                ))
                .map_err(|e| paramecia_core::Error::Msg(format!("Seek error: {}", e)))?;
            let mut remaining = size;
            while remaining > 0 {
                let to_read = remaining.min(BUFFER_SIZE);
                input_file
                    .read_exact(&mut buffer[..to_read])
                    .map_err(|e| paramecia_core::Error::Msg(format!("Read error: {}", e)))?;
                output
                    .write_all(&buffer[..to_read])
                    .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
                remaining -= to_read;
            }

            // Write padding
            let padding = 31 - (31 + size) % 32;
            if padding > 0 {
                output
                    .write_all(&vec![0u8; padding])
                    .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
            }
        }

        output
            .flush()
            .map_err(|e| paramecia_core::Error::Msg(format!("Flush error: {}", e)))?;

        Ok(())
    })();

    match write_result {
        Ok(()) => {
            std::fs::rename(&tmp_path, checkpoint_path).map_err(|e| {
                paramecia_core::Error::Msg(format!("Failed to rename temp file: {}", e))
            })?;
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

/// Save trained model to GGUF format.
///
/// This creates a new GGUF file with the updated weights while preserving
/// all metadata and non-trained tensors from the original model.
pub fn save_trained_model(
    model: &ModelWeights,
    original_path: &Path,
    output_path: &Path,
    extra_metadata: Option<&HashMap<String, paramecia_core::quantized::gguf_file::Value>>,
) -> Result<()> {
    use byteorder::{LittleEndian, WriteBytesExt};
    use paramecia_core::quantized::gguf_file::{self, Value, ValueType};
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

    info!("Saving trained model to {:?}...", output_path);

    // Read original GGUF
    let mut input_file = BufReader::new(File::open(original_path).map_err(|e| {
        paramecia_core::Error::Msg(format!("Failed to open original model: {}", e))
    })?);
    let input_content = gguf_file::Content::read(&mut input_file)
        .map_err(|e| paramecia_core::Error::Msg(format!("Failed to read GGUF: {}", e)))?;

    // Get trained tensor data from model
    let trained_tensors = model.quzo_qtensors();
    let trained_names: HashSet<String> = trained_tensors.iter().map(|(n, _)| n.clone()).collect();

    info!(
        "  {} trained tensors to update out of {} total",
        trained_names.len(),
        input_content.tensor_infos.len()
    );
    {
        let mut sorted_names: Vec<_> = trained_names.iter().cloned().collect();
        sorted_names.sort();
        for name in &sorted_names {
            debug!("    [trained] {}", name);
        }
    }

    // Create output file
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                paramecia_core::Error::Msg(format!("Failed to create output directory: {}", e))
            })?;
        }
    }

    let output_file = File::create(output_path)
        .map_err(|e| paramecia_core::Error::Msg(format!("Failed to create output file: {}", e)))?;
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, output_file);

    // Helper functions for GGUF writing
    fn write_string<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
        let bytes = s.as_bytes();
        w.write_u64::<LittleEndian>(bytes.len() as u64)?;
        w.write_all(bytes)?;
        Ok(())
    }

    fn value_type_to_u32(vt: ValueType) -> u32 {
        match vt {
            ValueType::U8 => 0,
            ValueType::I8 => 1,
            ValueType::U16 => 2,
            ValueType::I16 => 3,
            ValueType::U32 => 4,
            ValueType::I32 => 5,
            ValueType::F32 => 6,
            ValueType::Bool => 7,
            ValueType::String => 8,
            ValueType::Array => 9,
            ValueType::U64 => 10,
            ValueType::I64 => 11,
            ValueType::F64 => 12,
        }
    }

    fn write_value<W: Write>(w: &mut W, value: &Value) -> std::io::Result<()> {
        match value {
            Value::U8(v) => w.write_u8(*v)?,
            Value::I8(v) => w.write_i8(*v)?,
            Value::U16(v) => w.write_u16::<LittleEndian>(*v)?,
            Value::I16(v) => w.write_i16::<LittleEndian>(*v)?,
            Value::U32(v) => w.write_u32::<LittleEndian>(*v)?,
            Value::I32(v) => w.write_i32::<LittleEndian>(*v)?,
            Value::U64(v) => w.write_u64::<LittleEndian>(*v)?,
            Value::I64(v) => w.write_i64::<LittleEndian>(*v)?,
            Value::F32(v) => w.write_f32::<LittleEndian>(*v)?,
            Value::F64(v) => w.write_f64::<LittleEndian>(*v)?,
            Value::Bool(v) => w.write_u8(u8::from(*v))?,
            Value::String(v) => write_string(w, v.as_str())?,
            Value::Array(arr) => {
                let elem_type = if arr.is_empty() {
                    ValueType::U32
                } else {
                    arr[0].value_type()
                };
                w.write_u32::<LittleEndian>(value_type_to_u32(elem_type))?;
                w.write_u64::<LittleEndian>(arr.len() as u64)?;
                for elem in arr {
                    write_value(w, elem)?;
                }
            }
        }
        Ok(())
    }

    fn ggml_dtype_to_u32(dtype: paramecia_core::quantized::GgmlDType) -> u32 {
        use paramecia_core::quantized::GgmlDType;
        match dtype {
            GgmlDType::F32 => 0,
            GgmlDType::F16 => 1,
            GgmlDType::Q4_0 => 2,
            GgmlDType::Q4_1 => 3,
            GgmlDType::Q5_0 => 6,
            GgmlDType::Q5_1 => 7,
            GgmlDType::Q8_0 => 8,
            GgmlDType::Q8_1 => 9,
            GgmlDType::Q2K => 10,
            GgmlDType::Q3K => 11,
            GgmlDType::Q4K => 12,
            GgmlDType::Q5K => 13,
            GgmlDType::Q6K => 14,
            GgmlDType::Q8K => 15,
            GgmlDType::BF16 => 30,
        }
    }

    // Prepare metadata — start from original, then overlay any extra entries
    let mut merged_metadata = input_content.metadata.clone();
    if let Some(extra) = extra_metadata {
        for (k, v) in extra {
            merged_metadata
                .entry(k.clone())
                .and_modify(|existing| *existing = v.clone())
                .or_insert_with(|| v.clone());
        }
    }
    let mut metadata: Vec<(&str, &Value)> = merged_metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    metadata.sort_by_key(|(k, _)| *k);

    // Write header
    output
        .write_u32::<LittleEndian>(0x46554747) // "GGUF" magic
        .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
    output
        .write_u32::<LittleEndian>(3) // Version 3
        .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
    output
        .write_u64::<LittleEndian>(input_content.tensor_infos.len() as u64)
        .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
    output
        .write_u64::<LittleEndian>(metadata.len() as u64)
        .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;

    // Write metadata
    for (key, value) in &metadata {
        write_string(&mut output, key)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        output
            .write_u32::<LittleEndian>(value_type_to_u32(value.value_type()))
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        write_value(&mut output, value)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
    }

    // Sort tensor names for deterministic output
    let mut tensor_names: Vec<_> = input_content.tensor_infos.keys().cloned().collect();
    tensor_names.sort();

    // Calculate tensor sizes
    fn tensor_size_in_bytes(info: &gguf_file::TensorInfo) -> usize {
        let elem_count = info.shape.elem_count();
        let block_size = info.ggml_dtype.block_size();
        let type_size = info.ggml_dtype.type_size();
        elem_count / block_size * type_size
    }

    // Write tensor headers
    let mut data_offset: usize = 0;
    for name in &tensor_names {
        let info = &input_content.tensor_infos[name];
        let dims = info.shape.dims();

        write_string(&mut output, name)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        output
            .write_u32::<LittleEndian>(dims.len() as u32)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        for &dim in dims.iter().rev() {
            output
                .write_u64::<LittleEndian>(dim as u64)
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        }
        output
            .write_u32::<LittleEndian>(ggml_dtype_to_u32(info.ggml_dtype))
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        output
            .write_u64::<LittleEndian>(data_offset as u64)
            .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;

        let size = tensor_size_in_bytes(info);
        let padding = 31 - (31 + size) % 32;
        data_offset += size + padding;
    }

    // Pad header to 32-byte alignment
    let pos = output
        .stream_position()
        .map_err(|e| paramecia_core::Error::Msg(format!("Stream error: {}", e)))?
        as usize;
    let padding = 31 - (31 + pos) % 32;
    output
        .write_all(&vec![0u8; padding])
        .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;

    // Write tensor data
    info!("  Writing {} tensors...", tensor_names.len());
    const BUFFER_SIZE: usize = 8 * 1024 * 1024;
    let mut buffer = vec![0u8; BUFFER_SIZE];

    let trained_tensor_map: std::collections::HashMap<String, SharedQTensor> =
        trained_tensors.into_iter().collect();

    for (idx, name) in tensor_names.iter().enumerate() {
        let info = &input_content.tensor_infos[name];
        let size = tensor_size_in_bytes(info);

        if trained_names.contains(name) {
            // Write updated tensor data from trained model
            if let Some(qt) = trained_tensor_map.get(name) {
                let qt_read = qt.read().map_err(|e| {
                    paramecia_core::Error::Msg(format!("Failed to lock tensor {}: {}", name, e))
                })?;
                let data = qt_read.data()?;
                output
                    .write_all(&data)
                    .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
            } else {
                // Fallback: copy from original
                input_file
                    .seek(SeekFrom::Start(
                        input_content.tensor_data_offset + info.offset,
                    ))
                    .map_err(|e| paramecia_core::Error::Msg(format!("Seek error: {}", e)))?;
                let mut remaining = size;
                while remaining > 0 {
                    let to_read = remaining.min(BUFFER_SIZE);
                    input_file
                        .read_exact(&mut buffer[..to_read])
                        .map_err(|e| paramecia_core::Error::Msg(format!("Read error: {}", e)))?;
                    output
                        .write_all(&buffer[..to_read])
                        .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
                    remaining -= to_read;
                }
            }
        } else {
            // Copy unchanged tensor from original file
            input_file
                .seek(SeekFrom::Start(
                    input_content.tensor_data_offset + info.offset,
                ))
                .map_err(|e| paramecia_core::Error::Msg(format!("Seek error: {}", e)))?;
            let mut remaining = size;
            while remaining > 0 {
                let to_read = remaining.min(BUFFER_SIZE);
                input_file
                    .read_exact(&mut buffer[..to_read])
                    .map_err(|e| paramecia_core::Error::Msg(format!("Read error: {}", e)))?;
                output
                    .write_all(&buffer[..to_read])
                    .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
                remaining -= to_read;
            }
        }

        // Write padding
        let padding = 31 - (31 + size) % 32;
        if padding > 0 {
            output
                .write_all(&vec![0u8; padding])
                .map_err(|e| paramecia_core::Error::Msg(format!("Write error: {}", e)))?;
        }

        if (idx + 1) % 100 == 0 || idx + 1 == tensor_names.len() {
            debug!("  Written {}/{} tensors", idx + 1, tensor_names.len());
        }
    }

    output
        .flush()
        .map_err(|e| paramecia_core::Error::Msg(format!("Flush error: {}", e)))?;

    let output_size = std::fs::metadata(output_path)
        .map_err(|e| paramecia_core::Error::Msg(format!("Metadata error: {}", e)))?
        .len();
    info!("  Output size: {:.2} GB", output_size as f64 / 1e9);

    Ok(())
}
