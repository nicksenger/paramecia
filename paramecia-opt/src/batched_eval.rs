//! Batched evaluation for EGGROLL perturbations.
//!
//! This module provides utilities for efficient perturbation evaluation.
//!
//! # Two Batching Strategies
//!
//! ## Strategy 1: Batch across examples (RECOMMENDED - matches paper)
//! - Apply ONE perturbation to the model
//! - Run MULTIPLE examples/problems through it in a batch
//! - This is what the EGGROLL paper actually describes
//! - Maximizes GPU utilization within a single perturbation evaluation
//!
//! ## Strategy 2: Batch across perturbations (EXPERIMENTAL)
//! - Apply MULTIPLE perturbations simultaneously
//! - Requires stacking LoRA matrices: [num_perturbs, out_features, rank]
//! - More complex, may not provide benefits over Strategy 1
//! - Useful when examples are small but perturbations are many
//!
//! # Recommended Workflow (Strategy 1)
//!
//! ```ignore
//! for perturbation in population {
//!     // Apply this perturbation's LoRA to model
//!     model.set_lora(perturbation.lora_a, perturbation.lora_b, sigma);
//!     
//!     // Batch ALL problems/examples through the perturbed model
//!     let logits = model.forward(&all_examples)?;  // [num_examples, seq_len, vocab]
//!     
//!     // Compute fitness across all examples
//!     let fitness = compute_fitness(&logits, &targets)?;
//! }
//! ```
//!
//! This is simpler and matches the paper's description of EGGROLL.

use candle::{Device, IndexOp, Result, Tensor};
use std::collections::HashMap;

use crate::eggroll::{LayerConfig, PopulationMember};

// =============================================================================
// STRATEGY 1: Single-perturbation batched evaluation (RECOMMENDED)
// =============================================================================

/// Configuration for single-perturbation evaluation with batched examples.
/// This is the approach described in the EGGROLL paper.
#[derive(Debug, Clone)]
pub struct SinglePerturbationConfig {
    /// Sigma value for perturbation scaling.
    pub sigma: f64,
    /// Device for computation.
    pub device: Device,
}

impl Default for SinglePerturbationConfig {
    fn default() -> Self {
        Self {
            sigma: 0.001,
            device: Device::Cpu,
        }
    }
}

/// Prepared perturbation ready for efficient forward pass.
///
/// This is the paper-aligned approach: apply ONE perturbation to the model,
/// then batch MULTIPLE examples through it.
#[derive(Debug, Clone)]
pub struct PreparedPerturbation {
    /// Scale multipliers per layer: layer_name -> (1 + sigma * delta)
    pub scales: HashMap<String, Tensor>,
    /// LoRA adapters per layer: layer_name -> (A, B)
    pub lora: HashMap<String, (Tensor, Tensor)>,
    /// Sigma used for this perturbation
    pub sigma: f64,
    /// Member index (for result tracking)
    pub member_index: usize,
    /// Whether this is antithetic
    pub is_antithetic: bool,
}

impl PreparedPerturbation {
    /// Create from a PopulationMember.
    pub fn from_member(member: &PopulationMember, sigma: f64) -> Result<Self> {
        let mut scales = HashMap::new();
        let mut lora = HashMap::new();

        for (name, layer_pert) in &member.perturbations {
            // Convert scale delta to multiplier: 1 + sigma * delta
            if let Some(delta) = &layer_pert.scale_delta {
                let scaled_delta = (delta * sigma)?;
                let ones = Tensor::ones(delta.shape(), delta.dtype(), delta.device())?;
                let multiplier = (&ones + &scaled_delta)?;
                scales.insert(name.clone(), multiplier);
            }

            // Copy LoRA adapters
            if let Some(lora_pert) = &layer_pert.lora {
                lora.insert(name.clone(), (lora_pert.a.clone(), lora_pert.b.clone()));
            }
        }

        Ok(Self {
            scales,
            lora,
            sigma,
            member_index: member.member_index,
            is_antithetic: member.is_antithetic,
        })
    }

    /// Get scale multipliers for a layer.
    pub fn get_scales(&self, layer_name: &str) -> Option<&Tensor> {
        self.scales.get(layer_name)
    }

    /// Get LoRA adapters for a layer.
    pub fn get_lora(&self, layer_name: &str) -> Option<(&Tensor, &Tensor)> {
        self.lora.get(layer_name).map(|(a, b)| (a, b))
    }
}

/// Compute fitness for a single perturbation evaluated on batched examples.
///
/// This is the efficient, paper-aligned approach:
/// - One perturbation applied to the model
/// - Multiple examples batched through the perturbed model
/// - Fitness = average negative cross-entropy across examples
pub fn compute_single_perturbation_fitness(logits: &Tensor, targets: &Tensor) -> Result<f32> {
    // logits: [batch, seq_len, vocab_size] or [batch, vocab_size]
    // targets: [batch, seq_len] or [batch]

    let logits_dims = logits.dims();

    if logits_dims.len() == 3 {
        // Training mode: [batch, seq_len, vocab_size]
        let batch = logits_dims[0];
        let seq_len = logits_dims[1];
        let vocab_size = logits_dims[2];

        let logits_flat = logits.reshape((batch * seq_len, vocab_size))?;
        let targets_flat = targets.reshape(batch * seq_len)?;

        let log_softmax = candle_nn::ops::log_softmax(&logits_flat, 1)?;
        let loss = candle_nn::loss::nll(&log_softmax, &targets_flat)?;
        let loss_val = loss.to_scalar::<f32>()?;

        Ok(-loss_val) // Fitness = negative loss
    } else {
        // Inference mode: [batch, vocab_size]
        let log_softmax = candle_nn::ops::log_softmax(logits, 1)?;
        let loss = candle_nn::loss::nll(&log_softmax, targets)?;
        let loss_val = loss.to_scalar::<f32>()?;

        Ok(-loss_val)
    }
}

/// Compute fitness for a single perturbation with MoE auxiliary losses.
///
/// This version includes Z-loss and load balance loss as negative fitness terms,
/// which is important for training Mixture-of-Experts models like Qwen3-next.
///
/// # Arguments
/// * `logits` - Model output logits: [batch, seq_len, vocab_size]
/// * `targets` - Target token IDs: [batch, seq_len]
/// * `router_stats` - Router statistics from each MoE layer: Vec of (router_logits, selected_experts)
/// * `z_loss_weight` - Weight for Z-loss penalty (typically 0.001)
/// * `lb_loss_weight` - Weight for load balance loss penalty (typically 0.01)
/// * `num_experts` - Number of experts in the MoE layers
///
/// # Returns
/// Fitness = -cross_entropy - z_loss_weight * z_loss - lb_loss_weight * lb_loss
pub fn compute_single_perturbation_fitness_with_moe_loss(
    logits: &Tensor,
    targets: &Tensor,
    router_stats: &[(Tensor, Tensor)],
    z_loss_weight: f64,
    lb_loss_weight: f64,
    num_experts: usize,
) -> Result<f32> {
    // Base fitness from cross-entropy
    let base_fitness = compute_single_perturbation_fitness(logits, targets)?;

    if router_stats.is_empty() || (z_loss_weight == 0.0 && lb_loss_weight == 0.0) {
        return Ok(base_fitness);
    }

    let mut z_loss_total = 0.0f32;
    let mut lb_loss_total = 0.0f32;

    for (router_logits, selected_experts) in router_stats {
        // Flatten router logits to [total_tokens, num_experts]
        let router_logits_flat = router_logits.flatten(0, 1)?;

        // Z-loss: penalizes large router logits to prevent collapse
        if z_loss_weight > 0.0 {
            let z_loss = compute_z_loss_scalar(&router_logits_flat)?;
            z_loss_total += z_loss;
        }

        // Load balance loss: encourages uniform expert utilization
        if lb_loss_weight > 0.0 {
            let selected_flat = selected_experts.flatten(0, 1)?;
            let lb_loss =
                compute_load_balance_loss_scalar(&router_logits_flat, &selected_flat, num_experts)?;
            lb_loss_total += lb_loss;
        }
    }

    let num_layers = router_stats.len() as f32;
    z_loss_total /= num_layers;
    lb_loss_total /= num_layers;

    // Subtract auxiliary losses from fitness (they are penalties)
    let aux_penalty =
        (z_loss_weight as f32 * z_loss_total) + (lb_loss_weight as f32 * lb_loss_total);

    Ok(base_fitness - aux_penalty)
}

/// Compute Z-loss for a single layer's router logits.
/// Z-loss = mean(log_sum_exp(router_logits)^2)
fn compute_z_loss_scalar(router_logits: &Tensor) -> Result<f32> {
    // router_logits: [num_tokens, num_experts]

    // Compute log-sum-exp along expert dimension
    let max_val = router_logits.max(candle::D::Minus1)?;
    let max_val_keepdim = max_val.unsqueeze(candle::D::Minus1)?;
    let exp_shifted = router_logits.broadcast_sub(&max_val_keepdim)?.exp()?;
    let sum_exp = exp_shifted.sum(candle::D::Minus1)?;
    let log_sum_exp = (max_val + sum_exp.log()?)?;

    // Z-loss = mean(lse^2)
    let z_loss = (&log_sum_exp * &log_sum_exp)?.mean_all()?;
    z_loss.to_scalar::<f32>()
}

/// Compute load balance loss for a single layer.
/// LB-loss = num_experts * sum(f_i * p_i) where:
/// - f_i = fraction of tokens routed to expert i
/// - p_i = mean routing probability for expert i
fn compute_load_balance_loss_scalar(
    router_logits: &Tensor,
    selected_experts: &Tensor,
    num_experts: usize,
) -> Result<f32> {
    let device = router_logits.device();

    // Compute routing probabilities
    let router_probs = candle_nn::ops::softmax_last_dim(router_logits)?;

    // Compute f_i: fraction of tokens routed to each expert
    let selected_vec = selected_experts.to_vec2::<u32>().unwrap_or_else(|_| {
        // Fall back to 1D
        selected_experts
            .to_vec1::<u32>()
            .map(|v| vec![v])
            .unwrap_or_default()
    });

    let mut expert_counts = vec![0.0f32; num_experts];
    let mut total_selections = 0usize;

    for row in &selected_vec {
        for &expert_idx in row {
            if (expert_idx as usize) < num_experts {
                expert_counts[expert_idx as usize] += 1.0;
                total_selections += 1;
            }
        }
    }

    if total_selections == 0 {
        return Ok(0.0);
    }

    let f_i: Vec<f32> = expert_counts
        .iter()
        .map(|c| c / total_selections as f32)
        .collect();
    let f_i_tensor = Tensor::from_vec(f_i, num_experts, device)?;

    // Compute p_i: mean routing probability for each expert
    let p_i = router_probs.mean(0)?;

    // LB-loss = num_experts * sum(f_i * p_i)
    let lb_loss = (&f_i_tensor * &p_i)?.sum_all()?;
    let lb_loss = (lb_loss * (num_experts as f64))?.to_scalar::<f32>()?;

    Ok(lb_loss)
}

/// Configuration for chunked example evaluation.
#[derive(Debug, Clone)]
pub struct ChunkedEvalConfig {
    /// Maximum number of examples to process in a single forward pass.
    /// Controls GPU memory usage.
    pub example_batch_size: usize,
    /// Maximum sequence length. Longer sequences will be truncated.
    pub max_seq_len: usize,
    /// Whether to pad shorter sequences to max_seq_len.
    pub pad_to_max: bool,
    /// Padding token ID.
    pub pad_token_id: u32,
}

impl Default for ChunkedEvalConfig {
    fn default() -> Self {
        Self {
            // For agentic coding, contexts are large (8K-64K tokens typically).
            // Batch size of 2 is conservative for 32K context on 24GB GPU.
            example_batch_size: 2,
            // 32K context covers most agentic coding tasks.
            // Increase to 65536 or 131072 for very long contexts if GPU allows.
            max_seq_len: 32768,
            pad_to_max: false,
            pad_token_id: 0,
        }
    }
}

/// Result of chunked evaluation.
#[derive(Debug, Clone)]
pub struct ChunkedEvalResult {
    /// Per-example losses (negative for fitness).
    pub losses: Vec<f32>,
    /// Average fitness across all examples.
    pub mean_fitness: f32,
    /// Number of examples processed.
    pub num_examples: usize,
    /// Number of chunks used.
    pub num_chunks: usize,
}

/// Trait for models that support chunked example evaluation.
///
/// This is the recommended interface for EGGROLL evaluation:
/// - Apply ONE perturbation to the model
/// - Call `forward_chunked` to process examples in GPU-sized batches
pub trait ChunkedForward {
    /// Forward pass on a chunk of examples.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape [batch, seq_len]
    /// * `index_pos` - Starting position (0 for no cache)
    ///
    /// # Returns
    /// Logits of shape [batch, seq_len, vocab_size]
    fn forward_chunk(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor>;

    /// Clear any cached state between chunks if needed.
    fn clear_cache(&mut self);
}

/// Evaluate examples in chunks that fit in GPU memory.
///
/// This function:
/// 1. Splits examples into chunks of size `config.example_batch_size`
/// 2. Runs each chunk through the model
/// 3. Accumulates losses
/// 4. Returns average fitness
///
/// # Arguments
/// * `model` - Model with perturbation already applied
/// * `all_input_ids` - All example token IDs: Vec of [seq_len] tensors
/// * `all_targets` - All target token IDs: Vec of [seq_len] tensors
/// * `config` - Chunking configuration
///
/// # Returns
/// ChunkedEvalResult with per-example and mean fitness
pub fn evaluate_examples_chunked<M: ChunkedForward>(
    model: &mut M,
    all_input_ids: &[Tensor],
    all_targets: &[Tensor],
    config: &ChunkedEvalConfig,
) -> Result<ChunkedEvalResult> {
    if all_input_ids.is_empty() {
        return Ok(ChunkedEvalResult {
            losses: vec![],
            mean_fitness: 0.0,
            num_examples: 0,
            num_chunks: 0,
        });
    }

    let num_examples = all_input_ids.len();
    let num_chunks = (num_examples + config.example_batch_size - 1) / config.example_batch_size;
    let mut all_losses = Vec::with_capacity(num_examples);

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * config.example_batch_size;
        let end = (start + config.example_batch_size).min(num_examples);
        let chunk_size = end - start;

        // Get chunk of examples
        let chunk_inputs: Vec<_> = all_input_ids[start..end].to_vec();
        let chunk_targets: Vec<_> = all_targets[start..end].to_vec();

        // Pad/truncate and stack into batch tensor
        let (input_batch, target_batch) = prepare_batch(
            &chunk_inputs,
            &chunk_targets,
            config.max_seq_len,
            config.pad_to_max,
            config.pad_token_id,
        )?;

        // Clear cache before each chunk to avoid state bleeding
        model.clear_cache();

        // Forward pass
        let logits = model.forward_chunk(&input_batch, 0)?;

        // Compute per-example losses
        let chunk_losses = compute_per_example_loss(&logits, &target_batch, chunk_size)?;
        all_losses.extend(chunk_losses);
    }

    // Compute mean fitness (negative mean loss)
    let total_loss: f32 = all_losses.iter().sum();
    let mean_fitness = -total_loss / num_examples as f32;

    Ok(ChunkedEvalResult {
        losses: all_losses,
        mean_fitness,
        num_examples,
        num_chunks,
    })
}

/// Prepare a batch of examples with padding/truncation.
fn prepare_batch(
    inputs: &[Tensor],
    targets: &[Tensor],
    max_seq_len: usize,
    pad_to_max: bool,
    pad_token_id: u32,
) -> Result<(Tensor, Tensor)> {
    let batch_size = inputs.len();
    let device = inputs[0].device();

    // Find max sequence length in this batch (or use max_seq_len if pad_to_max)
    let actual_max_len = if pad_to_max {
        max_seq_len
    } else {
        inputs
            .iter()
            .map(|t| t.dim(0).unwrap_or(0).min(max_seq_len))
            .max()
            .unwrap_or(max_seq_len)
    };

    // Create padded tensors
    let mut input_data = vec![pad_token_id; batch_size * actual_max_len];
    let mut target_data = vec![pad_token_id; batch_size * actual_max_len];

    for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
        let input_vec = input.to_vec1::<u32>()?;
        let target_vec = target.to_vec1::<u32>()?;
        let len = input_vec.len().min(actual_max_len);

        for j in 0..len {
            input_data[i * actual_max_len + j] = input_vec[j];
            target_data[i * actual_max_len + j] = target_vec[j];
        }
    }

    let input_batch = Tensor::from_vec(input_data, (batch_size, actual_max_len), device)?;
    let target_batch = Tensor::from_vec(target_data, (batch_size, actual_max_len), device)?;

    Ok((input_batch, target_batch))
}

/// Compute per-example loss from batched logits.
fn compute_per_example_loss(
    logits: &Tensor,
    targets: &Tensor,
    batch_size: usize,
) -> Result<Vec<f32>> {
    let logits_dims = logits.dims();
    let seq_len = logits_dims[1];
    let vocab_size = logits_dims[2];

    let mut losses = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        // Get logits and targets for this example
        let example_logits = logits.i((i, .., ..))?; // [seq_len, vocab_size]
        let example_targets = targets.i((i, ..))?; // [seq_len]

        // Flatten for cross-entropy
        let logits_flat = example_logits.reshape((seq_len, vocab_size))?;

        let log_softmax = candle_nn::ops::log_softmax(&logits_flat, 1)?;
        let loss = candle_nn::loss::nll(&log_softmax, &example_targets)?;
        let loss_val = loss.to_scalar::<f32>()?;

        losses.push(loss_val);
    }

    Ok(losses)
}

// =============================================================================
// STRATEGY 2: Multi-perturbation batched evaluation (EXPERIMENTAL)
// =============================================================================

/// Configuration for batched perturbation evaluation.
#[derive(Debug, Clone)]
pub struct BatchedEvalConfig {
    /// Number of perturbations to evaluate in one batch.
    pub batch_size: usize,
    /// Sigma value for perturbation scaling.
    pub sigma: f64,
    /// Whether to use gradient checkpointing to reduce memory.
    pub gradient_checkpointing: bool,
    /// Device for computation.
    pub device: Device,
}

impl Default for BatchedEvalConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            sigma: 0.001,
            gradient_checkpointing: false,
            device: Device::Cpu,
        }
    }
}

/// Stacked LoRA perturbations for batched forward pass.
///
/// For a single layer, holds LoRA A and B matrices for all perturbations
/// in the batch, stacked along a new dimension.
#[derive(Debug, Clone)]
pub struct StackedLoRA {
    /// Stacked A matrices: [num_perturbs, out_features, rank]
    pub a_stacked: Tensor,
    /// Stacked B matrices: [num_perturbs, in_features, rank]  
    pub b_stacked: Tensor,
    /// Number of perturbations
    pub num_perturbs: usize,
    /// Rank of each LoRA
    pub rank: usize,
}

impl StackedLoRA {
    /// Create stacked LoRA from a batch of population members.
    pub fn from_members(
        members: &[PopulationMember],
        layer_name: &str,
        _device: &Device,
    ) -> Result<Option<Self>> {
        let loras: Vec<_> = members
            .iter()
            .filter_map(|m| {
                m.perturbations
                    .get(layer_name)
                    .and_then(|p| p.lora.as_ref())
            })
            .collect();

        if loras.is_empty() {
            return Ok(None);
        }

        let num_perturbs = loras.len();
        let rank = loras[0].rank;

        // Stack A matrices: [num_perturbs, out_features, rank]
        let a_tensors: Vec<_> = loras.iter().map(|l| l.a.clone()).collect();
        let a_stacked = Tensor::stack(&a_tensors, 0)?;

        // Stack B matrices: [num_perturbs, in_features, rank]
        let b_tensors: Vec<_> = loras.iter().map(|l| l.b.clone()).collect();
        let b_stacked = Tensor::stack(&b_tensors, 0)?;

        Ok(Some(Self {
            a_stacked,
            b_stacked,
            num_perturbs,
            rank,
        }))
    }

    /// Apply stacked LoRA to batched input.
    ///
    /// Input x: [num_perturbs * batch, seq_len, in_features]
    /// Output: [num_perturbs * batch, seq_len, out_features]
    ///
    /// Each slice of size `batch` uses the corresponding LoRA perturbation.
    pub fn apply_batched(&self, x: &Tensor, batch_size: usize, sigma: f64) -> Result<Tensor> {
        let scale = sigma / (self.rank as f64).sqrt();
        let x_dims = x.dims();

        // x is [num_perturbs * batch, seq_len, in_features]
        let total_batch = x_dims[0];
        let seq_len = x_dims[1];
        let in_features = x_dims[2];

        debug_assert_eq!(total_batch, self.num_perturbs * batch_size);

        // Reshape to [num_perturbs, batch, seq_len, in_features]
        let x_4d = x.reshape((self.num_perturbs, batch_size, seq_len, in_features))?;

        // For each perturbation i, compute: x[i] @ B[i] @ A[i]^T
        // This is a batched operation over the perturbation dimension

        // x_4d: [num_perturbs, batch, seq_len, in_features]
        // b_stacked: [num_perturbs, in_features, rank]
        // Want: [num_perturbs, batch, seq_len, rank]

        // Reshape x to [num_perturbs, batch * seq_len, in_features]
        let x_3d = x_4d.reshape((self.num_perturbs, batch_size * seq_len, in_features))?;

        // Batched matmul: x_3d @ b_stacked -> [num_perturbs, batch * seq_len, rank]
        let xb = x_3d.matmul(&self.b_stacked)?;

        // a_stacked^T: [num_perturbs, rank, out_features]
        let a_t = self.a_stacked.transpose(1, 2)?;

        // Batched matmul: xb @ a_t -> [num_perturbs, batch * seq_len, out_features]
        let out_3d = xb.matmul(&a_t)?;

        let out_features = self.a_stacked.dim(1)?;

        // Reshape back to [num_perturbs * batch, seq_len, out_features]
        let out = out_3d.reshape((self.num_perturbs * batch_size, seq_len, out_features))?;

        // Scale by sigma / sqrt(rank)
        out * scale
    }
}

/// Stacked scale perturbations for batched forward pass.
#[derive(Debug, Clone)]
pub struct StackedScales {
    /// Stacked scale deltas: [num_perturbs, num_blocks]
    pub scales_stacked: Tensor,
    /// Number of perturbations
    pub num_perturbs: usize,
}

impl StackedScales {
    /// Create stacked scales from a batch of population members.
    pub fn from_members(
        members: &[PopulationMember],
        layer_name: &str,
        _device: &Device,
    ) -> Result<Option<Self>> {
        let scales: Vec<_> = members
            .iter()
            .filter_map(|m| {
                m.perturbations
                    .get(layer_name)
                    .and_then(|p| p.scale_delta.as_ref())
            })
            .collect();

        if scales.is_empty() {
            return Ok(None);
        }

        let num_perturbs = scales.len();
        let scale_tensors: Vec<_> = scales.iter().map(|s| (*s).clone()).collect();
        let scales_stacked = Tensor::stack(&scale_tensors, 0)?;

        Ok(Some(Self {
            scales_stacked,
            num_perturbs,
        }))
    }

    /// Get the scale multipliers for a specific perturbation index.
    /// Returns 1.0 + sigma * delta for the i-th perturbation.
    pub fn get_scales(&self, perturbation_idx: usize, sigma: f64) -> Result<Tensor> {
        let delta = self.scales_stacked.i(perturbation_idx)?;
        let scaled_delta = (&delta * sigma)?;
        let ones = Tensor::ones(delta.shape(), delta.dtype(), delta.device())?;
        ones + scaled_delta
    }
}

/// Batched layer perturbations for all layers.
#[derive(Debug)]
pub struct BatchedPerturbations {
    /// Layer name -> stacked LoRA perturbations
    pub lora: HashMap<String, StackedLoRA>,
    /// Layer name -> stacked scale perturbations
    pub scales: HashMap<String, StackedScales>,
    /// Number of perturbations in this batch
    pub num_perturbs: usize,
    /// Configuration
    pub config: BatchedEvalConfig,
}

impl BatchedPerturbations {
    /// Create batched perturbations from a population of members.
    pub fn from_population(
        members: &[PopulationMember],
        layer_configs: &[LayerConfig],
        config: BatchedEvalConfig,
    ) -> Result<Self> {
        let num_perturbs = members.len();
        let mut lora = HashMap::new();
        let mut scales = HashMap::new();

        for layer_config in layer_configs {
            if layer_config.optimize_lora {
                if let Some(stacked) =
                    StackedLoRA::from_members(members, &layer_config.name, &config.device)?
                {
                    lora.insert(layer_config.name.clone(), stacked);
                }
            }

            if layer_config.optimize_scales {
                if let Some(stacked) =
                    StackedScales::from_members(members, &layer_config.name, &config.device)?
                {
                    scales.insert(layer_config.name.clone(), stacked);
                }
            }
        }

        Ok(Self {
            lora,
            scales,
            num_perturbs,
            config,
        })
    }

    /// Get the LoRA perturbation for a specific layer.
    pub fn get_lora(&self, layer_name: &str) -> Option<&StackedLoRA> {
        self.lora.get(layer_name)
    }

    /// Get the scale perturbation for a specific layer and perturbation index.
    pub fn get_scales(&self, layer_name: &str, perturbation_idx: usize) -> Result<Option<Tensor>> {
        if let Some(stacked) = self.scales.get(layer_name) {
            Ok(Some(
                stacked.get_scales(perturbation_idx, self.config.sigma)?,
            ))
        } else {
            Ok(None)
        }
    }

    /// Replicate input tensor for all perturbations.
    ///
    /// Takes input of shape [batch, seq_len] and returns [num_perturbs * batch, seq_len].
    pub fn replicate_input(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = x.dims2()?;

        // Expand to [num_perturbs, batch, seq_len]
        let x_expanded = x
            .unsqueeze(0)?
            .expand((self.num_perturbs, batch, seq_len))?;

        // Reshape to [num_perturbs * batch, seq_len]
        x_expanded.reshape((self.num_perturbs * batch, seq_len))
    }

    /// Split batched output back into per-perturbation results.
    ///
    /// Takes output of shape [num_perturbs * batch, ...] and returns
    /// a vector of [batch, ...] tensors.
    pub fn split_output(&self, x: &Tensor, batch_size: usize) -> Result<Vec<Tensor>> {
        let dims = x.dims();
        let rest_dims: Vec<_> = dims[1..].to_vec();

        // Reshape to [num_perturbs, batch, ...]
        let mut new_shape = vec![self.num_perturbs, batch_size];
        new_shape.extend(rest_dims.iter());
        let x_split = x.reshape(&new_shape[..])?;

        // Extract each perturbation
        let mut results = Vec::with_capacity(self.num_perturbs);
        for i in 0..self.num_perturbs {
            results.push(x_split.i(i)?);
        }

        Ok(results)
    }
}

/// Trait for models that support batched perturbation evaluation.
pub trait BatchedPerturbationModel {
    /// Forward pass with batched perturbations.
    ///
    /// The input should already be replicated for all perturbations
    /// (shape [num_perturbs * batch, seq_len]).
    ///
    /// Returns logits of shape [num_perturbs * batch, vocab_size] for next-token
    /// or [num_perturbs * batch, seq_len, vocab_size] for training.
    fn forward_batched(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        perturbations: &BatchedPerturbations,
        batch_size: usize,
    ) -> Result<Tensor>;
}

/// Compute fitness for batched outputs.
///
/// Given logits from batched forward and target tokens, compute
/// fitness (e.g., negative cross-entropy loss) for each perturbation.
pub fn compute_batched_fitness(
    logits: &Tensor,
    targets: &Tensor,
    num_perturbs: usize,
    batch_size: usize,
) -> Result<Vec<f32>> {
    // logits: [num_perturbs * batch, seq_len, vocab_size] or [num_perturbs * batch, vocab_size]
    // targets: [batch, seq_len] (will be replicated)

    let logits_dims = logits.dims();
    let is_training_mode = logits_dims.len() == 3;

    let mut fitnesses = Vec::with_capacity(num_perturbs);

    if is_training_mode {
        // Training mode: logits [num_perturbs * batch, seq_len, vocab_size]
        let seq_len = logits_dims[1];
        let vocab_size = logits_dims[2];

        // Reshape logits to [num_perturbs, batch, seq_len, vocab_size]
        let logits_4d = logits.reshape((num_perturbs, batch_size, seq_len, vocab_size))?;

        for i in 0..num_perturbs {
            let logits_i = logits_4d.i(i)?; // [batch, seq_len, vocab_size]

            // Flatten for cross-entropy: [batch * seq_len, vocab_size]
            let logits_flat = logits_i.reshape((batch_size * seq_len, vocab_size))?;
            let targets_flat = targets.reshape(batch_size * seq_len)?;

            // Compute cross-entropy loss
            let log_softmax = candle_nn::ops::log_softmax(&logits_flat, 1)?;
            let loss = candle_nn::loss::nll(&log_softmax, &targets_flat)?;
            let loss_val = loss.to_scalar::<f32>()?;

            // Fitness = negative loss (higher is better)
            fitnesses.push(-loss_val);
        }
    } else {
        // Inference mode: logits [num_perturbs * batch, vocab_size]
        let vocab_size = logits_dims[1];

        // Reshape logits to [num_perturbs, batch, vocab_size]
        let logits_3d = logits.reshape((num_perturbs, batch_size, vocab_size))?;

        // Use last target token for next-token prediction
        let targets_1d = targets.i((.., targets.dim(1)? - 1))?; // [batch]

        for i in 0..num_perturbs {
            let logits_i = logits_3d.i(i)?; // [batch, vocab_size]

            let log_softmax = candle_nn::ops::log_softmax(&logits_i, 1)?;
            let loss = candle_nn::loss::nll(&log_softmax, &targets_1d)?;
            let loss_val = loss.to_scalar::<f32>()?;

            fitnesses.push(-loss_val);
        }
    }

    Ok(fitnesses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stacked_lora_apply() -> Result<()> {
        let device = Device::Cpu;
        let num_perturbs = 3;
        let batch_size = 2;
        let seq_len = 4;
        let in_features = 8;
        let out_features = 6;
        let rank = 2;

        // Create stacked LoRA
        let a_stacked = Tensor::randn(0f32, 1f32, (num_perturbs, out_features, rank), &device)?;
        let b_stacked = Tensor::randn(0f32, 1f32, (num_perturbs, in_features, rank), &device)?;

        let stacked = StackedLoRA {
            a_stacked,
            b_stacked,
            num_perturbs,
            rank,
        };

        // Create batched input
        let x = Tensor::randn(
            0f32,
            1f32,
            (num_perturbs * batch_size, seq_len, in_features),
            &device,
        )?;

        // Apply batched LoRA
        let out = stacked.apply_batched(&x, batch_size, 0.01)?;

        assert_eq!(
            out.dims(),
            &[num_perturbs * batch_size, seq_len, out_features]
        );

        Ok(())
    }

    #[test]
    fn test_replicate_and_split() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let seq_len = 4;
        let num_perturbs = 3;

        let config = BatchedEvalConfig {
            batch_size: num_perturbs,
            ..Default::default()
        };

        let perturbations = BatchedPerturbations {
            lora: HashMap::new(),
            scales: HashMap::new(),
            num_perturbs,
            config,
        };

        // Create input
        let x = Tensor::randn(0f32, 1f32, (batch_size, seq_len), &device)?;

        // Replicate
        let x_replicated = perturbations.replicate_input(&x)?;
        assert_eq!(x_replicated.dims(), &[num_perturbs * batch_size, seq_len]);

        // Create dummy output and split
        let out = Tensor::randn(0f32, 1f32, (num_perturbs * batch_size, 10), &device)?;
        let splits = perturbations.split_output(&out, batch_size)?;

        assert_eq!(splits.len(), num_perturbs);
        for split in &splits {
            assert_eq!(split.dims(), &[batch_size, 10]);
        }

        Ok(())
    }
}
