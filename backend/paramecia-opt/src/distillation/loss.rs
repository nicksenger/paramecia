//! Distillation loss computation using sparse probability representation.
//!
//! The teacher model outputs are stored as:
//! - Top-k token IDs and log probabilities (typically k=100)
//! - Randomly sampled tail token IDs and log probabilities (typically 20)
//! - Total tail probability mass for importance weighting
//!
//! The KL-divergence loss is computed as:
//!
//! L = Σ_i p_teacher(i) * [log p_teacher(i) - log p_student(i)]
//!
//! For the sparse representation:
//! - Top-k terms are computed exactly
//! - Tail terms are importance-weighted by (tail_mass / n_tail_samples)

use paramecia_core::{Device, Result, Tensor, D};

use super::TuningData;

/// Configuration for distillation loss computation.
#[derive(Debug, Clone)]
pub struct DistillationLossConfig {
    /// Weight for the main distillation (KL-divergence) loss
    pub distill_weight: f64,
    /// Weight for Z-loss (router stability)
    pub z_loss_weight: f64,
    /// Weight for load balance loss (expert utilization)
    pub lb_loss_weight: f64,
    /// Temperature for softening distributions (1.0 = no softening)
    pub temperature: f64,
    /// Minimum probability for numerical stability
    pub min_prob: f64,
}

impl Default for DistillationLossConfig {
    fn default() -> Self {
        Self {
            distill_weight: 1.0,
            z_loss_weight: 0.001,
            lb_loss_weight: 0.01,
            temperature: 1.0,
            min_prob: 1e-10,
        }
    }
}

/// Configuration for Multi-Token Prediction (MTP) loss computation.
///
/// MTP extends the loss computation to include predictions at multiple
/// future positions, providing higher-resolution gradient signals for
/// zeroth-order optimization.
#[derive(Debug, Clone)]
pub struct MtpLossConfig {
    /// Number of MTP depths to use (default: 3)
    /// Each depth predicts one additional token into the future.
    pub num_depths: usize,
    /// Weight decay factor: λ_i = decay^i (default: 0.7)
    /// Controls how much weight is given to further predictions.
    pub decay_factor: f64,
    /// Whether to normalize weights to sum=1 (default: true)
    pub normalize_weights: bool,
}

impl Default for MtpLossConfig {
    fn default() -> Self {
        Self {
            num_depths: 3,
            // Use 0.5 decay for soft-target distillation - more aggressive than 0.7
            // to emphasize near-term predictions where gradient signal is stronger
            decay_factor: 0.5,
            normalize_weights: true,
        }
    }
}

impl MtpLossConfig {
    /// Compute the weights for main model and each MTP depth.
    ///
    /// Returns weights [w_main, w_mtp_0, w_mtp_1, ...] where:
    /// - w_main = 1.0 (main model loss)
    /// - w_mtp_i = decay_factor^i
    ///
    /// If normalize_weights is true, all weights are scaled to sum to 1.
    pub fn compute_weights(&self) -> Vec<f64> {
        let mut weights = Vec::with_capacity(1 + self.num_depths);
        weights.push(1.0); // Main model weight

        for i in 0..self.num_depths {
            weights.push(self.decay_factor.powi(i as i32));
        }

        if self.normalize_weights {
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }
}

/// Distillation loss computer.
pub struct DistillationLoss {
    config: DistillationLossConfig,
}

impl DistillationLoss {
    pub fn new(config: DistillationLossConfig) -> Self {
        Self { config }
    }

    /// Compute the sparse KL-divergence loss for a batch of positions.
    ///
    /// # Arguments
    /// * `student_logits` - Model output logits [batch, seq_len, vocab_size] or [seq_len, vocab_size]
    /// * `tuning_data` - Teacher probabilities and metadata
    /// * `chunk_assistant_positions` - List of (chunk_position, assistant_data_index) pairs
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Scalar loss tensor
    pub fn compute_distillation_loss(
        &self,
        student_logits: &Tensor,
        tuning_data: &TuningData,
        chunk_assistant_positions: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor> {
        if chunk_assistant_positions.is_empty() {
            // No assistant tokens in this chunk
            return Tensor::new(0.0f32, device);
        }

        let logits_shape = student_logits.dims();
        let (seq_dim, vocab_size) = if logits_shape.len() == 3 {
            // [batch, seq, vocab] - squeeze batch dim
            let logits_2d = student_logits.squeeze(0)?;
            (logits_2d.dim(0)?, logits_2d.dim(1)?)
        } else {
            // [seq, vocab]
            (student_logits.dim(0)?, student_logits.dim(1)?)
        };

        let student_logits_2d = if logits_shape.len() == 3 {
            student_logits.squeeze(0)?
        } else {
            student_logits.clone()
        };

        // Apply temperature scaling
        let scaled_logits = if (self.config.temperature - 1.0).abs() > 1e-6 {
            (&student_logits_2d / self.config.temperature)?
        } else {
            student_logits_2d
        };

        // Compute log softmax for student distribution
        let student_log_probs = paramecia_nn::ops::log_softmax(&scaled_logits, D::Minus1)?;

        // Transfer entire log_probs tensor to CPU once (avoid per-position GPU sync)
        let student_log_probs_cpu = student_log_probs.to_vec2::<f32>()?;

        let mut total_loss = 0.0f32;
        let mut n_positions = 0;

        for &(chunk_pos, data_idx) in chunk_assistant_positions {
            if chunk_pos >= seq_dim {
                continue;
            }
            if data_idx >= tuning_data.raw_probs.len() {
                return Err(paramecia_core::Error::Msg(format!(
                    "Assistant data index {} is out of range for raw_probs (len={})",
                    data_idx,
                    tuning_data.raw_probs.len()
                )));
            }

            // Get student log probs for this position (already on CPU)
            let pos_log_probs_vec = &student_log_probs_cpu[chunk_pos];

            let mut position_loss = 0.0f32;

            // Dense raw-probability path: full-vocab KL.
            if let Some(raw_probs) = tuning_data.raw_probs[data_idx].as_ref() {
                if raw_probs.len() != vocab_size {
                    return Err(paramecia_core::Error::Msg(format!(
                        "Raw teacher distribution length {} does not match student vocab size {}",
                        raw_probs.len(),
                        vocab_size
                    )));
                }

                let raw_sum: f32 = raw_probs
                    .iter()
                    .copied()
                    .filter(|p| p.is_finite() && *p > 0.0)
                    .sum();
                if !raw_sum.is_finite() || raw_sum <= 0.0 {
                    return Err(paramecia_core::Error::Msg(
                        "Raw teacher distribution must have positive finite mass".to_string(),
                    ));
                }

                for (token_id, &raw_prob) in raw_probs.iter().enumerate() {
                    if !raw_prob.is_finite() || raw_prob <= 0.0 {
                        continue;
                    }

                    let teacher_prob = (raw_prob / raw_sum).max(self.config.min_prob as f32);
                    let teacher_log_prob = teacher_prob.ln();
                    let student_log_prob = pos_log_probs_vec[token_id];

                    // KL = Σ p_teacher * (log p_teacher - log p_student)
                    position_loss += teacher_prob * (teacher_log_prob - student_log_prob);
                }
            } else {
                // Sparse top-k + tail path.
                let teacher_top_k_ids = &tuning_data.top_k_token_ids[data_idx];
                let teacher_top_k_log_probs = &tuning_data.top_k_log_probs[data_idx];
                let teacher_tail_ids = &tuning_data.tail_token_ids[data_idx];
                let teacher_tail_log_probs = &tuning_data.tail_log_probs[data_idx];

                // KL divergence for top-k tokens:
                // KL = Σ p_teacher * (log p_teacher - log p_student)
                for (i, &token_id) in teacher_top_k_ids.iter().enumerate() {
                    let token_id = token_id as usize;
                    if token_id >= vocab_size {
                        continue;
                    }

                    let teacher_log_prob = teacher_top_k_log_probs[i];
                    let teacher_prob = teacher_log_prob.exp().max(self.config.min_prob as f32);
                    let student_log_prob = pos_log_probs_vec[token_id];

                    // KL contribution: p_teacher * (log p_teacher - log p_student)
                    position_loss += teacher_prob * (teacher_log_prob - student_log_prob);
                }

                // KL divergence for sampled tail tokens.
                if !teacher_tail_ids.is_empty() {
                    for (i, &token_id) in teacher_tail_ids.iter().enumerate() {
                        let token_id = token_id as usize;
                        if token_id >= vocab_size {
                            continue;
                        }

                        let teacher_log_prob = teacher_tail_log_probs[i];
                        let teacher_prob = teacher_log_prob.exp().max(self.config.min_prob as f32);
                        let student_log_prob = pos_log_probs_vec[token_id];

                        position_loss += teacher_prob * (teacher_log_prob - student_log_prob);
                    }
                }
            }

            total_loss += position_loss;
            n_positions += 1;
        }

        if n_positions > 0 {
            total_loss /= n_positions as f32;
        }

        Tensor::new(total_loss * self.config.distill_weight as f32, device)
    }

    /// Compute Z-loss from router statistics.
    ///
    /// Z-loss = mean(log_sum_exp(router_logits)^2)
    ///
    /// This encourages router logits to stay near zero for numerical stability.
    pub fn compute_z_loss(
        &self,
        router_stats: &[(Tensor, Tensor)],
        device: &Device,
    ) -> Result<Tensor> {
        if router_stats.is_empty() || self.config.z_loss_weight <= 0.0 {
            return Tensor::new(0.0f32, device);
        }

        let mut total_z_loss = Tensor::new(0.0f32, device)?;
        let mut n_layers = 0;

        for (router_logits, _) in router_stats {
            // router_logits shape: [batch, seq, num_experts] or [seq, num_experts]
            let logits_flat = router_logits.flatten_all()?;
            let dims = router_logits.dims();

            // Reshape to [n_tokens, num_experts]
            let num_experts = *dims.last().unwrap_or(&1);
            let n_tokens = logits_flat.elem_count() / num_experts;

            if n_tokens == 0 || num_experts == 0 {
                continue;
            }

            let logits_2d = logits_flat.reshape((n_tokens, num_experts))?;

            // Compute log-sum-exp for each token
            let max_logits = logits_2d.max_keepdim(D::Minus1)?;
            let shifted = logits_2d.broadcast_sub(&max_logits)?;
            let exp_sum = shifted.exp()?.sum_keepdim(D::Minus1)?;
            let log_sum_exp = (exp_sum.log()? + max_logits)?;

            // Z-loss = mean(log_sum_exp^2)
            let z_loss = log_sum_exp.sqr()?.mean_all()?;
            // Transfer to target device for cross-GPU accumulation
            let z_loss = z_loss.to_device(device)?;

            total_z_loss = (total_z_loss + z_loss)?;
            n_layers += 1;
        }

        if n_layers > 0 {
            total_z_loss = (total_z_loss / n_layers as f64)?;
        }

        total_z_loss * self.config.z_loss_weight
    }

    /// Compute load balance loss from router statistics.
    ///
    /// LB_loss = num_experts * Σ_i (f_i * p_i)
    ///
    /// Where:
    /// - f_i = fraction of tokens routed to expert i
    /// - p_i = mean routing probability for expert i
    pub fn compute_load_balance_loss(
        &self,
        router_stats: &[(Tensor, Tensor)],
        device: &Device,
    ) -> Result<Tensor> {
        if router_stats.is_empty() || self.config.lb_loss_weight <= 0.0 {
            return Tensor::new(0.0f32, device);
        }

        let mut total_lb_loss = Tensor::new(0.0f32, device)?;
        let mut n_layers = 0;

        for (router_logits, selected_experts) in router_stats {
            let dims = router_logits.dims();
            let num_experts = *dims.last().unwrap_or(&1);

            if num_experts == 0 {
                continue;
            }

            // Flatten to [n_tokens, num_experts]
            let n_tokens = router_logits.elem_count() / num_experts;
            if n_tokens == 0 {
                continue;
            }

            let logits_2d = router_logits
                .flatten_all()?
                .reshape((n_tokens, num_experts))?;

            // Compute routing probabilities
            let routing_probs = paramecia_nn::ops::softmax(&logits_2d, D::Minus1)?;

            // p_i = mean probability assigned to expert i
            let mean_probs = routing_probs.mean(0)?; // [num_experts]

            // f_i = fraction of tokens where expert i is selected
            // selected_experts shape: [batch, seq, k] or [seq, k]
            let selected_flat = selected_experts.flatten_all()?.to_vec1::<u32>()?;

            let mut expert_counts = vec![0.0f32; num_experts];
            for &expert_id in &selected_flat {
                if (expert_id as usize) < num_experts {
                    expert_counts[expert_id as usize] += 1.0;
                }
            }

            let total_selections = selected_flat.len() as f32;
            if total_selections > 0.0 {
                for count in &mut expert_counts {
                    *count /= total_selections;
                }
            }

            // Create f_i on the same device as mean_probs to avoid cross-device multiply
            let probs_device = mean_probs.device().clone();
            let f_i = Tensor::new(expert_counts.as_slice(), &probs_device)?;

            // LB loss = num_experts * sum(f_i * p_i)
            let lb_loss = ((&f_i * &mean_probs)?.sum_all()? * num_experts as f64)?;
            // Transfer to target device for cross-GPU accumulation
            let lb_loss = lb_loss.to_device(device)?;

            total_lb_loss = (total_lb_loss + lb_loss)?;
            n_layers += 1;
        }

        if n_layers > 0 {
            total_lb_loss = (total_lb_loss / n_layers as f64)?;
        }

        total_lb_loss * self.config.lb_loss_weight
    }

    /// Compute total loss including distillation and auxiliary losses.
    pub fn compute_total_loss(
        &self,
        student_logits: &Tensor,
        router_stats: &[(Tensor, Tensor)],
        tuning_data: &TuningData,
        chunk_assistant_positions: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor> {
        let distill_loss = self.compute_distillation_loss(
            student_logits,
            tuning_data,
            chunk_assistant_positions,
            device,
        )?;

        let z_loss = self.compute_z_loss(router_stats, device)?;
        let lb_loss = self.compute_load_balance_loss(router_stats, device)?;

        (distill_loss + z_loss)? + lb_loss
    }

    /// Compute MTP loss across main model and MTP depth predictions.
    ///
    /// Label alignment for MTP:
    /// - Main model (depth 0): position i predicts token i+1
    /// - MTP depth d: position i predicts token i+d+2
    ///
    /// ```text
    /// Position:  0   1   2   3   4   5
    /// Token:     A   B   C   D   E   F
    /// Main:      B   C   D   E   F   -
    /// MTP d=0:   C   D   E   F   -   -
    /// MTP d=1:   D   E   F   -   -   -
    /// ```
    ///
    /// # Arguments
    /// * `main_logits` - Main model logits [batch, seq_len, vocab]
    /// * `mtp_logits` - MTP logits for each depth, each [batch, valid_len, vocab]
    /// * `router_stats` - Router statistics from main model
    /// * `tuning_data` - Teacher probabilities
    /// * `chunk_assistant_positions` - (chunk_pos, data_idx) pairs for loss computation
    /// * `mtp_config` - MTP configuration
    /// * `device` - Device for tensor operations
    #[allow(clippy::too_many_arguments)]
    pub fn compute_mtp_loss(
        &self,
        main_logits: &Tensor,
        mtp_logits: &[Tensor],
        router_stats: &[(Tensor, Tensor)],
        tuning_data: &TuningData,
        chunk_assistant_positions: &[(usize, usize)],
        mtp_config: &MtpLossConfig,
        device: &Device,
    ) -> Result<Tensor> {
        if chunk_assistant_positions.is_empty() {
            return Tensor::new(0.0f32, device);
        }

        let weights = mtp_config.compute_weights();

        // Compute main model distillation loss (weight = weights[0])
        let main_loss = self.compute_distillation_loss(
            main_logits,
            tuning_data,
            chunk_assistant_positions,
            device,
        )?;

        let mut total_loss = (main_loss * weights[0])?;

        // Compute MTP losses for each depth
        // MTP depth d: position i predicts token i+d+2, so we shift labels
        for (depth, mtp_logit) in mtp_logits.iter().enumerate() {
            if depth + 1 >= weights.len() {
                break;
            }

            let weight = weights[depth + 1];
            if weight < 1e-10 {
                continue;
            }

            // Shift positions: MTP depth d needs label at original_pos + d + 1
            // (main model predicts i+1, MTP d=0 predicts i+2, etc.)
            let label_shift = depth + 1;

            let shifted_positions: Vec<(usize, usize)> = chunk_assistant_positions
                .iter()
                .filter_map(|&(chunk_pos, data_idx)| {
                    // Check if shifted data_idx is valid
                    let shifted_data_idx = data_idx + label_shift;
                    if shifted_data_idx < tuning_data.n_assistant_tokens() {
                        // MTP logits are shorter: valid_len = seq_len - depth - 1
                        // Position in MTP logits corresponds to original chunk position
                        Some((chunk_pos, shifted_data_idx))
                    } else {
                        None
                    }
                })
                .collect();

            if shifted_positions.is_empty() {
                continue;
            }

            let mtp_depth_loss =
                self.compute_distillation_loss(mtp_logit, tuning_data, &shifted_positions, device)?;

            total_loss = (total_loss + (mtp_depth_loss * weight)?)?;
        }

        // Add auxiliary losses (Z-loss and load balance loss)
        let z_loss = self.compute_z_loss(router_stats, device)?;
        let lb_loss = self.compute_load_balance_loss(router_stats, device)?;

        (total_loss + z_loss)? + lb_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DistillationLossConfig::default();
        assert!((config.distill_weight - 1.0).abs() < 1e-6);
        assert!((config.z_loss_weight - 0.001).abs() < 1e-6);
        assert!((config.lb_loss_weight - 0.01).abs() < 1e-6);
    }
}
