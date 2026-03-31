//! MoE-specific support for QZO optimization
//!
//! This module provides auxiliary loss functions and utilities for fine-tuning
//! Mixture-of-Experts (MoE) models with QZO. The main challenges with MoE models are:
//!
//! 1. **Load Balancing**: Ensuring tokens are distributed evenly across experts
//! 2. **Router Stability**: Preventing router logits from growing unbounded
//! 3. **Expert Utilization**: Tracking which experts are being used
//!
//! # Example
//!
//! ```no_run
//! use paramecia_qzo::moe::{RouterStats, LoadBalanceLoss};
//! use paramecia_core::{Tensor, Result};
//!
//! fn example() -> Result<()> {
//!     // Collect router statistics during forward pass
//!     let stats = RouterStats::new(router_logits, selected_experts);
//!     
//!     // Compute load balancing loss
//!     let lb_loss = LoadBalanceLoss::new(0.01, 8);
//!     let aux_loss = lb_loss.compute(&stats)?;
//!     
//!     // Add to main loss
//!     let total_loss = (task_loss + aux_loss)?;
//!     Ok(())
//! }
//! ```

use paramecia_core::{Result, Tensor, D};

/// Statistics from MoE routing for load balancing
///
/// This struct captures the routing decisions made by an MoE layer,
/// which are used to compute auxiliary losses for load balancing.
#[derive(Debug, Clone)]
pub struct RouterStats {
    /// Router logits before softmax: [batch * seq_len, num_experts]
    pub router_logits: Tensor,
    /// Selected expert indices: [batch * seq_len, num_experts_per_tok]
    pub selected_experts: Tensor,
    /// Cached expert usage counts: [num_experts]
    expert_counts: Option<Tensor>,
}

impl RouterStats {
    /// Create new router statistics
    ///
    /// # Arguments
    ///
    /// * `router_logits` - Raw logits from the router network
    /// * `selected_experts` - Indices of experts selected for each token
    pub fn new(router_logits: Tensor, selected_experts: Tensor) -> Self {
        Self {
            router_logits,
            selected_experts,
            expert_counts: None,
        }
    }

    /// Compute expert usage counts from selected experts
    ///
    /// Returns a tensor of shape [num_experts] where each element is the
    /// number of tokens that were routed to that expert.
    pub fn compute_expert_counts(&mut self, num_experts: usize) -> Result<Tensor> {
        if let Some(ref counts) = self.expert_counts {
            return Ok(counts.clone());
        }

        let device = self.selected_experts.device();
        let selected_dims = self.selected_experts.dims();

        // Count occurrences of each expert
        let mut counts = vec![0.0f32; num_experts];

        // Handle both 1D and 2D selected_experts tensors
        if selected_dims.len() == 1 {
            let selected = self.selected_experts.to_vec1::<u32>()?;
            for &expert_idx in selected.iter() {
                if (expert_idx as usize) < num_experts {
                    counts[expert_idx as usize] += 1.0;
                }
            }
        } else {
            let selected = self.selected_experts.to_vec2::<u32>()?;
            for row in selected.iter() {
                for &expert_idx in row.iter() {
                    if (expert_idx as usize) < num_experts {
                        counts[expert_idx as usize] += 1.0;
                    }
                }
            }
        }

        let counts_tensor = Tensor::from_vec(counts, num_experts, device)?;
        self.expert_counts = Some(counts_tensor.clone());
        Ok(counts_tensor)
    }

    /// Get the number of tokens (batch_size * seq_len)
    pub fn num_tokens(&self) -> Result<usize> {
        self.router_logits.dim(0)
    }

    /// Get the number of experts
    pub fn num_experts(&self) -> Result<usize> {
        self.router_logits.dim(1)
    }
}

/// Auxiliary loss function for MoE load balancing
///
/// Implements the load balancing loss from the Switch Transformer paper:
/// "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
/// (https://arxiv.org/abs/2101.03961)
///
/// The loss encourages uniform distribution of tokens across experts:
/// ```text
/// aux_loss = alpha * num_experts * sum_i(f_i * P_i)
/// ```
///
/// where:
/// - `f_i` = fraction of tokens routed to expert i
/// - `P_i` = average router probability for expert i
/// - `alpha` = loss weight (typically 0.01)
///
/// # Example
///
/// ```no_run
/// use paramecia_qzo::moe::{LoadBalanceLoss, RouterStats};
///
/// let lb_loss = LoadBalanceLoss::new(0.01, 8);  // alpha=0.01, 8 experts
/// let aux_loss = lb_loss.compute(&router_stats)?;
/// # Ok::<(), paramecia_core::Error>(())
/// ```
pub struct LoadBalanceLoss {
    /// Weight for the auxiliary loss term (typically 0.01)
    pub alpha: f64,
    /// Number of experts in the MoE layer
    pub num_experts: usize,
}

impl LoadBalanceLoss {
    /// Create a new load balancing loss
    ///
    /// # Arguments
    ///
    /// * `alpha` - Loss weight (typically 0.01)
    /// * `num_experts` - Number of experts in the layer
    pub fn new(alpha: f64, num_experts: usize) -> Self {
        Self { alpha, num_experts }
    }

    /// Compute the load balancing auxiliary loss
    ///
    /// # Arguments
    ///
    /// * `stats` - Router statistics from the forward pass
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the auxiliary loss value
    pub fn compute(&self, stats: &mut RouterStats) -> Result<Tensor> {
        // Compute router probabilities via softmax
        let router_probs = paramecia_nn::ops::softmax_last_dim(&stats.router_logits)?;

        // Compute f_i: fraction of tokens assigned to each expert
        // Shape: [num_experts]
        let num_tokens = stats.num_tokens()? as f64;
        let expert_counts = stats.compute_expert_counts(self.num_experts)?;

        // Ensure expert_counts has the right shape [num_experts]
        let expert_counts = if expert_counts.dims().len() > 1 {
            expert_counts.flatten_all()?
        } else {
            expert_counts
        };

        let f_i = (expert_counts / num_tokens)?;

        // Compute P_i: average routing probability per expert
        // Shape: [num_experts]
        let p_i = router_probs.mean(0)?;

        // Ensure p_i also has shape [num_experts]
        let p_i = if p_i.dims().len() > 1 {
            p_i.flatten_all()?
        } else {
            p_i
        };

        // Both should now have shape [num_experts]
        // aux_loss = num_experts * sum(f_i * P_i)
        let loss = (&f_i * &p_i)?.sum_all()?;
        let loss = (loss * (self.num_experts as f64))?;
        let loss = (loss * self.alpha)?;

        Ok(loss)
    }
}

/// Z-loss for router logit stability
///
/// Implements the Z-loss from "ST-MoE: Designing Stable and Transferable Sparse Expert Models"
/// (https://arxiv.org/abs/2202.08906)
///
/// The Z-loss penalizes large router logits to improve training stability:
/// ```text
/// z_loss = alpha * mean(log(sum(exp(router_logits)))^2)
/// ```
///
/// This encourages the router to produce moderate logit values and prevents
/// numerical instability from exponentially large values.
///
/// # Example
///
/// ```no_run
/// use paramecia_qzo::moe::{ZLoss, RouterStats};
///
/// let z_loss = ZLoss::new(0.001);  // alpha=0.001
/// let aux_loss = z_loss.compute(&router_stats)?;
/// # Ok::<(), paramecia_core::Error>(())
/// ```
pub struct ZLoss {
    /// Weight for the Z-loss term (typically 0.001)
    pub alpha: f64,
}

impl ZLoss {
    /// Create a new Z-loss
    ///
    /// # Arguments
    ///
    /// * `alpha` - Loss weight (typically 0.001)
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Compute the Z-loss for router stability
    ///
    /// # Arguments
    ///
    /// * `stats` - Router statistics from the forward pass
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the Z-loss value
    pub fn compute(&self, stats: &RouterStats) -> Result<Tensor> {
        // log(sum(exp(x))) is computed via logsumexp for numerical stability
        let logsumexp = self.log_sum_exp(&stats.router_logits)?;

        // Z-loss = mean((logsumexp)^2)
        let z_loss = (&logsumexp * &logsumexp)?.mean_all()?;
        z_loss * self.alpha
    }

    /// Compute log-sum-exp along the last dimension
    fn log_sum_exp(&self, x: &Tensor) -> Result<Tensor> {
        // LogSumExp(x) = log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        let max_val = x.max(D::Minus1)?;
        let max_val_keepdim = max_val.unsqueeze(D::Minus1)?;
        let exp_shifted = x.broadcast_sub(&max_val_keepdim)?.exp()?;
        let sum_exp = exp_shifted.sum(D::Minus1)?;
        let log_sum = sum_exp.log()?;
        max_val + log_sum
    }
}

/// Expert usage metrics for monitoring
///
/// Tracks how evenly tokens are distributed across experts.
#[derive(Debug, Clone)]
pub struct ExpertMetrics {
    /// Usage count per expert
    pub expert_usage: Vec<f32>,
    /// Variance in expert usage (lower is better)
    pub load_variance: f32,
    /// Entropy of expert distribution (higher is better for balance)
    pub routing_entropy: f32,
}

impl ExpertMetrics {
    /// Compute metrics from router statistics
    pub fn from_stats(stats: &mut RouterStats) -> Result<Self> {
        let num_experts = stats.num_experts()?;
        let counts = stats.compute_expert_counts(num_experts)?;
        let expert_usage = counts.to_vec1::<f32>()?;

        // Compute variance
        let mean = expert_usage.iter().sum::<f32>() / num_experts as f32;
        let variance = expert_usage
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / num_experts as f32;

        // Compute entropy
        let total: f32 = expert_usage.iter().sum();
        let entropy = if total > 0.0 {
            expert_usage
                .iter()
                .map(|&count| {
                    if count > 0.0 {
                        let p = count / total;
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        } else {
            0.0
        };

        Ok(Self {
            expert_usage,
            load_variance: variance,
            routing_entropy: entropy,
        })
    }

    /// Check if experts are well-balanced (variance below threshold)
    pub fn is_balanced(&self, variance_threshold: f32) -> bool {
        self.load_variance < variance_threshold
    }

    /// Get the maximum imbalance ratio (max_usage / min_usage)
    pub fn imbalance_ratio(&self) -> f32 {
        let max_usage = self
            .expert_usage
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_usage = self
            .expert_usage
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        if min_usage > 0.0 {
            max_usage / min_usage
        } else {
            f32::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paramecia_core::Device;

    #[test]
    fn test_router_stats_expert_counts() -> Result<()> {
        let device = Device::Cpu;

        // Create fake router logits: 4 tokens, 3 experts
        let router_logits = Tensor::new(
            &[
                [1.0f32, 2.0, 3.0],
                [2.0, 1.0, 3.0],
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
            ],
            &device,
        )?;

        // Selected experts (top-1): expert indices
        // Token 0 -> expert 2, Token 1 -> expert 2, Token 2 -> expert 0, Token 3 -> expert 1
        let selected_experts = Tensor::new(&[[2u32], [2u32], [0u32], [1u32]], &device)?;

        let mut stats = RouterStats::new(router_logits, selected_experts);

        let counts = stats.compute_expert_counts(3)?;
        let counts_vec = counts.to_vec1::<f32>()?;

        assert_eq!(counts_vec, vec![1.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_load_balance_loss_balanced() -> Result<()> {
        let device = Device::Cpu;

        // Create balanced routing: each expert gets equal probability
        let router_logits = Tensor::new(
            &[[1.0f32, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            &device,
        )?;

        // Balanced selection: each expert selected once
        let selected_experts = Tensor::new(&[[0u32], [1u32], [2u32]], &device)?;

        let mut stats = RouterStats::new(router_logits, selected_experts);

        let lb_loss = LoadBalanceLoss::new(0.01, 3);
        let loss = lb_loss.compute(&mut stats)?;
        let loss_val = loss.to_vec0::<f32>()?;

        // With perfect balance, f_i = 1/3 for all i, P_i = 1/3 for all i
        // loss = 0.01 * 3 * sum(1/3 * 1/3) = 0.01 * 3 * (1/9 + 1/9 + 1/9) = 0.01 * 3 * 1/3 = 0.01
        assert!(
            (loss_val - 0.01).abs() < 1e-5,
            "Expected ~0.01, got {}",
            loss_val
        );
        Ok(())
    }

    #[test]
    fn test_load_balance_loss_imbalanced() -> Result<()> {
        let device = Device::Cpu;

        // Create imbalanced routing: one expert gets high probability
        let router_logits = Tensor::new(
            &[[10.0f32, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            &device,
        )?;

        // Imbalanced selection: all tokens go to expert 0
        let selected_experts = Tensor::new(&[[0u32], [0u32], [0u32]], &device)?;

        let mut stats = RouterStats::new(router_logits, selected_experts);

        let lb_loss = LoadBalanceLoss::new(0.01, 3);
        let loss = lb_loss.compute(&mut stats)?;
        let loss_val = loss.to_vec0::<f32>()?;

        // With complete imbalance, loss should be higher than balanced case
        assert!(
            loss_val > 0.01,
            "Expected loss > 0.01 for imbalanced routing, got {}",
            loss_val
        );
        Ok(())
    }

    #[test]
    fn test_z_loss() -> Result<()> {
        let device = Device::Cpu;

        // Test with moderate logits
        let router_logits_moderate = Tensor::new(&[[1.0f32, 2.0, 3.0], [2.0, 1.0, 3.0]], &device)?;
        let selected_experts = Tensor::new(&[[0u32], [1u32]], &device)?;
        let stats_moderate = RouterStats::new(router_logits_moderate, selected_experts.clone());

        let z_loss = ZLoss::new(0.001);
        let loss_moderate = z_loss.compute(&stats_moderate)?.to_vec0::<f32>()?;

        // Test with large logits
        let router_logits_large =
            Tensor::new(&[[10.0f32, 20.0, 30.0], [20.0, 10.0, 30.0]], &device)?;
        let stats_large = RouterStats::new(router_logits_large, selected_experts);

        let loss_large = z_loss.compute(&stats_large)?.to_vec0::<f32>()?;

        // Larger logits should produce larger Z-loss
        assert!(
            loss_large > loss_moderate,
            "Z-loss should be higher for larger logits"
        );
        Ok(())
    }

    #[test]
    fn test_expert_metrics() -> Result<()> {
        let device = Device::Cpu;

        let router_logits = Tensor::new(
            &[
                [1.0f32, 2.0, 3.0],
                [2.0, 1.0, 3.0],
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
            ],
            &device,
        )?;
        let selected_experts = Tensor::new(&[[2u32], [2u32], [0u32], [1u32]], &device)?;

        let mut stats = RouterStats::new(router_logits, selected_experts);
        let metrics = ExpertMetrics::from_stats(&mut stats)?;

        assert_eq!(metrics.expert_usage.len(), 3);
        assert!(metrics.routing_entropy > 0.0);
        assert!(!metrics.is_balanced(0.01)); // Should not be perfectly balanced

        Ok(())
    }

    #[test]
    fn test_router_stats_multiple_experts_per_token() -> Result<()> {
        let device = Device::Cpu;

        // 3 tokens, 4 experts, top-2 routing
        let router_logits = Tensor::new(
            &[
                [1.0f32, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [2.0, 2.0, 3.0, 1.0],
            ],
            &device,
        )?;

        // Each token selects 2 experts
        let selected_experts = Tensor::new(&[[3u32, 2], [0, 1], [2, 1]], &device)?;

        let mut stats = RouterStats::new(router_logits, selected_experts);
        let counts = stats.compute_expert_counts(4)?;
        let counts_vec = counts.to_vec1::<f32>()?;

        // Expert 0: 1, Expert 1: 2, Expert 2: 2, Expert 3: 1
        assert_eq!(counts_vec, vec![1.0, 2.0, 2.0, 1.0]);

        Ok(())
    }
}
