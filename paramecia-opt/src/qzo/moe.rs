//! MoE-specific support for QZO optimization
//!
//! This module provides auxiliary loss functions and utilities for fine-tuning
//! Mixture-of-Experts (MoE) models with QZO.

use candle::{Result, Tensor, D};

/// Statistics from MoE routing for load balancing
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
    pub fn new(router_logits: Tensor, selected_experts: Tensor) -> Self {
        Self {
            router_logits,
            selected_experts,
            expert_counts: None,
        }
    }

    pub fn compute_expert_counts(&mut self, num_experts: usize) -> Result<Tensor> {
        if let Some(ref counts) = self.expert_counts {
            return Ok(counts.clone());
        }

        let device = self.selected_experts.device();
        let selected_dims = self.selected_experts.dims();

        let mut counts = vec![0.0f32; num_experts];

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

    pub fn num_tokens(&self) -> Result<usize> {
        Ok(self.router_logits.dim(0)?)
    }

    pub fn num_experts(&self) -> Result<usize> {
        Ok(self.router_logits.dim(1)?)
    }
}

/// Auxiliary loss function for MoE load balancing
pub struct LoadBalanceLoss {
    /// Weight for the auxiliary loss term (typically 0.01)
    pub alpha: f64,
    /// Number of experts in the MoE layer
    pub num_experts: usize,
}

impl LoadBalanceLoss {
    pub fn new(alpha: f64, num_experts: usize) -> Self {
        Self { alpha, num_experts }
    }

    pub fn compute(&self, stats: &mut RouterStats) -> Result<Tensor> {
        let router_probs = candle_nn::ops::softmax_last_dim(&stats.router_logits)?;

        let num_tokens = stats.num_tokens()? as f64;
        let expert_counts = stats.compute_expert_counts(self.num_experts)?;

        let expert_counts = if expert_counts.dims().len() > 1 {
            expert_counts.flatten_all()?
        } else {
            expert_counts
        };

        let f_i = (expert_counts / num_tokens)?;
        let p_i = router_probs.mean(0)?;
        let p_i = if p_i.dims().len() > 1 {
            p_i.flatten_all()?
        } else {
            p_i
        };

        let loss = (&f_i * &p_i)?.sum_all()?;
        let loss = (loss * (self.num_experts as f64))?;
        let loss = (loss * self.alpha)?;

        Ok(loss)
    }
}

/// Z-loss for router logit stability
pub struct ZLoss {
    /// Weight for the Z-loss term (typically 0.001)
    pub alpha: f64,
}

impl ZLoss {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    pub fn compute(&self, stats: &RouterStats) -> Result<Tensor> {
        let logsumexp = self.log_sum_exp(&stats.router_logits)?;
        let z_loss = (&logsumexp * &logsumexp)?.mean_all()?;
        z_loss * self.alpha
    }

    fn log_sum_exp(&self, x: &Tensor) -> Result<Tensor> {
        let max_val = x.max(D::Minus1)?;
        let max_val_keepdim = max_val.unsqueeze(D::Minus1)?;
        let exp_shifted = x.broadcast_sub(&max_val_keepdim)?.exp()?;
        let sum_exp = exp_shifted.sum(D::Minus1)?;
        let log_sum = sum_exp.log()?;
        max_val + log_sum
    }
}

/// Expert usage metrics for monitoring
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
    pub fn from_stats(stats: &mut RouterStats) -> Result<Self> {
        let num_experts = stats.num_experts()?;
        let counts = stats.compute_expert_counts(num_experts)?;
        let expert_usage = counts.to_vec1::<f32>()?;

        let mean = expert_usage.iter().sum::<f32>() / num_experts as f32;
        let variance = expert_usage
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / num_experts as f32;

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

    pub fn is_balanced(&self, variance_threshold: f32) -> bool {
        self.load_variance < variance_threshold
    }

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
    use candle::Device;

    #[test]
    fn test_router_stats_expert_counts() -> Result<()> {
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
        let counts = stats.compute_expert_counts(3)?;
        let counts_vec = counts.to_vec1::<f32>()?;

        assert_eq!(counts_vec, vec![1.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn test_load_balance_loss_balanced() -> Result<()> {
        let device = Device::Cpu;
        let router_logits = Tensor::new(
            &[[1.0f32, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            &device,
        )?;
        let selected_experts = Tensor::new(&[[0u32], [1u32], [2u32]], &device)?;
        let mut stats = RouterStats::new(router_logits, selected_experts);

        let lb_loss = LoadBalanceLoss::new(0.01, 3);
        let loss = lb_loss.compute(&mut stats)?;
        let loss_val = loss.to_vec0::<f32>()?;

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
        let router_logits = Tensor::new(
            &[[10.0f32, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            &device,
        )?;
        let selected_experts = Tensor::new(&[[0u32], [0u32], [0u32]], &device)?;

        let mut stats = RouterStats::new(router_logits, selected_experts);
        let lb_loss = LoadBalanceLoss::new(0.01, 3);
        let loss = lb_loss.compute(&mut stats)?;
        let loss_val = loss.to_vec0::<f32>()?;

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

        let router_logits_moderate = Tensor::new(&[[1.0f32, 2.0, 3.0], [2.0, 1.0, 3.0]], &device)?;
        let selected_experts = Tensor::new(&[[0u32], [1u32]], &device)?;
        let stats_moderate = RouterStats::new(router_logits_moderate, selected_experts.clone());

        let z_loss = ZLoss::new(0.001);
        let loss_moderate = z_loss.compute(&stats_moderate)?.to_vec0::<f32>()?;

        let router_logits_large =
            Tensor::new(&[[10.0f32, 20.0, 30.0], [20.0, 10.0, 30.0]], &device)?;
        let stats_large = RouterStats::new(router_logits_large, selected_experts);

        let loss_large = z_loss.compute(&stats_large)?.to_vec0::<f32>()?;
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
        assert!(!metrics.is_balanced(0.01));

        Ok(())
    }
}
