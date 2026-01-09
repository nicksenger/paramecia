//! QZO: Quantized Zeroth-Order Optimization
//!
//! This module provides memory-efficient fine-tuning for quantized models using
//! zeroth-order optimization. Instead of storing gradients for all parameters,
//! QZO keeps the model quantized, only optimizes quantization scales, and uses
//! finite differences instead of backpropagation.

pub mod moe;

use candle::{Result, Tensor, Var};
use moe::{LoadBalanceLoss, RouterStats, ZLoss};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Parameters for the QZO optimizer
#[derive(Clone, Debug)]
pub struct ParamsQZO {
    /// Learning rate (η in the paper)
    pub lr: f64,
    /// Perturbation magnitude for finite differences (ε in the paper)
    pub epsilon: f64,
    /// Number of random directions to sample per update (typically 1)
    pub num_samples: usize,
    /// Clipping threshold for directional derivative clipping (C in the paper)
    /// Prevents unstable updates from large gradients.
    /// - For losses ~1-10: use 0.01-0.1
    /// - For losses ~100-1000: use 1.0-10.0
    /// - Too small = no learning, too large = instability
    pub clip_threshold: f64,
    /// Optional: Load balancing loss weight for MoE (typical: 0.01, or None for non-MoE)
    pub load_balance_alpha: Option<f64>,
    /// Optional: Z-loss weight for MoE router stability (typical: 0.001, or None)
    pub z_loss_alpha: Option<f64>,
}

impl Default for ParamsQZO {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            epsilon: 1e-3,
            num_samples: 1,
            clip_threshold: 0.01,
            load_balance_alpha: None,
            z_loss_alpha: None,
        }
    }
}

/// Loss function output that can include auxiliary losses for MoE
#[derive(Debug, Clone)]
pub struct LossOutput {
    /// Main task loss (e.g., cross-entropy, MSE)
    pub task_loss: Tensor,
    /// Optional router statistics for MoE load balancing
    pub router_stats: Option<Vec<RouterStats>>,
}

impl LossOutput {
    /// Create a simple loss without auxiliary components (for non-MoE models)
    pub fn from_tensor(loss: Tensor) -> Self {
        Self {
            task_loss: loss,
            router_stats: None,
        }
    }

    /// Create a loss with MoE router statistics
    pub fn with_router_stats(task_loss: Tensor, router_stats: Vec<RouterStats>) -> Self {
        Self {
            task_loss,
            router_stats: Some(router_stats),
        }
    }

    /// Compute total loss including auxiliary terms
    pub fn total_loss(
        &mut self,
        load_balance_alpha: Option<f64>,
        z_loss_alpha: Option<f64>,
        num_experts: usize,
    ) -> Result<Tensor> {
        let mut total = self.task_loss.clone();
        let task_val = total.to_vec0::<f32>().unwrap_or(0.0);

        if let Some(ref mut stats) = self.router_stats {
            let num_layers = stats.len();

            // Add load balancing loss (averaged over layers)
            if let Some(alpha) = load_balance_alpha {
                let lb_loss = LoadBalanceLoss::new(alpha, num_experts);
                let mut lb_sum = Tensor::new(0f32, self.task_loss.device())?;
                for stat in stats.iter_mut() {
                    let aux_loss = lb_loss.compute(stat)?;
                    lb_sum = (lb_sum + aux_loss)?;
                }
                if num_layers > 0 {
                    lb_sum = (lb_sum / num_layers as f64)?;
                }
                let lb_val = lb_sum.to_vec0::<f32>().unwrap_or(0.0);
                if task_val < 10.0 && lb_val > 10.0 {
                    eprintln!(
                        "WARNING: LB loss ({:.2}) >> task loss ({:.2})",
                        lb_val, task_val
                    );
                }
                total = (total + lb_sum)?;
            }

            // Add Z-loss (averaged over layers)
            if let Some(alpha) = z_loss_alpha {
                let z_loss = ZLoss::new(alpha);
                let mut z_sum = Tensor::new(0f32, self.task_loss.device())?;
                for stat in stats.iter() {
                    let aux_loss = z_loss.compute(stat)?;
                    z_sum = (z_sum + aux_loss)?;
                }
                if num_layers > 0 {
                    z_sum = (z_sum / num_layers as f64)?;
                }
                total = (total + z_sum)?;
            }
        }

        Ok(total)
    }
}

/// QZO: Quantized Zeroth-Order Optimizer
pub struct QZO {
    /// The scaling factor variables to optimize
    scale_vars: Vec<Var>,
    /// Optimizer parameters
    params: ParamsQZO,
    /// Random number generator for perturbations
    rng: StdRng,
    /// Track number of experts for MoE models
    num_experts: Option<usize>,
}

impl QZO {
    /// Create a new QZO optimizer
    pub fn new(scale_vars: Vec<Var>, params: ParamsQZO) -> Result<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Ok(Self {
            scale_vars,
            params,
            rng: StdRng::seed_from_u64(seed),
            num_experts: None,
        })
    }

    /// Create QZO with a specific seed for reproducibility
    pub fn new_with_seed(scale_vars: Vec<Var>, params: ParamsQZO, seed: u64) -> Result<Self> {
        Ok(Self {
            scale_vars,
            params,
            rng: StdRng::seed_from_u64(seed),
            num_experts: None,
        })
    }

    /// Create a new QZO optimizer for MoE models
    pub fn new_moe(scale_vars: Vec<Var>, params: ParamsQZO, num_experts: usize) -> Result<Self> {
        let mut optimizer = Self::new(scale_vars, params)?;
        optimizer.num_experts = Some(num_experts);
        Ok(optimizer)
    }

    /// Create a new QZO optimizer for MoE models with specific seed
    pub fn new_moe_with_seed(
        scale_vars: Vec<Var>,
        params: ParamsQZO,
        num_experts: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut optimizer = Self::new_with_seed(scale_vars, params, seed)?;
        optimizer.num_experts = Some(num_experts);
        Ok(optimizer)
    }

    /// Sample a random perturbation direction on a specific device
    fn sample_perturbation(
        &mut self,
        shape: &candle::Shape,
        device: &candle::Device,
    ) -> Result<Tensor> {
        let data: Vec<f32> = (0..shape.elem_count())
            .map(|_| self.rng.sample::<f32, _>(rand_distr::StandardNormal))
            .collect();
        Tensor::from_vec(data, shape.clone(), device)
    }

    /// Perform optimization step with MoE auxiliary losses
    pub fn step_with_aux<F>(&mut self, mut loss_fn: F) -> Result<f32>
    where
        F: FnMut() -> Result<LossOutput>,
    {
        let mut z_directions = Vec::new();
        for var in &self.scale_vars {
            let shape = var.shape().clone();
            let device = var.device().clone();
            z_directions.push((shape, device));
        }

        let z_directions: Result<Vec<_>> = z_directions
            .into_iter()
            .map(|(shape, device)| self.sample_perturbation(&shape, &device))
            .collect();
        let z_directions = z_directions?;

        let mut total_loss_plus = 0.0f32;
        let mut total_loss_minus = 0.0f32;

        for _ in 0..self.params.num_samples {
            for (var, z) in self.scale_vars.iter().zip(&z_directions) {
                let perturbed = (var.as_tensor() + &(z * self.params.epsilon)?)?;
                var.set(&perturbed)?;
            }

            let mut loss_output_plus = loss_fn()?;
            let loss_plus = loss_output_plus
                .total_loss(
                    self.params.load_balance_alpha,
                    self.params.z_loss_alpha,
                    self.num_experts.unwrap_or(1),
                )?
                .to_vec0::<f32>()?;
            total_loss_plus += loss_plus;

            for (var, z) in self.scale_vars.iter().zip(&z_directions) {
                let perturbed = (var.as_tensor() - &(z * (2.0 * self.params.epsilon))?)?;
                var.set(&perturbed)?;
            }

            let mut loss_output_minus = loss_fn()?;
            let loss_minus = loss_output_minus
                .total_loss(
                    self.params.load_balance_alpha,
                    self.params.z_loss_alpha,
                    self.num_experts.unwrap_or(1),
                )?
                .to_vec0::<f32>()?;
            total_loss_minus += loss_minus;

            let directional_deriv = (loss_plus - loss_minus) / (2.0 * self.params.epsilon as f32);
            let clip = self.params.clip_threshold as f32;
            let clipped_deriv = directional_deriv.clamp(-clip, clip);
            let scale = (directional_deriv.abs() / clip).clamp(1.0, 50.0);
            let effective_lr = self.params.lr * scale as f64;

            for (var, z) in self.scale_vars.iter().zip(&z_directions) {
                let restored = (var.as_tensor() + &(z * self.params.epsilon)?)?;
                let update_term = (z * (effective_lr * clipped_deriv as f64))?;
                let updated = (restored - update_term)?;
                var.set(&updated)?;
            }
        }

        let avg_loss_plus = total_loss_plus / self.params.num_samples as f32;
        let avg_loss_minus = total_loss_minus / self.params.num_samples as f32;
        Ok((avg_loss_plus + avg_loss_minus) / 2.0)
    }

    /// Perform a single optimization step using zero-order gradient estimation
    pub fn step<F>(&mut self, mut loss_fn: F) -> Result<f32>
    where
        F: FnMut() -> Result<Tensor>,
    {
        self.step_with_aux(|| Ok(LossOutput::from_tensor(loss_fn()?)))
    }

    /// Perform optimization step (legacy implementation)
    #[allow(dead_code)]
    fn step_legacy<F>(&mut self, mut loss_fn: F) -> Result<f32>
    where
        F: FnMut() -> Result<Tensor>,
    {
        let loss_base = loss_fn()?.to_vec0::<f32>()?;
        let mut total_loss = loss_base;

        let num_vars = self.scale_vars.len();
        for var_idx in 0..num_vars {
            let (shape, device) = {
                let var = &self.scale_vars[var_idx];
                (var.shape().clone(), var.device().clone())
            };

            let mut grad_sum = Tensor::zeros(&shape, candle::DType::F32, &device)?;

            for _ in 0..self.params.num_samples {
                let z = self.sample_perturbation(&shape, &device)?;
                let var = &self.scale_vars[var_idx];

                let perturbed = (var.as_tensor() + &(&z * self.params.epsilon)?)?;
                var.set(&perturbed)?;

                let loss_perturbed = loss_fn()?.to_vec0::<f32>()?;
                total_loss += loss_perturbed;

                let restored = (var.as_tensor() - &(&z * self.params.epsilon)?)?;
                var.set(&restored)?;

                let loss_delta = loss_perturbed - loss_base;
                let grad_coef = (loss_delta as f64) / self.params.epsilon;
                let grad_estimate = (&z * grad_coef)?;
                grad_sum = (grad_sum + grad_estimate)?;
            }

            let grad_avg = (grad_sum / self.params.num_samples as f64)?;
            let var = &self.scale_vars[var_idx];
            let update = (var.as_tensor() - &(&grad_avg * self.params.lr)?)?;
            var.set(&update)?;
        }

        Ok(total_loss / (1 + self.params.num_samples) as f32)
    }

    pub fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    pub fn epsilon(&self) -> f64 {
        self.params.epsilon
    }

    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.params.epsilon = epsilon;
    }

    pub fn num_samples(&self) -> usize {
        self.params.num_samples
    }

    pub fn set_num_samples(&mut self, num_samples: usize) {
        self.params.num_samples = num_samples;
    }

    pub fn load_balance_alpha(&self) -> Option<f64> {
        self.params.load_balance_alpha
    }

    pub fn set_load_balance_alpha(&mut self, alpha: Option<f64>) {
        self.params.load_balance_alpha = alpha;
    }

    pub fn z_loss_alpha(&self) -> Option<f64> {
        self.params.z_loss_alpha
    }

    pub fn set_z_loss_alpha(&mut self, alpha: Option<f64>) {
        self.params.z_loss_alpha = alpha;
    }

    pub fn num_experts(&self) -> Option<usize> {
        self.num_experts
    }

    pub fn scale_vars(&self) -> &[Var] {
        &self.scale_vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_qzo_basic() -> Result<()> {
        let device = Device::Cpu;
        let var = Var::from_tensor(&Tensor::new(&[2.0f32], &device)?)?;

        let params = ParamsQZO {
            lr: 0.1,
            epsilon: 0.01,
            num_samples: 2,
            clip_threshold: 0.1,
            load_balance_alpha: None,
            z_loss_alpha: None,
        };

        let mut optimizer = QZO::new_with_seed(vec![var.clone()], params, 42)?;

        for _ in 0..50 {
            optimizer.step(|| {
                let x = var.as_tensor();
                let target = Tensor::new(&[5.0f32], &device)?;
                let diff = (x - &target)?;
                let loss = (&diff * &diff)?.sum_all()?;
                Ok(loss)
            })?;
        }

        let final_val = var.to_vec1::<f32>()?;
        let first = final_val.first().copied().unwrap_or(0.0);
        assert!(
            (first - 5.0).abs() < 1.0,
            "Expected value near 5.0, got {}",
            first
        );

        Ok(())
    }

    #[test]
    fn test_qzo_moe_with_aux() -> Result<()> {
        let device = Device::Cpu;
        let var = Var::from_tensor(&Tensor::new(2.0f32, &device)?)?;

        let params = ParamsQZO {
            lr: 0.1,
            epsilon: 0.01,
            num_samples: 1,
            clip_threshold: 0.1,
            load_balance_alpha: None,
            z_loss_alpha: None,
        };

        let mut optimizer = QZO::new_moe_with_seed(vec![var.clone()], params, 42, 4)?;

        for _ in 0..5 {
            optimizer.step_with_aux(|| {
                let x = var.as_tensor();
                let target = Tensor::new(5.0f32, &device)?;
                let diff = (x - &target)?;
                let task_loss = (&diff * &diff)?;
                Ok(LossOutput::from_tensor(task_loss))
            })?;
        }

        let final_val = var.to_vec0::<f32>()?;
        assert!(
            (final_val - 5.0).abs() < 3.0,
            "Expected value near 5.0, got {}",
            final_val
        );

        Ok(())
    }
}
