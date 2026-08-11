//! QZO: Quantized Zeroth-Order Optimization
//!
//! This crate provides memory-efficient fine-tuning for quantized models using
//! zeroth-order optimization. Instead of storing gradients for all parameters,
//! QZO:
//!
//! 1. Keeps the model quantized (memory efficient)
//! 2. Only optimizes the scaling factors in quantization blocks
//! 3. Uses finite differences instead of backpropagation
//!
//! # Basic Example
//!
//! ```no_run
//! use paramecia_qzo::{QZO, ParamsQZO};
//! use paramecia_core::quantized::QTensor;
//! use paramecia_core::{Result, Tensor};
//!
//! fn example() -> Result<()> {
//!     // Load a quantized model
//!     let qtensor = QTensor::quantize(&tensor, paramecia_core::quantized::GgmlDType::Q4_0)?;
//!     
//!     // Extract trainable scales
//!     let scales = qtensor.extract_scales()?;
//!     
//!     // Create QZO optimizer
//!     let params = ParamsQZO::default();
//!     let mut optimizer = QZO::new(vec![scales.into()], params)?;
//!     
//!     // Training loop
//!     for batch in dataloader {
//!         optimizer.step(|| {
//!             // Compute loss with current scales
//!             let output = model_forward(&qtensor, &scales, &batch)?;
//!             compute_loss(&output, &batch.labels)
//!         })?;
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! # MoE Support
//!
//! For Mixture-of-Experts models, use `step_with_aux` to include load balancing:
//!
//! ```no_run
//! use paramecia_qzo::{QZO, ParamsQZO, LossOutput};
//! use paramecia_qzo::moe::RouterStats;
//!
//! fn moe_example() -> paramecia_core::Result<()> {
//!     let params = ParamsQZO {
//!         lr: 1e-4,
//!         epsilon: 1e-3,
//!         num_samples: 1,
//!         clip_threshold: 0.01,
//!         load_balance_alpha: Some(0.01),
//!         z_loss_alpha: Some(0.001),
//!     };
//!     
//!     let mut optimizer = QZO::new_moe(scale_vars, params, 8)?;
//!     
//!     optimizer.step_with_aux(|| {
//!         let (logits, router_stats) = model_forward_with_stats(&input)?;
//!         let task_loss = cross_entropy(&logits, &labels)?;
//!         Ok(LossOutput::with_router_stats(task_loss, router_stats))
//!     })?;
//!     
//!     Ok(())
//! }
//! ```

pub mod moe;

use paramecia_core::{Result, Tensor, Var};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use tracing::{debug, warn};
// but NOT for GPU operations (forward passes) because CUDA contexts are thread-local.

use moe::{LoadBalanceLoss, RouterStats, ZLoss};

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
    /// **IMPORTANT**: Should be scaled with your loss magnitude!
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
            clip_threshold: 0.01, // Default clipping threshold
            load_balance_alpha: None,
            z_loss_alpha: None,
        }
    }
}

/// Loss function output that can include auxiliary losses for MoE
///
/// This struct encapsulates both the main task loss and optional
/// router statistics for computing load balancing auxiliary losses.
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
    ///
    /// # Arguments
    ///
    /// * `task_loss` - Main task loss
    /// * `router_stats` - Router statistics from MoE layers
    pub fn with_router_stats(task_loss: Tensor, router_stats: Vec<RouterStats>) -> Self {
        Self {
            task_loss,
            router_stats: Some(router_stats),
        }
    }

    /// Compute total loss including auxiliary terms
    ///
    /// # Arguments
    ///
    /// * `load_balance_alpha` - Weight for load balancing loss (if Some)
    /// * `z_loss_alpha` - Weight for Z-loss (if Some)
    /// * `num_experts` - Number of experts per MoE layer
    ///
    /// # Returns
    ///
    /// Total loss = task_loss + load_balance_loss + z_loss
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
                    warn!("LB loss ({:.2}) >> task loss ({:.2})", lb_val, task_val);
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
///
/// Optimizes only the scaling factors of quantized tensors using
/// simultaneous perturbation stochastic approximation (SPSA).
///
/// Supports both standard models and Mixture-of-Experts (MoE) models
/// with auxiliary load balancing losses.
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
    ///
    /// # Arguments
    ///
    /// * `scale_vars` - Variables containing the scaling factors to optimize
    /// * `params` - Optimization parameters
    pub fn new(scale_vars: Vec<Var>, params: ParamsQZO) -> Result<Self> {
        // Use a random seed based on system time
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

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
    ///
    /// # Arguments
    ///
    /// * `scale_vars` - Variables containing the scaling factors to optimize
    /// * `params` - Optimization parameters (should include load_balance_alpha)
    /// * `num_experts` - Number of experts per MoE layer
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
        shape: &paramecia_core::Shape,
        device: &paramecia_core::Device,
    ) -> Result<Tensor> {
        // Sample from standard normal distribution
        let data: Vec<f32> = (0..shape.elem_count())
            .map(|_| self.rng.sample::<f32, _>(rand_distr::StandardNormal))
            .collect();
        Tensor::from_vec(data, shape.clone(), device)
    }

    /// Perform optimization step with MoE auxiliary losses
    ///
    /// This method supports both regular models and MoE models with load balancing.
    /// For MoE models, the loss function should return a `LossOutput` with router statistics.
    ///
    /// # Arguments
    ///
    /// * `loss_fn` - A closure that computes the loss and returns `LossOutput`
    ///
    /// # Returns
    ///
    /// The averaged total loss (task + auxiliary) from all forward passes
    ///
    /// # Example
    ///
    /// ```no_run
    /// use paramecia_qzo::{QZO, LossOutput};
    ///
    /// let loss = optimizer.step_with_aux(|| {
    ///     let (logits, router_stats) = model.forward_with_stats(&input)?;
    ///     let task_loss = cross_entropy(&logits, &labels)?;
    ///     Ok(LossOutput::with_router_stats(task_loss, router_stats))
    /// })?;
    /// # Ok::<(), paramecia_core::Error>(())
    /// ```
    /// Perform a single optimization step with auxiliary losses (MoE support)
    ///
    /// Implements Algorithm 1 from "Fine-tuning Quantized Neural Networks with
    /// Zeroth-order Optimization" using two-sided finite differences.
    ///
    /// # Algorithm
    /// 1. Sample random direction z (shared across all variables)
    /// 2. Perturb forward: Δ ← Δ + ε·z
    /// 3. Compute ℓ₊ = L(Δ ⊙ θ̄; B)
    /// 4. Perturb backward: Δ ← Δ - 2ε·z  
    /// 5. Compute ℓ₋ = L(Δ ⊙ θ̄; B)
    /// 6. Restore: Δ ← Δ + ε·z
    /// 7. Compute directional derivative: d = (ℓ₊ - ℓ₋)/(2ε)
    /// 8. Clip: d' = CLIP(d, -C, C)
    /// 9. Update: Δ ← Δ - η·d'·z
    ///
    /// # Arguments
    /// * `loss_fn` - Closure that computes loss (called 2 times for two-sided estimate)
    ///
    /// # Returns
    /// Average of forward and backward pass losses
    pub fn step_with_aux<F>(&mut self, mut loss_fn: F) -> Result<f32>
    where
        F: FnMut() -> Result<LossOutput>,
    {
        // Sample ONE random direction for ALL variables (shared seed approach from paper)
        let mut z_directions = Vec::new();
        for var in &self.scale_vars {
            let shape = var.shape().clone();
            let device = var.device().clone();
            z_directions.push((shape, device));
        }

        // Now sample perturbations without borrowing self
        let z_directions: Result<Vec<_>> = z_directions
            .into_iter()
            .map(|(shape, device)| self.sample_perturbation(&shape, &device))
            .collect();
        let z_directions = z_directions?;

        let _num_vars = self.scale_vars.len();

        // Accumulate losses over num_samples (typically 1)
        let mut total_loss_plus = 0.0f32;
        let mut total_loss_minus = 0.0f32;

        for _ in 0..self.params.num_samples {
            // Step 1: Perturb forward: Δ ← Δ + ε·z
            for (var, z) in self.scale_vars.iter().zip(&z_directions) {
                let perturbed = (var.as_tensor() + &(z * self.params.epsilon)?)?;
                var.set(&perturbed)?;
            }

            // Step 2: Forward pass: ℓ₊ = L(Δ ⊙ θ̄; B)
            let mut loss_output_plus = loss_fn()?;
            let loss_plus = loss_output_plus
                .total_loss(
                    self.params.load_balance_alpha,
                    self.params.z_loss_alpha,
                    self.num_experts.unwrap_or(1),
                )?
                .to_vec0::<f32>()?;
            total_loss_plus += loss_plus;

            // Step 3: Perturb backward: Δ ← Δ - 2ε·z (goes from +ε to -ε)
            for (var, z) in self.scale_vars.iter().zip(&z_directions) {
                let perturbed = (var.as_tensor() - &(z * (2.0 * self.params.epsilon))?)?;
                var.set(&perturbed)?;
            }

            // Step 4: Backward pass: ℓ₋ = L(Δ ⊙ θ̄; B)
            let mut loss_output_minus = loss_fn()?;
            let loss_minus = loss_output_minus
                .total_loss(
                    self.params.load_balance_alpha,
                    self.params.z_loss_alpha,
                    self.num_experts.unwrap_or(1),
                )?
                .to_vec0::<f32>()?;
            total_loss_minus += loss_minus;

            // Step 5: Compute directional derivative: d = (ℓ₊ - ℓ₋)/(2ε)
            let directional_deriv = (loss_plus - loss_minus) / (2.0 * self.params.epsilon as f32);

            // Step 6: DDC - Directional Derivative Clipping
            let clipped_deriv = directional_deriv.clamp(
                -self.params.clip_threshold as f32,
                self.params.clip_threshold as f32,
            );

            // Step 7: Restore to original: Δ ← Δ + ε·z (from -ε back to 0)
            // Then update: Δ ← Δ - η·d'·z
            for (var, z) in self.scale_vars.iter().zip(&z_directions) {
                // Restore from -ε to 0
                let restored = (var.as_tensor() + &(z * self.params.epsilon)?)?;

                // Apply update: Δ ← Δ - η·d'·z
                let update_term = (z * (self.params.lr * clipped_deriv as f64))?;
                let updated = (restored - update_term)?;
                var.set(&updated)?;
            }
        }

        // Return average loss from forward and backward passes
        let avg_loss_plus = total_loss_plus / self.params.num_samples as f32;
        let avg_loss_minus = total_loss_minus / self.params.num_samples as f32;
        Ok((avg_loss_plus + avg_loss_minus) / 2.0)
    }

    /// Perform a single optimization step using zero-order gradient estimation
    ///
    /// This is the standard step method for non-MoE models. For MoE models,
    /// use `step_with_aux` instead to include load balancing losses.
    ///
    /// # Arguments
    ///
    /// * `loss_fn` - A closure that computes the loss. This will be called multiple
    ///   times to estimate gradients via finite differences.
    ///
    /// # Returns
    ///
    /// The averaged loss value from all forward passes
    pub fn step<F>(&mut self, mut loss_fn: F) -> Result<f32>
    where
        F: FnMut() -> Result<Tensor>,
    {
        // Wrap the loss function to return LossOutput
        self.step_with_aux(|| Ok(LossOutput::from_tensor(loss_fn()?)))
    }

    /// Perform optimization step (old implementation, kept for reference)
    #[allow(dead_code)]
    fn step_legacy<F>(&mut self, mut loss_fn: F) -> Result<f32>
    where
        F: FnMut() -> Result<Tensor>,
    {
        // Evaluate baseline loss
        let loss_base = loss_fn()?.to_vec0::<f32>()?;

        // Accumulate gradient estimates
        let mut total_loss = loss_base;

        let num_vars = self.scale_vars.len();
        for var_idx in 0..num_vars {
            // Extract shape and device first (clone device to avoid borrow issues)
            let (shape, device) = {
                let var = &self.scale_vars[var_idx];
                (var.shape().clone(), var.device().clone())
            };

            let mut grad_sum = Tensor::zeros(&shape, paramecia_core::DType::F32, &device)?;

            // Sample multiple random directions
            for _ in 0..self.params.num_samples {
                // Sample random perturbation (on the same device as this variable)
                let z = self.sample_perturbation(&shape, &device)?;

                // Get var reference
                let var = &self.scale_vars[var_idx];

                // Perturb: var += epsilon * z
                let perturbed = (var.as_tensor() + &(&z * self.params.epsilon)?)?;
                var.set(&perturbed)?;

                // Evaluate loss at perturbed point
                let loss_perturbed = loss_fn()?.to_vec0::<f32>()?;
                total_loss += loss_perturbed;

                // Restore original value
                let restored = (var.as_tensor() - &(&z * self.params.epsilon)?)?;
                var.set(&restored)?;

                // Estimate gradient: (loss_delta / epsilon) * z
                let loss_delta = loss_perturbed - loss_base;
                let grad_coef = (loss_delta as f64) / self.params.epsilon;
                let grad_estimate = (&z * grad_coef)?;
                grad_sum = (grad_sum + grad_estimate)?;
            }

            // Average the gradient estimates
            let grad_avg = (grad_sum / self.params.num_samples as f64)?;

            // Update: var -= learning_rate * gradient
            let var = &self.scale_vars[var_idx];
            let update = (var.as_tensor() - &(&grad_avg * self.params.lr)?)?;
            var.set(&update)?;
        }

        // Return average loss
        Ok(total_loss / (1 + self.params.num_samples) as f32)
    }

    /// Get the current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    /// Set a new learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    /// Get the current epsilon (perturbation magnitude)
    pub fn epsilon(&self) -> f64 {
        self.params.epsilon
    }

    /// Set a new epsilon value
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.params.epsilon = epsilon;
    }

    /// Get the current number of samples per update
    pub fn num_samples(&self) -> usize {
        self.params.num_samples
    }

    /// Set a new number of samples per update
    pub fn set_num_samples(&mut self, num_samples: usize) {
        self.params.num_samples = num_samples;
    }

    /// Get the load balance alpha (MoE)
    pub fn load_balance_alpha(&self) -> Option<f64> {
        self.params.load_balance_alpha
    }

    /// Set the load balance alpha (MoE)
    pub fn set_load_balance_alpha(&mut self, alpha: Option<f64>) {
        self.params.load_balance_alpha = alpha;
    }

    /// Get the Z-loss alpha (MoE)
    pub fn z_loss_alpha(&self) -> Option<f64> {
        self.params.z_loss_alpha
    }

    /// Set the Z-loss alpha (MoE)
    pub fn set_z_loss_alpha(&mut self, alpha: Option<f64>) {
        self.params.z_loss_alpha = alpha;
    }

    /// Get the number of experts (if MoE)
    pub fn num_experts(&self) -> Option<usize> {
        self.num_experts
    }

    /// Get references to the scale variables being optimized
    ///
    /// Returns a slice of Var references that can be used to extract
    /// current scale values during training
    pub fn scale_vars(&self) -> &[Var] {
        &self.scale_vars
    }
}

// ============================================================================
// QuZO: Quantized Zeroth-Order Optimization (Paper Implementation)
// ============================================================================
//
// This implements the QuZO algorithm from "QuZO: Quantized Zeroth-Order
// Fine-Tuning for Large Language Models" (arXiv:2502.12346).
//
// Key difference from QZO (above):
// - QZO optimizes continuous scale factors only
// - QuZO optimizes discrete quantized weights directly using stochastic rounding

use half::f16;
use paramecia_core::quantized::GgmlDType;
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
use paramecia_core::quantized::QTensor;
use paramecia_core::quantized::SharedQTensor;
use std::collections::VecDeque;

/// Returns true for dtypes that use stochastic quantization/residual feedback.
fn tracks_error_feedback_residuals(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4K
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8K
    )
}

/// Cached GPU disable flags for each dtype.
/// Initialized once on first access.
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
struct GpuDisableFlags {
    all: bool,
    q2k: bool,
    q3k: bool,
    q5k: bool,
    q6k: bool,
    bf16: bool,
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
impl GpuDisableFlags {
    fn load() -> Self {
        Self {
            all: std::env::var("QUZO_DISABLE_CUDA").is_ok()
                || std::env::var("QUZO_DISABLE_GPU").is_ok(),
            q2k: std::env::var("QUZO_DISABLE_CUDA_Q2K").is_ok(),
            q3k: std::env::var("QUZO_DISABLE_CUDA_Q3K").is_ok(),
            q5k: std::env::var("QUZO_DISABLE_CUDA_Q5K").is_ok(),
            q6k: std::env::var("QUZO_DISABLE_CUDA_Q6K").is_ok(),
            bf16: std::env::var("QUZO_DISABLE_CUDA_BF16").is_ok(),
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
static GPU_DISABLE_FLAGS: std::sync::OnceLock<GpuDisableFlags> = std::sync::OnceLock::new();

/// Check if GPU perturbation is enabled for a specific dtype.
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
fn gpu_enabled_for_dtype(dtype: GgmlDType) -> bool {
    let flags = GPU_DISABLE_FLAGS.get_or_init(GpuDisableFlags::load);
    if flags.all {
        return false;
    }
    match dtype {
        GgmlDType::Q2K => !flags.q2k,
        GgmlDType::Q3K => !flags.q3k,
        GgmlDType::Q5K => !flags.q5k,
        GgmlDType::Q6K => !flags.q6k,
        GgmlDType::BF16 => !flags.bf16,
        _ => true,
    }
}

/// Check if a QTensor supports GPU-accelerated perturbation (CUDA, Metal, or Vulkan).
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
fn supports_gpu_perturb(qt: &QTensor) -> bool {
    let dtype_ok = matches!(
        qt.dtype(),
        paramecia_core::quantized::GgmlDType::Q8_0
            | paramecia_core::quantized::GgmlDType::Q4K
            | paramecia_core::quantized::GgmlDType::Q2K
            | paramecia_core::quantized::GgmlDType::Q3K
            | paramecia_core::quantized::GgmlDType::Q5K
            | paramecia_core::quantized::GgmlDType::Q6K
            | paramecia_core::quantized::GgmlDType::BF16
    );
    if !dtype_ok {
        return false;
    }
    #[cfg(feature = "cuda")]
    if qt.device().is_cuda() {
        return gpu_enabled_for_dtype(qt.dtype());
    }
    #[cfg(feature = "metal")]
    if qt.device().is_metal() {
        return gpu_enabled_for_dtype(qt.dtype());
    }
    #[cfg(feature = "vulkan")]
    if qt.device().is_vulkan() {
        return gpu_enabled_for_dtype(qt.dtype());
    }
    false
}

/// Check if a QTensor supports GPU-accelerated restore_and_update (CUDA, Metal, or Vulkan).
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
fn supports_gpu_restore_update(qt: &QTensor) -> bool {
    let dtype_ok = matches!(
        qt.dtype(),
        paramecia_core::quantized::GgmlDType::Q8_0
            | paramecia_core::quantized::GgmlDType::Q4K
            | paramecia_core::quantized::GgmlDType::Q2K
            | paramecia_core::quantized::GgmlDType::Q3K
            | paramecia_core::quantized::GgmlDType::Q5K
            | paramecia_core::quantized::GgmlDType::Q6K
            | paramecia_core::quantized::GgmlDType::BF16
    );
    if !dtype_ok {
        return false;
    }
    #[cfg(feature = "cuda")]
    if qt.device().is_cuda() {
        return true;
    }
    #[cfg(feature = "metal")]
    if qt.device().is_metal() {
        return true;
    }
    #[cfg(feature = "vulkan")]
    if qt.device().is_vulkan() {
        return true;
    }
    false
}

/// Generate a perturbation tensor deterministically from a seed and shape.
///
/// Always generates on CPU. The same seed + shape always produces the same tensor.
fn generate_perturbation(seed: u64, shape: &paramecia_core::Shape) -> Result<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..shape.elem_count())
        .map(|_| rng.sample::<f32, _>(rand_distr::StandardNormal))
        .collect();
    Tensor::from_vec(data, shape.clone(), &paramecia_core::Device::Cpu)
}

/// Two-seed perturbation for unbiased gradient estimation.
///
/// QuZO uses two independent stochastic quantizations of the same
/// continuous perturbation:
/// - seed_forward: Used for Q_1(u) in loss estimation
/// - seed_update: Used for Q_2(u) in weight update
///
/// This maintains E[u_{i,1} · u_{i,2}^T] = E[u_i · u_i^T], preserving
/// the unbiased gradient property despite quantization.
pub struct QuZOPerturbation {
    /// Continuous perturbation u_i ~ N(0, I)
    pub continuous: Tensor,
    /// Seed for stochastic rounding in forward perturbation
    pub seed_forward: u64,
    /// Seed for stochastic rounding in update step (must differ!)
    pub seed_update: u64,
}

impl QuZOPerturbation {
    /// Sample a new perturbation with two independent seeds.
    ///
    /// Note: Perturbations are always created on CPU because perturb_weights
    /// operates on CPU anyway. This avoids GPU->CPU sync overhead.
    pub fn sample(
        shape: &paramecia_core::Shape,
        _device: &paramecia_core::Device, // Ignored - always use CPU
        rng: &mut StdRng,
    ) -> Result<Self> {
        // Sample continuous perturbation from N(0, I) on CPU
        // The perturb_weights function operates on CPU, so keeping
        // perturbations on CPU avoids expensive GPU->CPU transfers.
        let data: Vec<f32> = (0..shape.elem_count())
            .map(|_| rng.sample::<f32, _>(rand_distr::StandardNormal))
            .collect();
        let continuous = Tensor::from_vec(data, shape.clone(), &paramecia_core::Device::Cpu)?;

        Ok(Self {
            continuous,
            seed_forward: rng.random(),
            seed_update: rng.random(),
        })
    }
}

// ============================================================================
// Philox 4x32 RNG (matches CUDA kernel for fused perturbation)
// ============================================================================

/// Philox 4x32-10 constants
const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;
const PHILOX_W0: u32 = 0x9E3779B9;
const PHILOX_W1: u32 = 0xBB67AE85;

/// Multiply and get high/low parts
#[inline]
fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    ((product >> 32) as u32, product as u32)
}

/// Single Philox round
#[inline]
fn philox_round(c: &mut [u32; 4], k0: u32, k1: u32) {
    let (hi0, lo0) = mulhilo32(c[0], PHILOX_M0);
    let (hi1, lo1) = mulhilo32(c[2], PHILOX_M1);

    let new_c0 = hi1 ^ c[1] ^ k0;
    let new_c1 = lo1;
    let new_c2 = hi0 ^ c[3] ^ k1;
    let new_c3 = lo0;

    c[0] = new_c0;
    c[1] = new_c1;
    c[2] = new_c2;
    c[3] = new_c3;
}

/// Generate 4 uint32 random values from seed and index using Philox-4x32-10
fn philox4x32(seed: u64, index: u64) -> [u32; 4] {
    let mut c = [index as u32, (index >> 32) as u32, 0u32, 0u32];

    let mut k0 = seed as u32;
    let mut k1 = (seed >> 32) as u32;

    for _ in 0..10 {
        philox_round(&mut c, k0, k1);
        k0 = k0.wrapping_add(PHILOX_W0);
        k1 = k1.wrapping_add(PHILOX_W1);
    }

    c
}

/// Generate Gaussian-distributed float using Box-Muller (matches CUDA kernel)
pub fn philox_gaussian(seed: u64, index: u64) -> f32 {
    let r = philox4x32(seed, index);

    // Box-Muller transform (matches CUDA kernel)
    let u1 = ((r[0] | 1) as f64) * (1.0 / 4294967296.0); // (0, 1]
    let u2 = (r[1] as f64) * (1.0 / 4294967296.0); // [0, 1)

    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;

    (radius * theta.cos()) as f32
}

/// Error feedback mode for accumulated error in quantized updates (QES-style).
///
/// When update magnitudes are smaller than the quantization grid spacing,
/// stochastic rounding produces zero updates. Error feedback accumulates
/// the fractional residuals so small gradients eventually cross the rounding threshold.
#[derive(Clone, Debug)]
pub enum ErrorFeedbackMode {
    /// FP16 residuals stored per element for quantized tensors that use stochastic rounding.
    /// F32/BF16 tensors skip residual storage.
    Persistent,
    /// Reconstruct residuals by replaying history. Memory: O(K * n_tensors * 16 bytes).
    Replay { max_history: usize },
}

/// Parse an error feedback mode string.
pub fn parse_error_feedback_mode(s: &Option<String>) -> Option<ErrorFeedbackMode> {
    match s.as_deref() {
        None | Some("none") | Some("") => None,
        Some("persistent") => Some(ErrorFeedbackMode::Persistent),
        Some(s) if s.starts_with("replay:") => {
            let n: usize = s[7..].parse().unwrap_or(50);
            Some(ErrorFeedbackMode::Replay { max_history: n })
        }
        Some(s) => {
            warn!("[QuZO] Unknown error feedback mode '{}', using none", s);
            None
        }
    }
}

struct ReplayRecord {
    seeds: Vec<(u64, u64)>,
    update_scale: f64,
}

struct ReplayHistory {
    entries: VecDeque<ReplayRecord>,
    max_history: usize,
}

/// Parameters for the QuZO optimizer
#[derive(Clone, Debug)]
pub struct ParamsQuZO {
    /// Learning rate (η)
    pub lr: f64,
    /// Perturbation magnitude for finite differences (ε)
    pub epsilon: f64,
    /// Number of perturbation samples per step
    pub num_samples: usize,
    /// Clipping threshold for directional derivative
    /// Should be scaled based on expected loss magnitude (e.g., 10-100 for losses around 20)
    pub clip_threshold: f64,
    /// Use fused perturbation during forward passes (CUDA only).
    ///
    /// When enabled, perturbation is generated on-the-fly during matmul using
    /// Philox RNG, eliminating the need to modify weights before forward passes.
    /// This provides significant speedup for CPU-offloaded experts.
    ///
    /// Requirements:
    /// - CUDA feature must be enabled
    /// - Only Q8_0 and Q4K quantization formats are supported for fused mode (BF16 uses standard perturbation)
    /// - Weights must be on CUDA device
    ///
    /// For unsupported configurations, falls back to regular perturbation.
    pub use_fused: bool,
    /// Optional per-tensor epsilon multipliers.
    ///
    /// When set, tensor i uses epsilon * epsilon_multipliers[i] for perturbation.
    /// This allows different perturbation magnitudes for different model components:
    /// - Embedding layers: typically smaller (0.1×)
    /// - Attention Q/K/V: standard (1.0×)
    /// - MoE gating: smaller for stability (0.5×)
    /// - MoE experts: larger for impact (2.0×)
    /// - MTP heads: moderate (1.5×)
    ///
    /// If None, all tensors use the base epsilon.
    /// Must have the same length as the number of tensors if set.
    pub epsilon_multipliers: Option<Vec<f64>>,
    /// Generate perturbation tensors lazily (one at a time, regenerated from seed).
    ///
    /// When false (default): all perturbation tensors are pre-allocated in parallel
    /// using rayon, then reused across the 3 perturbation phases. Fast but uses
    /// `total_elem_count * 4` bytes of memory for the perturbations.
    ///
    /// When true: each perturbation is regenerated from its seed on-demand and freed
    /// immediately. Uses 3x the CPU random generation time but keeps peak perturbation
    /// memory to just one tensor's worth.
    ///
    /// **Recommended on Apple Silicon** where CPU and GPU share unified memory —
    /// a 2GB quantized model can require 7-14GB of perturbation storage otherwise.
    /// On CUDA systems with separate CPU RAM, the default (false) is usually fine.
    pub lazy_perturbations: bool,
    /// Error feedback mode for accumulated residuals.
    /// None = original QuZO (no error accumulation).
    pub error_feedback: Option<ErrorFeedbackMode>,
    /// Temporal decay applied to residuals each step (gamma, 0-1). Default 0.9.
    pub error_decay: f64,
    /// How much the residual influences rounding (0.0-1.0). Default 1.0.
    /// 0.0 = purely stochastic (residual tracked but doesn't affect rounding)
    /// 1.0 = full error feedback (QES behavior)
    pub error_gain: f64,
}

impl Default for ParamsQuZO {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            epsilon: 1e-3,
            num_samples: 1,
            clip_threshold: 1.0,
            use_fused: true,
            epsilon_multipliers: None,
            lazy_perturbations: false,
            error_feedback: None,
            error_decay: 0.9,
            error_gain: 1.0,
        }
    }
}

/// QuZO: Quantized Zeroth-Order Optimizer for Discrete Weights
///
/// Unlike QZO which only optimizes scale factors, QuZO directly modifies
/// the discrete quantized weights using stochastic rounding to maintain
/// unbiased gradient estimation.
///
/// # Algorithm
///
/// For each step:
/// 1. Sample continuous perturbation u_i ~ N(0, I)
/// 2. Generate two quantized versions with different seeds:
///    - u_{i,1} = Q_1(u_i) for loss estimation
///    - u_{i,2} = Q_2(u_i) for weight update
/// 3. Compute loss sensitivity: μ = (L(w̄ + ε·u_{i,1}) - L(w̄ - ε·u_{i,1})) / (2ε)
/// 4. Update weights: w̄ ← w̄ - Q(η·μ/n · u_{i,2})
///
/// The key insight is that E[u_{i,1}·u_{i,2}^T] = E[u_i·u_i^T] when using
/// stochastic rounding, preserving the unbiased gradient property.
pub struct QuZO {
    /// Mutable references to quantized tensors being optimized
    qtensors: Vec<SharedQTensor>,
    /// Optimizer parameters
    params: ParamsQuZO,
    /// Random number generator
    rng: StdRng,
    /// Current step count
    step_count: usize,
    /// Per-tensor epsilon multipliers (cached from params for fast access)
    epsilon_multipliers: Vec<f64>,
    /// Per-tensor FP16 residuals (Persistent mode only)
    residuals: Option<Vec<Vec<f16>>>,
    /// Step history for replay reconstruction
    replay_history: Option<ReplayHistory>,
}

/// State preserved between decomposed ZO phases (perturb_up → perturb_down → update).
pub struct DecomposedZOState {
    seeds: Vec<(u64, u64)>,
    prealloc: Option<Vec<Tensor>>,
}

impl QuZO {
    /// Create a new QuZO optimizer.
    ///
    /// # Arguments
    /// * `qtensors` - SharedQTensor references to optimize
    /// * `params` - Optimization parameters
    pub fn new(qtensors: Vec<SharedQTensor>, params: ParamsQuZO) -> Result<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self::new_with_seed(qtensors, params, seed)
    }

    /// Create QuZO with a specific seed for reproducibility.
    pub fn new_with_seed(
        qtensors: Vec<SharedQTensor>,
        params: ParamsQuZO,
        seed: u64,
    ) -> Result<Self> {
        let epsilon_multipliers = params
            .epsilon_multipliers
            .clone()
            .unwrap_or_else(|| vec![1.0; qtensors.len()]);

        // Initialize error feedback state
        let (residuals, replay_history) = match &params.error_feedback {
            Some(ErrorFeedbackMode::Persistent) => {
                let res: Vec<Vec<f16>> = qtensors
                    .iter()
                    .map(|qt| {
                        let qt_read = qt.read().unwrap();
                        if tracks_error_feedback_residuals(qt_read.dtype()) {
                            vec![f16::ZERO; qt_read.shape().elem_count()]
                        } else {
                            Vec::new()
                        }
                    })
                    .collect();
                (Some(res), None)
            }
            Some(ErrorFeedbackMode::Replay { max_history }) => (
                None,
                Some(ReplayHistory {
                    entries: VecDeque::new(),
                    max_history: *max_history,
                }),
            ),
            None => (None, None),
        };

        Ok(Self {
            qtensors,
            params,
            rng: StdRng::seed_from_u64(seed),
            step_count: 0,
            epsilon_multipliers,
            residuals,
            replay_history,
        })
    }

    /// Perform a single QuZO optimization step.
    ///
    /// # Arguments
    /// * `loss_fn` - Closure that computes the loss
    ///   - Called twice per perturbation for two-sided (central) differences (default)
    ///   - Called twice per perturbation for one-sided differences (baseline + perturbed)
    ///
    /// # Returns
    /// Average loss from forward passes
    pub fn step<F>(&mut self, mut loss_fn: F) -> Result<f32>
    where
        F: FnMut() -> Result<Tensor>,
    {
        // Increment step counter
        self.step_count += 1;

        // Use fused kernel path if enabled (CUDA, Metal, or Vulkan)
        // but only when error feedback is disabled (residuals live on CPU)
        #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
        if self.params.use_fused
            && self.params.error_feedback.is_none()
            && std::env::var("PARAMECIA_DISABLE_FUSED").is_err()
        {
            return self.step_two_sided_fused(&mut loss_fn);
        }

        self.step_two_sided(&mut loss_fn)
    }

    /// Get the current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Two-sided (central) finite differences: g ≈ (L+ - L-) / 2ε
    /// More accurate but requires 3 perturbation phases per sample.
    ///
    /// Perturbations are generated lazily from seeds (one tensor at a time) to
    /// avoid allocating all perturbation tensors simultaneously. On Apple Silicon
    /// where CPU and GPU share unified memory, pre-allocating all perturbations
    /// would consume elem_count*4 bytes (potentially exceeding the memory pool).
    fn step_two_sided<F>(&mut self, loss_fn: &mut F) -> Result<f32>
    where
        F: FnMut() -> Result<Tensor>,
    {
        use std::time::Instant;

        let epsilon = self.params.epsilon as f32;
        let lr = self.params.lr as f32;
        let clip = self.params.clip_threshold as f32;

        let mut total_loss = 0.0f32;
        let mut loss_count = 0;

        // Timing accumulators (only used for first few steps)
        let mut time_sample = 0.0f32;
        let mut time_perturb_fwd = 0.0f32;
        let mut time_loss_plus = 0.0f32;
        let mut time_perturb_bwd = 0.0f32;
        let mut time_loss_minus = 0.0f32;
        let mut time_update = 0.0f32;

        let lazy = self.params.lazy_perturbations;
        let gain = self.params.error_gain as f32;

        // Apply decay to persistent residuals at step start
        if let Some(ref mut residuals) = self.residuals {
            let decay = self.params.error_decay as f32;
            for tensor_residuals in residuals.iter_mut() {
                for v in tensor_residuals.iter_mut() {
                    *v = f16::from_f32(v.to_f32() * decay);
                }
            }
        }

        // For replay mode, reconstruct residuals from history
        let mut replay_residuals = if self.replay_history.is_some() {
            Some(self.reconstruct_residuals()?)
        } else {
            None
        };

        for _ in 0..self.params.num_samples {
            // Step 1: Pre-generate seeds. If not lazy, also generate perturbation
            // tensors in parallel (fast but uses total_elem_count*4 bytes).
            let t0 = Instant::now();
            let seeds: Vec<(u64, u64)> = (0..self.qtensors.len())
                .map(|_| {
                    let seed_forward: u64 = self.rng.random();
                    let seed_update: u64 = self.rng.random();
                    (seed_forward, seed_update)
                })
                .collect();

            // Pre-allocate perturbation tensors in parallel (only when not lazy)
            let prealloc: Option<Vec<Tensor>> = if !lazy {
                let specs: Vec<_> = self
                    .qtensors
                    .iter()
                    .zip(&seeds)
                    .map(|(qt, &(seed_forward, _))| {
                        let shape = qt.read().unwrap().shape().clone();
                        (seed_forward, shape)
                    })
                    .collect();
                Some(
                    specs
                        .into_par_iter()
                        .map(|(seed, shape)| generate_perturbation(seed, &shape))
                        .collect::<Result<Vec<_>>>()?,
                )
            } else {
                None
            };
            time_sample += t0.elapsed().as_secs_f32();

            // Helper: get perturbation for tensor i (lazy or pre-allocated)
            macro_rules! get_perturb {
                ($i:expr, $shape:expr) => {
                    if let Some(ref tensors) = prealloc {
                        tensors[$i].clone()
                    } else {
                        generate_perturbation(seeds[$i].0, $shape)?
                    }
                };
            }

            // Step 2: Forward perturbation: w̄ + ε_i·Q_1(u)
            let t1 = Instant::now();
            for (i, (qt, &(seed_forward, _))) in self.qtensors.iter().zip(&seeds).enumerate() {
                let eps_i = epsilon * self.epsilon_multipliers[i] as f32;
                let qt_read = qt.read().unwrap();
                let continuous = get_perturb!(i, qt_read.shape());
                #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                let perturbed = if supports_gpu_perturb(&qt_read) {
                    qt_read.perturb_weights_gpu(&continuous, eps_i, seed_forward, true)?
                } else {
                    qt_read.perturb_weights(&continuous, eps_i, seed_forward, true)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                let perturbed = qt_read.perturb_weights(&continuous, eps_i, seed_forward, true)?;
                drop(qt_read);
                qt.replace(perturbed);
            }
            time_perturb_fwd += t1.elapsed().as_secs_f32();

            // Compute loss_plus (GPU operation)
            let t2 = Instant::now();
            let loss_plus = loss_fn()?.to_vec0::<f32>()?;
            time_loss_plus += t2.elapsed().as_secs_f32();
            total_loss += loss_plus;
            loss_count += 1;

            // Step 3: Backward perturbation: go from +ε_i to -ε_i (subtract 2ε_i)
            let t3 = Instant::now();
            for (i, (qt, &(seed_forward, _))) in self.qtensors.iter().zip(&seeds).enumerate() {
                let eps_i = epsilon * self.epsilon_multipliers[i] as f32;
                let qt_read = qt.read().unwrap();
                let continuous = get_perturb!(i, qt_read.shape());
                #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                let perturbed = if supports_gpu_perturb(&qt_read) {
                    qt_read.perturb_weights_gpu(&continuous, 2.0 * eps_i, seed_forward, false)?
                } else {
                    qt_read.perturb_weights(&continuous, 2.0 * eps_i, seed_forward, false)?
                };
                #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                let perturbed =
                    qt_read.perturb_weights(&continuous, 2.0 * eps_i, seed_forward, false)?;
                drop(qt_read);
                qt.replace(perturbed);
            }
            time_perturb_bwd += t3.elapsed().as_secs_f32();

            // Compute loss_minus (GPU operation)
            let t4 = Instant::now();
            let loss_minus = loss_fn()?.to_vec0::<f32>()?;
            time_loss_minus += t4.elapsed().as_secs_f32();
            total_loss += loss_minus;
            loss_count += 1;

            // Step 4: Compute directional derivative
            let mu = (loss_plus - loss_minus) / (2.0 * epsilon);

            // Step 5: Clip the directional derivative
            let mu_clipped = mu.clamp(-clip, clip);

            // Step 6: Combined restore-and-update
            let t5 = Instant::now();
            let update_scale = lr * mu_clipped / (self.params.num_samples as f32);

            // Get the active residuals (persistent or replay-reconstructed)
            let mut active_residuals: Option<&mut Vec<Vec<f16>>> = if self.residuals.is_some() {
                self.residuals.as_mut()
            } else {
                replay_residuals.as_mut()
            };

            for (i, (qt, &(seed_forward, seed_update))) in
                self.qtensors.iter().zip(&seeds).enumerate()
            {
                let eps_i = epsilon * self.epsilon_multipliers[i] as f32;
                let update_scale_i = update_scale * self.epsilon_multipliers[i] as f32;
                let qt_read = qt.read().unwrap();
                let continuous = get_perturb!(i, qt_read.shape());

                let apply_without_residual = || -> Result<_> {
                    // Original path (try GPU first, fall back to CPU)
                    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                    {
                        if supports_gpu_restore_update(&qt_read) {
                            qt_read.restore_and_update_gpu(
                                &continuous,
                                eps_i,
                                update_scale_i,
                                seed_forward,
                                seed_update,
                            )
                        } else {
                            qt_read.restore_and_update(
                                &continuous,
                                eps_i,
                                update_scale_i,
                                seed_forward,
                                seed_update,
                            )
                        }
                    }
                    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                    {
                        qt_read.restore_and_update(
                            &continuous,
                            eps_i,
                            update_scale_i,
                            seed_forward,
                            seed_update,
                        )
                    }
                };

                let updated = if tracks_error_feedback_residuals(qt_read.dtype()) {
                    if let Some(ref mut residuals) = active_residuals.as_mut() {
                        // Error feedback path: always CPU for quantized tensors.
                        qt_read.restore_and_update_with_residual(
                            &continuous,
                            eps_i,
                            update_scale_i,
                            seed_forward,
                            seed_update,
                            &mut residuals[i],
                            gain,
                        )?
                    } else {
                        apply_without_residual()?
                    }
                } else {
                    // F32/BF16 tensors are updated directly without residual tracking.
                    apply_without_residual()?
                };
                drop(qt_read);
                qt.replace(updated);
            }
            time_update += t5.elapsed().as_secs_f32();

            // Record step for replay reconstruction
            if let Some(ref mut history) = self.replay_history {
                history.entries.push_back(ReplayRecord {
                    seeds: seeds.clone(),
                    update_scale: update_scale as f64,
                });
                if history.max_history > 0 && history.entries.len() > history.max_history {
                    history.entries.pop_front();
                }
            }
        }

        debug!(
            "[QuZO timing] sample={:.3}s perturb_fwd={:.3}s loss+={:.3}s perturb_bwd={:.3}s loss-={:.3}s update={:.3}s",
            time_sample, time_perturb_fwd, time_loss_plus, time_perturb_bwd, time_loss_minus, time_update
        );

        Ok(total_loss / loss_count as f32)
    }

    // ========================================================================
    // Decomposed two-sided ZO: perturb_up → perturb_down → update
    // ========================================================================

    /// Phase 1: Apply +ε perturbation. The caller should then compute
    /// `loss_up` using the perturbed model externally.
    ///
    /// Returns a `DecomposedZOState` that must be passed to `perturb_down`.
    pub fn perturb_up(&mut self, seed: Option<u64>) -> Result<DecomposedZOState> {
        // Re-seed RNG when an external seed is provided
        if let Some(seed) = seed {
            self.rng = StdRng::seed_from_u64(seed);
        }

        let epsilon = self.params.epsilon as f32;
        let lazy = self.params.lazy_perturbations;

        // Generate seeds
        let seeds: Vec<(u64, u64)> = (0..self.qtensors.len())
            .map(|_| {
                let seed_forward: u64 = self.rng.random();
                let seed_update: u64 = self.rng.random();
                (seed_forward, seed_update)
            })
            .collect();

        // Pre-allocate perturbation tensors if not lazy
        let prealloc: Option<Vec<Tensor>> = if !lazy {
            let specs: Vec<_> = self
                .qtensors
                .iter()
                .zip(&seeds)
                .map(|(qt, &(seed_forward, _))| {
                    let shape = qt.read().unwrap().shape().clone();
                    (seed_forward, shape)
                })
                .collect();
            Some(
                specs
                    .into_par_iter()
                    .map(|(seed, shape)| generate_perturbation(seed, &shape))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };

        // Apply +ε perturbation
        for (i, (qt, &(seed_forward, _))) in self.qtensors.iter().zip(&seeds).enumerate() {
            let eps_i = epsilon * self.epsilon_multipliers[i] as f32;
            let qt_read = qt.read().unwrap();
            let continuous = if let Some(ref tensors) = prealloc {
                tensors[i].clone()
            } else {
                generate_perturbation(seeds[i].0, qt_read.shape())?
            };
            #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
            let perturbed = if supports_gpu_perturb(&qt_read) {
                qt_read.perturb_weights_gpu(&continuous, eps_i, seed_forward, true)?
            } else {
                qt_read.perturb_weights(&continuous, eps_i, seed_forward, true)?
            };
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
            let perturbed = qt_read.perturb_weights(&continuous, eps_i, seed_forward, true)?;
            drop(qt_read);
            qt.replace(perturbed);
        }

        Ok(DecomposedZOState { seeds, prealloc })
    }

    /// Phase 2: Go from +ε to -ε perturbation. The caller should then
    /// compute `loss_down` using the negatively-perturbed model.
    ///
    /// Must be called after `perturb_up` with the returned state.
    pub fn perturb_down(&mut self, state: &DecomposedZOState) -> Result<()> {
        let epsilon = self.params.epsilon as f32;

        for (i, (qt, &(seed_forward, _))) in self.qtensors.iter().zip(&state.seeds).enumerate() {
            let eps_i = epsilon * self.epsilon_multipliers[i] as f32;
            let qt_read = qt.read().unwrap();
            let continuous = if let Some(ref tensors) = state.prealloc {
                tensors[i].clone()
            } else {
                generate_perturbation(state.seeds[i].0, qt_read.shape())?
            };
            #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
            let perturbed = if supports_gpu_perturb(&qt_read) {
                qt_read.perturb_weights_gpu(&continuous, 2.0 * eps_i, seed_forward, false)?
            } else {
                qt_read.perturb_weights(&continuous, 2.0 * eps_i, seed_forward, false)?
            };
            #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
            let perturbed =
                qt_read.perturb_weights(&continuous, 2.0 * eps_i, seed_forward, false)?;
            drop(qt_read);
            qt.replace(perturbed);
        }

        Ok(())
    }

    /// Phase 3: Compute gradient estimate and update weights.
    ///
    /// Returns the average of `loss_up` and `loss_down`.
    pub fn update(
        &mut self,
        state: DecomposedZOState,
        loss_up: f32,
        loss_down: f32,
    ) -> Result<f32> {
        self.step_count += 1;

        let epsilon = self.params.epsilon as f32;
        let lr = self.params.lr as f32;
        let clip = self.params.clip_threshold as f32;
        let gain = self.params.error_gain as f32;

        let mu = (loss_up - loss_down) / (2.0 * epsilon);
        let mu_clipped = mu.clamp(-clip, clip);

        // num_samples=1 for decomposed API
        let update_scale = lr * mu_clipped;

        // Apply decay to persistent residuals
        if let Some(ref mut residuals) = self.residuals {
            let decay = self.params.error_decay as f32;
            for tensor_residuals in residuals.iter_mut() {
                for v in tensor_residuals.iter_mut() {
                    *v = f16::from_f32(v.to_f32() * decay);
                }
            }
        }

        // For replay mode, reconstruct residuals
        let mut replay_residuals = if self.replay_history.is_some() {
            Some(self.reconstruct_residuals()?)
        } else {
            None
        };

        let mut active_residuals: Option<&mut Vec<Vec<f16>>> = if self.residuals.is_some() {
            self.residuals.as_mut()
        } else {
            replay_residuals.as_mut()
        };

        for (i, (qt, &(seed_forward, seed_update))) in
            self.qtensors.iter().zip(&state.seeds).enumerate()
        {
            let eps_i = epsilon * self.epsilon_multipliers[i] as f32;
            let update_scale_i = update_scale * self.epsilon_multipliers[i] as f32;
            let qt_read = qt.read().unwrap();
            let continuous = if let Some(ref tensors) = state.prealloc {
                tensors[i].clone()
            } else {
                generate_perturbation(state.seeds[i].0, qt_read.shape())?
            };

            let apply_without_residual = || -> Result<_> {
                #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                {
                    if supports_gpu_restore_update(&qt_read) {
                        qt_read.restore_and_update_gpu(
                            &continuous,
                            eps_i,
                            update_scale_i,
                            seed_forward,
                            seed_update,
                        )
                    } else {
                        qt_read.restore_and_update(
                            &continuous,
                            eps_i,
                            update_scale_i,
                            seed_forward,
                            seed_update,
                        )
                    }
                }
                #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                {
                    qt_read.restore_and_update(
                        &continuous,
                        eps_i,
                        update_scale_i,
                        seed_forward,
                        seed_update,
                    )
                }
            };

            let updated = if tracks_error_feedback_residuals(qt_read.dtype()) {
                if let Some(ref mut residuals) = active_residuals.as_mut() {
                    qt_read.restore_and_update_with_residual(
                        &continuous,
                        eps_i,
                        update_scale_i,
                        seed_forward,
                        seed_update,
                        &mut residuals[i],
                        gain,
                    )?
                } else {
                    apply_without_residual()?
                }
            } else {
                apply_without_residual()?
            };
            drop(qt_read);
            qt.replace(updated);
        }

        // Record step for replay
        if let Some(ref mut history) = self.replay_history {
            history.entries.push_back(ReplayRecord {
                seeds: state.seeds.to_vec(),
                update_scale: update_scale as f64,
            });
            if history.max_history > 0 && history.entries.len() > history.max_history {
                history.entries.pop_front();
            }
        }

        Ok((loss_up + loss_down) / 2.0)
    }

    /// Fused two-sided finite differences using on-the-fly perturbation.
    ///
    /// This implementation uses the fused kernel that generates perturbation
    /// during matmul, eliminating the need to modify weights before forward passes.
    ///
    /// **Important**: This only works when ALL optimized tensors are on GPU with
    /// supported dtypes (Q8_0, Q4K, etc.). If any tensor is on CPU or unsupported,
    /// falls back to regular step_two_sided.
    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    fn step_two_sided_fused<F>(&mut self, loss_fn: &mut F) -> Result<f32>
    where
        F: FnMut() -> Result<Tensor>,
    {
        use paramecia_core::quantized::{
            clear_perturbation_state, set_perturbation_state, GgmlDType,
        };
        use std::time::Instant;

        // Check if all tensors support fused mode (GPU + supported dtype)
        // Also collect device pointers/IDs for per-tensor seeding
        let mut tensor_info: Vec<(paramecia_core::Shape, u64)> =
            Vec::with_capacity(self.qtensors.len());
        for qt in &self.qtensors {
            let qt_read = qt.read().unwrap();
            let dtype_ok = matches!(
                qt_read.dtype(),
                GgmlDType::Q8_0
                    | GgmlDType::Q4K
                    | GgmlDType::Q2K
                    | GgmlDType::Q3K
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
                    | GgmlDType::BF16
            );
            let gpu_ok = gpu_enabled_for_dtype(qt_read.dtype());
            let is_gpu = {
                let mut _gpu = false;
                #[cfg(feature = "cuda")]
                {
                    _gpu = _gpu || qt_read.device().is_cuda();
                }
                #[cfg(feature = "metal")]
                {
                    _gpu = _gpu || qt_read.device().is_metal();
                }
                #[cfg(feature = "vulkan")]
                {
                    _gpu = _gpu || qt_read.device().is_vulkan();
                }
                _gpu
            };

            if !is_gpu || !dtype_ok || !gpu_ok {
                // Fall back to regular step if any tensor doesn't support fused
                drop(qt_read);
                return self.step_two_sided(loss_fn);
            }

            // Get per-tensor ID for Philox RNG seeding (must match the fused kernel)
            let tensor_id = qt_read.fused_tensor_id();
            tensor_info.push((qt_read.shape().clone(), tensor_id));
        }

        let epsilon = self.params.epsilon as f32;
        let lr = self.params.lr as f32;
        let clip = self.params.clip_threshold as f32;

        let mut total_loss = 0.0f32;
        let mut loss_count = 0;

        // Timing accumulators
        let mut time_sample = 0.0f32;
        let mut time_loss_plus = 0.0f32;
        let mut time_loss_minus = 0.0f32;
        let mut time_update = 0.0f32;

        for _ in 0..self.params.num_samples {
            // Step 1: Generate seeds for this sample
            let t0 = Instant::now();
            let seed_forward: u64 = self.rng.random();
            let seed_update: u64 = self.rng.random();
            time_sample += t0.elapsed().as_secs_f32();

            // Step 2: Forward pass with +ε perturbation (using fused kernel)
            let t1 = Instant::now();
            set_perturbation_state(seed_forward, epsilon);
            let loss_plus = loss_fn()?.to_vec0::<f32>()?;
            clear_perturbation_state();
            time_loss_plus += t1.elapsed().as_secs_f32();
            total_loss += loss_plus;
            loss_count += 1;

            // Step 3: Forward pass with -ε perturbation (using fused kernel)
            let t2 = Instant::now();
            set_perturbation_state(seed_forward, -epsilon);
            let loss_minus = loss_fn()?.to_vec0::<f32>()?;
            clear_perturbation_state();
            time_loss_minus += t2.elapsed().as_secs_f32();
            total_loss += loss_minus;
            loss_count += 1;

            // Step 4: Compute directional derivative
            let mu = (loss_plus - loss_minus) / (2.0 * epsilon);

            // Step 5: Clip the directional derivative
            let mu_clipped = mu.clamp(-clip, clip);

            // Step 6: Apply update using Philox-generated perturbation
            // Use effective_seed = seed ^ device_ptr to match the fused kernel
            let t3 = Instant::now();
            let update_scale = lr * mu_clipped / (self.params.num_samples as f32);

            for (qt, (shape, tensor_id)) in self.qtensors.iter().zip(&tensor_info) {
                let qt_read = qt.read().unwrap();

                // Compute effective seed matching the fused kernel
                let effective_seed = seed_forward ^ tensor_id;

                // Generate perturbation using Philox RNG (matches fused kernel)
                let elem_count = shape.elem_count();
                let perturb_data: Vec<f32> = (0..elem_count)
                    .map(|i| philox_gaussian(effective_seed, i as u64))
                    .collect();
                let perturb_tensor =
                    Tensor::from_vec(perturb_data, shape.clone(), &paramecia_core::Device::Cpu)?;

                // Apply update: w̄ ← w̄ - Q(update_scale · u)
                // (no restoration needed since weights were never modified)
                let updated =
                    qt_read.apply_quantized_update(&perturb_tensor, update_scale, seed_update)?;
                drop(qt_read);
                qt.replace(updated);
            }
            time_update += t3.elapsed().as_secs_f32();
        }

        debug!(
            "[QuZO fused timing] sample={:.3}s loss+={:.3}s loss-={:.3}s update={:.3}s",
            time_sample, time_loss_plus, time_loss_minus, time_update
        );

        Ok(total_loss / loss_count as f32)
    }

    /// Reconstruct residuals by replaying history (for Replay mode).
    fn reconstruct_residuals(&self) -> Result<Vec<Vec<f16>>> {
        let history = self.replay_history.as_ref().unwrap();
        let mut residuals: Vec<Vec<f16>> = self
            .qtensors
            .iter()
            .map(|qt| {
                let qt_read = qt.read().unwrap();
                if tracks_error_feedback_residuals(qt_read.dtype()) {
                    vec![f16::ZERO; qt_read.shape().elem_count()]
                } else {
                    Vec::new()
                }
            })
            .collect();

        let decay = self.params.error_decay as f32;
        let gain = self.params.error_gain as f32;

        for entry in &history.entries {
            // Apply decay
            for r in residuals.iter_mut() {
                for v in r.iter_mut() {
                    *v = f16::from_f32(v.to_f32() * decay);
                }
            }
            // Replay each tensor's update
            for (i, qt) in self.qtensors.iter().enumerate() {
                let qt_read = qt.read().unwrap();
                if !tracks_error_feedback_residuals(qt_read.dtype()) {
                    continue;
                }
                let (_, seed_update) = entry.seeds[i];
                let update_scale_i = (entry.update_scale * self.epsilon_multipliers[i]) as f32;
                let perturbation = generate_perturbation(entry.seeds[i].0, qt_read.shape())?;
                let perturb_data = perturbation.flatten_all()?.to_vec1::<f32>()?;
                qt_read.simulate_update_with_residual(
                    &perturb_data,
                    update_scale_i,
                    seed_update,
                    &mut residuals[i],
                    gain,
                )?;
            }
        }

        Ok(residuals)
    }

    /// Get the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    /// Set the learning rate.
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    /// Get epsilon (perturbation magnitude).
    pub fn epsilon(&self) -> f64 {
        self.params.epsilon
    }

    /// Set epsilon.
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.params.epsilon = epsilon;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paramecia_core::Device;

    #[test]
    fn test_qzo_basic() -> Result<()> {
        // Test basic QZO optimization
        let device = Device::Cpu;

        // Create a simple variable to optimize (target: 5.0)
        let var = Var::from_tensor(&Tensor::new(&[2.0f32], &device)?)?;

        let params = ParamsQZO {
            lr: 0.1,
            epsilon: 0.01,
            num_samples: 2,
            clip_threshold: 1.0,
            load_balance_alpha: None,
            z_loss_alpha: None,
        };

        let mut optimizer = QZO::new_with_seed(vec![var.clone()], params, 42)?;

        // Optimize towards x = 5.0
        for _ in 0..50 {
            optimizer.step(|| {
                let x = var.as_tensor();
                let target = Tensor::new(&[5.0f32], &device)?;
                let diff = (x - &target)?;
                let loss = (&diff * &diff)?.sum_all()?;
                Ok(loss)
            })?;
        }

        // Check that we moved towards the target
        let final_val = var.to_vec1::<f32>()?[0];
        assert!(
            (final_val - 5.0).abs() < 1.0,
            "Expected value near 5.0, got {}",
            final_val
        );

        Ok(())
    }

    #[test]
    fn test_qzo_moe_with_aux() -> Result<()> {
        // Test QZO with MoE support (basic functionality)
        let device = Device::Cpu;

        // Create a simple scalar variable to optimize
        let var = Var::from_tensor(&Tensor::new(2.0f32, &device)?)?;

        let params = ParamsQZO {
            lr: 0.1,
            epsilon: 0.01,
            num_samples: 1,
            clip_threshold: 1.0,
            load_balance_alpha: None,
            z_loss_alpha: None,
        };

        let mut optimizer = QZO::new_moe_with_seed(vec![var.clone()], params, 42, 4)?;

        // Training loop using step_with_aux (MoE-compatible interface)
        for _ in 0..5 {
            optimizer.step_with_aux(|| {
                let x = var.as_tensor();
                let target = Tensor::new(5.0f32, &device)?;
                let diff = (x - &target)?;
                let task_loss = (&diff * &diff)?;
                Ok(LossOutput::from_tensor(task_loss))
            })?;
        }

        // Should converge
        let final_val = var.to_vec0::<f32>()?;
        assert!(
            (final_val - 5.0).abs() < 3.0,
            "Expected value near 5.0, got {}",
            final_val
        );

        Ok(())
    }

    #[test]
    fn test_moe_aux_losses_directly() -> Result<()> {
        // Test MoE auxiliary losses directly (not through optimizer)
        let device = Device::Cpu;

        // Create router stats
        let router_logits = Tensor::new(
            &[
                [1.0f32, 2.0, 3.0, 4.0],
                [2.0, 1.0, 4.0, 3.0],
                [3.0, 4.0, 1.0, 2.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
            &device,
        )?;
        let selected_experts = Tensor::new(&[[3u32], [2u32], [1u32], [0u32]], &device)?;
        let mut stats = RouterStats::new(router_logits, selected_experts);

        // Test load balancing loss
        let lb_loss = LoadBalanceLoss::new(0.01, 4);
        let aux_loss = lb_loss.compute(&mut stats)?;
        assert!(aux_loss.to_vec0::<f32>()? > 0.0);

        // Test Z-loss
        let z_loss = ZLoss::new(0.001);
        let z_aux_loss = z_loss.compute(&stats)?;
        assert!(z_aux_loss.to_vec0::<f32>()? > 0.0);

        Ok(())
    }

    #[test]
    fn test_loss_output_from_tensor() -> Result<()> {
        let device = Device::Cpu;
        let loss = Tensor::new(1.5f32, &device)?; // Scalar tensor

        let loss_output = LossOutput::from_tensor(loss.clone());

        assert!(loss_output.router_stats.is_none());
        let mut loss_output_mut = loss_output;
        let total = loss_output_mut.total_loss(None, None, 1)?;
        let total_val = total.to_vec0::<f32>()?;

        assert!((total_val - 1.5).abs() < 1e-6);
        Ok(())
    }
}
