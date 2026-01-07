//! EGGROLL: Evolution Guided General Optimization via Low-rank Learning
//!
//! This crate implements the EGGROLL algorithm for efficient evolution strategies
//! optimization of neural networks. EGGROLL uses low-rank perturbations to scale
//! ES to billion-parameter models.
//!
//! # Key Features
//!
//! - **Low-rank perturbations**: Instead of full-rank E ∈ ℝ^(m×n), uses E = (1/√r)AB^T
//! - **Memory efficient**: Storage reduced from O(mn) to O(r(m+n)) per layer
//! - **Seed-based reconstruction**: Workers reconstruct perturbations from shared seeds
//! - **Rank-based fitness shaping**: Robust to fitness scaling issues
//! - **Momentum**: Accelerated convergence with momentum-based updates
//! - **Adaptive σ**: Perturbation magnitude adapts during training
//!
//! # Algorithm
//!
//! For each worker i in parallel:
//! 1. Reconstruct A_i, B_i from shared seed + member index
//! 2. Form perturbation E_i = (1/√r) × A_i × B_i^T
//! 3. Evaluate fitness f_i = f(μ + σ × E_i)
//! 4. Compute local update: α × rank(f_i) × E_i
//! 5. All-reduce: μ ← μ + (1/N) × Σ updates
//!
//! # Reference
//!
//! "Evolution Strategies at the Hyperscale" (arXiv:2511.16652)
//! Sarkar et al., 2025

use candle::{DType, Device, Result, Tensor};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::collections::HashMap;

/// Parameters for the EGGROLL optimizer
#[derive(Clone, Debug)]
pub struct EggrollParams {
    /// Learning rate for parameter updates
    pub lr: f64,
    /// Initial perturbation magnitude (σ in the paper)
    pub sigma: f64,
    /// Minimum sigma (for adaptive scheduling)
    pub sigma_min: f64,
    /// Sigma decay rate per generation
    pub sigma_decay: f64,
    /// Whether to use adaptive sigma based on fitness variance
    pub adaptive_sigma: bool,
    /// Rank of the low-rank perturbations (r in the paper)
    pub rank: usize,
    /// Population size (number of parallel perturbations)
    pub population_size: usize,
    /// Whether to use antithetic sampling (mirrored perturbations)
    pub antithetic: bool,
    /// Learning rate for scale factors (can be different from LoRA lr)
    pub scale_lr: Option<f64>,
    /// Momentum coefficient (0 = no momentum, 0.9 = typical)
    pub momentum: f64,
    /// Whether to use rank-based fitness shaping
    pub rank_based_fitness: bool,
    /// Base seed for deterministic perturbation generation
    pub base_seed: u64,
}

impl Default for EggrollParams {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            sigma: 0.001,
            sigma_min: 1e-5,
            sigma_decay: 0.999,
            adaptive_sigma: true,
            rank: 1,
            population_size: 8,
            antithetic: true,
            scale_lr: Some(0.01),
            momentum: 0.9,
            rank_based_fitness: true,
            base_seed: 0, // Will be set from system time if 0
        }
    }
}

/// A low-rank perturbation E = (1/√r) × A × B^T
#[derive(Debug, Clone)]
pub struct LowRankPerturbation {
    /// A matrix: (out_features, rank)
    pub a: Tensor,
    /// B matrix: (in_features, rank)
    pub b: Tensor,
    /// Rank of the perturbation
    pub rank: usize,
}

impl LowRankPerturbation {
    /// Create a new random low-rank perturbation
    pub fn new(
        out_features: usize,
        in_features: usize,
        rank: usize,
        device: &Device,
    ) -> Result<Self> {
        let a = Tensor::randn(0f32, 1f32, (out_features, rank), device)?;
        let b = Tensor::randn(0f32, 1f32, (in_features, rank), device)?;
        Ok(Self { a, b, rank })
    }

    /// Create a perturbation from a seed (deterministic reconstruction)
    pub fn from_seed(
        out_features: usize,
        in_features: usize,
        rank: usize,
        seed: u64,
        device: &Device,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);

        let a_data: Vec<f32> = (0..out_features * rank)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let b_data: Vec<f32> = (0..in_features * rank)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();

        let a = Tensor::from_vec(a_data, (out_features, rank), device)?;
        let b = Tensor::from_vec(b_data, (in_features, rank), device)?;

        Ok(Self { a, b, rank })
    }

    /// Create from existing A and B tensors
    pub fn from_tensors(a: Tensor, b: Tensor) -> Result<Self> {
        let rank = a.dim(1)?;
        Ok(Self { a, b, rank })
    }

    /// Compute the full perturbation matrix: (1/√r) × A × B^T
    /// Returns tensor of shape (out_features, in_features)
    pub fn to_full(&self) -> Result<Tensor> {
        let scale = 1.0 / (self.rank as f64).sqrt();
        let ab_t = self.a.matmul(&self.b.t()?)?;
        ab_t * scale
    }

    /// Apply the perturbation to an input: x @ (W + σE)^T = x @ W^T + σ/√r × (x @ B) @ A^T
    /// This is the efficient computation that avoids materializing the full perturbation
    pub fn apply_to_input(&self, x: &Tensor, sigma: f64) -> Result<Tensor> {
        let scale = sigma / (self.rank as f64).sqrt();

        // Handle batched input: flatten to 2D, apply LoRA, reshape back
        let x_dims = x.dims();
        if x_dims.len() == 3 {
            let (batch, seq, features) = (x_dims[0], x_dims[1], x_dims[2]);
            let x_2d = x.reshape((batch * seq, features))?;
            let xb = x_2d.matmul(&self.b)?;
            let out_2d = xb.matmul(&self.a.t()?)?;
            let out_features = self.a.dim(0)?;
            let out = out_2d.reshape((batch, seq, out_features))?;
            out * scale
        } else {
            let xb = x.matmul(&self.b)?;
            let xba_t = xb.matmul(&self.a.t()?)?;
            xba_t * scale
        }
    }

    /// Scale the perturbation by a factor (for fitness-weighted updates)
    pub fn scale(&self, factor: f64) -> Result<Self> {
        Ok(Self {
            a: (&self.a * factor)?,
            b: self.b.clone(),
            rank: self.rank,
        })
    }

    /// Add another perturbation to this one
    pub fn add(&self, other: &LowRankPerturbation) -> Result<Self> {
        let a = Tensor::cat(&[&self.a, &other.a], 1)?;
        let b = Tensor::cat(&[&self.b, &other.b], 1)?;
        Ok(Self {
            a,
            b,
            rank: self.rank + other.rank,
        })
    }

    /// Create a zero perturbation (for momentum initialization)
    pub fn zeros(
        out_features: usize,
        in_features: usize,
        rank: usize,
        device: &Device,
    ) -> Result<Self> {
        let a = Tensor::zeros((out_features, rank), DType::F32, device)?;
        let b = Tensor::zeros((in_features, rank), DType::F32, device)?;
        Ok(Self { a, b, rank })
    }
}

/// Perturbations for a single layer (both scale and LoRA)
#[derive(Debug, Clone)]
pub struct LayerPerturbations {
    /// Scale perturbation (additive delta to scale multipliers)
    pub scale_delta: Option<Tensor>,
    /// Low-rank adapter perturbation
    pub lora: Option<LowRankPerturbation>,
}

/// A population member with perturbations for all layers
#[derive(Debug, Clone)]
pub struct PopulationMember {
    /// Layer name -> perturbations
    pub perturbations: HashMap<String, LayerPerturbations>,
    /// Fitness score for main model layers (filled after evaluation)
    pub fitness: Option<f32>,
    /// Fitness score for MTP head layers (percentage of verified predictions)
    /// This is used exclusively for MTP layer weight updates
    pub mtp_fitness: Option<f32>,
    /// Whether this is a mirrored (antithetic) sample
    pub is_antithetic: bool,
    /// Seed used to generate this member's perturbations
    pub seed: u64,
    /// Member index within the population
    pub member_index: usize,
}

impl PopulationMember {
    /// Create a new population member with empty perturbations
    pub fn new() -> Self {
        Self {
            perturbations: HashMap::new(),
            fitness: None,
            mtp_fitness: None,
            is_antithetic: false,
            seed: 0,
            member_index: 0,
        }
    }

    /// Create a population member with seed (for reconstruction)
    pub fn with_seed(seed: u64, member_index: usize) -> Self {
        Self {
            perturbations: HashMap::new(),
            fitness: None,
            mtp_fitness: None,
            is_antithetic: false,
            seed,
            member_index,
        }
    }

    /// Create the antithetic (negated) version of this member
    pub fn antithetic(&self) -> Result<Self> {
        let mut perturbations = HashMap::new();
        for (name, layer_pert) in &self.perturbations {
            let scale_delta = if let Some(sd) = &layer_pert.scale_delta {
                Some(sd.neg()?)
            } else {
                None
            };
            let lora = if let Some(l) = &layer_pert.lora {
                Some(LowRankPerturbation {
                    a: l.a.neg()?,
                    b: l.b.clone(),
                    rank: l.rank,
                })
            } else {
                None
            };
            perturbations.insert(name.clone(), LayerPerturbations { scale_delta, lora });
        }
        Ok(Self {
            perturbations,
            fitness: None,
            mtp_fitness: None,
            is_antithetic: true,
            seed: self.seed,
            member_index: self.member_index,
        })
    }
}

impl Default for PopulationMember {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for which layers to optimize
#[derive(Debug, Clone, PartialEq)]
pub struct LayerConfig {
    /// Layer name (e.g., "blk.0.attn_q")
    pub name: String,
    /// Shape of the weight matrix (out_features, in_features)
    pub shape: (usize, usize),
    /// Number of scale blocks (for quantized layers)
    pub num_scale_blocks: Option<usize>,
    /// Whether to optimize scales
    pub optimize_scales: bool,
    /// Whether to optimize with LoRA
    pub optimize_lora: bool,
    /// Layer-specific learning rate multiplier (default 1.0)
    pub lr_multiplier: f64,
    /// Whether this is an MTP (Multi-Token Prediction) layer.
    /// MTP layers use mtp_fitness instead of regular fitness for updates.
    pub is_mtp: bool,
}

impl LayerConfig {
    /// Create a new layer config with default lr_multiplier
    pub fn new(
        name: String,
        shape: (usize, usize),
        num_scale_blocks: Option<usize>,
        optimize_scales: bool,
        optimize_lora: bool,
    ) -> Self {
        Self {
            name,
            shape,
            num_scale_blocks,
            optimize_scales,
            optimize_lora,
            lr_multiplier: 1.0,
            is_mtp: false,
        }
    }

    /// Mark this layer as an MTP layer (uses mtp_fitness for updates)
    pub fn as_mtp(mut self) -> Self {
        self.is_mtp = true;
        self
    }

    /// Check if this layer name indicates an MTP layer
    pub fn is_mtp_layer(name: &str) -> bool {
        name.contains(".mtp.") || name.starts_with("mtp.")
    }
}

/// Momentum state for a layer
/// NOTE: Reserved for future momentum-based optimization.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct LayerMomentum {
    /// Scale momentum
    scale: Option<Tensor>,
    /// LoRA A momentum
    lora_a: Option<Tensor>,
}

/// EGGROLL Optimizer
///
/// Manages a population of perturbations and performs evolution strategy updates.
pub struct Eggroll {
    /// Optimizer parameters
    params: EggrollParams,
    /// Layer configurations
    layer_configs: Vec<LayerConfig>,
    /// Current population
    population: Vec<PopulationMember>,
    /// Random number generator (for non-reproducible operations)
    #[allow(dead_code)]
    rng: StdRng,
    /// Device for tensor operations
    device: Device,
    /// Current mean scale values (the parameters being optimized)
    mean_scales: HashMap<String, Tensor>,
    /// Current mean LoRA values (accumulated low-rank updates)
    mean_lora: HashMap<String, LowRankPerturbation>,
    /// Momentum buffers for scales
    momentum_scales: HashMap<String, Tensor>,
    /// Momentum buffers for LoRA
    momentum_lora: HashMap<String, Tensor>,
    /// Generation counter
    generation: usize,
    /// Current sigma (may be adapted)
    current_sigma: f64,
    /// Fitness history for adaptive sigma
    fitness_history: Vec<f32>,
}

impl Eggroll {
    /// Create a new EGGROLL optimizer
    pub fn new(
        params: EggrollParams,
        layer_configs: Vec<LayerConfig>,
        device: &Device,
    ) -> Result<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = if params.base_seed == 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        } else {
            params.base_seed
        };

        Self::new_with_seed(params, layer_configs, device, seed)
    }

    /// Create a new EGGROLL optimizer with a specific seed
    pub fn new_with_seed(
        mut params: EggrollParams,
        layer_configs: Vec<LayerConfig>,
        device: &Device,
        seed: u64,
    ) -> Result<Self> {
        params.base_seed = seed;
        let mut mean_scales = HashMap::new();
        let mean_lora = HashMap::new();
        let momentum_scales = HashMap::new();
        let momentum_lora = HashMap::new();

        // Initialize mean scales to ones (no modification initially)
        for config in &layer_configs {
            if config.optimize_scales {
                if let Some(num_blocks) = config.num_scale_blocks {
                    let scales = Tensor::ones(num_blocks, DType::F32, device)?;
                    mean_scales.insert(config.name.clone(), scales);
                }
            }
        }

        let current_sigma = params.sigma;

        Ok(Self {
            params,
            layer_configs,
            population: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            device: device.clone(),
            mean_scales,
            mean_lora,
            momentum_scales,
            momentum_lora,
            generation: 0,
            current_sigma,
            fitness_history: Vec::new(),
        })
    }

    /// Get the generation seed for deterministic perturbation reconstruction
    pub fn generation_seed(&self) -> u64 {
        self.params
            .base_seed
            .wrapping_add(self.generation as u64 * 1_000_000)
    }

    /// Get the seed for a specific member within the current generation
    pub fn member_seed(&self, member_index: usize) -> u64 {
        self.generation_seed()
            .wrapping_add(member_index as u64 * 10_000)
    }

    /// Get the seed for a specific layer within a member
    pub fn layer_seed(&self, member_index: usize, layer_index: usize) -> u64 {
        self.member_seed(member_index)
            .wrapping_add(layer_index as u64)
    }

    /// Sample a new population of perturbations
    pub fn sample_population(&mut self) -> Result<()> {
        self.population.clear();

        let base_pop_size = if self.params.antithetic {
            self.params.population_size / 2
        } else {
            self.params.population_size
        };

        for member_idx in 0..base_pop_size {
            let member = self.sample_member(member_idx)?;
            if self.params.antithetic {
                let anti = member.antithetic()?;
                self.population.push(member);
                self.population.push(anti);
            } else {
                self.population.push(member);
            }
        }

        Ok(())
    }

    /// Sample perturbations for a single population member using deterministic seed
    fn sample_member(&mut self, member_index: usize) -> Result<PopulationMember> {
        let member_seed = self.member_seed(member_index);
        let mut member = PopulationMember::with_seed(member_seed, member_index);

        for (layer_idx, config) in self.layer_configs.iter().enumerate() {
            let layer_seed = self.layer_seed(member_index, layer_idx);
            let mut rng = StdRng::seed_from_u64(layer_seed);

            let mut layer_pert = LayerPerturbations {
                scale_delta: None,
                lora: None,
            };

            // Sample scale perturbation
            if config.optimize_scales {
                if let Some(num_blocks) = config.num_scale_blocks {
                    let data: Vec<f32> = (0..num_blocks)
                        .map(|_| rng.sample::<f32, _>(StandardNormal))
                        .collect();
                    let delta = Tensor::from_vec(data, num_blocks, &self.device)?;
                    layer_pert.scale_delta = Some(delta);
                }
            }

            // Sample LoRA perturbation
            if config.optimize_lora {
                let (out_features, in_features) = config.shape;
                let lora = LowRankPerturbation::from_seed(
                    out_features,
                    in_features,
                    self.params.rank,
                    layer_seed.wrapping_add(1_000_000), // Different seed for LoRA
                    &self.device,
                )?;
                layer_pert.lora = Some(lora);
            }

            if layer_pert.scale_delta.is_some() || layer_pert.lora.is_some() {
                member.perturbations.insert(config.name.clone(), layer_pert);
            }
        }

        Ok(member)
    }

    /// Reconstruct a population member from seed (for distributed workers)
    pub fn reconstruct_member(
        &self,
        member_index: usize,
        is_antithetic: bool,
    ) -> Result<PopulationMember> {
        let member_seed = self.member_seed(member_index);
        let mut member = PopulationMember::with_seed(member_seed, member_index);

        for (layer_idx, config) in self.layer_configs.iter().enumerate() {
            let layer_seed = self.layer_seed(member_index, layer_idx);
            let mut rng = StdRng::seed_from_u64(layer_seed);

            let mut layer_pert = LayerPerturbations {
                scale_delta: None,
                lora: None,
            };

            if config.optimize_scales {
                if let Some(num_blocks) = config.num_scale_blocks {
                    let data: Vec<f32> = (0..num_blocks)
                        .map(|_| rng.sample::<f32, _>(StandardNormal))
                        .collect();
                    let mut delta = Tensor::from_vec(data, num_blocks, &self.device)?;
                    if is_antithetic {
                        delta = delta.neg()?;
                    }
                    layer_pert.scale_delta = Some(delta);
                }
            }

            if config.optimize_lora {
                let (out_features, in_features) = config.shape;
                let mut lora = LowRankPerturbation::from_seed(
                    out_features,
                    in_features,
                    self.params.rank,
                    layer_seed.wrapping_add(1_000_000),
                    &self.device,
                )?;
                if is_antithetic {
                    lora.a = lora.a.neg()?;
                }
                layer_pert.lora = Some(lora);
            }

            if layer_pert.scale_delta.is_some() || layer_pert.lora.is_some() {
                member.perturbations.insert(config.name.clone(), layer_pert);
            }
        }

        member.is_antithetic = is_antithetic;
        Ok(member)
    }

    /// Get the current population
    pub fn population(&self) -> &[PopulationMember] {
        &self.population
    }

    /// Get mutable access to population (for setting fitness)
    pub fn population_mut(&mut self) -> &mut [PopulationMember] {
        &mut self.population
    }

    /// Get the population size
    pub fn population_size(&self) -> usize {
        self.population.len()
    }

    /// Get the scale values for a specific member
    pub fn get_member_scales(&self, member_idx: usize) -> Result<HashMap<String, Tensor>> {
        let member = &self.population[member_idx];
        let mut scales = HashMap::new();

        for (name, mean_scale) in &self.mean_scales {
            let scale = if let Some(pert) = member.perturbations.get(name) {
                if let Some(delta) = &pert.scale_delta {
                    let sigma = self.params.scale_lr.unwrap_or(self.current_sigma);
                    (mean_scale + &(delta * sigma)?)?
                } else {
                    mean_scale.clone()
                }
            } else {
                mean_scale.clone()
            };
            scales.insert(name.clone(), scale);
        }

        Ok(scales)
    }

    /// Get the combined LoRA for a specific member and layer
    pub fn get_member_lora(
        &self,
        member_idx: usize,
        layer_name: &str,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let perturbation = self.population[member_idx]
            .perturbations
            .get(layer_name)
            .and_then(|p| p.lora.as_ref());

        match (self.mean_lora.get(layer_name), perturbation) {
            (Some(mean), Some(pert)) => {
                let a = (&mean.a + &pert.a)?;
                let b = mean.b.clone();
                Ok(Some((a, b)))
            }
            (None, Some(pert)) => Ok(Some((pert.a.clone(), pert.b.clone()))),
            (Some(mean), None) => Ok(Some((mean.a.clone(), mean.b.clone()))),
            (None, None) => Ok(None),
        }
    }

    /// Get the mean scales
    pub fn mean_scales(&self) -> &HashMap<String, Tensor> {
        &self.mean_scales
    }

    /// Get the mean LoRA adapters
    pub fn mean_lora(&self) -> &HashMap<String, LowRankPerturbation> {
        &self.mean_lora
    }

    /// Replace mean scales (used when importing checkpoints)
    pub fn set_mean_scales(&mut self, scales: HashMap<String, Tensor>) {
        self.mean_scales = scales;
    }

    /// Replace mean LoRA adapters (used when importing checkpoints)
    pub fn set_mean_lora(&mut self, loras: HashMap<String, LowRankPerturbation>) {
        self.mean_lora = loras;
    }

    /// Get current sigma (may differ from initial if adaptive)
    pub fn sigma(&self) -> f64 {
        self.current_sigma
    }

    /// Get learning rate
    pub fn lr(&self) -> f64 {
        self.params.lr
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.params.rank
    }

    /// Compute rank-based fitness shaping
    /// Converts raw fitness to centered ranks in [-0.5, 0.5]
    fn rank_based_fitness_shaping(&self, fitnesses: &[f32]) -> Vec<f32> {
        let n = fitnesses.len();
        if n == 0 {
            return vec![];
        }

        // Get sorted indices (ascending order)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            fitnesses[a]
                .partial_cmp(&fitnesses[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign centered ranks
        let mut shaped = vec![0.0f32; n];
        for (rank, &idx) in indices.iter().enumerate() {
            // Map rank to [-0.5, 0.5]
            shaped[idx] = if n > 1 {
                (rank as f32 / (n - 1) as f32) - 0.5
            } else {
                0.0
            };
        }

        shaped
    }

    /// Compute normalized fitness (z-score normalization)
    fn normalize_fitness(&self, fitnesses: &[f32]) -> Vec<f32> {
        let n = fitnesses.len() as f32;
        let mean: f32 = fitnesses.iter().sum::<f32>() / n;
        let std = (fitnesses.iter().map(|f| (f - mean).powi(2)).sum::<f32>() / n)
            .sqrt()
            .max(1e-8);

        fitnesses.iter().map(|f| (f - mean) / std).collect()
    }

    /// Adapt sigma based on fitness variance
    fn adapt_sigma(&mut self, fitness_variance: f32) {
        if !self.params.adaptive_sigma {
            // Simple decay when not using adaptive sigma
            self.current_sigma =
                (self.current_sigma * self.params.sigma_decay).max(self.params.sigma_min);
            return;
        }

        // Store fitness variance in history
        self.fitness_history.push(fitness_variance);

        // Keep last 10 generations
        if self.fitness_history.len() > 10 {
            self.fitness_history.remove(0);
        }

        // If fitness variance is low (stuck), increase sigma; if high (progressing), decrease
        if self.fitness_history.len() >= 3 {
            let recent_avg: f32 =
                self.fitness_history.iter().sum::<f32>() / self.fitness_history.len() as f32;

            if recent_avg < 0.01 {
                // Very low variance - likely stuck, increase exploration
                self.current_sigma = (self.current_sigma * 1.1).min(self.params.sigma * 10.0);
            } else {
                // Normal progress, apply decay
                self.current_sigma =
                    (self.current_sigma * self.params.sigma_decay).max(self.params.sigma_min);
            }
        }
    }

    /// Perform the EGGROLL update step after fitness evaluation
    pub fn step(&mut self) -> Result<f32> {
        // Collect fitness values for main model
        let fitnesses: Vec<f32> = self
            .population
            .iter()
            .map(|m| m.fitness.expect("Fitness not set for population member"))
            .collect();

        // Collect MTP fitness values (use 0.0 if not set)
        let mtp_fitnesses: Vec<f32> = self
            .population
            .iter()
            .map(|m| m.mtp_fitness.unwrap_or(0.0))
            .collect();

        let n = fitnesses.len() as f32;
        let mean_fitness: f32 = fitnesses.iter().sum::<f32>() / n;
        let _mean_mtp_fitness: f32 = mtp_fitnesses.iter().sum::<f32>() / n;

        // Compute fitness variance for adaptive sigma
        let fitness_variance = fitnesses
            .iter()
            .map(|f| (f - mean_fitness).powi(2))
            .sum::<f32>()
            / n;

        // Apply fitness shaping for main model
        let shaped_fitness = if self.params.rank_based_fitness {
            self.rank_based_fitness_shaping(&fitnesses)
        } else {
            self.normalize_fitness(&fitnesses)
        };

        // Apply fitness shaping for MTP (only if any MTP fitness is set)
        let has_mtp_fitness = self.population.iter().any(|m| m.mtp_fitness.is_some());
        let shaped_mtp_fitness = if has_mtp_fitness {
            if self.params.rank_based_fitness {
                self.rank_based_fitness_shaping(&mtp_fitnesses)
            } else {
                self.normalize_fitness(&mtp_fitnesses)
            }
        } else {
            vec![0.0; self.population.len()]
        };

        // Update mean scales with momentum
        let scale_lr = self.params.scale_lr.unwrap_or(self.params.lr);
        for config in &self.layer_configs {
            if !config.optimize_scales {
                continue;
            }

            let name = &config.name;
            let effective_lr = scale_lr * config.lr_multiplier;

            // Use MTP fitness for MTP layers, regular fitness for others
            let fitness_to_use = if config.is_mtp {
                &shaped_mtp_fitness
            } else {
                &shaped_fitness
            };

            if let Some(mean_scale) = self.mean_scales.get_mut(name) {
                let mut gradient = Tensor::zeros(mean_scale.shape(), DType::F32, &self.device)?;

                for (member, &sf) in self.population.iter().zip(fitness_to_use.iter()) {
                    if let Some(pert) = member.perturbations.get(name) {
                        if let Some(delta) = &pert.scale_delta {
                            let weighted = (delta * (sf as f64))?;
                            gradient = (gradient + weighted)?;
                        }
                    }
                }

                // Average gradient
                gradient = (gradient / n as f64)?;

                // Apply momentum
                let momentum_coeff = self.params.momentum;
                let momentum = self.momentum_scales.entry(name.clone()).or_insert_with(|| {
                    Tensor::zeros(mean_scale.shape(), DType::F32, &self.device).unwrap()
                });

                let new_momentum = ((momentum.clone() * momentum_coeff)? + &gradient)?;
                *momentum = new_momentum.clone();

                // Update parameters: μ ← μ + lr × momentum
                *mean_scale = (mean_scale.clone() + (new_momentum * effective_lr)?)?;
            }
        }

        // Update mean LoRA adapters with momentum
        for config in &self.layer_configs {
            if !config.optimize_lora {
                continue;
            }

            let name = &config.name;
            let effective_lr = self.params.lr * config.lr_multiplier;

            // Use MTP fitness for MTP layers, regular fitness for others
            let fitness_to_use = if config.is_mtp {
                &shaped_mtp_fitness
            } else {
                &shaped_fitness
            };

            let mut a_gradient =
                Tensor::zeros((config.shape.0, self.params.rank), DType::F32, &self.device)?;

            for (member, &sf) in self.population.iter().zip(fitness_to_use.iter()) {
                if let Some(pert) = member.perturbations.get(name) {
                    if let Some(lora) = &pert.lora {
                        let weighted_a = (&lora.a * (sf as f64))?;
                        a_gradient = (a_gradient + weighted_a)?;
                    }
                }
            }

            // Average gradient
            a_gradient = (a_gradient / n as f64)?;

            // Apply momentum
            let momentum_coeff = self.params.momentum;
            let momentum_key = format!("{}_a", name);
            let momentum = self
                .momentum_lora
                .entry(momentum_key.clone())
                .or_insert_with(|| {
                    Tensor::zeros((config.shape.0, self.params.rank), DType::F32, &self.device)
                        .unwrap()
                });

            let new_momentum = ((momentum.clone() * momentum_coeff)? + &a_gradient)?;
            *momentum = new_momentum.clone();

            // Update mean LoRA
            if let Some(mean_lora) = self.mean_lora.get_mut(name) {
                mean_lora.a = (&mean_lora.a + (new_momentum * effective_lr)?)?;
            } else {
                // Initialize mean LoRA with the update
                let b = if let Some(member) = self.population.first() {
                    if let Some(pert) = member.perturbations.get(name) {
                        if let Some(lora) = &pert.lora {
                            lora.b.clone()
                        } else {
                            Tensor::randn(
                                0f32,
                                1f32,
                                (config.shape.1, self.params.rank),
                                &self.device,
                            )?
                        }
                    } else {
                        Tensor::randn(0f32, 1f32, (config.shape.1, self.params.rank), &self.device)?
                    }
                } else {
                    Tensor::randn(0f32, 1f32, (config.shape.1, self.params.rank), &self.device)?
                };

                self.mean_lora.insert(
                    name.clone(),
                    LowRankPerturbation {
                        a: (new_momentum * effective_lr)?,
                        b,
                        rank: self.params.rank,
                    },
                );
            }
        }

        // Adapt sigma
        self.adapt_sigma(fitness_variance);

        self.generation += 1;
        Ok(mean_fitness)
    }

    /// Get the current generation number
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Clear the population (after step)
    pub fn clear_population(&mut self) {
        self.population.clear();
    }

    /// Get layer configs
    pub fn layer_configs(&self) -> &[LayerConfig] {
        &self.layer_configs
    }

    /// Get base seed
    pub fn base_seed(&self) -> u64 {
        self.params.base_seed
    }
}

/// Helper trait for models that can be optimized with EGGROLL
pub trait EggrollModel {
    /// Get layer configurations for EGGROLL optimization
    fn eggroll_layer_configs(&self) -> Vec<LayerConfig>;

    /// Apply scales and LoRA perturbations from a population member
    fn apply_perturbations(
        &mut self,
        scales: &HashMap<String, Tensor>,
        loras: &HashMap<String, &LowRankPerturbation>,
        sigma: f64,
    ) -> Result<()>;

    /// Clear all perturbations
    fn clear_perturbations(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_rank_perturbation() -> Result<()> {
        let device = Device::Cpu;
        let pert = LowRankPerturbation::new(64, 128, 4, &device)?;

        assert_eq!(pert.a.dims(), &[64, 4]);
        assert_eq!(pert.b.dims(), &[128, 4]);

        let full = pert.to_full()?;
        assert_eq!(full.dims(), &[64, 128]);

        Ok(())
    }

    #[test]
    fn test_seed_based_reconstruction() -> Result<()> {
        let device = Device::Cpu;
        let seed = 12345u64;

        let pert1 = LowRankPerturbation::from_seed(64, 128, 4, seed, &device)?;
        let pert2 = LowRankPerturbation::from_seed(64, 128, 4, seed, &device)?;

        // Same seed should produce identical perturbations
        let diff_a = (&pert1.a - &pert2.a)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let diff_b = (&pert1.b - &pert2.b)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;

        assert!(diff_a < 1e-6, "A matrices should be identical");
        assert!(diff_b < 1e-6, "B matrices should be identical");

        Ok(())
    }

    #[test]
    fn test_rank_based_fitness() -> Result<()> {
        let device = Device::Cpu;
        let configs = vec![LayerConfig::new(
            "test".to_string(),
            (32, 64),
            Some(16),
            true,
            false,
        )];

        let params = EggrollParams {
            rank_based_fitness: true,
            ..Default::default()
        };

        let optimizer = Eggroll::new_with_seed(params, configs, &device, 42)?;

        // Test rank-based shaping
        let fitnesses = vec![0.1, 0.5, 0.3, 0.9];
        let shaped = optimizer.rank_based_fitness_shaping(&fitnesses);

        // 0.1 is lowest (rank 0) -> -0.5
        // 0.3 is next (rank 1) -> -0.166...
        // 0.5 is next (rank 2) -> 0.166...
        // 0.9 is highest (rank 3) -> 0.5
        assert!((shaped[0] - (-0.5)).abs() < 0.01);
        assert!((shaped[3] - 0.5).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_member_reconstruction() -> Result<()> {
        let device = Device::Cpu;
        let configs = vec![LayerConfig::new(
            "test".to_string(),
            (32, 64),
            Some(16),
            true,
            true,
        )];

        let params = EggrollParams {
            population_size: 4,
            rank: 2,
            antithetic: true,
            ..Default::default()
        };

        let mut optimizer = Eggroll::new_with_seed(params, configs, &device, 42)?;
        optimizer.sample_population()?;

        // Reconstruct member 0 and compare
        let reconstructed = optimizer.reconstruct_member(0, false)?;
        let original = &optimizer.population()[0];

        let orig_delta = original
            .perturbations
            .get("test")
            .unwrap()
            .scale_delta
            .as_ref()
            .unwrap();
        let recon_delta = reconstructed
            .perturbations
            .get("test")
            .unwrap()
            .scale_delta
            .as_ref()
            .unwrap();

        let diff = (orig_delta - recon_delta)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-6, "Reconstructed member should match original");

        Ok(())
    }

    #[test]
    fn test_antithetic_sampling() -> Result<()> {
        let device = Device::Cpu;
        let configs = vec![LayerConfig::new(
            "test".to_string(),
            (32, 64),
            Some(16),
            true,
            false,
        )];

        let params = EggrollParams {
            population_size: 4,
            antithetic: true,
            ..Default::default()
        };

        let mut optimizer = Eggroll::new_with_seed(params, configs, &device, 42)?;
        optimizer.sample_population()?;

        let pop = optimizer.population();
        assert_eq!(pop.len(), 4);
        assert!(!pop[0].is_antithetic);
        assert!(pop[1].is_antithetic);

        // Check that antithetic pairs sum to zero
        let delta0 = pop[0]
            .perturbations
            .get("test")
            .unwrap()
            .scale_delta
            .as_ref()
            .unwrap();
        let delta1 = pop[1]
            .perturbations
            .get("test")
            .unwrap()
            .scale_delta
            .as_ref()
            .unwrap();
        let sum = (delta0 + delta1)?;
        let sum_val = sum.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(sum_val < 1e-6, "Antithetic pairs should sum to zero");

        Ok(())
    }

    #[test]
    fn test_momentum_update() -> Result<()> {
        let device = Device::Cpu;
        let configs = vec![LayerConfig::new(
            "test".to_string(),
            (8, 16),
            Some(4),
            true,
            false,
        )];

        let params = EggrollParams {
            population_size: 4,
            momentum: 0.9,
            rank_based_fitness: true,
            antithetic: false,
            ..Default::default()
        };

        let mut optimizer = Eggroll::new_with_seed(params, configs, &device, 42)?;

        // Run a few generations
        for gen in 0..3 {
            optimizer.sample_population()?;

            // Set some fitness values
            for (i, member) in optimizer.population_mut().iter_mut().enumerate() {
                member.fitness = Some((i as f32 + 1.0) / 5.0);
            }

            let mean_fitness = optimizer.step()?;
            assert!(
                mean_fitness > 0.0,
                "Generation {} should have positive mean fitness",
                gen
            );
        }

        // Check that momentum was accumulated
        assert!(
            !optimizer.momentum_scales.is_empty(),
            "Should have momentum state"
        );

        Ok(())
    }
}
