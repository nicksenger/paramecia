//! Knowledge distillation training using QuZO optimization.
//!
//! This module provides tools for training Qwen3-Next models using knowledge
//! distillation from teacher model outputs stored in binary tuning files.
//!
//! ## Overview
//!
//! The distillation process uses Quantized Zeroth-Order (QuZO) optimization
//! to train the model without backpropagation. This enables training of
//! quantized models directly on consumer hardware.
//!
//! ## Workflow
//!
//! 1. **Weight Perturbation**: Perturb weights by +εz (random direction)
//! 2. **Chunked Forward Pass**: Process long sequences in chunks, accumulating
//!    DeltaNet state and loss across chunks
//! 3. **Batch Accumulation**: Accumulate loss across N conversations
//! 4. **Opposite Perturbation**: Repeat with -εz perturbation
//! 5. **Gradient Estimation**: Estimate gradient using central differences
//! 6. **Update**: Apply optimizer update (Adam/SGD)
//!
//! ## Loss Components
//!
//! - **Distillation Loss**: KL-divergence between teacher and student distributions
//!   using sparse probability representation (top-k + importance-weighted tail samples)
//! - **Z-Loss**: Encourages router logits to stay near zero for numerical stability
//! - **Load Balance Loss**: Encourages balanced expert utilization across the batch

mod loss;
mod trainer;
mod training_step;
mod tuning_reader;

pub use loss::{DistillationLoss, DistillationLossConfig, MtpLossConfig};
pub use trainer::{
    save_trained_model, update_gguf_metadata, DistillationTrainer, DistillationTrainerConfig,
    EpsilonConfig, OptimizeTensors,
};
pub use training_step::{parse_optimize_tensors, run_training_step_with_grad_accum};
pub use tuning_reader::{PaddedTuningData, TuningData, TuningDataset};
