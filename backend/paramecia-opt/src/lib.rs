//! Paramecia optimization library
//!
//! This crate provides QuZO (Quantized Zeroth-Order) optimization for fine-tuning
//! quantized language models without backpropagation.
//!
//! ## Modules
//!
//! - [`distillation`]: Knowledge distillation training using teacher model outputs
//! - [`fuse`]: Task arithmetic fusion of multiple GGUF models
//! - [`prune`]: Expert and layer pruning for GGUF models
//!
//! ## QuZO Optimization
//!
//! QuZO (Quantized Zeroth-Order) optimization enables training of quantized models
//! without backpropagation by using finite differences to estimate gradients.

#[allow(dead_code)]
mod distillation;
mod fuse;
#[allow(dead_code)]
mod prune;
#[allow(dead_code)]
mod qzo;
#[allow(dead_code)]
mod tune;

pub use distillation::{
    parse_optimize_tensors, run_training_step_with_grad_accum, save_trained_model,
    update_gguf_metadata, DistillationLoss, DistillationLossConfig, EpsilonConfig, MtpLossConfig,
    OptimizeTensors, TuningData,
};
pub use fuse::{fuse_models, parse_model_spec, FuseOptions, QuantConflictStrategy};
pub use prune::{prune_experts, prune_layers, PruneExpertsOptions, PruneLayersOptions};
pub use qzo::{DecomposedZOState, ErrorFeedbackMode, ParamsQuZO, QuZO};
