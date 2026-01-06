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

pub mod distillation;
pub mod fuse;
pub mod prune;
pub mod qzo;
pub mod tune;

// Re-export QuZO (Quantized Zeroth-Order optimization for discrete weights)
pub use crate::qzo::{
    parse_error_feedback_mode, DecomposedZOState, ErrorFeedbackMode, ParamsQuZO, QuZO,
    QuZOPerturbation,
};

// Re-export distillation types
pub use distillation::{
    load_tokenizer_for_training, parse_optimize_tensors, run_training_step_with_grad_accum,
    save_trained_model, update_gguf_metadata, DistillationLoss, DistillationLossConfig,
    DistillationTrainer, DistillationTrainerConfig, EpsilonConfig, MtpLossConfig, OptimizeTensors,
    StepStats, TrainingCheckpoint, TrainingState, TuningData, TuningDataset, TuningReader,
};
