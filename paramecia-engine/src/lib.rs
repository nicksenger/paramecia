//! Paramecia Engine — centralized inference and training logic.
//!
//! This crate provides `ModelEngine` which consolidates duplicated
//! inference/training loops from paramecia-controller, paramecia-text,
//! and the various examples into a single crate.

pub mod builder;
pub mod distribution;
pub mod executor;
pub(crate) mod model_actor;
pub mod tensor_trace_agg;
pub mod training;
pub mod types;

pub use builder::ModelEngineBuilder;
pub use executor::{describe_model, fuse_models, ModelEngine};
pub use model_actor::TrainingConfig;
pub use types::*;

// Re-export types needed by downstream crates (paramecia-text)
pub use paramecia_model::models::qwen3_next::{
    DeviceOffloadMode, KvCacheQuantization, LayerDeviceMap,
};
pub use paramecia_model::token_output_stream::TokenOutputStream;
pub use paramecia_model::YarnConfig;
pub use paramecia_opt::fuse::parse_model_spec;
