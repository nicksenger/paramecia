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
pub use executor::{
    describe_model, fuse_models, graft_composite_from_paths, prune_experts, prune_layers,
    update_model_metadata, ModelEngine,
};
pub use model_actor::TrainingConfig;
pub use types::*;

// Re-export types needed by downstream crates (paramecia-text)
pub use paramecia_model::gguf_file;
pub use paramecia_model::vis;
pub use paramecia_model::{
    select_best_device, DType, Device, DeviceOffloadMode, GgmlDType, KvCacheQuantization,
    LayerDeviceMap, Tensor, TokenOutputStream, YarnConfig,
};
pub use paramecia_opt::{parse_model_spec, PerturbationMode};

/// Build the model computation graph used by the visualizer.
pub fn model_visualization_graph() -> vis::Graph {
    paramecia_model::visualization_graph()
}
