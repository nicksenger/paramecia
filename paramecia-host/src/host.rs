//! Host state for the WASM controller.
//!
//! Contains `HostState` (the WASI context, model registries, and UUID generation)
//! and the WIT trait placeholder implementations. All actual logic is handled
//! by linker registrations in runtime.rs.

#![allow(clippy::manual_async_fn)]

use crate::paramecia::host::types as wit;
use paramecia_bridge::ControllerHostEndpoint;
use paramecia_engine::types::Uuid;
use paramecia_engine::{DeviceOffloadMode, KvCacheQuantization, ModelEngine, TrainingConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use wasmtime::component::{Accessor, HasData, ResourceTable, StreamReader};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

/// A loaded model in the registry.
pub struct ModelEntry {
    pub executor: ModelEngine,
    pub source_path: PathBuf,
}

/// Tracks which model owns a snapshot.
pub struct SnapshotEntry {
    pub model_id: Uuid,
    pub executor_snapshot_id: Uuid,
}

/// Configuration for building new ModelEngines via load-model.
#[derive(Clone)]
pub struct ModelBuilderConfig {
    pub tokenizer_path: Option<PathBuf>,
    pub tokenizer_repo: String,
    pub cpu: bool,
    pub offload_mode: DeviceOffloadMode,
    pub kv_cache_quant: KvCacheQuantization,
    pub layer_split: Option<String>,
    pub prefetch: bool,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: usize,
    pub tail_samples: usize,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub penalty_last_n: usize,
    pub snapshot_dir: PathBuf,
    pub inference_mtp_speculation_depth: Option<usize>,
    pub training_mtp_speculation_depth: Option<usize>,
    pub mtp_allow_inference_override: bool,
    pub training_config: TrainingConfig,
}

/// Host state - multi-model registries with UUID-based resource tracking
pub struct HostState {
    /// Model registry: UUID → ModelEntry (unified executor per model)
    pub models: HashMap<Uuid, ModelEntry>,
    /// Checkpoint registry: UUID → file path on disk
    pub checkpoints: HashMap<Uuid, PathBuf>,
    /// In-memory snapshot registry: UUID → snapshot entry
    pub snapshots: HashMap<Uuid, SnapshotEntry>,
    /// Persisted snapshot registry: UUID → file path on disk
    pub persisted_snapshots: HashMap<Uuid, PathBuf>,
    /// Per-model cancellation senders for predict-completion
    pub prediction_cancels: Arc<Mutex<HashMap<Uuid, oneshot::Sender<()>>>>,
    /// Per-model cancellation senders for training
    pub training_cancels: Arc<Mutex<HashMap<Uuid, oneshot::Sender<()>>>>,
    /// Path to the default model GGUF (used for Weights::HostDefault)
    pub default_model_path: PathBuf,
    /// Directory for snapshot files
    pub snapshot_dir: PathBuf,
    /// Directory for checkpoint files
    pub checkpoint_dir: PathBuf,
    /// Device for model loading
    pub device: paramecia_engine::Device,
    /// Configuration for building new ModelEngines
    pub builder_config: ModelBuilderConfig,
    /// UUID generation counter
    pub next_uuid: u64,
    /// WASI context
    pub wasi: WasiCtx,
    /// Resource table (required by WASI)
    pub table: ResourceTable,
    /// Interactive endpoint for connect() handler (taken once on first connect call)
    pub interactive_endpoint: Option<ControllerHostEndpoint>,
}

impl HostState {
    /// Generate a new unique UUID.
    pub fn next_uuid(&mut self) -> Uuid {
        let id = self.next_uuid;
        self.next_uuid += 1;
        (0, id)
    }

    /// Look up a model executor by UUID.
    pub fn lookup_model(&self, model_id: &Uuid) -> Result<&ModelEntry, wit::Error> {
        self.models
            .get(model_id)
            .ok_or_else(|| wit::Error::ModelError(format!("Model {:?} not found", model_id)))
    }
}

impl WasiView for HostState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl HasData for HostState {
    type Data<'a> = &'a mut HostState;
}

// Empty types::Host (no functions in the types interface)
impl crate::paramecia::host::types::Host for HostState {}

// --- structure ---

impl crate::paramecia::host::structure::Host for HostState {}

impl crate::paramecia::host::structure::HostWithStore for HostState {
    fn load_model<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _source: wit::Weights,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Model, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "load_model is handled by the linker registration",
            ))
        }
    }

    fn unload_model<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "unload_model is handled by the linker registration",
            ))
        }
    }

    fn save_checkpoint<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Checkpoint, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "save_checkpoint is handled by the linker registration",
            ))
        }
    }

    fn delete_checkpoint<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _checkpoint: wit::Checkpoint,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "delete_checkpoint is handled by the linker registration",
            ))
        }
    }

    fn describe_model<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _source: wit::Weights,
    ) -> impl std::future::Future<
        Output = wasmtime::Result<Result<wit::ModelDescription, wit::Error>>,
    > + Send {
        async move {
            Err(wasmtime::Error::msg(
                "describe_model is handled by the linker registration",
            ))
        }
    }

    fn update_metadata<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _checkpoint: wit::Checkpoint,
        _metadata: Vec<(String, wit::MetadataValue)>,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "update_metadata is handled by the linker registration",
            ))
        }
    }
}

// --- structure-ext ---

impl crate::paramecia::host::structure_ext::Host for HostState {
    fn drop_snapshot(
        &mut self,
        _snapshot: wit::Snapshot,
    ) -> impl std::future::Future<Output = wasmtime::Result<()>> + Send {
        std::future::ready(Err(wasmtime::Error::msg(
            "drop_snapshot is handled by the linker registration",
        )))
    }
}

impl crate::paramecia::host::structure_ext::HostWithStore for HostState {
    fn take_snapshot<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Snapshot, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "take_snapshot is handled by the linker registration",
            ))
        }
    }

    fn persist_snapshot<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _handle: wit::Snapshot,
    ) -> impl std::future::Future<
        Output = wasmtime::Result<Result<wit::PersistedSnapshot, wit::Error>>,
    > + Send {
        async move {
            Err(wasmtime::Error::msg(
                "persist_snapshot is handled by the linker registration",
            ))
        }
    }

    fn delete_snapshot<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _snapshot: wit::PersistedSnapshot,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "delete_snapshot is handled by the linker registration",
            ))
        }
    }

    fn restore_snapshot<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
        _snapshot: wit::Snapshot,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "restore_snapshot is handled by the linker registration",
            ))
        }
    }

    fn load_snapshot<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
        _snapshot: wit::PersistedSnapshot,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Snapshot, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "load_snapshot is handled by the linker registration",
            ))
        }
    }

    fn reseed<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _seed: u64,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "reseed is handled by the linker registration",
            ))
        }
    }

    fn prune_experts<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _source: wit::Weights,
        _request: wit::ExpertPruningRequest,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Checkpoint, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "prune_experts is handled by the linker registration",
            ))
        }
    }

    fn prune_layers<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _source: wit::Weights,
        _retained: wit::Layers,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Checkpoint, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "prune_layers is handled by the linker registration",
            ))
        }
    }

    fn graft<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _composite: wit::ModelComposite,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Checkpoint, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "graft is handled by the linker registration",
            ))
        }
    }

    fn fuse<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _base: wit::Weights,
        _members: Vec<wit::FusionMember>,
        _strategy: wit::QuantConflictStrategy,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Checkpoint, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "fuse is handled by the linker registration",
            ))
        }
    }
}

// --- inference ---

impl crate::paramecia::host::inference::Host for HostState {
    fn fill_context(
        &mut self,
        _model: wit::Model,
        _input: Vec<wit::ModelInput>,
    ) -> impl std::future::Future<Output = wasmtime::Result<StreamReader<Result<u32, wit::Error>>>> + Send
    {
        std::future::ready(Err(wasmtime::Error::msg(
            "fill_context is handled by the linker registration",
        )))
    }

    fn predict_completion(
        &mut self,
        _model: wit::Model,
    ) -> impl std::future::Future<
        Output = wasmtime::Result<StreamReader<Result<wit::Predicted, wit::Error>>>,
    > + Send {
        std::future::ready(Err(wasmtime::Error::msg(
            "predict_completion is handled by the linker registration",
        )))
    }
}

impl crate::paramecia::host::inference::HostWithStore for HostState {
    fn cancel_prediction<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "cancel_prediction is handled by the linker registration",
            ))
        }
    }
}

// --- inference-ext ---

impl crate::paramecia::host::inference_ext::Host for HostState {
    fn predict_completions_batched(
        &mut self,
        _model: wit::Model,
        _input: Vec<Vec<wit::ModelInput>>,
    ) -> impl std::future::Future<
        Output = wasmtime::Result<StreamReader<Result<Vec<wit::Predicted>, wit::Error>>>,
    > + Send {
        std::future::ready(Err(wasmtime::Error::msg(
            "predict_completions_batched is handled by the linker registration",
        )))
    }

    fn commit_token(
        &mut self,
        _model: wit::Model,
        _token: u32,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        std::future::ready(Err(wasmtime::Error::msg(
            "commit_token is handled by the linker registration",
        )))
    }
}

impl crate::paramecia::host::inference_ext::HostWithStore for HostState {
    fn predict_token<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Predicted, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "predict_token is handled by the linker registration",
            ))
        }
    }
}

// --- training ---

impl crate::paramecia::host::training::Host for HostState {
    fn train_model(
        &mut self,
        _model: wit::Model,
        _data: StreamReader<wit::TrainingSample>,
    ) -> impl std::future::Future<
        Output = wasmtime::Result<
            Result<StreamReader<Result<wit::StepResult, wit::Error>>, wit::Error>,
        >,
    > + Send {
        std::future::ready(Err(wasmtime::Error::msg(
            "train_model is handled by the linker registration",
        )))
    }

    fn validate_model(
        &mut self,
        _model: wit::Model,
        _data: StreamReader<wit::TrainingSample>,
    ) -> impl std::future::Future<
        Output = wasmtime::Result<
            Result<StreamReader<Result<wit::StepResult, wit::Error>>, wit::Error>,
        >,
    > + Send {
        std::future::ready(Err(wasmtime::Error::msg(
            "validate_model is handled by the linker registration",
        )))
    }
}

impl crate::paramecia::host::training::HostWithStore for HostState {
    fn cancel_training<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "cancel_training is handled by the linker registration",
            ))
        }
    }

    fn set_hyper_parameters<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
        _update: wit::HyperParameterUpdate,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<(), wit::Error>>> + Send {
        async move {
            Err(wasmtime::Error::msg(
                "set_hyper_parameters is handled by the linker registration",
            ))
        }
    }
}

// --- interactive ---
// connect() is async in WIT and handled by the linker in runtime.rs

// --- training-ext ---

impl crate::paramecia::host::training_ext::Host for HostState {}

impl crate::paramecia::host::training_ext::HostWithStore for HostState {
    fn perturb_up<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _model: wit::Model,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::PositiveModel, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "perturb_up is handled by the linker registration",
            ))
        }
    }

    fn perturb_down<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _perturbed: wit::PositiveModel,
        _loss_up: f32,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::NegativeModel, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "perturb_down is handled by the linker registration",
            ))
        }
    }

    fn update<T: 'static>(
        _accessor: &Accessor<T, Self>,
        _perturbed: wit::NegativeModel,
        _loss_down: f32,
    ) -> impl std::future::Future<Output = wasmtime::Result<Result<wit::Model, wit::Error>>> + Send
    {
        async move {
            Err(wasmtime::Error::msg(
                "update is handled by the linker registration",
            ))
        }
    }
}
