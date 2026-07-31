//! Rust-native types mirroring the WIT interface types.
//!
//! These are plain Rust structs with no wasmtime dependency, allowing the engine
//! to be used from both WASM host (paramecia-controller) and native (paramecia-text) contexts.

/// UUID type as (high, low) u64 pair.
pub type Uuid = (u64, u64);

/// Dedicated ID types — all backed by (u64, u64) but semantically distinct.
pub type ModelId = (u64, u64);
pub type CheckpointId = (u64, u64);
pub type SnapshotId = (u64, u64);
pub type ToolCallId = (u64, u64);
pub type SampleId = (u64, u64);

/// A model actively loaded on the host.
#[derive(Debug, Clone)]
pub struct Model {
    pub id: ModelId,
    pub n_experts: u32,
    pub n_layers: u32,
}

/// A saved model checkpoint.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub id: CheckpointId,
}

/// Snapshot of a model's internal state.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub id: SnapshotId,
    pub state_position: u64,
    pub num_tokens: u32,
}

/// Persisted snapshot of a model's state.
#[derive(Debug, Clone)]
pub struct PersistedSnapshot {
    pub id: SnapshotId,
}

/// Model weights residing on disk.
#[derive(Debug, Clone)]
pub enum Weights {
    /// Weights provided via config/environment/argument to the host application.
    HostDefault,
    /// Weights saved during controller execution.
    Checkpoint(Checkpoint),
    /// Weights from a specific path.
    Path(String),
}

/// Per-layer expert indices for pruning.
#[derive(Debug, Clone)]
pub struct Experts {
    pub indices: Vec<Vec<u32>>,
}

/// Weighted expert contributions for Shapley-based expert merging.
#[derive(Debug, Clone)]
pub struct ExpertContributions {
    pub weights: Vec<Vec<f32>>,
}

/// Request for expert pruning, bundling strategy with its required data.
#[derive(Debug, Clone)]
pub enum ExpertPruningRequest {
    /// Drop pruned experts entirely.
    Naive(Experts),
    /// Merge pruned expert knowledge back into retained experts.
    ShapleyMerge(ExpertMergeSpec),
}

/// Expert pruning via Shapley merging.
#[derive(Debug, Clone)]
pub struct ExpertMergeSpec {
    pub retained: Experts,
    pub contributions: ExpertContributions,
}

/// Layer indices for pruning.
#[derive(Debug, Clone)]
pub struct Layers {
    pub indices: Vec<u32>,
}

/// A member in task-arithmetic model fusion.
#[derive(Debug, Clone)]
pub struct FusionMember {
    pub weights: Weights,
    pub contribution: f32,
}

/// How to resolve differing quantization dtypes across fusion members.
#[derive(Debug, Clone, Copy)]
pub enum QuantConflictStrategy {
    Reject,
    Highest,
    Lowest,
}

/// Identifies a specific layer within a model's weights.
#[derive(Debug, Clone)]
pub struct LayerRef {
    pub source: Weights,
    pub layer_idx: u32,
}

/// Identifies a specific expert within a model.
#[derive(Debug, Clone)]
pub struct ExpertRef {
    pub layer: LayerRef,
    pub expert_idx: u32,
}

/// Mapping of architectural elements for grafting.
/// Each layer is independently sourced from a specific layer in a model.
#[derive(Debug, Clone)]
pub struct ModelComposite {
    pub embedding: Weights,
    pub layers: Vec<LayerRef>,
    pub lm_head: Option<Weights>,
    pub mtp_head: Option<Weights>,
}

/// A single entry in a logit distribution (token ID + log probability).
#[derive(Debug, Clone)]
pub struct LogitEntry {
    pub token_id: u32,
    pub log_prob: f32,
}

/// A soft token position for dark-knowledge transfer between model forward passes.
/// Carries the predicted (committed) token ID and a top-K distribution from a teacher model.
#[derive(Debug, Clone)]
pub struct SoftToken {
    /// The predicted (committed) token ID for this position.
    pub predicted: u32,
    /// Top-K logit entries representing the teacher model's distribution at this position.
    pub dark_knowledge: Vec<LogitEntry>,
}

/// A predicted token with distribution information.
#[derive(Debug, Clone)]
pub struct Predicted {
    pub token_id: u32,
    pub text: Option<String>,
    pub top_k: Vec<LogitEntry>,
    pub tail: Vec<LogitEntry>,
    pub tail_mass: f32,
    /// Expert indices from MoE routing, flattened across layers.
    /// Empty for non-MoE models or when capture is disabled.
    pub expert_indices: Vec<u32>,
}

/// Inputs provided for a model forward pass.
#[derive(Debug, Clone)]
pub enum ModelInput {
    /// Text context (tokenized by the host).
    Text(String),
    /// Specific token IDs.
    Tokens(Vec<u32>),
    /// Soft prompt: a sequence of soft tokens carrying predicted token IDs and
    /// dark-knowledge distributions. The host computes a weighted embedding per position
    /// from each soft token's dark_knowledge, producing a [1, seq_len, hidden_dim] tensor.
    Soft(Vec<SoftToken>),
}

/// A segment of training data: either masked context or teacher-labeled generation.
#[derive(Debug, Clone)]
pub enum TrainingData {
    /// Non-generated text (masked during loss computation).
    Context(ModelInput),
    /// Teacher model output — logits for generated tokens.
    Target(Vec<Predicted>),
}

/// A complete training sample with interleaved context and generation segments.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub id: SampleId,
    pub data: Vec<TrainingData>,
}

/// A batch of training data for fine-grained training operations.
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub data: Vec<TrainingData>,
}

/// Result of a training step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub loss: f64,
    pub sample_ids: Vec<SampleId>,
    pub n_tokens: u64,
}

/// A model which has been perturbed in the positive direction.
#[derive(Debug, Clone)]
pub struct PositiveModel {
    pub model: Model,
}

/// A model which has been perturbed in the negative direction.
#[derive(Debug, Clone)]
pub struct NegativeModel {
    pub model: Model,
}

/// Parameters for error feedback in quantized zeroth-order optimization.
#[derive(Debug, Clone)]
pub struct ErrorFeedbackParams {
    pub decay: f64,
    pub gain: f64,
}

/// Parameters for replay-based error feedback.
#[derive(Debug, Clone)]
pub struct ReplayParams {
    pub steps: u32,
    pub decay: f64,
    pub gain: f64,
}

/// Error feedback mode for accumulated residuals in quantized zeroth-order optimization.
#[derive(Debug, Clone)]
pub enum ErrorFeedbackMode {
    /// No error feedback.
    None,
    /// FP16 residuals stored per element.
    Persistent(ErrorFeedbackParams),
    /// Reconstruct residuals by replaying the last N steps.
    Replay(ReplayParams),
}

/// Model component types for per-component epsilon multipliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpsilonComponent {
    Embedding,
    Attention,
    MoeGating,
    MoeExperts,
    MoeExpertBanks,
    Mtp,
    Norms,
    Ssm,
    Other,
}

/// All-optional record for dynamically adjusting training hyperparameters.
#[derive(Debug, Clone, Default)]
pub struct HyperParameterUpdate {
    pub mtp_decay: Option<f64>,
    pub num_speculative: Option<u8>,
    pub temperature: Option<f64>,
    pub lb_loss: Option<f64>,
    pub z_loss: Option<f64>,
    pub clip_threshold: Option<f64>,
    /// QuZO learning rate.
    pub lr: Option<f64>,
    /// QuZO base perturbation magnitude.
    pub epsilon: Option<f64>,
    /// Per-component epsilon multipliers.
    pub epsilon_multipliers: Option<Vec<(EpsilonComponent, f64)>>,
    /// Error feedback mode.
    pub error_feedback: Option<ErrorFeedbackMode>,
}

/// GGML quantization dtype.
#[derive(Debug, Clone, Copy)]
pub enum GgmlDtype {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

/// Tensor descriptor from a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<u32>,
    pub dtype: GgmlDtype,
}

/// A GGUF metadata value.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

/// Description of a model's tensors and metadata, read from GGUF headers.
#[derive(Debug, Clone)]
pub struct ModelDescription {
    pub n_experts: u32,
    pub n_layers: u32,
    pub n_total_parameters: u64,
    pub n_active_parameters: u64,
    pub metadata: Vec<(String, MetadataValue)>,
    pub tensors: Vec<GgufTensor>,
}

/// Unified errors that can occur during engine operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("Not available: {0}")]
    NotAvailable(String),
    #[error("Train error: {0}")]
    TrainError(String),
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),
    #[error("Invalid state: {0}")]
    InvalidState(String),
    #[error("Model error: {0}")]
    ModelError(String),
    #[error("Tokenize error: {0}")]
    TokenizeError(String),
    #[error("Snapshot error: {0}")]
    SnapshotError(String),
}
