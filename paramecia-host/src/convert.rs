//! Bidirectional conversions between executor types and WIT-generated types.
//!
//! The executor crate defines Rust-native types mirroring the WIT interface.
//! This module provides `From` impls so the controller can convert at the
//! boundary between the actor (executor types) and the WASM guest (WIT types).

use crate::paramecia::host::types as wit;
use paramecia_engine::types as exec;

// ---------------------------------------------------------------------------
// Executor → WIT
// ---------------------------------------------------------------------------

impl From<exec::LogitEntry> for wit::LogitEntry {
    fn from(e: exec::LogitEntry) -> Self {
        wit::LogitEntry {
            token_id: e.token_id,
            log_prob: e.log_prob,
        }
    }
}

impl From<exec::Predicted> for wit::Predicted {
    fn from(p: exec::Predicted) -> Self {
        wit::Predicted {
            token_id: p.token_id,
            text: p.text,
            top_k: p.top_k.into_iter().map(Into::into).collect(),
            tail: p.tail.into_iter().map(Into::into).collect(),
            tail_mass: p.tail_mass,
            expert_indices: p.expert_indices,
        }
    }
}

impl From<exec::Model> for wit::Model {
    fn from(m: exec::Model) -> Self {
        wit::Model {
            id: m.id,
            n_experts: m.n_experts,
            n_layers: m.n_layers,
        }
    }
}

impl From<exec::Checkpoint> for wit::Checkpoint {
    fn from(c: exec::Checkpoint) -> Self {
        wit::Checkpoint { id: c.id }
    }
}

impl From<exec::Snapshot> for wit::Snapshot {
    fn from(s: exec::Snapshot) -> Self {
        wit::Snapshot {
            id: s.id,
            state_position: s.state_position,
            num_tokens: s.num_tokens,
        }
    }
}

impl From<exec::PersistedSnapshot> for wit::PersistedSnapshot {
    fn from(p: exec::PersistedSnapshot) -> Self {
        wit::PersistedSnapshot { id: p.id }
    }
}

impl From<exec::Weights> for wit::Weights {
    fn from(w: exec::Weights) -> Self {
        match w {
            exec::Weights::HostDefault => wit::Weights::HostDefault,
            exec::Weights::Checkpoint(c) => wit::Weights::Checkpoint(c.into()),
            exec::Weights::Path(p) => wit::Weights::Path(p),
        }
    }
}

impl From<exec::StepResult> for wit::StepResult {
    fn from(r: exec::StepResult) -> Self {
        wit::StepResult {
            loss: r.loss,
            sample_ids: r.sample_ids,
            n_tokens: r.n_tokens,
        }
    }
}

impl From<exec::PositiveModel> for wit::PositiveModel {
    fn from(p: exec::PositiveModel) -> Self {
        wit::PositiveModel {
            model: p.model.into(),
        }
    }
}

impl From<exec::NegativeModel> for wit::NegativeModel {
    fn from(n: exec::NegativeModel) -> Self {
        wit::NegativeModel {
            model: n.model.into(),
        }
    }
}

impl From<exec::ModelInput> for wit::ModelInput {
    fn from(i: exec::ModelInput) -> Self {
        match i {
            exec::ModelInput::Text(s) => wit::ModelInput::Text(s),
            exec::ModelInput::Tokens(ids) => wit::ModelInput::Tokens(ids),
            exec::ModelInput::Soft(tokens) => {
                wit::ModelInput::Soft(tokens.into_iter().map(|t| wit::SoftToken {
                    predicted: t.predicted,
                    dark_knowledge: t.dark_knowledge.into_iter().map(Into::into).collect(),
                }).collect())
            }
        }
    }
}

impl From<exec::GgmlDtype> for wit::GgmlDtype {
    fn from(d: exec::GgmlDtype) -> Self {
        match d {
            exec::GgmlDtype::F32 => wit::GgmlDtype::F32,
            exec::GgmlDtype::F16 => wit::GgmlDtype::F16,
            exec::GgmlDtype::Bf16 => wit::GgmlDtype::Bf16,
            exec::GgmlDtype::Q4_0 => wit::GgmlDtype::Q40,
            exec::GgmlDtype::Q4_1 => wit::GgmlDtype::Q41,
            exec::GgmlDtype::Q5_0 => wit::GgmlDtype::Q50,
            exec::GgmlDtype::Q5_1 => wit::GgmlDtype::Q51,
            exec::GgmlDtype::Q8_0 => wit::GgmlDtype::Q80,
            exec::GgmlDtype::Q8_1 => wit::GgmlDtype::Q81,
            exec::GgmlDtype::Q2K => wit::GgmlDtype::Q2k,
            exec::GgmlDtype::Q3K => wit::GgmlDtype::Q3k,
            exec::GgmlDtype::Q4K => wit::GgmlDtype::Q4k,
            exec::GgmlDtype::Q5K => wit::GgmlDtype::Q5k,
            exec::GgmlDtype::Q6K => wit::GgmlDtype::Q6k,
            exec::GgmlDtype::Q8K => wit::GgmlDtype::Q8k,
        }
    }
}

impl From<exec::GgufTensor> for wit::GgufTensor {
    fn from(t: exec::GgufTensor) -> Self {
        wit::GgufTensor {
            name: t.name,
            shape: t.shape,
            dtype: t.dtype.into(),
        }
    }
}

fn metadata_value_to_json(v: &exec::MetadataValue) -> String {
    match v {
        exec::MetadataValue::U8(x) => x.to_string(),
        exec::MetadataValue::I8(x) => x.to_string(),
        exec::MetadataValue::U16(x) => x.to_string(),
        exec::MetadataValue::I16(x) => x.to_string(),
        exec::MetadataValue::U32(x) => x.to_string(),
        exec::MetadataValue::I32(x) => x.to_string(),
        exec::MetadataValue::U64(x) => x.to_string(),
        exec::MetadataValue::I64(x) => x.to_string(),
        exec::MetadataValue::F32(x) => x.to_string(),
        exec::MetadataValue::F64(x) => x.to_string(),
        exec::MetadataValue::Bool(x) => x.to_string(),
        exec::MetadataValue::String(x) => {
            serde_json::to_string(x).unwrap_or_else(|_| format!("\"{}\"", x))
        }
        exec::MetadataValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(metadata_value_to_json).collect();
            format!("[{}]", items.join(","))
        }
    }
}

impl From<exec::MetadataValue> for wit::MetadataValue {
    fn from(v: exec::MetadataValue) -> Self {
        match v {
            exec::MetadataValue::U8(x) => wit::MetadataValue::U8(x),
            exec::MetadataValue::I8(x) => wit::MetadataValue::S8(x),
            exec::MetadataValue::U16(x) => wit::MetadataValue::U16(x),
            exec::MetadataValue::I16(x) => wit::MetadataValue::S16(x),
            exec::MetadataValue::U32(x) => wit::MetadataValue::U32(x),
            exec::MetadataValue::I32(x) => wit::MetadataValue::S32(x),
            exec::MetadataValue::U64(x) => wit::MetadataValue::U64(x),
            exec::MetadataValue::I64(x) => wit::MetadataValue::S64(x),
            exec::MetadataValue::F32(x) => wit::MetadataValue::F32(x),
            exec::MetadataValue::F64(x) => wit::MetadataValue::F64(x),
            exec::MetadataValue::Bool(x) => wit::MetadataValue::Bool(x),
            exec::MetadataValue::String(x) => wit::MetadataValue::String(x),
            exec::MetadataValue::Array(ref arr) => {
                let items: Vec<String> = arr.iter().map(metadata_value_to_json).collect();
                wit::MetadataValue::Array(format!("[{}]", items.join(",")))
            }
        }
    }
}

impl From<exec::ModelDescription> for wit::ModelDescription {
    fn from(d: exec::ModelDescription) -> Self {
        wit::ModelDescription {
            n_experts: d.n_experts,
            n_layers: d.n_layers,
            n_total_parameters: d.n_total_parameters,
            n_active_parameters: d.n_active_parameters,
            metadata: d.metadata.into_iter().map(|(k, v)| (k, v.into())).collect(),
            tensors: d.tensors.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<exec::Error> for wit::Error {
    fn from(e: exec::Error) -> Self {
        match e {
            exec::Error::NotAvailable(s) => wit::Error::NotAvailable(s),
            exec::Error::TrainError(s) => wit::Error::TrainError(s),
            exec::Error::CheckpointError(s) => wit::Error::CheckpointError(s),
            exec::Error::InvalidState(s) => wit::Error::InvalidState(s),
            exec::Error::ModelError(s) => wit::Error::ModelError(s),
            exec::Error::TokenizeError(s) => wit::Error::TokenizeError(s),
            exec::Error::SnapshotError(s) => wit::Error::SnapshotError(s),
        }
    }
}

// ---------------------------------------------------------------------------
// WIT → Executor
// ---------------------------------------------------------------------------

impl From<wit::LogitEntry> for exec::LogitEntry {
    fn from(e: wit::LogitEntry) -> Self {
        exec::LogitEntry {
            token_id: e.token_id,
            log_prob: e.log_prob,
        }
    }
}

impl From<wit::Predicted> for exec::Predicted {
    fn from(p: wit::Predicted) -> Self {
        exec::Predicted {
            token_id: p.token_id,
            text: p.text,
            top_k: p.top_k.into_iter().map(Into::into).collect(),
            tail: p.tail.into_iter().map(Into::into).collect(),
            tail_mass: p.tail_mass,
            expert_indices: p.expert_indices,
        }
    }
}

impl From<wit::Model> for exec::Model {
    fn from(m: wit::Model) -> Self {
        exec::Model {
            id: m.id,
            n_experts: m.n_experts,
            n_layers: m.n_layers,
        }
    }
}

impl From<wit::Checkpoint> for exec::Checkpoint {
    fn from(c: wit::Checkpoint) -> Self {
        exec::Checkpoint { id: c.id }
    }
}

impl From<wit::Snapshot> for exec::Snapshot {
    fn from(s: wit::Snapshot) -> Self {
        exec::Snapshot {
            id: s.id,
            state_position: s.state_position,
            num_tokens: s.num_tokens,
        }
    }
}

impl From<wit::PersistedSnapshot> for exec::PersistedSnapshot {
    fn from(p: wit::PersistedSnapshot) -> Self {
        exec::PersistedSnapshot { id: p.id }
    }
}

impl From<wit::Weights> for exec::Weights {
    fn from(w: wit::Weights) -> Self {
        match w {
            wit::Weights::HostDefault => exec::Weights::HostDefault,
            wit::Weights::Checkpoint(c) => exec::Weights::Checkpoint(c.into()),
            wit::Weights::Path(p) => exec::Weights::Path(p),
        }
    }
}

impl From<wit::ModelInput> for exec::ModelInput {
    fn from(i: wit::ModelInput) -> Self {
        match i {
            wit::ModelInput::Text(s) => exec::ModelInput::Text(s),
            wit::ModelInput::Tokens(ids) => exec::ModelInput::Tokens(ids),
            wit::ModelInput::Soft(tokens) => {
                exec::ModelInput::Soft(tokens.into_iter().map(|t| exec::SoftToken {
                    predicted: t.predicted,
                    dark_knowledge: t.dark_knowledge.into_iter().map(Into::into).collect(),
                }).collect())
            }
        }
    }
}

impl From<wit::TrainingData> for exec::TrainingData {
    fn from(d: wit::TrainingData) -> Self {
        match d {
            wit::TrainingData::Context(input) => exec::TrainingData::Context(input.into()),
            wit::TrainingData::Target(preds) => {
                exec::TrainingData::Target(preds.into_iter().map(Into::into).collect())
            }
        }
    }
}

impl From<wit::TrainingSample> for exec::TrainingSample {
    fn from(s: wit::TrainingSample) -> Self {
        exec::TrainingSample {
            id: s.id,
            data: s.data.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<wit::ErrorFeedbackParams> for exec::ErrorFeedbackParams {
    fn from(p: wit::ErrorFeedbackParams) -> Self {
        exec::ErrorFeedbackParams {
            decay: p.decay,
            gain: p.gain,
        }
    }
}

impl From<wit::ReplayParams> for exec::ReplayParams {
    fn from(r: wit::ReplayParams) -> Self {
        exec::ReplayParams {
            steps: r.steps,
            decay: r.decay,
            gain: r.gain,
        }
    }
}

impl From<wit::ErrorFeedbackMode> for exec::ErrorFeedbackMode {
    fn from(m: wit::ErrorFeedbackMode) -> Self {
        match m {
            wit::ErrorFeedbackMode::None => exec::ErrorFeedbackMode::None,
            wit::ErrorFeedbackMode::Persistent(p) => exec::ErrorFeedbackMode::Persistent(p.into()),
            wit::ErrorFeedbackMode::Replay(r) => exec::ErrorFeedbackMode::Replay(r.into()),
        }
    }
}

impl From<wit::EpsilonComponent> for exec::EpsilonComponent {
    fn from(c: wit::EpsilonComponent) -> Self {
        match c {
            wit::EpsilonComponent::Embedding => exec::EpsilonComponent::Embedding,
            wit::EpsilonComponent::Attention => exec::EpsilonComponent::Attention,
            wit::EpsilonComponent::MoeGating => exec::EpsilonComponent::MoeGating,
            wit::EpsilonComponent::MoeExperts => exec::EpsilonComponent::MoeExperts,
            wit::EpsilonComponent::MoeExpertBanks => exec::EpsilonComponent::MoeExpertBanks,
            wit::EpsilonComponent::Mtp => exec::EpsilonComponent::Mtp,
            wit::EpsilonComponent::Norms => exec::EpsilonComponent::Norms,
            wit::EpsilonComponent::Ssm => exec::EpsilonComponent::Ssm,
            wit::EpsilonComponent::Other => exec::EpsilonComponent::Other,
        }
    }
}

impl From<wit::HyperParameterUpdate> for exec::HyperParameterUpdate {
    fn from(u: wit::HyperParameterUpdate) -> Self {
        exec::HyperParameterUpdate {
            mtp_decay: u.mtp_decay,
            num_speculative: u.num_speculative,
            temperature: u.temperature,
            lb_loss: u.lb_loss,
            z_loss: u.z_loss,
            clip_threshold: u.clip_threshold,
            lr: u.lr,
            epsilon: u.epsilon,
            epsilon_multipliers: u
                .epsilon_multipliers
                .map(|v| v.into_iter().map(|(c, val)| (c.into(), val)).collect()),
            error_feedback: u.error_feedback.map(Into::into),
        }
    }
}

impl From<wit::Experts> for exec::Experts {
    fn from(e: wit::Experts) -> Self {
        exec::Experts { indices: e.indices }
    }
}

impl From<wit::ExpertContributions> for exec::ExpertContributions {
    fn from(c: wit::ExpertContributions) -> Self {
        exec::ExpertContributions { weights: c.weights }
    }
}

impl From<wit::ExpertMergeSpec> for exec::ExpertMergeSpec {
    fn from(s: wit::ExpertMergeSpec) -> Self {
        exec::ExpertMergeSpec {
            retained: s.retained.into(),
            contributions: s.contributions.into(),
        }
    }
}

impl From<wit::ExpertPruningRequest> for exec::ExpertPruningRequest {
    fn from(r: wit::ExpertPruningRequest) -> Self {
        match r {
            wit::ExpertPruningRequest::Naive(e) => exec::ExpertPruningRequest::Naive(e.into()),
            wit::ExpertPruningRequest::ShapleyMerge(s) => {
                exec::ExpertPruningRequest::ShapleyMerge(s.into())
            }
        }
    }
}

impl From<wit::Layers> for exec::Layers {
    fn from(l: wit::Layers) -> Self {
        exec::Layers { indices: l.indices }
    }
}

impl From<wit::LayerRef> for exec::LayerRef {
    fn from(r: wit::LayerRef) -> Self {
        exec::LayerRef {
            source: r.source.into(),
            layer_idx: r.layer_idx,
        }
    }
}

impl From<wit::PositiveModel> for exec::PositiveModel {
    fn from(p: wit::PositiveModel) -> Self {
        exec::PositiveModel {
            model: p.model.into(),
        }
    }
}

impl From<wit::NegativeModel> for exec::NegativeModel {
    fn from(n: wit::NegativeModel) -> Self {
        exec::NegativeModel {
            model: n.model.into(),
        }
    }
}

impl From<wit::MetadataValue> for exec::MetadataValue {
    fn from(v: wit::MetadataValue) -> Self {
        match v {
            wit::MetadataValue::U8(x) => exec::MetadataValue::U8(x),
            wit::MetadataValue::S8(x) => exec::MetadataValue::I8(x),
            wit::MetadataValue::U16(x) => exec::MetadataValue::U16(x),
            wit::MetadataValue::S16(x) => exec::MetadataValue::I16(x),
            wit::MetadataValue::U32(x) => exec::MetadataValue::U32(x),
            wit::MetadataValue::S32(x) => exec::MetadataValue::I32(x),
            wit::MetadataValue::U64(x) => exec::MetadataValue::U64(x),
            wit::MetadataValue::S64(x) => exec::MetadataValue::I64(x),
            wit::MetadataValue::F32(x) => exec::MetadataValue::F32(x),
            wit::MetadataValue::F64(x) => exec::MetadataValue::F64(x),
            wit::MetadataValue::Bool(x) => exec::MetadataValue::Bool(x),
            wit::MetadataValue::String(x) => exec::MetadataValue::String(x),
            wit::MetadataValue::Array(s) => exec::MetadataValue::String(s),
        }
    }
}

impl From<wit::QuantConflictStrategy> for exec::QuantConflictStrategy {
    fn from(s: wit::QuantConflictStrategy) -> Self {
        match s {
            wit::QuantConflictStrategy::Reject => exec::QuantConflictStrategy::Reject,
            wit::QuantConflictStrategy::Highest => exec::QuantConflictStrategy::Highest,
            wit::QuantConflictStrategy::Lowest => exec::QuantConflictStrategy::Lowest,
        }
    }
}

impl From<wit::FusionMember> for exec::FusionMember {
    fn from(m: wit::FusionMember) -> Self {
        exec::FusionMember {
            weights: m.weights.into(),
            contribution: m.contribution,
        }
    }
}

impl From<wit::ModelComposite> for exec::ModelComposite {
    fn from(c: wit::ModelComposite) -> Self {
        exec::ModelComposite {
            embedding: c.embedding.into(),
            layers: c.layers.into_iter().map(Into::into).collect(),
            lm_head: c.lm_head.map(Into::into),
            mtp_head: c.mtp_head.map(Into::into),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience conversion helper
// ---------------------------------------------------------------------------

/// Convert an executor result to a WIT result using the unified Error type.
pub fn to_wit<T, W>(result: Result<T, exec::Error>) -> Result<W, wit::Error>
where
    T: Into<W>,
{
    result.map(Into::into).map_err(Into::into)
}
