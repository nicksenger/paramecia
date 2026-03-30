use paramecia_core::Tensor;
use paramecia_tensor::glowstick::Shape;
use paramecia_tensor::Tensor as TypedTensor;

use super::kv_cache::LayerSnapshot;

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct AttentionTensors<QS, KS, VS, GS>
where
    QS: Shape,
    KS: Shape,
    VS: Shape,
    GS: Shape,
{
    pub(super) q: TypedTensor<QS>,
    pub(super) k: TypedTensor<KS>,
    pub(super) v: TypedTensor<VS>,
    pub(super) gate: TypedTensor<GS>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct MoeRouting<LogitsS, IndicesS, WeightsS>
where
    LogitsS: Shape,
    IndicesS: Shape,
    WeightsS: Shape,
{
    pub(super) logits: TypedTensor<LogitsS>,
    pub(super) indices: TypedTensor<IndicesS>,
    pub(super) weights: TypedTensor<WeightsS>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct LinearRecurrentState<StateS, ConvS, GateS>
where
    StateS: Shape,
    ConvS: Shape,
    GateS: Shape,
{
    pub(super) ssm_state: TypedTensor<StateS>,
    pub(super) conv_state: TypedTensor<ConvS>,
    pub(super) gate_cumsum_offset: Option<TypedTensor<GateS>>,
}

/// Result of a speculative decoding step.
#[derive(Debug)]
pub struct SpeculativeResult {
    /// Main model's predicted token
    pub main_token: Tensor,
    /// Main model's logits for the predicted token
    pub main_logits: Tensor,
    /// Speculative tokens from MTP (may be empty if MTP not available)
    pub spec_tokens: Vec<Tensor>,
    /// Logits for each speculative token
    pub spec_logits: Vec<Tensor>,
    /// Layer snapshots for rollback (from snapshot_cache)
    pub snapshots: Vec<LayerSnapshot>,
}

/// Result of verification step.
#[derive(Debug)]
pub struct VerificationResult {
    /// Number of draft tokens that were accepted
    pub num_accepted: usize,
    /// Logits for the next token after accepted prefix (None if all accepted)
    pub next_logits: Option<Tensor>,
}
