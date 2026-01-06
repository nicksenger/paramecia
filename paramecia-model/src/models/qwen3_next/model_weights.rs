use crate::quantized_nn::RmsNorm;
use inception::{primitive, Inception};
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator, CombinatorTraceExt, If, LiftResult, MapErr, SwitchRef, Then, TryFoldRange,
    WrapOk,
};
use paramecia_core::quantized::{gguf_file, QTensor, SharedQTensor};
use paramecia_core::{DType, Device, Result, Tensor, D};
use paramecia_nn::{Activation, Embedding, Module};
use paramecia_tensor::glowstick::{num::U1, Shape1, Shape2, Shape3};
use paramecia_tensor::{
    contiguous::ContiguousOp, narrow_dyn_start::NarrowDynStartOp, qmatmul_op::QMatMulOp,
    rms_norm::RmsNormOp, squeeze::SqueezeOp, to_device::ToDeviceOp, to_dtype::ToDtypeOp,
    Error as TensorError,
};
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::path::Path;
use std::sync::Arc;
use tracing::{trace, warn};

use super::config::{DeviceOffloadMode, KvCacheQuantization, LayerDeviceMap};
use super::expert_cache::{should_cache_experts, ExpertCache};
use super::full_attention::FullAttention;
use super::gguf_loader::Gguf;
use super::kv_cache::{
    LayerSnapshot, PreallocatedKvCache, PreallocatedQuantizedKvCache, PrefixCache, RecurrentState,
};
use super::linear_attention::LinearAttention;
use super::moe::{
    default_cache_capacity, ExpertWeightTensor, MoeBlock, MoeDispatchPrep, MoeExperts,
    MoeGroupAssignments, MoePrefetchGraph, MoeRouteRemap, SharedExpert,
};
use super::mtp::MtpHead;
use super::rope::{RotaryEmbedding, YarnConfig};
use super::shape::{Hidden3, TopIndices2, B, E, N, S, V};
use super::types::{SpeculativeResult, VerificationResult};
use super::utils::{log_shape, log_typed_qmatmul_shape};
use super::{
    is_gpu_device, layer_mask, transfer_to, AttentionLayer, LayerAttentionDispatchGraph,
    LayerForwardGraph, LayerForwardMode, LayerMoeForwardGraph, LayerWeights, TypedTensor,
};

type TQMatMul<S> = paramecia_tensor::QMatMul<S>;
type LmHeadShape = Shape2<V, S>;
type TExpertVec = paramecia_tensor::Tensor<Shape1<E>>;
type TTopIndices2 = paramecia_tensor::Tensor<TopIndices2>;
type LayerRunState = (TypedTensor<Hidden3>, Option<Vec<(Tensor, Tensor)>>);
type HiddenLast3 = Shape3<B, U1, S>;
type LogitsLast3 = Shape3<B, U1, V>;
type THiddenLast = TypedTensor<HiddenLast3>;
type TLogitsLast = TypedTensor<LogitsLast3>;
type TLogits2 = TypedTensor<Shape2<B, V>>;
type TLogits3 = TypedTensor<Shape3<B, N, V>>;

#[cfg(feature = "qwen3next_80b_a3b")]
const MODEL_NODE_LABEL: &str = "Qwen3-Next-80B-A3B";
#[cfg(feature = "qwen35moe_35b_a3b")]
const MODEL_NODE_LABEL: &str = "Qwen3.5-35B-A3B";
#[cfg(feature = "qwen35moe_122b_a10b")]
const MODEL_NODE_LABEL: &str = "Qwen3.5-122B-A10B";
#[cfg(feature = "qwen35moe_397b_a17b")]
const MODEL_NODE_LABEL: &str = "Qwen3.5-397B-A17B";
#[cfg(feature = "qwen35_0p8b")]
const MODEL_NODE_LABEL: &str = "Qwen3.5-0.8B";
#[cfg(feature = "qwen35_4b")]
const MODEL_NODE_LABEL: &str = "Qwen3.5-4B";
#[cfg(feature = "qwen35_9b")]
const MODEL_NODE_LABEL: &str = "Qwen3.5-9B";
#[cfg(feature = "qwen35_27b")]
const MODEL_NODE_LABEL: &str = "Qwen3.5-27B";
#[cfg(not(any(
    feature = "qwen3next_80b_a3b",
    feature = "qwen35moe_35b_a3b",
    feature = "qwen35moe_122b_a10b",
    feature = "qwen35moe_397b_a17b",
    feature = "qwen35_0p8b",
    feature = "qwen35_4b",
    feature = "qwen35_9b",
    feature = "qwen35_27b"
)))]
const MODEL_NODE_LABEL: &str = "Qwen3.5-0.8B";

// ============================================================================
// Model
// ============================================================================

#[derive(Debug, Clone)]
pub(super) struct ModelConfig {
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) head_dim: usize,
    pub(super) num_layers: usize,
    pub(super) hidden_size: usize,
    pub(super) max_position_embeddings: usize,
    pub(super) rms_norm_eps: f64,
    pub(super) rope_freq_base: f64,
    /// Number of dimensions to rotate with RoPE (partial rotary embedding).
    /// For Qwen3-Next this is typically head_dim / 4 = 64 (out of 256).
    /// Only the first n_rot dimensions are rotated, the rest pass through unchanged.
    pub(super) n_rot: usize,
    /// Whether metadata indicates interleaved mRoPE frequency layout.
    /// This is not a switch to adjacent-pair rotation; Qwen3.5 still uses
    /// half-split RoPE rotation semantics.
    pub(super) rope_interleaved: bool,
    pub(super) num_experts: usize,
    pub(super) num_experts_per_tok: usize,
    pub(super) ssm_d_inner: usize,
    pub(super) ssm_d_state: usize,
    pub(super) ssm_n_groups: usize,
    pub(super) ssm_dt_rank: usize,
    /// Whether linear-attention V-related tensors use tiled head order in GGUF.
    /// Qwen3.5 conversion uses tiled order and requires matching Q/K head expansion.
    pub(super) linear_v_heads_tiled_order: bool,
    pub(super) recurrent_layers: Vec<bool>,
    pub(super) dtype: DType,
    /// YARN configuration for extended context. None means no YARN (use standard RoPE).
    pub(super) yarn_config: Option<YarnConfig>,
}

#[derive(Debug)]
pub(super) struct DecodeWorkspace {
    is_multi: bool,
    layer_devices: Vec<Device>,
}

#[derive(Debug)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: TQMatMul<LmHeadShape>,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
    /// Layer-to-device mapping for multi-GPU layer parallelism.
    layer_device_map: LayerDeviceMap,
    /// Prefetch pipeline coordinator for hiding transfer latency
    prefetch_pipeline: Option<crate::layer_pipeline::PrefetchPipelineCoordinator>,
    /// SharedQTensors for QuZO training (empty for inference mode)
    shared_tensors: Vec<(String, SharedQTensor)>,
    /// Expert indices from the last forward pass (for tuning output).
    /// Each tensor has shape [batch*seq, num_experts_per_tok] with u32 expert IDs.
    /// Only populated when `set_capture_expert_indices(true)` is called.
    last_expert_indices: Option<Vec<Tensor>>,
    /// Whether to capture expert indices during forward passes.
    capture_expert_indices: bool,
    /// MTP (Multi-Token Prediction) head for speculative decoding.
    /// None if the model doesn't include MTP weights.
    mtp_head: Option<MtpHead>,
    /// Cached decode-only routing/workspace metadata (L=1 path).
    decode_workspace: Option<DecodeWorkspace>,
}

struct LayerRunCtx<'a> {
    layer: &'a mut LayerWeights,
    layer_idx: usize,
    is_multi: bool,
    layer_device_map: &'a LayerDeviceMap,
    per_device_masks: &'a [(String, Tensor)],
    mask: Option<&'a Tensor>,
    offset: usize,
    mode: LayerForwardMode,
}

struct LayerLoopCtx<'a> {
    layers: &'a mut [LayerWeights],
    is_multi: bool,
    layer_device_map: &'a LayerDeviceMap,
    per_device_masks: &'a [(String, Tensor)],
    mask: Option<&'a Tensor>,
    offset: usize,
    mode: LayerForwardMode,
}
struct DecodeLayerStepCtx<'a> {
    layers: &'a mut [LayerWeights],
    is_multi: bool,
    layer_devices: &'a [Device],
    offset: usize,
}

type LayerRunTransferResult = std::result::Result<TypedTensor<Hidden3>, paramecia_core::Error>;
type LayerRunForwardLift = LiftResult<
    LayerRunForwardOp,
    LayerRunTransferResult,
    (TypedTensor<Hidden3>, Option<(Tensor, Tensor)>),
>;

#[derive(Debug, Default, Clone, Copy)]
struct LayerRunTransferOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<LayerRunCtx<'a>> for LayerRunTransferOp {
    type In = TypedTensor<Hidden3>;
    type Out = Result<TypedTensor<Hidden3>>;

    fn forward(&mut self, ctx: &mut LayerRunCtx<'a>, input: Self::In) -> Self::Out {
        let hidden = if ctx.is_multi {
            transfer_to(
                input.into_inner(),
                ctx.layer_device_map.device_for_layer(ctx.layer_idx),
            )?
        } else {
            input.into_inner()
        };
        Ok(hidden.try_into()?)
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerRunTransferOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("LayerRunTransfer").with_output_type::<Result<TypedTensor<Hidden3>>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerRunForwardOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<LayerRunCtx<'a>> for LayerRunForwardOp {
    type In = TypedTensor<Hidden3>;
    type Out = Result<(TypedTensor<Hidden3>, Option<(Tensor, Tensor)>)>;

    fn forward(&mut self, ctx: &mut LayerRunCtx<'a>, input: Self::In) -> Self::Out {
        let mask_ref = layer_mask(
            ctx.layer_idx,
            ctx.is_multi,
            ctx.layer_device_map,
            ctx.per_device_masks,
            ctx.mask,
        );
        ctx.layer
            .forward_inner_typed(input, mask_ref, ctx.offset, ctx.mode)
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerRunForwardOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph("LayerRunForward", <LayerForwardGraph as Vis>::visualize())
            .with_output_type::<Result<(TypedTensor<Hidden3>, Option<(Tensor, Tensor)>)>>()
    }
}

type LayerRunOp = Then<LayerRunTransferOp, LayerRunForwardLift>;
fn layer_run_op() -> LayerRunOp {
    Then::new(LayerRunTransferOp, LiftResult::new(LayerRunForwardOp))
}

type LayerLoopInvokeResult =
    std::result::Result<(LayerRunState, Option<(Tensor, Tensor)>), paramecia_core::Error>;
type LayerLoopCollectLift =
    LiftResult<LayerLoopCollectStatsOp, LayerLoopInvokeResult, LayerRunState>;

fn layer_loop_collect_stats(
    input: (LayerRunState, Option<(Tensor, Tensor)>),
) -> Result<LayerRunState> {
    let ((output, mut all_stats), stats) = input;
    if let (Some(ref mut all), Some(s)) = (&mut all_stats, stats) {
        all.push(s);
    }
    Ok((output, all_stats))
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerLoopInvokeRunOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<LayerLoopCtx<'a>> for LayerLoopInvokeRunOp {
    type In = (LayerRunState, usize);
    type Out = Result<(LayerRunState, Option<(Tensor, Tensor)>)>;

    fn forward(&mut self, ctx: &mut LayerLoopCtx<'a>, input: Self::In) -> Self::Out {
        let ((h_typed, all_stats), layer_idx) = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;
        let mut run_ctx = LayerRunCtx {
            layer,
            layer_idx,
            is_multi: ctx.is_multi,
            layer_device_map: ctx.layer_device_map,
            per_device_masks: ctx.per_device_masks,
            mask: ctx.mask,
            offset: ctx.offset,
            mode: ctx.mode,
        };
        let mut step = layer_run_op();
        let (output, stats) = step.traced_forward(&mut run_ctx, h_typed)?;
        Ok(((output, all_stats), stats))
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerLoopInvokeRunOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph("LayerLoopInvokeRun", <LayerRunOp as Vis>::visualize())
            .with_output_type::<Result<(LayerRunState, Option<(Tensor, Tensor)>)>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerLoopCollectStatsOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for LayerLoopCollectStatsOp {
    type In = (LayerRunState, Option<(Tensor, Tensor)>);
    type Out = Result<LayerRunState>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        layer_loop_collect_stats(input)
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerLoopCollectStatsOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("LayerLoopCollectStats").with_output_type::<Result<LayerRunState>>()
    }
}

type LayerLoopStepOp = Then<LayerLoopInvokeRunOp, LayerLoopCollectLift>;
fn layer_loop_step_op() -> LayerLoopStepOp {
    Then::new(
        LayerLoopInvokeRunOp,
        LiftResult::new(LayerLoopCollectStatsOp),
    )
}

type DecodeLayerTransferResult =
    std::result::Result<(TypedTensor<Hidden3>, usize), paramecia_core::Error>;
type DecodeLayerForwardLift =
    LiftResult<DecodeLayerForwardOp, DecodeLayerTransferResult, TypedTensor<Hidden3>>;

#[derive(Debug, Default, Clone, Copy)]
struct DecodeLayerTransferOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<DecodeLayerStepCtx<'a>> for DecodeLayerTransferOp {
    type In = (TypedTensor<Hidden3>, usize);
    type Out = Result<(TypedTensor<Hidden3>, usize)>;

    fn forward(&mut self, ctx: &mut DecodeLayerStepCtx<'a>, input: Self::In) -> Self::Out {
        let (h, layer_idx) = input;
        let h = if ctx.is_multi {
            let target = ctx.layer_devices.get(layer_idx).ok_or_else(|| {
                paramecia_core::Error::Msg(format!(
                    "missing decode layer device {} (len {})",
                    layer_idx,
                    ctx.layer_devices.len()
                ))
            })?;
            transfer_to(h.into_inner(), target)?.try_into()?
        } else {
            h
        };
        Ok((h, layer_idx))
    }
}
#[primitive(property = Visualize)]
impl Vis for DecodeLayerTransferOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("DecodeLayerTransfer")
            .with_output_type::<Result<(TypedTensor<Hidden3>, usize)>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct DecodeLayerForwardOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<DecodeLayerStepCtx<'a>> for DecodeLayerForwardOp {
    type In = (TypedTensor<Hidden3>, usize);
    type Out = Result<TypedTensor<Hidden3>>;

    fn forward(&mut self, ctx: &mut DecodeLayerStepCtx<'a>, input: Self::In) -> Self::Out {
        let (h, layer_idx) = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "decode layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;
        layer.forward_typed(h, None, ctx.offset)
    }
}
#[primitive(property = Visualize)]
impl Vis for DecodeLayerForwardOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "DecodeLayerForward",
            <LayerForwardGraph as Vis>::visualize(),
        )
        .with_output_type::<Result<TypedTensor<Hidden3>>>()
    }
}

type DecodeLayerStepOp = Then<DecodeLayerTransferOp, DecodeLayerForwardLift>;
fn decode_layer_step_op() -> DecodeLayerStepOp {
    Then::new(DecodeLayerTransferOp, LiftResult::new(DecodeLayerForwardOp))
}

struct TrainPipelineState {
    h: TypedTensor<Hidden3>,
    pending_moe: Option<usize>,
    pending_residual: Option<TypedTensor<Hidden3>>,
    all_router_stats: Vec<(Tensor, Tensor)>,
}

struct PrefetchPipelineState {
    h: TypedTensor<Hidden3>,
    pending_moe: Option<usize>,
    pending_residual: Option<TypedTensor<Hidden3>>,
    expert_indices: Vec<Tensor>,
}

struct TrainPipelineStepCtx<'a> {
    layers: &'a mut [LayerWeights],
    pipeline: &'a crate::layer_pipeline::PrefetchPipelineCoordinator,
    is_multi: bool,
    layer_device_map: &'a LayerDeviceMap,
    device_masks: &'a [(String, Tensor)],
    causal_mask: Option<&'a Tensor>,
    offset: usize,
    activation_dtype: DType,
    num_layers: usize,
}
struct PrefetchPipelineStepCtx<'a> {
    layers: &'a mut [LayerWeights],
    pipeline: &'a crate::layer_pipeline::PrefetchPipelineCoordinator,
    is_multi: bool,
    layer_device_map: &'a LayerDeviceMap,
    device_masks: &'a [(String, Tensor)],
    causal_mask: Option<&'a Tensor>,
    offset: usize,
    activation_dtype: DType,
    num_layers: usize,
    capture_experts: bool,
}
struct TrainStepResolved {
    state: TrainPipelineState,
    layer_idx: usize,
    is_last: bool,
}

struct TrainBranchInput {
    state: TrainPipelineState,
    layer_idx: usize,
    h_after_attn: TypedTensor<Hidden3>,
    ffn_residual: TypedTensor<Hidden3>,
    h_ffn_normed_typed: TypedTensor<Hidden3>,
    is_last: bool,
}

struct TrainAttnNormed {
    state: TrainPipelineState,
    layer_idx: usize,
    is_last: bool,
    inp_sa: TypedTensor<Hidden3>,
    h_normed_typed: TypedTensor<Hidden3>,
}

struct TrainAttnForwarded {
    state: TrainPipelineState,
    layer_idx: usize,
    is_last: bool,
    inp_sa: TypedTensor<Hidden3>,
    attn_out_typed: TypedTensor<Hidden3>,
}

type TrainResolveResult = std::result::Result<TrainStepResolved, paramecia_core::Error>;
type TrainBranchInputResult = std::result::Result<TrainBranchInput, paramecia_core::Error>;

fn train_is_last_branch(input: &TrainBranchInput) -> usize {
    if input.is_last {
        0
    } else {
        1
    }
}

struct TrainLastMoeForwarded {
    state: TrainPipelineState,
    ffn_residual: TypedTensor<Hidden3>,
    moe_out_typed: TypedTensor<Hidden3>,
    stats: (Tensor, Tensor),
}

struct TrainNonLastPrefetched {
    state: TrainPipelineState,
    layer_idx: usize,
    h_after_attn: TypedTensor<Hidden3>,
    ffn_residual: TypedTensor<Hidden3>,
    h_ffn_normed_typed: TypedTensor<Hidden3>,
}

#[derive(Debug, Default, Clone, Copy)]
struct TrainResolveOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<TrainPipelineStepCtx<'a>> for TrainResolveOp {
    type In = (TrainPipelineState, usize);
    type Out = Result<TrainStepResolved>;

    fn forward(&mut self, ctx: &mut TrainPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let (mut state, layer_idx) = input;
        let is_last = layer_idx + 1 == ctx.num_layers;

        if ctx.is_multi {
            state.h = transfer_to(
                state.h.into_inner(),
                ctx.layer_device_map.device_for_layer(layer_idx),
            )?
            .try_into()?;
        }

        if let Some(prev_layer) = state.pending_moe.take() {
            let target_dev = ctx.layer_device_map.device_for_layer(layer_idx);
            let moe_out = ctx.pipeline.wait_for_result(prev_layer, target_dev)?;
            let moe_out = if moe_out.dtype() != ctx.activation_dtype {
                moe_out.to_dtype(ctx.activation_dtype)?
            } else {
                moe_out
            };
            let moe_out_typed: TypedTensor<Hidden3> = moe_out.try_into()?;
            let residual = state
                .pending_residual
                .take()
                .ok_or_else(|| paramecia_core::Error::Msg("pending_residual missing".into()))?;
            state.h = paramecia_tensor::residual_add!(&residual, &moe_out_typed)?;
        }

        Ok(TrainStepResolved {
            state,
            layer_idx,
            is_last,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for TrainResolveOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("TrainResolve").with_output_type::<Result<TrainStepResolved>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct TrainAttnNormOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<TrainPipelineStepCtx<'a>> for TrainAttnNormOp {
    type In = TrainStepResolved;
    type Out = Result<TrainAttnNormed>;

    fn forward(&mut self, ctx: &mut TrainPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let TrainStepResolved {
            state,
            layer_idx,
            is_last,
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "train pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let inp_sa = state.h.clone();
        let h_normed = layer.attn_norm.forward(state.h.inner())?;
        let h_normed = if h_normed.dtype() != ctx.activation_dtype {
            h_normed.to_dtype(ctx.activation_dtype)?
        } else {
            h_normed
        };
        log_shape("train_pipe.attn_norm", &h_normed);
        let h_normed_typed: TypedTensor<Hidden3> = h_normed.try_into()?;

        Ok(TrainAttnNormed {
            state,
            layer_idx,
            is_last,
            inp_sa,
            h_normed_typed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for TrainAttnNormOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("TrainAttnNorm").with_output_type::<Result<TrainAttnNormed>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct TrainAttnForwardOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<TrainPipelineStepCtx<'a>> for TrainAttnForwardOp {
    type In = TrainAttnNormed;
    type Out = Result<TrainAttnForwarded>;

    fn forward(&mut self, ctx: &mut TrainPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let TrainAttnNormed {
            state,
            layer_idx,
            is_last,
            inp_sa,
            h_normed_typed,
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "train pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;
        let mask_ref = layer_mask(
            layer_idx,
            ctx.is_multi,
            ctx.layer_device_map,
            ctx.device_masks,
            ctx.causal_mask,
        );

        let attn_out = match &mut layer.attn {
            AttentionLayer::Full(attn) => {
                attn.forward_typed(&h_normed_typed, mask_ref, ctx.offset)?
            }
            AttentionLayer::Linear(attn) => attn.forward_typed(&h_normed_typed)?,
        };
        log_shape("train_pipe.attn_out", attn_out.inner());
        let attn_out_typed: TypedTensor<Hidden3> =
            if attn_out.inner().dtype() != ctx.activation_dtype {
                attn_out
                    .inner()
                    .to_dtype(ctx.activation_dtype)?
                    .try_into()?
            } else {
                attn_out
            };

        Ok(TrainAttnForwarded {
            state,
            layer_idx,
            is_last,
            inp_sa,
            attn_out_typed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for TrainAttnForwardOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "TrainAttnForward",
            Graph::sequence(
                <LayerAttentionDispatchGraph as Vis>::visualize(),
                Graph::custom_leaf("CastToActivationDtypeIfNeeded"),
            ),
        )
        .with_output_type::<Result<TrainAttnForwarded>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct TrainFfnPrepOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<TrainPipelineStepCtx<'a>> for TrainFfnPrepOp {
    type In = TrainAttnForwarded;
    type Out = Result<TrainBranchInput>;

    fn forward(&mut self, ctx: &mut TrainPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let TrainAttnForwarded {
            state,
            layer_idx,
            is_last,
            inp_sa,
            attn_out_typed,
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "train pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let h_after_attn: TypedTensor<Hidden3> =
            paramecia_tensor::residual_add!(&attn_out_typed, &inp_sa)?;
        let ffn_residual = h_after_attn.clone();
        let h_ffn_normed = layer.ffn_norm.forward(h_after_attn.inner())?;
        let h_ffn_normed = if h_ffn_normed.dtype() != ctx.activation_dtype {
            h_ffn_normed.to_dtype(ctx.activation_dtype)?
        } else {
            h_ffn_normed
        };
        log_shape("train_pipe.ffn_norm", &h_ffn_normed);
        let h_ffn_normed_typed: TypedTensor<Hidden3> = h_ffn_normed.try_into()?;

        Ok(TrainBranchInput {
            state,
            layer_idx,
            h_after_attn,
            ffn_residual,
            h_ffn_normed_typed,
            is_last,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for TrainFfnPrepOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("TrainFfnPrep").with_output_type::<Result<TrainBranchInput>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct TrainLastForwardMoeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<TrainPipelineStepCtx<'a>> for TrainLastForwardMoeOp {
    type In = TrainBranchInput;
    type Out = Result<TrainLastMoeForwarded>;

    fn forward(&mut self, ctx: &mut TrainPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let TrainBranchInput {
            state,
            layer_idx,
            ffn_residual,
            h_ffn_normed_typed,
            ..
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "train pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let (moe_out_typed, stats) = layer
            .moe_block
            .forward_with_stats_typed(&h_ffn_normed_typed)?;
        Ok(TrainLastMoeForwarded {
            state,
            ffn_residual,
            moe_out_typed,
            stats,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for TrainLastForwardMoeOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "TrainLastForwardMoe",
            Graph::sequence(
                <LayerMoeForwardGraph as Vis>::visualize(),
                Graph::custom_leaf("CaptureRouterStats"),
            ),
        )
        .with_output_type::<Result<TrainLastMoeForwarded>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct TrainLastFinalizeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<TrainPipelineStepCtx<'a>> for TrainLastFinalizeOp {
    type In = TrainLastMoeForwarded;
    type Out = Result<TrainPipelineState>;

    fn forward(&mut self, ctx: &mut TrainPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let TrainLastMoeForwarded {
            mut state,
            ffn_residual,
            moe_out_typed,
            stats,
        } = input;
        state.all_router_stats.push(stats);
        let moe_out_typed: TypedTensor<Hidden3> =
            if moe_out_typed.inner().dtype() != ctx.activation_dtype {
                moe_out_typed
                    .inner()
                    .to_dtype(ctx.activation_dtype)?
                    .try_into()?
            } else {
                moe_out_typed
            };
        state.h = paramecia_tensor::residual_add!(&moe_out_typed, &ffn_residual)?;
        log_shape("train_pipe.layer_out_last", state.h.inner());
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl Vis for TrainLastFinalizeOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("TrainLastFinalize").with_output_type::<Result<TrainPipelineState>>()
    }
}

type TrainLastForwardResult = Result<TrainLastMoeForwarded>;
type TrainLastFinalizeLift =
    LiftResult<TrainLastFinalizeOp, TrainLastForwardResult, TrainPipelineState>;
type TrainRunLastOp = Then<TrainLastForwardMoeOp, TrainLastFinalizeLift>;

#[derive(Debug, Default, Clone, Copy)]
struct TrainNonLastPrefetchOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<TrainPipelineStepCtx<'a>> for TrainNonLastPrefetchOp {
    type In = TrainBranchInput;
    type Out = Result<TrainNonLastPrefetched>;

    fn forward(&mut self, ctx: &mut TrainPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let TrainBranchInput {
            mut state,
            layer_idx,
            h_after_attn,
            ffn_residual,
            h_ffn_normed_typed,
            ..
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "train pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let (routing_logits, top_weights_typed, remapped_indices_typed, _original_indices) =
            layer.moe_block.route_for_prefetch(&h_ffn_normed_typed)?;
        log_shape("train_pipe.routing_logits", &routing_logits);
        log_shape("train_pipe.routing_weights", top_weights_typed.inner());
        log_shape("train_pipe.routing_indices", remapped_indices_typed.inner());
        state
            .all_router_stats
            .push((routing_logits, remapped_indices_typed.inner().clone()));

        let (hidden_flat_typed, indices_flat_typed, weights_flat_typed, dispatch_dims) =
            layer.moe_block.prepare_dispatch_for_prefetch(
                h_ffn_normed_typed.clone(),
                remapped_indices_typed,
                top_weights_typed,
            )?;
        log_shape("train_pipe.hidden_flat", hidden_flat_typed.inner());
        ctx.pipeline.submit_prefetch(
            layer_idx,
            hidden_flat_typed.inner(),
            indices_flat_typed.inner(),
            weights_flat_typed.inner(),
            dispatch_dims,
        )?;

        Ok(TrainNonLastPrefetched {
            state,
            layer_idx,
            h_after_attn,
            ffn_residual,
            h_ffn_normed_typed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for TrainNonLastPrefetchOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "TrainNonLastPrefetch",
            Graph::sequence(
                <MoePrefetchGraph as Vis>::visualize(),
                Graph::custom_leaf("StorePendingMoeAndResidual"),
            ),
        )
        .with_output_type::<Result<TrainNonLastPrefetched>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct TrainNonLastFinalizeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<TrainPipelineStepCtx<'a>> for TrainNonLastFinalizeOp {
    type In = TrainNonLastPrefetched;
    type Out = Result<TrainPipelineState>;

    fn forward(&mut self, ctx: &mut TrainPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let TrainNonLastPrefetched {
            mut state,
            layer_idx,
            h_after_attn,
            ffn_residual,
            h_ffn_normed_typed,
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "train pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let adjusted_residual: TypedTensor<Hidden3> =
            if let Some(ref mut shared_exp) = layer.moe_block.shared_expert {
                let shared_out = shared_exp.forward(&h_ffn_normed_typed)?;
                log_shape("train_pipe.shared_expert_out", shared_out.inner());
                paramecia_tensor::residual_add!(&ffn_residual, &shared_out)?
            } else {
                ffn_residual
            };

        state.pending_moe = Some(layer_idx);
        state.pending_residual = Some(adjusted_residual);
        state.h = h_after_attn;
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl Vis for TrainNonLastFinalizeOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("TrainNonLastFinalize").with_output_type::<Result<TrainPipelineState>>()
    }
}

type TrainNonLastPrefetchResult = Result<TrainNonLastPrefetched>;
type TrainNonLastFinalizeLift =
    LiftResult<TrainNonLastFinalizeOp, TrainNonLastPrefetchResult, TrainPipelineState>;
type TrainRunNonLastOp = Then<TrainNonLastPrefetchOp, TrainNonLastFinalizeLift>;

type TrainAttnNormResult = Result<TrainAttnNormed>;
type TrainAttnForwardLift = LiftResult<TrainAttnForwardOp, TrainAttnNormResult, TrainAttnForwarded>;
type TrainAttnForwardResult = Result<TrainAttnForwarded>;
type TrainFfnPrepLift = LiftResult<TrainFfnPrepOp, TrainAttnForwardResult, TrainBranchInput>;
type TrainAttnFfnFlow = Then<TrainAttnNormOp, Then<TrainAttnForwardLift, TrainFfnPrepLift>>;
fn train_attn_ffn_flow() -> TrainAttnFfnFlow {
    Then::new(
        TrainAttnNormOp,
        Then::new(
            LiftResult::new(TrainAttnForwardOp),
            LiftResult::new(TrainFfnPrepOp),
        ),
    )
}
type TrainAttnFfnLiftFlow = LiftResult<TrainAttnFfnFlow, TrainResolveResult, TrainBranchInput>;
type TrainBranchSwitchFlow = SwitchRef<TrainBranchInput, TrainRunLastOp, TrainRunNonLastOp>;
type TrainBranchLiftFlow =
    LiftResult<TrainBranchSwitchFlow, TrainBranchInputResult, TrainPipelineState>;
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct TrainPipelineStep(TrainResolveOp, TrainAttnFfnLiftFlow, TrainBranchLiftFlow);
fn train_pipeline_step_op() -> TrainPipelineStep {
    TrainPipelineStep(
        TrainResolveOp,
        LiftResult::new(train_attn_ffn_flow()),
        LiftResult::new(SwitchRef::new(
            train_is_last_branch,
            Then::new(TrainLastForwardMoeOp, LiftResult::new(TrainLastFinalizeOp)),
            Then::new(
                TrainNonLastPrefetchOp,
                LiftResult::new(TrainNonLastFinalizeOp),
            ),
        )),
    )
}

struct PrefetchStepResolved {
    state: PrefetchPipelineState,
    layer_idx: usize,
    is_last: bool,
}

struct PrefetchBranchInput {
    state: PrefetchPipelineState,
    layer_idx: usize,
    h_after_attn: TypedTensor<Hidden3>,
    ffn_residual: TypedTensor<Hidden3>,
    h_ffn_normed_typed: TypedTensor<Hidden3>,
    is_last: bool,
}

struct PrefetchAttnNormed {
    state: PrefetchPipelineState,
    layer_idx: usize,
    is_last: bool,
    inp_sa: TypedTensor<Hidden3>,
    h_normed_typed: TypedTensor<Hidden3>,
}

struct PrefetchAttnForwarded {
    state: PrefetchPipelineState,
    layer_idx: usize,
    is_last: bool,
    inp_sa: TypedTensor<Hidden3>,
    attn_out_typed: TypedTensor<Hidden3>,
}

type PrefetchResolveResult = std::result::Result<PrefetchStepResolved, paramecia_core::Error>;
type PrefetchBranchInputResult = std::result::Result<PrefetchBranchInput, paramecia_core::Error>;

fn prefetch_is_last_branch(input: &PrefetchBranchInput) -> usize {
    if input.is_last {
        0
    } else {
        1
    }
}

struct PrefetchLastMoeForwarded {
    state: PrefetchPipelineState,
    ffn_residual: TypedTensor<Hidden3>,
    moe_out_typed: TypedTensor<Hidden3>,
    stats: (Tensor, Tensor),
}

struct PrefetchNonLastPrefetched {
    state: PrefetchPipelineState,
    layer_idx: usize,
    h_after_attn: TypedTensor<Hidden3>,
    ffn_residual: TypedTensor<Hidden3>,
    h_ffn_normed_typed: TypedTensor<Hidden3>,
}

#[derive(Debug, Default, Clone, Copy)]
struct PrefetchResolveOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchPipelineStepCtx<'a>> for PrefetchResolveOp {
    type In = (PrefetchPipelineState, usize);
    type Out = Result<PrefetchStepResolved>;

    fn forward(&mut self, ctx: &mut PrefetchPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let (mut state, layer_idx) = input;
        let is_last = layer_idx + 1 == ctx.num_layers;

        if ctx.is_multi {
            state.h = transfer_to(
                state.h.into_inner(),
                ctx.layer_device_map.device_for_layer(layer_idx),
            )?
            .try_into()?;
        }

        if let Some(prev_layer) = state.pending_moe.take() {
            let target_dev = ctx.layer_device_map.device_for_layer(layer_idx);
            let moe_out = ctx.pipeline.wait_for_result(prev_layer, target_dev)?;
            let moe_out = if moe_out.dtype() != ctx.activation_dtype {
                moe_out.to_dtype(ctx.activation_dtype)?
            } else {
                moe_out
            };
            let moe_out_typed: TypedTensor<Hidden3> = moe_out.try_into()?;
            let residual = state
                .pending_residual
                .take()
                .ok_or_else(|| paramecia_core::Error::Msg("pending_residual missing".into()))?;
            state.h = paramecia_tensor::residual_add!(&residual, &moe_out_typed)?;
        }

        Ok(PrefetchStepResolved {
            state,
            layer_idx,
            is_last,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchResolveOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("PrefetchResolve").with_output_type::<Result<PrefetchStepResolved>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct PrefetchAttnNormOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchPipelineStepCtx<'a>> for PrefetchAttnNormOp {
    type In = PrefetchStepResolved;
    type Out = Result<PrefetchAttnNormed>;

    fn forward(&mut self, ctx: &mut PrefetchPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let PrefetchStepResolved {
            state,
            layer_idx,
            is_last,
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "prefetch pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let inp_sa = state.h.clone();
        let h_normed = layer.attn_norm.forward(state.h.inner())?;
        let h_normed = if h_normed.dtype() != ctx.activation_dtype {
            h_normed.to_dtype(ctx.activation_dtype)?
        } else {
            h_normed
        };
        log_shape("prefetch.attn_norm", &h_normed);
        let h_normed_typed: TypedTensor<Hidden3> = h_normed.try_into()?;

        Ok(PrefetchAttnNormed {
            state,
            layer_idx,
            is_last,
            inp_sa,
            h_normed_typed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchAttnNormOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("PrefetchAttnNorm").with_output_type::<Result<PrefetchAttnNormed>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct PrefetchAttnForwardOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchPipelineStepCtx<'a>> for PrefetchAttnForwardOp {
    type In = PrefetchAttnNormed;
    type Out = Result<PrefetchAttnForwarded>;

    fn forward(&mut self, ctx: &mut PrefetchPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let PrefetchAttnNormed {
            state,
            layer_idx,
            is_last,
            inp_sa,
            h_normed_typed,
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "prefetch pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;
        let layer_mask_ref = layer_mask(
            layer_idx,
            ctx.is_multi,
            ctx.layer_device_map,
            ctx.device_masks,
            ctx.causal_mask,
        );

        let attn_out = match &mut layer.attn {
            AttentionLayer::Full(attn) => {
                attn.forward_typed(&h_normed_typed, layer_mask_ref, ctx.offset)?
            }
            AttentionLayer::Linear(attn) => attn.forward_typed(&h_normed_typed)?,
        };
        log_shape("prefetch.attn_out", attn_out.inner());
        let attn_out_typed: TypedTensor<Hidden3> =
            if attn_out.inner().dtype() != ctx.activation_dtype {
                attn_out
                    .inner()
                    .to_dtype(ctx.activation_dtype)?
                    .try_into()?
            } else {
                attn_out
            };

        Ok(PrefetchAttnForwarded {
            state,
            layer_idx,
            is_last,
            inp_sa,
            attn_out_typed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchAttnForwardOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "PrefetchAttnForward",
            Graph::sequence(
                <LayerAttentionDispatchGraph as Vis>::visualize(),
                Graph::custom_leaf("CastToActivationDtypeIfNeeded"),
            ),
        )
        .with_output_type::<Result<PrefetchAttnForwarded>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct PrefetchFfnPrepOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchPipelineStepCtx<'a>> for PrefetchFfnPrepOp {
    type In = PrefetchAttnForwarded;
    type Out = Result<PrefetchBranchInput>;

    fn forward(&mut self, ctx: &mut PrefetchPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let PrefetchAttnForwarded {
            state,
            layer_idx,
            is_last,
            inp_sa,
            attn_out_typed,
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "prefetch pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let h_after_attn: TypedTensor<Hidden3> =
            paramecia_tensor::residual_add!(&attn_out_typed, &inp_sa)?;
        log_shape("prefetch.h_after_attn", h_after_attn.inner());
        let ffn_residual = h_after_attn.clone();
        let h_ffn_normed = layer.ffn_norm.forward(h_after_attn.inner())?;
        let h_ffn_normed = if h_ffn_normed.dtype() != ctx.activation_dtype {
            h_ffn_normed.to_dtype(ctx.activation_dtype)?
        } else {
            h_ffn_normed
        };
        log_shape("prefetch.ffn_norm", &h_ffn_normed);
        let h_ffn_normed_typed: TypedTensor<Hidden3> = h_ffn_normed.try_into()?;

        Ok(PrefetchBranchInput {
            state,
            layer_idx,
            h_after_attn,
            ffn_residual,
            h_ffn_normed_typed,
            is_last,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchFfnPrepOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("PrefetchFfnPrep").with_output_type::<Result<PrefetchBranchInput>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct PrefetchLastForwardMoeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchPipelineStepCtx<'a>> for PrefetchLastForwardMoeOp {
    type In = PrefetchBranchInput;
    type Out = Result<PrefetchLastMoeForwarded>;

    fn forward(&mut self, ctx: &mut PrefetchPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let PrefetchBranchInput {
            state,
            layer_idx,
            ffn_residual,
            h_ffn_normed_typed,
            ..
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "prefetch pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let (moe_out_typed, stats) = layer
            .moe_block
            .forward_with_stats_typed(&h_ffn_normed_typed)?;
        Ok(PrefetchLastMoeForwarded {
            state,
            ffn_residual,
            moe_out_typed,
            stats,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchLastForwardMoeOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "PrefetchLastForwardMoe",
            Graph::sequence(
                <LayerMoeForwardGraph as Vis>::visualize(),
                Graph::custom_leaf("CaptureOrStoreRouterStats"),
            ),
        )
        .with_output_type::<Result<PrefetchLastMoeForwarded>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct PrefetchLastFinalizeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchPipelineStepCtx<'a>> for PrefetchLastFinalizeOp {
    type In = PrefetchLastMoeForwarded;
    type Out = Result<PrefetchPipelineState>;

    fn forward(&mut self, ctx: &mut PrefetchPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let PrefetchLastMoeForwarded {
            mut state,
            ffn_residual,
            moe_out_typed,
            stats,
        } = input;
        if ctx.capture_experts {
            let (_router_logits, indices) = stats;
            state
                .expert_indices
                .push(ModelWeights::normalize_captured_expert_indices(indices)?);
        }
        let moe_out_typed: TypedTensor<Hidden3> =
            if moe_out_typed.inner().dtype() != ctx.activation_dtype {
                moe_out_typed
                    .inner()
                    .to_dtype(ctx.activation_dtype)?
                    .try_into()?
            } else {
                moe_out_typed
            };
        state.h = paramecia_tensor::residual_add!(&moe_out_typed, &ffn_residual)?;
        log_shape("prefetch.layer_out_last", state.h.inner());
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchLastFinalizeOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("PrefetchLastFinalize")
            .with_output_type::<Result<PrefetchPipelineState>>()
    }
}

type PrefetchLastForwardResult = Result<PrefetchLastMoeForwarded>;
type PrefetchLastFinalizeLift =
    LiftResult<PrefetchLastFinalizeOp, PrefetchLastForwardResult, PrefetchPipelineState>;
type PrefetchRunLastOp = Then<PrefetchLastForwardMoeOp, PrefetchLastFinalizeLift>;

#[derive(Debug, Default, Clone, Copy)]
struct PrefetchNonLastPrefetchOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchPipelineStepCtx<'a>> for PrefetchNonLastPrefetchOp {
    type In = PrefetchBranchInput;
    type Out = Result<PrefetchNonLastPrefetched>;

    fn forward(&mut self, ctx: &mut PrefetchPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let PrefetchBranchInput {
            mut state,
            layer_idx,
            h_after_attn,
            ffn_residual,
            h_ffn_normed_typed,
            ..
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "prefetch pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let (routing_logits, top_weights_typed, remapped_indices_typed, _original_indices) =
            layer.moe_block.route_for_prefetch(&h_ffn_normed_typed)?;
        log_shape("prefetch.routing_logits", &routing_logits);
        log_shape("prefetch.routing_weights", top_weights_typed.inner());
        log_shape("prefetch.routing_indices", remapped_indices_typed.inner());
        if ctx.capture_experts {
            state
                .expert_indices
                .push(ModelWeights::normalize_captured_expert_indices(
                    remapped_indices_typed.inner().clone(),
                )?);
        }

        let (hidden_flat_typed, indices_flat_typed, weights_flat_typed, dispatch_dims) =
            layer.moe_block.prepare_dispatch_for_prefetch(
                h_ffn_normed_typed.clone(),
                remapped_indices_typed,
                top_weights_typed,
            )?;
        log_shape("prefetch.hidden_flat", hidden_flat_typed.inner());
        ctx.pipeline.submit_prefetch(
            layer_idx,
            hidden_flat_typed.inner(),
            indices_flat_typed.inner(),
            weights_flat_typed.inner(),
            dispatch_dims,
        )?;

        Ok(PrefetchNonLastPrefetched {
            state,
            layer_idx,
            h_after_attn,
            ffn_residual,
            h_ffn_normed_typed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchNonLastPrefetchOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "PrefetchNonLastPrefetch",
            Graph::sequence(
                <MoePrefetchGraph as Vis>::visualize(),
                Graph::custom_leaf("StorePendingMoeAndResidual"),
            ),
        )
        .with_output_type::<Result<PrefetchNonLastPrefetched>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct PrefetchNonLastFinalizeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchPipelineStepCtx<'a>> for PrefetchNonLastFinalizeOp {
    type In = PrefetchNonLastPrefetched;
    type Out = Result<PrefetchPipelineState>;

    fn forward(&mut self, ctx: &mut PrefetchPipelineStepCtx<'a>, input: Self::In) -> Self::Out {
        let PrefetchNonLastPrefetched {
            mut state,
            layer_idx,
            h_after_attn,
            ffn_residual,
            h_ffn_normed_typed,
        } = input;
        let len = ctx.layers.len();
        let layer = ctx.layers.get_mut(layer_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "prefetch pipeline layer index {} out of bounds (len {})",
                layer_idx, len
            ))
        })?;

        let adjusted_residual: TypedTensor<Hidden3> =
            if let Some(ref mut shared_exp) = layer.moe_block.shared_expert {
                let shared_out = shared_exp.forward(&h_ffn_normed_typed)?;
                log_shape("prefetch.shared_expert_out", shared_out.inner());
                paramecia_tensor::residual_add!(&ffn_residual, &shared_out)?
            } else {
                ffn_residual
            };

        state.pending_moe = Some(layer_idx);
        state.pending_residual = Some(adjusted_residual);
        state.h = h_after_attn;
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchNonLastFinalizeOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("PrefetchNonLastFinalize")
            .with_output_type::<Result<PrefetchPipelineState>>()
    }
}

type PrefetchNonLastPrefetchResult = Result<PrefetchNonLastPrefetched>;
type PrefetchNonLastFinalizeLift =
    LiftResult<PrefetchNonLastFinalizeOp, PrefetchNonLastPrefetchResult, PrefetchPipelineState>;
type PrefetchRunNonLastOp = Then<PrefetchNonLastPrefetchOp, PrefetchNonLastFinalizeLift>;

type PrefetchAttnNormResult = Result<PrefetchAttnNormed>;
type PrefetchAttnForwardLift =
    LiftResult<PrefetchAttnForwardOp, PrefetchAttnNormResult, PrefetchAttnForwarded>;
type PrefetchAttnForwardResult = Result<PrefetchAttnForwarded>;
type PrefetchFfnPrepLift =
    LiftResult<PrefetchFfnPrepOp, PrefetchAttnForwardResult, PrefetchBranchInput>;
type PrefetchAttnFfnFlow =
    Then<PrefetchAttnNormOp, Then<PrefetchAttnForwardLift, PrefetchFfnPrepLift>>;
fn prefetch_attn_ffn_flow() -> PrefetchAttnFfnFlow {
    Then::new(
        PrefetchAttnNormOp,
        Then::new(
            LiftResult::new(PrefetchAttnForwardOp),
            LiftResult::new(PrefetchFfnPrepOp),
        ),
    )
}
type PrefetchAttnFfnLiftFlow =
    LiftResult<PrefetchAttnFfnFlow, PrefetchResolveResult, PrefetchBranchInput>;
type PrefetchBranchSwitchFlow =
    SwitchRef<PrefetchBranchInput, PrefetchRunLastOp, PrefetchRunNonLastOp>;
type PrefetchBranchLiftFlow =
    LiftResult<PrefetchBranchSwitchFlow, PrefetchBranchInputResult, PrefetchPipelineState>;
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct PrefetchPipelineStep(
    PrefetchResolveOp,
    PrefetchAttnFfnLiftFlow,
    PrefetchBranchLiftFlow,
);
fn prefetch_pipeline_step_op() -> PrefetchPipelineStep {
    PrefetchPipelineStep(
        PrefetchResolveOp,
        LiftResult::new(prefetch_attn_ffn_flow()),
        LiftResult::new(SwitchRef::new(
            prefetch_is_last_branch,
            Then::new(
                PrefetchLastForwardMoeOp,
                LiftResult::new(PrefetchLastFinalizeOp),
            ),
            Then::new(
                PrefetchNonLastPrefetchOp,
                LiftResult::new(PrefetchNonLastFinalizeOp),
            ),
        )),
    )
}
type ToDeviceHiddenFlow = MapErr<
    ToDeviceOp<Hidden3>,
    TypedTensor<Hidden3>,
    TypedTensor<Hidden3>,
    TensorError,
    paramecia_core::Error,
>;
type ToPrimaryFlow = If<
    ToDeviceHiddenFlow,
    WrapOk<TypedTensor<Hidden3>, paramecia_core::Error>,
    Result<TypedTensor<Hidden3>>,
>;

fn to_primary_flow(ldm: &LayerDeviceMap) -> ToPrimaryFlow {
    If::new(
        ldm.is_multi_gpu(),
        MapErr::new(ToDeviceOp::new(ldm.primary_device().clone())),
        WrapOk::default(),
    )
}

type NormHiddenFlow = MapErr<
    RmsNormOp<Hidden3>,
    TypedTensor<Hidden3>,
    TypedTensor<Hidden3>,
    TensorError,
    paramecia_core::Error,
>;

fn norm_hidden_flow(norm: &RmsNorm) -> NormHiddenFlow {
    MapErr::new(RmsNormOp::new_with_shared(
        norm.weight().clone(),
        norm.eps(),
        norm.shared_weight().cloned(),
        norm.zero_centered(),
    ))
}

type LastTokenSliceOp = NarrowDynStartOp<Hidden3, U1, U1>;

type OutputHeadAtStep1 = LiftResult<
    ContiguousOp<HiddenLast3>,
    std::result::Result<THiddenLast, TensorError>,
    THiddenLast,
>;
type OutputHeadAtStep2 =
    LiftResult<ToDtypeOp<HiddenLast3>, std::result::Result<THiddenLast, TensorError>, THiddenLast>;
type OutputHeadAtStep3 = LiftResult<
    QMatMulOp<LmHeadShape, HiddenLast3, LogitsLast3>,
    std::result::Result<THiddenLast, TensorError>,
    TLogitsLast,
>;
type OutputHeadAtStep4 =
    LiftResult<SqueezeOp<LogitsLast3, U1>, std::result::Result<TLogitsLast, TensorError>, TLogits2>;
type OutputHeadAtOp = Then<
    LastTokenSliceOp,
    Then<OutputHeadAtStep1, Then<OutputHeadAtStep2, Then<OutputHeadAtStep3, OutputHeadAtStep4>>>,
>;
fn output_head_at_op(lm_head: &TQMatMul<LmHeadShape>, last_pos: usize) -> OutputHeadAtOp {
    Then::new(
        LastTokenSliceOp::new(last_pos),
        Then::new(
            LiftResult::new(ContiguousOp::default()),
            Then::new(
                LiftResult::new(ToDtypeOp::new(DType::F32)),
                Then::new(
                    LiftResult::new(QMatMulOp::new(lm_head.clone())),
                    LiftResult::new(SqueezeOp::default()),
                ),
            ),
        ),
    )
}

type OutputHeadAllStep2 = LiftResult<
    QMatMulOp<LmHeadShape, Hidden3, Shape3<B, N, V>>,
    std::result::Result<TypedTensor<Hidden3>, TensorError>,
    TLogits3,
>;
type OutputHeadAllOp = Then<ToDtypeOp<Hidden3>, OutputHeadAllStep2>;
fn output_head_all_op(lm_head: &TQMatMul<LmHeadShape>) -> OutputHeadAllOp {
    Then::new(
        ToDtypeOp::new(DType::F32),
        LiftResult::new(QMatMulOp::new(lm_head.clone())),
    )
}

#[allow(dead_code)]
struct ModelForwardCtx<'a> {
    model: &'a mut ModelWeights,
}

#[allow(dead_code)]
#[derive(Clone)]
struct ForwardEmbeddedState {
    h_typed: TypedTensor<Hidden3>,
    offset: usize,
    batch_size: usize,
    seq_len: usize,
    embed_ms: f64,
    profile: bool,
    prefetch_enabled: bool,
    capture_experts: bool,
}

#[allow(dead_code)]
struct EmbeddedTensorInfo {
    h: Tensor,
    offset: usize,
    embed_ms: f64,
    profile: bool,
    prefetch_enabled: bool,
    capture_experts: bool,
}

#[allow(dead_code)]
struct ForwardLayerOutput {
    h_typed: TypedTensor<Hidden3>,
    seq_len: usize,
    embed_ms: f64,
    layers_ms: f64,
    profile: bool,
}

#[allow(dead_code)]
struct GeneralHeadNormedState {
    h_normed: TypedTensor<Hidden3>,
    seq_len: usize,
    embed_ms: f64,
    layers_ms: f64,
    profile: bool,
}

type ModelEmbedForwardResult = Result<EmbeddedTensorInfo>;
#[derive(Debug, Default, Clone, Copy)]
struct ModelEmbedForwardOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<ModelForwardCtx<'a>> for ModelEmbedForwardOp {
    type In = (Tensor, usize);
    type Out = Result<EmbeddedTensorInfo>;

    fn forward(&mut self, ctx: &mut ModelForwardCtx<'a>, input: Self::In) -> Self::Out {
        let (input_ids, offset) = input;
        let t_embed = std::time::Instant::now();
        let h = ctx
            .model
            .embed_tokens
            .forward(&input_ids)?
            .to_dtype(ctx.model.dtype)?;
        log_shape("model.embedding", &h);
        let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
        Ok(EmbeddedTensorInfo {
            h,
            offset,
            embed_ms,
            profile: std::env::var("PARAMECIA_PROFILE").is_ok(),
            prefetch_enabled: ctx.model.prefetch_pipeline.is_some(),
            capture_experts: ctx.model.capture_expert_indices,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for ModelEmbedForwardOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("ModelEmbedForward").with_output_type::<Result<EmbeddedTensorInfo>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ModelEmbedFinalizeOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for ModelEmbedFinalizeOp {
    type In = EmbeddedTensorInfo;
    type Out = Result<ForwardEmbeddedState>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let EmbeddedTensorInfo {
            h,
            offset,
            embed_ms,
            profile,
            prefetch_enabled,
            capture_experts,
        } = input;
        let dims = h.dims();
        let batch_size = dims.first().copied().unwrap_or(0);
        let seq_len = dims.get(1).copied().unwrap_or(0);
        let h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;
        Ok(ForwardEmbeddedState {
            h_typed,
            offset,
            batch_size,
            seq_len,
            embed_ms,
            profile,
            prefetch_enabled,
            capture_experts,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for ModelEmbedFinalizeOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("ModelEmbedFinalize").with_output_type::<Result<ForwardEmbeddedState>>()
    }
}
type ModelEmbedFinalizeLift =
    LiftResult<ModelEmbedFinalizeOp, ModelEmbedForwardResult, ForwardEmbeddedState>;
type ModelEmbedResult = Result<ForwardEmbeddedState>;

#[allow(dead_code)]
#[derive(Inception)]
#[inception(properties = [Visualize])]
struct DecodeLayerLoopExec(
    TryFoldRange<DecodeLayerStepOp, TypedTensor<Hidden3>, paramecia_core::Error>,
);
#[primitive(property = Arrow)]
impl<'a> Combinator<ModelForwardCtx<'a>> for DecodeLayerLoopExec {
    type In = ForwardEmbeddedState;
    type Out = Result<TypedTensor<Hidden3>>;

    fn forward(&mut self, ctx: &mut ModelForwardCtx<'a>, input: Self::In) -> Self::Out {
        let ws = ctx.model.ensure_decode_workspace()?;
        let is_multi = ws.is_multi;
        let layer_devices = ws.layer_devices.clone();
        let num_layers = ctx.model.layers.len();
        let mut decode_ctx = DecodeLayerStepCtx {
            layers: &mut ctx.model.layers,
            is_multi,
            layer_devices: &layer_devices,
            offset: input.offset,
        };
        let decode_step: DecodeLayerStepOp = decode_layer_step_op();
        let mut decode_fold = TryFoldRange::new(decode_step);
        decode_fold.traced_forward(&mut decode_ctx, (input.h_typed, 0..num_layers))
    }
}
type DecodeLayersStep = DecodeLayerLoopExec;
type DecodeLayersResult = Result<TypedTensor<Hidden3>>;
type DecodeHeadLogitsStep =
    MapErr<OutputHeadAllOp, TypedTensor<Hidden3>, TLogits3, TensorError, paramecia_core::Error>;
type DecodeHeadNormThenLogits =
    Then<NormHiddenFlow, LiftResult<DecodeHeadLogitsStep, Result<TypedTensor<Hidden3>>, TLogits3>>;
type DecodeHeadNormLift =
    LiftResult<DecodeHeadNormThenLogits, Result<TypedTensor<Hidden3>>, TLogits3>;
type DecodeSqueezeLogitsOp = SqueezeOp<Shape3<B, N, V>, U1>;
type DecodeSqueezeLogitsLift = LiftResult<DecodeSqueezeLogitsOp, Result<TLogits3>, TLogits2>;
type DecodePostLayersFlow = Then<ToPrimaryFlow, Then<DecodeHeadNormLift, DecodeSqueezeLogitsLift>>;
type DecodePostLayersLift = LiftResult<DecodePostLayersFlow, DecodeLayersResult, TLogits2>;
type DecodeSingleFastPathFlow = Then<DecodeLayersStep, DecodePostLayersLift>;

#[allow(dead_code)]
#[derive(Inception)]
#[inception(properties = [Visualize])]
struct PrefetchLayerLoopExec(
    TryFoldRange<PrefetchPipelineStep, PrefetchPipelineState, paramecia_core::Error>,
);
#[primitive(property = Arrow)]
impl<'a> Combinator<ModelForwardCtx<'a>> for PrefetchLayerLoopExec {
    type In = ForwardEmbeddedState;
    type Out = Result<ForwardLayerOutput>;

    fn forward(&mut self, ctx: &mut ModelForwardCtx<'a>, input: Self::In) -> Self::Out {
        let ForwardEmbeddedState {
            h_typed,
            offset,
            batch_size,
            seq_len,
            embed_ms,
            profile,
            capture_experts,
            ..
        } = input;
        let pipeline = ctx.model.prefetch_pipeline.as_ref().ok_or_else(|| {
            paramecia_core::Error::Msg("Prefetch pipeline not enabled".to_string())
        })?;
        let num_layers = ctx.model.layers.len();
        let (_batch_size, _seq_len, _hidden_dim) = h_typed.inner().dims3()?;
        let activation_dtype = h_typed.inner().dtype();
        let is_multi = ctx.model.layer_device_map.is_multi_gpu();
        let causal_mask = if seq_len == 1 {
            None
        } else {
            Some(ctx.model.causal_mask(batch_size, seq_len, offset)?)
        };
        let device_masks = if seq_len == 1 {
            Vec::new()
        } else {
            ctx.model.per_device_masks(causal_mask.as_ref())?
        };
        let init_state = PrefetchPipelineState {
            h: h_typed,
            pending_moe: None,
            pending_residual: None,
            expert_indices: if capture_experts {
                Vec::with_capacity(num_layers)
            } else {
                Vec::new()
            },
        };
        let mut step_ctx = PrefetchPipelineStepCtx {
            layers: &mut ctx.model.layers,
            pipeline,
            is_multi,
            layer_device_map: &ctx.model.layer_device_map,
            device_masks: &device_masks,
            causal_mask: causal_mask.as_ref(),
            offset,
            activation_dtype,
            num_layers,
            capture_experts,
        };
        let t_layers = std::time::Instant::now();
        let step: PrefetchPipelineStep = prefetch_pipeline_step_op();
        let mut fold = TryFoldRange::new(step);
        let mut state = fold.traced_forward(&mut step_ctx, (init_state, 0..num_layers))?;

        if state.pending_moe.is_some() || state.pending_residual.is_some() {
            paramecia_core::bail!("prefetch pipeline ended with unresolved pending MoE state");
        }

        if capture_experts {
            ctx.model.last_expert_indices = Some(state.expert_indices);
        }
        if is_multi {
            state.h = transfer_to(
                state.h.into_inner(),
                ctx.model.layer_device_map.primary_device(),
            )?
            .try_into()?;
        }

        let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
        Ok(ForwardLayerOutput {
            h_typed: state.h,
            seq_len,
            embed_ms,
            layers_ms,
            profile,
        })
    }
}

#[allow(dead_code)]
#[derive(Inception)]
#[inception(properties = [Visualize])]
struct CaptureLayerLoopExec(TryFoldRange<LayerLoopStepOp, LayerRunState, paramecia_core::Error>);
#[primitive(property = Arrow)]
impl<'a> Combinator<ModelForwardCtx<'a>> for CaptureLayerLoopExec {
    type In = ForwardEmbeddedState;
    type Out = Result<ForwardLayerOutput>;

    fn forward(&mut self, ctx: &mut ModelForwardCtx<'a>, input: Self::In) -> Self::Out {
        let ForwardEmbeddedState {
            h_typed,
            offset,
            batch_size,
            seq_len,
            embed_ms,
            profile,
            ..
        } = input;
        let causal_mask = if seq_len == 1 {
            None
        } else {
            Some(ctx.model.causal_mask(batch_size, seq_len, offset)?)
        };
        let device_masks = if seq_len == 1 {
            Vec::new()
        } else {
            ctx.model.per_device_masks(causal_mask.as_ref())?
        };
        let is_multi = ctx.model.layer_device_map.is_multi_gpu();
        let num_layers = ctx.model.layers.len();
        let mut loop_ctx = LayerLoopCtx {
            layers: &mut ctx.model.layers,
            is_multi,
            layer_device_map: &ctx.model.layer_device_map,
            per_device_masks: &device_masks,
            mask: causal_mask.as_ref(),
            offset,
            mode: LayerForwardMode::WithStats,
        };
        let t_layers = std::time::Instant::now();
        let step: LayerLoopStepOp = layer_loop_step_op();
        let mut fold = TryFoldRange::new(step);
        let (h_typed, all_stats) = fold.traced_forward(
            &mut loop_ctx,
            (
                (h_typed, Some(Vec::with_capacity(num_layers))),
                0..num_layers,
            ),
        )?;
        let stats = all_stats.ok_or_else(|| {
            paramecia_core::Error::Msg(
                "run_layer_loop WithStats returned no router stats".to_string(),
            )
        })?;
        let expert_indices = stats
            .into_iter()
            .map(|(_router_logits, indices)| {
                ModelWeights::normalize_captured_expert_indices(indices)
            })
            .collect::<Result<Vec<_>>>()?;
        ctx.model.last_expert_indices = Some(expert_indices);
        let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
        Ok(ForwardLayerOutput {
            h_typed,
            seq_len,
            embed_ms,
            layers_ms,
            profile,
        })
    }
}

#[allow(dead_code)]
#[derive(Inception)]
#[inception(properties = [Visualize])]
struct NormalLayerLoopExec(TryFoldRange<LayerLoopStepOp, LayerRunState, paramecia_core::Error>);
#[primitive(property = Arrow)]
impl<'a> Combinator<ModelForwardCtx<'a>> for NormalLayerLoopExec {
    type In = ForwardEmbeddedState;
    type Out = Result<ForwardLayerOutput>;

    fn forward(&mut self, ctx: &mut ModelForwardCtx<'a>, input: Self::In) -> Self::Out {
        let ForwardEmbeddedState {
            h_typed,
            offset,
            batch_size,
            seq_len,
            embed_ms,
            profile,
            ..
        } = input;
        let causal_mask = if seq_len == 1 {
            None
        } else {
            Some(ctx.model.causal_mask(batch_size, seq_len, offset)?)
        };
        let device_masks = if seq_len == 1 {
            Vec::new()
        } else {
            ctx.model.per_device_masks(causal_mask.as_ref())?
        };
        let num_layers = ctx.model.layers.len();
        let mut loop_ctx = LayerLoopCtx {
            layers: &mut ctx.model.layers,
            is_multi: ctx.model.layer_device_map.is_multi_gpu(),
            layer_device_map: &ctx.model.layer_device_map,
            per_device_masks: &device_masks,
            mask: causal_mask.as_ref(),
            offset,
            mode: LayerForwardMode::Normal,
        };
        let t_layers = std::time::Instant::now();
        let step: LayerLoopStepOp = layer_loop_step_op();
        let mut fold = TryFoldRange::new(step);
        let (h_typed, _) = fold.traced_forward(&mut loop_ctx, ((h_typed, None), 0..num_layers))?;
        let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;
        Ok(ForwardLayerOutput {
            h_typed,
            seq_len,
            embed_ms,
            layers_ms,
            profile,
        })
    }
}

type GeneralPrefetchStep = PrefetchLayerLoopExec;
type GeneralCaptureStep = CaptureLayerLoopExec;
type GeneralNormalStep = NormalLayerLoopExec;
type LayerPathResult = Result<ForwardLayerOutput>;
type GeneralCaptureOrNormalFlow =
    SwitchRef<ForwardEmbeddedState, GeneralCaptureStep, GeneralNormalStep>;
type GeneralLayerPathSelectFlow =
    SwitchRef<ForwardEmbeddedState, GeneralPrefetchStep, GeneralCaptureOrNormalFlow>;

#[allow(dead_code)]
#[derive(Debug, Default, Clone, Copy)]
struct GeneralHeadToPrimaryOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<ModelForwardCtx<'a>> for GeneralHeadToPrimaryOp {
    type In = ForwardLayerOutput;
    type Out = Result<ForwardLayerOutput>;

    fn forward(&mut self, ctx: &mut ModelForwardCtx<'a>, input: Self::In) -> Self::Out {
        let ForwardLayerOutput {
            h_typed,
            seq_len,
            embed_ms,
            layers_ms,
            profile,
        } = input;
        let mut to_primary = to_primary_flow(&ctx.model.layer_device_map);
        let h_typed = to_primary.traced_forward(&mut (), h_typed)?;
        Ok(ForwardLayerOutput {
            h_typed,
            seq_len,
            embed_ms,
            layers_ms,
            profile,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for GeneralHeadToPrimaryOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph("GeneralHeadToPrimary", <ToPrimaryFlow as Vis>::visualize())
            .with_output_type::<Result<ForwardLayerOutput>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct GeneralHeadNormApplyOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<ModelForwardCtx<'a>> for GeneralHeadNormApplyOp {
    type In = ForwardLayerOutput;
    type Out = Result<GeneralHeadNormedState>;

    fn forward(&mut self, ctx: &mut ModelForwardCtx<'a>, input: Self::In) -> Self::Out {
        let ForwardLayerOutput {
            h_typed,
            seq_len,
            embed_ms,
            layers_ms,
            profile,
        } = input;
        let mut norm = norm_hidden_flow(&ctx.model.norm);
        let h_normed = norm.traced_forward(&mut (), h_typed)?;
        Ok(GeneralHeadNormedState {
            h_normed,
            seq_len,
            embed_ms,
            layers_ms,
            profile,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for GeneralHeadNormApplyOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph("GeneralHeadNormApply", <NormHiddenFlow as Vis>::visualize())
            .with_output_type::<Result<GeneralHeadNormedState>>()
    }
}

type GeneralHeadToPrimaryResult = Result<ForwardLayerOutput>;
type GeneralHeadNormLift =
    LiftResult<GeneralHeadNormApplyOp, GeneralHeadToPrimaryResult, GeneralHeadNormedState>;

#[derive(Debug, Default, Clone, Copy)]
struct GeneralHeadOutputOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<ModelForwardCtx<'a>> for GeneralHeadOutputOp {
    type In = GeneralHeadNormedState;
    type Out = Result<Tensor>;

    fn forward(&mut self, ctx: &mut ModelForwardCtx<'a>, input: Self::In) -> Self::Out {
        let GeneralHeadNormedState {
            h_normed,
            seq_len,
            embed_ms,
            layers_ms,
            profile,
        } = input;
        let t_head = std::time::Instant::now();
        let span_output = ctx.model.span_output.clone();
        let _enter_output = span_output.enter();
        let logits = ctx.model.output_head_at(&h_normed, seq_len - 1)?;
        log_shape("model.logits", &logits);
        #[cfg(feature = "vulkan")]
        if profile {
            if let Device::Vulkan(vk) = logits.device() {
                vk.flush()?;
            }
        }
        let head_ms = t_head.elapsed().as_secs_f64() * 1000.0;
        if profile {
            trace!(
                embed_ms = format_args!("{:.1}", embed_ms),
                layers_ms = format_args!("{:.1}", layers_ms),
                head_ms = format_args!("{:.1}", head_ms),
                total_ms = format_args!("{:.1}", embed_ms + layers_ms + head_ms),
                "Forward pass profile (prefetch)"
            );
        }
        Ok(logits)
    }
}
#[primitive(property = Visualize)]
impl Vis for GeneralHeadOutputOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "GeneralHeadOutput",
            Graph::sequence(
                <OutputHeadAtOp as Vis>::visualize(),
                Graph::custom_leaf("ProfileAndTraceHeadMs"),
            ),
        )
        .with_output_type::<Result<Tensor>>()
    }
}
type GeneralHeadOutputLift =
    LiftResult<GeneralHeadOutputOp, Result<GeneralHeadNormedState>, Tensor>;
type GeneralHeadFlow =
    Then<GeneralHeadToPrimaryOp, Then<GeneralHeadNormLift, GeneralHeadOutputLift>>;
type GeneralHeadLift = LiftResult<GeneralHeadFlow, LayerPathResult, Tensor>;
type GeneralForwardPathFlow = Then<GeneralLayerPathSelectFlow, GeneralHeadLift>;

type ForwardPathSelectFlow =
    SwitchRef<ForwardEmbeddedState, DecodeSingleFastPathFlow, GeneralForwardPathFlow>;

type ModelForwardTail = LiftResult<ForwardPathSelectFlow, ModelEmbedResult, Tensor>;
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct ModelForward(
    ModelEmbedForwardOp,
    ModelEmbedFinalizeLift,
    ModelForwardTail,
);

#[primitive(property = Visualize)]
impl Vis for ModelWeights {
    fn visualize() -> Graph {
        type ModelForwardVisFlow = ModelForward;
        Graph::wrap_subgraph(MODEL_NODE_LABEL, <ModelForwardVisFlow as Vis>::visualize())
    }
}

impl ModelWeights {
    fn normalize_captured_expert_indices(indices: Tensor) -> Result<Tensor> {
        let dims = indices.dims().to_vec();
        match dims.as_slice() {
            // Already flattened: [tokens, topk]
            [_, _] => {
                let typed: TTopIndices2 = indices.try_into()?;
                Ok(typed.into_inner())
            }
            // Layer-local stats may be [batch, seq, topk]; flatten to [tokens, topk].
            [b, n, tk] => {
                let flat = indices.contiguous()?.reshape((b * n, *tk))?;
                let typed: TTopIndices2 = flat.try_into()?;
                Ok(typed.into_inner())
            }
            _ => paramecia_core::bail!(
                "captured expert indices must have shape [T, Tk] or [B, N, Tk], got rank {} shape {:?}",
                dims.len(),
                dims
            ),
        }
    }

    /// Compute logits from a prefill hidden state at a specific position.
    /// Narrows to `last_pos`, makes contiguous, converts to F32, applies typed lm_head, squeezes.
    fn output_head_at(&self, h: &TypedTensor<Hidden3>, last_pos: usize) -> Result<Tensor> {
        let mut head_at = output_head_at_op(&self.lm_head, last_pos);
        head_at
            .traced_forward(&mut (), h.clone())
            .map(TypedTensor::into_inner)
            .map_err(paramecia_core::Error::from)
    }

    /// Compute logits for all sequence positions (training / verify paths).
    /// Converts to F32 if needed, applies typed lm_head, returns full [B, L, V] tensor.
    fn output_head_all(&self, h: &TypedTensor<Hidden3>) -> Result<Tensor> {
        let mut head_all = output_head_all_op(&self.lm_head);
        head_all
            .traced_forward(&mut (), h.clone())
            .map(TypedTensor::into_inner)
            .map_err(paramecia_core::Error::from)
    }

    /// Run the hidden state through all transformer layers.
    ///
    /// Common infrastructure for all forward variants: handles device transfer,
    /// mask selection, and per-layer dispatch via `LayerForwardMode`.
    /// Does NOT include the final device transfer to primary — callers use
    /// `transfer_to_primary()` for that so they can capture pre-transfer state if needed.
    fn run_layer_loop(
        &mut self,
        h: TypedTensor<Hidden3>,
        mask: Option<&Tensor>,
        per_device_masks: &[(String, Tensor)],
        offset: usize,
        mode: LayerForwardMode,
    ) -> Result<(TypedTensor<Hidden3>, Option<Vec<(Tensor, Tensor)>>)> {
        let is_multi = self.layer_device_map.is_multi_gpu();
        let num_layers = self.layers.len();
        let collect_stats = mode == LayerForwardMode::WithStats;
        let mut all_stats: Option<Vec<(Tensor, Tensor)>> = if collect_stats {
            Some(Vec::with_capacity(num_layers))
        } else {
            None
        };
        let mut ctx = LayerLoopCtx {
            layers: &mut self.layers,
            is_multi,
            layer_device_map: &self.layer_device_map,
            per_device_masks,
            mask,
            offset,
            mode,
        };
        let step: LayerLoopStepOp = layer_loop_step_op();
        let mut fold = TryFoldRange::new(step);
        fold.traced_forward(&mut ctx, ((h, all_stats.take()), 0..num_layers))
    }

    /// Transfer hidden state to primary device if running multi-GPU.
    fn transfer_to_primary(&self, h: Tensor) -> Result<Tensor> {
        if self.layer_device_map.is_multi_gpu() {
            transfer_to(h, self.layer_device_map.primary_device())
        } else {
            Ok(h)
        }
    }

    fn transfer_to_primary_typed(&self, h: TypedTensor<Hidden3>) -> Result<TypedTensor<Hidden3>> {
        let out = self.transfer_to_primary(h.into_inner())?;
        Ok(out.try_into()?)
    }

    fn ensure_decode_workspace(&mut self) -> Result<&DecodeWorkspace> {
        if self.decode_workspace.is_none() {
            let is_multi = self.layer_device_map.is_multi_gpu();
            let mut layer_devices = Vec::with_capacity(self.layers.len());
            for i in 0..self.layers.len() {
                layer_devices.push(self.layer_device_map.device_for_layer(i).clone());
            }
            self.decode_workspace = Some(DecodeWorkspace {
                is_multi,
                layer_devices,
            });
        }
        self.decode_workspace
            .as_ref()
            .ok_or_else(|| paramecia_core::Error::Msg("decode workspace unavailable".to_string()))
    }

    fn forward_decode_with_workspace(
        &mut self,
        h_in: TypedTensor<Hidden3>,
        offset: usize,
        embed_ms: f64,
        profile: bool,
    ) -> Result<Tensor> {
        self.forward_decode_typed(h_in, offset, embed_ms, profile)
    }

    /// Typed decode path: validates the hidden state shape [B, 1, hidden_size] entering
    /// the layer loop, then enforces shape preservation through every layer via typed forward.
    /// Falls back to untyped ops for norm + output head (which change the shape).
    fn forward_decode_typed(
        &mut self,
        h_in: TypedTensor<Hidden3>,
        offset: usize,
        embed_ms: f64,
        profile: bool,
    ) -> Result<Tensor> {
        let ws = self.ensure_decode_workspace()?;
        let is_multi = ws.is_multi;
        let layer_devices = ws.layer_devices.clone();

        let t_layers = std::time::Instant::now();
        let num_layers = self.layers.len();
        let mut decode_ctx = DecodeLayerStepCtx {
            layers: &mut self.layers,
            is_multi,
            layer_devices: &layer_devices,
            offset,
        };
        let decode_step: DecodeLayerStepOp = decode_layer_step_op();
        let mut decode_fold = TryFoldRange::new(decode_step);
        let h = decode_fold.traced_forward(&mut decode_ctx, (h_in, 0..num_layers))?;
        let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;

        let t_head = std::time::Instant::now();
        let span_output = self.span_output.clone();
        let _enter_output = span_output.enter();
        let mut post_layers = Then::new(
            to_primary_flow(&self.layer_device_map),
            LiftResult::new(Then::new(
                norm_hidden_flow(&self.norm),
                LiftResult::new(MapErr::new(output_head_all_op(&self.lm_head))),
            )),
        );
        let logits = post_layers
            .traced_forward(&mut (), h)?
            .into_inner()
            .squeeze(1)?;
        #[cfg(feature = "vulkan")]
        if profile {
            if let Device::Vulkan(vk) = logits.device() {
                vk.flush()?;
            }
        }
        let head_ms = t_head.elapsed().as_secs_f64() * 1000.0;

        if profile {
            trace!(
                embed_ms = format_args!("{:.1}", embed_ms),
                layers_ms = format_args!("{:.1}", layers_ms),
                head_ms = format_args!("{:.1}", head_ms),
                total_ms = format_args!("{:.1}", embed_ms + layers_ms + head_ms),
                "Forward pass profile (typed)"
            );
        }

        Ok(logits)
    }

    fn read_model_config<R: Read + Seek>(gg: &Gguf<R>) -> Result<ModelConfig> {
        let md_get_any = |keys: &[&str]| -> Result<&paramecia_core::quantized::gguf_file::Value> {
            keys.iter()
                .find_map(|k| gg.metadata().get(*k))
                .ok_or_else(|| {
                    paramecia_core::Error::Msg(format!(
                        "cannot find any metadata key in {:?}",
                        keys
                    ))
                })
        };
        let md_get_opt = |keys: &[&str]| -> Option<&paramecia_core::quantized::gguf_file::Value> {
            keys.iter().find_map(|k| gg.metadata().get(*k))
        };

        let num_attention_heads = md_get_any(&[
            "qwen35.attention.head_count",
            "qwen35moe.attention.head_count",
            "qwen3_5_moe.attention.head_count",
            "qwen3_5.attention.head_count",
            "qwen3next.attention.head_count",
            "qwen3moe.attention.head_count",
            "llama.attention.head_count",
        ])?
        .to_u32()? as usize;
        let num_key_value_heads = md_get_any(&[
            "qwen35.attention.head_count_kv",
            "qwen35moe.attention.head_count_kv",
            "qwen3_5_moe.attention.head_count_kv",
            "qwen3_5.attention.head_count_kv",
            "qwen3next.attention.head_count_kv",
            "qwen3moe.attention.head_count_kv",
            "llama.attention.head_count_kv",
        ])?
        .to_u32()? as usize;
        let head_dim = md_get_any(&[
            "qwen35.attention.key_length",
            "qwen35moe.attention.key_length",
            "qwen3_5_moe.attention.key_length",
            "qwen3_5.attention.key_length",
            "qwen3next.attention.key_length",
            "qwen3moe.attention.key_length",
            "llama.attention.key_length",
        ])?
        .to_u32()? as usize;
        let num_layers = md_get_any(&[
            "qwen35.block_count",
            "qwen35moe.block_count",
            "qwen3_5_moe.block_count",
            "qwen3_5.block_count",
            "qwen3next.block_count",
            "qwen3moe.block_count",
            "llama.block_count",
        ])?
        .to_u32()? as usize;
        let hidden_size = md_get_any(&[
            "qwen35.embedding_length",
            "qwen35moe.embedding_length",
            "qwen3_5_moe.embedding_length",
            "qwen3_5.embedding_length",
            "qwen3next.embedding_length",
            "qwen3moe.embedding_length",
            "llama.embedding_length",
        ])?
        .to_u32()? as usize;
        let max_position_embeddings = md_get_any(&[
            "qwen35.context_length",
            "qwen35moe.context_length",
            "qwen3_5_moe.context_length",
            "qwen3_5.context_length",
            "qwen3next.context_length",
            "qwen3moe.context_length",
            "llama.context_length",
        ])?
        .to_u32()? as usize;
        let rms_norm_eps = md_get_any(&[
            "qwen35.attention.layer_norm_rms_epsilon",
            "qwen35moe.attention.layer_norm_rms_epsilon",
            "qwen3_5_moe.attention.layer_norm_rms_epsilon",
            "qwen3_5.attention.layer_norm_rms_epsilon",
            "qwen3next.attention.layer_norm_rms_epsilon",
            "qwen3moe.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_rms_epsilon",
        ])?
        .to_f32()? as f64;
        let rope_freq_base = md_get_any(&[
            "qwen35.rope.freq_base",
            "qwen35moe.rope.freq_base",
            "qwen3_5_moe.rope.freq_base",
            "qwen3_5.rope.freq_base",
            "qwen3next.rope.freq_base",
            "qwen3moe.rope.freq_base",
            "llama.rope.freq_base",
        ])?
        .to_f32()? as f64;

        // Read n_rot (rope dimension count) - only these dimensions get rotated.
        // For Qwen3-family MoE this is typically 25% of head_dim (64 out of 256).
        // The remaining dimensions pass through unchanged.
        let n_rot = md_get_opt(&[
            "qwen35.rope.dimension_count",
            "qwen35moe.rope.dimension_count",
            "qwen3_5_moe.rope.dimension_count",
            "qwen3_5.rope.dimension_count",
            "qwen3next.rope.dimension_count",
            "qwen3moe.rope.dimension_count",
            "llama.rope.dimension_count",
        ])
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or(head_dim / 4);

        let rope_dimension_sections = md_get_opt(&[
            "qwen35.rope.dimension_sections",
            "qwen35moe.rope.dimension_sections",
            "qwen3_5_moe.rope.dimension_sections",
            "qwen3_5.rope.dimension_sections",
            "qwen3next.rope.dimension_sections",
            "qwen3moe.rope.dimension_sections",
            "llama.rope.dimension_sections",
        ])
        .and_then(|v| v.to_vec().ok())
        .and_then(|values| {
            let mut sections = Vec::with_capacity(values.len());
            for value in values {
                let section = value
                    .to_u32()
                    .ok()
                    .map(|v| v as usize)
                    .or_else(|| value.to_u64().ok().map(|v| v as usize));
                match section {
                    Some(v) => sections.push(v),
                    None => return None,
                }
            }
            Some(sections)
        });

        let rope_interleaved = md_get_opt(&[
            "qwen35.rope.interleaved",
            "qwen35moe.rope.interleaved",
            "qwen3_5_moe.rope.interleaved",
            "qwen3_5.rope.interleaved",
            "qwen3next.rope.interleaved",
            "qwen3moe.rope.interleaved",
            "llama.rope.interleaved",
        ])
        .and_then(|v| v.to_bool().ok())
        // GGUF does not always export explicit interleaved mRoPE metadata, but
        // dimension sections indicate that mRoPE layout is in use.
        .unwrap_or_else(|| {
            rope_dimension_sections
                .as_ref()
                .map(|s| !s.is_empty())
                .unwrap_or(false)
        });

        if let Some(sections) = rope_dimension_sections.as_ref() {
            let total_pairs: usize = sections.iter().sum();
            let expected_pairs = n_rot / 2;
            if total_pairs != expected_pairs {
                warn!(
                    total_pairs,
                    expected_pairs, n_rot, "rope.dimension_sections sum does not match n_rot/2"
                );
            }
        }

        let architecture = md_get_opt(&["general.architecture"])
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_ascii_lowercase());
        let dense_qwen35 = matches!(
            architecture.as_deref(),
            Some("qwen35") | Some("qwen3_5") | Some("qwen3.5")
        );

        let num_experts = md_get_opt(&[
            "qwen35.expert_count",
            "qwen35moe.expert_count",
            "qwen3_5_moe.expert_count",
            "qwen3_5.expert_count",
            "qwen3next.expert_count",
            "qwen3moe.expert_count",
        ])
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or_else(|| if dense_qwen35 { 1 } else { 256 });
        let num_experts_per_tok = md_get_opt(&[
            "qwen35.expert_used_count",
            "qwen35moe.expert_used_count",
            "qwen3_5_moe.expert_used_count",
            "qwen3_5.expert_used_count",
            "qwen3next.expert_used_count",
            "qwen3moe.expert_used_count",
        ])
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or_else(|| if dense_qwen35 { 1 } else { 8 });

        let ssm_d_inner = md_get_opt(&[
            "qwen35.ssm.inner_size",
            "qwen35.ssm.d_inner",
            "qwen35moe.ssm.inner_size",
            "qwen35moe.ssm.d_inner",
            "qwen3_5_moe.ssm.inner_size",
            "qwen3_5_moe.ssm.d_inner",
            "qwen3_5.ssm.inner_size",
            "qwen3_5.ssm.d_inner",
            "qwen3next.ssm.inner_size",
            "qwen3next.ssm.d_inner",
            "qwen3moe.ssm.inner_size",
        ])
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or(hidden_size * 2);
        let ssm_d_state = md_get_opt(&[
            "qwen35.ssm.state_size",
            "qwen35.ssm.d_state",
            "qwen35moe.ssm.state_size",
            "qwen35moe.ssm.d_state",
            "qwen3_5_moe.ssm.state_size",
            "qwen3_5_moe.ssm.d_state",
            "qwen3_5.ssm.state_size",
            "qwen3_5.ssm.d_state",
            "qwen3next.ssm.state_size",
            "qwen3next.ssm.d_state",
            "qwen3moe.ssm.state_size",
        ])
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or(128);
        let ssm_n_groups = md_get_opt(&[
            "qwen35.ssm.group_count",
            "qwen35.ssm.n_groups",
            "qwen35moe.ssm.group_count",
            "qwen35moe.ssm.n_groups",
            "qwen3_5_moe.ssm.group_count",
            "qwen3_5_moe.ssm.n_groups",
            "qwen3_5.ssm.group_count",
            "qwen3_5.ssm.n_groups",
            "qwen3next.ssm.group_count",
            "qwen3next.ssm.n_groups",
            "qwen3moe.ssm.group_count",
        ])
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or(16);
        let ssm_dt_rank = md_get_opt(&[
            "qwen35.ssm.time_step_rank",
            "qwen35.ssm.dt_rank",
            "qwen35moe.ssm.time_step_rank",
            "qwen35moe.ssm.dt_rank",
            "qwen3_5_moe.ssm.time_step_rank",
            "qwen3_5_moe.ssm.dt_rank",
            "qwen3_5.ssm.time_step_rank",
            "qwen3_5.ssm.dt_rank",
            "qwen3next.ssm.time_step_rank",
            "qwen3next.ssm.dt_rank",
            "qwen3moe.ssm.time_step_rank",
        ])
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or(32);

        let linear_v_heads_tiled_order = architecture
            .as_deref()
            .map(|a| a == "qwen35" || a == "qwen35moe")
            .or_else(|| {
                if md_get_opt(&["qwen35moe.block_count", "qwen3_5_moe.block_count"]).is_some() {
                    Some(true)
                } else if md_get_opt(&["qwen35.block_count", "qwen3_5.block_count"]).is_some() {
                    Some(true)
                } else {
                    None
                }
            })
            .unwrap_or(false);

        // Determine recurrent layers (linear attention without RoPE).
        let recurrent_layers: Vec<bool> = (0..num_layers)
            .map(|i| {
                md_get_opt(&[
                    &format!("qwen35.layer.{}.is_recurrent", i),
                    &format!("qwen35moe.layer.{}.is_recurrent", i),
                    &format!("qwen3_5_moe.layer.{}.is_recurrent", i),
                    &format!("qwen3_5.layer.{}.is_recurrent", i),
                    &format!("qwen3next.layer.{}.is_recurrent", i),
                    &format!("qwen3moe.layer.{}.is_recurrent", i),
                ])
                .and_then(|v| v.to_bool().ok())
                .unwrap_or((i + 1) % 4 != 0)
            })
            .collect();

        // Read YARN configuration from metadata (if present)
        let yarn_config = Self::read_yarn_config(gg, max_position_embeddings);

        Ok(ModelConfig {
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            num_layers,
            hidden_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_freq_base,
            n_rot,
            rope_interleaved,
            num_experts,
            num_experts_per_tok,
            ssm_d_inner,
            ssm_d_state,
            ssm_n_groups,
            ssm_dt_rank,
            linear_v_heads_tiled_order,
            recurrent_layers,
            dtype: DType::F32,
            yarn_config,
        })
    }

    /// Read YARN configuration from GGUF metadata.
    /// Returns None if YARN is not configured or target context equals original.
    fn read_yarn_config<R: Read + Seek>(
        gg: &Gguf<R>,
        max_position_embeddings: usize,
    ) -> Option<YarnConfig> {
        let md_get_any = |keys: &[&str]| keys.iter().find_map(|s| gg.metadata().get(*s));

        let scaling_type = md_get_any(&[
            "qwen35.rope.scaling.type",
            "qwen35moe.rope.scaling.type",
            "qwen3_5_moe.rope.scaling.type",
            "qwen3_5.rope.scaling.type",
            "qwen3next.rope.scaling.type",
            "qwen3moe.rope.scaling.type",
            "llama.rope.scaling.type",
        ])
        .and_then(|v| v.to_string().ok());

        let scale_factor = md_get_any(&[
            "qwen35.rope.scaling.factor",
            "qwen35moe.rope.scaling.factor",
            "qwen3_5_moe.rope.scaling.factor",
            "qwen3_5.rope.scaling.factor",
            "qwen3next.rope.scaling.factor",
            "qwen3moe.rope.scaling.factor",
            "llama.rope.scaling.factor",
            "qwen35.rope.scale_linear",
            "qwen35moe.rope.scale_linear",
            "qwen3_5_moe.rope.scale_linear",
            "qwen3_5.rope.scale_linear",
            "qwen3next.rope.scale_linear",
            "llama.rope.scale_linear",
        ])
        .and_then(|v| v.to_f32().ok());

        let original_context = md_get_any(&[
            "qwen35.rope.scaling.original_context_length",
            "qwen35moe.rope.scaling.original_context_length",
            "qwen3_5_moe.rope.scaling.original_context_length",
            "qwen3_5.rope.scaling.original_context_length",
            "qwen3next.rope.scaling.original_context_length",
            "qwen3moe.rope.scaling.original_context_length",
            "llama.rope.scaling.original_context_length",
        ])
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize);

        let beta_fast = md_get_any(&[
            "qwen35.rope.scaling.yarn.beta_fast",
            "qwen35moe.rope.scaling.yarn.beta_fast",
            "qwen3_5_moe.rope.scaling.yarn.beta_fast",
            "qwen3_5.rope.scaling.yarn.beta_fast",
            "qwen3next.rope.scaling.yarn.beta_fast",
            "llama.rope.scaling.yarn.beta_fast",
        ])
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(32.0);

        let beta_slow = md_get_any(&[
            "qwen35.rope.scaling.yarn.beta_slow",
            "qwen35moe.rope.scaling.yarn.beta_slow",
            "qwen3_5_moe.rope.scaling.yarn.beta_slow",
            "qwen3_5.rope.scaling.yarn.beta_slow",
            "qwen3next.rope.scaling.yarn.beta_slow",
            "llama.rope.scaling.yarn.beta_slow",
        ])
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(1.0);

        let attn_factor = md_get_any(&[
            "qwen35.rope.scaling.yarn.attn_factor",
            "qwen35moe.rope.scaling.yarn.attn_factor",
            "qwen3_5_moe.rope.scaling.yarn.attn_factor",
            "qwen3_5.rope.scaling.yarn.attn_factor",
            "qwen3next.rope.scaling.yarn.attn_factor",
            "llama.rope.scaling.yarn.attn_factor",
        ])
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(1.0);

        let is_yarn = scaling_type
            .as_ref()
            .map(|t| t.to_lowercase() == "yarn")
            .unwrap_or(false);

        if is_yarn || scale_factor.is_some() || original_context.is_some() {
            let orig_ctx = original_context.unwrap_or_else(|| {
                if let Some(sf) = scale_factor {
                    (max_position_embeddings as f32 / sf) as usize
                } else {
                    max_position_embeddings
                }
            });

            let target_ctx = if let Some(sf) = scale_factor {
                (orig_ctx as f32 * sf) as usize
            } else {
                max_position_embeddings
            };

            if target_ctx > orig_ctx {
                return Some(YarnConfig {
                    original_context: orig_ctx,
                    target_context: target_ctx,
                    beta_fast,
                    beta_slow,
                    attn_factor,
                });
            }
        }

        None
    }

    fn build_layers<R: Read + Seek, F>(
        gg: &mut Gguf<R>,
        config: &ModelConfig,
        rotary: Arc<RotaryEmbedding>,
        kv_cache_quantization: KvCacheQuantization,
        layer_device_map: &LayerDeviceMap,
        mut load_moe_block: F,
    ) -> Result<Vec<LayerWeights>>
    where
        F: FnMut(&mut Gguf<R>, &str, &Device, usize) -> Result<MoeBlock>,
    {
        let rotary_map: std::collections::HashMap<String, Arc<RotaryEmbedding>> =
            if layer_device_map.is_multi_gpu() {
                let mut map = std::collections::HashMap::new();
                for i in 0..config.num_layers {
                    let dev = layer_device_map.device_for_layer(i);
                    let key = format!("{:?}", dev);
                    if let std::collections::hash_map::Entry::Vacant(e) = map.entry(key) {
                        let r = Arc::new(RotaryEmbedding::new(
                            config.dtype,
                            config.head_dim,
                            config.n_rot,
                            config.max_position_embeddings,
                            config.rope_freq_base,
                            config.rope_interleaved,
                            config.yarn_config.as_ref(),
                            dev,
                        )?);
                        e.insert(r);
                    }
                }
                map
            } else {
                std::collections::HashMap::new()
            };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("blk.{}", i);

            let layer_device = layer_device_map.device_for_layer(i);
            let prev_device = if layer_device_map.is_multi_gpu() {
                Some(gg.set_device(layer_device.clone()))
            } else {
                None
            };

            let attn_norm = gg.rms_norm(
                &format!("{}.attn_norm.weight", prefix),
                config.rms_norm_eps,
                config.dtype,
            )?;
            let ffn_norm = gg
                .rms_norm(
                    &format!("{}.ffn_norm.weight", prefix),
                    config.rms_norm_eps,
                    config.dtype,
                )
                .or_else(|_| {
                    gg.rms_norm(
                        &format!("{}.post_attention_norm.weight", prefix),
                        config.rms_norm_eps,
                        config.dtype,
                    )
                })?;

            let is_recurrent = config.recurrent_layers.get(i).copied().unwrap_or(false)
                || gg
                    .try_tensor(&format!("{}.ssm_in.weight", prefix))?
                    .is_some()
                || gg.try_tensor(&format!("{}.ssm_in", prefix))?.is_some()
                || gg.try_tensor(&format!("{}.ssm_a", prefix))?.is_some();

            let layer_rotary = if layer_device_map.is_multi_gpu() {
                let key = format!("{:?}", layer_device);
                rotary_map
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| rotary.clone())
            } else {
                rotary.clone()
            };

            let attn = if is_recurrent {
                AttentionLayer::Linear(LinearAttention::new(
                    gg,
                    &prefix,
                    config.ssm_d_inner,
                    config.ssm_d_state,
                    config.ssm_n_groups,
                    config.ssm_dt_rank,
                    config.hidden_size,
                    config.rms_norm_eps,
                    config.dtype,
                    config.linear_v_heads_tiled_order,
                )?)
            } else {
                AttentionLayer::Full(FullAttention::new(
                    gg,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.rms_norm_eps,
                    layer_rotary,
                    &prefix,
                    config.dtype,
                    kv_cache_quantization,
                    config.max_position_embeddings,
                )?)
            };

            let moe_block = load_moe_block(gg, &prefix, layer_device, i)?;

            if let Some(prev) = prev_device {
                gg.set_device(prev);
            }

            layers.push(LayerWeights {
                attn,
                moe_block,
                attn_norm,
                ffn_norm,
            });
        }

        Ok(layers)
    }

    fn load_expert_tensor(path: &Path, tensor_name: &str, device: &Device) -> Result<QTensor> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut gg = Gguf::new(content, file, device.clone(), device.clone());
        gg.tensor(tensor_name)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_moe_block_with_devices<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        config: &ModelConfig,
        path: &Path,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
        compute_device: &Device,
        layer_idx: usize,
    ) -> Result<MoeBlock> {
        if config.num_experts == 1 {
            return MoeBlock::new(
                gg,
                prefix,
                config.num_experts,
                config.num_experts_per_tok,
                config.dtype,
                compute_device,
                default_cache_capacity(config.num_experts_per_tok),
                layer_idx,
            );
        }

        let gate_exps = Self::load_expert_tensor(
            path,
            &format!("{}.ffn_gate_exps.weight", prefix),
            gate_device,
        )?;
        let up_exps =
            Self::load_expert_tensor(path, &format!("{}.ffn_up_exps.weight", prefix), up_device)?;
        let down_exps = Self::load_expert_tensor(
            path,
            &format!("{}.ffn_down_exps.weight", prefix),
            down_device,
        )?;
        let cache_capacity = default_cache_capacity(config.num_experts_per_tok);

        let gate = gg
            .typed_qmatmul::<paramecia_tensor::glowstick::Shape2<super::shape::E, super::shape::S>>(
                &format!("{}.ffn_gate_inp.weight", prefix),
            )?;
        let shared_expert = SharedExpert::new(gg, prefix, config.dtype)?;

        let expert_mask: Option<TExpertVec> = gg
            .try_tensor(&format!("{}.expert_mask", prefix))?
            .map(|qt| -> Result<TExpertVec> { Ok(qt.dequantize(compute_device)?.try_into()?) })
            .transpose()?;

        let expert_remap: Option<TExpertVec> = gg
            .try_tensor(&format!("{}.expert_remap", prefix))?
            .map(|qt| -> Result<TExpertVec> { Ok(qt.dequantize(compute_device)?.try_into()?) })
            .transpose()?;

        let gate_name = format!("{}.ffn_gate_exps.weight", prefix);
        let up_name = format!("{}.ffn_up_exps.weight", prefix);
        let down_name = format!("{}.ffn_down_exps.weight", prefix);
        let gate_exps = SharedQTensor::new(gate_exps);
        let up_exps = SharedQTensor::new(up_exps);
        let down_exps = SharedQTensor::new(down_exps);
        gg.shared_tensors.push((gate_name, gate_exps.clone()));
        gg.shared_tensors.push((up_name, up_exps.clone()));
        gg.shared_tensors.push((down_name, down_exps.clone()));
        let gate_exps = ExpertWeightTensor::new(gate_exps, "gate_exps")?;
        let up_exps = ExpertWeightTensor::new(up_exps, "up_exps")?;
        let down_exps = ExpertWeightTensor::new(down_exps, "down_exps")?;

        let cache = if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
            Some(ExpertCache::new(compute_device.clone(), cache_capacity))
        } else {
            None
        };

        let training_cache =
            if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
                Some(ExpertCache::new(compute_device.clone(), cache_capacity))
            } else {
                None
            };

        let experts = MoeExperts {
            gate_exps,
            up_exps,
            down_exps,
            act_fn: Activation::Silu,
            span: tracing::span!(tracing::Level::TRACE, "moe-experts"),
            cache,
            training_cache,
            compute_device: compute_device.clone(),
            custom_gate_block_mults: None,
            custom_up_block_mults: None,
            custom_down_block_mults: None,
            gpu_hot_cache: None,
            gpu_device: None,
        };

        Ok(MoeBlock {
            experts,
            shared_expert,
            route_remap: MoeRouteRemap::new(
                gate.clone(),
                config.num_experts_per_tok,
                expert_mask.clone(),
                expert_remap.clone(),
            ),
            dispatch_prep: MoeDispatchPrep::new(config.num_experts_per_tok),
            group_assignments: MoeGroupAssignments::new(config.num_experts),
            select_prefetch: super::moe::MoePrefetchSelect::new(),
            select_sequential_exec: super::moe::MoeSequentialExecSelect::new(),
            gate,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            span: tracing::span!(tracing::Level::TRACE, "moe-block"),
            layer_idx,
            expert_mask,
            expert_remap,
        })
    }

    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone(), device.clone());
        let config = Self::read_model_config(&gg)?;
        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, config.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            config.rope_interleaved,
            config.yarn_config.as_ref(),
            device,
        )?);

        let layers = Self::build_layers(
            &mut gg,
            &config,
            rotary.clone(),
            KvCacheQuantization::F16,
            &LayerDeviceMap::single(device.clone()),
            |gg, prefix, compute_device, layer_idx| {
                MoeBlock::new(
                    gg,
                    prefix,
                    config.num_experts,
                    config.num_experts_per_tok,
                    config.dtype,
                    compute_device,
                    0,
                    layer_idx,
                )
            },
        )?;

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head = gg
            .typed_qmatmul::<LmHeadShape>("output.weight")
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embd.weight"))
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embdd.weight"))?;
        log_typed_qmatmul_shape("lm_head", &lm_head);

        let mtp_head = MtpHead::new(
            &mut gg,
            config.head_dim,
            config.rms_norm_eps,
            rotary.clone(),
            config.num_experts,
            config.num_experts_per_tok,
            config.dtype,
        )?;

        let shared_tensors = gg.take_shared_tensors();
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: config.dtype,
            span,
            span_output,
            layer_device_map: LayerDeviceMap::single(device.clone()),
            prefetch_pipeline: None,
            shared_tensors,
            last_expert_indices: None,
            capture_expert_indices: false,
            mtp_head,
            decode_workspace: None,
        })
    }

    pub fn from_gguf_with_device_map<P: AsRef<Path>>(
        path: P,
        device: &Device,
        expert_device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_kv_cache_config(
            path,
            device,
            expert_device,
            expert_device,
            expert_device,
            KvCacheQuantization::F16,
        )
    }

    pub fn from_gguf_with_expert_device_map<P: AsRef<Path>>(
        path: P,
        device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_kv_cache_config(
            path,
            device,
            gate_device,
            up_device,
            down_device,
            KvCacheQuantization::F16,
        )
    }

    /// Load model from GGUF with explicit device placement for expert weights and KV-cache settings.
    pub fn from_gguf_with_kv_cache_config<P: AsRef<Path>>(
        path: P,
        device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let content = gguf_file::Content::read(&mut file)?;
        if device.same_device(gate_device)
            && device.same_device(up_device)
            && device.same_device(down_device)
            && matches!(kv_cache_quantization, KvCacheQuantization::F16)
        {
            return Self::from_gguf(content, &mut file, device);
        }
        Self::from_gguf_with_devices(
            content,
            &mut file,
            path.as_ref(),
            device,
            gate_device,
            up_device,
            down_device,
            kv_cache_quantization,
        )
    }

    /// Load model from GGUF with device offload mode.
    pub fn from_gguf_with_offload_mode<P: AsRef<Path>>(
        path: P,
        device: &Device,
        offload_mode: DeviceOffloadMode,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        Self::from_gguf_with_offload_and_yarn(
            path,
            device,
            offload_mode,
            kv_cache_quantization,
            None,
        )
    }

    /// Load model from GGUF with device offloading, KV cache quantization, and optional YARN config.
    pub fn from_gguf_with_offload_and_yarn<P: AsRef<Path>>(
        path: P,
        device: &Device,
        offload_mode: DeviceOffloadMode,
        kv_cache_quantization: KvCacheQuantization,
        yarn_config: Option<YarnConfig>,
    ) -> Result<Self> {
        let (gate_device, up_device, down_device) = offload_mode.get_expert_devices(device);
        let mut file = std::fs::File::open(path.as_ref())?;
        let ct = gguf_file::Content::read(&mut file)?;
        Self::from_gguf_with_devices_and_yarn(
            ct,
            &mut file,
            path.as_ref(),
            device,
            &gate_device,
            &up_device,
            &down_device,
            kv_cache_quantization,
            yarn_config,
        )
    }

    /// Load model from GGUF with multi-GPU layer parallelism.
    pub fn from_gguf_with_layer_split<P: AsRef<Path>>(
        path: P,
        layer_device_map: LayerDeviceMap,
        offload_mode: DeviceOffloadMode,
        kv_cache_quantization: KvCacheQuantization,
        yarn_config: Option<YarnConfig>,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let ct = gguf_file::Content::read(&mut file)?;
        let primary_device = layer_device_map.primary_device().clone();
        let mut gg = Gguf::new(ct, file, primary_device.clone(), primary_device.clone());
        let mut config = Self::read_model_config(&gg)?;

        if yarn_config.is_some() {
            config.yarn_config = yarn_config;
        }

        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(
            embed_tensor.dequantize(&primary_device)?,
            config.hidden_size,
        );

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            config.rope_interleaved,
            config.yarn_config.as_ref(),
            &primary_device,
        )?);

        let layers = Self::build_layers(
            &mut gg,
            &config,
            rotary.clone(),
            kv_cache_quantization,
            &layer_device_map,
            |gg, prefix, layer_device, layer_idx| {
                let (gate_dev, up_dev, down_dev) = offload_mode.get_expert_devices(layer_device);
                Self::build_moe_block_with_devices(
                    gg,
                    prefix,
                    &config,
                    path.as_ref(),
                    &gate_dev,
                    &up_dev,
                    &down_dev,
                    layer_device,
                    layer_idx,
                )
            },
        )?;

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head = gg
            .typed_qmatmul::<LmHeadShape>("output.weight")
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embd.weight"))
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embdd.weight"))?;

        let last_layer_device = layer_device_map
            .device_for_layer(config.num_layers.saturating_sub(1))
            .clone();
        let _prev = gg.set_device(last_layer_device.clone());
        let mtp_rotary = if !last_layer_device.same_device(&primary_device) {
            Arc::new(RotaryEmbedding::new(
                config.dtype,
                config.head_dim,
                config.n_rot,
                config.max_position_embeddings,
                config.rope_freq_base,
                config.rope_interleaved,
                config.yarn_config.as_ref(),
                &last_layer_device,
            )?)
        } else {
            rotary.clone()
        };
        let mtp_head = MtpHead::new(
            &mut gg,
            config.head_dim,
            config.rms_norm_eps,
            mtp_rotary,
            config.num_experts,
            config.num_experts_per_tok,
            config.dtype,
        )?;

        let shared_tensors = gg.take_shared_tensors();
        let device = primary_device;
        let span = tracing::span!(tracing::Level::TRACE, "qwen3-next");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        tracing::info!(
            "Multi-GPU layer split: {} GPUs, {} layers",
            layer_device_map.num_gpus(),
            config.num_layers
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: config.dtype,
            span,
            span_output,
            layer_device_map,
            prefetch_pipeline: None,
            shared_tensors,
            last_expert_indices: None,
            capture_expert_indices: false,
            mtp_head,
            decode_workspace: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn from_gguf_with_devices_and_yarn<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        path: &Path,
        device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
        kv_cache_quantization: KvCacheQuantization,
        yarn_override: Option<YarnConfig>,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone(), device.clone());
        let mut config = Self::read_model_config(&gg)?;

        if yarn_override.is_some() {
            config.yarn_config = yarn_override;
        }

        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, config.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            config.rope_interleaved,
            config.yarn_config.as_ref(),
            device,
        )?);

        let layers = Self::build_layers(
            &mut gg,
            &config,
            rotary.clone(),
            kv_cache_quantization,
            &LayerDeviceMap::single(device.clone()),
            |gg, prefix, compute_device, layer_idx| {
                Self::build_moe_block_with_devices(
                    gg,
                    prefix,
                    &config,
                    path,
                    gate_device,
                    up_device,
                    down_device,
                    compute_device,
                    layer_idx,
                )
            },
        )?;

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head = gg
            .typed_qmatmul::<LmHeadShape>("output.weight")
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embd.weight"))
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embdd.weight"))?;
        log_typed_qmatmul_shape("lm_head", &lm_head);

        let mtp_head = MtpHead::new(
            &mut gg,
            config.head_dim,
            config.rms_norm_eps,
            rotary.clone(),
            config.num_experts,
            config.num_experts_per_tok,
            config.dtype,
        )?;

        let shared_tensors = gg.take_shared_tensors();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: config.dtype,
            span: tracing::span!(tracing::Level::TRACE, "qwen3-next"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
            layer_device_map: LayerDeviceMap::single(device.clone()),
            prefetch_pipeline: None,
            shared_tensors,
            last_expert_indices: None,
            capture_expert_indices: false,
            mtp_head,
            decode_workspace: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn from_gguf_with_devices<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        path: &Path,
        device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone(), device.clone());
        let config = Self::read_model_config(&gg)?;
        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, config.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            config.rope_interleaved,
            config.yarn_config.as_ref(),
            device,
        )?);

        let layers = Self::build_layers(
            &mut gg,
            &config,
            rotary.clone(),
            kv_cache_quantization,
            &LayerDeviceMap::single(device.clone()),
            |gg, prefix, compute_device, layer_idx| {
                Self::build_moe_block_with_devices(
                    gg,
                    prefix,
                    &config,
                    path,
                    gate_device,
                    up_device,
                    down_device,
                    compute_device,
                    layer_idx,
                )
            },
        )?;

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head = gg
            .typed_qmatmul::<LmHeadShape>("output.weight")
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embd.weight"))
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embdd.weight"))?;
        log_typed_qmatmul_shape("lm_head", &lm_head);

        let mtp_head = MtpHead::new(
            &mut gg,
            config.head_dim,
            config.rms_norm_eps,
            rotary.clone(),
            config.num_experts,
            config.num_experts_per_tok,
            config.dtype,
        )?;

        let shared_tensors = gg.take_shared_tensors();
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: config.dtype,
            span,
            span_output,
            layer_device_map: LayerDeviceMap::single(device.clone()),
            prefetch_pipeline: None,
            shared_tensors,
            last_expert_indices: None,
            capture_expert_indices: false,
            mtp_head,
            decode_workspace: None,
        })
    }

    pub fn from_gguf_file<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        Self::from_gguf_with_kv_cache_config(
            path,
            device,
            device,
            device,
            device,
            KvCacheQuantization::F16,
        )
    }

    pub fn from_gguf_with_yarn<P: AsRef<Path>>(
        path: P,
        device: &Device,
        yarn_config: Option<YarnConfig>,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let ct = gguf_file::Content::read(&mut file)?;
        let gg = Gguf::new(ct, file, device.clone(), device.clone());
        let mut config = Self::read_model_config(&gg)?;

        if yarn_config.is_some() {
            config.yarn_config = yarn_config;
        }

        Self::from_gguf_with_config_inner(gg, config, device, KvCacheQuantization::F16)
    }

    fn from_gguf_with_config_inner<R: Read + Seek>(
        mut gg: Gguf<R>,
        config: ModelConfig,
        device: &Device,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, config.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            config.rope_interleaved,
            config.yarn_config.as_ref(),
            device,
        )?);

        let layers = Self::build_layers(
            &mut gg,
            &config,
            rotary.clone(),
            kv_cache_quantization,
            &LayerDeviceMap::single(device.clone()),
            |gg, prefix, compute_device, layer_idx| {
                MoeBlock::new(
                    gg,
                    prefix,
                    config.num_experts,
                    config.num_experts_per_tok,
                    config.dtype,
                    compute_device,
                    0,
                    layer_idx,
                )
            },
        )?;

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head = gg
            .typed_qmatmul::<LmHeadShape>("output.weight")
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embd.weight"))
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embdd.weight"))?;
        log_typed_qmatmul_shape("lm_head", &lm_head);

        let mtp_head = MtpHead::new(
            &mut gg,
            config.head_dim,
            config.rms_norm_eps,
            rotary.clone(),
            config.num_experts,
            config.num_experts_per_tok,
            config.dtype,
        )?;

        let shared_tensors = gg.take_shared_tensors();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: config.dtype,
            span: tracing::span!(tracing::Level::TRACE, "qwen3-next"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
            layer_device_map: LayerDeviceMap::single(device.clone()),
            prefetch_pipeline: None,
            shared_tensors,
            last_expert_indices: None,
            capture_expert_indices: false,
            mtp_head,
            decode_workspace: None,
        })
    }

    pub fn from_gguf_for_training<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        Self::from_gguf_for_training_with_offload(
            path,
            device,
            DeviceOffloadMode::ExpertsOnCpu,
            KvCacheQuantization::Q8_0,
        )
    }

    pub fn from_gguf_for_training_with_offload<P: AsRef<Path>>(
        path: P,
        device: &Device,
        offload_mode: DeviceOffloadMode,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let content = gguf_file::Content::read(&mut file)?;
        let (expert_device, _, _) = offload_mode.get_expert_devices(device);
        Self::from_gguf_training_inner(
            content,
            &mut file,
            device,
            &expert_device,
            kv_cache_quantization,
        )
    }

    fn from_gguf_training_inner<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        expert_device: &Device,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone(), expert_device.clone());
        let config = Self::read_model_config(&gg)?;

        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, config.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            config.rope_interleaved,
            config.yarn_config.as_ref(),
            device,
        )?);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("blk.{}", i);

            let attn_norm = gg.rms_norm(
                &format!("{}.attn_norm.weight", prefix),
                config.rms_norm_eps,
                config.dtype,
            )?;
            let ffn_norm = gg
                .rms_norm(
                    &format!("{}.ffn_norm.weight", prefix),
                    config.rms_norm_eps,
                    config.dtype,
                )
                .or_else(|_| {
                    gg.rms_norm(
                        &format!("{}.post_attention_norm.weight", prefix),
                        config.rms_norm_eps,
                        config.dtype,
                    )
                })?;

            let is_recurrent = config.recurrent_layers.get(i).copied().unwrap_or(false)
                || gg
                    .try_tensor(&format!("{}.ssm_in.weight", prefix))?
                    .is_some()
                || gg.try_tensor(&format!("{}.ssm_in", prefix))?.is_some()
                || gg.try_tensor(&format!("{}.ssm_a", prefix))?.is_some();

            let attn = if is_recurrent {
                AttentionLayer::Linear(LinearAttention::new(
                    &mut gg,
                    &prefix,
                    config.ssm_d_inner,
                    config.ssm_d_state,
                    config.ssm_n_groups,
                    config.ssm_dt_rank,
                    config.hidden_size,
                    config.rms_norm_eps,
                    config.dtype,
                    config.linear_v_heads_tiled_order,
                )?)
            } else {
                AttentionLayer::Full(FullAttention::new(
                    &mut gg,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.rms_norm_eps,
                    rotary.clone(),
                    &prefix,
                    config.dtype,
                    kv_cache_quantization,
                    config.max_position_embeddings,
                )?)
            };

            let moe_block = MoeBlock::new(
                &mut gg,
                &prefix,
                config.num_experts,
                config.num_experts_per_tok,
                config.dtype,
                device,
                0,
                i,
            )?;

            layers.push(LayerWeights {
                attn,
                moe_block,
                attn_norm,
                ffn_norm,
            });
        }

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head = gg
            .typed_qmatmul::<LmHeadShape>("output.weight")
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embd.weight"))
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embdd.weight"))?;
        log_typed_qmatmul_shape("lm_head", &lm_head);

        let mtp_head = MtpHead::new(
            &mut gg,
            config.head_dim,
            config.rms_norm_eps,
            rotary.clone(),
            config.num_experts,
            config.num_experts_per_tok,
            config.dtype,
        )?;

        let shared_tensors = gg.take_shared_tensors();
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: config.dtype,
            span,
            span_output,
            layer_device_map: LayerDeviceMap::single(device.clone()),
            prefetch_pipeline: None,
            shared_tensors,
            last_expert_indices: None,
            capture_expert_indices: false,
            mtp_head,
            decode_workspace: None,
        })
    }

    pub fn from_gguf_for_training_with_layer_split<P: AsRef<Path>>(
        path: P,
        layer_device_map: LayerDeviceMap,
        offload_mode: DeviceOffloadMode,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let ct = gguf_file::Content::read(&mut file)?;

        let primary_device = layer_device_map.primary_device().clone();
        let (expert_device, _, _) = offload_mode.get_expert_devices(&primary_device);
        let mut gg = Gguf::new(ct, &mut file, primary_device.clone(), expert_device);

        let config = Self::read_model_config(&gg)?;

        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(
            embed_tensor.dequantize(&primary_device)?,
            config.hidden_size,
        );

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            config.rope_interleaved,
            config.yarn_config.as_ref(),
            &primary_device,
        )?);

        let rotary_map: HashMap<String, Arc<RotaryEmbedding>> = if layer_device_map.is_multi_gpu() {
            let mut map = HashMap::new();
            for i in 0..config.num_layers {
                let dev = layer_device_map.device_for_layer(i);
                let key = format!("{:?}", dev);
                if let std::collections::hash_map::Entry::Vacant(e) = map.entry(key) {
                    let r = Arc::new(RotaryEmbedding::new(
                        config.dtype,
                        config.head_dim,
                        config.n_rot,
                        config.max_position_embeddings,
                        config.rope_freq_base,
                        config.rope_interleaved,
                        config.yarn_config.as_ref(),
                        dev,
                    )?);
                    e.insert(r);
                }
            }
            map
        } else {
            HashMap::new()
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("blk.{}", i);
            let layer_device = layer_device_map.device_for_layer(i);

            let prev_device = if layer_device_map.is_multi_gpu() {
                let (expert_dev, _, _) = offload_mode.get_expert_devices(layer_device);
                gg.set_expert_device(expert_dev);
                Some(gg.set_device(layer_device.clone()))
            } else {
                None
            };

            let layer_rotary = if layer_device_map.is_multi_gpu() {
                let key = format!("{:?}", layer_device);
                rotary_map
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| rotary.clone())
            } else {
                rotary.clone()
            };

            let attn_norm = gg.rms_norm(
                &format!("{}.attn_norm.weight", prefix),
                config.rms_norm_eps,
                config.dtype,
            )?;
            let ffn_norm = gg
                .rms_norm(
                    &format!("{}.ffn_norm.weight", prefix),
                    config.rms_norm_eps,
                    config.dtype,
                )
                .or_else(|_| {
                    gg.rms_norm(
                        &format!("{}.post_attention_norm.weight", prefix),
                        config.rms_norm_eps,
                        config.dtype,
                    )
                })?;

            let is_recurrent = config.recurrent_layers.get(i).copied().unwrap_or(false)
                || gg
                    .try_tensor(&format!("{}.ssm_in.weight", prefix))?
                    .is_some()
                || gg.try_tensor(&format!("{}.ssm_in", prefix))?.is_some()
                || gg.try_tensor(&format!("{}.ssm_a", prefix))?.is_some();

            let attn = if is_recurrent {
                AttentionLayer::Linear(LinearAttention::new(
                    &mut gg,
                    &prefix,
                    config.ssm_d_inner,
                    config.ssm_d_state,
                    config.ssm_n_groups,
                    config.ssm_dt_rank,
                    config.hidden_size,
                    config.rms_norm_eps,
                    config.dtype,
                    config.linear_v_heads_tiled_order,
                )?)
            } else {
                AttentionLayer::Full(FullAttention::new(
                    &mut gg,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.rms_norm_eps,
                    layer_rotary,
                    &prefix,
                    config.dtype,
                    kv_cache_quantization,
                    config.max_position_embeddings,
                )?)
            };

            let moe_block = MoeBlock::new(
                &mut gg,
                &prefix,
                config.num_experts,
                config.num_experts_per_tok,
                config.dtype,
                layer_device,
                0,
                i,
            )?;

            layers.push(LayerWeights {
                attn,
                moe_block,
                attn_norm,
                ffn_norm,
            });

            if let Some(prev) = prev_device {
                gg.set_device(prev);
            }
        }

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head = gg
            .typed_qmatmul::<LmHeadShape>("output.weight")
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embd.weight"))
            .or_else(|_| gg.typed_qmatmul::<LmHeadShape>("token_embdd.weight"))?;

        let last_layer_device = layer_device_map.device_for_layer(config.num_layers - 1);
        let last_layer_rotary = if layer_device_map.is_multi_gpu() {
            let key = format!("{:?}", last_layer_device);
            rotary_map
                .get(&key)
                .cloned()
                .unwrap_or_else(|| rotary.clone())
        } else {
            rotary.clone()
        };
        if layer_device_map.is_multi_gpu() {
            let (expert_dev, _, _) = offload_mode.get_expert_devices(last_layer_device);
            gg.set_expert_device(expert_dev);
            gg.set_device(last_layer_device.clone());
        }
        let mtp_head = MtpHead::new(
            &mut gg,
            config.head_dim,
            config.rms_norm_eps,
            last_layer_rotary,
            config.num_experts,
            config.num_experts_per_tok,
            config.dtype,
        )?;

        let shared_tensors = gg.take_shared_tensors();
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: primary_device,
            dtype: config.dtype,
            span,
            span_output,
            layer_device_map,
            prefetch_pipeline: None,
            shared_tensors,
            last_expert_indices: None,
            capture_expert_indices: false,
            mtp_head,
            decode_workspace: None,
        })
    }

    pub fn quzo_qtensors(&self) -> Vec<(String, SharedQTensor)> {
        self.shared_tensors.clone()
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    fn batched_causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        padding_lengths: &[usize],
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let kv_len = tgt + offset;
        let mut mask = Vec::with_capacity(b * tgt * kv_len);
        for &pad_len in padding_lengths.iter().take(b) {
            for i in 0..tgt {
                for j in 0..kv_len {
                    let causal_ok = j <= i + offset;
                    let not_padding = j >= pad_len;
                    mask.push(if causal_ok && not_padding { 0. } else { minf });
                }
            }
        }
        Tensor::from_slice(&mask, (b, 1, tgt, kv_len), &self.device)?.to_dtype(self.dtype)
    }

    fn per_device_masks(&self, mask: Option<&Tensor>) -> Result<Vec<(String, Tensor)>> {
        let mut masks = Vec::new();
        if let Some(m) = mask {
            if self.layer_device_map.is_multi_gpu() {
                let mut seen = std::collections::HashSet::new();
                for i in 0..self.layers.len() {
                    let dev = self.layer_device_map.device_for_layer(i);
                    let key = format!("{:?}", dev);
                    if seen.insert(key.clone()) {
                        masks.push((key, m.to_device(dev)?));
                    }
                }
            }
        }
        Ok(masks)
    }

    pub fn enable_gpu_hot_cache(&mut self, capacity: usize) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let gpu_dev = self.layer_device_map.device_for_layer(i);
            if is_gpu_device(gpu_dev) {
                layer
                    .moe_block
                    .experts
                    .enable_gpu_hot_cache(gpu_dev.clone(), capacity);
            }
        }
    }

    pub fn enable_prefetch_pipeline(&mut self) -> Result<()> {
        let first_expert_device = self.layers[0]
            .moe_block
            .experts
            .gate_exps
            .read()
            .unwrap()
            .device();
        if is_gpu_device(&first_expert_device) {
            warn!(
                "Skipping prefetch pipeline - experts are on GPU. Use --offload=experts or --no-prefetch."
            );
            return Ok(());
        }

        let mut gate_weights = Vec::with_capacity(self.layers.len());
        let mut up_weights = Vec::with_capacity(self.layers.len());
        let mut down_weights = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            gate_weights.push(layer.moe_block.experts.gate_exps.as_shared().clone());
            up_weights.push(layer.moe_block.experts.up_exps.as_shared().clone());
            down_weights.push(layer.moe_block.experts.down_exps.as_shared().clone());
        }

        let num_experts = self.layers[0]
            .moe_block
            .experts
            .gate_exps
            .read()
            .unwrap()
            .shape()
            .dims()[0];

        let pipeline = crate::layer_pipeline::PrefetchPipelineCoordinator::new(
            self.device.clone(),
            self.layers.len(),
            gate_weights,
            up_weights,
            down_weights,
            num_experts,
            #[cfg(feature = "vulkan")]
            self.device.as_vulkan_device().ok().cloned(),
        );

        self.prefetch_pipeline = Some(pipeline);
        Ok(())
    }

    pub fn has_prefetch_pipeline(&self) -> bool {
        self.prefetch_pipeline.is_some()
    }

    fn forward_impl(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let t_embed = std::time::Instant::now();
        let h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;
        log_shape("model.embedding", &h);
        let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
        self.forward_with_embeddings_inner(&h, offset, embed_ms)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.forward_impl(input, offset)
    }

    pub fn forward_with_embeddings(
        &mut self,
        embeddings: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        self.forward_with_embeddings_inner(embeddings, offset, 0.0)
    }

    fn forward_with_embeddings_inner(
        &mut self,
        h: &Tensor,
        offset: usize,
        embed_ms: f64,
    ) -> Result<Tensor> {
        let span = self.span.clone();
        let _enter = span.enter();
        let dims = h.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let decode_single = l == 1;
        let profile = std::env::var("PARAMECIA_PROFILE").is_ok();

        let mut h_typed: TypedTensor<Hidden3> = h.clone().contiguous()?.try_into()?;

        if decode_single && self.prefetch_pipeline.is_none() && !self.capture_expert_indices {
            return self.forward_decode_with_workspace(h_typed, offset, embed_ms, profile);
        }

        let causal_mask = if decode_single {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        let device_masks = if decode_single {
            Vec::new()
        } else {
            self.per_device_masks(causal_mask.as_ref())?
        };

        let t_layers = std::time::Instant::now();
        if self.prefetch_pipeline.is_some() {
            h_typed = self.forward_prefetch_pipelined(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                offset,
            )?;
        } else if self.capture_expert_indices {
            let (out, all_stats) = self.run_layer_loop(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                offset,
                LayerForwardMode::WithStats,
            )?;
            h_typed = out;
            let stats = all_stats.ok_or_else(|| {
                paramecia_core::Error::Msg(
                    "run_layer_loop WithStats returned no router stats".to_string(),
                )
            })?;
            let expert_indices: Vec<Tensor> = stats
                .into_iter()
                .map(|(_router_logits, indices)| Self::normalize_captured_expert_indices(indices))
                .collect::<Result<Vec<_>>>()?;
            self.last_expert_indices = Some(expert_indices);
        } else {
            let (out, _) = self.run_layer_loop(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                offset,
                LayerForwardMode::Normal,
            )?;
            h_typed = out;
        }
        let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;

        let t_head = std::time::Instant::now();
        log_shape("model.pre_norm", h_typed.inner());
        let tail = Then::new(
            norm_hidden_flow(&self.norm),
            LiftResult::new(MapErr::new(output_head_at_op(&self.lm_head, l - 1))),
        );
        let mut head_flow = Then::new(
            to_primary_flow(&self.layer_device_map),
            LiftResult::new(tail),
        );
        let span_output = self.span_output.clone();
        let _enter_output = span_output.enter();
        let logits = head_flow.traced_forward(&mut (), h_typed)?.into_inner();
        log_shape("model.logits", &logits);
        #[cfg(feature = "vulkan")]
        if profile {
            if let Device::Vulkan(vk) = logits.device() {
                vk.flush()?;
            }
        }
        let head_ms = t_head.elapsed().as_secs_f64() * 1000.0;

        if profile {
            trace!(
                embed_ms = format_args!("{:.1}", embed_ms),
                layers_ms = format_args!("{:.1}", layers_ms),
                head_ms = format_args!("{:.1}", head_ms),
                total_ms = format_args!("{:.1}", embed_ms + layers_ms + head_ms),
                "Forward pass profile (prefetch)"
            );
        }

        Ok(logits)
    }

    pub fn forward_chunked(
        &mut self,
        input: &Tensor,
        offset: usize,
        chunk_size: Option<usize>,
    ) -> Result<Tensor> {
        let span = self.span.clone();
        let _enter = span.enter();
        let dims = input.dims();
        let l = dims.get(1).copied().unwrap_or(0);

        const DEFAULT_PREFILL_CHUNK_SIZE: usize = 512;
        let chunk_size = chunk_size.unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE);

        if l <= chunk_size {
            return self.forward(input, offset);
        }

        let mut current_offset = offset;
        let mut chunk_start = 0;

        while chunk_start < l {
            let chunk_end = (chunk_start + chunk_size).min(l);
            let chunk_len = chunk_end - chunk_start;
            let is_last_chunk = chunk_end == l;

            let chunk_input = input.narrow(1, chunk_start, chunk_len)?;

            if is_last_chunk {
                let logits = self.forward(&chunk_input, current_offset)?;
                return Ok(logits);
            } else {
                self.forward_state_update_only(&chunk_input, current_offset)?;
            }

            current_offset += chunk_len;
            chunk_start = chunk_end;
        }

        paramecia_core::bail!("forward_chunked: unexpected empty sequence")
    }

    fn forward_prefetch_pipelined(
        &mut self,
        initial_hidden: TypedTensor<Hidden3>,
        causal_mask: Option<&Tensor>,
        device_masks: &[(String, Tensor)],
        offset: usize,
    ) -> Result<TypedTensor<Hidden3>> {
        let pipeline = self.prefetch_pipeline.as_ref().ok_or_else(|| {
            paramecia_core::Error::Msg("Prefetch pipeline not enabled".to_string())
        })?;

        let num_layers = self.layers.len();
        let (_batch_size, _seq_len, _hidden_dim) = initial_hidden.inner().dims3()?;
        let activation_dtype = initial_hidden.inner().dtype();
        let is_multi = self.layer_device_map.is_multi_gpu();
        let capture_experts = self.capture_expert_indices;
        let init_state = PrefetchPipelineState {
            h: initial_hidden,
            pending_moe: None,
            pending_residual: None,
            expert_indices: if capture_experts {
                Vec::with_capacity(num_layers)
            } else {
                Vec::new()
            },
        };
        let mut step_ctx = PrefetchPipelineStepCtx {
            layers: &mut self.layers,
            pipeline,
            is_multi,
            layer_device_map: &self.layer_device_map,
            device_masks,
            causal_mask,
            offset,
            activation_dtype,
            num_layers,
            capture_experts,
        };
        let step: PrefetchPipelineStep = prefetch_pipeline_step_op();
        let mut fold = TryFoldRange::new(step);
        let mut state = fold.traced_forward(&mut step_ctx, (init_state, 0..num_layers))?;

        if state.pending_moe.is_some() || state.pending_residual.is_some() {
            paramecia_core::bail!("prefetch pipeline ended with unresolved pending MoE state");
        }

        if capture_experts {
            self.last_expert_indices = Some(state.expert_indices);
        }

        if is_multi {
            state.h = transfer_to(state.h.into_inner(), self.layer_device_map.primary_device())?
                .try_into()?;
        }

        Ok(state.h)
    }

    pub fn forward_all_positions(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let span = self.span.clone();
        let _enter = span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;
        let mut h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;
        log_shape("all_pos.embedding", h_typed.inner());

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        let device_masks = self.per_device_masks(causal_mask.as_ref())?;

        if self.prefetch_pipeline.is_some() {
            h_typed = self.forward_prefetch_pipelined(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                offset,
            )?;
        } else {
            let (out, _) = self.run_layer_loop(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                offset,
                LayerForwardMode::Normal,
            )?;
            h_typed = out;
        }

        h_typed = self.transfer_to_primary_typed(h_typed)?;
        log_shape("all_pos.pre_norm", h_typed.inner());
        let h: TypedTensor<Hidden3> = self.norm.forward(h_typed.inner())?.try_into()?;
        log_shape("all_pos.post_norm", h.inner());
        let span_output = self.span_output.clone();
        let _enter = span_output.enter();

        let logits = self.output_head_all(&h)?;
        log_shape("all_pos.logits", &logits);
        Ok(logits)
    }

    pub fn verify_with_state_materialization(
        &mut self,
        input: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let span = self.span.clone();
        let _enter = span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);

        let h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;
        let mut h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;
        log_shape("verify.embedding", h_typed.inner());

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        let device_masks = self.per_device_masks(causal_mask.as_ref())?;

        let (out, _) = self.run_layer_loop(
            h_typed,
            causal_mask.as_ref(),
            &device_masks,
            offset,
            LayerForwardMode::WithStateMaterialization,
        )?;
        h_typed = out;

        let all_hidden = h_typed.inner().clone();
        log_shape("verify.all_hidden", &all_hidden);

        h_typed = self.transfer_to_primary_typed(h_typed)?;
        let h_normed: TypedTensor<Hidden3> = self.norm.forward(h_typed.inner())?.try_into()?;
        log_shape("verify.post_norm", h_normed.inner());
        let _enter = self.span_output.enter();

        let logits = self.output_head_all(&h_normed)?;
        log_shape("verify.logits", &logits);

        Ok((logits, all_hidden))
    }

    pub fn restore_to_intermediate_state(&mut self, index: usize, offset: usize) {
        if self.layer_device_map.is_multi_gpu() {
            let num_layers = self.layers.len();
            let mut last_key = String::new();
            for i in 0..num_layers {
                let dev = self.layer_device_map.device_for_layer(i);
                let key = format!("{:?}", dev);
                if key != last_key {
                    let _ctx = Tensor::zeros(1, paramecia_core::DType::U8, dev);
                    last_key = key;
                }
                self.layers[i].restore_to_intermediate_state(index, offset);
            }
        } else {
            for layer in &mut self.layers {
                layer.restore_to_intermediate_state(index, offset);
            }
        }
    }

    pub fn clear_intermediate_states(&mut self) {
        if self.layer_device_map.is_multi_gpu() {
            let num_layers = self.layers.len();
            let mut last_key = String::new();
            for i in 0..num_layers {
                let dev = self.layer_device_map.device_for_layer(i);
                let key = format!("{:?}", dev);
                if key != last_key {
                    let _ctx = Tensor::zeros(1, paramecia_core::DType::U8, dev);
                    last_key = key;
                }
                self.layers[i].clear_intermediate_states();
            }
        } else {
            for layer in &mut self.layers {
                layer.clear_intermediate_states();
            }
        }
    }

    pub fn forward_embeddings(&mut self, input: &Tensor) -> Result<Tensor> {
        let span = self.span.clone();
        let _enter = span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;
        let h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, 0)?)
        };
        let device_masks = self.per_device_masks(causal_mask.as_ref())?;

        let (h, _) = self.run_layer_loop(
            h_typed,
            causal_mask.as_ref(),
            &device_masks,
            0,
            LayerForwardMode::Normal,
        )?;
        let h = self.transfer_to_primary_typed(h)?.into_inner();
        let h = self.norm.forward(&h)?;

        let embeddings = h.mean(1)?;
        Ok(embeddings)
    }

    pub fn forward_embeddings_last(&mut self, input: &Tensor) -> Result<Tensor> {
        let span = self.span.clone();
        let _enter = span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;
        let h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, 0)?)
        };
        let device_masks = self.per_device_masks(causal_mask.as_ref())?;

        let (h, _) = self.run_layer_loop(
            h_typed,
            causal_mask.as_ref(),
            &device_masks,
            0,
            LayerForwardMode::Normal,
        )?;
        let h = self.transfer_to_primary_typed(h)?.into_inner();
        let h = self.norm.forward(&h)?;

        let embeddings = h.narrow(1, l - 1, 1)?.squeeze(1)?;
        Ok(embeddings)
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    pub fn set_kv_cache_quantization(&mut self, quant: KvCacheQuantization) {
        for layer in &mut self.layers {
            if let AttentionLayer::Full(ref mut attn) = layer.attn {
                attn.kv_cache_quantization = quant;
                attn.preallocated_cache = None;
                attn.quantized_cache = None;
                attn.kv_cache = None;
            }
        }
    }

    pub fn truncate_cache(&mut self, new_len: usize) {
        for layer in &mut self.layers {
            layer.truncate_cache(new_len);
        }
    }

    pub fn snapshot_cache(&mut self) -> Result<Vec<LayerSnapshot>> {
        self.layers
            .iter_mut()
            .map(|layer| layer.snapshot_cache())
            .collect()
    }

    pub fn restore_cache(&mut self, snapshots: Vec<LayerSnapshot>) {
        for (layer, snapshot) in self.layers.iter_mut().zip(snapshots) {
            layer.restore_cache(snapshot);
        }
    }

    pub fn forward_training(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let span = self.span.clone();
        let _enter = span.enter();
        let (b, l) = input_ids.dims2()?;
        let h = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        let h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;
        log_shape("train.embedding", h_typed.inner());

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, seqlen_offset)?)
        };

        let device_masks = self.per_device_masks(causal_mask.as_ref())?;

        if self.prefetch_pipeline.is_some() {
            return self.forward_training_pipelined(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                seqlen_offset,
            );
        }

        let (h, all_stats) = self.run_layer_loop(
            h_typed,
            causal_mask.as_ref(),
            &device_masks,
            seqlen_offset,
            LayerForwardMode::WithStats,
        )?;
        let all_router_stats = all_stats.ok_or_else(|| {
            paramecia_core::Error::Msg(
                "run_layer_loop WithStats returned no router stats".to_string(),
            )
        })?;
        let h = self.transfer_to_primary_typed(h)?;

        log_shape("train.pre_norm", h.inner());
        let h: TypedTensor<Hidden3> = self.norm.forward(h.inner())?.try_into()?;
        log_shape("train.post_norm", h.inner());
        let span_output = self.span_output.clone();
        let _enter2 = span_output.enter();
        let logits = self.output_head_all(&h)?;
        log_shape("train.logits", &logits);

        Ok((logits, all_router_stats))
    }

    fn forward_training_pipelined(
        &mut self,
        h: TypedTensor<Hidden3>,
        causal_mask: Option<&Tensor>,
        device_masks: &[(String, Tensor)],
        offset: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let pipeline = self.prefetch_pipeline.as_ref().ok_or_else(|| {
            paramecia_core::Error::Msg("Prefetch pipeline not enabled".to_string())
        })?;

        let num_layers = self.layers.len();
        let (_batch_size, _seq_len, _hidden_dim) = h.inner().dims3()?;
        let activation_dtype = h.inner().dtype();
        let is_multi = self.layer_device_map.is_multi_gpu();
        let init_state = TrainPipelineState {
            h,
            pending_moe: None,
            pending_residual: None,
            all_router_stats: Vec::new(),
        };
        let mut step_ctx = TrainPipelineStepCtx {
            layers: &mut self.layers,
            pipeline,
            is_multi,
            layer_device_map: &self.layer_device_map,
            device_masks,
            causal_mask,
            offset,
            activation_dtype,
            num_layers,
        };
        let step: TrainPipelineStep = train_pipeline_step_op();
        let mut fold = TryFoldRange::new(step);
        let mut state = fold.traced_forward(&mut step_ctx, (init_state, 0..num_layers))?;

        if state.pending_moe.is_some() || state.pending_residual.is_some() {
            paramecia_core::bail!("training pipeline ended with unresolved pending MoE state");
        }

        if is_multi {
            state.h = transfer_to(state.h.into_inner(), self.layer_device_map.primary_device())?
                .try_into()?;
        }
        let h = state.h;
        let all_router_stats = state.all_router_stats;

        log_shape("train_pipe.pre_norm", h.inner());
        let h: TypedTensor<Hidden3> = self.norm.forward(h.inner())?.try_into()?;
        log_shape("train_pipe.post_norm", h.inner());
        let logits = self.output_head_all(&h)?;
        log_shape("train_pipe.logits", &logits);
        Ok((logits, all_router_stats))
    }

    pub fn forward_state_update_only(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<()> {
        let span = self.span.clone();
        let _enter = span.enter();
        let (b, l) = input_ids.dims2()?;
        let h = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        let h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, seqlen_offset)?)
        };

        let device_masks = self.per_device_masks(causal_mask.as_ref())?;

        if self.prefetch_pipeline.is_some() {
            let _ = self.forward_training_pipelined(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                seqlen_offset,
            )?;
            return Ok(());
        }

        let _ = self.run_layer_loop(
            h_typed,
            causal_mask.as_ref(),
            &device_masks,
            seqlen_offset,
            LayerForwardMode::Normal,
        )?;

        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub fn forward_training_with_mtp(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        num_depths: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>, Vec<Tensor>)> {
        let span = self.span.clone();
        let _enter = span.enter();
        let (b, l) = input_ids.dims2()?;

        let token_embeds = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        let h_typed: TypedTensor<Hidden3> = token_embeds.clone().contiguous()?.try_into()?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, seqlen_offset)?)
        };
        let device_masks = self.per_device_masks(causal_mask.as_ref())?;
        let is_multi = self.layer_device_map.is_multi_gpu();
        let num_layers = self.layers.len();

        let (h, all_stats) = self.run_layer_loop(
            h_typed,
            causal_mask.as_ref(),
            &device_masks,
            seqlen_offset,
            LayerForwardMode::WithStats,
        )?;
        let all_router_stats = all_stats.ok_or_else(|| {
            paramecia_core::Error::Msg(
                "run_layer_loop WithStats returned no router stats".to_string(),
            )
        })?;

        // Capture pre-norm hidden before device transfer (MTP needs it on last layer's device)
        let pre_norm_hidden = h.clone();
        let h = self.transfer_to_primary_typed(h)?;

        let h_normed: TypedTensor<Hidden3> = self.norm.forward(h.inner())?.try_into()?;
        let span_output = self.span_output.clone();
        let _enter2 = span_output.enter();
        let main_logits = self.output_head_all(&h_normed)?;

        let mtp_logits = if let Some(ref mut mtp_head) = self.mtp_head {
            let token_embeds_for_mtp: TypedTensor<Hidden3> = if is_multi {
                transfer_to(
                    token_embeds,
                    self.layer_device_map.device_for_layer(num_layers - 1),
                )?
                .contiguous()?
                .try_into()?
            } else {
                token_embeds.clone().contiguous()?.try_into()?
            };
            let logits = mtp_head.forward_training_batch(
                &pre_norm_hidden,
                &token_embeds_for_mtp,
                &self.lm_head,
                seqlen_offset,
                num_depths,
            )?;
            if is_multi {
                logits
                    .into_iter()
                    .map(|t| transfer_to(t, self.layer_device_map.primary_device()))
                    .collect::<Result<Vec<_>>>()?
            } else {
                logits
            }
        } else {
            Vec::new()
        };

        Ok((main_logits, all_router_stats, mtp_logits))
    }

    #[allow(clippy::type_complexity)]
    pub fn forward_training_with_mtp_weighted(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        weighted_embeds_per_depth: &[Tensor],
        num_depths: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>, Vec<Tensor>)> {
        let span = self.span.clone();
        let _enter = span.enter();
        let (b, l) = input_ids.dims2()?;

        let token_embeds = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        let h_typed: TypedTensor<Hidden3> = token_embeds.contiguous()?.try_into()?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, seqlen_offset)?)
        };
        let device_masks = self.per_device_masks(causal_mask.as_ref())?;
        let is_multi = self.layer_device_map.is_multi_gpu();
        let num_layers = self.layers.len();

        let (h, all_stats) = self.run_layer_loop(
            h_typed,
            causal_mask.as_ref(),
            &device_masks,
            seqlen_offset,
            LayerForwardMode::WithStats,
        )?;
        let all_router_stats = all_stats.ok_or_else(|| {
            paramecia_core::Error::Msg(
                "run_layer_loop WithStats returned no router stats".to_string(),
            )
        })?;

        // Capture pre-norm hidden before device transfer (MTP needs it on last layer's device)
        let pre_norm_hidden = h.clone();
        let h = self.transfer_to_primary_typed(h)?;

        let h_normed: TypedTensor<Hidden3> = self.norm.forward(h.inner())?.try_into()?;
        let span_output = self.span_output.clone();
        let _enter2 = span_output.enter();
        let main_logits = self.output_head_all(&h_normed)?;

        let mtp_logits = if let Some(ref mut mtp_head) = self.mtp_head {
            if !weighted_embeds_per_depth.is_empty() {
                let weighted_embeds_mtp: Vec<TypedTensor<Hidden3>> = if is_multi {
                    let last_dev = self.layer_device_map.device_for_layer(num_layers - 1);
                    weighted_embeds_per_depth
                        .iter()
                        .map(|t| -> Result<TypedTensor<Hidden3>> {
                            let t_rank3 = if t.rank() == 2 {
                                t.unsqueeze(0)?
                            } else if t.rank() == 3 {
                                t.clone()
                            } else {
                                paramecia_core::bail!(
                                    "weighted embed expected rank 2 or 3, got rank {}",
                                    t.rank()
                                );
                            };
                            transfer_to(t_rank3, last_dev)?
                                .contiguous()?
                                .try_into()
                                .map_err(|e: paramecia_tensor::Error| {
                                    paramecia_core::Error::Msg(e.to_string())
                                })
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    weighted_embeds_per_depth
                        .iter()
                        .map(|t| -> Result<TypedTensor<Hidden3>> {
                            let t_rank3 = if t.rank() == 2 {
                                t.unsqueeze(0)?
                            } else if t.rank() == 3 {
                                t.clone()
                            } else {
                                paramecia_core::bail!(
                                    "weighted embed expected rank 2 or 3, got rank {}",
                                    t.rank()
                                );
                            };
                            t_rank3.contiguous()?.try_into().map_err(
                                |e: paramecia_tensor::Error| {
                                    paramecia_core::Error::Msg(e.to_string())
                                },
                            )
                        })
                        .collect::<Result<Vec<_>>>()?
                };
                let logits = mtp_head.forward_training_batch_weighted(
                    &pre_norm_hidden,
                    &weighted_embeds_mtp,
                    &self.lm_head,
                    seqlen_offset,
                    num_depths,
                )?;
                if is_multi {
                    logits
                        .into_iter()
                        .map(|t| transfer_to(t, self.layer_device_map.primary_device()))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    logits
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        Ok((main_logits, all_router_stats, mtp_logits))
    }

    pub fn compute_z_loss(
        &self,
        router_stats: &[(Tensor, Tensor)],
        z_loss_weight: f64,
    ) -> Result<Tensor> {
        if router_stats.is_empty() {
            return Tensor::new(0.0f32, &self.device);
        }

        let mut total_z_loss: Option<Tensor> = None;

        for (router_logits, _) in router_stats {
            let router_logits = router_logits.flatten(0, 1)?;
            let max_val = router_logits.max(D::Minus1)?;
            let max_val_keepdim = max_val.unsqueeze(D::Minus1)?;
            let exp_shifted = router_logits.broadcast_sub(&max_val_keepdim)?.exp()?;
            let sum_exp = exp_shifted.sum(D::Minus1)?;
            let log_sum_exp = (max_val + sum_exp.log()?)?;

            let layer_z_loss = (&log_sum_exp * &log_sum_exp)?.mean_all()?;
            let layer_z_loss = layer_z_loss.to_device(&self.device)?;

            total_z_loss = Some(match total_z_loss {
                Some(acc) => (acc + layer_z_loss)?,
                None => layer_z_loss,
            });
        }

        let z_loss = match total_z_loss {
            Some(z_loss) => z_loss,
            None => Tensor::new(0.0f32, &self.device)?,
        };
        let num_layers = router_stats.len() as f64;
        (z_loss / num_layers)? * z_loss_weight
    }

    pub fn compute_load_balance_loss(
        &self,
        router_stats: &[(Tensor, Tensor)],
        lb_loss_weight: f64,
    ) -> Result<Tensor> {
        if router_stats.is_empty() {
            return Tensor::new(0.0f32, &self.device);
        }

        let num_experts = self.num_experts();
        let mut total_lb_loss: Option<Tensor> = None;

        for (router_logits, selected_experts) in router_stats {
            let router_logits = router_logits.flatten(0, 1)?;
            let selected_experts = selected_experts.flatten(0, 1)?;

            let router_probs = paramecia_nn::ops::softmax_last_dim(&router_logits)?;

            let selected_flat = selected_experts.to_vec2::<u32>()?;
            let mut expert_counts = vec![0.0f32; num_experts];
            for row in &selected_flat {
                for &expert_idx in row {
                    if (expert_idx as usize) < num_experts {
                        expert_counts[expert_idx as usize] += 1.0;
                    }
                }
            }
            let total_selections =
                selected_flat.len() * selected_flat.first().map_or(1, |r| r.len());
            let logits_device = router_logits.device().clone();
            let f_i = Tensor::from_vec(
                expert_counts
                    .iter()
                    .map(|c| c / total_selections as f32)
                    .collect::<Vec<_>>(),
                num_experts,
                &logits_device,
            )?;

            let p_i = router_probs.mean(0)?;

            let lb_loss = (&f_i * &p_i)?.sum_all()?;
            let lb_loss = (lb_loss * (num_experts as f64))?;
            let lb_loss = lb_loss.to_device(&self.device)?;

            total_lb_loss = Some(match total_lb_loss {
                Some(acc) => (acc + lb_loss)?,
                None => lb_loss,
            });
        }

        let lb_loss = match total_lb_loss {
            Some(lb_loss) => lb_loss,
            None => Tensor::new(0.0f32, &self.device)?,
        };
        let num_layers = router_stats.len() as f64;
        (lb_loss / num_layers)? * lb_loss_weight
    }

    pub fn compute_auxiliary_loss(
        &self,
        router_stats: &[(Tensor, Tensor)],
        z_loss_weight: f64,
        lb_loss_weight: f64,
    ) -> Result<Tensor> {
        let z_loss = self.compute_z_loss(router_stats, z_loss_weight)?;
        let lb_loss = self.compute_load_balance_loss(router_stats, lb_loss_weight)?;
        z_loss + lb_loss
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    pub fn reinit_caches(&mut self, batch_size: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.clear_cache();
            if let AttentionLayer::Full(ref mut attn) = layer.attn {
                if let Some(ref mut cache) = attn.preallocated_cache {
                    cache.resize_batch(batch_size, self.dtype)?;
                }
                if let Some(ref mut cache) = attn.quantized_cache {
                    cache.resize_batch(batch_size)?;
                }
            }
        }
        Ok(())
    }

    pub fn forward_batched(
        &mut self,
        input: &Tensor,
        offset: usize,
        padding_lengths: &[usize],
    ) -> Result<Tensor> {
        let t_embed = std::time::Instant::now();
        let h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;
        let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
        self.forward_with_embeddings_batched(&h, offset, embed_ms, padding_lengths)
    }

    fn forward_with_embeddings_batched(
        &mut self,
        h: &Tensor,
        offset: usize,
        embed_ms: f64,
        padding_lengths: &[usize],
    ) -> Result<Tensor> {
        let span = self.span.clone();
        let _enter = span.enter();
        let dims = h.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let profile = std::env::var("PARAMECIA_PROFILE").is_ok();

        let h = if l > 1 {
            let mut mask_data = vec![1.0f32; b * l];
            for (i, &pad_len) in padding_lengths.iter().enumerate() {
                for j in 0..pad_len {
                    mask_data[i * l + j] = 0.0;
                }
            }
            let pad_mask =
                Tensor::from_slice(&mask_data, (b, l, 1), h.device())?.to_dtype(self.dtype)?;
            h.broadcast_mul(&pad_mask)?
        } else {
            h.clone()
        };
        let mut h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;

        let causal_mask = if l == 1 && padding_lengths.iter().all(|&p| p == 0) {
            None
        } else {
            Some(self.batched_causal_mask(b, l, offset, padding_lengths)?)
        };

        let device_masks = self.per_device_masks(causal_mask.as_ref())?;
        let is_multi = self.layer_device_map.is_multi_gpu();

        let t_layers = std::time::Instant::now();
        if self.prefetch_pipeline.is_some() {
            h_typed = self.forward_prefetch_pipelined(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                offset,
            )?;
        } else {
            let (out, _) = self.run_layer_loop(
                h_typed,
                causal_mask.as_ref(),
                &device_masks,
                offset,
                LayerForwardMode::Normal,
            )?;
            h_typed = out;
        }
        let layers_ms = t_layers.elapsed().as_secs_f64() * 1000.0;

        if self.prefetch_pipeline.is_none() && is_multi {
            h_typed = self.transfer_to_primary_typed(h_typed)?;
        }

        let t_head = std::time::Instant::now();
        let h: TypedTensor<Hidden3> = self.norm.forward(h_typed.inner())?.try_into()?;
        let span_output = self.span_output.clone();
        let _enter_output = span_output.enter();
        let logits = self.output_head_at(&h, l - 1)?;
        #[cfg(feature = "vulkan")]
        if profile {
            if let Device::Vulkan(vk) = h.inner().device() {
                vk.flush()?;
            }
        }
        let head_ms = t_head.elapsed().as_secs_f64() * 1000.0;

        if profile {
            trace!(
                embed_ms = format_args!("{:.1}", embed_ms),
                layers_ms = format_args!("{:.1}", layers_ms),
                head_ms = format_args!("{:.1}", head_ms),
                total_ms = format_args!("{:.1}", embed_ms + layers_ms + head_ms),
                "Forward pass profile (batched)"
            );
        }

        Ok(logits)
    }

    pub fn save_prefix_cache(&self, prefix_tokens: Vec<u32>) -> PrefixCache {
        let layer_caches = self
            .layers
            .iter()
            .map(|layer| layer.save_cache_for_prefix())
            .collect();

        PrefixCache {
            prefix_tokens,
            layer_caches,
        }
    }

    pub fn restore_prefix_cache(&mut self, cache: &PrefixCache) -> Result<usize> {
        if cache.layer_caches.len() != self.layers.len() {
            paramecia_core::bail!(
                "Prefix cache layer count mismatch: {} vs {}",
                cache.layer_caches.len(),
                self.layers.len()
            );
        }

        for (layer, entry) in self.layers.iter_mut().zip(cache.layer_caches.iter()) {
            layer.restore_cache_from_prefix(entry)?;
        }

        Ok(cache.prefix_tokens.len())
    }

    pub fn cache_seq_len(&self) -> usize {
        for layer in &self.layers {
            if let AttentionLayer::Full(attn) = &layer.attn {
                return attn.cache_seq_len();
            }
        }
        0
    }

    pub fn all_qtensors(&self) -> Vec<(String, &QTensor)> {
        let mut tensors = Vec::new();

        if let Some(qt) = self.lm_head.qtensor() {
            tensors.push(("output.weight".to_string(), qt));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prefix = format!("blk.{}", layer_idx);

            match &layer.attn {
                AttentionLayer::Full(attn) => {
                    if let Some(qt) = attn.wq.qtensor() {
                        tensors.push((format!("{}.attn_q.weight", prefix), qt));
                    }
                    if let Some(qt) = attn.wk.qtensor() {
                        tensors.push((format!("{}.attn_k.weight", prefix), qt));
                    }
                    if let Some(qt) = attn.wv.qtensor() {
                        tensors.push((format!("{}.attn_v.weight", prefix), qt));
                    }
                    if let Some(qt) = attn.wo.qtensor() {
                        tensors.push((format!("{}.attn_output.weight", prefix), qt));
                    }
                }
                AttentionLayer::Linear(attn) => {
                    let (main_qt, main_name, gate_qt) = attn.ssm_in.qtensors_for_save();
                    if let Some(qt) = main_qt {
                        tensors.push((format!("{}.{}", prefix, main_name), qt));
                    }
                    if let Some(qt) = gate_qt {
                        tensors.push((format!("{}.attn_gate", prefix), qt));
                    }
                    let (ba_qt_0, ba_qt_1) = attn.ssm_beta_alpha.qtensors_for_save();
                    if let Some((name, qt)) = ba_qt_0 {
                        tensors.push((format!("{}.{}", prefix, name), qt));
                    }
                    if let Some((name, qt)) = ba_qt_1 {
                        tensors.push((format!("{}.{}", prefix, name), qt));
                    }
                    if let Some(qt) = attn.ssm_out.qtensor() {
                        tensors.push((format!("{}.ssm_out", prefix), qt));
                    }
                }
            }

            if let Some(qt) = layer.moe_block.gate.qtensor() {
                tensors.push((format!("{}.ffn_gate_inp.weight", prefix), qt));
            }
        }

        tensors
    }

    fn block_count(qtensor: &QTensor) -> usize {
        qtensor.shape().elem_count() / qtensor.dtype().block_size()
    }

    pub fn all_qtensors_with_info(&self) -> Vec<(String, &QTensor, usize)> {
        let mut tensors = Vec::new();

        if let Some(qt) = self.lm_head.qtensor() {
            tensors.push(("lm_head".to_string(), qt, Self::block_count(qt)));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layer_{}", layer_idx);

            match &layer.attn {
                AttentionLayer::Full(attn) => {
                    if let Some(qt) = attn.wq.qtensor() {
                        tensors.push((format!("{}_attn_q", prefix), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.wk.qtensor() {
                        tensors.push((format!("{}_attn_k", prefix), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.wv.qtensor() {
                        tensors.push((format!("{}_attn_v", prefix), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.wo.qtensor() {
                        tensors.push((format!("{}_attn_o", prefix), qt, Self::block_count(qt)));
                    }
                }
                AttentionLayer::Linear(attn) => {
                    let (main_qt, main_name, gate_qt) = attn.ssm_in.qtensors_for_save();
                    if let Some(qt) = main_qt {
                        tensors.push((
                            format!("{}_{}", prefix, main_name),
                            qt,
                            Self::block_count(qt),
                        ));
                    }
                    if let Some(qt) = gate_qt {
                        tensors.push((format!("{}_attn_gate", prefix), qt, Self::block_count(qt)));
                    }
                    let (ba_qt_0, ba_qt_1) = attn.ssm_beta_alpha.qtensors_for_save();
                    if let Some((name, qt)) = ba_qt_0 {
                        tensors.push((format!("{}_{}", prefix, name), qt, Self::block_count(qt)));
                    }
                    if let Some((name, qt)) = ba_qt_1 {
                        tensors.push((format!("{}_{}", prefix, name), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.ssm_out.qtensor() {
                        tensors.push((format!("{}_ssm_out", prefix), qt, Self::block_count(qt)));
                    }
                }
            }

            if let Some(qt) = layer.moe_block.gate.qtensor() {
                tensors.push((format!("{}_router_gate", prefix), qt, Self::block_count(qt)));
            }
        }

        tensors
    }

    pub fn num_experts(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.moe_block.num_experts)
    }

    pub fn num_experts_per_token(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.moe_block.num_experts_per_tok)
    }

    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    pub fn embedding_weights(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn vocab_size(&self) -> usize {
        self.embed_tokens
            .embeddings()
            .dims()
            .first()
            .copied()
            .unwrap_or(0)
    }

    pub fn lm_head(&self) -> &TQMatMul<LmHeadShape> {
        &self.lm_head
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn set_capture_expert_indices(&mut self, capture: bool) {
        self.capture_expert_indices = capture;
        if !capture {
            self.last_expert_indices = None;
        }
    }

    pub fn is_capturing_expert_indices(&self) -> bool {
        self.capture_expert_indices
    }

    pub fn take_last_expert_indices(&mut self) -> Option<Vec<Tensor>> {
        self.last_expert_indices.take()
    }

    // ========================================================================
    // MTP Speculative Decoding Methods
    // ========================================================================

    pub fn has_mtp(&self) -> bool {
        self.mtp_head.is_some()
    }

    pub fn clear_mtp_caches(&mut self) {
        if let Some(ref mut mtp) = self.mtp_head {
            if self.layer_device_map.is_multi_gpu() {
                let mtp_device = self
                    .layer_device_map
                    .device_for_layer(self.layers.len().saturating_sub(1));
                let _ctx_guard = Tensor::zeros(1, paramecia_core::DType::U8, mtp_device);
            }
            if let Err(err) = mtp.clear_caches() {
                warn!(error = %err, "failed to clear MTP caches");
            }
        }
    }

    pub fn speculative_step(
        &mut self,
        input: &Tensor,
        offset: usize,
        num_speculative: usize,
    ) -> Result<SpeculativeResult> {
        let snapshots = self.snapshot_cache()?;

        let span = self.span.clone();
        let _enter = span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;
        let mut h_typed: TypedTensor<Hidden3> = h.contiguous()?.try_into()?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        let device_masks = self.per_device_masks(causal_mask.as_ref())?;
        let (h_after_layers, _) = self.run_layer_loop(
            h_typed,
            causal_mask.as_ref(),
            &device_masks,
            offset,
            LayerForwardMode::Normal,
        )?;
        h_typed = h_after_layers;

        let h_last = h_typed.inner();
        let base_hidden_for_mtp: TypedTensor<Hidden3> = h_last
            .narrow(1, h_last.dim(1)? - 1, 1)?
            .contiguous()?
            .try_into()?;

        h_typed = self.transfer_to_primary_typed(h_typed)?;

        let h_normed: TypedTensor<Hidden3> = self.norm.forward(h_typed.inner())?.try_into()?;
        let main_logits = self
            .output_head_at(&h_normed, h_normed.inner().dim(1)? - 1)?
            .squeeze(0)?;
        let main_token = main_logits.argmax(D::Minus1)?;

        let (spec_tokens, spec_logits) = if let Some(ref mut mtp) = self.mtp_head {
            if num_speculative > 0 {
                let base_hidden = base_hidden_for_mtp;
                let base_offset = offset + l;

                mtp.forward_batched(
                    &base_hidden,
                    &main_token,
                    &self.embed_tokens,
                    &self.lm_head,
                    base_offset,
                    num_speculative,
                )?
            } else {
                (vec![], vec![])
            }
        } else {
            (vec![], vec![])
        };

        Ok(SpeculativeResult {
            main_token,
            main_logits,
            spec_tokens,
            spec_logits,
            snapshots,
        })
    }

    pub fn verify_and_accept(
        &mut self,
        draft_tokens: &Tensor,
        _snapshots: Vec<LayerSnapshot>,
        offset: usize,
    ) -> Result<VerificationResult> {
        let draft_len = draft_tokens.dim(1)?;
        if draft_len <= 1 {
            return Ok(VerificationResult {
                num_accepted: 0,
                next_logits: None,
            });
        }

        let (verify_logits, _all_hidden) =
            self.verify_with_state_materialization(draft_tokens, offset)?;

        let num_spec = draft_len - 1;

        let verify_logits_for_spec = verify_logits.narrow(1, 0, num_spec)?;
        let verified_tokens = verify_logits_for_spec.argmax(D::Minus1)?;

        let spec_tokens = draft_tokens.narrow(1, 1, num_spec)?;

        let spec_flat = spec_tokens.flatten_all()?;
        let verify_flat = verified_tokens.flatten_all()?;

        let matches = spec_flat.eq(&verify_flat)?;
        let matches_vec: Vec<u8> = matches.to_vec1()?;

        let num_accepted = matches_vec.iter().take_while(|&&m| m == 1).count();

        let next_logits = if num_accepted < num_spec {
            Some(
                verify_logits
                    .narrow(1, num_accepted, 1)?
                    .squeeze(1)?
                    .squeeze(0)?,
            )
        } else {
            Some(
                verify_logits
                    .narrow(1, num_spec, 1)?
                    .squeeze(1)?
                    .squeeze(0)?,
            )
        };

        if num_accepted < num_spec {
            self.restore_to_intermediate_state(num_accepted, offset);
            self.clear_mtp_caches();
        }

        self.clear_intermediate_states();

        drop(_all_hidden);
        let _primary_ctx_guard = if self.layer_device_map.is_multi_gpu() {
            Some(Tensor::zeros(
                1,
                paramecia_core::DType::U8,
                self.layer_device_map.primary_device(),
            )?)
        } else {
            None
        };

        Ok(VerificationResult {
            num_accepted,
            next_logits,
        })
    }

    // ============================================================================
    // Snapshot methods
    // ============================================================================

    pub fn save_snapshot<P: AsRef<Path>>(
        &self,
        path: P,
        state_position: usize,
        tokens: &[u32],
    ) -> Result<()> {
        let config = crate::snapshot::SnapshotConfig {
            num_layers: self.layers.len(),
            hidden_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            max_position_embeddings: 0,
            rope_freq_base: 0.0,
            n_rot: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            recurrent_layers: vec![],
        };

        let mut layer_states = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter() {
            let state = match &layer.attn {
                AttentionLayer::Full(full_attn) => {
                    if let Some(ref cache) = full_attn.preallocated_cache {
                        crate::snapshot::LayerState::FullAttention {
                            k_cache: Some(crate::snapshot::CachedTensor::from_tensor(
                                &cache.k_cache.inner().narrow(2, 0, cache.seq_len)?,
                            )?),
                            v_cache: Some(crate::snapshot::CachedTensor::from_tensor(
                                &cache.v_cache.inner().narrow(2, 0, cache.seq_len)?,
                            )?),
                            seq_len: cache.seq_len,
                        }
                    } else if let Some(ref qcache) = full_attn.quantized_cache {
                        let (k_data, v_data) = qcache.snapshot_quantized_bytes()?;

                        crate::snapshot::LayerState::FullAttention {
                            k_cache: Some(crate::snapshot::CachedTensor::from_quantized(
                                k_data,
                                qcache.ggml_dtype,
                                vec![
                                    qcache.batch_size,
                                    qcache.num_kv_heads,
                                    qcache.seq_len,
                                    qcache.head_dim,
                                ],
                                qcache.bytes_per_row,
                                qcache.padded_head_dim,
                                qcache.max_seq_len,
                            )),
                            v_cache: Some(crate::snapshot::CachedTensor::from_quantized(
                                v_data,
                                qcache.ggml_dtype,
                                vec![
                                    qcache.batch_size,
                                    qcache.num_kv_heads,
                                    qcache.seq_len,
                                    qcache.head_dim,
                                ],
                                qcache.bytes_per_row,
                                qcache.padded_head_dim,
                                qcache.max_seq_len,
                            )),
                            seq_len: qcache.seq_len,
                        }
                    } else {
                        crate::snapshot::LayerState::FullAttention {
                            k_cache: None,
                            v_cache: None,
                            seq_len: 0,
                        }
                    }
                }
                AttentionLayer::Linear(linear_attn) => {
                    if let Some(ref state) = linear_attn.recurrent_state {
                        crate::snapshot::LayerState::LinearAttention {
                            ssm_state: Some(crate::snapshot::CachedTensor::from_tensor(
                                state.ssm_state_ref(),
                            )?),
                            conv_state: Some(crate::snapshot::CachedTensor::from_tensor(
                                state.conv_state_ref(),
                            )?),
                            gate_cumsum_offset: state
                                .gate_offset_ref()
                                .map(crate::snapshot::CachedTensor::from_tensor)
                                .transpose()?,
                        }
                    } else {
                        crate::snapshot::LayerState::LinearAttention {
                            ssm_state: None,
                            conv_state: None,
                            gate_cumsum_offset: None,
                        }
                    }
                }
            };
            layer_states.push(state);
        }

        let snapshot = crate::snapshot::Snapshot {
            state_position,
            tokens: tokens.to_vec(),
            layer_states,
            config,
        };

        crate::snapshot::save_snapshot(&snapshot, path)?;

        Ok(())
    }

    pub fn load_snapshot<P: AsRef<Path>>(&mut self, path: P) -> Result<crate::snapshot::Snapshot> {
        let snapshot = crate::snapshot::load_snapshot(path)?;

        let model_config = crate::snapshot::SnapshotConfig {
            num_layers: self.layers.len(),
            hidden_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            max_position_embeddings: 0,
            rope_freq_base: 0.0,
            n_rot: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            ssm_d_inner: 0,
            ssm_d_state: 0,
            recurrent_layers: vec![],
        };

        let warnings = crate::snapshot::validate_config(&model_config, &snapshot.config)?;
        for warning in warnings {
            tracing::warn!("Snapshot validation: {}", warning);
        }

        if snapshot.layer_states.len() != self.layers.len() {
            paramecia_core::bail!(
                "Layer count mismatch: model has {}, snapshot has {}",
                self.layers.len(),
                snapshot.layer_states.len()
            );
        }

        for (layer, layer_state) in self.layers.iter_mut().zip(snapshot.layer_states.iter()) {
            match (&mut layer.attn, layer_state) {
                (
                    AttentionLayer::Full(full_attn),
                    crate::snapshot::LayerState::FullAttention {
                        k_cache,
                        v_cache,
                        seq_len,
                    },
                ) => {
                    if let (Some(k), Some(v)) = (k_cache, v_cache) {
                        match (k, v) {
                            (
                                crate::snapshot::CachedTensor::Float { .. },
                                crate::snapshot::CachedTensor::Float { .. },
                            ) => {
                                let k_tensor = k.to_tensor(&self.device)?;
                                let v_tensor = v.to_tensor(&self.device)?;

                                full_attn.preallocated_cache = None;
                                full_attn.quantized_cache = None;

                                let batch_size = k_tensor.dim(0)?;
                                let num_kv_heads = k_tensor.dim(1)?;
                                let head_dim = k_tensor.dim(3)?;
                                let max_seq = k_tensor.dim(2)?;

                                let k_full = Tensor::zeros(
                                    (batch_size, num_kv_heads, max_seq, head_dim),
                                    k_tensor.dtype(),
                                    &self.device,
                                )?;
                                let v_full = Tensor::zeros(
                                    (batch_size, num_kv_heads, max_seq, head_dim),
                                    v_tensor.dtype(),
                                    &self.device,
                                )?;

                                let k_full = k_full.slice_scatter(&k_tensor, 2, 0)?;
                                let v_full = v_full.slice_scatter(&v_tensor, 2, 0)?;

                                full_attn.preallocated_cache = Some(PreallocatedKvCache {
                                    k_cache: k_full.try_into()?,
                                    v_cache: v_full.try_into()?,
                                    seq_len: *seq_len,
                                    max_seq_len: max_seq,
                                    batch_size,
                                });
                            }
                            (
                                crate::snapshot::CachedTensor::Quantized {
                                    data: k_data,
                                    dtype: k_dtype,
                                    shape: k_shape,
                                    bytes_per_row: saved_bytes_per_row,
                                    padded_head_dim: saved_padded_head_dim,
                                    max_seq_len: saved_max_seq_len,
                                },
                                crate::snapshot::CachedTensor::Quantized {
                                    data: v_data,
                                    dtype: _v_dtype,
                                    ..
                                },
                            ) => {
                                full_attn.preallocated_cache = None;
                                full_attn.quantized_cache = None;

                                let batch_size = k_shape[0];
                                let num_kv_heads = k_shape[1];
                                let head_dim = k_shape[3];

                                let block_size = k_dtype.block_size();
                                let total_rows = batch_size * num_kv_heads * saved_max_seq_len;
                                let total_bytes = total_rows * saved_bytes_per_row;

                                let mut k_cache = vec![0u8; total_bytes];
                                let mut v_cache = vec![0u8; total_bytes];

                                k_cache[..k_data.len()].copy_from_slice(k_data);
                                v_cache[..v_data.len()].copy_from_slice(v_data);

                                let mut qcache = PreallocatedQuantizedKvCache {
                                    k_cache,
                                    v_cache,
                                    ggml_dtype: *k_dtype,
                                    seq_len: *seq_len,
                                    max_seq_len: *saved_max_seq_len,
                                    batch_size,
                                    num_kv_heads,
                                    head_dim,
                                    padded_head_dim: *saved_padded_head_dim,
                                    block_size,
                                    bytes_per_row: *saved_bytes_per_row,
                                    device: self.device.clone(),
                                    #[cfg(feature = "vulkan")]
                                    k_gpu_storage: None,
                                    #[cfg(feature = "vulkan")]
                                    v_gpu_storage: None,
                                    #[cfg(feature = "cuda")]
                                    k_gpu_storage: None,
                                    #[cfg(feature = "cuda")]
                                    v_gpu_storage: None,
                                };
                                qcache.rebuild_gpu_storage_from_host()?;

                                full_attn.quantized_cache = Some(qcache);
                            }
                            _ => {
                                paramecia_core::bail!(
                                    "KV cache type mismatch (float vs quantized)"
                                );
                            }
                        }
                    }
                }
                (
                    AttentionLayer::Linear(linear_attn),
                    crate::snapshot::LayerState::LinearAttention {
                        ssm_state,
                        conv_state,
                        gate_cumsum_offset,
                    },
                ) => {
                    if let Some(ssm) = ssm_state {
                        let ssm_tensor = ssm.to_tensor(&self.device)?;
                        let conv_tensor = conv_state
                            .as_ref()
                            .map(|c| c.to_tensor(&self.device))
                            .transpose()?;
                        let gate_tensor = gate_cumsum_offset
                            .as_ref()
                            .map(|g| g.to_tensor(&self.device))
                            .transpose()?;
                        let (b, _, _, _) = ssm_tensor.dims4()?;
                        let conv_tensor = match conv_tensor {
                            Some(t) => t,
                            None => Tensor::zeros(
                                (
                                    b,
                                    linear_attn.d_inner
                                        + 2 * linear_attn.n_groups * linear_attn.d_state,
                                    linear_attn.conv_kernel_size - 1,
                                ),
                                self.dtype,
                                &self.device,
                            )?,
                        };
                        let restored = if let Some(gate) = gate_tensor {
                            RecurrentState::with_gate_offset(ssm_tensor, conv_tensor, gate)?
                        } else {
                            RecurrentState::new(ssm_tensor, conv_tensor)?
                        };
                        linear_attn.recurrent_state = Some(restored);
                    }
                }
                _ => {
                    paramecia_core::bail!("Layer type mismatch during snapshot restore");
                }
            }
        }

        Ok(snapshot)
    }

    pub fn validate_snapshot<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let snapshot = crate::snapshot::load_snapshot(path)?;

        if snapshot.layer_states.len() != self.layers.len() {
            return Ok(false);
        }

        Ok(true)
    }
}
