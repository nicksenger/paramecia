use super::expert_cache::{should_cache_experts, CachedMatmuls, ExpertCache, GpuHotExpertCache};
use super::gguf_loader::Gguf;
use super::is_gpu_device;
use super::shape::{BlockMults1, Tk, B, E, N, S, SI, T};
use super::utils::{log_shape, log_typed_qmatmul_shape};
use inception::{primitive, Inception};
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, BoolToIndex2, Combinator, CombinatorTraceExt, ConstOut, Fanout, First,
    FlattenTripleResult, FromOp, Identity, InjectConst, LiftResult, MapErr, MapOk, OptionThen,
    Second, SetInsertOp, Switch, Then, Third, TryFoldRange, TryFoldVec, TryFromOp, WrapOk, Zip,
    Zip3, ZipOk, ZipOk3,
};
use paramecia_core::quantized::{GgmlDType, QStorage, QTensor, SharedQTensor};
use paramecia_core::{DType, Device, Result, Tensor};
use paramecia_nn::Activation;
use paramecia_tensor::dims2::Dims2Op;
use paramecia_tensor::dims3::Dims3Op;
use paramecia_tensor::flatten_prefix2::FlattenPrefix2Op;
use paramecia_tensor::from_vec_on_device::{FromVec1OnDeviceOp, FromVecColOnDeviceOp};
use paramecia_tensor::glowstick::num::{Unsigned, U1, U2};
use paramecia_tensor::glowstick::{Shape1, Shape2, Shape3};
use paramecia_tensor::group_topk_assignments::GroupTopKAssignmentsOp;
use paramecia_tensor::index_add_dim0::IndexAddDim0Op;
use paramecia_tensor::index_select_dim0::IndexSelectDim0Op;
use paramecia_tensor::into_inner::{IntoInnerOp, IntoInnerResultOp};
use paramecia_tensor::qmatmul_op::QMatMulOp;
use paramecia_tensor::remap_indices::RemapIndicesOp as TensorRemapIndicesOp;
use paramecia_tensor::tensor_device_info::{TensorDeviceInfo, TensorDeviceInfoOp};
use paramecia_tensor::topk_from_logits::TopkFromLogitsOp as TensorTopkFromLogitsOp;
use paramecia_tensor::Tensor as TTensor;
use paramecia_tensor::{
    broadcast_add::BroadcastAddOp, broadcast_mul::BroadcastMulOp, cast_like::CastLikeOp,
    clamp::ClampOp, contiguous::ContiguousOp, residual_add::ResidualAddOp, sigmoid::SigmoidOp,
    silu::SiluOp, sum_dim::SumDimOp,
};
use std::collections::HashSet;
use std::io::{Read, Seek};
use std::ops::Deref;
use std::sync::Arc;
use tracing::debug;

type TQMatMul<S> = paramecia_tensor::QMatMul<S>;
type THidden = TTensor<Shape3<B, N, S>>;
type TIntermediate = TTensor<Shape3<B, N, SI>>;
type TRouterLogits = TTensor<Shape3<B, N, E>>;
type TTopWeights = TTensor<Shape3<B, N, Tk>>;
type TTopIndices = TTensor<Shape3<B, N, Tk>>;
type TExpertVec = TTensor<Shape1<E>>;
type TSharedGate = TTensor<Shape2<U1, S>>;
type TGateScore = TTensor<Shape3<B, N, U1>>;
type THiddenFlat = TTensor<Shape2<T, S>>;
type TTopFlat = TTensor<Shape2<T, Tk>>;
type TTopX = TTensor<Shape1<T>>;
type TWeightCol = TTensor<Shape2<T, U1>>;
type TBlockMults = TTensor<BlockMults1>;
type MoeDispatchPrefetchShape = (usize, usize, usize);
type MoeDispatchPrefetchOut = (THiddenFlat, TTopFlat, TTopFlat, MoeDispatchPrefetchShape);
type CpuFusedExpertAssignment = (usize, Vec<u32>, Vec<f32>);
type CpuFusedExpertAssignments = Vec<CpuFusedExpertAssignment>;
type RemapTopIndicesFlow = MapErr<
    TensorRemapIndicesOp<Shape3<B, N, Tk>, Shape1<E>>,
    TTopIndices,
    TTopIndices,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

#[derive(Clone)]
pub(super) struct ExpertWeightTensor {
    inner: SharedQTensor,
    role: &'static str,
}

impl ExpertWeightTensor {
    pub(super) fn new(inner: SharedQTensor, role: &'static str) -> Result<Self> {
        let shape = inner
            .read()
            .map_err(|_| {
                paramecia_core::Error::Msg(format!(
                    "failed to lock shared expert tensor for role `{role}`"
                ))
            })?
            .shape()
            .clone();
        if shape.dims().len() != 3 {
            paramecia_core::bail!(
                "expert tensor `{role}` expected rank 3 [experts, out, in], got shape {:?}",
                shape
            );
        }
        let dims = shape.dims();
        let expected_gate_up = [E::USIZE, SI::USIZE, S::USIZE];
        let expected_down = [E::USIZE, S::USIZE, SI::USIZE];
        let expected = if role.contains("down_exps") {
            Some(expected_down)
        } else if role.contains("gate_exps") || role.contains("up_exps") {
            Some(expected_gate_up)
        } else {
            None
        };
        if let Some(expected) = expected {
            if dims != expected {
                paramecia_core::bail!(
                    "expert tensor `{role}` expected shape {:?}, got {:?}",
                    expected,
                    dims
                );
            }
        }
        debug!(role, shape = ?shape, "Loaded typed expert tensor");
        Ok(Self { inner, role })
    }

    pub(super) fn as_shared(&self) -> &SharedQTensor {
        &self.inner
    }
}

impl Deref for ExpertWeightTensor {
    type Target = SharedQTensor;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::fmt::Debug for ExpertWeightTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.inner.read().ok().map(|t| t.shape().clone());
        f.debug_struct("ExpertWeightTensor")
            .field("role", &self.role)
            .field("shape", &shape)
            .finish()
    }
}

#[derive(Clone)]
pub(super) struct MoeDispatchPrepared {
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    hidden_flat: THiddenFlat,
    indices_flat: TTopFlat,
    weights_flat: TTopFlat,
}

impl std::fmt::Debug for MoeDispatchPrepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeDispatchPrepared")
            .field("batch_size", &self.batch_size)
            .field("seq_len", &self.seq_len)
            .field("hidden_dim", &self.hidden_dim)
            .finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum MoeDispatchPath {
    CpuFused,
    Batched,
    GpuGrouped,
    Sequential,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MoePathCtx {
    training_mode: bool,
    supports_batched: bool,
    all_experts_on_cpu: bool,
    hidden_on_gpu: bool,
    hidden_is_vulkan: bool,
}

#[derive(Debug, Clone)]
pub(super) struct MoeAssignments {
    top_x: Vec<Vec<u32>>,
    selected_rws: Vec<Vec<f32>>,
}
impl TryFrom<MoeAssignmentVecs> for MoeAssignments {
    type Error = paramecia_core::Error;

    fn try_from(value: MoeAssignmentVecs) -> std::result::Result<Self, Self::Error> {
        let (top_x, selected_rws) = value;
        Ok(Self {
            top_x,
            selected_rws,
        })
    }
}

type MoeAssignmentVecs = (Vec<Vec<u32>>, Vec<Vec<f32>>);
type MoeGroupAssignmentsTensorStep = MapErr<
    GroupTopKAssignmentsOp<Shape2<T, Tk>>,
    (TTopFlat, TTopFlat),
    MoeAssignmentVecs,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MoeGroupAssignmentsBuildStep =
    TryFromOp<MoeAssignmentVecs, MoeAssignments, paramecia_core::Error>;
type MoeGroupAssignmentsBuildLift =
    LiftResult<MoeGroupAssignmentsBuildStep, Result<MoeAssignmentVecs>, MoeAssignments>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoeGroupAssignments(MoeGroupAssignmentsTensorStep, MoeGroupAssignmentsBuildLift);

impl MoeGroupAssignments {
    pub(super) fn new(num_experts: usize) -> Self {
        Self(
            MapErr::new(GroupTopKAssignmentsOp::new(num_experts)),
            LiftResult::new(TryFromOp::default()),
        )
    }
}
impl std::fmt::Debug for MoeGroupAssignments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeGroupAssignments").finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum MoePrefetchMode {
    None,
    Inference,
    Training,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MoePrefetchContext {
    training_mode: bool,
    inference_cache_enabled: bool,
    training_cache_enabled: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MoePrefetchInferenceCondOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for MoePrefetchInferenceCondOp {
    type In = MoePrefetchContext;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        !input.training_mode && input.inference_cache_enabled
    }
}
#[primitive(property = Visualize)]
impl Vis for MoePrefetchInferenceCondOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MoePrefetchInferenceCond").with_output_type::<bool>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MoePrefetchTrainingCondOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for MoePrefetchTrainingCondOp {
    type In = MoePrefetchContext;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input.training_mode && input.training_cache_enabled
    }
}
#[primitive(property = Visualize)]
impl Vis for MoePrefetchTrainingCondOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MoePrefetchTrainingCond").with_output_type::<bool>()
    }
}

type MoePrefetchPair = (bool, MoePrefetchContext);
type MoePrefetchPrimarySelector = Then<First<bool, MoePrefetchContext>, BoolToIndex2>;
type MoePrefetchTrainingSwitch =
    Switch<BoolToIndex2, ConstOut<bool, MoePrefetchMode>, ConstOut<bool, MoePrefetchMode>>;
type MoePrefetchFallbackFlow = Then<
    Second<bool, MoePrefetchContext>,
    Then<MoePrefetchTrainingCondOp, MoePrefetchTrainingSwitch>,
>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoePrefetchSelect(
    Fanout<MoePrefetchContext>,
    Zip<MoePrefetchInferenceCondOp, Identity<MoePrefetchContext>>,
    Switch<
        MoePrefetchPrimarySelector,
        ConstOut<MoePrefetchPair, MoePrefetchMode>,
        MoePrefetchFallbackFlow,
    >,
);
impl MoePrefetchSelect {
    pub(super) fn new() -> Self {
        Self(
            Fanout::default(),
            Zip::new(MoePrefetchInferenceCondOp, Identity::default()),
            Switch::new(
                Then::new(First::default(), BoolToIndex2),
                ConstOut::new(MoePrefetchMode::Inference),
                Then::new(
                    Second::default(),
                    Then::new(
                        MoePrefetchTrainingCondOp,
                        Switch::new(
                            BoolToIndex2,
                            ConstOut::new(MoePrefetchMode::Training),
                            ConstOut::new(MoePrefetchMode::None),
                        ),
                    ),
                ),
            ),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum MoeSequentialExecMode {
    Parallel,
    Sequential,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MoeSequentialExecContext {
    remaining_experts_len: usize,
    all_experts_on_cpu: bool,
    training_mode: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MoeSequentialParallelCondOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for MoeSequentialParallelCondOp {
    type In = MoeSequentialExecContext;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input.remaining_experts_len > 1 && input.all_experts_on_cpu && !input.training_mode
    }
}
#[primitive(property = Visualize)]
impl Vis for MoeSequentialParallelCondOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MoeSequentialParallelCond").with_output_type::<bool>()
    }
}

type MoeSequentialModeSwitch = Switch<
    BoolToIndex2,
    ConstOut<bool, MoeSequentialExecMode>,
    ConstOut<bool, MoeSequentialExecMode>,
>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoeSequentialExecSelect(MoeSequentialParallelCondOp, MoeSequentialModeSwitch);
impl MoeSequentialExecSelect {
    pub(super) fn new() -> Self {
        Self(
            MoeSequentialParallelCondOp,
            Switch::new(
                BoolToIndex2,
                ConstOut::new(MoeSequentialExecMode::Parallel),
                ConstOut::new(MoeSequentialExecMode::Sequential),
            ),
        )
    }
}

struct PrefetchInferenceExpertCtx<'a> {
    cache: Option<&'a mut ExpertCache>,
    gate_exps: Option<&'a SharedQTensor>,
    up_exps: Option<&'a SharedQTensor>,
    down_exps: Option<&'a SharedQTensor>,
}
#[derive(Debug, Default, Clone, Copy)]
struct PrefetchInferenceExpertOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchInferenceExpertCtx<'a>> for PrefetchInferenceExpertOp {
    type In = ((), usize);
    type Out = Result<()>;

    fn forward(&mut self, ctx: &mut PrefetchInferenceExpertCtx<'a>, input: Self::In) -> Self::Out {
        let (_, expert_idx) = input;
        let Some(cache) = ctx.cache.as_deref_mut() else {
            return Ok(());
        };
        let gate_exps = ctx.gate_exps.ok_or_else(|| {
            paramecia_core::Error::Msg("inference prefetch missing gate experts".to_string())
        })?;
        let up_exps = ctx.up_exps.ok_or_else(|| {
            paramecia_core::Error::Msg("inference prefetch missing up experts".to_string())
        })?;
        let down_exps = ctx.down_exps.ok_or_else(|| {
            paramecia_core::Error::Msg("inference prefetch missing down experts".to_string())
        })?;
        let _ = cache.get_or_prepare(expert_idx, gate_exps, up_exps, down_exps)?;
        Ok(())
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchInferenceExpertOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("PrefetchInferenceExpert").with_output_type::<Result<()>>()
    }
}
fn prefetch_inference_expert_op() -> PrefetchInferenceExpertOp {
    PrefetchInferenceExpertOp
}

struct PrefetchTrainingExpertCtx<'a> {
    experts: Option<&'a mut MoeExperts>,
}
#[derive(Debug, Default, Clone, Copy)]
struct PrefetchTrainingExpertOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<PrefetchTrainingExpertCtx<'a>> for PrefetchTrainingExpertOp {
    type In = ((), usize);
    type Out = Result<()>;

    fn forward(&mut self, ctx: &mut PrefetchTrainingExpertCtx<'a>, input: Self::In) -> Self::Out {
        let (_, expert_idx) = input;
        let Some(experts) = ctx.experts.as_deref_mut() else {
            return Ok(());
        };

        let in_cache = experts
            .training_cache
            .as_ref()
            .map(|c| c.entries.contains_key(&expert_idx))
            .unwrap_or(false);
        if in_cache {
            return Ok(());
        }

        let entry = experts.build_scaled_mats(expert_idx)?;
        if let Some(cache) = experts.training_cache.as_mut() {
            cache.evict_if_needed();
            cache.entries.insert(expert_idx, entry);
            cache.touch(expert_idx);
        }
        Ok(())
    }
}
#[primitive(property = Visualize)]
impl Vis for PrefetchTrainingExpertOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("PrefetchTrainingExpert").with_output_type::<Result<()>>()
    }
}
fn prefetch_training_expert_op() -> PrefetchTrainingExpertOp {
    PrefetchTrainingExpertOp
}

#[derive(Debug, Clone)]
struct GpuCachedApplyState {
    ys: Tensor,
    gpu_processed: HashSet<usize>,
}

struct ApplyGpuCachedExpertCtx<'a> {
    experts: &'a mut MoeExperts,
    hidden_flat: &'a Tensor,
    top_x: &'a [Vec<u32>],
    selected_rws: &'a [Vec<f32>],
    gpu_device: &'a Device,
}
#[derive(Clone)]
struct GpuCachedPrepared {
    ys_typed: THiddenFlat,
    top_x_tensor: TTopX,
    selected_rws_tensor: TWeightCol,
    hidden_on_gpu: Tensor,
    cached_mats: CachedMatmuls,
    expert_idx: usize,
    gpu_processed: HashSet<usize>,
}

#[derive(Clone)]
struct GpuCachedForwarded {
    ys_typed: THiddenFlat,
    top_x_tensor: TTopX,
    selected_rws_tensor: TWeightCol,
    result: THiddenFlat,
    expert_idx: usize,
    gpu_processed: HashSet<usize>,
}

#[derive(Debug, Default, Clone, Copy)]
struct PrepareGpuCachedDispatchOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<ApplyGpuCachedExpertCtx<'a>> for PrepareGpuCachedDispatchOp {
    type In = (GpuCachedApplyState, (usize, CachedMatmuls));
    type Out = Result<GpuCachedPrepared>;

    fn forward(&mut self, ctx: &mut ApplyGpuCachedExpertCtx<'a>, input: Self::In) -> Self::Out {
        let (state, (expert_idx, cached_mats)) = input;
        let top_x_expert = ctx.top_x.get(expert_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "gpu cached apply: expert index {} out of bounds (top_x len {})",
                expert_idx,
                ctx.top_x.len()
            ))
        })?;
        let selected_rws_expert = ctx.selected_rws.get(expert_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "gpu cached apply: expert index {} out of bounds (selected_rws len {})",
                expert_idx,
                ctx.selected_rws.len()
            ))
        })?;
        if top_x_expert.len() != selected_rws_expert.len() {
            return Err(paramecia_core::Error::Msg(format!(
                "gpu cached apply: assignment/weight len mismatch for expert {} ({} vs {})",
                expert_idx,
                top_x_expert.len(),
                selected_rws_expert.len()
            )));
        }

        let ys_typed: THiddenFlat = state.ys.try_into()?;
        let hidden_flat_typed: THiddenFlat = ctx.hidden_flat.clone().try_into()?;
        let mut from_top_x = FromVec1OnDeviceOp::<Shape1<T>, u32>::default();
        let top_x_tensor = <FromVec1OnDeviceOp<Shape1<T>, u32> as Combinator<()>>::forward(
            &mut from_top_x,
            &mut (),
            (top_x_expert.clone(), ctx.hidden_flat.device().clone()),
        )?;
        let mut from_selected_rws = FromVecColOnDeviceOp::<Shape2<T, U1>, f32>::default();
        let selected_rws_col =
            <FromVecColOnDeviceOp<Shape2<T, U1>, f32> as Combinator<()>>::forward(
                &mut from_selected_rws,
                &mut (),
                (
                    selected_rws_expert.clone(),
                    ctx.hidden_flat.device().clone(),
                ),
            )?;
        let mut cast_like = CastLikeOp::<Shape2<T, S>, Shape2<T, U1>>::default();
        let (_, selected_rws_tensor): (THiddenFlat, TWeightCol) =
            <CastLikeOp<Shape2<T, S>, Shape2<T, U1>> as Combinator<()>>::forward(
                &mut cast_like,
                &mut (),
                (hidden_flat_typed.clone(), selected_rws_col),
            )?;
        let mut gather = GatherHiddenToCore::new();
        let current_state = <GatherHiddenToCore as Combinator<()>>::forward(
            &mut gather,
            &mut (),
            (hidden_flat_typed, top_x_tensor.clone()),
        )?;
        let hidden_on_gpu = current_state.to_device(ctx.gpu_device)?;
        Ok(GpuCachedPrepared {
            ys_typed,
            top_x_tensor,
            selected_rws_tensor,
            hidden_on_gpu,
            cached_mats,
            expert_idx,
            gpu_processed: state.gpu_processed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for PrepareGpuCachedDispatchOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "PrepareGpuCachedDispatch",
            Graph::sequence(
                <FromVec1OnDeviceOp<Shape1<T>, u32> as Vis>::visualize(),
                Graph::sequence(
                    <FromVecColOnDeviceOp<Shape2<T, U1>, f32> as Vis>::visualize(),
                    Graph::sequence(
                        <CastLikeOp<Shape2<T, S>, Shape2<T, U1>> as Vis>::visualize(),
                        Graph::sequence(
                            <GatherHiddenToCore as Vis>::visualize(),
                            <paramecia_tensor::to_device::ToDeviceOp<Shape2<T, S>> as Vis>::visualize(),
                        ),
                    ),
                ),
            ),
        )
        .with_output_type::<Result<GpuCachedPrepared>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ApplyGpuCachedForwardOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<ApplyGpuCachedExpertCtx<'a>> for ApplyGpuCachedForwardOp {
    type In = GpuCachedPrepared;
    type Out = Result<GpuCachedForwarded>;

    fn forward(&mut self, ctx: &mut ApplyGpuCachedExpertCtx<'a>, input: Self::In) -> Self::Out {
        let result = ctx.experts.forward_with_cached(
            &input.hidden_on_gpu,
            &input.cached_mats,
            ctx.gpu_device,
        )?;
        let result: THiddenFlat = result.to_device(ctx.hidden_flat.device())?.try_into()?;
        Ok(GpuCachedForwarded {
            ys_typed: input.ys_typed,
            top_x_tensor: input.top_x_tensor,
            selected_rws_tensor: input.selected_rws_tensor,
            result,
            expert_idx: input.expert_idx,
            gpu_processed: input.gpu_processed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for ApplyGpuCachedForwardOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("ApplyGpuCachedForward").with_output_type::<Result<GpuCachedForwarded>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct FinalizeGpuCachedDispatchOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for FinalizeGpuCachedDispatchOp {
    type In = GpuCachedForwarded;
    type Out = Result<GpuCachedApplyState>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let weighted: THiddenFlat =
            paramecia_tensor::broadcast_mul!(input.result, input.selected_rws_tensor)?;
        let mut index_add = IndexAddDim0Op::<Shape2<T, S>, Shape1<T>, Shape2<T, S>>::default();
        let ys_typed =
            <IndexAddDim0Op<Shape2<T, S>, Shape1<T>, Shape2<T, S>> as Combinator<()>>::forward(
                &mut index_add,
                &mut (),
                (input.ys_typed, input.top_x_tensor, weighted),
            )?;
        let mut set_insert = SetInsertOp::<usize>::default();
        let gpu_processed = <SetInsertOp<usize> as Combinator<()>>::forward(
            &mut set_insert,
            &mut (),
            (input.gpu_processed, input.expert_idx),
        );
        Ok(GpuCachedApplyState {
            ys: ys_typed.into_inner(),
            gpu_processed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for FinalizeGpuCachedDispatchOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "FinalizeGpuCachedDispatch",
            Graph::sequence(
                <BroadcastMulOp<Shape2<T, S>, Shape2<T, U1>> as Vis>::visualize(),
                Graph::sequence(
                    <IndexAddDim0Op<Shape2<T, S>, Shape1<T>, Shape2<T, S>> as Vis>::visualize(),
                    Graph::sequence(
                        <IntoInnerOp<Shape2<T, S>> as Vis>::visualize(),
                        <SetInsertOp<usize> as Vis>::visualize(),
                    ),
                ),
            ),
        )
        .with_output_type::<Result<GpuCachedApplyState>>()
    }
}

type ApplyGpuCachedForwardLift =
    LiftResult<ApplyGpuCachedForwardOp, Result<GpuCachedPrepared>, GpuCachedForwarded>;
type ApplyGpuCachedFinalizeLift =
    LiftResult<FinalizeGpuCachedDispatchOp, Result<GpuCachedForwarded>, GpuCachedApplyState>;
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct ApplyGpuCachedExpert(
    PrepareGpuCachedDispatchOp,
    ApplyGpuCachedForwardLift,
    ApplyGpuCachedFinalizeLift,
);
impl ApplyGpuCachedExpert {
    fn new() -> Self {
        Self(
            PrepareGpuCachedDispatchOp,
            LiftResult::new(ApplyGpuCachedForwardOp),
            LiftResult::new(FinalizeGpuCachedDispatchOp),
        )
    }
}
impl std::fmt::Debug for ApplyGpuCachedExpert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApplyGpuCachedExpert").finish()
    }
}
fn apply_gpu_cached_expert_op() -> ApplyGpuCachedExpert {
    ApplyGpuCachedExpert::new()
}

struct ApplySequentialExpertCtx<'a> {
    experts: &'a mut MoeExperts,
    hidden_flat: &'a Tensor,
    top_x: &'a [Vec<u32>],
    selected_rws: &'a [Vec<f32>],
}
type GatherHiddenSelectFlow = MapErr<
    IndexSelectDim0Op<Shape2<T, S>, Shape1<T>, Shape2<T, S>>,
    (THiddenFlat, TTopX),
    THiddenFlat,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type GatherHiddenContiguousFlow = MapErr<
    ContiguousOp<Shape2<T, S>>,
    THiddenFlat,
    THiddenFlat,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type GatherHiddenContiguousLift =
    LiftResult<GatherHiddenContiguousFlow, Result<THiddenFlat>, THiddenFlat>;
type GatherHiddenIntoCoreLift = IntoInnerResultOp<Shape2<T, S>, paramecia_core::Error>;
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct GatherHiddenToCore(
    GatherHiddenSelectFlow,
    GatherHiddenContiguousLift,
    GatherHiddenIntoCoreLift,
);
impl GatherHiddenToCore {
    fn new() -> Self {
        Self(
            GatherHiddenSelectFlow::new(IndexSelectDim0Op::default()),
            LiftResult::new(GatherHiddenContiguousFlow::new(ContiguousOp::default())),
            IntoInnerResultOp::default(),
        )
    }
}
impl std::fmt::Debug for GatherHiddenToCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatherHiddenToCore").finish()
    }
}

#[derive(Clone)]
struct SequentialPrepared {
    ys_typed: THiddenFlat,
    top_x_tensor: TTopX,
    selected_rws_tensor: TWeightCol,
    current_state: Tensor,
    expert_idx: usize,
}

#[derive(Clone)]
struct SequentialForwarded {
    ys_typed: THiddenFlat,
    top_x_tensor: TTopX,
    selected_rws_tensor: TWeightCol,
    current_hidden_states: THiddenFlat,
}

#[derive(Debug, Default, Clone, Copy)]
struct PrepareSequentialDispatchOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<ApplySequentialExpertCtx<'a>> for PrepareSequentialDispatchOp {
    type In = (Tensor, usize);
    type Out = Result<SequentialPrepared>;

    fn forward(&mut self, ctx: &mut ApplySequentialExpertCtx<'a>, input: Self::In) -> Self::Out {
        let (ys, expert_idx) = input;
        let top_x_expert = ctx.top_x.get(expert_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "sequential apply: expert index {} out of bounds (top_x len {})",
                expert_idx,
                ctx.top_x.len()
            ))
        })?;
        let selected_rws_expert = ctx.selected_rws.get(expert_idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "sequential apply: expert index {} out of bounds (selected_rws len {})",
                expert_idx,
                ctx.selected_rws.len()
            ))
        })?;
        if top_x_expert.len() != selected_rws_expert.len() {
            return Err(paramecia_core::Error::Msg(format!(
                "sequential apply: assignment/weight len mismatch for expert {} ({} vs {})",
                expert_idx,
                top_x_expert.len(),
                selected_rws_expert.len()
            )));
        }

        let ys_typed: THiddenFlat = ys.try_into()?;
        let hidden_flat_typed: THiddenFlat = ctx.hidden_flat.clone().try_into()?;
        let mut from_top_x = FromVec1OnDeviceOp::<Shape1<T>, u32>::default();
        let top_x_tensor = <FromVec1OnDeviceOp<Shape1<T>, u32> as Combinator<()>>::forward(
            &mut from_top_x,
            &mut (),
            (top_x_expert.clone(), ctx.hidden_flat.device().clone()),
        )?;
        let mut from_selected_rws = FromVecColOnDeviceOp::<Shape2<T, U1>, f32>::default();
        let selected_rws_col =
            <FromVecColOnDeviceOp<Shape2<T, U1>, f32> as Combinator<()>>::forward(
                &mut from_selected_rws,
                &mut (),
                (
                    selected_rws_expert.clone(),
                    ctx.hidden_flat.device().clone(),
                ),
            )?;
        let mut cast_like = CastLikeOp::<Shape2<T, S>, Shape2<T, U1>>::default();
        let (_, selected_rws_tensor): (THiddenFlat, TWeightCol) =
            <CastLikeOp<Shape2<T, S>, Shape2<T, U1>> as Combinator<()>>::forward(
                &mut cast_like,
                &mut (),
                (hidden_flat_typed.clone(), selected_rws_col),
            )?;
        let mut gather = GatherHiddenToCore::new();
        let current_state = <GatherHiddenToCore as Combinator<()>>::forward(
            &mut gather,
            &mut (),
            (hidden_flat_typed, top_x_tensor.clone()),
        )?;
        Ok(SequentialPrepared {
            ys_typed,
            top_x_tensor,
            selected_rws_tensor,
            current_state,
            expert_idx,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for PrepareSequentialDispatchOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "PrepareSequentialDispatch",
            Graph::sequence(
                <FromVec1OnDeviceOp<Shape1<T>, u32> as Vis>::visualize(),
                Graph::sequence(
                    <FromVecColOnDeviceOp<Shape2<T, U1>, f32> as Vis>::visualize(),
                    Graph::sequence(
                        <CastLikeOp<Shape2<T, S>, Shape2<T, U1>> as Vis>::visualize(),
                        <GatherHiddenToCore as Vis>::visualize(),
                    ),
                ),
            ),
        )
        .with_output_type::<Result<SequentialPrepared>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct ApplySequentialExpertForwardOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<ApplySequentialExpertCtx<'a>> for ApplySequentialExpertForwardOp {
    type In = SequentialPrepared;
    type Out = Result<SequentialForwarded>;

    fn forward(&mut self, ctx: &mut ApplySequentialExpertCtx<'a>, input: Self::In) -> Self::Out {
        let current_hidden_states: THiddenFlat = ctx
            .experts
            .forward_expert(&input.current_state, input.expert_idx)?
            .try_into()?;
        Ok(SequentialForwarded {
            ys_typed: input.ys_typed,
            top_x_tensor: input.top_x_tensor,
            selected_rws_tensor: input.selected_rws_tensor,
            current_hidden_states,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for ApplySequentialExpertForwardOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("ApplySequentialExpertForward")
            .with_output_type::<Result<SequentialForwarded>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct FinalizeSequentialDispatchOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for FinalizeSequentialDispatchOp {
    type In = SequentialForwarded;
    type Out = Result<Tensor>;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        let weighted: THiddenFlat = paramecia_tensor::broadcast_mul!(
            input.current_hidden_states,
            input.selected_rws_tensor
        )?;
        let mut index_add = IndexAddDim0Op::<Shape2<T, S>, Shape1<T>, Shape2<T, S>>::default();
        let updated =
            <IndexAddDim0Op<Shape2<T, S>, Shape1<T>, Shape2<T, S>> as Combinator<()>>::forward(
                &mut index_add,
                &mut (),
                (input.ys_typed, input.top_x_tensor, weighted),
            )?;
        Ok(updated.into_inner())
    }
}
#[primitive(property = Visualize)]
impl Vis for FinalizeSequentialDispatchOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "FinalizeSequentialDispatch",
            Graph::sequence(
                <BroadcastMulOp<Shape2<T, S>, Shape2<T, U1>> as Vis>::visualize(),
                Graph::sequence(
                    <IndexAddDim0Op<Shape2<T, S>, Shape1<T>, Shape2<T, S>> as Vis>::visualize(),
                    <IntoInnerOp<Shape2<T, S>> as Vis>::visualize(),
                ),
            ),
        )
        .with_output_type::<Result<Tensor>>()
    }
}

type ApplySequentialForwardLift =
    LiftResult<ApplySequentialExpertForwardOp, Result<SequentialPrepared>, SequentialForwarded>;
type ApplySequentialFinalizeLift =
    LiftResult<FinalizeSequentialDispatchOp, Result<SequentialForwarded>, Tensor>;
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct ApplySequentialExpert(
    PrepareSequentialDispatchOp,
    ApplySequentialForwardLift,
    ApplySequentialFinalizeLift,
);
impl ApplySequentialExpert {
    fn new() -> Self {
        Self(
            PrepareSequentialDispatchOp,
            LiftResult::new(ApplySequentialExpertForwardOp),
            LiftResult::new(FinalizeSequentialDispatchOp),
        )
    }
}
impl std::fmt::Debug for ApplySequentialExpert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApplySequentialExpert").finish()
    }
}
fn apply_sequential_expert_op() -> ApplySequentialExpert {
    ApplySequentialExpert::new()
}
#[allow(dead_code)]
type PrefetchInferenceExpertVisOp = PrefetchInferenceExpertOp;
#[allow(dead_code)]
type PrefetchTrainingExpertVisOp = PrefetchTrainingExpertOp;
#[allow(dead_code)]
type ApplyGpuCachedExpertVisOp = ApplyGpuCachedExpert;
#[allow(dead_code)]
type ApplySequentialExpertVisOp = ApplySequentialExpert;

#[derive(Debug, Clone)]
struct CpuFusedCountState {
    expert_counts: Vec<usize>,
}

#[derive(Debug, Clone)]
struct CpuFusedAssignState {
    token_ids_by_expert: Vec<Vec<u32>>,
    routing_weights_by_expert: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct CpuFusedCollectState {
    token_ids_by_expert: Vec<Vec<u32>>,
    routing_weights_by_expert: Vec<Vec<f32>>,
    expert_assignments: Vec<(usize, Vec<u32>, Vec<f32>)>,
}

#[derive(Debug)]
struct CpuFusedCountExpertStepOp<'a> {
    indices_data: &'a [u32],
    num_experts: usize,
}
impl<'a> CpuFusedCountExpertStepOp<'a> {
    fn new(indices_data: &'a [u32], num_experts: usize) -> Self {
        Self {
            indices_data,
            num_experts,
        }
    }
}
#[primitive(property = Arrow)]
impl<'a> Combinator for CpuFusedCountExpertStepOp<'a> {
    type In = (CpuFusedCountState, usize);
    type Out = Result<CpuFusedCountState>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let (mut state, idx) = input;
        let expert_id = self.indices_data.get(idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "cpu fused count: index {} out of bounds (len {})",
                idx,
                self.indices_data.len()
            ))
        })?;
        let expert = *expert_id as usize;
        if expert < self.num_experts {
            let len = state.expert_counts.len();
            if expert >= len {
                return Err(paramecia_core::Error::Msg(format!(
                    "cpu fused count: expert {} out of bounds (len {})",
                    expert, len
                )));
            }
            state.expert_counts[expert] += 1;
        }
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl<'a> Vis for CpuFusedCountExpertStepOp<'a> {
    fn visualize() -> Graph {
        Graph::custom_leaf("CpuFusedCountExpertStep")
            .with_output_type::<Result<CpuFusedCountState>>()
    }
}

#[derive(Debug)]
struct CpuFusedAssignPairStepOp<'a> {
    indices_data: &'a [u32],
    weights_data: &'a [f32],
    num_experts: usize,
    top_k: usize,
}
impl<'a> CpuFusedAssignPairStepOp<'a> {
    fn new(
        indices_data: &'a [u32],
        weights_data: &'a [f32],
        num_experts: usize,
        top_k: usize,
    ) -> Self {
        Self {
            indices_data,
            weights_data,
            num_experts,
            top_k,
        }
    }
}
#[primitive(property = Arrow)]
impl<'a> Combinator for CpuFusedAssignPairStepOp<'a> {
    type In = (CpuFusedAssignState, usize);
    type Out = Result<CpuFusedAssignState>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let (mut state, idx) = input;
        let expert = *self.indices_data.get(idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "cpu fused assign: index {} out of bounds for indices (len {})",
                idx,
                self.indices_data.len()
            ))
        })? as usize;
        if expert >= self.num_experts {
            return Ok(state);
        }
        let weight = *self.weights_data.get(idx).ok_or_else(|| {
            paramecia_core::Error::Msg(format!(
                "cpu fused assign: index {} out of bounds for weights (len {})",
                idx,
                self.weights_data.len()
            ))
        })?;
        let token_id = if self.top_k == 0 { 0 } else { idx / self.top_k };
        let token_id_u32 = u32::try_from(token_id).map_err(|_| {
            paramecia_core::Error::Msg(format!(
                "cpu fused assign: token index {} does not fit in u32",
                token_id
            ))
        })?;
        let tokens_len = state.token_ids_by_expert.len();
        if expert >= tokens_len {
            return Err(paramecia_core::Error::Msg(format!(
                "cpu fused assign: expert {} out of bounds (token buckets len {})",
                expert, tokens_len
            )));
        }
        let weights_len = state.routing_weights_by_expert.len();
        if expert >= weights_len {
            return Err(paramecia_core::Error::Msg(format!(
                "cpu fused assign: expert {} out of bounds (weight buckets len {})",
                expert, weights_len
            )));
        }
        state.token_ids_by_expert[expert].push(token_id_u32);
        state.routing_weights_by_expert[expert].push(weight);
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl<'a> Vis for CpuFusedAssignPairStepOp<'a> {
    fn visualize() -> Graph {
        Graph::custom_leaf("CpuFusedAssignPairStep")
            .with_output_type::<Result<CpuFusedAssignState>>()
    }
}

#[derive(Debug, Clone, Default)]
struct CpuFusedCollectAssignmentsStepOp;
#[primitive(property = Arrow)]
impl Combinator for CpuFusedCollectAssignmentsStepOp {
    type In = (CpuFusedCollectState, usize);
    type Out = Result<CpuFusedCollectState>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let (mut state, expert_idx) = input;
        if expert_idx >= state.token_ids_by_expert.len()
            || expert_idx >= state.routing_weights_by_expert.len()
        {
            return Err(paramecia_core::Error::Msg(format!(
                "cpu fused collect: expert {} out of bounds (tokens {}, weights {})",
                expert_idx,
                state.token_ids_by_expert.len(),
                state.routing_weights_by_expert.len()
            )));
        }
        if state.token_ids_by_expert[expert_idx].is_empty() {
            return Ok(state);
        }
        let token_ids = std::mem::take(&mut state.token_ids_by_expert[expert_idx]);
        let routing_weights = std::mem::take(&mut state.routing_weights_by_expert[expert_idx]);
        if token_ids.len() != routing_weights.len() {
            return Err(paramecia_core::Error::Msg(format!(
                "cpu fused collect: assignment/weight len mismatch for expert {} ({} vs {})",
                expert_idx,
                token_ids.len(),
                routing_weights.len()
            )));
        }
        state
            .expert_assignments
            .push((expert_idx, token_ids, routing_weights));
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl Vis for CpuFusedCollectAssignmentsStepOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("CpuFusedCollectAssignmentsStep")
            .with_output_type::<Result<CpuFusedCollectState>>()
    }
}

pub(super) type DispatchShapeOp = MapErr<
    Dims3Op<Shape3<B, N, S>>,
    THidden,
    (usize, usize, usize),
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type DispatchHiddenFlatFlow = MapErr<
    FlattenPrefix2Op<Shape3<B, N, S>, Shape2<T, S>>,
    THidden,
    THiddenFlat,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type DispatchTopFlatFlow = MapErr<
    FlattenPrefix2Op<Shape3<B, N, Tk>, Shape2<T, Tk>>,
    TTopIndices,
    TTopFlat,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type DispatchShapeHiddenPair = ((usize, usize, usize), THiddenFlat);
type DispatchPreparedInput = (DispatchShapeHiddenPair, TTopFlat, TTopFlat);
type DispatchPreparedResult = std::result::Result<DispatchPreparedInput, paramecia_core::Error>;
impl TryFrom<DispatchPreparedInput> for MoeDispatchPrepared {
    type Error = paramecia_core::Error;

    fn try_from(input: DispatchPreparedInput) -> std::result::Result<Self, Self::Error> {
        let (((batch_size, seq_len, hidden_dim), hidden_flat), indices_flat, weights_flat) = input;
        Ok(Self {
            batch_size,
            seq_len,
            hidden_dim,
            hidden_flat,
            indices_flat,
            weights_flat,
        })
    }
}
type BuildDispatchPreparedOp =
    TryFromOp<DispatchPreparedInput, MoeDispatchPrepared, paramecia_core::Error>;

type HiddenDispatchPrepFlow = Then<
    Fanout<THidden>,
    Then<
        Zip<DispatchShapeOp, DispatchHiddenFlatFlow>,
        ZipOk<(usize, usize, usize), THiddenFlat, paramecia_core::Error>,
    >,
>;
fn hidden_dispatch_prep_flow() -> HiddenDispatchPrepFlow {
    Then::new(
        Fanout::default(),
        Then::new(
            Zip::new(
                MapErr::new(Dims3Op::default()),
                MapErr::new(FlattenPrefix2Op::default()),
            ),
            ZipOk::default(),
        ),
    )
}

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoeDispatchPrep(
    Zip3<HiddenDispatchPrepFlow, DispatchTopFlatFlow, DispatchTopFlatFlow>,
    ZipOk3<DispatchShapeHiddenPair, TTopFlat, TTopFlat, paramecia_core::Error>,
    LiftResult<BuildDispatchPreparedOp, DispatchPreparedResult, MoeDispatchPrepared>,
);
impl MoeDispatchPrep {
    pub(super) fn new(_num_experts_per_tok: usize) -> Self {
        Self(
            Zip3::new(
                hidden_dispatch_prep_flow(),
                MapErr::new(FlattenPrefix2Op::default()),
                MapErr::new(FlattenPrefix2Op::default()),
            ),
            ZipOk3::default(),
            LiftResult::new(TryFromOp::default()),
        )
    }
}
impl std::fmt::Debug for MoeDispatchPrep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeDispatchPrep").finish()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MoePathUseCpuFusedCondOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for MoePathUseCpuFusedCondOp {
    type In = MoePathCtx;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        !input.training_mode
            && input.all_experts_on_cpu
            && input.hidden_on_gpu
            && input.hidden_is_vulkan
    }
}
#[primitive(property = Visualize)]
impl Vis for MoePathUseCpuFusedCondOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MoePathUseCpuFusedCond").with_output_type::<bool>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MoePathUseBatchedCondOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for MoePathUseBatchedCondOp {
    type In = MoePathCtx;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        !input.training_mode && input.supports_batched
    }
}
#[primitive(property = Visualize)]
impl Vis for MoePathUseBatchedCondOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MoePathUseBatchedCond").with_output_type::<bool>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MoePathUseGpuGroupedCondOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for MoePathUseGpuGroupedCondOp {
    type In = MoePathCtx;
    type Out = bool;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        !input.training_mode && input.all_experts_on_cpu && input.hidden_on_gpu
    }
}
#[primitive(property = Visualize)]
impl Vis for MoePathUseGpuGroupedCondOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MoePathUseGpuGroupedCond").with_output_type::<bool>()
    }
}

type MoePathPair = (bool, MoePathCtx);
type MoePathPrimarySelector = Then<First<bool, MoePathCtx>, BoolToIndex2>;
type MoePathGpuGroupedSwitch =
    Switch<BoolToIndex2, ConstOut<bool, MoeDispatchPath>, ConstOut<bool, MoeDispatchPath>>;
type MoePathAfterBatchedFalse =
    Then<Second<bool, MoePathCtx>, Then<MoePathUseGpuGroupedCondOp, MoePathGpuGroupedSwitch>>;
type MoePathBatchedPairFlow =
    Then<Fanout<MoePathCtx>, Zip<MoePathUseBatchedCondOp, Identity<MoePathCtx>>>;
type MoePathBatchedSwitch = Switch<
    MoePathPrimarySelector,
    ConstOut<MoePathPair, MoeDispatchPath>,
    MoePathAfterBatchedFalse,
>;
type MoePathAfterCpuFalse =
    Then<Second<bool, MoePathCtx>, Then<MoePathBatchedPairFlow, MoePathBatchedSwitch>>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoePathSelect(
    Fanout<MoePathCtx>,
    Zip<MoePathUseCpuFusedCondOp, Identity<MoePathCtx>>,
    Switch<MoePathPrimarySelector, ConstOut<MoePathPair, MoeDispatchPath>, MoePathAfterCpuFalse>,
);
impl MoePathSelect {
    pub(super) fn new() -> Self {
        Self(
            Fanout::default(),
            Zip::new(MoePathUseCpuFusedCondOp, Identity::default()),
            Switch::new(
                Then::new(First::default(), BoolToIndex2),
                ConstOut::new(MoeDispatchPath::CpuFused),
                Then::new(
                    Second::default(),
                    Then::new(
                        Then::new(
                            Fanout::default(),
                            Zip::new(MoePathUseBatchedCondOp, Identity::default()),
                        ),
                        Switch::new(
                            Then::new(First::default(), BoolToIndex2),
                            ConstOut::new(MoeDispatchPath::Batched),
                            Then::new(
                                Second::default(),
                                Then::new(
                                    MoePathUseGpuGroupedCondOp,
                                    Switch::new(
                                        BoolToIndex2,
                                        ConstOut::new(MoeDispatchPath::GpuGrouped),
                                        ConstOut::new(MoeDispatchPath::Sequential),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    }
}

type MoePathCfg = (bool, bool, bool);
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct MoePathExtractHiddenOp;
#[primitive(property = Arrow)]
impl<Ctx> Combinator<Ctx> for MoePathExtractHiddenOp {
    type In = MoeDispatchPrepared;
    type Out = THiddenFlat;

    fn forward(&mut self, _ctx: &mut Ctx, input: Self::In) -> Self::Out {
        input.hidden_flat
    }
}
#[primitive(property = Visualize)]
impl Vis for MoePathExtractHiddenOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MoePathExtractHidden").with_output_type::<THiddenFlat>()
    }
}
type MoePathDeviceOp = TensorDeviceInfoOp<Shape2<T, S>>;
type MoePathCfgInjectOp = InjectConst<THiddenFlat, MoePathCfg>;
type MoePathCfgFlow = Then<MoePathCfgInjectOp, Second<THiddenFlat, MoePathCfg>>;
type MoePathInfoCfgZip = Zip<MoePathDeviceOp, MoePathCfgFlow>;
impl From<(TensorDeviceInfo, MoePathCfg)> for MoePathCtx {
    fn from(input: (TensorDeviceInfo, MoePathCfg)) -> Self {
        let (device_info, (training_mode, supports_batched, all_experts_on_cpu)) = input;
        Self {
            training_mode,
            supports_batched,
            all_experts_on_cpu,
            hidden_on_gpu: device_info.is_gpu,
            hidden_is_vulkan: device_info.is_vulkan,
        }
    }
}
type MoePathBuildOp = FromOp<(TensorDeviceInfo, MoePathCfg), MoePathCtx>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoePathContext(
    MoePathExtractHiddenOp,
    Fanout<THiddenFlat>,
    MoePathInfoCfgZip,
    MoePathBuildOp,
);
impl MoePathContext {
    pub(super) fn new(
        training_mode: bool,
        supports_batched: bool,
        all_experts_on_cpu: bool,
    ) -> Self {
        Self(
            MoePathExtractHiddenOp,
            Fanout::default(),
            Zip::new(
                TensorDeviceInfoOp::default(),
                Then::new(
                    InjectConst::new((training_mode, supports_batched, all_experts_on_cpu)),
                    Second::default(),
                ),
            ),
            FromOp::default(),
        )
    }
}

pub(super) type MoePathResolveFlow = Then<MoePathContext, MoePathSelect>;
pub(super) fn moe_path_resolve_flow(
    training_mode: bool,
    supports_batched: bool,
    all_experts_on_cpu: bool,
) -> MoePathResolveFlow {
    Then::new(
        MoePathContext::new(training_mode, supports_batched, all_experts_on_cpu),
        MoePathSelect::new(),
    )
}

type SharedGateProjFlow = MapErr<
    QMatMulOp<Shape2<SI, S>, Shape3<B, N, S>, Shape3<B, N, SI>>,
    THidden,
    TIntermediate,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedUpProjFlow = MapErr<
    QMatMulOp<Shape2<SI, S>, Shape3<B, N, S>, Shape3<B, N, SI>>,
    THidden,
    TIntermediate,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedSiluFlow = MapErr<
    SiluOp<Shape3<B, N, SI>>,
    TIntermediate,
    TIntermediate,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedMulFlow = MapErr<
    paramecia_tensor::broadcast_mul::BroadcastMulOp<Shape3<B, N, SI>, Shape3<B, N, SI>>,
    (TIntermediate, TIntermediate),
    TIntermediate,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedContigFlow = MapErr<
    ContiguousOp<Shape3<B, N, SI>>,
    TIntermediate,
    TIntermediate,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedDownProjFlow = MapErr<
    QMatMulOp<Shape2<S, SI>, Shape3<B, N, SI>, Shape3<B, N, S>>,
    TIntermediate,
    THidden,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type SharedSiluMulFlow = Then<
    Zip<SharedSiluFlow, WrapOk<TIntermediate, paramecia_core::Error>>,
    Then<
        ZipOk<TIntermediate, TIntermediate, paramecia_core::Error>,
        LiftResult<
            SharedMulFlow,
            std::result::Result<(TIntermediate, TIntermediate), paramecia_core::Error>,
            TIntermediate,
        >,
    >,
>;
fn shared_silu_mul_flow() -> SharedSiluMulFlow {
    Then::new(
        Zip::new(MapErr::new(SiluOp::default()), WrapOk::default()),
        Then::new(
            ZipOk::default(),
            LiftResult::new(MapErr::new(
                paramecia_tensor::broadcast_mul::BroadcastMulOp::default(),
            )),
        ),
    )
}

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct SharedFfn(
    Fanout<THidden>,
    Zip<SharedGateProjFlow, SharedUpProjFlow>,
    ZipOk<TIntermediate, TIntermediate, paramecia_core::Error>,
    LiftResult<
        SharedSiluMulFlow,
        std::result::Result<(TIntermediate, TIntermediate), paramecia_core::Error>,
        TIntermediate,
    >,
    LiftResult<
        SharedContigFlow,
        std::result::Result<TIntermediate, paramecia_core::Error>,
        TIntermediate,
    >,
    LiftResult<
        SharedDownProjFlow,
        std::result::Result<TIntermediate, paramecia_core::Error>,
        THidden,
    >,
);
impl SharedFfn {
    fn new(
        gate_proj: TQMatMul<Shape2<SI, S>>,
        up_proj: TQMatMul<Shape2<SI, S>>,
        down_proj: TQMatMul<Shape2<S, SI>>,
    ) -> Self {
        Self(
            Fanout::default(),
            Zip::new(
                MapErr::new(QMatMulOp::new(gate_proj)),
                MapErr::new(QMatMulOp::new(up_proj)),
            ),
            ZipOk::default(),
            LiftResult::new(shared_silu_mul_flow()),
            LiftResult::new(MapErr::new(ContiguousOp::default())),
            LiftResult::new(MapErr::new(QMatMulOp::new(down_proj))),
        )
    }
}
impl std::fmt::Debug for SharedFfn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedFfn").finish()
    }
}

type SharedGateAlignFlow = MapErr<
    CastLikeOp<Shape3<B, N, S>, Shape2<U1, S>>,
    (THidden, TSharedGate),
    (THidden, TSharedGate),
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedGateMulFlow = MapErr<
    BroadcastMulOp<Shape3<B, N, S>, Shape2<U1, S>>,
    (THidden, TSharedGate),
    THidden,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedGateScoreFlow = MapErr<
    SumDimOp<Shape3<B, N, S>, U2>,
    THidden,
    TGateScore,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedGateSigmoidFlow = MapErr<
    SigmoidOp<Shape3<B, N, U1>>,
    TGateScore,
    TGateScore,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type SharedGateComputeFlow = Then<
    SharedGateAlignFlow,
    Then<
        LiftResult<
            SharedGateMulFlow,
            std::result::Result<(THidden, TSharedGate), paramecia_core::Error>,
            THidden,
        >,
        Then<
            LiftResult<
                SharedGateScoreFlow,
                std::result::Result<THidden, paramecia_core::Error>,
                TGateScore,
            >,
            LiftResult<
                SharedGateSigmoidFlow,
                std::result::Result<TGateScore, paramecia_core::Error>,
                TGateScore,
            >,
        >,
    >,
>;
fn shared_gate_compute_flow() -> SharedGateComputeFlow {
    Then::new(
        MapErr::new(CastLikeOp::default()),
        Then::new(
            LiftResult::new(MapErr::new(BroadcastMulOp::default())),
            Then::new(
                LiftResult::new(MapErr::new(SumDimOp::default())),
                LiftResult::new(MapErr::new(SigmoidOp::default())),
            ),
        ),
    )
}

type SharedOutputCastFlow = MapErr<
    CastLikeOp<Shape3<B, N, U1>, Shape3<B, N, S>>,
    (TGateScore, THidden),
    (TGateScore, THidden),
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type SharedOutputMulFlow = MapErr<
    BroadcastMulOp<Shape3<B, N, U1>, Shape3<B, N, S>>,
    (TGateScore, THidden),
    THidden,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type SharedGateApplyFlow = Then<
    SharedOutputCastFlow,
    LiftResult<
        SharedOutputMulFlow,
        std::result::Result<(TGateScore, THidden), paramecia_core::Error>,
        THidden,
    >,
>;
fn shared_gate_apply_flow() -> SharedGateApplyFlow {
    Then::new(
        MapErr::new(CastLikeOp::default()),
        LiftResult::new(MapErr::new(BroadcastMulOp::default())),
    )
}

#[allow(dead_code)]
struct SharedExpertForwardGraph;
#[primitive(property = Visualize)]
impl Vis for SharedExpertForwardGraph {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "SharedExpertForwardGraph",
            <SharedExpertForward as Vis>::visualize(),
        )
    }
}

type SharedGateInjectOp = InjectConst<THidden, TSharedGate>;

type SharedExpertGatePathFlow = Then<SharedGateInjectOp, SharedGateComputeFlow>;
fn shared_expert_gate_path_flow(shared_gate: TSharedGate) -> SharedExpertGatePathFlow {
    Then::new(InjectConst::new(shared_gate), shared_gate_compute_flow())
}

type CE = paramecia_core::Error;

pub(super) type ResidualAddHiddenFlow = MapErr<
    ResidualAddOp<Shape3<B, N, S>>,
    (THidden, THidden),
    THidden,
    paramecia_tensor::Error,
    CE,
>;

/// Full shared expert forward flow (excluding initial contiguous, handled in forward()).
///
/// Data flow:
/// ```text
/// THidden → fanout → ┬─ gate_inject(shared_gate) → gate_compute → TGateScore
///                     └─ ffn → THidden
///          → join results → gate_apply((TGateScore, THidden)) → THidden
/// ```
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct SharedExpertForward(
    // Step 0: THidden → (THidden, THidden)
    Fanout<THidden>,
    // Step 1: (THidden, THidden) → (Result<TGateScore, CE>, Result<THidden, CE>)
    Zip<SharedExpertGatePathFlow, SharedFfn>,
    // Step 2: (Result<TGateScore, CE>, Result<THidden, CE>) → Result<(TGateScore, THidden), CE>
    ZipOk<TGateScore, THidden, CE>,
    // Step 3: Result<(TGateScore, THidden), CE> → Result<THidden, CE>
    LiftResult<SharedGateApplyFlow, std::result::Result<(TGateScore, THidden), CE>, THidden>,
);
impl SharedExpertForward {
    fn new(shared_gate: TSharedGate, ffn: SharedFfn) -> Self {
        Self(
            Fanout::default(),
            Zip::new(shared_expert_gate_path_flow(shared_gate), ffn),
            ZipOk::default(),
            LiftResult::new(shared_gate_apply_flow()),
        )
    }
}
impl std::fmt::Debug for SharedExpertForward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedExpertForward").finish()
    }
}

type RouteMaskApplyOp = MapErr<
    BroadcastAddOp<Shape3<B, N, E>, Shape1<E>>,
    (TRouterLogits, TExpertVec),
    TRouterLogits,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type RouteMaskOp = Then<
    InjectConst<TRouterLogits, Option<TExpertVec>>,
    OptionThen<RouteMaskApplyOp, TRouterLogits, TExpertVec, paramecia_core::Error>,
>;

pub(super) type RouteTopKFromLogitsOp = MapErr<
    TensorTopkFromLogitsOp<Shape3<B, N, E>, Shape3<B, N, Tk>, Shape3<B, N, Tk>>,
    TRouterLogits,
    (TTopWeights, TTopIndices),
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type RouteClampFlow = MapErr<
    ClampOp<Shape3<B, N, E>>,
    TRouterLogits,
    TRouterLogits,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type RouteAttachTopKFlow =
    FlattenTripleResult<TRouterLogits, TTopWeights, TTopIndices, paramecia_core::Error>;

type RouteTopKBranchFlow = Then<
    RouteClampFlow,
    LiftResult<
        RouteTopKFromLogitsOp,
        std::result::Result<TRouterLogits, paramecia_core::Error>,
        (TTopWeights, TTopIndices),
    >,
>;
fn route_topk_branch_flow(num_experts_per_tok: usize) -> RouteTopKBranchFlow {
    Then::new(
        MapErr::new(ClampOp::new(-100.0, 100.0)),
        LiftResult::new(MapErr::new(TensorTopkFromLogitsOp::new(
            num_experts_per_tok,
        ))),
    )
}

type RouteTopKPostMaskFlow = Then<
    Fanout<TRouterLogits>,
    Then<
        Zip<WrapOk<TRouterLogits, paramecia_core::Error>, RouteTopKBranchFlow>,
        Then<
            ZipOk<TRouterLogits, (TTopWeights, TTopIndices), paramecia_core::Error>,
            LiftResult<
                RouteAttachTopKFlow,
                std::result::Result<
                    (TRouterLogits, (TTopWeights, TTopIndices)),
                    paramecia_core::Error,
                >,
                (TRouterLogits, TTopWeights, TTopIndices),
            >,
        >,
    >,
>;
fn route_topk_post_mask_flow(num_experts_per_tok: usize) -> RouteTopKPostMaskFlow {
    Then::new(
        Fanout::default(),
        Then::new(
            Zip::new(
                WrapOk::default(),
                route_topk_branch_flow(num_experts_per_tok),
            ),
            Then::new(
                ZipOk::default(),
                LiftResult::new(RouteAttachTopKFlow::default()),
            ),
        ),
    )
}

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct RouteTopK(
    RouteMaskOp,
    LiftResult<
        RouteTopKPostMaskFlow,
        std::result::Result<TRouterLogits, paramecia_core::Error>,
        (TRouterLogits, TTopWeights, TTopIndices),
    >,
);
impl RouteTopK {
    pub(super) fn new(num_experts_per_tok: usize, expert_mask: Option<TExpertVec>) -> Self {
        Self(
            Then::new(
                InjectConst::new(expert_mask),
                OptionThen::new(MapErr::new(BroadcastAddOp::default())),
            ),
            LiftResult::new(route_topk_post_mask_flow(num_experts_per_tok)),
        )
    }
}
impl std::fmt::Debug for RouteTopK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouteTopK").finish()
    }
}

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoeRoute(MoeRouteProjectOp, MoeRouteTopKOp);
impl std::fmt::Debug for MoeRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeRoute").finish()
    }
}
impl MoeRoute {
    pub(super) fn new(
        gate: TQMatMul<Shape2<E, S>>,
        num_experts_per_tok: usize,
        expert_mask: Option<TExpertVec>,
    ) -> Self {
        Self(
            MapErr::new(QMatMulOp::new(gate)),
            LiftResult::new(RouteTopK::new(num_experts_per_tok, expert_mask)),
        )
    }
}

type MoeRouteProjectOp = MapErr<
    QMatMulOp<Shape2<E, S>, Shape3<B, N, S>, Shape3<B, N, E>>,
    THidden,
    TRouterLogits,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MoeRouteTopKOp = MapOk<
    RouteTopK,
    TRouterLogits,
    (TRouterLogits, TTopWeights, TTopIndices),
    paramecia_core::Error,
>;

type MoeRouteJoinResult = std::result::Result<
    (THidden, (TRouterLogits, TTopWeights, TTopIndices)),
    paramecia_core::Error,
>;
type MoeRouteJoinInput = (THidden, (TRouterLogits, TTopWeights, TTopIndices));

pub(super) struct MoeRouteRemapOutput {
    hidden_states: THidden,
    router_logits: Tensor,
    top_weights: TTopWeights,
    remapped_indices: TTopIndices,
    top_indices: Tensor,
}

impl TryFrom<(MoeRouteJoinInput, TTopIndices)> for MoeRouteRemapOutput {
    type Error = paramecia_core::Error;

    fn try_from(value: (MoeRouteJoinInput, TTopIndices)) -> std::result::Result<Self, Self::Error> {
        let (
            (hidden_states, (router_logits_typed, top_weights, top_indices_typed)),
            remapped_indices,
        ) = value;
        Ok(Self {
            hidden_states,
            router_logits: router_logits_typed.into_inner(),
            top_weights,
            remapped_indices,
            top_indices: top_indices_typed.inner().clone(),
        })
    }
}
type MoeTopIndicesFromRouteJoinOp = Then<
    Second<THidden, (TRouterLogits, TTopWeights, TTopIndices)>,
    Third<TRouterLogits, TTopWeights, TTopIndices>,
>;
type MoeRemapFromRouteIndicesFlow = Then<MoeTopIndicesFromRouteJoinOp, RemapTopIndicesFlow>;
type MoeRemapFromRouteZip =
    Zip<WrapOk<MoeRouteJoinInput, paramecia_core::Error>, MoeRemapFromRouteIndicesFlow>;
type MoeRemapFromRouteBuild = LiftResult<
    TryFromOp<(MoeRouteJoinInput, TTopIndices), MoeRouteRemapOutput, paramecia_core::Error>,
    std::result::Result<(MoeRouteJoinInput, TTopIndices), paramecia_core::Error>,
    MoeRouteRemapOutput,
>;
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoeRemapFromRouteStep(
    Fanout<MoeRouteJoinInput>,
    MoeRemapFromRouteZip,
    ZipOk<MoeRouteJoinInput, TTopIndices, paramecia_core::Error>,
    MoeRemapFromRouteBuild,
);
impl MoeRemapFromRouteStep {
    fn new(expert_remap: Option<TExpertVec>) -> Self {
        Self(
            Fanout::default(),
            Zip::new(
                WrapOk::default(),
                Then::new(
                    Then::new(
                        Second::<THidden, (TRouterLogits, TTopWeights, TTopIndices)>::default(),
                        Third::<TRouterLogits, TTopWeights, TTopIndices>::default(),
                    ),
                    MapErr::new(TensorRemapIndicesOp::new(expert_remap)),
                ),
            ),
            ZipOk::default(),
            LiftResult::new(TryFromOp::default()),
        )
    }
}
impl std::fmt::Debug for MoeRemapFromRouteStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeRemapFromRouteStep").finish()
    }
}

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
pub(super) struct MoeRouteRemap(
    Fanout<THidden>,
    Zip<WrapOk<THidden, paramecia_core::Error>, MoeRoute>,
    ZipOk<THidden, (TRouterLogits, TTopWeights, TTopIndices), paramecia_core::Error>,
    LiftResult<MoeRemapFromRouteStep, MoeRouteJoinResult, MoeRouteRemapOutput>,
);
impl MoeRouteRemap {
    pub(super) fn new(
        gate: TQMatMul<Shape2<E, S>>,
        num_experts_per_tok: usize,
        expert_mask: Option<TExpertVec>,
        expert_remap: Option<TExpertVec>,
    ) -> Self {
        Self(
            Fanout::default(),
            Zip::new(
                WrapOk::default(),
                MoeRoute::new(gate, num_experts_per_tok, expert_mask),
            ),
            ZipOk::default(),
            LiftResult::new(MoeRemapFromRouteStep::new(expert_remap)),
        )
    }
}
impl std::fmt::Debug for MoeRouteRemap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeRouteRemap").finish()
    }
}

#[allow(dead_code)]
pub(super) struct MoeForwardGraph;
#[primitive(property = Visualize)]
impl Vis for MoeForwardGraph {
    fn visualize() -> Graph {
        let g = Graph::sequence(
            <MoeRouteRemap as Vis>::visualize(),
            <MoeDispatchPrep as Vis>::visualize(),
        );
        let g = Graph::sequence(g, <MoePathResolveFlow as Vis>::visualize());
        let g = Graph::sequence(g, <MoeGroupAssignments as Vis>::visualize());
        let g = Graph::sequence(g, <MoePrefetchSelect as Vis>::visualize());
        let g = Graph::sequence(
            g,
            <TryFoldVec<PrefetchInferenceExpertVisOp, (), usize, paramecia_core::Error> as Vis>::visualize(),
        );
        let g = Graph::sequence(
            g,
            <TryFoldVec<PrefetchTrainingExpertVisOp, (), usize, paramecia_core::Error> as Vis>::visualize(),
        );
        let g = Graph::sequence(g, <MoeSequentialExecSelect as Vis>::visualize());
        let g = Graph::sequence(
            g,
            <TryFoldVec<
                ApplyGpuCachedExpertVisOp,
                GpuCachedApplyState,
                (usize, CachedMatmuls),
                paramecia_core::Error,
            > as Vis>::visualize(),
        );
        let g = Graph::sequence(
            g,
            <TryFoldVec<ApplySequentialExpertVisOp, Tensor, usize, paramecia_core::Error> as Vis>::visualize(),
        );
        let g = Graph::sequence(
            g,
            <TryFoldRange<
                CpuFusedCountExpertStepOp<'static>,
                CpuFusedCountState,
                paramecia_core::Error,
            > as Vis>::visualize(),
        );
        let g = Graph::sequence(
            g,
            <TryFoldRange<
                CpuFusedAssignPairStepOp<'static>,
                CpuFusedAssignState,
                paramecia_core::Error,
            > as Vis>::visualize(),
        );
        let g = Graph::sequence(
            g,
            <TryFoldRange<
                CpuFusedCollectAssignmentsStepOp,
                CpuFusedCollectState,
                paramecia_core::Error,
            > as Vis>::visualize(),
        );
        let g = Graph::sequence(g, <SharedExpertForwardGraph as Vis>::visualize());
        Graph::wrap_custom_subgraph("MoeForwardGraph", g)
    }
}

#[allow(dead_code)]
pub(super) struct MoePrefetchGraph;
#[primitive(property = Visualize)]
impl Vis for MoePrefetchGraph {
    fn visualize() -> Graph {
        let mode_select = <MoePrefetchSelect as Vis>::visualize();
        let mode_paths = Graph::zip_custom(
            "MoePrefetchModePaths",
            vec![
                (
                    "inference",
                    <TryFoldVec<PrefetchInferenceExpertVisOp, (), usize, paramecia_core::Error> as Vis>::visualize(),
                ),
                (
                    "training",
                    <TryFoldVec<PrefetchTrainingExpertVisOp, (), usize, paramecia_core::Error> as Vis>::visualize(),
                ),
                ("none", Graph::custom_leaf("NoPrefetch")),
            ],
        );
        Graph::wrap_custom_subgraph("MoePrefetchGraph", Graph::sequence(mode_select, mode_paths))
    }
}

pub(super) struct MoeExperts {
    pub(super) gate_exps: ExpertWeightTensor,
    pub(super) up_exps: ExpertWeightTensor,
    pub(super) down_exps: ExpertWeightTensor,
    pub(super) act_fn: Activation,
    pub(super) span: tracing::Span,
    pub(super) cache: Option<ExpertCache>,
    pub(super) training_cache: Option<ExpertCache>,
    pub(super) compute_device: Device,
    pub(super) custom_gate_block_mults: Option<TBlockMults>,
    pub(super) custom_up_block_mults: Option<TBlockMults>,
    pub(super) custom_down_block_mults: Option<TBlockMults>,
    /// GPU hot expert cache - keeps frequently used experts on GPU
    /// even when main expert weights are on CPU
    pub(super) gpu_hot_cache: Option<GpuHotExpertCache>,
    /// The GPU device for hot caching (may differ from compute_device)
    pub(super) gpu_device: Option<Device>,
}

impl std::fmt::Debug for MoeExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeExperts")
            .field("gate_exps", &self.gate_exps)
            .field("up_exps", &self.up_exps)
            .field("down_exps", &self.down_exps)
            .field("act_fn", &self.act_fn)
            .field("compute_device", &self.compute_device)
            .field("gpu_hot_cache", &self.gpu_hot_cache)
            .finish()
    }
}

impl MoeExperts {
    fn load_expert_tensor_3d<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        exps_suffix: &str,
        dense_suffix: &str,
    ) -> Result<SharedQTensor> {
        let exps_name = format!("{}.{}", prefix, exps_suffix);
        if gg.try_tensor(&exps_name)?.is_some() {
            return gg.shared_expert_tensor(&exps_name);
        }
        let dense_name = format!("{}.{}", prefix, dense_suffix);
        gg.shared_expert_tensor(&dense_name)?.unsqueeze(0)
    }

    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        compute_device: &Device,
        cache_capacity: usize,
    ) -> Result<Self> {
        Self::new_with_gpu_cache(gg, prefix, compute_device, cache_capacity, None)
    }

    pub(super) fn new_with_gpu_cache<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        compute_device: &Device,
        cache_capacity: usize,
        gpu_device: Option<Device>,
    ) -> Result<Self> {
        let gate_exps = ExpertWeightTensor::new(
            Self::load_expert_tensor_3d(gg, prefix, "ffn_gate_exps.weight", "ffn_gate.weight")?,
            "gate_exps",
        )?;
        let up_exps = ExpertWeightTensor::new(
            Self::load_expert_tensor_3d(gg, prefix, "ffn_up_exps.weight", "ffn_up.weight")?,
            "up_exps",
        )?;
        let down_exps = ExpertWeightTensor::new(
            Self::load_expert_tensor_3d(gg, prefix, "ffn_down_exps.weight", "ffn_down.weight")?,
            "down_exps",
        )?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "moe-experts");

        // Get number of experts from shape
        let num_experts = gate_exps
            .read()
            .unwrap()
            .shape()
            .dims()
            .first()
            .copied()
            .unwrap_or(64);

        let cache = if cache_capacity == 0 {
            None
        } else if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
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

        // Create GPU hot cache if:
        // 1. We have a GPU device
        // 2. Experts are on CPU (so we benefit from caching hot experts on GPU)
        let gpu_hot_cache = if let Some(ref gpu_dev) = gpu_device {
            if is_gpu_device(gpu_dev) && !is_gpu_device(compute_device) {
                // Cache up to 16 hot experts on GPU (configurable)
                Some(GpuHotExpertCache::new(gpu_dev.clone(), num_experts, 16))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            gate_exps,
            up_exps,
            down_exps,
            act_fn,
            span,
            cache,
            training_cache,
            compute_device: compute_device.clone(),
            custom_gate_block_mults: None,
            custom_up_block_mults: None,
            custom_down_block_mults: None,
            gpu_hot_cache,
            gpu_device,
        })
    }

    /// Enable GPU hot caching for CPU-offloaded experts
    pub(super) fn enable_gpu_hot_cache(&mut self, gpu_device: Device, capacity: usize) {
        let num_experts = self
            .gate_exps
            .read()
            .unwrap()
            .shape()
            .dims()
            .first()
            .copied()
            .unwrap_or(64);
        if is_gpu_device(&gpu_device) && !is_gpu_device(&self.compute_device) {
            self.gpu_hot_cache = Some(GpuHotExpertCache::new(
                gpu_device.clone(),
                num_experts,
                capacity,
            ));
            self.gpu_device = Some(gpu_device);
        }
    }

    /// Expert-parallel forward pass using fused SwiGLU and parallel matmul
    ///
    /// This processes all (token, expert) pairs in parallel:
    /// 1. Gathers only the active expert weights (reduces memory access)
    /// 2. Uses parallel indexed matmul for gate+up projections
    /// 3. Applies fused SwiGLU activation
    /// 4. Uses parallel indexed matmul for down projection
    /// 5. Scatter-adds weighted expert outputs
    ///
    /// hidden_states: [n_tokens, hidden_dim]
    /// expert_indices: [n_tokens, top_k] (u32)
    /// expert_weights: [n_tokens, top_k] (f32)
    /// Returns: [n_tokens, hidden_dim]
    /// Batched forward pass using indexed_moe_forward CUDA kernels
    ///
    /// This uses the optimized quantized matmul kernels that keep weights
    /// in quantized form, avoiding dequantization overhead.
    ///
    /// hidden_states: [n_tokens, hidden_dim]
    /// expert_indices: [n_tokens, top_k] (u32)
    /// expert_weights: [n_tokens, top_k] (f32)
    /// Returns: [n_tokens, hidden_dim]
    pub(super) fn forward_batched(
        &self,
        hidden_states: &THiddenFlat,
        expert_indices: &TTopFlat,
        expert_weights: &TTopFlat,
        top_k: usize,
    ) -> Result<THiddenFlat> {
        let _enter = self.span.enter();
        let mut dims2 = Dims2Op::<Shape2<T, S>>::default();
        let (n_tokens, hidden_dim) = <Dims2Op<Shape2<T, S>> as Combinator<()>>::forward(
            &mut dims2,
            &mut (),
            hidden_states.clone(),
        )?;

        // Reshape hidden states for indexed_moe_forward: [n_tokens, 1, hidden_dim]
        // Note: We expect F32 input (model uses F32 activations to match llama.cpp)
        let hidden_expanded = hidden_states.inner().unsqueeze(1)?;

        // Fused gate+up+SwiGLU: computes silu(gate @ x) * (up @ x) in a single GPU dispatch.
        // Reduces 4 dispatches + barriers (gate matmul, up matmul, swiglu) to 1.
        // Falls back to separate calls + swiglu if fused kernel not available.
        let gate_lock = self.gate_exps.read().unwrap();
        let up_lock = self.up_exps.read().unwrap();
        let activated = QTensor::indexed_moe_gate_up_swiglu(
            &gate_lock,
            &up_lock,
            &hidden_expanded,
            expert_indices.inner(),
        )?;
        drop(gate_lock);
        drop(up_lock);

        // For down projection, reshape activated for indexed_moe_forward
        // activated: [n_tokens, top_k, n_ff]
        // Need: [n_tokens * top_k, 1, n_ff] with indices [n_tokens * top_k, 1]
        let n_ff = activated.dims()[2];
        let activated_flat = activated.reshape((n_tokens * top_k, 1, n_ff))?;

        // Expand indices: [n_tokens, top_k] -> [n_tokens * top_k, 1]
        let indices_flat = expert_indices.inner().reshape((n_tokens * top_k,))?;
        let indices_expanded = indices_flat.unsqueeze(1)?;

        // Down projection
        let down_out = self
            .down_exps
            .read()
            .unwrap()
            .indexed_moe_forward(&activated_flat, &indices_expanded)?;
        // down_out: [n_tokens * top_k, 1, hidden_dim]
        let down_out = down_out.reshape((n_tokens, top_k, hidden_dim))?;

        // Weight and sum expert outputs
        let weights_expanded = expert_weights.inner().unsqueeze(2)?;
        let weighted = down_out.broadcast_mul(&weights_expanded)?;
        let out_typed = weighted.sum(1)?.try_into()?;
        Ok(out_typed)
    }

    fn build_cpu_fused_expert_assignments(
        &self,
        indices_data: &[u32],
        weights_data: &[f32],
        n_tokens: usize,
        top_k: usize,
        num_experts: usize,
    ) -> Result<CpuFusedExpertAssignments> {
        let expected_pairs = n_tokens.checked_mul(top_k).ok_or_else(|| {
            paramecia_core::Error::Msg("forward_batched_cpu_fused: token/top_k product over".into())
        })?;
        if indices_data.len() != expected_pairs || weights_data.len() != expected_pairs {
            return Err(paramecia_core::Error::Msg(format!(
                "forward_batched_cpu_fused: routing data size mismatch (indices {}, weights {}, expected {})",
                indices_data.len(),
                weights_data.len(),
                expected_pairs
            )));
        }

        let count_step = CpuFusedCountExpertStepOp::new(indices_data, num_experts);
        let mut count_fold = TryFoldRange::<
            CpuFusedCountExpertStepOp<'_>,
            CpuFusedCountState,
            paramecia_core::Error,
        >::new(count_step);
        let count_state = count_fold.traced_forward(
            &mut (),
            (
                CpuFusedCountState {
                    expert_counts: vec![0usize; num_experts],
                },
                0..expected_pairs,
            ),
        )?;

        let assign_state = CpuFusedAssignState {
            token_ids_by_expert: count_state
                .expert_counts
                .iter()
                .map(|&count| Vec::with_capacity(count))
                .collect(),
            routing_weights_by_expert: count_state
                .expert_counts
                .iter()
                .map(|&count| Vec::with_capacity(count))
                .collect(),
        };
        let assign_step =
            CpuFusedAssignPairStepOp::new(indices_data, weights_data, num_experts, top_k);
        let mut assign_fold = TryFoldRange::<
            CpuFusedAssignPairStepOp<'_>,
            CpuFusedAssignState,
            paramecia_core::Error,
        >::new(assign_step);
        let assign_state =
            assign_fold.traced_forward(&mut (), (assign_state, 0..expected_pairs))?;

        let collect_state = CpuFusedCollectState {
            token_ids_by_expert: assign_state.token_ids_by_expert,
            routing_weights_by_expert: assign_state.routing_weights_by_expert,
            expert_assignments: Vec::new(),
        };
        let mut collect_fold = TryFoldRange::<
            CpuFusedCollectAssignmentsStepOp,
            CpuFusedCollectState,
            paramecia_core::Error,
        >::new(CpuFusedCollectAssignmentsStepOp);
        let collect_state =
            collect_fold.traced_forward(&mut (), (collect_state, 0..num_experts))?;
        Ok(collect_state.expert_assignments)
    }

    /// Fused MoE forward that keeps all computation on CPU with minimal GPU transfers.
    ///
    /// For Vulkan (and other GPU backends where quantized matmul falls back to CPU),
    /// the standard forward_batched creates ~9 GPU↔CPU transfers per layer (each
    /// indexed_moe_forward downloads x + ids and uploads result). This fused path:
    /// 1. Downloads hidden_states + indices + weights to CPU once (3 transfers)
    /// 2. Does gate/up matmul → SwiGLU → down matmul → weight+sum ALL on CPU
    /// 3. Uploads final result back to GPU once (1 transfer)
    ///    = 4 transfers instead of ~15+
    pub(super) fn forward_batched_cpu_fused(
        &self,
        hidden_states: &THiddenFlat,
        expert_indices: &TTopFlat,
        expert_weights: &TTopFlat,
        top_k: usize,
    ) -> Result<THiddenFlat> {
        let _enter = self.span.enter();
        let original_device = hidden_states.inner().device().clone();
        let original_dtype = hidden_states.inner().dtype();
        let mut dims2 = Dims2Op::<Shape2<T, S>>::default();
        let (n_tokens, hidden_dim) = <Dims2Op<Shape2<T, S>> as Combinator<()>>::forward(
            &mut dims2,
            &mut (),
            hidden_states.clone(),
        )?;

        // Move everything to CPU - use batched download on Vulkan to reduce flushes.
        #[cfg(feature = "vulkan")]
        paramecia_core::vulkan_backend::device::set_transfer_label("moe_cpu_fused:to_cpu");

        #[cfg(feature = "vulkan")]
        let (hidden_cpu, indices_data, weights_data) = if let Device::Vulkan(vk) =
            hidden_states.inner().device()
        {
            // Batch all 3 downloads into a single flush (saves 2 flushes per MoE layer).
            let hidden_src = hidden_states.inner().contiguous()?;
            let indices_src = if expert_indices.inner().dtype() == DType::U32 {
                expert_indices.inner().contiguous()?
            } else {
                expert_indices.inner().to_dtype(DType::U32)?.contiguous()?
            };
            let weights_src = expert_weights.inner().contiguous()?;

            let h_storage = hidden_src.storage_and_layout().0;
            let i_storage = indices_src.storage_and_layout().0;
            let w_storage = weights_src.storage_and_layout().0;
            let h_buf = match &*h_storage {
                paramecia_core::Storage::Vulkan(s) => s.vk_buffer_arc()?,
                _ => {
                    return Err(paramecia_core::Error::Msg(
                        "forward_batched_cpu_fused: hidden tensor is not Vulkan storage".into(),
                    ));
                }
            };
            let i_buf =
                match &*i_storage {
                    paramecia_core::Storage::Vulkan(s) => s.vk_buffer_arc()?,
                    _ => return Err(paramecia_core::Error::Msg(
                        "forward_batched_cpu_fused: expert_indices tensor is not Vulkan storage"
                            .into(),
                    )),
                };
            let w_buf =
                match &*w_storage {
                    paramecia_core::Storage::Vulkan(s) => s.vk_buffer_arc()?,
                    _ => return Err(paramecia_core::Error::Msg(
                        "forward_batched_cpu_fused: expert_weights tensor is not Vulkan storage"
                            .into(),
                    )),
                };

            let h_bytes = (hidden_src.elem_count() * hidden_src.dtype().size_in_bytes()) as u64;
            let i_bytes = (indices_src.elem_count() * indices_src.dtype().size_in_bytes()) as u64;
            let w_bytes = (weights_src.elem_count() * weights_src.dtype().size_in_bytes()) as u64;

            let h_dl = vk.prepare_download(h_buf, h_bytes)?;
            let i_dl = vk.prepare_download(i_buf, i_bytes)?;
            let w_dl = vk.prepare_download(w_buf, w_bytes)?;

            vk.flush()?; // Single flush for all 3 copies

            let h_data: Vec<f32> = match hidden_src.dtype() {
                DType::F32 => vk.complete_download(h_dl, hidden_src.elem_count())?,
                DType::F16 => vk
                    .complete_download::<u16>(h_dl, hidden_src.elem_count())?
                    .into_iter()
                    .map(|bits| half::f16::from_bits(bits).to_f32())
                    .collect(),
                DType::BF16 => vk
                    .complete_download::<u16>(h_dl, hidden_src.elem_count())?
                    .into_iter()
                    .map(|bits| half::bf16::from_bits(bits).to_f32())
                    .collect(),
                other => {
                    return Err(paramecia_core::Error::Msg(format!(
                        "forward_batched_cpu_fused: unsupported hidden dtype for Vulkan download {:?}",
                        other
                    )));
                }
            };
            let hidden_cpu = Tensor::from_vec(h_data, hidden_src.shape(), &Device::Cpu)?;

            let indices_data: Vec<u32> = vk.complete_download(i_dl, indices_src.elem_count())?;
            let weights_data: Vec<f32> = match weights_src.dtype() {
                DType::F32 => vk.complete_download(w_dl, weights_src.elem_count())?,
                DType::F16 => vk
                    .complete_download::<u16>(w_dl, weights_src.elem_count())?
                    .into_iter()
                    .map(|bits| half::f16::from_bits(bits).to_f32())
                    .collect(),
                DType::BF16 => vk
                    .complete_download::<u16>(w_dl, weights_src.elem_count())?
                    .into_iter()
                    .map(|bits| half::bf16::from_bits(bits).to_f32())
                    .collect(),
                other => {
                    return Err(paramecia_core::Error::Msg(format!(
                        "forward_batched_cpu_fused: unsupported routing-weight dtype for Vulkan download {:?}",
                        other
                    )));
                }
            };

            (hidden_cpu, indices_data, weights_data)
        } else {
            let hidden_cpu = hidden_states
                .inner()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?;
            let indices_data: Vec<u32> = expert_indices
                .inner()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1()?;
            let weights_data: Vec<f32> = expert_weights
                .inner()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            (hidden_cpu, indices_data, weights_data)
        };

        #[cfg(not(feature = "vulkan"))]
        let (hidden_cpu, indices_data, weights_data) = {
            let hidden_cpu = hidden_states
                .inner()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?;
            let indices_data: Vec<u32> = expert_indices
                .inner()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::U32)?
                .flatten_all()?
                .to_vec1()?;
            let weights_data: Vec<f32> = expert_weights
                .inner()
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            (hidden_cpu, indices_data, weights_data)
        };

        let num_experts = self
            .gate_exps
            .read()
            .unwrap()
            .shape()
            .dims()
            .first()
            .copied()
            .ok_or_else(|| {
                paramecia_core::Error::Msg(
                    "forward_batched_cpu_fused: missing expert dimension".into(),
                )
            })?;

        let expert_assignments = self.build_cpu_fused_expert_assignments(
            &indices_data,
            &weights_data,
            n_tokens,
            top_k,
            num_experts,
        )?;

        // CPU-side MoE compute with per-expert parallelism and no giant intermediates.
        let result = crate::expert_pipeline::process_experts_tokio(
            &hidden_cpu,
            &self.gate_exps,
            &self.up_exps,
            &self.down_exps,
            &expert_assignments,
            hidden_dim,
        )?;

        // Upload final result back to GPU (single transfer)
        #[cfg(feature = "vulkan")]
        paramecia_core::vulkan_backend::device::set_transfer_label("moe_cpu_fused:to_gpu");
        let r = result
            .to_dtype(original_dtype)?
            .to_device(&original_device)?;
        #[cfg(feature = "vulkan")]
        paramecia_core::vulkan_backend::device::set_transfer_label("");
        let r_typed = r.try_into()?;
        Ok(r_typed)
    }

    /// Check if batched forward is supported (CUDA + supported quant type + ALL experts on GPU)
    pub(super) fn supports_batched_forward(&self) -> bool {
        use paramecia_core::quantized::GgmlDType;

        // Check if we're on a GPU (CUDA or Metal)
        if !is_gpu_device(&self.compute_device) {
            return false;
        }

        // Check if weights have supported dtype
        let gate_lock = self.gate_exps.read().unwrap();
        let gate_dtype = gate_lock.dtype();
        let supported = matches!(
            gate_dtype,
            GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8_0
        );

        // All expert projections must be on GPU for indexed_moe_forward
        supported
            && is_gpu_device(&gate_lock.device())
            && is_gpu_device(&self.up_exps.read().unwrap().device())
            && is_gpu_device(&self.down_exps.read().unwrap().device())
    }

    pub(super) fn forward_expert(
        &mut self,
        hidden_states: &Tensor,
        expert_idx: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_device = hidden_states.device();
        let training_mode = self.custom_gate_block_mults.is_some();

        // First, check GPU hot cache for non-training mode
        // This is the fast path: expert is already on GPU
        if !training_mode {
            if let Some(ref mut gpu_cache) = self.gpu_hot_cache {
                // Record usage and check if hot
                let is_hot = gpu_cache.record_usage(expert_idx);

                // Try to get from GPU cache
                if let Some(cached_mats) = gpu_cache.get(expert_idx) {
                    // Fast path: use GPU-cached expert
                    let gpu_device = gpu_cache.gpu_device.clone();
                    let hidden_on_gpu = if !hidden_states.device().same_device(&gpu_device) {
                        hidden_states.to_device(&gpu_device)?
                    } else {
                        hidden_states.clone()
                    };
                    let result =
                        self.forward_with_cached(&hidden_on_gpu, &cached_mats, &gpu_device)?;
                    return if !result.device().same_device(input_device) {
                        result.to_device(input_device)
                    } else {
                        Ok(result)
                    };
                }

                // If hot but not cached, promote to GPU
                if is_hot {
                    if let Ok(entry) = gpu_cache.build_gpu_entry(
                        expert_idx,
                        &self.gate_exps,
                        &self.up_exps,
                        &self.down_exps,
                    ) {
                        let gpu_device = gpu_cache.gpu_device.clone();
                        gpu_cache.insert(expert_idx, entry.clone());

                        // Use the newly cached entry
                        let hidden_on_gpu = if !hidden_states.device().same_device(&gpu_device) {
                            hidden_states.to_device(&gpu_device)?
                        } else {
                            hidden_states.clone()
                        };
                        let result =
                            self.forward_with_cached(&hidden_on_gpu, &entry, &gpu_device)?;
                        return if !result.device().same_device(input_device) {
                            result.to_device(input_device)
                        } else {
                            Ok(result)
                        };
                    }
                }
            }
        }

        // Fall back to original path
        let mats = if training_mode {
            // Training mode: use training_cache if available
            if let Some(cache) = &self.training_cache {
                if cache.enabled() {
                    if let Some(entry) = cache.entries.get(&expert_idx).cloned() {
                        if let Some(cache) = &mut self.training_cache {
                            cache.touch(expert_idx);
                        }
                        entry
                    } else {
                        let entry = self.build_scaled_mats(expert_idx)?;
                        if let Some(cache) = &mut self.training_cache {
                            cache.evict_if_needed();
                            cache.entries.insert(expert_idx, entry.clone());
                            cache.touch(expert_idx);
                        }
                        entry
                    }
                } else {
                    self.build_scaled_mats(expert_idx)?
                }
            } else {
                self.build_scaled_mats(expert_idx)?
            }
        } else if let Some(cache) = &mut self.cache {
            if cache.enabled() {
                cache.get_or_prepare(expert_idx, &self.gate_exps, &self.up_exps, &self.down_exps)?
            } else {
                self.build_uncached_mats(expert_idx)?
            }
        } else {
            self.build_uncached_mats(expert_idx)?
        };

        self.forward_with_cached(hidden_states, &mats, input_device)
    }

    pub(super) fn build_uncached_mats(&self, expert_idx: usize) -> Result<CachedMatmuls> {
        let gate_qtensor = self.gate_exps.read().unwrap().slice_first_dim(expert_idx)?;
        let up_qtensor = self.up_exps.read().unwrap().slice_first_dim(expert_idx)?;
        let down_qtensor = self.down_exps.read().unwrap().slice_first_dim(expert_idx)?;

        let (gate_tensor, gate_device) = self.materialize_for_compute(gate_qtensor)?;
        let (up_tensor, up_device) = self.materialize_for_compute(up_qtensor)?;
        let (down_tensor, down_device) = self.materialize_for_compute(down_qtensor)?;

        Ok(CachedMatmuls {
            gate: Arc::new(paramecia_core::quantized::QMatMul::from_arc(gate_tensor)?.try_into()?),
            up: Arc::new(paramecia_core::quantized::QMatMul::from_arc(up_tensor)?.try_into()?),
            down: Arc::new(paramecia_core::quantized::QMatMul::from_arc(down_tensor)?.try_into()?),
            gate_device,
            up_device,
            down_device,
        })
    }

    pub(super) fn materialize_for_compute(
        &self,
        qtensor: QTensor,
    ) -> Result<(Arc<QTensor>, Device)> {
        if self.compute_device.same_device(&qtensor.device())
            || matches!(self.compute_device, Device::Cpu)
        {
            let device = qtensor.device();
            return Ok((Arc::new(qtensor), device));
        }

        let data = qtensor.data()?;
        let storage = QStorage::from_data(data, &self.compute_device, qtensor.dtype())?;
        let moved = QTensor::new(storage, qtensor.shape().clone())?;
        Ok((Arc::new(moved), self.compute_device.clone()))
    }

    pub(super) fn build_scaled_mats(&self, expert_idx: usize) -> Result<CachedMatmuls> {
        let gate_lock = self.gate_exps.read().unwrap();
        let up_lock = self.up_exps.read().unwrap();
        let down_lock = self.down_exps.read().unwrap();

        let gate_qtensor = gate_lock.slice_first_dim(expert_idx)?;
        let up_qtensor = up_lock.slice_first_dim(expert_idx)?;
        let down_qtensor = down_lock.slice_first_dim(expert_idx)?;

        let gate_src_device = gate_lock.device();
        let up_src_device = up_lock.device();
        let down_src_device = down_lock.device();

        let gate_mults = self
            .custom_gate_block_mults
            .as_ref()
            .map(|t| self.extract_expert_scales(t, expert_idx, &gate_qtensor))
            .transpose()?
            .ok_or_else(|| paramecia_core::Error::Msg("missing gate multipliers".to_string()))?;
        let up_mults = self
            .custom_up_block_mults
            .as_ref()
            .map(|t| self.extract_expert_scales(t, expert_idx, &up_qtensor))
            .transpose()?
            .ok_or_else(|| paramecia_core::Error::Msg("missing up multipliers".to_string()))?;
        let down_mults = self
            .custom_down_block_mults
            .as_ref()
            .map(|t| self.extract_expert_scales(t, expert_idx, &down_qtensor))
            .transpose()?
            .ok_or_else(|| paramecia_core::Error::Msg("missing down multipliers".to_string()))?;

        // Move multipliers to the source device for scale modification
        let gate_mults = if !gate_mults.device().same_device(&gate_src_device) {
            gate_mults.to_device(&gate_src_device)?
        } else {
            gate_mults
        };
        let up_mults = if !up_mults.device().same_device(&up_src_device) {
            up_mults.to_device(&up_src_device)?
        } else {
            up_mults
        };
        let down_mults = if !down_mults.device().same_device(&down_src_device) {
            down_mults.to_device(&down_src_device)?
        } else {
            down_mults
        };

        let gate_mults = gate_mults.to_dtype(DType::F32)?;
        let up_mults = up_mults.to_dtype(DType::F32)?;
        let down_mults = down_mults.to_dtype(DType::F32)?;

        // Apply scale modifications on source device
        let gate_modified = gate_qtensor.modify_block_scales(&gate_mults)?;
        let up_modified = up_qtensor.modify_block_scales(&up_mults)?;
        let down_modified = down_qtensor.modify_block_scales(&down_mults)?;

        // Materialize modified tensors to compute device
        let (gate_tensor, gate_device) = self.materialize_for_compute(gate_modified)?;
        let (up_tensor, up_device) = self.materialize_for_compute(up_modified)?;
        let (down_tensor, down_device) = self.materialize_for_compute(down_modified)?;

        Ok(CachedMatmuls {
            gate: Arc::new(paramecia_core::quantized::QMatMul::from_arc(gate_tensor)?.try_into()?),
            up: Arc::new(paramecia_core::quantized::QMatMul::from_arc(up_tensor)?.try_into()?),
            down: Arc::new(paramecia_core::quantized::QMatMul::from_arc(down_tensor)?.try_into()?),
            gate_device,
            up_device,
            down_device,
        })
    }

    pub(super) fn forward_with_cached(
        &self,
        hidden_states: &Tensor,
        mats: &CachedMatmuls,
        input_device: &Device,
    ) -> Result<Tensor> {
        self.forward_expert_with_qmatmul(
            hidden_states,
            mats.gate.as_ref(),
            mats.up.as_ref(),
            mats.down.as_ref(),
            input_device,
            &mats.gate_device,
            &mats.up_device,
            &mats.down_device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_expert_with_qmatmul(
        &self,
        hidden_states: &Tensor,
        gate_qmatmul: &TQMatMul<Shape2<SI, S>>,
        up_qmatmul: &TQMatMul<Shape2<SI, S>>,
        down_qmatmul: &TQMatMul<Shape2<S, SI>>,
        input_device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
    ) -> Result<Tensor> {
        let xs_for_gate = if !input_device.same_device(gate_device) {
            hidden_states.to_device(gate_device)?
        } else {
            hidden_states.clone()
        };
        let gate = gate_qmatmul
            .forward_untyped(&xs_for_gate.contiguous()?)?
            .apply(&self.act_fn)?;

        let xs_for_up = if !input_device.same_device(up_device) {
            hidden_states.to_device(up_device)?
        } else {
            hidden_states.clone()
        };
        let up = up_qmatmul.forward_untyped(&xs_for_up.contiguous()?)?;

        let gate_on_up_device = if !gate.device().same_device(up_device) {
            gate.to_device(up_device)?
        } else {
            gate
        };
        let gated = (gate_on_up_device * up)?.contiguous()?;

        let gated_for_down = if !gated.device().same_device(down_device) {
            gated.to_device(down_device)?
        } else {
            gated
        };
        let output = down_qmatmul.forward_untyped(&gated_for_down)?;

        if !output.device().same_device(input_device) {
            output.to_device(input_device)
        } else {
            Ok(output)
        }
    }

    pub(super) fn extract_expert_scales(
        &self,
        all_multipliers: &TBlockMults,
        expert_idx: usize,
        expert_qtensor: &QTensor,
    ) -> Result<Tensor> {
        let num_blocks = expert_qtensor.shape().elem_count() / expert_qtensor.dtype().block_size();
        let blocks_per_expert = num_blocks;
        let start_block = expert_idx * blocks_per_expert;
        all_multipliers
            .inner()
            .narrow(0, start_block, blocks_per_expert)?
            .contiguous()
    }

    #[allow(dead_code)]
    pub(super) fn set_block_multipliers(
        &mut self,
        gate_mults: Tensor,
        up_mults: Tensor,
        down_mults: Tensor,
    ) -> Result<()> {
        self.custom_gate_block_mults = Some(gate_mults.try_into()?);
        self.custom_up_block_mults = Some(up_mults.try_into()?);
        self.custom_down_block_mults = Some(down_mults.try_into()?);
        // Clear training cache since mults have changed
        if let Some(cache) = &mut self.training_cache {
            cache.entries.clear();
            cache.lru.clear();
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn clear_block_multipliers(&mut self) {
        self.custom_gate_block_mults = None;
        self.custom_up_block_mults = None;
        self.custom_down_block_mults = None;
        // Clear training cache when exiting training mode
        if let Some(cache) = &mut self.training_cache {
            cache.entries.clear();
            cache.lru.clear();
        }
    }
}

/// Shared expert FFN (dense, not MoE)
pub(super) struct SharedExpert {
    flow: SharedExpertForward,
}

impl std::fmt::Debug for SharedExpert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedExpert")
            .field("", &self.flow)
            .finish()
    }
}

impl SharedExpert {
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        dtype: DType,
    ) -> Result<Option<Self>> {
        #[allow(clippy::too_many_arguments)]
        let up_proj = match gg
            .try_typed_qmatmul::<Shape2<SI, S>>(&format!("{}.ffn_up_shexp.weight", prefix))?
        {
            Some(p) => p,
            None => return Ok(None),
        };
        let gate_proj =
            gg.typed_qmatmul::<Shape2<SI, S>>(&format!("{}.ffn_gate_shexp.weight", prefix))?;
        let down_proj =
            gg.typed_qmatmul::<Shape2<S, SI>>(&format!("{}.ffn_down_shexp.weight", prefix))?;
        log_typed_qmatmul_shape(&format!("{}.shexp.gate_proj", prefix), &gate_proj);
        log_typed_qmatmul_shape(&format!("{}.shexp.up_proj", prefix), &up_proj);
        log_typed_qmatmul_shape(&format!("{}.shexp.down_proj", prefix), &down_proj);
        // ffn_gate_inp_shexp is a 1D tensor [n_embd] used for dot product
        // Normalize to [1, n_embd] for typed broadcast_mul with [B, N, S]
        let shared_gate_raw = gg
            .tensor(&format!("{}.ffn_gate_inp_shexp.weight", prefix))
            .or_else(|_| gg.tensor(&format!("{}.ffn_gate_inp_shexp", prefix)))?
            .dequantize(&gg.device)?
            .to_dtype(dtype)?;
        let shared_gate: TSharedGate = match shared_gate_raw.dims() {
            [d] if *d == <S as Unsigned>::USIZE => shared_gate_raw
                .unsqueeze(0)?
                .try_into()
                .map_err(paramecia_core::Error::from),
            [1, d] if *d == <S as Unsigned>::USIZE => shared_gate_raw
                .try_into()
                .map_err(paramecia_core::Error::from),
            dims => paramecia_core::bail!(
                "shared expert gate expected [S] or [1,S] with S={}, got {:?}",
                <S as Unsigned>::USIZE,
                dims
            ),
        }?;

        Ok(Some(Self {
            flow: SharedExpertForward::new(
                shared_gate,
                SharedFfn::new(gate_proj.clone(), up_proj.clone(), down_proj.clone()),
            ),
        }))
    }

    pub(super) fn forward(&mut self, hidden_states: &THidden) -> Result<THidden> {
        let hs_typed: THidden = paramecia_tensor::contiguous!(hidden_states.clone())?;
        self.flow.traced_forward(&mut (), hs_typed)
    }
}

pub(super) fn default_cache_capacity(num_experts_per_tok: usize) -> usize {
    let env_override = std::env::var("QWEN3NEXT_MOE_CACHE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if let Some(value) = env_override {
        return value;
    }

    let scaled = num_experts_per_tok.saturating_mul(3);
    let buffered = num_experts_per_tok.saturating_add(4);
    let base = std::cmp::max(scaled, buffered);
    std::cmp::max(base, 8)
}

pub(super) struct MoeBlock {
    pub(super) experts: MoeExperts,
    pub(super) shared_expert: Option<SharedExpert>,
    pub(super) gate: TQMatMul<Shape2<E, S>>,
    pub(super) num_experts: usize,
    pub(super) num_experts_per_tok: usize,
    pub(super) span: tracing::Span,
    /// Layer index for this MoE block
    pub(super) layer_idx: usize,
    /// Optional expert mask for pruning: 0.0 for kept experts, -inf for pruned
    /// Shape: [num_experts_original]
    pub(super) expert_mask: Option<TExpertVec>,
    /// Optional expert remap for pruned models: maps original expert index to new index
    /// Shape: [num_experts_original], dtype: u16 stored as f16 bits
    pub(super) expert_remap: Option<TExpertVec>,
    pub(super) route_remap: MoeRouteRemap,
    pub(super) dispatch_prep: MoeDispatchPrep,
    pub(super) group_assignments: MoeGroupAssignments,
    pub(super) select_prefetch: MoePrefetchSelect,
    pub(super) select_sequential_exec: MoeSequentialExecSelect,
}

impl std::fmt::Debug for MoeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeBlock")
            .field("num_experts", &self.num_experts)
            .field("num_experts_per_tok", &self.num_experts_per_tok)
            .field("layer_idx", &self.layer_idx)
            .field("has_expert_mask", &self.expert_mask.is_some())
            .field("has_expert_remap", &self.expert_remap.is_some())
            .finish()
    }
}

impl MoeBlock {
    fn synthetic_single_expert_gate<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        dtype: DType,
    ) -> Result<TQMatMul<Shape2<E, S>>> {
        if <E as Unsigned>::USIZE != 1 {
            paramecia_core::bail!(
                "missing {}.ffn_gate_inp.weight for multi-expert model (E={})",
                prefix,
                <E as Unsigned>::USIZE
            );
        }
        let gate_dense = Tensor::zeros((1, <S as Unsigned>::USIZE), dtype, &gg.device)?;
        let gate_q = QTensor::quantize(&gate_dense, GgmlDType::F32)?;
        let gate_shared = SharedQTensor::new(gate_q);
        gg.shared_tensors.push((
            format!("{}.ffn_gate_inp.weight.synthetic", prefix),
            gate_shared.clone(),
        ));
        TQMatMul::<Shape2<E, S>>::from_shared(gate_shared).map_err(|e| {
            paramecia_core::Error::Msg(format!("failed to build synthetic routing gate: {e}"))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_experts: usize,
        num_experts_per_tok: usize,
        dtype: DType,
        compute_device: &Device,
        cache_capacity: usize,
        layer_idx: usize,
    ) -> Result<Self> {
        if num_experts != <E as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for num_experts: runtime={} type-level={}",
                num_experts,
                <E as Unsigned>::USIZE
            );
        }
        let cache_capacity = if cache_capacity == 0 {
            default_cache_capacity(num_experts_per_tok)
        } else {
            cache_capacity
        };

        let experts = MoeExperts::new(gg, prefix, compute_device, cache_capacity)?;
        let shared_expert = SharedExpert::new(gg, prefix, dtype)?;
        let gate = match gg
            .try_typed_qmatmul::<Shape2<E, S>>(&format!("{}.ffn_gate_inp.weight", prefix))?
        {
            Some(gate) => gate,
            None => Self::synthetic_single_expert_gate(gg, prefix, dtype)?,
        };
        log_typed_qmatmul_shape(&format!("{}.moe.gate", prefix), &gate);

        // Try to load expert mask and remap for pruned models
        let expert_mask = gg
            .try_tensor(&format!("{}.expert_mask", prefix))?
            .map(|qt| -> Result<TExpertVec> { Ok(qt.dequantize(compute_device)?.try_into()?) })
            .transpose()?;

        let expert_remap = gg
            .try_tensor(&format!("{}.expert_remap", prefix))?
            .map(|qt| -> Result<TExpertVec> { Ok(qt.dequantize(compute_device)?.try_into()?) })
            .transpose()?;

        let span = tracing::span!(tracing::Level::TRACE, "moe-block");
        Ok(Self {
            experts,
            shared_expert,
            route_remap: MoeRouteRemap::new(
                gate.clone(),
                num_experts_per_tok,
                expert_mask.clone(),
                expert_remap.clone(),
            ),
            dispatch_prep: MoeDispatchPrep::new(num_experts_per_tok),
            group_assignments: MoeGroupAssignments::new(num_experts),
            select_prefetch: MoePrefetchSelect::new(),
            select_sequential_exec: MoeSequentialExecSelect::new(),
            gate,
            num_experts,
            num_experts_per_tok,
            span,
            layer_idx,
            expert_mask,
            expert_remap,
        })
    }

    #[allow(dead_code)]
    pub(super) fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let hs_typed: THidden = hidden_states.contiguous()?.try_into()?;
        let (output, _) = self.forward_with_stats_typed(&hs_typed)?;
        Ok(output.into_inner())
    }

    pub(super) fn forward_typed(&mut self, hidden_states: &THidden) -> Result<THidden> {
        let (output, _) = self.forward_with_stats_typed(hidden_states)?;
        Ok(output)
    }

    pub(super) fn forward_with_stats_typed(
        &mut self,
        hidden_states: &THidden,
    ) -> Result<(THidden, (Tensor, Tensor))> {
        let _enter = self.span.enter();

        // Typed gate + masked top-k routing: [B, N, S] → [B, N, E], [B, N, Tk], [B, N, Tk]
        let hs_typed: THidden = paramecia_tensor::contiguous!(hidden_states.clone())?;
        let routed = self.route_remap.traced_forward(&mut (), hs_typed.clone())?;
        log_shape("moe.router_logits", &routed.router_logits);

        // Route tokens to experts
        drop(_enter);
        let moe_output: THidden = self.apply_experts(
            &routed.hidden_states,
            &routed.remapped_indices,
            &routed.top_weights,
        )?;
        log_shape("moe.expert_output", moe_output.inner());

        // Add shared expert output if present (arrow-composed)
        let final_output: THidden = if let Some(ref mut shared_exp) = self.shared_expert {
            let shared_out = shared_exp.forward(&hs_typed)?;
            log_shape("moe.shared_expert_output", shared_out.inner());
            ResidualAddHiddenFlow::new(ResidualAddOp::default())
                .traced_forward(&mut (), (moe_output, shared_out))?
        } else {
            moe_output
        };
        log_shape("moe.final_output", final_output.inner());

        Ok((final_output, (routed.router_logits, routed.top_indices)))
    }

    pub(super) fn route_for_prefetch(
        &mut self,
        hidden_states: &THidden,
    ) -> Result<(Tensor, TTopWeights, TTopIndices, Tensor)> {
        let routed = self
            .route_remap
            .traced_forward(&mut (), hidden_states.clone())?;
        Ok((
            routed.router_logits,
            routed.top_weights,
            routed.remapped_indices,
            routed.top_indices,
        ))
    }

    pub(super) fn prepare_dispatch_for_prefetch(
        &mut self,
        hidden_states: THidden,
        expert_indices: TTopIndices,
        expert_weights: TTopWeights,
    ) -> Result<MoeDispatchPrefetchOut> {
        let prepared = self
            .dispatch_prep
            .traced_forward(&mut (), (hidden_states, expert_indices, expert_weights))?;
        Ok((
            prepared.hidden_flat,
            prepared.indices_flat,
            prepared.weights_flat,
            (prepared.batch_size, prepared.seq_len, prepared.hidden_dim),
        ))
    }

    pub(super) fn apply_experts(
        &mut self,
        hidden_states: &THidden,
        expert_indices: &TTopIndices,
        expert_weights: &TTopWeights,
    ) -> Result<THidden> {
        let prepared = self.dispatch_prep.traced_forward(
            &mut (),
            (
                hidden_states.clone(),
                expert_indices.clone(),
                expert_weights.clone(),
            ),
        )?;
        let hidden_flat = &prepared.hidden_flat;
        let indices_flat = &prepared.indices_flat;
        let weights_flat = &prepared.weights_flat;

        // Check if we can use the batched path
        // The batched path uses indexed_moe_forward CUDA kernels which:
        // 1. Avoid GPU->CPU sync for expert indices (major bottleneck)
        // 2. Process all experts in parallel on GPU
        // 3. Handle input quantization to Q8_1 internally
        //
        // Even for single tokens, batched is faster than sequential due to
        // avoiding the synchronization overhead.
        let training_mode = self.experts.custom_gate_block_mults.is_some();
        let supports_batched = self.experts.supports_batched_forward();

        // Check if all experts are on CPU (can use optimized CPU paths)
        let all_experts_on_cpu = !is_gpu_device(&self.experts.gate_exps.read().unwrap().device())
            && !is_gpu_device(&self.experts.up_exps.read().unwrap().device())
            && !is_gpu_device(&self.experts.down_exps.read().unwrap().device());

        // Check if hidden states are on GPU (can use GPU-side grouping)
        let hidden_on_gpu = is_gpu_device(hidden_flat.inner().device());
        let mut resolve_path =
            moe_path_resolve_flow(training_mode, supports_batched, all_experts_on_cpu);
        let dispatch_path = resolve_path.traced_forward(&mut (), prepared.clone());
        let use_cpu_fused = matches!(dispatch_path, MoeDispatchPath::CpuFused);
        let use_batched = matches!(dispatch_path, MoeDispatchPath::Batched);
        let use_gpu_grouped = matches!(dispatch_path, MoeDispatchPath::GpuGrouped);

        if std::env::var("PARAMECIA_PROFILE").is_ok() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            let log_every = std::env::var("PARAMECIA_LOG_MOE_PATH_EVERY")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if log_every || !LOGGED.swap(true, Ordering::Relaxed) {
                debug!(
                    training = training_mode,
                    batched = use_batched,
                    cpu_fused = use_cpu_fused,
                    gpu_grouped = use_gpu_grouped,
                    experts_on_cpu = all_experts_on_cpu,
                    hidden_on_gpu = hidden_on_gpu,
                    device = ?hidden_flat.inner().device(),
                    gate_dev = ?self.experts.gate_exps.read().unwrap().device(),
                    "MoE path selection"
                );
            }
        }

        let result_flat: THiddenFlat = match dispatch_path {
            MoeDispatchPath::CpuFused => {
                // Vulkan: fused CPU path with minimal GPU transfers
                self.experts.forward_batched_cpu_fused(
                    hidden_flat,
                    indices_flat,
                    weights_flat,
                    self.num_experts_per_tok,
                )?
            }
            MoeDispatchPath::Batched => {
                // Use batched indexed_moe_forward CUDA kernels (fastest - all on GPU)
                self.experts.forward_batched(
                    hidden_flat,
                    indices_flat,
                    weights_flat,
                    self.num_experts_per_tok,
                )?
            }
            MoeDispatchPath::GpuGrouped => {
                // GPU-side grouping: sort tokens by expert on GPU, then process on CPU
                // This minimizes GPU->CPU sync by doing grouping on GPU
                crate::ops::process_moe_gpu_grouped(
                    hidden_flat.inner(),
                    indices_flat.inner(),
                    weights_flat.inner(),
                    &self.experts.gate_exps,
                    &self.experts.up_exps,
                    &self.experts.down_exps,
                    self.num_experts,
                )?
                .try_into()?
            }
            MoeDispatchPath::Sequential => {
                // Sequential processing with Rayon parallelization
                self.apply_experts_sequential(hidden_flat, indices_flat, weights_flat)?
            }
        };

        result_flat
            .into_inner()
            .reshape((prepared.batch_size, prepared.seq_len, prepared.hidden_dim))?
            .try_into()
            .map_err(paramecia_core::Error::from)
    }

    /// Sequential expert application (fallback for training mode or unsupported configs)
    pub(super) fn apply_experts_sequential(
        &mut self,
        hidden_flat: &THiddenFlat,
        indices_flat: &TTopFlat,
        weights_flat: &TTopFlat,
    ) -> Result<THiddenFlat> {
        let mut dims2 = Dims2Op::<Shape2<T, S>>::default();
        let (_total_tokens, hidden_dim) = <Dims2Op<Shape2<T, S>> as Combinator<()>>::forward(
            &mut dims2,
            &mut (),
            hidden_flat.clone(),
        )?;
        let assignments = self
            .group_assignments
            .traced_forward(&mut (), (indices_flat.clone(), weights_flat.clone()))?;
        let top_x = &assignments.top_x;
        let selected_rws = &assignments.selected_rws;
        let active_experts = self.collect_active_experts(top_x);

        let training_mode = self.experts.custom_gate_block_mults.is_some();
        let prefetch_mode = self.select_prefetch.traced_forward(
            &mut (),
            MoePrefetchContext {
                training_mode,
                inference_cache_enabled: self
                    .experts
                    .cache
                    .as_ref()
                    .map(|c| c.enabled())
                    .unwrap_or(false),
                training_cache_enabled: self
                    .experts
                    .training_cache
                    .as_ref()
                    .map(|c| c.enabled())
                    .unwrap_or(false),
            },
        );
        self.prefetch_active_experts(prefetch_mode, &active_experts)?;

        let mut ys = hidden_flat.inner().zeros_like()?;

        // First, check if GPU hot cache is available and collect cached experts
        let (gpu_device_opt, cached_entries) = self.collect_gpu_cached_entries(&active_experts);

        // Process GPU-cached experts first (fast path)
        let gpu_processed = self.apply_gpu_cached_entries(
            &mut ys,
            hidden_flat.inner(),
            top_x,
            selected_rws,
            gpu_device_opt.as_ref(),
            &cached_entries,
        )?;

        // Get remaining experts not processed by GPU cache
        let remaining_experts: Vec<usize> = active_experts
            .iter()
            .filter(|&&idx| !gpu_processed.contains(&idx))
            .copied()
            .collect();

        // Check if we should use parallel processing for remaining
        // Parallel is beneficial when we have multiple experts and ALL weights are on CPU
        // (the parallel path processes entire expert on CPU, so mixed GPU/CPU doesn't work)
        let all_experts_on_cpu = !is_gpu_device(&self.experts.gate_exps.read().unwrap().device())
            && !is_gpu_device(&self.experts.up_exps.read().unwrap().device())
            && !is_gpu_device(&self.experts.down_exps.read().unwrap().device());
        let exec_mode = self.select_sequential_exec.traced_forward(
            &mut (),
            MoeSequentialExecContext {
                remaining_experts_len: remaining_experts.len(),
                all_experts_on_cpu,
                training_mode,
            },
        );

        self.apply_remaining_experts(
            &mut ys,
            hidden_flat.inner(),
            hidden_dim,
            top_x,
            selected_rws,
            &remaining_experts,
            exec_mode,
        )?;

        let ys_typed = ys.try_into()?;
        Ok(ys_typed)
    }

    fn collect_active_experts(&self, top_x: &[Vec<u32>]) -> Vec<usize> {
        (0..self.num_experts)
            .filter(|&idx| !top_x[idx].is_empty())
            .collect()
    }

    fn prefetch_active_experts(
        &mut self,
        mode: MoePrefetchMode,
        active_experts: &[usize],
    ) -> Result<()> {
        match mode {
            MoePrefetchMode::Inference => self.prefetch_inference_experts(active_experts),
            MoePrefetchMode::Training => self.prefetch_training_experts(active_experts),
            MoePrefetchMode::None => Ok(()),
        }
    }

    fn prefetch_inference_experts(&mut self, active_experts: &[usize]) -> Result<()> {
        let mut ctx = PrefetchInferenceExpertCtx {
            cache: self.experts.cache.as_mut(),
            gate_exps: Some(self.experts.gate_exps.as_shared()),
            up_exps: Some(self.experts.up_exps.as_shared()),
            down_exps: Some(self.experts.down_exps.as_shared()),
        };
        if ctx.cache.is_none() {
            return Ok(());
        }
        let step = prefetch_inference_expert_op();
        let mut fold =
            TryFoldVec::<PrefetchInferenceExpertOp, (), usize, paramecia_core::Error>::new(step);
        fold.traced_forward(&mut ctx, ((), active_experts.to_vec()))
    }

    fn prefetch_training_experts(&mut self, active_experts: &[usize]) -> Result<()> {
        let mut ctx = PrefetchTrainingExpertCtx {
            experts: Some(&mut self.experts),
        };
        let step = prefetch_training_expert_op();
        let mut fold =
            TryFoldVec::<PrefetchTrainingExpertOp, (), usize, paramecia_core::Error>::new(step);
        fold.traced_forward(&mut ctx, ((), active_experts.to_vec()))
    }

    fn collect_gpu_cached_entries(
        &mut self,
        active_experts: &[usize],
    ) -> (Option<Device>, Vec<(usize, CachedMatmuls)>) {
        if let Some(ref mut gpu_cache) = self.experts.gpu_hot_cache {
            let gpu_device = gpu_cache.gpu_device.clone();
            let mut cached = Vec::new();

            for &expert_idx in active_experts {
                gpu_cache.record_usage(expert_idx);
                if let Some(mats) = gpu_cache.get(expert_idx) {
                    cached.push((expert_idx, mats));
                }
            }
            (Some(gpu_device), cached)
        } else {
            (None, Vec::new())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_gpu_cached_entries(
        &mut self,
        ys: &mut Tensor,
        hidden_flat: &Tensor,
        top_x: &[Vec<u32>],
        selected_rws: &[Vec<f32>],
        gpu_device_opt: Option<&Device>,
        cached_entries: &[(usize, CachedMatmuls)],
    ) -> Result<std::collections::HashSet<usize>> {
        let mut gpu_processed = HashSet::new();
        let Some(gpu_device) = gpu_device_opt else {
            return Ok(gpu_processed);
        };
        let mut ctx = ApplyGpuCachedExpertCtx {
            experts: &mut self.experts,
            hidden_flat,
            top_x,
            selected_rws,
            gpu_device,
        };
        let step = apply_gpu_cached_expert_op();
        let init_state = GpuCachedApplyState {
            ys: ys.clone(),
            gpu_processed: HashSet::new(),
        };
        let mut fold = TryFoldVec::<
            ApplyGpuCachedExpert,
            GpuCachedApplyState,
            (usize, CachedMatmuls),
            paramecia_core::Error,
        >::new(step);
        let state = fold.traced_forward(&mut ctx, (init_state, cached_entries.to_vec()))?;
        *ys = state.ys;
        gpu_processed.extend(state.gpu_processed);
        Ok(gpu_processed)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_remaining_experts(
        &mut self,
        ys: &mut Tensor,
        hidden_flat: &Tensor,
        hidden_dim: usize,
        top_x: &[Vec<u32>],
        selected_rws: &[Vec<f32>],
        remaining_experts: &[usize],
        exec_mode: MoeSequentialExecMode,
    ) -> Result<()> {
        if matches!(exec_mode, MoeSequentialExecMode::Parallel) {
            // Parallel expert processing using Tokio's blocking pool
            let expert_assignments: Vec<(usize, Vec<u32>, Vec<f32>)> = remaining_experts
                .iter()
                .map(|&idx| (idx, top_x[idx].clone(), selected_rws[idx].clone()))
                .collect();

            let result = crate::expert_pipeline::process_experts_tokio(
                hidden_flat,
                &self.experts.gate_exps,
                &self.experts.up_exps,
                &self.experts.down_exps,
                &expert_assignments,
                hidden_dim,
            )?;

            *ys = ys.add(&result)?;
        } else {
            let mut ctx = ApplySequentialExpertCtx {
                experts: &mut self.experts,
                hidden_flat,
                top_x,
                selected_rws,
            };
            let step = apply_sequential_expert_op();
            let mut fold =
                TryFoldVec::<ApplySequentialExpert, Tensor, usize, paramecia_core::Error>::new(
                    step,
                );
            *ys = fold.traced_forward(&mut ctx, (ys.clone(), remaining_experts.to_vec()))?;
        }

        Ok(())
    }
}
