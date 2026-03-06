use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use glowstick::num::{Unsigned, U0, U1, U2, U3};
use glowstick::Shape4;
use paramecia_core::{DType, Device, Result, Tensor, D};
use paramecia_nn::{Embedding, Module};
use paramecia_tensor::glowstick::{Shape1, Shape2, Shape3};
use paramecia_tensor::Error as TensorError;
use paramecia_tensor::Tensor as TTensor;
use std::io::{Read, Seek};
use std::sync::Arc;

use inception::{primitive, Inception};
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, ChunkVecOp, Combinator, CombinatorTraceExt, EnumerateVecOp, Fanout, Fanout3,
    GroupByIndexOp, LiftResult, MapErr, MapOk, OptionThen, Swap2, Then, TryFoldRange,
    TryFoldSliceMut, TryFromOp, TryIndexMutCtx, WrapOk, Zip, Zip3, ZipOk, ZipOk3, ZipVecOp,
};

use paramecia_tensor::dims3::Dims3Op;
use paramecia_tensor::flatten::FlattenOp;
use paramecia_tensor::flatten_prefix2::FlattenPrefix2Op;
use paramecia_tensor::from_vec_on_device::{FromVec1OnDeviceOp, FromVecColOnDeviceOp};
use paramecia_tensor::index_add_dim0::IndexAddDim0Op;
use paramecia_tensor::index_select_dim0::IndexSelectDim0Op;
use paramecia_tensor::narrow::NarrowOp;
use paramecia_tensor::qmatmul_from_qtensor::QMatMulFromQTensorOp;
use paramecia_tensor::qmatmul_op::QMatMulOp;
use paramecia_tensor::to_vec::CastFlattenToVec1PairOp;
use paramecia_tensor::topk_from_logits::TopkFromLogitsOp as TensorTopkFromLogitsOp;
use paramecia_tensor::unflatten_last::UnflattenLastOp;
use paramecia_tensor::{
    broadcast_mul::BroadcastMulOp, cast_like::CastLikeOp, sigmoid::SigmoidOp, silu::SiluOp,
    sum_dim::SumDimOp,
};
use paramecia_tensor::{
    contiguous::ContiguousOp, residual_add::ResidualAddOp, rms_norm::RmsNormOp,
    transpose::TransposeOp,
};

use super::gguf_loader::Gguf;
use super::moe::{DispatchShapeOp, ExpertWeightTensor, ResidualAddHiddenFlow};
use super::rope::RotaryEmbedding;
use super::shape::{Lk, Sx2, Tk, A, AH, AH2, B, E, H, H2, K, KH, N, S, S2, SI, T, V};
use super::utils::{log_shape, log_typed_qmatmul_shape};

type TQMatMul<Sh> = paramecia_tensor::QMatMul<Sh>;
type TLmHead = TQMatMul<Shape2<V, S>>;
type THidden = TTensor<Shape3<B, N, S>>;
type THiddenCat = TTensor<Shape3<B, N, Sx2>>;
type TAttnHidden = TTensor<Shape3<B, N, AH>>;
type TQProj = TTensor<Shape3<B, N, AH2>>;
type TKProj = TTensor<Shape3<B, N, KH>>;
type TVProj = TTensor<Shape3<B, N, KH>>;
type TQSplitAll = TTensor<Shape4<B, N, A, H2>>;
type TQSplit = TTensor<Shape4<B, N, A, H>>;
type TKV4 = TTensor<Shape4<B, N, K, H>>;
type TKVCache = TTensor<Shape4<B, K, Lk, H>>;
type TQHeads = TTensor<Shape4<B, A, N, H>>;
type TKHeads = TTensor<Shape4<B, K, N, H>>;
type TVHeads = TTensor<Shape4<B, K, N, H>>;
type TIntermediate = TTensor<Shape3<B, N, SI>>;
type TRouterLogits = TTensor<Shape3<B, N, E>>;
type TTopWeights = TTensor<Shape3<B, N, Tk>>;
type TTopIndices = TTensor<Shape3<B, N, Tk>>;
type TSharedGate = TTensor<Shape2<U1, S>>;
type TSharedGateQt = paramecia_tensor::SharedQTensor<Shape2<U1, S>>;
type TGateScore = TTensor<Shape3<B, N, U1>>;
type THiddenFlat = TTensor<Shape2<T, S>>;
type TTopFlat = TTensor<Shape2<T, Tk>>;
type TIntermediateFlat = TTensor<Shape2<T, SI>>;
type TWeightCol = TTensor<Shape2<T, U1>>;

type KvReshapeOpFlow = MapErr<
    UnflattenLastOp<Shape3<B, N, KH>, Shape4<B, N, K, H>>,
    TKProj,
    TKV4,
    TensorError,
    paramecia_core::Error,
>;

type QSplitPrepFlow = Then<
    Zip<QNormTransposeOp, WrapOk<TQSplit, paramecia_core::Error>>,
    ZipOk<TQHeads, TQSplit, paramecia_core::Error>,
>;
fn q_split_prep_flow(q_norm: RmsNorm) -> QSplitPrepFlow {
    Then::new(
        Zip::new(q_norm_transpose_op(q_norm), WrapOk::default()),
        ZipOk::default(),
    )
}

type QPrepFlow = Then<
    SplitGatedQFlow,
    LiftResult<
        QSplitPrepFlow,
        std::result::Result<(TQSplit, TQSplit), paramecia_core::Error>,
        (TQHeads, TQSplit),
    >,
>;
fn q_prep_flow(q_norm: RmsNorm) -> QPrepFlow {
    Then::new(
        split_gated_q_flow(),
        LiftResult::new(q_split_prep_flow(q_norm)),
    )
}

type QProjFlow = MapErr<
    QMatMulOp<Shape2<AH2, S>, Shape3<B, N, S>, Shape3<B, N, AH2>>,
    THidden,
    TQProj,
    TensorError,
    paramecia_core::Error,
>;
type KProjFlow = MapErr<
    QMatMulOp<Shape2<KH, S>, Shape3<B, N, S>, Shape3<B, N, KH>>,
    THidden,
    TKProj,
    TensorError,
    paramecia_core::Error,
>;
type VProjFlow = MapErr<
    QMatMulOp<Shape2<KH, S>, Shape3<B, N, S>, Shape3<B, N, KH>>,
    THidden,
    TVProj,
    TensorError,
    paramecia_core::Error,
>;
type KPrepFromProjFlow = Then<
    KvReshapeOpFlow,
    LiftResult<KNormTransposeOp, std::result::Result<TKV4, paramecia_core::Error>, TKHeads>,
>;
type VPrepFromProjFlow = Then<
    KvReshapeOpFlow,
    LiftResult<VTransposeOp, std::result::Result<TKV4, paramecia_core::Error>, TVHeads>,
>;
type QkvQBranchFlow = Then<
    QProjFlow,
    LiftResult<QPrepFlow, std::result::Result<TQProj, paramecia_core::Error>, (TQHeads, TQSplit)>,
>;
type QkvKBranchFlow = Then<
    KProjFlow,
    LiftResult<KPrepFromProjFlow, std::result::Result<TKProj, paramecia_core::Error>, TKHeads>,
>;
type QkvVBranchFlow = Then<
    VProjFlow,
    LiftResult<VPrepFromProjFlow, std::result::Result<TVProj, paramecia_core::Error>, TVHeads>,
>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct QkvPrep(
    Fanout3<THidden>,
    Zip3<QkvQBranchFlow, QkvKBranchFlow, QkvVBranchFlow>,
    ZipOk3<(TQHeads, TQSplit), TKHeads, TVHeads, paramecia_core::Error>,
);
impl QkvPrep {
    fn new(
        wq: TQMatMul<Shape2<AH2, S>>,
        wk: TQMatMul<Shape2<KH, S>>,
        wv: TQMatMul<Shape2<KH, S>>,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
    ) -> Self {
        Self(
            Fanout3::default(),
            Zip3::new(
                Then::new(
                    MapErr::new(QMatMulOp::new(wq)),
                    LiftResult::new(q_prep_flow(q_norm)),
                ),
                Then::new(
                    MapErr::new(QMatMulOp::new(wk)),
                    LiftResult::new(Then::new(
                        MapErr::new(UnflattenLastOp::new(K::USIZE, H::USIZE)),
                        LiftResult::new(k_norm_transpose_op(k_norm)),
                    )),
                ),
                Then::new(
                    MapErr::new(QMatMulOp::new(wv)),
                    LiftResult::new(Then::new(
                        MapErr::new(UnflattenLastOp::new(K::USIZE, H::USIZE)),
                        LiftResult::new(v_transpose_op()),
                    )),
                ),
            ),
            ZipOk3::default(),
        )
    }
}
impl std::fmt::Debug for QkvPrep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QkvPrep").finish()
    }
}

type GateFlattenFlow = MapErr<
    FlattenOp<Shape4<B, N, A, H>, U2, U3>,
    TQSplit,
    TAttnHidden,
    TensorError,
    paramecia_core::Error,
>;

type GateSigmoidFlow = Then<
    Zip<
        WrapOk<TAttnHidden, paramecia_core::Error>,
        MapErr<
            SigmoidOp<Shape3<B, N, AH>>,
            TAttnHidden,
            TAttnHidden,
            TensorError,
            paramecia_core::Error,
        >,
    >,
    ZipOk<TAttnHidden, TAttnHidden, paramecia_core::Error>,
>;
fn gate_sigmoid_flow() -> GateSigmoidFlow {
    Then::new(
        Zip::new(WrapOk::default(), MapErr::new(SigmoidOp::default())),
        ZipOk::default(),
    )
}

type GateCastLikeFlow = MapErr<
    CastLikeOp<Shape3<B, N, AH>>,
    (TAttnHidden, TAttnHidden),
    (TAttnHidden, TAttnHidden),
    TensorError,
    paramecia_core::Error,
>;
type GateMulFlow = MapErr<
    BroadcastMulOp<Shape3<B, N, AH>, Shape3<B, N, AH>>,
    (TAttnHidden, TAttnHidden),
    TAttnHidden,
    TensorError,
    paramecia_core::Error,
>;
type GateApplyFlow = Then<
    GateCastLikeFlow,
    Then<
        LiftResult<
            GateSigmoidFlow,
            std::result::Result<(TAttnHidden, TAttnHidden), paramecia_core::Error>,
            (TAttnHidden, TAttnHidden),
        >,
        LiftResult<
            GateMulFlow,
            std::result::Result<(TAttnHidden, TAttnHidden), paramecia_core::Error>,
            TAttnHidden,
        >,
    >,
>;
fn gate_apply_flow() -> GateApplyFlow {
    Then::new(
        MapErr::new(CastLikeOp::default()),
        Then::new(
            LiftResult::new(gate_sigmoid_flow()),
            LiftResult::new(MapErr::new(BroadcastMulOp::default())),
        ),
    )
}

type PostAttnGatePrepareFlow = Then<
    Zip<WrapOk<TAttnHidden, paramecia_core::Error>, GateFlattenFlow>,
    ZipOk<TAttnHidden, TAttnHidden, paramecia_core::Error>,
>;
type PostAttnGateFlow = Then<
    PostAttnGatePrepareFlow,
    LiftResult<
        GateApplyFlow,
        std::result::Result<(TAttnHidden, TAttnHidden), paramecia_core::Error>,
        TAttnHidden,
    >,
>;
fn post_attn_gate_flow() -> PostAttnGateFlow {
    Then::new(
        Then::new(
            Zip::new(WrapOk::default(), MapErr::new(FlattenOp::default())),
            ZipOk::default(),
        ),
        LiftResult::new(gate_apply_flow()),
    )
}

type OutputProjectFlow = QMatMulOp<Shape2<S, AH>, Shape3<B, N, AH>, Shape3<B, N, S>>;
fn output_project_flow(wo: TQMatMul<Shape2<S, AH>>) -> OutputProjectFlow {
    QMatMulOp::new(wo)
}

#[allow(dead_code)]
struct MtpAttentionForwardGraph;
#[primitive(property = Visualize)]
impl Vis for MtpAttentionForwardGraph {
    fn visualize() -> Graph {
        let g = Graph::sequence(
            <QkvPrep as Vis>::visualize(),
            <PostAttnGateFlow as Vis>::visualize(),
        );
        let g = Graph::sequence(g, <OutputProjectFlow as Vis>::visualize());
        Graph::wrap_custom_subgraph("MtpAttentionForwardGraph", g)
    }
}

type QReshapeSplitFlow = MapErr<
    UnflattenLastOp<Shape3<B, N, AH2>, Shape4<B, N, A, H2>>,
    TQProj,
    TQSplitAll,
    TensorError,
    paramecia_core::Error,
>;

type QTakeQFlow = MapErr<
    NarrowOp<Shape4<B, N, A, H2>, U3, U0, H>,
    TQSplitAll,
    TQSplit,
    TensorError,
    paramecia_core::Error,
>;
type QTakeGateFlow = MapErr<
    NarrowOp<Shape4<B, N, A, H2>, U3, H, H>,
    TQSplitAll,
    TQSplit,
    TensorError,
    paramecia_core::Error,
>;

type QSplitGateBranchesFlow = Then<
    Fanout<TQSplitAll>,
    Then<Zip<QTakeQFlow, QTakeGateFlow>, ZipOk<TQSplit, TQSplit, paramecia_core::Error>>,
>;
fn q_split_gate_branches_flow() -> QSplitGateBranchesFlow {
    Then::new(
        Fanout::default(),
        Then::new(
            Zip::new(
                MapErr::new(NarrowOp::default()),
                MapErr::new(NarrowOp::default()),
            ),
            ZipOk::default(),
        ),
    )
}

type SplitGatedQFlow = Then<
    QReshapeSplitFlow,
    LiftResult<
        QSplitGateBranchesFlow,
        std::result::Result<TQSplitAll, paramecia_core::Error>,
        (TQSplit, TQSplit),
    >,
>;
fn split_gated_q_flow() -> SplitGatedQFlow {
    Then::new(
        MapErr::new(UnflattenLastOp::new(A::USIZE, H2::USIZE)),
        LiftResult::new(q_split_gate_branches_flow()),
    )
}

type QRmsNormFlow =
    MapErr<RmsNormOp<Shape4<B, N, A, H>>, TQSplit, TQSplit, TensorError, paramecia_core::Error>;
type KRmsNormFlow =
    MapErr<RmsNormOp<Shape4<B, N, K, H>>, TKV4, TKV4, TensorError, paramecia_core::Error>;
type QTransposeFlow = MapErr<
    TransposeOp<Shape4<B, N, A, H>, U1, U2>,
    TQSplit,
    TQHeads,
    TensorError,
    paramecia_core::Error,
>;
type KTransposeFlow = MapErr<
    TransposeOp<Shape4<B, N, K, H>, U1, U2>,
    TKV4,
    TKHeads,
    TensorError,
    paramecia_core::Error,
>;
type VTransposeFlow = MapErr<
    TransposeOp<Shape4<B, N, K, H>, U1, U2>,
    TKV4,
    TVHeads,
    TensorError,
    paramecia_core::Error,
>;
type QContiguousFlow =
    MapErr<ContiguousOp<Shape4<B, A, N, H>>, TQHeads, TQHeads, TensorError, paramecia_core::Error>;
type KContiguousFlow =
    MapErr<ContiguousOp<Shape4<B, K, N, H>>, TKHeads, TKHeads, TensorError, paramecia_core::Error>;
type VContiguousFlow =
    MapErr<ContiguousOp<Shape4<B, K, N, H>>, TVHeads, TVHeads, TensorError, paramecia_core::Error>;
type QNormTransposeOp = Then<
    QRmsNormFlow,
    Then<
        LiftResult<QTransposeFlow, std::result::Result<TQSplit, paramecia_core::Error>, TQHeads>,
        LiftResult<QContiguousFlow, std::result::Result<TQHeads, paramecia_core::Error>, TQHeads>,
    >,
>;
fn q_norm_transpose_op(norm: RmsNorm) -> QNormTransposeOp {
    let rms = RmsNormOp::new_with_shared(
        norm.weight().clone(),
        norm.eps(),
        norm.shared_weight().cloned(),
        norm.zero_centered(),
    );
    Then::new(
        MapErr::new(rms),
        Then::new(
            LiftResult::new(MapErr::new(TransposeOp::default())),
            LiftResult::new(MapErr::new(ContiguousOp::default())),
        ),
    )
}

type KNormTransposeOp = Then<
    KRmsNormFlow,
    Then<
        LiftResult<KTransposeFlow, std::result::Result<TKV4, paramecia_core::Error>, TKHeads>,
        LiftResult<KContiguousFlow, std::result::Result<TKHeads, paramecia_core::Error>, TKHeads>,
    >,
>;
fn k_norm_transpose_op(norm: RmsNorm) -> KNormTransposeOp {
    let rms = RmsNormOp::new_with_shared(
        norm.weight().clone(),
        norm.eps(),
        norm.shared_weight().cloned(),
        norm.zero_centered(),
    );
    Then::new(
        MapErr::new(rms),
        Then::new(
            LiftResult::new(MapErr::new(TransposeOp::default())),
            LiftResult::new(MapErr::new(ContiguousOp::default())),
        ),
    )
}

type VTransposeOp = Then<
    VTransposeFlow,
    LiftResult<VContiguousFlow, std::result::Result<TVHeads, paramecia_core::Error>, TVHeads>,
>;
fn v_transpose_op() -> VTransposeOp {
    Then::new(
        MapErr::new(TransposeOp::default()),
        LiftResult::new(MapErr::new(ContiguousOp::default())),
    )
}

type MtpGateProjFlow = MapErr<
    QMatMulOp<Shape2<SI, S>, Shape3<B, N, S>, Shape3<B, N, SI>>,
    THidden,
    TIntermediate,
    TensorError,
    paramecia_core::Error,
>;
type MtpUpProjFlow = MapErr<
    QMatMulOp<Shape2<SI, S>, Shape3<B, N, S>, Shape3<B, N, SI>>,
    THidden,
    TIntermediate,
    TensorError,
    paramecia_core::Error,
>;
type MtpSiluFlow = MapErr<
    SiluOp<Shape3<B, N, SI>>,
    TIntermediate,
    TIntermediate,
    TensorError,
    paramecia_core::Error,
>;
type MtpMulFlow = MapErr<
    BroadcastMulOp<Shape3<B, N, SI>, Shape3<B, N, SI>>,
    (TIntermediate, TIntermediate),
    TIntermediate,
    TensorError,
    paramecia_core::Error,
>;
type MtpContigFlow = MapErr<
    ContiguousOp<Shape3<B, N, SI>>,
    TIntermediate,
    TIntermediate,
    TensorError,
    paramecia_core::Error,
>;
type MtpDownProjFlow = MapErr<
    QMatMulOp<Shape2<S, SI>, Shape3<B, N, SI>, Shape3<B, N, S>>,
    TIntermediate,
    THidden,
    TensorError,
    paramecia_core::Error,
>;

type MtpSiluMulFlow = Then<
    Zip<MtpSiluFlow, WrapOk<TIntermediate, paramecia_core::Error>>,
    Then<
        ZipOk<TIntermediate, TIntermediate, paramecia_core::Error>,
        LiftResult<
            MtpMulFlow,
            std::result::Result<(TIntermediate, TIntermediate), paramecia_core::Error>,
            TIntermediate,
        >,
    >,
>;
fn mtp_silu_mul_flow() -> MtpSiluMulFlow {
    Then::new(
        Zip::new(MapErr::new(SiluOp::default()), WrapOk::default()),
        Then::new(
            ZipOk::default(),
            LiftResult::new(MapErr::new(BroadcastMulOp::default())),
        ),
    )
}

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpSharedFfn(
    Fanout<THidden>,
    Zip<MtpGateProjFlow, MtpUpProjFlow>,
    ZipOk<TIntermediate, TIntermediate, paramecia_core::Error>,
    LiftResult<
        MtpSiluMulFlow,
        std::result::Result<(TIntermediate, TIntermediate), paramecia_core::Error>,
        TIntermediate,
    >,
    LiftResult<
        MtpContigFlow,
        std::result::Result<TIntermediate, paramecia_core::Error>,
        TIntermediate,
    >,
    LiftResult<MtpDownProjFlow, std::result::Result<TIntermediate, paramecia_core::Error>, THidden>,
);
impl MtpSharedFfn {
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
            LiftResult::new(mtp_silu_mul_flow()),
            LiftResult::new(MapErr::new(ContiguousOp::default())),
            LiftResult::new(MapErr::new(QMatMulOp::new(down_proj))),
        )
    }
}
impl std::fmt::Debug for MtpSharedFfn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpSharedFfn").finish()
    }
}

type MtpSharedGateAlignFlow = MapErr<
    CastLikeOp<Shape3<B, N, S>, Shape2<U1, S>>,
    (THidden, TSharedGate),
    (THidden, TSharedGate),
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MtpSharedGateMulFlow = MapErr<
    BroadcastMulOp<Shape3<B, N, S>, Shape2<U1, S>>,
    (THidden, TSharedGate),
    THidden,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MtpSharedGateScoreFlow = MapErr<
    SumDimOp<Shape3<B, N, S>, U2>,
    THidden,
    TGateScore,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MtpSharedGateSigmoidFlow = MapErr<
    SigmoidOp<Shape3<B, N, U1>>,
    TGateScore,
    TGateScore,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type MtpSharedGateComputeFlow = Then<
    MtpSharedGateAlignFlow,
    Then<
        LiftResult<
            MtpSharedGateMulFlow,
            std::result::Result<(THidden, TSharedGate), paramecia_core::Error>,
            THidden,
        >,
        Then<
            LiftResult<
                MtpSharedGateScoreFlow,
                std::result::Result<THidden, paramecia_core::Error>,
                TGateScore,
            >,
            LiftResult<
                MtpSharedGateSigmoidFlow,
                std::result::Result<TGateScore, paramecia_core::Error>,
                TGateScore,
            >,
        >,
    >,
>;
fn mtp_shared_gate_compute_flow() -> MtpSharedGateComputeFlow {
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

type MtpSharedOutputCastFlow = MapErr<
    CastLikeOp<Shape3<B, N, U1>, Shape3<B, N, S>>,
    (TGateScore, THidden),
    (TGateScore, THidden),
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MtpSharedOutputMulFlow = MapErr<
    BroadcastMulOp<Shape3<B, N, U1>, Shape3<B, N, S>>,
    (TGateScore, THidden),
    THidden,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type MtpSharedGateApplyFlow = Then<
    MtpSharedOutputCastFlow,
    LiftResult<
        MtpSharedOutputMulFlow,
        std::result::Result<(TGateScore, THidden), paramecia_core::Error>,
        THidden,
    >,
>;
fn mtp_shared_gate_apply_flow() -> MtpSharedGateApplyFlow {
    Then::new(
        MapErr::new(CastLikeOp::default()),
        LiftResult::new(MapErr::new(BroadcastMulOp::default())),
    )
}

#[allow(dead_code)]
struct MtpSharedExpertForwardGraph;
#[primitive(property = Visualize)]
impl Vis for MtpSharedExpertForwardGraph {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "MtpSharedExpertForwardGraph",
            <MtpSharedExpertForward as Vis>::visualize(),
        )
    }
}

type CE = paramecia_core::Error;

/// Prepares the shared gate tensor for MTP, handling conditional dequant/cast.
struct MtpGatePrepOp {
    shared_gate: TSharedGate,
    shared_gate_qt: Option<TSharedGateQt>,
}
#[primitive(property = Arrow)]
impl Combinator for MtpGatePrepOp {
    type In = THidden;
    type Out = std::result::Result<(THidden, TSharedGate), CE>;
    fn forward(&mut self, _ctx: &mut (), input: THidden) -> Self::Out {
        let gate = if let Some(ref qt) = self.shared_gate_qt {
            MtpSharedExpert::normalize_shared_gate(
                qt.dequant_to(input.inner().dtype(), input.inner().device())?
                    .into_inner(),
            )?
        } else {
            self.shared_gate
                .inner()
                .to_dtype(input.inner().dtype())?
                .to_device(input.inner().device())?
                .try_into()?
        };
        Ok((input, gate))
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpGatePrepOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpGatePrep")
            .with_output_type::<std::result::Result<(THidden, TSharedGate), CE>>()
    }
}

type MtpGatePathFlow = Then<
    MtpGatePrepOp,
    LiftResult<
        MtpSharedGateComputeFlow,
        std::result::Result<(THidden, TSharedGate), CE>,
        TGateScore,
    >,
>;
fn mtp_gate_path_flow(
    shared_gate: TSharedGate,
    shared_gate_qt: Option<TSharedGateQt>,
) -> MtpGatePathFlow {
    Then::new(
        MtpGatePrepOp {
            shared_gate,
            shared_gate_qt,
        },
        LiftResult::new(mtp_shared_gate_compute_flow()),
    )
}

/// Full MTP shared expert forward flow (excluding initial contiguous, handled in forward()).
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpSharedExpertForward(
    // Step 0: THidden → (THidden, THidden)
    Fanout<THidden>,
    // Step 1: (THidden, THidden) → (Result<TGateScore, CE>, Result<THidden, CE>)
    Zip<MtpGatePathFlow, MtpSharedFfn>,
    // Step 2: → Result<(TGateScore, THidden), CE>
    ZipOk<TGateScore, THidden, CE>,
    // Step 3: → Result<THidden, CE>
    LiftResult<MtpSharedGateApplyFlow, std::result::Result<(TGateScore, THidden), CE>, THidden>,
);
impl MtpSharedExpertForward {
    fn new(
        shared_gate: TSharedGate,
        shared_gate_qt: Option<TSharedGateQt>,
        ffn: MtpSharedFfn,
    ) -> Self {
        Self(
            Fanout::default(),
            Zip::new(mtp_gate_path_flow(shared_gate, shared_gate_qt), ffn),
            ZipOk::default(),
            LiftResult::new(mtp_shared_gate_apply_flow()),
        )
    }
}
impl std::fmt::Debug for MtpSharedExpertForward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpSharedExpertForward").finish()
    }
}

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpRoute(MtpRouteProjectOp, MtpRouteTopKOp);
impl std::fmt::Debug for MtpRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpRoute").finish()
    }
}
impl MtpRoute {
    fn new(gate: TQMatMul<Shape2<E, S>>, num_experts_per_tok: usize) -> Self {
        Self(
            MapErr::new(QMatMulOp::new(gate)),
            LiftResult::new(MapErr::new(TensorTopkFromLogitsOp::new(
                num_experts_per_tok,
            ))),
        )
    }
}

type MtpRouteProjectOp = MapErr<
    QMatMulOp<Shape2<E, S>, Shape3<B, N, S>, Shape3<B, N, E>>,
    THidden,
    TRouterLogits,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MtpTopKSelectOp = MapErr<
    TensorTopkFromLogitsOp<Shape3<B, N, E>, Shape3<B, N, Tk>, Shape3<B, N, Tk>>,
    TRouterLogits,
    (TTopWeights, TTopIndices),
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MtpRouteTopKOp =
    MapOk<MtpTopKSelectOp, TRouterLogits, (TTopWeights, TTopIndices), paramecia_core::Error>;

type MtpDispatchHiddenFlatOp = MapErr<
    FlattenPrefix2Op<Shape3<B, N, S>, Shape2<T, S>>,
    THidden,
    THiddenFlat,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;
type MtpDispatchTopFlatOp = MapErr<
    FlattenPrefix2Op<Shape3<B, N, Tk>, Shape2<T, Tk>>,
    TTopIndices,
    TTopFlat,
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type MtpRoutingCastHostOp = MapErr<
    CastFlattenToVec1PairOp<Shape2<T, Tk>, Shape2<T, Tk>, f32, u32>,
    (TTopFlat, TTopFlat),
    (Vec<f32>, Vec<u32>),
    paramecia_tensor::Error,
    paramecia_core::Error,
>;

type MtpDispatchShapeHidden = ((usize, usize, usize), THiddenFlat);
type MtpDispatchPrepJoin = (MtpDispatchShapeHidden, (Vec<u32>, Vec<f32>));
type MtpDispatchPrepResult = std::result::Result<MtpDispatchPrepJoin, paramecia_core::Error>;

#[derive(Clone)]
struct MtpDispatchPrepared {
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    hidden_flat: THiddenFlat,
    top_k_indices_vec: Vec<u32>,
    top_k_weights_vec: Vec<f32>,
}

impl TryFrom<MtpDispatchPrepJoin> for MtpDispatchPrepared {
    type Error = paramecia_core::Error;

    fn try_from(input: MtpDispatchPrepJoin) -> std::result::Result<Self, Self::Error> {
        let (
            ((batch_size, seq_len, hidden_dim), hidden_flat),
            (top_k_indices_vec, top_k_weights_vec),
        ) = input;
        Ok(Self {
            batch_size,
            seq_len,
            hidden_dim,
            hidden_flat,
            top_k_indices_vec,
            top_k_weights_vec,
        })
    }
}
type BuildMtpDispatchPreparedOp =
    TryFromOp<MtpDispatchPrepJoin, MtpDispatchPrepared, paramecia_core::Error>;

type MtpHiddenDispatchPrepFlow = Then<
    Fanout<THidden>,
    Then<
        Zip<DispatchShapeOp, MtpDispatchHiddenFlatOp>,
        ZipOk<(usize, usize, usize), THiddenFlat, paramecia_core::Error>,
    >,
>;
fn mtp_hidden_dispatch_prep_flow() -> MtpHiddenDispatchPrepFlow {
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

type MtpRoutingVecFlow = Then<
    Zip<MtpDispatchTopFlatOp, MtpDispatchTopFlatOp>,
    ZipOk<TTopFlat, TTopFlat, paramecia_core::Error>,
>;
fn mtp_routing_vec_flow() -> MtpRoutingVecFlow {
    Then::new(
        Zip::new(
            MapErr::new(FlattenPrefix2Op::default()),
            MapErr::new(FlattenPrefix2Op::default()),
        ),
        ZipOk::default(),
    )
}

#[derive(Debug, Clone)]
struct MtpDispatchExpertLoopOp {
    gate_exps: ExpertWeightTensor,
    up_exps: ExpertWeightTensor,
    down_exps: ExpertWeightTensor,
    num_experts: usize,
    num_experts_per_tok: usize,
}
impl MtpDispatchExpertLoopOp {
    fn new(
        gate_exps: ExpertWeightTensor,
        up_exps: ExpertWeightTensor,
        down_exps: ExpertWeightTensor,
        num_experts: usize,
        num_experts_per_tok: usize,
    ) -> Self {
        Self {
            gate_exps,
            up_exps,
            down_exps,
            num_experts,
            num_experts_per_tok,
        }
    }
}
#[primitive(property = Arrow)]
impl Combinator for MtpDispatchExpertLoopOp {
    type In = MtpDispatchPrepared;
    type Out = Result<THidden>;
    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let prepared = input;
        let b = prepared.batch_size;
        let l = prepared.seq_len;
        let hidden_dim = prepared.hidden_dim;
        let token_count = b.checked_mul(l).ok_or_else(|| {
            paramecia_core::Error::Msg("mtp dispatch token count over".to_string())
        })?;
        let hidden_flat = prepared.hidden_flat;
        let mut ys: THiddenFlat = Tensor::zeros(
            (token_count, hidden_dim),
            hidden_flat.inner().dtype(),
            hidden_flat.inner().device(),
        )?
        .try_into()?;
        let device = hidden_flat.inner().device();
        let mut zip_vec = ZipVecOp::<u32, f32>::default();
        let routed_pairs = <ZipVecOp<u32, f32> as Combinator<()>>::forward(
            &mut zip_vec,
            &mut (),
            (prepared.top_k_indices_vec, prepared.top_k_weights_vec),
        )
        .map_err(|e| paramecia_core::Error::Msg(format!("mtp dispatch route zip failed: {e}")))?;
        let mut chunk_vec = ChunkVecOp::<(u32, f32)>::new(self.num_experts_per_tok);
        let token_routes = <ChunkVecOp<(u32, f32)> as Combinator<()>>::forward(
            &mut chunk_vec,
            &mut (),
            routed_pairs,
        )
        .map_err(|e| paramecia_core::Error::Msg(format!("mtp dispatch route chunk failed: {e}")))?;
        if token_routes.len() != token_count {
            return Err(paramecia_core::Error::Msg(format!(
                "mtp dispatch route token count mismatch: got {}, expected {}",
                token_routes.len(),
                token_count
            )));
        }
        let mut enumerate_vec = EnumerateVecOp::<Vec<(u32, f32)>>::default();
        let indexed_token_routes = <EnumerateVecOp<Vec<(u32, f32)>> as Combinator<()>>::forward(
            &mut enumerate_vec,
            &mut (),
            token_routes,
        );
        let mut expert_pairs = Vec::new();
        for (token_idx, routes) in indexed_token_routes {
            for (expert_idx, weight) in routes {
                expert_pairs.push((expert_idx as usize, (token_idx as u32, weight)));
            }
        }
        let mut group_by_expert = GroupByIndexOp::<(u32, f32)>::default();
        let grouped_assignments = <GroupByIndexOp<(u32, f32)> as Combinator<()>>::forward(
            &mut group_by_expert,
            &mut (),
            (self.num_experts, expert_pairs),
        )
        .map_err(|e| {
            paramecia_core::Error::Msg(format!("mtp dispatch expert grouping failed: {e}"))
        })?;

        for (expert_idx, assignments) in grouped_assignments.into_iter().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let (token_indices, weights): (Vec<u32>, Vec<f32>) = assignments.into_iter().unzip();

            let mut from_token_indices = FromVec1OnDeviceOp::<Shape1<T>, u32>::default();
            let token_tensor = <FromVec1OnDeviceOp<Shape1<T>, u32> as Combinator<()>>::forward(
                &mut from_token_indices,
                &mut (),
                (token_indices, device.clone()),
            )?;
            let mut from_weights = FromVecColOnDeviceOp::<Shape2<T, U1>, f32>::default();
            let weights_col =
                <FromVecColOnDeviceOp<Shape2<T, U1>, f32> as Combinator<()>>::forward(
                    &mut from_weights,
                    &mut (),
                    (weights, device.clone()),
                )?;
            let weights_tensor_typed: TWeightCol = weights_col
                .to_dtype(hidden_flat.inner().dtype())?
                .to_device(hidden_flat.inner().device())?;
            let mut index_select =
                IndexSelectDim0Op::<Shape2<T, S>, Shape1<T>, Shape2<T, S>>::default();
            let expert_input: THiddenFlat = <IndexSelectDim0Op<
                Shape2<T, S>,
                Shape1<T>,
                Shape2<T, S>,
            > as Combinator<()>>::forward(
                &mut index_select,
                &mut (),
                (hidden_flat.clone(), token_tensor.clone()),
            )?;

            let gate_slice = self
                .gate_exps
                .read()
                .map_err(|_| {
                    paramecia_core::Error::Msg("failed to lock mtp gate expert tensor".to_string())
                })?
                .slice_first_dim(expert_idx)?;
            let up_slice = self
                .up_exps
                .read()
                .map_err(|_| {
                    paramecia_core::Error::Msg("failed to lock mtp up expert tensor".to_string())
                })?
                .slice_first_dim(expert_idx)?;
            let down_slice = self
                .down_exps
                .read()
                .map_err(|_| {
                    paramecia_core::Error::Msg("failed to lock mtp down expert tensor".to_string())
                })?
                .slice_first_dim(expert_idx)?;

            let mut gate_qmatmul = QMatMulFromQTensorOp::<Shape2<SI, S>>::default();
            let gate_mm: TQMatMul<Shape2<SI, S>> =
                <QMatMulFromQTensorOp<Shape2<SI, S>> as Combinator<()>>::forward(
                    &mut gate_qmatmul,
                    &mut (),
                    gate_slice,
                )?;
            let mut up_qmatmul = QMatMulFromQTensorOp::<Shape2<SI, S>>::default();
            let up_mm: TQMatMul<Shape2<SI, S>> =
                <QMatMulFromQTensorOp<Shape2<SI, S>> as Combinator<()>>::forward(
                    &mut up_qmatmul,
                    &mut (),
                    up_slice,
                )?;
            let mut down_qmatmul = QMatMulFromQTensorOp::<Shape2<S, SI>>::default();
            let down_mm: TQMatMul<Shape2<S, SI>> =
                <QMatMulFromQTensorOp<Shape2<S, SI>> as Combinator<()>>::forward(
                    &mut down_qmatmul,
                    &mut (),
                    down_slice,
                )?;

            let gate_out: TIntermediateFlat = gate_mm.forward(&expert_input)?;
            let up_out: TIntermediateFlat = up_mm.forward(&expert_input)?;
            let gate_out = paramecia_tensor::silu!(gate_out)?;
            let activated: TIntermediateFlat = paramecia_tensor::broadcast_mul!(gate_out, up_out)?;
            let expert_output: THiddenFlat = down_mm.forward(&activated)?;

            let weighted: THiddenFlat =
                paramecia_tensor::broadcast_mul!(&expert_output, weights_tensor_typed)?;
            let mut index_add = IndexAddDim0Op::<Shape2<T, S>, Shape1<T>, Shape2<T, S>>::default();
            ys =
                <IndexAddDim0Op<Shape2<T, S>, Shape1<T>, Shape2<T, S>> as Combinator<()>>::forward(
                    &mut index_add,
                    &mut (),
                    (ys, token_tensor, weighted),
                )?;
        }

        let out: THidden = ys
            .inner()
            .reshape((b, l, hidden_dim))?
            .try_into()
            .map_err(paramecia_core::Error::from)?;
        Ok(out)
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpDispatchExpertLoopOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpDispatchExpertLoop").with_output_type::<Result<THidden>>()
    }
}

type MtpDispatchLoopStep = LiftResult<
    MtpDispatchExpertLoopOp,
    std::result::Result<MtpDispatchPrepared, paramecia_core::Error>,
    THidden,
>;

type MtpRouteJoinResult =
    std::result::Result<(THidden, (TTopWeights, TTopIndices)), paramecia_core::Error>;
#[derive(Debug, Default, Clone, Copy)]
struct MtpPrepareHiddenStepOp;
#[primitive(property = Arrow)]
impl Combinator for MtpPrepareHiddenStepOp {
    type In = THidden;
    type Out = std::result::Result<MtpDispatchShapeHidden, paramecia_core::Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let mut hidden_flow = mtp_hidden_dispatch_prep_flow();
        <MtpHiddenDispatchPrepFlow as Combinator<()>>::forward(&mut hidden_flow, &mut (), input)
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpPrepareHiddenStepOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "MtpPrepareHiddenStep",
            <MtpHiddenDispatchPrepFlow as Vis>::visualize(),
        )
        .with_output_type::<std::result::Result<MtpDispatchShapeHidden, paramecia_core::Error>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpPrepareRouteStepOp;
#[primitive(property = Arrow)]
impl Combinator for MtpPrepareRouteStepOp {
    type In = (TTopWeights, TTopIndices);
    type Out = std::result::Result<(Vec<u32>, Vec<f32>), paramecia_core::Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let mut route_flow = mtp_routing_vec_flow();
        let flat_pair =
            <MtpRoutingVecFlow as Combinator<()>>::forward(&mut route_flow, &mut (), input)?;
        let mut host_cast = MtpRoutingCastHostOp::new(CastFlattenToVec1PairOp::default());
        let host_pair =
            <MtpRoutingCastHostOp as Combinator<()>>::forward(&mut host_cast, &mut (), flat_pair)?;
        let mut swap = Swap2::<Vec<f32>, Vec<u32>>::default();
        Ok(<Swap2<Vec<f32>, Vec<u32>> as Combinator<()>>::forward(
            &mut swap,
            &mut (),
            host_pair,
        ))
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpPrepareRouteStepOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "MtpPrepareRouteStep",
            Graph::sequence(
                <MtpRoutingVecFlow as Vis>::visualize(),
                Graph::sequence(
                    <MtpRoutingCastHostOp as Vis>::visualize(),
                    <Swap2<Vec<f32>, Vec<u32>> as Vis>::visualize(),
                ),
            ),
        )
        .with_output_type::<std::result::Result<(Vec<u32>, Vec<f32>), paramecia_core::Error>>()
    }
}

type MtpPrepareDispatchFromRouteOp = Then<
    Zip<MtpPrepareHiddenStepOp, MtpPrepareRouteStepOp>,
    Then<
        ZipOk<MtpDispatchShapeHidden, (Vec<u32>, Vec<f32>), paramecia_core::Error>,
        LiftResult<BuildMtpDispatchPreparedOp, MtpDispatchPrepResult, MtpDispatchPrepared>,
    >,
>;
fn mtp_prepare_dispatch_from_route_op() -> MtpPrepareDispatchFromRouteOp {
    Then::new(
        Zip::new(MtpPrepareHiddenStepOp, MtpPrepareRouteStepOp),
        Then::new(ZipOk::default(), LiftResult::new(TryFromOp::default())),
    )
}
type MtpRouteDispatchPrepLift =
    LiftResult<MtpPrepareDispatchFromRouteOp, MtpRouteJoinResult, MtpDispatchPrepared>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpRouteDispatch(
    Fanout<THidden>,
    Zip<WrapOk<THidden, paramecia_core::Error>, MtpRoute>,
    ZipOk<THidden, (TTopWeights, TTopIndices), paramecia_core::Error>,
    MtpRouteDispatchPrepLift,
    MtpDispatchLoopStep,
);
impl MtpRouteDispatch {
    fn new(
        gate: TQMatMul<Shape2<E, S>>,
        gate_exps: ExpertWeightTensor,
        up_exps: ExpertWeightTensor,
        down_exps: ExpertWeightTensor,
        num_experts: usize,
        num_experts_per_tok: usize,
    ) -> Self {
        Self(
            Fanout::default(),
            Zip::new(
                WrapOk::default(),
                MtpRoute::new(gate.clone(), num_experts_per_tok),
            ),
            ZipOk::default(),
            LiftResult::new(mtp_prepare_dispatch_from_route_op()),
            LiftResult::new(MtpDispatchExpertLoopOp::new(
                gate_exps,
                up_exps,
                down_exps,
                num_experts,
                num_experts_per_tok,
            )),
        )
    }
}
impl std::fmt::Debug for MtpRouteDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpRouteDispatch").finish()
    }
}

#[allow(dead_code)]
struct MtpMoeForwardGraph;
#[primitive(property = Visualize)]
impl Vis for MtpMoeForwardGraph {
    fn visualize() -> Graph {
        let route = <MtpRouteDispatch as Vis>::visualize();
        let shared_select = Graph::zip_custom(
            "MtpSharedExpertSelect",
            vec![
                (
                    "present",
                    Graph::sequence(
                        <MtpSharedExpertForwardGraph as Vis>::visualize(),
                        Graph::custom_leaf("ResidualAdd"),
                    ),
                ),
                ("absent", Graph::custom_leaf("Identity")),
            ],
        );
        Graph::wrap_custom_subgraph("MtpMoeForwardGraph", Graph::sequence(route, shared_select))
    }
}

/// Inner flow for MtpMoeBlock when shared expert is present:
/// route_dispatch and shared_expert run in parallel, results combined via residual add.
#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpMoeWithSharedInner(
    // Step 0: THidden → (THidden, THidden)
    Fanout<THidden>,
    // Step 1: → (Result<THidden, CE>, Result<THidden, CE>)
    Zip<MtpRouteDispatch, MtpSharedExpertForward>,
    // Step 2: → Result<(THidden, THidden), CE>
    ZipOk<THidden, THidden, CE>,
    // Step 3: → Result<THidden, CE>
    LiftResult<ResidualAddHiddenFlow, std::result::Result<(THidden, THidden), CE>, THidden>,
);
impl MtpMoeWithSharedInner {
    fn new(route_dispatch: MtpRouteDispatch, shared_expert: MtpSharedExpertForward) -> Self {
        Self(
            Fanout::default(),
            Zip::new(route_dispatch, shared_expert),
            ZipOk::default(),
            LiftResult::new(MapErr::new(ResidualAddOp::default())),
        )
    }
}
impl std::fmt::Debug for MtpMoeWithSharedInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpMoeWithSharedInner").finish()
    }
}

/// Full MtpMoeBlock forward flow: contiguous → route (+ optional shared expert residual).
///
/// Uses a construction-time enum to avoid constructing unused branch weights.
enum MtpMoeBlockForwardFlow {
    WithShared(MtpMoeWithSharedInner),
    RouteOnly(MtpRouteDispatch),
}
impl MtpMoeBlockForwardFlow {
    fn new(route_dispatch: MtpRouteDispatch, shared_expert: Option<MtpSharedExpert>) -> Self {
        if let Some(shared) = shared_expert {
            Self::WithShared(MtpMoeWithSharedInner::new(route_dispatch, shared.flow))
        } else {
            Self::RouteOnly(route_dispatch)
        }
    }
}
#[primitive(property = Arrow)]
impl Combinator for MtpMoeBlockForwardFlow {
    type In = THidden;
    type Out = std::result::Result<THidden, CE>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        match self {
            Self::WithShared(flow) => {
                <MtpMoeWithSharedInner as Combinator<()>>::forward(flow, &mut (), input)
            }
            Self::RouteOnly(flow) => {
                <MtpRouteDispatch as Combinator<()>>::forward(flow, &mut (), input)
            }
        }
    }
}
impl std::fmt::Debug for MtpMoeBlockForwardFlow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpMoeBlockForward").finish()
    }
}

// ============================================================================
// MTP (Multi-Token Prediction) Head for Speculative Decoding
// ============================================================================

/// MTP projection layer (top-level tensors).
///
/// Projects concatenated [hidden_state || token_embedding] to MTP hidden dimension.
/// Architecture:
/// ```text
/// hidden_state (from main model's final layer) + embedding(x_{t+1})
///     |
///     v
/// [pre_fc_norm_hidden(hidden) || pre_fc_norm_embedding(embedding)]  -> [batch, seq, hidden*2]
///     |
///     v
/// mtp.fc (linear projection) -> [batch, seq, mtp_hidden_size]
/// ```
#[derive(Debug)]
pub(super) struct MtpProjection {
    /// RmsNorm for hidden state input: mtp.pre_fc_norm_hidden.weight
    pub(super) pre_fc_norm_hidden: RmsNorm,
    /// RmsNorm for embedding input: mtp.pre_fc_norm_embedding.weight
    pub(super) pre_fc_norm_embedding: RmsNorm,
    /// Linear projection: mtp.fc.weight [mtp_hidden_size, hidden_size * 2]
    pub(super) fc: TQMatMul<Shape2<S, S2>>,
    /// Final output norm before lm_head: mtp.norm.weight
    /// CRITICAL: Must apply before shared lm_head
    pub(super) output_norm: RmsNorm,
}

/// MTP attention layer (full attention with partial RoPE).
///
/// Uses the same RoPE configuration as the main model (partial rotation with n_rot dimensions).
/// KV cache is cleared after each speculation round (fresh start from h_t).
pub(super) struct MtpAttention {
    #[allow(dead_code)]
    pub(super) wo: TQMatMul<Shape2<S, AH>>,
    pub(super) num_heads: usize,
    #[allow(dead_code)]
    pub(super) num_kv_heads: usize,
    pub(super) num_kv_groups: usize,
    /// Number of heads for output projection (may differ from num_heads)
    pub(super) o_num_heads: usize,
    pub(super) head_dim: usize,
    /// Shared with main model, respects n_rot (partial RoPE)
    pub(super) rotary_emb: Arc<RotaryEmbedding>,
    /// KV cache - cleared after each verification round
    pub(super) kv_cache: Option<(TKVCache, TKVCache)>,
    /// Whether Q projection outputs [Q, gate] (gated attention)
    pub(super) gated_attention: bool,
    /// Persistent Arrow stages for projection/reshape.
    qkv_prep: QkvPrep,
    post_attn_gate: PostAttnGateFlow,
    output_project: OutputProjectFlow,
}

impl std::fmt::Debug for MtpAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpAttention")
            .field("wo", &self.wo)
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("num_kv_groups", &self.num_kv_groups)
            .field("o_num_heads", &self.o_num_heads)
            .field("head_dim", &self.head_dim)
            .field("rotary_emb", &self.rotary_emb)
            .field("has_kv_cache", &self.kv_cache.is_some())
            .field("gated_attention", &self.gated_attention)
            .field("qkv_prep", &self.qkv_prep)
            .field("post_attn_gate", &"PostAttnGateFlow")
            .field("output_project", &"QMatMulOp")
            .finish()
    }
}

/// MTP MoE block (mirrors main MoeBlock structure but may use different expert count).
pub(super) struct MtpMoeBlock {
    flow: MtpMoeBlockForwardFlow,
}
impl std::fmt::Debug for MtpMoeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpMoeBlock").finish()
    }
}

/// Shared expert for MTP MoE (dense FFN, not routed)
pub(super) struct MtpSharedExpert {
    flow: MtpSharedExpertForward,
}

impl std::fmt::Debug for MtpSharedExpert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpSharedExpert")
            .field("", &self.flow)
            .finish()
    }
}

/// Complete MTP transformer block.
///
/// Architecture per block:
/// ```text
/// x -> in_norm -> MtpAttention -> residual -> post_attn_norm -> MtpMoeBlock -> residual -> output
/// ```
#[derive(Debug)]
pub(super) struct MtpBlock {
    pub(super) attn: MtpAttention,
    pub(super) moe_block: MtpMoeBlock,
    pub(super) in_norm: RmsNorm,
    pub(super) post_attn_norm: RmsNorm,
}

/// MTP head for multi-token prediction (speculative decoding).
///
/// Enables predicting multiple future tokens in a single forward pass.
/// Used for speculative decoding where MTP generates draft tokens that
/// are then verified by the main model.
///
/// ## Architecture
///
/// Given main model hidden state h_t and predicted token x_{t+1}:
/// 1. Project [h_t || embed(x_{t+1})] through MTP projection
/// 2. Process through MTP transformer blocks (attention + MoE)
/// 3. Apply output norm and shared lm_head to get x_{t+2} logits
/// 4. Repeat for additional speculation depths
///
/// ## RoPE Position Offset
///
/// MTP predicts token x_{t+k+1} given context at position t+k.
/// The RoPE position for depth k MTP is `base_offset + k` where
/// base_offset is the main model's current position.
#[derive(Debug)]
pub struct MtpHead {
    pub(super) projection: MtpProjection,
    pub(super) blocks: Vec<MtpBlock>,
    /// Hidden size of MTP layers (may differ from main model)
    #[allow(dead_code)]
    pub(super) hidden_size: usize,
}

fn run_mtp_blocks(blocks: &mut [MtpBlock], h: THidden, offset: usize) -> Result<THidden> {
    let block_count = blocks.len();
    let mut step_ctx = MtpBlockForwardStepCtx { blocks };
    let step = TryIndexMutCtx::new(
        mtp_block_slice,
        mtp_block_index,
        mtp_block_forward_indexed,
        mtp_block_index_oob,
    );
    let mut fold = TryFoldRange::<_, MtpBlockState, paramecia_core::Error>::new(step);
    let (h, _offset) = fold.traced_forward(&mut step_ctx, ((h, offset), 0..block_count))?;
    Ok(h)
}

fn clear_mtp_block_caches(blocks: &mut [MtpBlock]) -> Result<()> {
    let block_count = blocks.len();
    let mut ctx = MtpBlockClearCtx { blocks };
    let mut fold = TryFoldSliceMut::new(
        mtp_clear_block_slice,
        mtp_clear_block_step,
        mtp_block_index_oob,
    );
    fold.traced_forward(&mut ctx, ((), 0..block_count))
}

struct MtpBlockForwardStepCtx<'a> {
    blocks: &'a mut [MtpBlock],
}
fn mtp_block_slice<'a>(ctx: &'a mut MtpBlockForwardStepCtx<'_>) -> &'a mut [MtpBlock] {
    ctx.blocks
}
type MtpBlockState = (THidden, usize);
fn mtp_block_index(input: &(MtpBlockState, usize)) -> usize {
    input.1
}
fn mtp_block_index_oob(idx: usize, len: usize) -> paramecia_core::Error {
    paramecia_core::Error::Msg(format!(
        "mtp block index {} out of bounds (len {})",
        idx, len
    ))
}
fn mtp_block_forward_indexed(
    block: &mut MtpBlock,
    input: (MtpBlockState, usize),
) -> Result<MtpBlockState> {
    let ((h, offset), _idx) = input;
    let out = block.forward(&h, offset)?;
    Ok((out, offset))
}

struct MtpBlockClearCtx<'a> {
    blocks: &'a mut [MtpBlock],
}
fn mtp_clear_block_slice<'a>(ctx: &'a mut MtpBlockClearCtx<'_>) -> &'a mut [MtpBlock] {
    ctx.blocks
}
fn mtp_clear_block_step(block: &mut MtpBlock, state: (), _idx: usize) -> Result<()> {
    block.clear_kv_cache();
    Ok(state)
}

struct MtpBatchedDepthState {
    current_hidden: THidden,
    current_token: Tensor,
    spec_tokens: Vec<Tensor>,
    spec_logits: Vec<Tensor>,
}

struct MtpBatchedWithEmbed {
    state: MtpBatchedDepthState,
    depth: usize,
    token_embed: Tensor,
}

struct MtpBatchedWithHidden {
    state: MtpBatchedDepthState,
    hidden: THidden,
}

struct MtpBatchedDepthStepCtx<'a> {
    projection: &'a mut MtpProjection,
    blocks: &'a mut [MtpBlock],
    embed_tokens: &'a Embedding,
    lm_head: &'a TLmHead,
    mtp_device: Device,
    primary_device: Device,
    base_offset: usize,
}
#[derive(Debug, Default, Clone, Copy)]
struct MtpBatchedEmbedOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpBatchedDepthStepCtx<'a>> for MtpBatchedEmbedOp {
    type In = (MtpBatchedDepthState, usize);
    type Out = Result<MtpBatchedWithEmbed>;

    fn forward(&mut self, ctx: &mut MtpBatchedDepthStepCtx<'a>, input: Self::In) -> Self::Out {
        let (state, depth) = input;
        let token_for_embed = state.current_token.flatten_all()?.reshape((1, 1))?;
        let token_embed = ctx.embed_tokens.forward(&token_for_embed)?;
        let token_embed = super::transfer_to(token_embed, &ctx.mtp_device)?;
        Ok(MtpBatchedWithEmbed {
            state,
            depth,
            token_embed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpBatchedEmbedOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpBatchedEmbed").with_output_type::<Result<MtpBatchedWithEmbed>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpBatchedProjectBlocksOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpBatchedDepthStepCtx<'a>> for MtpBatchedProjectBlocksOp {
    type In = MtpBatchedWithEmbed;
    type Out = Result<MtpBatchedWithHidden>;

    fn forward(&mut self, ctx: &mut MtpBatchedDepthStepCtx<'a>, input: Self::In) -> Self::Out {
        let MtpBatchedWithEmbed {
            state,
            depth,
            token_embed,
        } = input;
        let hidden_normed: THidden = ctx
            .projection
            .pre_fc_norm_hidden
            .forward(state.current_hidden.inner())?
            .try_into()?;
        let embed_normed: THidden = ctx
            .projection
            .pre_fc_norm_embedding
            .forward(&token_embed)?
            .try_into()?;

        let combined_inputs = [&embed_normed, &hidden_normed];
        let combined: THiddenCat = paramecia_tensor::cat!(combined_inputs.as_slice(), U2 => Sx2)?;
        let combined_typed: paramecia_tensor::Tensor<Shape3<B, N, S2>> =
            combined.into_inner().try_into()?;
        let hidden: THidden = ctx.projection.fc.forward(&combined_typed)?;
        let hidden = run_mtp_blocks(ctx.blocks, hidden, ctx.base_offset + depth)?;
        Ok(MtpBatchedWithHidden { state, hidden })
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpBatchedProjectBlocksOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpBatchedProjectBlocks")
            .with_output_type::<Result<MtpBatchedWithHidden>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpBatchedFinalizeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpBatchedDepthStepCtx<'a>> for MtpBatchedFinalizeOp {
    type In = MtpBatchedWithHidden;
    type Out = Result<MtpBatchedDepthState>;

    fn forward(&mut self, ctx: &mut MtpBatchedDepthStepCtx<'a>, input: Self::In) -> Self::Out {
        let MtpBatchedWithHidden { mut state, hidden } = input;
        let hidden_normed = ctx.projection.output_norm.forward(hidden.inner())?;
        let hidden_for_lm = super::transfer_to(hidden_normed, &ctx.primary_device)?;
        let hidden_for_lm_typed: THidden = hidden_for_lm.contiguous()?.try_into()?;
        let logits = MtpHead::lm_head_forward(ctx.lm_head, &hidden_for_lm_typed)?.squeeze(1)?;
        let token = logits.argmax(D::Minus1)?;

        state.spec_logits.push(logits);
        state.spec_tokens.push(token.clone());
        state.current_token = token;
        state.current_hidden = hidden;
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpBatchedFinalizeOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpBatchedFinalize").with_output_type::<Result<MtpBatchedDepthState>>()
    }
}

type MtpBatchedEmbedResult = std::result::Result<MtpBatchedWithEmbed, paramecia_core::Error>;
type MtpBatchedProjectLift =
    LiftResult<MtpBatchedProjectBlocksOp, MtpBatchedEmbedResult, MtpBatchedWithHidden>;
type MtpBatchedProjectResult = std::result::Result<MtpBatchedWithHidden, paramecia_core::Error>;
type MtpBatchedFinalizeLift =
    LiftResult<MtpBatchedFinalizeOp, MtpBatchedProjectResult, MtpBatchedDepthState>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpBatchedDepthStep(
    MtpBatchedEmbedOp,
    MtpBatchedProjectLift,
    MtpBatchedFinalizeLift,
);
impl MtpBatchedDepthStep {
    fn new() -> Self {
        Self(
            MtpBatchedEmbedOp,
            LiftResult::new(MtpBatchedProjectBlocksOp),
            LiftResult::new(MtpBatchedFinalizeOp),
        )
    }
}
impl std::fmt::Debug for MtpBatchedDepthStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpBatchedDepthStep").finish()
    }
}
fn mtp_batched_depth_step_op() -> MtpBatchedDepthStep {
    MtpBatchedDepthStep::new()
}

struct MtpTrainingDepthState {
    logits: Vec<Tensor>,
}

struct MtpTrainingDepthStepCtx<'a> {
    projection: &'a mut MtpProjection,
    blocks: &'a mut [MtpBlock],
    hidden_states: &'a THidden,
    token_embeds: &'a THidden,
    lm_head: &'a TLmHead,
    base_offset: usize,
    seq_len: usize,
}

struct MtpTrainingDepthPrepared {
    depth: usize,
    hidden_slice: THidden,
    embed_slice: THidden,
}

struct MtpTrainingDepthProjected {
    state: MtpTrainingDepthState,
    hidden: THidden,
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpTrainingDepthResolveOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpTrainingDepthStepCtx<'a>> for MtpTrainingDepthResolveOp {
    type In = (MtpTrainingDepthState, usize);
    type Out = Result<(MtpTrainingDepthState, Option<MtpTrainingDepthPrepared>)>;

    fn forward(&mut self, ctx: &mut MtpTrainingDepthStepCtx<'a>, input: Self::In) -> Self::Out {
        let (state, depth) = input;
        let valid_len = ctx.seq_len.saturating_sub(depth + 1);
        if valid_len == 0 {
            return Ok((state, None));
        }

        let hidden_slice: THidden = ctx
            .hidden_states
            .inner()
            .narrow(1, 0, valid_len)?
            .contiguous()?
            .try_into()?;
        let embed_offset = depth + 1;
        let embed_slice: THidden = ctx
            .token_embeds
            .inner()
            .narrow(1, embed_offset, valid_len)?
            .contiguous()?
            .try_into()?;

        Ok((
            state,
            Some(MtpTrainingDepthPrepared {
                depth,
                hidden_slice,
                embed_slice,
            }),
        ))
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpTrainingDepthResolveOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpTrainingDepthResolve")
            .with_output_type::<Result<(MtpTrainingDepthState, Option<MtpTrainingDepthPrepared>)>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpTrainingDepthProjectBlocksOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpTrainingDepthStepCtx<'a>> for MtpTrainingDepthProjectBlocksOp {
    type In = (MtpTrainingDepthState, MtpTrainingDepthPrepared);
    type Out = Result<MtpTrainingDepthProjected>;

    fn forward(&mut self, ctx: &mut MtpTrainingDepthStepCtx<'a>, input: Self::In) -> Self::Out {
        let (state, prepared) = input;
        let hidden_normed: THidden = ctx
            .projection
            .pre_fc_norm_hidden
            .forward(prepared.hidden_slice.inner())?
            .try_into()?;
        let embed_normed: THidden = ctx
            .projection
            .pre_fc_norm_embedding
            .forward(prepared.embed_slice.inner())?
            .try_into()?;

        let combined_inputs = [&embed_normed, &hidden_normed];
        let combined: THiddenCat = paramecia_tensor::cat!(combined_inputs.as_slice(), U2 => Sx2)?;
        let combined_typed: paramecia_tensor::Tensor<Shape3<B, N, S2>> =
            combined.into_inner().try_into()?;
        let hidden: THidden = ctx.projection.fc.forward(&combined_typed)?;
        let hidden = run_mtp_blocks(ctx.blocks, hidden, ctx.base_offset + prepared.depth)?;
        Ok(MtpTrainingDepthProjected { state, hidden })
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpTrainingDepthProjectBlocksOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpTrainingDepthProjectBlocks")
            .with_output_type::<Result<MtpTrainingDepthProjected>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpTrainingDepthFinalizeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpTrainingDepthStepCtx<'a>> for MtpTrainingDepthFinalizeOp {
    type In = MtpTrainingDepthProjected;
    type Out = Result<MtpTrainingDepthState>;

    fn forward(&mut self, ctx: &mut MtpTrainingDepthStepCtx<'a>, input: Self::In) -> Self::Out {
        let MtpTrainingDepthProjected { mut state, hidden } = input;
        let hidden_final = ctx.projection.output_norm.forward(hidden.inner())?;
        let hidden_final_typed: THidden = hidden_final.contiguous()?.try_into()?;
        let logits = MtpHead::lm_head_forward(ctx.lm_head, &hidden_final_typed)?;
        state.logits.push(logits);
        clear_mtp_block_caches(ctx.blocks)?;
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpTrainingDepthFinalizeOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpTrainingDepthFinalize")
            .with_output_type::<Result<MtpTrainingDepthState>>()
    }
}

type MtpTrainingDepthProjectResult =
    std::result::Result<MtpTrainingDepthProjected, paramecia_core::Error>;
type MtpTrainingDepthFinalizeLift =
    LiftResult<MtpTrainingDepthFinalizeOp, MtpTrainingDepthProjectResult, MtpTrainingDepthState>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpTrainingDepthRun(
    MtpTrainingDepthProjectBlocksOp,
    MtpTrainingDepthFinalizeLift,
);
impl MtpTrainingDepthRun {
    fn new() -> Self {
        Self(
            MtpTrainingDepthProjectBlocksOp,
            LiftResult::new(MtpTrainingDepthFinalizeOp),
        )
    }
}
impl std::fmt::Debug for MtpTrainingDepthRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpTrainingDepthRun").finish()
    }
}

type MtpTrainingDepthOptionRun = OptionThen<
    MtpTrainingDepthRun,
    MtpTrainingDepthState,
    MtpTrainingDepthPrepared,
    paramecia_core::Error,
>;
type MtpTrainingDepthResolveResult = std::result::Result<
    (MtpTrainingDepthState, Option<MtpTrainingDepthPrepared>),
    paramecia_core::Error,
>;
type MtpTrainingDepthOptionLift =
    LiftResult<MtpTrainingDepthOptionRun, MtpTrainingDepthResolveResult, MtpTrainingDepthState>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpTrainingDepthStep(MtpTrainingDepthResolveOp, MtpTrainingDepthOptionLift);
impl MtpTrainingDepthStep {
    fn new() -> Self {
        Self(
            MtpTrainingDepthResolveOp,
            LiftResult::new(OptionThen::new(MtpTrainingDepthRun::new())),
        )
    }
}
impl std::fmt::Debug for MtpTrainingDepthStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpTrainingDepthStep").finish()
    }
}
fn mtp_training_depth_step_op() -> MtpTrainingDepthStep {
    MtpTrainingDepthStep::new()
}

struct MtpTrainingWeightedDepthStepCtx<'a> {
    projection: &'a mut MtpProjection,
    blocks: &'a mut [MtpBlock],
    hidden_states: &'a THidden,
    weighted_embeds_per_depth: &'a [THidden],
    lm_head: &'a TLmHead,
    base_offset: usize,
    seq_len: usize,
}

struct MtpTrainingWeightedDepthPrepared {
    depth: usize,
    hidden_slice: THidden,
    embed_slice: THidden,
}

struct MtpTrainingWeightedDepthProjected {
    state: MtpTrainingDepthState,
    hidden: THidden,
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpTrainingWeightedDepthResolveOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpTrainingWeightedDepthStepCtx<'a>> for MtpTrainingWeightedDepthResolveOp {
    type In = (MtpTrainingDepthState, usize);
    type Out = Result<(
        MtpTrainingDepthState,
        Option<MtpTrainingWeightedDepthPrepared>,
    )>;

    fn forward(
        &mut self,
        ctx: &mut MtpTrainingWeightedDepthStepCtx<'a>,
        input: Self::In,
    ) -> Self::Out {
        let (state, depth) = input;
        let weighted_embeds = match ctx.weighted_embeds_per_depth.get(depth) {
            Some(v) => v,
            None => return Ok((state, None)),
        };
        let valid_len = weighted_embeds.inner().dim(1)?;
        if valid_len == 0 {
            return Ok((state, None));
        }

        let embed_slice: THidden = weighted_embeds.clone();
        let hidden_slice: THidden = if valid_len < ctx.seq_len {
            ctx.hidden_states
                .inner()
                .narrow(1, 0, valid_len)?
                .contiguous()?
                .try_into()?
        } else {
            ctx.hidden_states.clone()
        };

        Ok((
            state,
            Some(MtpTrainingWeightedDepthPrepared {
                depth,
                hidden_slice,
                embed_slice,
            }),
        ))
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpTrainingWeightedDepthResolveOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpTrainingWeightedDepthResolve").with_output_type::<Result<(
            MtpTrainingDepthState,
            Option<MtpTrainingWeightedDepthPrepared>,
        )>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpTrainingWeightedDepthProjectBlocksOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpTrainingWeightedDepthStepCtx<'a>>
    for MtpTrainingWeightedDepthProjectBlocksOp
{
    type In = (MtpTrainingDepthState, MtpTrainingWeightedDepthPrepared);
    type Out = Result<MtpTrainingWeightedDepthProjected>;

    fn forward(
        &mut self,
        ctx: &mut MtpTrainingWeightedDepthStepCtx<'a>,
        input: Self::In,
    ) -> Self::Out {
        let (state, prepared) = input;
        let hidden_normed: THidden = ctx
            .projection
            .pre_fc_norm_hidden
            .forward(prepared.hidden_slice.inner())?
            .try_into()?;
        let embed_normed: THidden = ctx
            .projection
            .pre_fc_norm_embedding
            .forward(prepared.embed_slice.inner())?
            .try_into()?;
        let combined_inputs = [&embed_normed, &hidden_normed];
        let combined: THiddenCat = paramecia_tensor::cat!(combined_inputs.as_slice(), U2 => Sx2)?;
        let combined_typed: paramecia_tensor::Tensor<Shape3<B, N, S2>> =
            combined.into_inner().try_into()?;
        let hidden: THidden = ctx.projection.fc.forward(&combined_typed)?;
        let hidden = run_mtp_blocks(ctx.blocks, hidden, ctx.base_offset + prepared.depth)?;
        Ok(MtpTrainingWeightedDepthProjected { state, hidden })
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpTrainingWeightedDepthProjectBlocksOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpTrainingWeightedDepthProjectBlocks")
            .with_output_type::<Result<MtpTrainingWeightedDepthProjected>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MtpTrainingWeightedDepthFinalizeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<MtpTrainingWeightedDepthStepCtx<'a>> for MtpTrainingWeightedDepthFinalizeOp {
    type In = MtpTrainingWeightedDepthProjected;
    type Out = Result<MtpTrainingDepthState>;

    fn forward(
        &mut self,
        ctx: &mut MtpTrainingWeightedDepthStepCtx<'a>,
        input: Self::In,
    ) -> Self::Out {
        let MtpTrainingWeightedDepthProjected { mut state, hidden } = input;
        let hidden_final = ctx.projection.output_norm.forward(hidden.inner())?;
        let hidden_final_typed: THidden = hidden_final.contiguous()?.try_into()?;
        let logits = MtpHead::lm_head_forward(ctx.lm_head, &hidden_final_typed)?;
        state.logits.push(logits);
        clear_mtp_block_caches(ctx.blocks)?;
        Ok(state)
    }
}
#[primitive(property = Visualize)]
impl Vis for MtpTrainingWeightedDepthFinalizeOp {
    fn visualize() -> Graph {
        Graph::custom_leaf("MtpTrainingWeightedDepthFinalize")
            .with_output_type::<Result<MtpTrainingDepthState>>()
    }
}

type MtpTrainingWeightedDepthProjectResult =
    std::result::Result<MtpTrainingWeightedDepthProjected, paramecia_core::Error>;
type MtpTrainingWeightedDepthFinalizeLift = LiftResult<
    MtpTrainingWeightedDepthFinalizeOp,
    MtpTrainingWeightedDepthProjectResult,
    MtpTrainingDepthState,
>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpTrainingWeightedDepthRun(
    MtpTrainingWeightedDepthProjectBlocksOp,
    MtpTrainingWeightedDepthFinalizeLift,
);
impl MtpTrainingWeightedDepthRun {
    fn new() -> Self {
        Self(
            MtpTrainingWeightedDepthProjectBlocksOp,
            LiftResult::new(MtpTrainingWeightedDepthFinalizeOp),
        )
    }
}
impl std::fmt::Debug for MtpTrainingWeightedDepthRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpTrainingWeightedDepthRun").finish()
    }
}

type MtpTrainingWeightedDepthOptionRun = OptionThen<
    MtpTrainingWeightedDepthRun,
    MtpTrainingDepthState,
    MtpTrainingWeightedDepthPrepared,
    paramecia_core::Error,
>;
type MtpTrainingWeightedDepthResolveResult = std::result::Result<
    (
        MtpTrainingDepthState,
        Option<MtpTrainingWeightedDepthPrepared>,
    ),
    paramecia_core::Error,
>;
type MtpTrainingWeightedDepthOptionLift = LiftResult<
    MtpTrainingWeightedDepthOptionRun,
    MtpTrainingWeightedDepthResolveResult,
    MtpTrainingDepthState,
>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct MtpTrainingWeightedDepthStep(
    MtpTrainingWeightedDepthResolveOp,
    MtpTrainingWeightedDepthOptionLift,
);
impl MtpTrainingWeightedDepthStep {
    fn new() -> Self {
        Self(
            MtpTrainingWeightedDepthResolveOp,
            LiftResult::new(OptionThen::new(MtpTrainingWeightedDepthRun::new())),
        )
    }
}
impl std::fmt::Debug for MtpTrainingWeightedDepthStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpTrainingWeightedDepthStep").finish()
    }
}
fn mtp_training_weighted_depth_step_op() -> MtpTrainingWeightedDepthStep {
    MtpTrainingWeightedDepthStep::new()
}

impl MtpAttention {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
        dtype: DType,
    ) -> Result<Self> {
        if head_dim != <H as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for MTP head_dim: runtime={} type-level={}",
                head_dim,
                <H as Unsigned>::USIZE
            );
        }

        // Typed loading — shape validation happens at load time
        let wq = gg.typed_qmatmul::<Shape2<AH2, S>>(&format!("{}.attn_q.weight", prefix))?;
        let wk = gg.typed_qmatmul::<Shape2<KH, S>>(&format!("{}.attn_k.weight", prefix))?;
        let wv = gg.typed_qmatmul::<Shape2<KH, S>>(&format!("{}.attn_v.weight", prefix))?;
        let wo = gg.typed_qmatmul::<Shape2<S, AH>>(&format!("{}.attn_o.weight", prefix))?;

        log_typed_qmatmul_shape(&format!("{}.wq", prefix), &wq);
        log_typed_qmatmul_shape(&format!("{}.wk", prefix), &wk);
        log_typed_qmatmul_shape(&format!("{}.wv", prefix), &wv);
        log_typed_qmatmul_shape(&format!("{}.wo", prefix), &wo);

        // Derive runtime values from type constants
        use paramecia_tensor::glowstick::num::Unsigned;
        let num_heads = <A as Unsigned>::USIZE;
        let num_kv_heads = <K as Unsigned>::USIZE;
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            paramecia_core::bail!(
                "invalid MTP attention head configuration: num_heads={}, num_kv_heads={}",
                num_heads,
                num_kv_heads
            );
        }
        let num_kv_groups = num_heads / num_kv_heads;
        let o_num_heads = <A as Unsigned>::USIZE;

        // Detect gated attention from Q output dim
        let q_out_dim = wq.out_dim().unwrap_or(num_heads * head_dim);
        let expected_q_dim = num_heads * head_dim;
        let is_gated = q_out_dim == expected_q_dim * 2;

        let q_norm = gg.shared_rms_norm(
            &format!("{}.attn_q_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;
        let k_norm = gg.shared_rms_norm(
            &format!("{}.attn_k_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;
        let qkv_prep = QkvPrep::new(wq.clone(), wk.clone(), wv.clone(), q_norm, k_norm);

        Ok(Self {
            wo: wo.clone(),
            num_heads,
            num_kv_heads,
            num_kv_groups,
            o_num_heads,
            head_dim,
            rotary_emb,
            kv_cache: None,
            gated_attention: is_gated,
            qkv_prep,
            post_attn_gate: post_attn_gate_flow(),
            output_project: output_project_flow(wo.clone()),
        })
    }

    pub(super) fn forward(&mut self, x: &THidden, offset: usize) -> Result<THidden> {
        let x = x.inner();
        let (b, l, _) = x.dims3()?;
        log_shape("mtp_attn.input", x);

        let x_typed: THidden = x.contiguous()?.try_into()?;

        // ── Q/K/V projection + prep ─────────────────────────────────────────
        debug_assert!(
            self.gated_attention,
            "wq typed as AH2 implies gated attention"
        );

        let ((q, gate), k, v) = self.qkv_prep.traced_forward(&mut (), x_typed)?;
        log_shape("mtp_attn.q_proj", q.inner());

        // Apply RoPE (uses n_rot for partial rotation) on typed head tensors
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let q = q.into_inner();

        let k_step: TKVCache = k.into_inner().try_into()?;
        let v_step: TKVCache = v.into_inner().try_into()?;

        // KV cache handling (simple typed cat-based for MTP's short sequences)
        let (k, v): (TKVCache, TKVCache) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k_inputs = [prev_k, &k_step];
                let v_inputs = [prev_v, &v_step];
                let k: TKVCache = paramecia_tensor::cat!(k_inputs.as_slice(), U2 => Lk)?;
                let v: TKVCache = paramecia_tensor::cat!(v_inputs.as_slice(), U2 => Lk)?;
                let k: TKVCache = paramecia_tensor::contiguous!(k)?;
                let v: TKVCache = paramecia_tensor::contiguous!(v)?;
                (k, v)
            }
            None => (
                paramecia_tensor::contiguous!(k_step)?,
                paramecia_tensor::contiguous!(v_step)?,
            ),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let k = k.into_inner();
        let v = v.into_inner();

        // Attention scale with YARN if enabled
        let base_scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        let yarn_scale = self.rotary_emb.attention_scale();
        let attn_scale = base_scale * yarn_scale;

        // Compute attention
        let attn_output = {
            // Repeat KV heads for grouped-query attention
            let k = repeat_kv(k, self.num_kv_groups)?;
            let v = repeat_kv(v, self.num_kv_groups)?;

            let scale = attn_scale as f64;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn_scores = (q.contiguous()?.matmul(&k_t)? * scale)?;
            let attn_weights = paramecia_nn::ops::softmax_last_dim(&attn_scores)?;
            attn_weights.matmul(&v.contiguous()?)?
        };

        // Attention output is [batch, heads, seq, head_dim]
        // Need to reduce to o_num_heads if different from num_heads
        let attn_output = if self.num_heads != self.o_num_heads {
            // Reduce heads: average groups of (num_heads / o_num_heads) heads
            let head_group_size = self.num_heads / self.o_num_heads;
            let grouped =
                attn_output.reshape((b, self.o_num_heads, head_group_size, l, self.head_dim))?;
            let reduced = grouped.mean(2)?;
            reduced
                .transpose(1, 2)?
                .reshape((b, l, self.o_num_heads * self.head_dim))?
        } else {
            attn_output
                .transpose(1, 2)?
                .reshape((b, l, self.num_heads * self.head_dim))?
        };

        let attn_output: TAttnHidden = attn_output.try_into()?;

        // Apply gating (structurally guaranteed by AH2 Q projection type)
        let final_output = self
            .post_attn_gate
            .traced_forward(&mut (), (attn_output, gate))?;

        // Typed output projection: [B, N, AH] → [B, N, S]
        let result = self.output_project.traced_forward(&mut (), final_output)?;
        log_shape("mtp_attn.wo_output", result.inner());
        Ok(result)
    }

    pub(super) fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

impl MtpSharedExpert {
    fn normalize_shared_gate(gate: Tensor) -> Result<TSharedGate> {
        match gate.dims() {
            [d] if *d == <S as Unsigned>::USIZE => {
                gate.unsqueeze(0)?.try_into().map_err(Into::into)
            }
            [1, d] if *d == <S as Unsigned>::USIZE => gate.try_into().map_err(Into::into),
            dims => paramecia_core::bail!(
                "mtp shared expert gate expected [S] or [1,S] with S={}, got {:?}",
                <S as Unsigned>::USIZE,
                dims
            ),
        }
    }

    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        dtype: DType,
    ) -> Result<Option<Self>> {
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
        let gate_name = format!("{}.shared_expert_gate.weight", prefix);
        let shared_gate_qt = gg
            .shared_tensor_typed::<Shape2<U1, S>>(&gate_name)
            .or_else(|_| {
                gg.shared_tensor_typed::<Shape2<U1, S>>(&format!("{}.shared_expert_gate", prefix))
            })?;
        let shared_gate_raw = shared_gate_qt.dequant_to(dtype, &gg.device)?.into_inner();
        let shared_gate = Self::normalize_shared_gate(shared_gate_raw)?;

        log_typed_qmatmul_shape(&format!("{}.gate_proj", prefix), &gate_proj);
        log_typed_qmatmul_shape(&format!("{}.up_proj", prefix), &up_proj);
        log_typed_qmatmul_shape(&format!("{}.down_proj", prefix), &down_proj);

        Ok(Some(Self {
            flow: MtpSharedExpertForward::new(
                shared_gate,
                Some(shared_gate_qt),
                MtpSharedFfn::new(gate_proj, up_proj, down_proj),
            ),
        }))
    }
}

impl MtpMoeBlock {
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_experts: usize,
        num_experts_per_tok: usize,
        dtype: DType,
    ) -> Result<Self> {
        let gate_exps = gg.shared_expert_tensor(&format!("{}.ffn_gate_exps.weight", prefix))?;
        let up_exps = gg.shared_expert_tensor(&format!("{}.ffn_up_exps.weight", prefix))?;
        let down_exps = gg.shared_expert_tensor(&format!("{}.ffn_down_exps.weight", prefix))?;
        let gate_exps = ExpertWeightTensor::new(gate_exps, "mtp_gate_exps")?;
        let up_exps = ExpertWeightTensor::new(up_exps, "mtp_up_exps")?;
        let down_exps = ExpertWeightTensor::new(down_exps, "mtp_down_exps")?;

        // Infer num_experts from gate tensor shape (first dimension)
        let inferred_num_experts = gate_exps
            .read()
            .map_err(|_| {
                paramecia_core::Error::Msg("failed to lock mtp gate experts tensor".to_string())
            })?
            .shape()
            .dims()[0];
        if num_experts != inferred_num_experts {
            paramecia_core::bail!(
                "MTP MoE num_experts mismatch: config={} weights={}",
                num_experts,
                inferred_num_experts
            );
        }
        if num_experts != <E as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for MTP num_experts: runtime={} type-level={}",
                num_experts,
                <E as Unsigned>::USIZE
            );
        }
        let gate = gg.typed_qmatmul::<Shape2<E, S>>(&format!("{}.shared_gate.weight", prefix))?;

        log_typed_qmatmul_shape(&format!("{}.moe_gate", prefix), &gate);

        let shared_expert = MtpSharedExpert::new(gg, prefix, dtype)?;

        let route_dispatch = MtpRouteDispatch::new(
            gate,
            gate_exps,
            up_exps,
            down_exps,
            num_experts,
            num_experts_per_tok,
        );

        Ok(Self {
            flow: MtpMoeBlockForwardFlow::new(route_dispatch, shared_expert),
        })
    }

    pub(super) fn forward(&mut self, hidden_states: &THidden) -> Result<THidden> {
        let hs_typed: THidden = paramecia_tensor::contiguous!(hidden_states.clone())?;
        self.flow.traced_forward(&mut (), hs_typed)
    }
}

/// Wrapper op: applies RmsNorm to THidden.
struct RmsNormHiddenOp<'a>(&'a RmsNorm);
#[primitive(property = Arrow)]
impl<'a> Combinator for RmsNormHiddenOp<'a> {
    type In = THidden;
    type Out = std::result::Result<THidden, CE>;
    fn forward(&mut self, _ctx: &mut (), input: THidden) -> Self::Out {
        Ok(self.0.forward(input.inner())?.try_into()?)
    }
}
#[primitive(property = Visualize)]
impl<'a> Vis for RmsNormHiddenOp<'a> {
    fn visualize() -> Graph {
        Graph::custom_leaf("RmsNormHidden").with_output_type::<std::result::Result<THidden, CE>>()
    }
}

/// Wrapper op: runs MtpAttention forward at a given offset.
struct MtpAttnForwardOp<'a>(&'a mut MtpAttention, usize);
#[primitive(property = Arrow)]
impl<'a> Combinator for MtpAttnForwardOp<'a> {
    type In = THidden;
    type Out = std::result::Result<THidden, CE>;
    fn forward(&mut self, _ctx: &mut (), input: THidden) -> Self::Out {
        self.0.forward(&input, self.1)
    }
}
#[primitive(property = Visualize)]
impl<'a> Vis for MtpAttnForwardOp<'a> {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "MtpAttnForward",
            <MtpAttentionForwardGraph as Vis>::visualize(),
        )
        .with_output_type::<std::result::Result<THidden, CE>>()
    }
}

/// Wrapper op: runs MtpMoeBlock forward.
struct MtpMoeForwardOp<'a>(&'a mut MtpMoeBlock);
#[primitive(property = Arrow)]
impl<'a> Combinator for MtpMoeForwardOp<'a> {
    type In = THidden;
    type Out = std::result::Result<THidden, CE>;
    fn forward(&mut self, _ctx: &mut (), input: THidden) -> Self::Out {
        self.0.forward(&input)
    }
}
#[primitive(property = Visualize)]
impl<'a> Vis for MtpMoeForwardOp<'a> {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph("MtpMoeForward", <MtpMoeForwardGraph as Vis>::visualize())
            .with_output_type::<std::result::Result<THidden, CE>>()
    }
}

impl MtpBlock {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        num_experts: usize,
        num_experts_per_tok: usize,
        prefix: &str,
        dtype: DType,
    ) -> Result<Self> {
        let attn = MtpAttention::new(gg, head_dim, rms_norm_eps, rotary_emb, prefix, dtype)?;

        let moe_block = MtpMoeBlock::new(gg, prefix, num_experts, num_experts_per_tok, dtype)?;

        let in_norm =
            gg.shared_rms_norm(&format!("{}.in_norm.weight", prefix), rms_norm_eps, dtype)?;
        let post_attn_norm = gg.shared_rms_norm(
            &format!("{}.post_attn_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            attn,
            moe_block,
            in_norm,
            post_attn_norm,
        })
    }

    pub(super) fn forward(&mut self, x: &THidden, offset: usize) -> Result<THidden> {
        log_shape("mtp_block.input", x.inner());

        // Phase 1: x → fanout → (in_norm → attn, identity) → residual_add
        let (x_norm, x_res) = Fanout::default().traced_forward(&mut (), x.clone());
        let h_attn = RmsNormHiddenOp(&self.in_norm).traced_forward(&mut (), x_norm)?;
        let h_attn = MtpAttnForwardOp(&mut self.attn, offset).traced_forward(&mut (), h_attn)?;
        log_shape("mtp_block.post_attn", h_attn.inner());
        let h: THidden = ResidualAddHiddenFlow::new(ResidualAddOp::default())
            .traced_forward(&mut (), (h_attn, x_res))?;

        // Phase 2: h → fanout → (post_attn_norm → moe, identity) → residual_add
        let (h_norm, h_res) = Fanout::default().traced_forward(&mut (), h);
        let ffn_input = RmsNormHiddenOp(&self.post_attn_norm).traced_forward(&mut (), h_norm)?;
        let ffn_out = MtpMoeForwardOp(&mut self.moe_block).traced_forward(&mut (), ffn_input)?;
        log_shape("mtp_block.post_moe", ffn_out.inner());
        let out: THidden = ResidualAddHiddenFlow::new(ResidualAddOp::default())
            .traced_forward(&mut (), (ffn_out, h_res))?;

        Ok(out)
    }

    pub(super) fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache();
    }
}

impl MtpHead {
    /// Load MTP head from GGUF if MTP tensors are present.
    ///
    /// Returns None if the model doesn't have MTP tensors.
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        num_experts: usize,
        num_experts_per_tok: usize,
        dtype: DType,
    ) -> Result<Option<Self>> {
        // Check if MTP tensors exist by trying to load the fc projection
        let fc = match gg.try_typed_qmatmul::<Shape2<S, S2>>("mtp.fc.weight")? {
            Some(fc) => fc,
            None => return Ok(None), // No MTP in this model
        };

        log_typed_qmatmul_shape("mtp.fc", &fc);

        // Determine MTP hidden size from fc weight shape
        let hidden_size = fc.out_dim().unwrap_or(2048);

        // Load projection layer tensors
        let pre_fc_norm_hidden =
            gg.rms_norm("mtp.pre_fc_norm_hidden.weight", rms_norm_eps, dtype)?;
        let pre_fc_norm_embedding =
            gg.rms_norm("mtp.pre_fc_norm_embedding.weight", rms_norm_eps, dtype)?;
        let output_norm = gg.rms_norm("mtp.norm.weight", rms_norm_eps, dtype)?;

        let projection = MtpProjection {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            output_norm,
        };

        // Load MTP blocks (may be multiple, check for blk.0.mtp, blk.1.mtp, etc.)
        let mut blocks = Vec::new();
        let mut block_idx = 0;
        loop {
            let prefix = format!("blk.{}.mtp", block_idx);
            // Check if this block exists
            if gg
                .try_typed_qmatmul::<Shape2<AH2, S>>(&format!("{}.attn_q.weight", prefix))?
                .is_none()
            {
                break;
            }

            let block = MtpBlock::new(
                gg,
                head_dim,
                rms_norm_eps,
                rotary_emb.clone(),
                num_experts,
                num_experts_per_tok,
                &prefix,
                dtype,
            )?;
            blocks.push(block);
            block_idx += 1;
        }

        if blocks.is_empty() {
            // Try loading from single "mtp" prefix for models with one block
            let prefix = "mtp";
            if gg
                .try_typed_qmatmul::<Shape2<AH2, S>>(&format!("{}.attn_q.weight", prefix))?
                .is_some()
            {
                let block = MtpBlock::new(
                    gg,
                    head_dim,
                    rms_norm_eps,
                    rotary_emb.clone(),
                    num_experts,
                    num_experts_per_tok,
                    prefix,
                    dtype,
                )?;
                blocks.push(block);
            }
        }

        tracing::info!(
            "Loaded MTP head with {} blocks, hidden_size={}",
            blocks.len(),
            hidden_size
        );

        Ok(Some(Self {
            projection,
            blocks,
            hidden_size,
        }))
    }

    /// Forward the core lm_head QMatMul with F16 dispatch.
    fn lm_head_forward(lm_head: &TLmHead, x: &THidden) -> Result<Tensor> {
        lm_head.forward_untyped(x.inner())
    }

    /// BATCHED forward pass for multi-depth MTP speculation.
    ///
    /// Processes speculation depths sequentially but each depth is a single
    /// efficient forward pass. For tree-parallel speculation, this could be
    /// extended to process all depths in a single batched operation.
    ///
    /// # Arguments
    /// * `hidden_state` - [batch, 1, hidden_size] from main model's final layer
    /// * `initial_token` - [batch] first predicted token (x_{t+1})
    /// * `embed_tokens` - Token embedding layer (shared with main model)
    /// * `lm_head` - Output projection (shared with main model)
    /// * `base_offset` - Position offset for RoPE (main model's current position)
    /// * `num_depths` - Number of speculative tokens to generate
    ///
    /// # Returns
    /// * `(spec_tokens, spec_logits)` - Vectors of predicted tokens and their logits
    pub fn forward_batched(
        &mut self,
        hidden_state: &THidden,
        initial_token: &Tensor,
        embed_tokens: &Embedding,
        lm_head: &TLmHead,
        base_offset: usize,
        num_depths: usize,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        // CLEAR MTP KV cache (fresh start from h_t each speculation round)
        self.clear_caches()?;

        // MTP head lives on the last layer's device (may differ from primary in multi-GPU).
        // embed_tokens and lm_head are on primary. We transfer embeddings to MTP device
        // for norm/projection/blocks, then transfer back to primary for lm_head.
        let mtp_device = hidden_state.inner().device().clone();
        // primary_device inferred from initial_token (always on primary)
        let primary_device = initial_token.device().clone();

        let init_state = MtpBatchedDepthState {
            current_hidden: hidden_state.clone(),
            current_token: initial_token.clone(),
            spec_tokens: Vec::with_capacity(num_depths),
            spec_logits: Vec::with_capacity(num_depths),
        };
        let mut step_ctx = MtpBatchedDepthStepCtx {
            projection: &mut self.projection,
            blocks: &mut self.blocks,
            embed_tokens,
            lm_head,
            mtp_device,
            primary_device,
            base_offset,
        };
        let step = mtp_batched_depth_step_op();
        let mut fold =
            TryFoldRange::<MtpBatchedDepthStep, MtpBatchedDepthState, paramecia_core::Error>::new(
                step,
            );
        let state = fold.traced_forward(&mut step_ctx, (init_state, 0..num_depths))?;
        Ok((state.spec_tokens, state.spec_logits))
    }

    /// Clear MTP KV caches (required after each verification round).
    pub fn clear_caches(&mut self) -> Result<()> {
        clear_mtp_block_caches(&mut self.blocks)
    }

    /// Batched forward pass for training that processes all sequence positions in parallel.
    ///
    /// Unlike inference which processes autoregressively, this method computes
    /// MTP predictions for all positions at once, suitable for training.
    ///
    /// # Arguments
    /// * `hidden_states` - Pre-norm hidden states from main model [batch, seq_len, hidden]
    /// * `token_embeds` - Token embeddings [batch, seq_len, hidden]
    /// * `lm_head` - Shared LM head for final projection
    /// * `base_offset` - Position offset for RoPE
    /// * `num_depths` - Number of MTP depths to compute
    ///
    /// # Returns
    /// Vector of logits for each MTP depth, each [batch, valid_len, vocab]
    /// where valid_len = seq_len - depth - 1 due to label alignment.
    pub fn forward_training_batch(
        &mut self,
        hidden_states: &THidden,
        token_embeds: &THidden,
        lm_head: &TLmHead,
        base_offset: usize,
        num_depths: usize,
    ) -> Result<Vec<Tensor>> {
        let (_, seq_len, _) = hidden_states.inner().dims3()?;

        // Clear MTP KV caches for fresh training pass
        self.clear_caches()?;

        let depth_limit = num_depths.min(seq_len.saturating_sub(1));
        let init_state = MtpTrainingDepthState {
            logits: Vec::with_capacity(depth_limit),
        };
        let mut step_ctx = MtpTrainingDepthStepCtx {
            projection: &mut self.projection,
            blocks: &mut self.blocks,
            hidden_states,
            token_embeds,
            lm_head,
            base_offset,
            seq_len,
        };
        let step = mtp_training_depth_step_op();
        let mut fold = TryFoldRange::<
            MtpTrainingDepthStep,
            MtpTrainingDepthState,
            paramecia_core::Error,
        >::new(step);
        let state = fold.traced_forward(&mut step_ctx, (init_state, 0..depth_limit))?;
        Ok(state.logits)
    }

    /// Batched forward pass for training with pre-computed weighted embeddings.
    ///
    /// This variant uses weighted embeddings (expectation over teacher distribution)
    /// instead of single token embeddings, providing smoother gradient estimates
    /// for zeroth-order optimization.
    ///
    /// # Arguments
    /// * `hidden_states` - Pre-norm hidden states from main model [batch, seq_len, hidden]
    /// * `weighted_embeds_per_depth` - Weighted embeddings for each depth, each [valid_len, hidden]
    /// * `lm_head` - Shared LM head for final projection
    /// * `base_offset` - Position offset for RoPE
    /// * `num_depths` - Number of MTP depths to compute
    ///
    /// # Returns
    /// Vector of logits for each MTP depth, each [batch, valid_len, vocab]
    pub fn forward_training_batch_weighted(
        &mut self,
        hidden_states: &THidden,
        weighted_embeds_per_depth: &[THidden],
        lm_head: &TLmHead,
        base_offset: usize,
        num_depths: usize,
    ) -> Result<Vec<Tensor>> {
        let (_b, seq_len, _) = hidden_states.inner().dims3()?;

        // Clear MTP KV caches for fresh training pass
        self.clear_caches()?;

        let depth_limit = num_depths.min(weighted_embeds_per_depth.len());
        let init_state = MtpTrainingDepthState {
            logits: Vec::with_capacity(depth_limit),
        };
        let mut step_ctx = MtpTrainingWeightedDepthStepCtx {
            projection: &mut self.projection,
            blocks: &mut self.blocks,
            hidden_states,
            weighted_embeds_per_depth,
            lm_head,
            base_offset,
            seq_len,
        };
        let step = mtp_training_weighted_depth_step_op();
        let mut fold = TryFoldRange::<
            MtpTrainingWeightedDepthStep,
            MtpTrainingDepthState,
            paramecia_core::Error,
        >::new(step);
        let state = fold.traced_forward(&mut step_ctx, (init_state, 0..depth_limit))?;
        Ok(state.logits)
    }
}
