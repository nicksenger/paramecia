use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use glowstick::num::Unsigned;
use glowstick::Shape4;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
use paramecia_core::quantized::GgmlDType;
use paramecia_core::quantized::QTensor;
use paramecia_core::{DType, Error, Tensor};
use paramecia_tensor::glowstick::num::{U0, U1, U2, U3};
use paramecia_tensor::glowstick::{Shape2, Shape3};
use paramecia_tensor::Error as TensorError;
use paramecia_tensor::Tensor as TTensor;

use inception::{primitive, Inception};
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, CombinatorTraceExt, Fanout, Fanout3, LiftResult, MapErr, Then, WrapOk, Zip, Zip3, ZipOk,
    ZipOk3,
};

use paramecia_tensor::flatten::FlattenOp;
use paramecia_tensor::narrow::NarrowOp;
use paramecia_tensor::qmatmul_op::QMatMulOp;
use paramecia_tensor::unflatten_last::UnflattenLastOp;
use paramecia_tensor::{broadcast_mul::BroadcastMulOp, cast_like::CastLikeOp, sigmoid::SigmoidOp};
use paramecia_tensor::{contiguous::ContiguousOp, rms_norm::RmsNormOp, transpose::TransposeOp};
use std::io::{Read, Seek};
use std::sync::Arc;
#[cfg(any(feature = "cuda", feature = "vulkan"))]
use tracing::warn;

use super::config::KvCacheQuantization;
use super::gguf_loader::Gguf;
use super::kv_cache::{
    KvCacheStorage, PreallocatedKvCache, PreallocatedQuantizedKvCache, PrefixCacheEntry,
};
use super::rope::RotaryEmbedding;
use super::shape::{AttnScores4, Lk, Lq, A, AH, AH2, B, H, H2, K, KH, N, S};
use super::utils::{log_shape, log_typed_qmatmul_shape};

type TQMatMul<S> = paramecia_tensor::QMatMul<S>;
type THidden = TTensor<Shape3<B, N, S>>;
type TAttnHidden = TTensor<Shape3<B, N, AH>>;
type TQProj = TTensor<Shape3<B, N, AH2>>;
type TKProj = TTensor<Shape3<B, N, KH>>;
type TVProj = TTensor<Shape3<B, N, KH>>;
type TQSplitAll = TTensor<Shape4<B, N, A, H2>>;
type TQSplit = TTensor<Shape4<B, N, A, H>>;
type TKV4 = TTensor<Shape4<B, N, K, H>>;
type TQHeads = TTensor<Shape4<B, A, N, H>>;
type TKHeads = TTensor<Shape4<B, K, N, H>>;
type TVHeads = TTensor<Shape4<B, K, N, H>>;
type TAttnMask = TTensor<Shape4<B, U1, Lq, Lk>>;

#[allow(dead_code)]
pub(super) struct FullAttentionForwardGraph;
#[primitive(property = Visualize)]
impl Vis for FullAttentionForwardGraph {
    fn visualize() -> Graph {
        let g = Graph::sequence(
            <QkvPrep as Vis>::visualize(),
            <PostAttnGateFlow as Vis>::visualize(),
        );
        let g = Graph::sequence(g, <OutputProjectFlow as Vis>::visualize());
        Graph::wrap_custom_subgraph("FullAttentionForwardGraph", g)
    }
}

type KvReshapeOpFlow =
    MapErr<UnflattenLastOp<Shape3<B, N, KH>, Shape4<B, N, K, H>>, TKProj, TKV4, TensorError, Error>;

type QSplitPrepFlow =
    Then<Zip<QNormTransposeOp, WrapOk<TQSplit, Error>>, ZipOk<TQHeads, TQSplit, Error>>;
fn q_split_prep_flow(q_norm: RmsNorm) -> QSplitPrepFlow {
    Then::new(
        Zip::new(q_norm_transpose_op(q_norm), WrapOk::default()),
        ZipOk::default(),
    )
}

type QPrepFlow = Then<
    SplitGatedQFlow,
    LiftResult<QSplitPrepFlow, std::result::Result<(TQSplit, TQSplit), Error>, (TQHeads, TQSplit)>,
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
    Error,
>;
type KProjFlow = MapErr<
    QMatMulOp<Shape2<KH, S>, Shape3<B, N, S>, Shape3<B, N, KH>>,
    THidden,
    TKProj,
    TensorError,
    Error,
>;
type VProjFlow = MapErr<
    QMatMulOp<Shape2<KH, S>, Shape3<B, N, S>, Shape3<B, N, KH>>,
    THidden,
    TVProj,
    TensorError,
    Error,
>;
type KPrepFromProjFlow =
    Then<KvReshapeOpFlow, LiftResult<KNormTransposeOp, std::result::Result<TKV4, Error>, TKHeads>>;
type VPrepFromProjFlow =
    Then<KvReshapeOpFlow, LiftResult<VTransposeOp, std::result::Result<TKV4, Error>, TVHeads>>;
type QkvQBranchFlow =
    Then<QProjFlow, LiftResult<QPrepFlow, std::result::Result<TQProj, Error>, (TQHeads, TQSplit)>>;
type QkvKBranchFlow =
    Then<KProjFlow, LiftResult<KPrepFromProjFlow, std::result::Result<TKProj, Error>, TKHeads>>;
type QkvVBranchFlow =
    Then<VProjFlow, LiftResult<VPrepFromProjFlow, std::result::Result<TVProj, Error>, TVHeads>>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct QkvPrep(
    Fanout3<THidden>,
    Zip3<QkvQBranchFlow, QkvKBranchFlow, QkvVBranchFlow>,
    ZipOk3<(TQHeads, TQSplit), TKHeads, TVHeads, Error>,
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

type GateFlattenFlow =
    MapErr<FlattenOp<Shape4<B, N, A, H>, U2, U3>, TQSplit, TAttnHidden, TensorError, Error>;

type GateSigmoidFlow = Then<
    Zip<
        WrapOk<TAttnHidden, Error>,
        MapErr<SigmoidOp<Shape3<B, N, AH>>, TAttnHidden, TAttnHidden, TensorError, Error>,
    >,
    ZipOk<TAttnHidden, TAttnHidden, Error>,
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
    Error,
>;
type GateMulFlow = MapErr<
    BroadcastMulOp<Shape3<B, N, AH>, Shape3<B, N, AH>>,
    (TAttnHidden, TAttnHidden),
    TAttnHidden,
    TensorError,
    Error,
>;
type GateApplyFlow = Then<
    GateCastLikeFlow,
    Then<
        LiftResult<
            GateSigmoidFlow,
            std::result::Result<(TAttnHidden, TAttnHidden), Error>,
            (TAttnHidden, TAttnHidden),
        >,
        LiftResult<
            GateMulFlow,
            std::result::Result<(TAttnHidden, TAttnHidden), Error>,
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

type PostAttnGatePrepareFlow =
    Then<Zip<WrapOk<TAttnHidden, Error>, GateFlattenFlow>, ZipOk<TAttnHidden, TAttnHidden, Error>>;
type PostAttnGateFlow = Then<
    PostAttnGatePrepareFlow,
    LiftResult<GateApplyFlow, std::result::Result<(TAttnHidden, TAttnHidden), Error>, TAttnHidden>,
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

type QReshapeSplitFlow = MapErr<
    UnflattenLastOp<Shape3<B, N, AH2>, Shape4<B, N, A, H2>>,
    TQProj,
    TQSplitAll,
    TensorError,
    Error,
>;

type QTakeQFlow =
    MapErr<NarrowOp<Shape4<B, N, A, H2>, U3, U0, H>, TQSplitAll, TQSplit, TensorError, Error>;
type QTakeGateFlow =
    MapErr<NarrowOp<Shape4<B, N, A, H2>, U3, H, H>, TQSplitAll, TQSplit, TensorError, Error>;

type QSplitGateBranchesFlow =
    Then<Fanout<TQSplitAll>, Then<Zip<QTakeQFlow, QTakeGateFlow>, ZipOk<TQSplit, TQSplit, Error>>>;
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
    LiftResult<QSplitGateBranchesFlow, std::result::Result<TQSplitAll, Error>, (TQSplit, TQSplit)>,
>;
fn split_gated_q_flow() -> SplitGatedQFlow {
    Then::new(
        MapErr::new(UnflattenLastOp::new(A::USIZE, H2::USIZE)),
        LiftResult::new(q_split_gate_branches_flow()),
    )
}

type QRmsNormFlow = MapErr<RmsNormOp<Shape4<B, N, A, H>>, TQSplit, TQSplit, TensorError, Error>;
type KRmsNormFlow = MapErr<RmsNormOp<Shape4<B, N, K, H>>, TKV4, TKV4, TensorError, Error>;
type QTransposeFlow =
    MapErr<TransposeOp<Shape4<B, N, A, H>, U1, U2>, TQSplit, TQHeads, TensorError, Error>;
type KTransposeFlow =
    MapErr<TransposeOp<Shape4<B, N, K, H>, U1, U2>, TKV4, TKHeads, TensorError, Error>;
type VTransposeFlow =
    MapErr<TransposeOp<Shape4<B, N, K, H>, U1, U2>, TKV4, TVHeads, TensorError, Error>;
type QContiguousFlow =
    MapErr<ContiguousOp<Shape4<B, A, N, H>>, TQHeads, TQHeads, TensorError, Error>;
type KContiguousFlow =
    MapErr<ContiguousOp<Shape4<B, K, N, H>>, TKHeads, TKHeads, TensorError, Error>;
type VContiguousFlow =
    MapErr<ContiguousOp<Shape4<B, K, N, H>>, TVHeads, TVHeads, TensorError, Error>;
type QNormTransposeOp = Then<
    QRmsNormFlow,
    Then<
        LiftResult<QTransposeFlow, std::result::Result<TQSplit, Error>, TQHeads>,
        LiftResult<QContiguousFlow, std::result::Result<TQHeads, Error>, TQHeads>,
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
        LiftResult<KTransposeFlow, std::result::Result<TKV4, Error>, TKHeads>,
        LiftResult<KContiguousFlow, std::result::Result<TKHeads, Error>, TKHeads>,
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

type VTransposeOp =
    Then<VTransposeFlow, LiftResult<VContiguousFlow, std::result::Result<TVHeads, Error>, TVHeads>>;
fn v_transpose_op() -> VTransposeOp {
    Then::new(
        MapErr::new(TransposeOp::default()),
        LiftResult::new(MapErr::new(ContiguousOp::default())),
    )
}

// ============================================================================
// Full Attention (for non-recurrent layers)
// ============================================================================

pub(super) struct FullAttention {
    /// Q projection (may be joint Q+Gate or just Q): [8192, 2048]
    pub(super) wq: TQMatMul<Shape2<AH2, S>>,
    /// K projection: [512, 2048]
    pub(super) wk: TQMatMul<Shape2<KH, S>>,
    /// V projection: [512, 2048]
    pub(super) wv: TQMatMul<Shape2<KH, S>>,
    /// Output projection: [2048, 4096]
    pub(super) wo: TQMatMul<Shape2<S, AH>>,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    /// Number of query heads per KV head (for GQA).
    pub(super) num_kv_groups: usize,
    pub(super) head_dim: usize,
    /// Whether Q projection outputs Q+Gate (true) or just Q (false)
    pub(super) gated_attention: bool,
    pub(super) rotary_emb: Arc<RotaryEmbedding>,
    /// Pre-allocated KV cache for O(1) token insertion (used when not quantizing)
    pub(super) preallocated_cache: Option<PreallocatedKvCache>,
    /// Pre-allocated quantized KV cache for O(1) token insertion with quantization
    pub(super) quantized_cache: Option<PreallocatedQuantizedKvCache>,
    /// Legacy KV cache storage (no longer used - kept for potential future use)
    #[allow(dead_code)]
    pub(super) kv_cache: Option<KvCacheStorage>,
    pub(super) kv_cache_quantization: KvCacheQuantization,
    /// Maximum sequence length for pre-allocated cache
    pub(super) max_seq_len: usize,
    /// Persistent Arrow stages for projection/reshape.
    qkv_prep: QkvPrep,
    post_attn_gate: PostAttnGateFlow,
    output_project: OutputProjectFlow,
    #[allow(dead_code)]
    pub(super) span: tracing::Span,
}

impl std::fmt::Debug for FullAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullAttention")
            .field("wq", &self.wq)
            .field("wk", &self.wk)
            .field("wv", &self.wv)
            .field("wo", &self.wo)
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("num_kv_groups", &self.num_kv_groups)
            .field("head_dim", &self.head_dim)
            .field("gated_attention", &self.gated_attention)
            .field("rotary_emb", &self.rotary_emb)
            .field("has_preallocated_cache", &self.preallocated_cache.is_some())
            .field("has_quantized_cache", &self.quantized_cache.is_some())
            .field("has_legacy_kv_cache", &self.kv_cache.is_some())
            .field("kv_cache_quantization", &self.kv_cache_quantization)
            .field("max_seq_len", &self.max_seq_len)
            .field("qkv_prep", &self.qkv_prep)
            .field("post_attn_gate", &"PostAttnGateFlow")
            .field("output_project", &"QMatMulOp")
            .finish()
    }
}

impl FullAttention {
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[inline]
    pub(super) fn q8_flash_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("PARAMECIA_DISABLE_Q8_FLASH").is_err())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
        dtype: DType,
        kv_cache_quantization: KvCacheQuantization,
        max_seq_len: usize,
    ) -> Result<Self, Error> {
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            paramecia_core::bail!(
                "invalid attention head configuration: num_heads={}, num_kv_heads={}",
                num_heads,
                num_kv_heads
            );
        }
        if num_heads != <A as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for attention heads: runtime={} type-level={}",
                num_heads,
                <A as Unsigned>::USIZE
            );
        }
        if num_kv_heads != <K as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for KV heads: runtime={} type-level={}",
                num_kv_heads,
                <K as Unsigned>::USIZE
            );
        }
        if head_dim != <H as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for head_dim: runtime={} type-level={}",
                head_dim,
                <H as Unsigned>::USIZE
            );
        }
        let num_kv_groups = num_heads / num_kv_heads;

        let wq = gg.typed_qmatmul::<Shape2<AH2, S>>(&format!("{}.attn_q.weight", prefix))?;
        let wk = gg.typed_qmatmul::<Shape2<KH, S>>(&format!("{}.attn_k.weight", prefix))?;
        let wv = gg.typed_qmatmul::<Shape2<KH, S>>(&format!("{}.attn_v.weight", prefix))?;
        let wo = gg.typed_qmatmul::<Shape2<S, AH>>(&format!("{}.attn_output.weight", prefix))?;
        log_typed_qmatmul_shape(&format!("{}.wq", prefix), &wq);
        log_typed_qmatmul_shape(&format!("{}.wk", prefix), &wk);
        log_typed_qmatmul_shape(&format!("{}.wv", prefix), &wv);
        log_typed_qmatmul_shape(&format!("{}.wo", prefix), &wo);

        // Detect if this is gated attention by checking Q projection output dimension
        // Gated: outputs num_heads * head_dim * 2 (Q + gate)
        // Non-gated: outputs num_heads * head_dim (just Q)
        let q_out_dim = wq.out_dim().unwrap_or(num_heads * head_dim);
        let expected_q_dim = num_heads * head_dim;
        let gated_attention = q_out_dim == expected_q_dim * 2;

        let q_norm = gg.rms_norm(
            &format!("{}.attn_q_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;
        let k_norm = gg.rms_norm(
            &format!("{}.attn_k_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;

        let span = tracing::span!(tracing::Level::TRACE, "full-attn");
        let qkv_prep = QkvPrep::new(wq.clone(), wk.clone(), wv.clone(), q_norm, k_norm);

        Ok(Self {
            wq,
            wk,
            wv,
            wo: wo.clone(),
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            gated_attention,
            rotary_emb,
            preallocated_cache: None,
            quantized_cache: None,
            kv_cache: None,
            kv_cache_quantization,
            max_seq_len,
            qkv_prep,
            post_attn_gate: post_attn_gate_flow(),
            output_project: output_project_flow(wo.clone()),
            span,
        })
    }

    fn normalize_attn_mask(attn_mask: Option<&Tensor>) -> Result<Option<TAttnMask>, Error> {
        match attn_mask {
            Some(mask) => Ok(Some(mask.clone().try_into()?)),
            None => Ok(None),
        }
    }

    fn apply_attn_mask(
        attn_scores: Tensor,
        attn_mask: Option<&TAttnMask>,
    ) -> Result<Tensor, Error> {
        let attn_scores_typed: TTensor<AttnScores4> = attn_scores.try_into()?;
        let masked = match attn_mask {
            None => attn_scores_typed.into_inner(),
            Some(mask) => paramecia_tensor::broadcast_add!(&attn_scores_typed, mask)?.into_inner(),
        };
        Ok(masked)
    }

    pub(super) fn forward_typed(
        &mut self,
        x: &THidden,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<THidden, Error> {
        let x = x.inner();
        let (b, l, _) = x.dims3()?;
        let typed_attn_mask = Self::normalize_attn_mask(attn_mask)?;

        // ── 1. Q/K/V projection + prep ─────────────────────────────────────
        // wq is typed as Shape2<AH2, S>, so gated attention is structurally guaranteed
        debug_assert!(
            self.gated_attention,
            "wq typed as AH2 implies gated attention"
        );

        let ((q, gate), k, v) = self
            .qkv_prep
            .traced_forward(&mut (), x.clone().try_into()?)?;

        log_shape("full_attn.q_split", q.inner());
        log_shape("full_attn.gate_split", gate.inner());

        log_shape("full_attn.q_normed", q.inner());
        log_shape("full_attn.k_normed", k.inner());
        log_shape("full_attn.v_reshaped", v.inner());
        log_shape("full_attn.q_transposed", q.inner());
        log_shape("full_attn.k_transposed", k.inner());

        // Apply RoPE on typed head tensors
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let q = q.into_inner();
        let k = k.into_inner();
        let v = v.into_inner();
        log_shape("full_attn.q_rope", &q);
        log_shape("full_attn.k_rope", &k);

        // KV-cache handling with optional quantization
        // For Q8_0/Q4K: use PreallocatedQuantizedKvCache for memory efficiency
        // For F16/BF16: use PreallocatedKvCache with appropriate dtype
        // KV-cache handling with optional quantization
        // For Q8_0/Q4K: use PreallocatedQuantizedKvCache for memory efficiency
        // For F16/BF16: use PreallocatedKvCache with appropriate dtype
        let (k, v, q8_storage) =
            if let Some(ggml_dtype) = self.kv_cache_quantization.to_ggml_dtype() {
                // Quantized KV cache mode (Q8_0 or Q4K)
                if self.quantized_cache.is_none() {
                    self.quantized_cache = Some(PreallocatedQuantizedKvCache::new(
                        b,
                        self.num_kv_heads,
                        self.max_seq_len,
                        self.head_dim,
                        ggml_dtype,
                        k.device(),
                    )?);
                }

                let cache = self.quantized_cache.as_mut().ok_or_else(|| {
                    paramecia_core::Error::Msg("missing quantized KV cache".to_string())
                })?;

                if offset != cache.seq_len {
                    if offset <= cache.seq_len {
                        cache.truncate(offset);
                    } else {
                        paramecia_core::bail!(
                            "KV cache offset mismatch: offset={} but cache has {} tokens",
                            offset,
                            cache.seq_len
                        );
                    }
                }

                cache.append_quantized(&k, &v)?;

                #[cfg(any(feature = "cuda", feature = "vulkan"))]
                let q8_storage = if ggml_dtype == GgmlDType::Q8_0 && Self::q8_flash_enabled() {
                    match cache.get_kv_storage() {
                        Ok((k_storage, v_storage, k_layout, v_layout)) => {
                            Some((k_storage, v_storage, k_layout, v_layout))
                        }
                        Err(e) => {
                            use std::sync::atomic::{AtomicBool, Ordering};
                            static LOGGED: AtomicBool = AtomicBool::new(false);
                            if !LOGGED.swap(true, Ordering::Relaxed) {
                                warn!(error = ?e, "get_kv_storage failed");
                            }
                            None
                        }
                    }
                } else {
                    None
                };
                #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
                let q8_storage: Option<(
                    paramecia_core::quantized::QStorage,
                    paramecia_core::quantized::QStorage,
                    paramecia_core::Layout,
                    paramecia_core::Layout,
                )> = None;

                if q8_storage.is_some() {
                    (k, v, q8_storage)
                } else {
                    let (cached_k, cached_v) = cache.get_kv()?;
                    let cached_k = cached_k.to_dtype(q.dtype())?.contiguous()?;
                    let cached_v = cached_v.to_dtype(q.dtype())?.contiguous()?;
                    (cached_k, cached_v, q8_storage)
                }
            } else {
                // Non-quantized mode (F16 or BF16)
                // Optionally convert K/V to the cache dtype before storing
                let cache_dtype = self.kv_cache_quantization.cache_dtype();
                let (k_for_cache, v_for_cache) = if let Some(dtype) = cache_dtype {
                    if k.dtype() != dtype {
                        (
                            k.to_dtype(dtype)?.contiguous()?,
                            v.to_dtype(dtype)?.contiguous()?,
                        )
                    } else {
                        (k, v)
                    }
                } else {
                    (k, v)
                };

                if self.preallocated_cache.is_none() {
                    let cache_dtype = k_for_cache.dtype();
                    self.preallocated_cache = Some(PreallocatedKvCache::new(
                        b,
                        self.num_kv_heads,
                        self.max_seq_len,
                        self.head_dim,
                        cache_dtype,
                        k_for_cache.device(),
                    )?);
                }

                let cache = self
                    .preallocated_cache
                    .as_mut()
                    .ok_or_else(|| paramecia_core::Error::Msg("missing KV cache".to_string()))?;

                if offset != cache.seq_len {
                    if offset <= cache.seq_len {
                        cache.truncate(offset);
                    } else {
                        paramecia_core::bail!(
                            "KV cache offset mismatch: offset={} but cache has {} tokens",
                            offset,
                            cache.seq_len
                        );
                    }
                }

                let (cached_k, cached_v) = cache.append(&k_for_cache, &v_for_cache)?;

                // Convert back to query dtype for attention if cache dtype differs
                let (cached_k, cached_v) = if cached_k.dtype() != q.dtype() {
                    (
                        cached_k.to_dtype(q.dtype())?.contiguous()?,
                        cached_v.to_dtype(q.dtype())?.contiguous()?,
                    )
                } else {
                    (cached_k, cached_v)
                };

                (cached_k, cached_v, None)
            };
        let k_typed: TTensor<Shape4<B, K, Lk, H>> = k.try_into()?;
        let v_typed: TTensor<Shape4<B, K, Lk, H>> = v.try_into()?;
        let k = k_typed.into_inner();
        let v = v_typed.into_inner();

        // Compute attention - use flash attention when available, otherwise standard SDPA
        #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
        let attn_output = if let Some((k_storage, v_storage, k_layout, v_layout)) = q8_storage {
            // Use Q8_0 flash attention kernel
            // q is currently [b, h, l, d], transpose to [b, l, h, d]
            let q_perm = q.transpose(1, 2)?.contiguous()?;
            let q_l = q_perm.layout();

            // Use causal masking with an explicit offset into the KV cache.
            // This keeps correctness for chunked prefill and single-token decode.
            let seq_k = self.cache_seq_len();
            let q_offset = seq_k.saturating_sub(l);
            let causal = true;

            // Attention scale: 1/sqrt(head_dim) * YARN_scale
            let base_scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
            let yarn_scale = self.rotary_emb.attention_scale();
            let attn_scale = base_scale * yarn_scale;

            crate::ops::flash_attn_q8(
                &q_perm,
                &k_storage,
                &v_storage,
                &q_l,
                &k_layout,
                &v_layout,
                attn_scale,
                b,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                l,
                seq_k,
                q_offset,
                causal,
            )?
            .reshape((b, l, self.num_heads * self.head_dim))?
        } else {
            // Attention scale: 1/sqrt(head_dim) * YARN_scale
            let base_scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
            let yarn_scale = self.rotary_emb.attention_scale();
            let attn_scale = base_scale * yarn_scale;

            // Repeat KV heads for grouped-query attention
            let k = repeat_kv(k, self.num_kv_groups)?;
            let v = repeat_kv(v, self.num_kv_groups)?;

            // Compute attention
            let scale = attn_scale as f64;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn_scores = (q.contiguous()?.matmul(&k_t)? * scale)?;
            let attn_scores = Self::apply_attn_mask(attn_scores, typed_attn_mask.as_ref())?;

            // Clamp attention scores to prevent softmax overflow during training
            let attn_scores_clamped = attn_scores.clamp(-100.0, 100.0)?;
            let attn_weights = paramecia_nn::ops::softmax_last_dim(&attn_scores_clamped)?;
            let attn_output = attn_weights.matmul(&v.contiguous()?)?;

            // Reshape attention output
            attn_output
                .transpose(1, 2)?
                .reshape((b, l, self.num_heads * self.head_dim))?
                .contiguous()?
        };
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
        let attn_output = {
            // Suppress unused variable warning when cuda/metal/vulkan is disabled
            let _ = &q8_storage;

            // Attention scale: 1/sqrt(head_dim) * YARN_scale
            let base_scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
            let yarn_scale = self.rotary_emb.attention_scale();
            let attn_scale = base_scale * yarn_scale;

            // Repeat KV heads for grouped-query attention
            let k = repeat_kv(k, self.num_kv_groups)?;
            let v = repeat_kv(v, self.num_kv_groups)?;

            // Compute attention
            let scale = attn_scale as f64;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn_scores = (q.contiguous()?.matmul(&k_t)? * scale)?;
            let attn_scores = Self::apply_attn_mask(attn_scores, typed_attn_mask.as_ref())?;

            // Clamp attention scores to prevent softmax overflow during training
            let attn_scores_clamped = attn_scores.clamp(-100.0, 100.0)?;
            let attn_weights = paramecia_nn::ops::softmax_last_dim(&attn_scores_clamped)?;
            let attn_output = attn_weights.matmul(&v.contiguous()?)?;

            // Reshape attention output
            attn_output
                .transpose(1, 2)?
                .reshape((b, l, self.num_heads * self.head_dim))?
                .contiguous()?
        };

        let attn_output: TAttnHidden = attn_output.try_into()?;
        log_shape("full_attn.attn_output", attn_output.inner());

        // Apply gating (structurally guaranteed by AH2 Q projection type)
        let final_output = self
            .post_attn_gate
            .traced_forward(&mut (), (attn_output, gate))?;

        log_shape("full_attn.gated_output", final_output.inner());

        // Typed output projection: [B, N, AH] → [B, N, S]
        let result = self.output_project.traced_forward(&mut (), final_output)?;
        log_shape("full_attn.wo_output", result.inner());
        Ok(result)
    }

    #[allow(dead_code)]
    pub(super) fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor, Error> {
        let x_typed: THidden = x.contiguous()?.try_into()?;
        Ok(self
            .forward_typed(&x_typed, attn_mask, offset)?
            .into_inner())
    }

    /// Legacy method for storing K/V to cache with optional quantization.
    ///
    /// **Deprecated:** This is superseded by `PreallocatedQuantizedKvCache` which is now
    /// used directly in the forward pass when `kv_cache_quantization` is Q8_0 or Q4K.
    /// Kept for reference and potential future use cases.
    #[allow(dead_code)]
    pub(super) fn store_kv_cache(&mut self, k: &Tensor, v: &Tensor, b: usize) -> Result<(), Error> {
        if let Some(ggml_dtype) = self.kv_cache_quantization.to_ggml_dtype() {
            let block_size = self.kv_cache_quantization.block_size();

            let (k_for_quant, v_for_quant) = if !self.head_dim.is_multiple_of(block_size) {
                let padded_head_dim = self.head_dim.div_ceil(block_size) * block_size;
                let pad_size = padded_head_dim - self.head_dim;

                let k_padded = {
                    let zeros = Tensor::zeros(
                        (b, self.num_kv_heads, k.dim(2)?, pad_size),
                        k.dtype(),
                        k.device(),
                    )?;
                    Tensor::cat(&[k, &zeros], 3)?
                };
                let v_padded = {
                    let zeros = Tensor::zeros(
                        (b, self.num_kv_heads, v.dim(2)?, pad_size),
                        v.dtype(),
                        v.device(),
                    )?;
                    Tensor::cat(&[v, &zeros], 3)?
                };
                (k_padded, v_padded)
            } else {
                (k.clone(), v.clone())
            };

            let k_flat = k_for_quant.flatten_to(k_for_quant.rank() - 1)?;
            let v_flat = v_for_quant.flatten_to(v_for_quant.rank() - 1)?;
            let k_flat = k_flat.to_dtype(DType::F32)?;
            let v_flat = v_flat.to_dtype(DType::F32)?;

            let k_qtensor = QTensor::quantize(&k_flat, ggml_dtype)?;
            let v_qtensor = QTensor::quantize(&v_flat, ggml_dtype)?;

            let k_padded_shape = k_for_quant.dims().to_vec();
            let v_padded_shape = v_for_quant.dims().to_vec();

            self.kv_cache = Some(KvCacheStorage::Quantized(
                k_qtensor,
                v_qtensor,
                k_padded_shape,
                v_padded_shape,
            ));
        } else {
            self.kv_cache = Some(KvCacheStorage::Float(
                k.clone().try_into()?,
                v.clone().try_into()?,
            ));
        }
        Ok(())
    }

    pub(super) fn clear_kv_cache(&mut self) {
        // Clear pre-allocated cache (reset head position to 0, keeps buffer allocated)
        if let Some(ref mut cache) = self.preallocated_cache {
            cache.clear();
        }
        // Clear pre-allocated quantized cache
        if let Some(ref mut cache) = self.quantized_cache {
            cache.clear();
        }
        // Clear legacy cache (shouldn't be used anymore)
        self.kv_cache = None;
    }

    /// Truncate the KV cache to a given sequence length.
    /// Used for speculative decoding when draft tokens are rejected.
    pub(super) fn truncate_kv_cache(&mut self, new_len: usize) {
        if let Some(ref mut cache) = self.preallocated_cache {
            cache.truncate(new_len);
        }
        if let Some(ref mut cache) = self.quantized_cache {
            cache.truncate(new_len);
        }
    }

    /// Set the maximum sequence length for the pre-allocated KV cache.
    ///
    /// This should be called before the first forward pass. If called after
    /// the cache is initialized, it will clear the existing cache.
    #[allow(dead_code)]
    pub(super) fn set_max_seq_len(&mut self, max_seq_len: usize) {
        if self.max_seq_len != max_seq_len {
            self.max_seq_len = max_seq_len;
            // Clear existing caches so they're re-created with new size
            self.preallocated_cache = None;
            self.quantized_cache = None;
        }
    }

    /// Save the current KV cache state for prefix caching.
    /// Returns a deep copy of the K/V tensors that can be restored later.
    pub(super) fn save_kv_state(&self) -> PrefixCacheEntry {
        // Check quantized cache first (for Q8_0/Q4K modes)
        if let Some(ref cache) = self.quantized_cache {
            if cache.seq_len > 0 {
                // Dequantize the cache for storage
                if let Ok((k, v)) = cache.get_kv() {
                    if let (Ok(k_cloned), Ok(v_cloned)) = (k.contiguous(), v.contiguous()) {
                        let k_cache = match k_cloned.try_into() {
                            Ok(t) => t,
                            Err(_) => return PrefixCacheEntry::Empty,
                        };
                        let v_cache = match v_cloned.try_into() {
                            Ok(t) => t,
                            Err(_) => return PrefixCacheEntry::Empty,
                        };
                        return PrefixCacheEntry::FullAttention {
                            k_cache,
                            v_cache,
                            seq_len: cache.seq_len,
                        };
                    }
                }
            }
        }
        // Check non-quantized cache (for F16/BF16 modes)
        if let Some(ref cache) = self.preallocated_cache {
            if cache.seq_len > 0 {
                // Extract the valid portion of the cache and clone it
                if let (Ok(k), Ok(v)) = (
                    cache.k_cache.inner().narrow(2, 0, cache.seq_len),
                    cache.v_cache.inner().narrow(2, 0, cache.seq_len),
                ) {
                    if let (Ok(k_cloned), Ok(v_cloned)) = (k.contiguous(), v.contiguous()) {
                        let k_cache = match k_cloned.try_into() {
                            Ok(t) => t,
                            Err(_) => return PrefixCacheEntry::Empty,
                        };
                        let v_cache = match v_cloned.try_into() {
                            Ok(t) => t,
                            Err(_) => return PrefixCacheEntry::Empty,
                        };
                        return PrefixCacheEntry::FullAttention {
                            k_cache,
                            v_cache,
                            seq_len: cache.seq_len,
                        };
                    }
                }
            }
        }
        PrefixCacheEntry::Empty
    }

    /// Restore KV cache state from a prefix cache entry.
    /// This allows resuming generation from a previously cached prefix.
    pub(super) fn restore_kv_state(&mut self, entry: &PrefixCacheEntry) -> Result<(), Error> {
        match entry {
            PrefixCacheEntry::FullAttention {
                k_cache,
                v_cache,
                seq_len,
            } => {
                let b = k_cache.inner().dim(0)?;
                let device = k_cache.inner().device();

                // Check if we're using quantized caching
                if let Some(ggml_dtype) = self.kv_cache_quantization.to_ggml_dtype() {
                    // Initialize quantized cache if needed
                    if self.quantized_cache.is_none() {
                        self.quantized_cache = Some(PreallocatedQuantizedKvCache::new(
                            b,
                            self.num_kv_heads,
                            self.max_seq_len,
                            self.head_dim,
                            ggml_dtype,
                            device,
                        )?);
                    }

                    // Clear and restore by appending the saved K/V
                    if let Some(ref mut cache) = self.quantized_cache {
                        cache.clear();
                        cache.append(k_cache.inner(), v_cache.inner())?;
                    }
                } else {
                    // Non-quantized mode: restore to preallocated cache
                    if self.preallocated_cache.is_none() {
                        let num_kv_heads = k_cache.inner().dim(1)?;
                        let head_dim = k_cache.inner().dim(3)?;
                        let dtype = k_cache.inner().dtype();

                        self.preallocated_cache = Some(PreallocatedKvCache::new(
                            b,
                            num_kv_heads,
                            self.max_seq_len,
                            head_dim,
                            dtype,
                            device,
                        )?);
                    }

                    if let Some(ref mut cache) = self.preallocated_cache {
                        // Copy the saved K/V into the pre-allocated buffer
                        cache.k_cache = cache
                            .k_cache
                            .inner()
                            .slice_scatter(k_cache.inner(), 2, 0)?
                            .try_into()?;
                        cache.v_cache = cache
                            .v_cache
                            .inner()
                            .slice_scatter(v_cache.inner(), 2, 0)?
                            .try_into()?;
                        cache.seq_len = *seq_len;
                    }
                }
                Ok(())
            }
            PrefixCacheEntry::Empty => {
                self.clear_kv_cache();
                Ok(())
            }
            PrefixCacheEntry::LinearAttention { .. } => {
                // Wrong entry type for full attention - clear cache
                self.clear_kv_cache();
                Ok(())
            }
        }
    }

    /// Get the current cache sequence length.
    pub(super) fn cache_seq_len(&self) -> usize {
        // Check quantized cache
        if let Some(ref cache) = self.quantized_cache {
            return cache.seq_len;
        }
        // Fall back to non-quantized cache
        self.preallocated_cache
            .as_ref()
            .map(|c| c.seq_len)
            .unwrap_or(0)
    }
}
