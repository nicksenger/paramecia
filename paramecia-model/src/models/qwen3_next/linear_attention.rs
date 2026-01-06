use crate::quantized_nn::RmsNorm;
use inception::Inception;
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, CombinatorTraceExt, LiftResult, MapErr, Zip, ZipOk,
};
use paramecia_core::{DType, Device, Result, Tensor};
use paramecia_tensor::broadcast_mul::BroadcastMulOp;
use paramecia_tensor::glowstick::num::Unsigned;
use paramecia_tensor::glowstick::num::U1;
use paramecia_tensor::glowstick::{Shape1, Shape2, Shape3, Shape4};
use paramecia_tensor::qmatmul_op::QMatMulOp;
use paramecia_tensor::rms_norm::RmsNormOp;
use paramecia_tensor::silu::SiluOp;
use paramecia_tensor::Error as TensorError;
use std::io::{Read, Seek};
use tracing::warn;

use super::gguf_loader::Gguf;
use super::kv_cache::{PrefixCacheEntry, RecurrentState};
use super::shape::{
    BaDim, Ch, ConvKernel, DInner, DState, DtRank, NGroups, QkvDim, QkvzDim, B, C, N, S,
};
use super::utils::{
    create_causal_mask, create_identity_mask, l2_normalize, log_shape, log_typed_qmatmul_shape,
    pad_tensor, softplus, solve_lower_triangular,
};

type TQMatMul<S> = paramecia_tensor::QMatMul<S>;
type THidden = paramecia_tensor::Tensor<Shape3<B, N, S>>;
type TQkvSeq = paramecia_tensor::Tensor<Shape3<B, QkvDim, N>>;
type TConvState = paramecia_tensor::Tensor<Shape3<B, QkvDim, Ch>>;
type TSsmState = paramecia_tensor::Tensor<Shape4<B, DtRank, DState, DState>>;
type TDeltaNet4 = paramecia_tensor::Tensor<Shape4<B, N, DtRank, DState>>;
type TDeltaGate3 = paramecia_tensor::Tensor<Shape3<B, N, DtRank>>;
type TSsmConv = paramecia_tensor::Tensor<Shape2<QkvDim, ConvKernel>>;
type TSsmHead = paramecia_tensor::Tensor<Shape1<DtRank>>;
type TSharedSsmHead = paramecia_tensor::SharedQTensor<Shape1<DtRank>>;
type TSharedSsmConv = paramecia_tensor::SharedQTensor<Shape2<QkvDim, ConvKernel>>;
type TCausalMask = paramecia_tensor::Tensor<Shape4<U1, U1, C, C>>;
type TDInner = paramecia_tensor::Tensor<Shape3<B, N, DInner>>;
type DeltaNet4Shape = Shape4<B, N, DtRank, DState>;

// ── Arrow-composed flows ──────────────────────────────────────────────────

// Output projection: [B, N, DInner] → [B, N, S]
type SsmOutProjectFlow = QMatMulOp<Shape2<S, DInner>, Shape3<B, N, DInner>, Shape3<B, N, S>>;
fn ssm_out_project_flow(wout: TQMatMul<Shape2<S, DInner>>) -> SsmOutProjectFlow {
    QMatMulOp::new(wout)
}

// Gated norm: (input, gate) → RmsNorm(input) * SiLU(gate)
type CoreError = paramecia_core::Error;
type DnRmsNormFlow =
    MapErr<RmsNormOp<DeltaNet4Shape>, TDeltaNet4, TDeltaNet4, TensorError, CoreError>;
type DnSiluFlow = MapErr<SiluOp<DeltaNet4Shape>, TDeltaNet4, TDeltaNet4, TensorError, CoreError>;
type DnBroadcastMulFlow = MapErr<
    BroadcastMulOp<DeltaNet4Shape, DeltaNet4Shape>,
    (TDeltaNet4, TDeltaNet4),
    TDeltaNet4,
    TensorError,
    CoreError,
>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct GatedNorm(
    Zip<DnRmsNormFlow, DnSiluFlow>,
    ZipOk<TDeltaNet4, TDeltaNet4, CoreError>,
    LiftResult<
        DnBroadcastMulFlow,
        std::result::Result<(TDeltaNet4, TDeltaNet4), CoreError>,
        TDeltaNet4,
    >,
);
impl GatedNorm {
    fn new(norm: &RmsNorm) -> Self {
        let rms = RmsNormOp::new_with_shared(
            norm.weight().clone(),
            norm.eps(),
            norm.shared_weight().cloned(),
            norm.zero_centered(),
        );
        Self(
            Zip::new(MapErr::new(rms), MapErr::new(SiluOp::default())),
            ZipOk::default(),
            LiftResult::new(MapErr::new(BroadcastMulOp::default())),
        )
    }
}
impl std::fmt::Debug for GatedNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedNorm").finish()
    }
}

// Visualization graph for linear attention forward
#[allow(dead_code)]
pub(super) struct LinearAttentionForwardGraph;
impl LinearAttentionForwardGraph {
    #[allow(dead_code)]
    pub(super) fn graph() -> Graph {
        let ba_proj = Graph::custom_leaf("BetaAlphaProject");
        let qkvz_parse = Graph::custom_leaf("QkvzParse");
        let beta_alpha_gate = Graph::custom_leaf("BetaAlphaGate");
        let conv_silu = Graph::custom_leaf("ConvSiLU");
        let post_conv_split = Graph::custom_leaf("PostConvSplit");
        let delta_net_dispatch = Graph::custom_leaf("DeltaNetDispatch");
        let gated_norm = <GatedNorm as Vis>::visualize();
        let out_proj = <SsmOutProjectFlow as Vis>::visualize();

        let g = Graph::sequence(ba_proj, qkvz_parse);
        let g = Graph::sequence(g, beta_alpha_gate);
        let g = Graph::sequence(g, conv_silu);
        let g = Graph::sequence(g, post_conv_split);
        let g = Graph::sequence(g, delta_net_dispatch);
        let g = Graph::sequence(g, gated_norm);
        let g = Graph::sequence(g, out_proj);
        Graph::wrap_custom_subgraph("LinearAttentionForward", g)
    }
}

// ============================================================================
// Linear Attention (Gated Delta Net for recurrent layers)
// ============================================================================

/// Chunk size for delta net processing (matches llama.cpp CHUNK_SIZE)
#[allow(dead_code)]
pub(super) const DELTA_NET_CHUNK_SIZE: usize = 64;

/// Cached masks for a given sequence length
pub(super) struct CachedMasks {
    pub(super) seq_len: usize,
    pub(super) causal_mask: TCausalMask, // Strictly lower triangular
    pub(super) causal_diag_mask: TCausalMask, // Lower triangular including diagonal
}

impl std::fmt::Debug for CachedMasks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedMasks")
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

/// Mode for the delta net kernel dispatch.
/// Controls which kernel is used and whether intermediate states are materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeltaNetMode {
    /// Standard inference: dispatch based on sequence length
    /// (autoregressive for l=1, fused batch for small l, chunked for large l)
    Normal,
    /// Parallel state materialization: stores intermediate states at each position
    /// for O(1) state slicing during speculative decoding verification
    WithStateMaterialization,
}

/// SSM input projection — fused QKVZ or split QKV + gate.
pub(super) enum SsmInProjection {
    /// Fused QKVZ: single weight projects to Q, K, V, Z interleaved per group
    Fused(TQMatMul<Shape2<QkvzDim, S>>),
    /// Split: separate QKV and gate (Z) projections (e.g. Qwen3-Coder-Next GGUFs)
    Split {
        qkv: TQMatMul<Shape2<QkvDim, S>>,
        gate: TQMatMul<Shape2<DInner, S>>,
    },
}

impl SsmInProjection {
    /// Forward pass returning an untyped tensor (output is immediately reshaped/narrowed).
    fn forward_untyped(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Fused(proj) => proj.forward_untyped(x),
            Self::Split { qkv, .. } => qkv.forward_untyped(x),
        }
    }

    /// Forward the gate projection (split variant only).
    fn forward_gate_untyped(&self, x: &Tensor) -> Option<Result<Tensor>> {
        match self {
            Self::Fused(_) => None,
            Self::Split { gate, .. } => Some(gate.forward_untyped(x)),
        }
    }

    fn is_split(&self) -> bool {
        matches!(self, Self::Split { .. })
    }

    /// Get qtensors for serialization. Returns (main_qtensor, main_name_suffix, gate_qtensor).
    pub(super) fn qtensors_for_save(
        &self,
    ) -> (
        Option<&paramecia_core::quantized::QTensor>,
        &'static str,
        Option<&paramecia_core::quantized::QTensor>,
    ) {
        match self {
            Self::Fused(proj) => (proj.qtensor(), "ssm_in", None),
            Self::Split { qkv, gate } => (qkv.qtensor(), "attn_qkv", gate.qtensor()),
        }
    }
}

impl std::fmt::Debug for SsmInProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fused(q) => f.debug_tuple("Fused").field(q).finish(),
            Self::Split { qkv, gate } => f
                .debug_struct("Split")
                .field("qkv", qkv)
                .field("gate", gate)
                .finish(),
        }
    }
}

/// Beta/alpha projection for linear attention.
pub(super) enum BetaAlphaProjection {
    /// Fused beta+alpha projection: [BaDim, S]
    Fused(TQMatMul<Shape2<BaDim, S>>),
    /// Split beta/alpha projections: [DtRank, S] each
    Split {
        beta: TQMatMul<Shape2<DtRank, S>>,
        alpha: TQMatMul<Shape2<DtRank, S>>,
    },
}

impl BetaAlphaProjection {
    /// Returns tensor refs for serialization as (name_suffix, qtensor).
    pub(super) fn qtensors_for_save(
        &self,
    ) -> (
        Option<(&'static str, &paramecia_core::quantized::QTensor)>,
        Option<(&'static str, &paramecia_core::quantized::QTensor)>,
    ) {
        match self {
            Self::Fused(fused) => (fused.qtensor().map(|qt| ("ssm_ba", qt)), None),
            Self::Split { beta, alpha } => (
                beta.qtensor().map(|qt| ("ssm_beta", qt)),
                alpha.qtensor().map(|qt| ("ssm_alpha", qt)),
            ),
        }
    }

    /// Compute beta/alpha gate tensors with shape [B, N, num_v_heads].
    fn forward_beta_alpha(
        &self,
        x: &Tensor,
        batch: usize,
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fused(fused) => {
                let mixed_ba = fused.forward_untyped(x)?;
                log_shape("linear_attn.mixed_ba", &mixed_ba);

                let ba_new_dim = 2 * num_v_heads / num_k_heads;
                let beta_size = num_v_heads / num_k_heads;
                let mixed_ba = mixed_ba.reshape((batch, seq_len, num_k_heads, ba_new_dim))?;
                let beta =
                    mixed_ba
                        .narrow(3, 0, beta_size)?
                        .reshape((batch, seq_len, num_v_heads))?;
                let alpha = mixed_ba.narrow(3, beta_size, beta_size)?.reshape((
                    batch,
                    seq_len,
                    num_v_heads,
                ))?;
                Ok((beta, alpha))
            }
            Self::Split { beta, alpha } => {
                let beta = beta.forward_untyped(x)?;
                let alpha = alpha.forward_untyped(x)?;
                log_shape("linear_attn.beta_split", &beta);
                log_shape("linear_attn.alpha_split", &alpha);
                Ok((beta, alpha))
            }
        }
    }
}

impl std::fmt::Debug for BetaAlphaProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fused(fused) => f.debug_tuple("Fused").field(fused).finish(),
            Self::Split { beta, alpha } => f
                .debug_struct("Split")
                .field("beta", beta)
                .field("alpha", alpha)
                .finish(),
        }
    }
}

pub(super) struct LinearAttention {
    /// Input projection for Q, K, V, Z
    pub(super) ssm_in: SsmInProjection,
    /// Input projection for beta and alpha.
    pub(super) ssm_beta_alpha: BetaAlphaProjection,
    /// 1D convolution weights
    pub(super) ssm_conv1d: TSsmConv,
    /// F32 copy of convolution weights to avoid per-step dtype conversion.
    pub(super) ssm_conv1d_f32: TSsmConv,
    /// Output projection: [2048, 4096]
    pub(super) ssm_out: TQMatMul<Shape2<S, DInner>>,
    /// Gated normalization weights
    pub(super) ssm_norm: RmsNorm,
    /// A log parameter (for decay)
    pub(super) ssm_a: TSsmHead,
    /// dt bias
    pub(super) ssm_dt: TSsmHead,
    /// Shared SSM parameters for training (QuZO perturbation visibility)
    pub(super) shared_ssm_a: Option<TSharedSsmHead>,
    pub(super) shared_ssm_conv1d: Option<TSharedSsmConv>,
    pub(super) shared_ssm_dt: Option<TSharedSsmHead>,
    /// Configuration
    pub(super) d_inner: usize,
    pub(super) d_state: usize,
    pub(super) n_groups: usize,
    pub(super) dt_rank: usize,
    /// True when linear-attention V-head tensors in GGUF are stored in tiled head order
    /// ([k0, k1, ..., kN, k0, k1, ...]) rather than grouped order
    /// ([k0, k0, k1, k1, ...]).
    ///
    /// Qwen3.5 GGUF conversion applies this reordering for linear-attention tensors.
    pub(super) v_heads_tiled_order: bool,
    pub(super) conv_kernel_size: usize,
    #[allow(dead_code)]
    pub(super) hidden_size: usize,
    pub(super) rms_norm_eps: f64,
    /// Pre-computed scale factor: 1/sqrt(head_v_dim)
    pub(super) scale: f64,
    /// Recurrent state
    pub(super) recurrent_state: Option<RecurrentState>,
    /// Cached masks for reuse
    pub(super) cached_masks: Option<CachedMasks>,
    #[allow(dead_code)]
    pub(super) span: tracing::Span,
    /// Persistent Arrow stages
    out_project: SsmOutProjectFlow,
    gated_norm_flow: GatedNorm,
}

impl std::fmt::Debug for LinearAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearAttention")
            .field("ssm_in", &self.ssm_in)
            .field("ssm_beta_alpha", &self.ssm_beta_alpha)
            .field("ssm_out", &self.ssm_out)
            .field("ssm_norm", &self.ssm_norm)
            .field("d_inner", &self.d_inner)
            .field("d_state", &self.d_state)
            .field("n_groups", &self.n_groups)
            .field("dt_rank", &self.dt_rank)
            .field("v_heads_tiled_order", &self.v_heads_tiled_order)
            .field("conv_kernel_size", &self.conv_kernel_size)
            .field("hidden_size", &self.hidden_size)
            .field("rms_norm_eps", &self.rms_norm_eps)
            .field("scale", &self.scale)
            .field("recurrent_state", &self.recurrent_state)
            .field("cached_masks", &self.cached_masks)
            .finish()
    }
}

impl LinearAttention {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        d_inner: usize,
        d_state: usize,
        n_groups: usize,
        dt_rank: usize,
        hidden_size: usize,
        rms_norm_eps: f64,
        dtype: DType,
        v_heads_tiled_order: bool,
    ) -> Result<Self> {
        if dt_rank == 0 || n_groups == 0 {
            paramecia_core::bail!(
                "invalid linear-attention config: n_groups={}, dt_rank={}",
                n_groups,
                dt_rank
            );
        }
        if hidden_size != <S as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for hidden_size: runtime={} type-level={}",
                hidden_size,
                <S as Unsigned>::USIZE
            );
        }
        if d_inner != <DInner as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for d_inner: runtime={} type-level={}",
                d_inner,
                <DInner as Unsigned>::USIZE
            );
        }
        if d_state != <DState as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for d_state: runtime={} type-level={}",
                d_state,
                <DState as Unsigned>::USIZE
            );
        }
        if n_groups != <NGroups as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for n_groups: runtime={} type-level={}",
                n_groups,
                <NGroups as Unsigned>::USIZE
            );
        }
        if dt_rank != <DtRank as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for dt_rank: runtime={} type-level={}",
                dt_rank,
                <DtRank as Unsigned>::USIZE
            );
        }

        // SSM tensors: llama.cpp uses .weight suffix for most, see tensor loading code
        // Some GGUFs (e.g. Qwen3-Coder-Next) split ssm_in into attn_qkv + attn_gate
        let ssm_in = match gg
            .typed_qmatmul::<Shape2<QkvzDim, S>>(&format!("{}.ssm_in.weight", prefix))
            .or_else(|_| gg.typed_qmatmul::<Shape2<QkvzDim, S>>(&format!("{}.ssm_in", prefix)))
        {
            Ok(fused) => {
                log_typed_qmatmul_shape(&format!("{}.ssm_in", prefix), &fused);
                SsmInProjection::Fused(fused)
            }
            Err(_) => {
                // Split variant: load attn_qkv (Q,K,V) and attn_gate (Z) separately
                let qkv = gg
                    .typed_qmatmul::<Shape2<QkvDim, S>>(&format!("{}.attn_qkv.weight", prefix))
                    .or_else(|_| {
                        gg.typed_qmatmul::<Shape2<QkvDim, S>>(&format!("{}.attn_qkv", prefix))
                    })?;
                let gate = gg
                    .typed_qmatmul::<Shape2<DInner, S>>(&format!("{}.attn_gate.weight", prefix))
                    .or_else(|_| {
                        gg.typed_qmatmul::<Shape2<DInner, S>>(&format!("{}.attn_gate", prefix))
                    })?;
                log_typed_qmatmul_shape(&format!("{}.attn_qkv", prefix), &qkv);
                log_typed_qmatmul_shape(&format!("{}.attn_gate", prefix), &gate);
                SsmInProjection::Split { qkv, gate }
            }
        };
        // Beta/alpha projection:
        // - fused: ssm_ba / ssm_beta_alpha [64, 2048]
        // - split: ssm_beta + ssm_alpha [32, 2048] each (Qwen3.5-MoE GGUF)
        let ssm_beta_alpha = match gg
            .typed_qmatmul::<Shape2<BaDim, S>>(&format!("{}.ssm_ba.weight", prefix))
            .or_else(|_| gg.typed_qmatmul::<Shape2<BaDim, S>>(&format!("{}.ssm_ba", prefix)))
            .or_else(|_| {
                gg.typed_qmatmul::<Shape2<BaDim, S>>(&format!("{}.ssm_beta_alpha.weight", prefix))
            })
            .or_else(|_| {
                gg.typed_qmatmul::<Shape2<BaDim, S>>(&format!("{}.ssm_beta_alpha", prefix))
            }) {
            Ok(fused) => {
                log_typed_qmatmul_shape(&format!("{}.ssm_beta_alpha", prefix), &fused);
                BetaAlphaProjection::Fused(fused)
            }
            Err(_) => {
                let beta = gg
                    .typed_qmatmul::<Shape2<DtRank, S>>(&format!("{}.ssm_beta.weight", prefix))
                    .or_else(|_| {
                        gg.typed_qmatmul::<Shape2<DtRank, S>>(&format!("{}.ssm_beta", prefix))
                    })?;
                let alpha = gg
                    .typed_qmatmul::<Shape2<DtRank, S>>(&format!("{}.ssm_alpha.weight", prefix))
                    .or_else(|_| {
                        gg.typed_qmatmul::<Shape2<DtRank, S>>(&format!("{}.ssm_alpha", prefix))
                    })?;
                log_typed_qmatmul_shape(&format!("{}.ssm_beta", prefix), &beta);
                log_typed_qmatmul_shape(&format!("{}.ssm_alpha", prefix), &alpha);
                BetaAlphaProjection::Split { beta, alpha }
            }
        };
        let shared_ssm_conv1d = gg
            .shared_tensor_typed::<Shape2<QkvDim, ConvKernel>>(&format!(
                "{}.ssm_conv1d.weight",
                prefix
            ))
            .or_else(|_| {
                gg.shared_tensor_typed::<Shape2<QkvDim, ConvKernel>>(&format!(
                    "{}.ssm_conv1d",
                    prefix
                ))
            })?;
        let ssm_conv1d: TSsmConv = shared_ssm_conv1d.dequant_to(dtype, &gg.device)?;
        let ssm_conv1d_f32: TSsmConv = ssm_conv1d.inner().to_dtype(DType::F32)?.try_into()?;
        let ssm_out = gg
            .typed_qmatmul::<Shape2<S, DInner>>(&format!("{}.ssm_out.weight", prefix))
            .or_else(|_| gg.typed_qmatmul::<Shape2<S, DInner>>(&format!("{}.ssm_out", prefix)))?;
        log_typed_qmatmul_shape(&format!("{}.ssm_out", prefix), &ssm_out);
        let ssm_norm = gg
            .shared_rms_norm(&format!("{}.ssm_norm.weight", prefix), rms_norm_eps, dtype)
            .or_else(|_| {
                gg.shared_rms_norm(&format!("{}.ssm_norm", prefix), rms_norm_eps, dtype)
            })?;
        // ssm_a has no suffix in llama.cpp (LLM_TENSOR_SSM_A_NOSCAN)
        let shared_ssm_a =
            gg.shared_tensor_typed::<Shape1<DtRank>>(&format!("{}.ssm_a", prefix))?;
        let ssm_a: TSsmHead = shared_ssm_a.dequant_to(dtype, &gg.device)?;
        // ssm_dt uses "bias" suffix in llama.cpp, not "weight"
        let shared_ssm_dt = gg
            .shared_tensor_typed::<Shape1<DtRank>>(&format!("{}.ssm_dt.bias", prefix))
            .or_else(|_| gg.shared_tensor_typed::<Shape1<DtRank>>(&format!("{}.ssm_dt", prefix)))
            .or_else(|_| {
                gg.shared_tensor_typed::<Shape1<DtRank>>(&format!("{}.ssm_dt.weight", prefix))
            })?;
        let ssm_dt: TSsmHead = shared_ssm_dt.dequant_to(dtype, &gg.device)?;

        let conv_kernel_size = <ConvKernel as Unsigned>::USIZE;

        let span = tracing::span!(tracing::Level::TRACE, "linear-attn");

        // Pre-compute scale factor: 1/sqrt(head_v_dim) where head_v_dim = d_inner / num_v_heads
        let scale = 1.0 / ((d_inner / dt_rank) as f64).sqrt();

        let out_project = ssm_out_project_flow(ssm_out.clone());
        let gated_norm_flow = GatedNorm::new(&ssm_norm);

        Ok(Self {
            ssm_in,
            ssm_beta_alpha,
            ssm_conv1d,
            ssm_conv1d_f32,
            ssm_out,
            ssm_norm,
            ssm_a,
            ssm_dt,
            shared_ssm_a: Some(shared_ssm_a),
            shared_ssm_conv1d: Some(shared_ssm_conv1d),
            shared_ssm_dt: Some(shared_ssm_dt),
            d_inner,
            d_state,
            n_groups,
            dt_rank,
            v_heads_tiled_order,
            conv_kernel_size,
            hidden_size,
            rms_norm_eps,
            scale,
            recurrent_state: None,
            cached_masks: None,
            span,
            out_project,
            gated_norm_flow,
        })
    }

    #[inline]
    pub(super) fn use_live_shared_ssm_weights() -> bool {
        use std::sync::OnceLock;
        static LIVE: OnceLock<bool> = OnceLock::new();
        *LIVE.get_or_init(|| std::env::var("PARAMECIA_LIVE_SSM_WEIGHTS").is_ok())
    }

    #[inline]
    pub(super) fn ssm_param_on(
        &self,
        device: &Device,
        dtype: DType,
        param: &Tensor,
    ) -> Result<Tensor> {
        let t = if param.device().same_device(device) {
            param.clone()
        } else {
            param.to_device(device)?
        };
        if t.dtype() == dtype {
            Ok(t)
        } else {
            t.to_dtype(dtype)
        }
    }

    pub(super) fn forward_typed(&mut self, x: &THidden) -> Result<THidden> {
        self.forward_inner_typed(x, DeltaNetMode::Normal)
    }

    pub(super) fn forward_with_state_materialization_typed(
        &mut self,
        x: &THidden,
    ) -> Result<THidden> {
        self.forward_inner_typed(x, DeltaNetMode::WithStateMaterialization)
    }

    /// Unified forward implementation for all DeltaNet modes.
    ///
    /// Handles both standard inference (with decode optimizations) and
    /// state-materializing verification in a single code path.
    /// The `mode` parameter controls kernel dispatch:
    /// - `Normal`: autoregressive (l=1), fused batch (l<=16), or chunked (l>16)
    /// - `WithStateMaterialization`: parallel kernel that records intermediate states
    fn forward_inner_typed(&mut self, x: &THidden, mode: DeltaNetMode) -> Result<THidden> {
        let x_inner = x.inner();
        let (b, l, _) = x_inner.dims3()?;
        let decode_single = l == 1;

        let x_contiguous = x_inner.contiguous()?;

        // Compute dimensions
        let head_k_dim = self.d_state;
        let num_k_heads = self.n_groups;
        let num_v_heads = self.dt_rank;
        let head_v_dim = self.d_inner / num_v_heads;

        if head_k_dim != head_v_dim {
            paramecia_core::bail!(
                "qwen3-next expects head_k_dim ({}) to match head_v_dim ({})",
                head_k_dim,
                head_v_dim
            );
        }

        // Stage 1: beta/alpha projection.
        // Fused path parses [B,N,BaDim] into beta+alpha per KV-group.
        // Split path loads beta/alpha directly from separate projections.
        let (beta, alpha) = self.ssm_beta_alpha.forward_beta_alpha(
            &x_contiguous,
            b,
            l,
            num_k_heads,
            num_v_heads,
        )?;

        // Stage 2: QKVZ parse — handle fused (ssm_in) vs split (attn_qkv + attn_gate)
        // Decode single-token optimization: flatten to 2D for matmul then reshape back
        let x_proj = if decode_single {
            Some(x_contiguous.flatten(0, 1)?)
        } else {
            None
        };

        let v_size = head_v_dim * num_v_heads / num_k_heads;
        let qkvz_new_dim = 2 * head_k_dim + 2 * v_size;

        let (q_flat, k_flat, v_flat, z) = if self.ssm_in.is_split() {
            // Split variant: attn_qkv output is flat [Q_all, K_all, V_all]
            // attn_gate output is flat Z — no per-group interleaving
            let qkv = if let Some(ref x_flat) = x_proj {
                self.ssm_in.forward_untyped(x_flat)?.reshape((
                    b,
                    l,
                    head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads,
                ))?
            } else {
                self.ssm_in.forward_untyped(&x_contiguous)?
            };
            let z = if let Some(ref x_flat) = x_proj {
                self.ssm_in
                    .forward_gate_untyped(x_flat)
                    .unwrap()?
                    .reshape((b, l, head_v_dim * num_v_heads))?
            } else {
                self.ssm_in.forward_gate_untyped(&x_contiguous).unwrap()?
            };
            let q_dim = head_k_dim * num_k_heads;
            let k_dim = head_k_dim * num_k_heads;
            let v_dim = head_v_dim * num_v_heads;
            let q_flat = qkv.narrow(2, 0, q_dim)?;
            let k_flat = qkv.narrow(2, q_dim, k_dim)?;
            let v_flat = qkv.narrow(2, q_dim + k_dim, v_dim)?;
            (q_flat, k_flat, v_flat, z)
        } else {
            // Fused variant: ssm_in output is per-group interleaved [Q,K,V,Z]
            let mixed = if let Some(ref x_flat) = x_proj {
                self.ssm_in
                    .forward_untyped(x_flat)?
                    .reshape((b, l, num_k_heads * qkvz_new_dim))?
            } else {
                self.ssm_in.forward_untyped(&x_contiguous)?
            };
            let mixed_qkvz = mixed.reshape((b, l, num_k_heads, qkvz_new_dim))?;
            let q = mixed_qkvz.narrow(3, 0, head_k_dim)?;
            let k = mixed_qkvz.narrow(3, head_k_dim, head_k_dim)?;
            let v = mixed_qkvz.narrow(3, 2 * head_k_dim, v_size)?;
            let z = mixed_qkvz.narrow(3, 2 * head_k_dim + v_size, v_size)?;
            let q_flat = q.reshape((b, l, head_k_dim * num_k_heads))?.contiguous()?;
            let k_flat = k.reshape((b, l, head_k_dim * num_k_heads))?.contiguous()?;
            let v_flat = v.reshape((b, l, head_v_dim * num_v_heads))?.contiguous()?;
            let z_flat = z.reshape((b, l, head_v_dim * num_v_heads))?.contiguous()?;
            (q_flat, k_flat, v_flat, z_flat)
        };
        log_shape("linear_attn.q_flat", &q_flat);
        log_shape("linear_attn.k_flat", &k_flat);
        log_shape("linear_attn.v_flat", &v_flat);
        log_shape("linear_attn.z", &z);

        // Stage 3: Gate computation from beta/alpha
        log_shape("linear_attn.beta", &beta);
        log_shape("linear_attn.alpha", &alpha);

        // ssm_dt and ssm_a have shape [num_v_heads]
        let (ssm_dt, ssm_a) = if Self::use_live_shared_ssm_weights() {
            let ssm_dt = if let Some(ref sq) = self.shared_ssm_dt {
                sq.dequant_to(alpha.dtype(), alpha.device())?.into_inner()
            } else {
                self.ssm_param_on(alpha.device(), alpha.dtype(), self.ssm_dt.inner())?
            };
            let ssm_a = if let Some(ref sq) = self.shared_ssm_a {
                sq.dequant_to(alpha.dtype(), alpha.device())?.into_inner()
            } else {
                self.ssm_param_on(alpha.device(), alpha.dtype(), self.ssm_a.inner())?
            };
            (ssm_dt, ssm_a)
        } else {
            (
                self.ssm_param_on(alpha.device(), alpha.dtype(), self.ssm_dt.inner())?,
                self.ssm_param_on(alpha.device(), alpha.dtype(), self.ssm_a.inner())?,
            )
        };
        let alpha_biased = alpha.broadcast_add(&ssm_dt)?;
        let alpha_softplus = softplus(&alpha_biased)?;
        // ssm_a contains negative values for decay (it's -exp(A_log))
        let gate = alpha_softplus.broadcast_mul(&ssm_a)?;

        // Stage 4: Concatenate + conv + SiLU
        let qkv_cat: TQkvSeq = Tensor::cat(&[&q_flat, &k_flat, &v_flat], 2)?
            .transpose(1, 2)? // [b, qkv_dim, l]
            .try_into()?;
        let conv_out = self.apply_conv(&qkv_cat)?;
        let conv_out = paramecia_nn::ops::silu(conv_out.inner())?;
        log_shape("linear_attn.conv_silu_out", &conv_out);

        // Stage 5: Split back Q, K, V after conv + reshape + head repeat
        // Decode fast-path keeps conv_out in [b, qkv_dim, 1] and avoids full transpose.
        let (q_conv, k_conv, v_conv) = if decode_single {
            let q_conv = conv_out
                .narrow(1, 0, head_k_dim * num_k_heads)?
                .transpose(1, 2)?
                .reshape((b, 1, num_k_heads, head_k_dim))?;
            let k_conv = conv_out
                .narrow(1, head_k_dim * num_k_heads, head_k_dim * num_k_heads)?
                .transpose(1, 2)?
                .reshape((b, 1, num_k_heads, head_k_dim))?;
            let v_conv = conv_out
                .narrow(1, head_k_dim * num_k_heads * 2, head_v_dim * num_v_heads)?
                .transpose(1, 2)?
                .reshape((b, 1, num_v_heads, head_v_dim))?;
            (q_conv, k_conv, v_conv)
        } else {
            let conv_out = conv_out.transpose(1, 2)?; // [b, l, qkv_dim]
            let q_conv = conv_out.narrow(2, 0, head_k_dim * num_k_heads)?;
            let k_conv = conv_out.narrow(2, head_k_dim * num_k_heads, head_k_dim * num_k_heads)?;
            let v_conv =
                conv_out.narrow(2, head_k_dim * num_k_heads * 2, head_v_dim * num_v_heads)?;
            let q_conv = q_conv.reshape((b, l, num_k_heads, head_k_dim))?;
            let k_conv = k_conv.reshape((b, l, num_k_heads, head_k_dim))?;
            let v_conv = v_conv.reshape((b, l, num_v_heads, head_v_dim))?;
            (q_conv, k_conv, v_conv)
        };

        // Repeat Q and K if num_k_heads != num_v_heads.
        //
        // Two possible target head orders:
        // 1) Grouped: [k0, k0, k1, k1, ...] (repeat_interleave style)
        // 2) Tiled:   [k0, k1, ..., kN, k0, k1, ...]
        //
        // Qwen3.5 GGUF conversion stores linear-attention V-related tensors in tiled order,
        // so Q/K must use the same tiled order to align head-wise.
        let (q_conv, k_conv) = if num_k_heads != num_v_heads {
            let repeat_factor = num_v_heads / num_k_heads;
            let (q_repeated, k_repeated) = if self.v_heads_tiled_order {
                let q_tiled = q_conv
                    .unsqueeze(2)?
                    .expand((b, l, repeat_factor, num_k_heads, head_k_dim))?
                    .contiguous()?
                    .reshape((b, l, num_v_heads, head_k_dim))?;
                let k_tiled = k_conv
                    .unsqueeze(2)?
                    .expand((b, l, repeat_factor, num_k_heads, head_k_dim))?
                    .contiguous()?
                    .reshape((b, l, num_v_heads, head_k_dim))?;
                (q_tiled, k_tiled)
            } else {
                let q_grouped = q_conv
                    .unsqueeze(3)?
                    .expand((b, l, num_k_heads, repeat_factor, head_k_dim))?
                    .contiguous()?
                    .reshape((b, l, num_v_heads, head_k_dim))?;
                let k_grouped = k_conv
                    .unsqueeze(3)?
                    .expand((b, l, num_k_heads, repeat_factor, head_k_dim))?
                    .contiguous()?
                    .reshape((b, l, num_v_heads, head_k_dim))?;
                (q_grouped, k_grouped)
            };
            (q_repeated, k_repeated)
        } else {
            (q_conv, k_conv)
        };
        let q_conv_typed: TDeltaNet4 = q_conv.try_into()?;
        let k_conv_typed: TDeltaNet4 = k_conv.try_into()?;
        let v_conv_typed: TDeltaNet4 = v_conv.try_into()?;
        let gate_typed: TDeltaGate3 = gate.try_into()?;
        let beta_typed: TDeltaGate3 = beta.try_into()?;

        // Stage 6: Delta net kernel dispatch (mode-dependent)
        const AUTOREGRESSIVE_LOOP_THRESHOLD: usize = 16;

        let attn_out_typed: TDeltaNet4 = match mode {
            DeltaNetMode::Normal if l == 1 => {
                // Autoregressive: fused kernel handles L2 norm + sigmoid internally
                self.delta_net_autoregressive_typed(
                    &q_conv_typed,
                    &k_conv_typed,
                    &v_conv_typed,
                    &gate_typed,
                    &beta_typed,
                )?
            }
            DeltaNetMode::Normal if l <= AUTOREGRESSIVE_LOOP_THRESHOLD => {
                // Multi-token fused update
                self.delta_net_fused_batch_typed(
                    &q_conv_typed,
                    &k_conv_typed,
                    &v_conv_typed,
                    &gate_typed,
                    &beta_typed,
                )?
            }
            DeltaNetMode::Normal => {
                // Chunked/single-chunk: pre-process Q/K/beta before dispatch
                let q_norm: TDeltaNet4 = crate::ops::l2_normalize_scale(
                    q_conv_typed.inner(),
                    self.scale as f32,
                    self.rms_norm_eps,
                )?
                .try_into()?;
                let k_norm: TDeltaNet4 =
                    l2_normalize(k_conv_typed.inner(), self.rms_norm_eps)?.try_into()?;
                let beta_contig = beta_typed.inner().contiguous()?;
                let beta_sig: TDeltaGate3 = paramecia_nn::ops::sigmoid(&beta_contig)?.try_into()?;
                self.delta_net_chunked_typed(
                    &q_norm,
                    &k_norm,
                    &v_conv_typed,
                    &gate_typed,
                    &beta_sig,
                )?
            }
            DeltaNetMode::WithStateMaterialization => {
                // Parallel kernel that records intermediate states for verification
                let q_norm: TDeltaNet4 = crate::ops::l2_normalize_scale(
                    q_conv_typed.inner(),
                    self.scale as f32,
                    self.rms_norm_eps,
                )?
                .try_into()?;
                let k_norm: TDeltaNet4 =
                    l2_normalize(k_conv_typed.inner(), self.rms_norm_eps)?.try_into()?;
                let beta_contig = beta_typed.inner().contiguous()?;
                let beta_sig: TDeltaGate3 = paramecia_nn::ops::sigmoid(&beta_contig)?.try_into()?;
                self.delta_net_single_chunk_with_states_typed(
                    &q_norm,
                    &k_norm,
                    &v_conv_typed,
                    &gate_typed,
                    &beta_sig,
                    l,
                )?
            }
        };
        log_shape("linear_attn.delta_net_out", attn_out_typed.inner());

        // Stage 7: Gated norm (per-head RMS norm × SiLU gate) — arrow flow
        let z: TDeltaNet4 = z.reshape((b, l, num_v_heads, head_v_dim))?.try_into()?;
        let attn_out_norm: TDeltaNet4 = self
            .gated_norm_flow
            .traced_forward(&mut (), (attn_out_typed, z))?;
        let attn_out_norm = attn_out_norm
            .inner()
            .reshape((b, l, num_v_heads * head_v_dim))?;
        log_shape("linear_attn.gated_norm_out", &attn_out_norm);

        // Stage 8: Typed output projection: [B, N, DInner] → [B, N, S] — arrow flow
        let out_typed: TDInner = attn_out_norm.contiguous()?.try_into()?;
        let result: THidden = self.out_project.traced_forward(&mut (), out_typed)?;
        log_shape("linear_attn.ssm_out", result.inner());
        Ok(result)
    }

    pub(super) fn apply_conv(&mut self, qkv: &TQkvSeq) -> Result<TQkvSeq> {
        let qkv = qkv.inner();
        let (b, dim, l) = qkv.dims3()?;
        // Handle convolution state
        let conv_state_len = self.conv_kernel_size - 1;

        if l == 1 {
            let prev_state: TConvState = if let Some(ref state) = self.recurrent_state {
                state.conv_state_ref().clone().try_into()?
            } else {
                Tensor::zeros((b, dim, conv_state_len), qkv.dtype(), qkv.device())?.try_into()?
            };
            let conv_input: TQkvSeq = Tensor::cat(&[prev_state.inner(), qkv], 2)?.try_into()?;
            let new_conv_state: TConvState = conv_input
                .inner()
                .narrow(2, 1, conv_state_len)?
                .contiguous()?
                .try_into()?;

            let conv_out = self.depthwise_conv1d(&conv_input)?;

            if let Some(ref mut state) = self.recurrent_state {
                state.set_conv_state(new_conv_state.into_inner())?;
            } else {
                let num_heads = self.dt_rank; // num_v_heads
                let state_dim = self.d_inner / self.dt_rank;
                let ssm_state: TSsmState = Tensor::zeros(
                    (b, num_heads, state_dim, state_dim),
                    qkv.dtype(),
                    qkv.device(),
                )?
                .try_into()?;
                self.recurrent_state = Some(RecurrentState::new(
                    ssm_state.into_inner(),
                    new_conv_state.into_inner(),
                )?);
            }

            return Ok(conv_out);
        }

        let (conv_input, new_conv_state): (TQkvSeq, TConvState) = if let Some(ref state) =
            self.recurrent_state
        {
            // Concatenate previous state with current input
            let prev_state: TConvState = state.conv_state_ref().clone().try_into()?;
            let input: TQkvSeq = Tensor::cat(&[prev_state.inner(), qkv], 2)?.try_into()?;

            // Extract new state (last conv_state_len elements)
            let new_state: TConvState = if l >= conv_state_len {
                qkv.narrow(2, l - conv_state_len, conv_state_len)?
                    .contiguous()?
                    .try_into()?
            } else {
                // Need to combine old state with new input
                let old_part = prev_state.inner().narrow(2, l, conv_state_len - l)?;
                Tensor::cat(&[&old_part, qkv], 2)?.try_into()?
            };
            (input, new_state)
        } else {
            // No previous state - pad with zeros
            let zeros: TConvState =
                Tensor::zeros((b, dim, conv_state_len), qkv.dtype(), qkv.device())?.try_into()?;
            let input: TQkvSeq = Tensor::cat(&[zeros.inner(), qkv], 2)?.try_into()?;

            // Extract new state
            let new_state: TConvState = if l >= conv_state_len {
                qkv.narrow(2, l - conv_state_len, conv_state_len)?
                    .contiguous()?
                    .try_into()?
            } else {
                let old_zeros =
                    Tensor::zeros((b, dim, conv_state_len - l), qkv.dtype(), qkv.device())?;
                Tensor::cat(&[&old_zeros, qkv], 2)?.try_into()?
            };
            (input, new_state)
        };

        // Apply 1D convolution (ssm_conv style)
        // conv1d weight is [kernel_size, channels]
        // We need to do depthwise conv
        let conv_out = self.depthwise_conv1d(&conv_input)?;

        // Update state - don't initialize SSM state here, let delta_net functions handle it
        // Just update conv_state. IMPORTANT: preserve backup tensors for snapshot/restore!
        if let Some(ref mut state) = self.recurrent_state {
            // Just update the conv_state in place, preserving ssm_state and backups
            state.set_conv_state(new_conv_state.into_inner())?;
        } else {
            // Initialize SSM state with correct shape: [b, num_heads, state_dim, state_dim]
            let num_heads = self.dt_rank; // num_v_heads
            let state_dim = self.d_inner / self.dt_rank;
            let ssm_state: TSsmState = Tensor::zeros(
                (b, num_heads, state_dim, state_dim),
                qkv.dtype(),
                qkv.device(),
            )?
            .try_into()?;
            self.recurrent_state = Some(RecurrentState::new(
                ssm_state.into_inner(),
                new_conv_state.into_inner(),
            )?);
        }

        Ok(conv_out)
    }

    pub(super) fn depthwise_conv1d(&self, input: &TQkvSeq) -> Result<TQkvSeq> {
        let input = input.inner();
        // input: [b, channels, input_len]
        // ssm_conv1d: [channels, kernel_size] (GGUF stores it this way)
        // Default inference path uses resident weights.
        // Set PARAMECIA_LIVE_SSM_WEIGHTS=1 to re-read shared tensors every forward.
        let conv_weight = if Self::use_live_shared_ssm_weights() {
            if let Some(ref sq) = self.shared_ssm_conv1d {
                sq.dequant_to(input.dtype(), input.device())?.into_inner()
            } else {
                self.ssm_param_on(input.device(), input.dtype(), self.ssm_conv1d.inner())?
            }
        } else {
            self.ssm_param_on(input.device(), input.dtype(), self.ssm_conv1d.inner())?
        };
        // Note: depthwise_conv1d kernel requires F32, convert if needed
        let input_dtype = input.dtype();
        let input_f32 = if input_dtype != DType::F32 {
            input.to_dtype(DType::F32)?
        } else {
            input.clone()
        };
        let weight_f32 = if Self::use_live_shared_ssm_weights() {
            conv_weight.to_dtype(DType::F32)?
        } else if self
            .ssm_conv1d_f32
            .inner()
            .device()
            .same_device(input.device())
        {
            self.ssm_conv1d_f32.inner().clone()
        } else {
            self.ssm_conv1d_f32.inner().to_device(input.device())?
        };
        let result = crate::ops::depthwise_conv1d(&input_f32, &weight_f32)?;
        // Convert back to original dtype
        if input_dtype != DType::F32 {
            result.to_dtype(input_dtype)?.try_into().map_err(Into::into)
        } else {
            result.try_into().map_err(Into::into)
        }
    }

    fn delta_net_autoregressive_typed(
        &mut self,
        q: &TDeltaNet4,
        k: &TDeltaNet4,
        v: &TDeltaNet4,
        gate: &TDeltaGate3,
        beta: &TDeltaGate3,
    ) -> Result<TDeltaNet4> {
        self.delta_net_autoregressive(q.inner(), k.inner(), v.inner(), gate.inner(), beta.inner())?
            .try_into()
            .map_err(Into::into)
    }

    fn delta_net_chunked_typed(
        &mut self,
        q: &TDeltaNet4,
        k: &TDeltaNet4,
        v: &TDeltaNet4,
        gate: &TDeltaGate3,
        beta: &TDeltaGate3,
    ) -> Result<TDeltaNet4> {
        self.delta_net_chunked(q.inner(), k.inner(), v.inner(), gate.inner(), beta.inner())?
            .try_into()
            .map_err(Into::into)
    }

    fn delta_net_single_chunk_with_states_typed(
        &mut self,
        q: &TDeltaNet4,
        k: &TDeltaNet4,
        v: &TDeltaNet4,
        gate: &TDeltaGate3,
        beta: &TDeltaGate3,
        l: usize,
    ) -> Result<TDeltaNet4> {
        self.delta_net_single_chunk_with_states(
            q.inner(),
            k.inner(),
            v.inner(),
            gate.inner(),
            beta.inner(),
            l,
        )?
        .try_into()
        .map_err(Into::into)
    }

    fn delta_net_fused_batch_typed(
        &mut self,
        q: &TDeltaNet4,
        k: &TDeltaNet4,
        v: &TDeltaNet4,
        gate: &TDeltaGate3,
        beta: &TDeltaGate3,
    ) -> Result<TDeltaNet4> {
        self.delta_net_fused_batch(q.inner(), k.inner(), v.inner(), gate.inner(), beta.inner())?
            .try_into()
            .map_err(Into::into)
    }

    pub(super) fn delta_net_autoregressive(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        // Autoregressive (single token) delta net computation using fused CUDA kernel
        // Input shapes: q, k: [b, 1, num_heads, head_k_dim], v: [b, 1, num_heads, head_v_dim]
        // gate, beta: [b, 1, num_heads]
        let (b, _l, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            paramecia_core::bail!(
                "delta_net_autoregressive expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // Squeeze the sequence dimension: [b, num_heads, head_dim]
        let q_sq = q.squeeze(1)?.contiguous()?;
        let k_sq = k.squeeze(1)?.contiguous()?;
        let v_sq = v.squeeze(1)?.contiguous()?;
        let gate_sq = gate.squeeze(1)?.contiguous()?; // [b, num_heads]
        let beta_sq = beta.squeeze(1)?.contiguous()?; // [b, num_heads] (pre-sigmoid)

        // Get or initialize state: [b, num_heads, head_v_dim, head_v_dim]
        let state = if let Some(ref rs) = self.recurrent_state {
            rs.ssm_state_ref().contiguous()?
        } else {
            Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?
        };

        // Use fused CUDA kernel for autoregressive step
        let (output, new_state) = crate::ops::delta_net_autoregressive_step(
            &q_sq,
            &k_sq,
            &v_sq,
            &gate_sq,
            &beta_sq,
            &state,
            self.scale as f32,
            self.rms_norm_eps as f32,
        )?;

        // Store updated state and update gate offset
        if let Some(ref mut rs) = self.recurrent_state {
            rs.set_ssm_state(new_state)?;
            // Update gate offset: add current gate to accumulated offset
            let next_offset = Some(if let Some(offset) = rs.gate_offset_ref() {
                (offset + &gate_sq)?
            } else {
                gate_sq.clone()
            });
            rs.set_gate_offset(next_offset)?;
        } else {
            let conv_state = Tensor::zeros(
                (
                    b,
                    self.d_inner + 2 * self.n_groups * self.d_state,
                    self.conv_kernel_size - 1,
                ),
                q.dtype(),
                q.device(),
            )?;
            let mut rs = RecurrentState::new(new_state, conv_state)?;
            rs.set_gate_offset(Some(gate_sq.clone()))?;
            self.recurrent_state = Some(rs);
        }

        // Reshape output: [b, num_heads, head_v_dim] -> [b, 1, num_heads, head_v_dim]
        output.unsqueeze(1)
    }
    pub(super) fn delta_net_chunked(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            paramecia_core::bail!(
                "delta_net_chunked expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // For short sequences, use non-chunked path (avoids chunking overhead)
        if l <= DELTA_NET_CHUNK_SIZE {
            return self.delta_net_single_chunk(q, k, v, gate, beta, l);
        }

        // === CHUNKED PATH for longer sequences ===
        // Following llama.cpp: pad to multiples of CHUNK_SIZE and process chunk by chunk
        // NOTE: q/k are already L2-normalized and beta is already sigmoided by the caller.

        let chunk_size = DELTA_NET_CHUNK_SIZE;
        let pad = (chunk_size - l % chunk_size) % chunk_size;
        let padded_len = l + pad;
        let n_chunks = padded_len / chunk_size;

        // Transpose to [b, num_heads, l, head_dim] for batch matmul
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        log_shape("chunked.q_t", &q_t);
        log_shape("chunked.k_t", &k_t);
        log_shape("chunked.v_t", &v_t);

        // beta: [b, l, num_heads] -> [b, num_heads, l, 1] (already sigmoided)
        let beta_expanded = beta.transpose(1, 2)?.unsqueeze(3)?;

        // v_beta = v * beta, k_beta = k * beta
        let v_beta = v_t.broadcast_mul(&beta_expanded)?;
        let k_beta = k_t.broadcast_mul(&beta_expanded)?;

        // Pad tensors to padded_len if needed
        // Note: gate operations use F32 for cumsum/exp to avoid overflow; model uses F32 activations
        let (q_t, k_t, v_beta, k_beta, gate_t) = if pad > 0 {
            let q_t = pad_tensor(&q_t, 2, pad)?;
            let k_t = pad_tensor(&k_t, 2, pad)?;
            let v_beta = pad_tensor(&v_beta, 2, pad)?;
            let k_beta = pad_tensor(&k_beta, 2, pad)?;
            let gate_t = gate.transpose(1, 2)?.contiguous()?;
            let gate_t = pad_tensor(&gate_t, 2, pad)?;
            (q_t, k_t, v_beta, k_beta, gate_t)
        } else {
            let gate_t = gate.transpose(1, 2)?.contiguous()?;
            (q_t, k_t, v_beta, k_beta, gate_t)
        };

        // Create chunk-size masks in F32 for decay computation (matches model dtype)
        let chunk_causal_mask: TCausalMask =
            create_causal_mask(chunk_size, q.device(), DType::F32)?.try_into()?;
        let chunk_identity: TCausalMask =
            create_identity_mask(chunk_size, q.device(), DType::F32)?.try_into()?;
        let chunk_causal_diag_mask: TCausalMask =
            (chunk_causal_mask.inner() + chunk_identity.inner())?.try_into()?;

        // Initialize or get state and gate offset.
        // Keep state transposed for the chunk loop to avoid repeated transpose churn.
        let (mut state_t, mut running_gate_offset) = if let Some(ref rs) = self.recurrent_state {
            let offset = rs.gate_offset_ref().cloned();
            (rs.ssm_state_ref().transpose(2, 3)?, offset)
        } else {
            let state = Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?;
            (state.transpose(2, 3)?, None)
        };

        // Cumulative sum of gate for decay computation (per chunk)
        // NOTE: This uses per-chunk cumsum to avoid exp() overflow with very long sequences.
        // The chunk-local cumsum is correct for within-chunk decay computation.
        // For prefix cache continuation, the state already encodes the prefix decay.
        // gate_t is F32 to avoid overflow in cumsum/exp
        let gate_cumsum = {
            let gate_reshaped = gate_t.reshape((b, num_heads, n_chunks, chunk_size))?;
            let gate_flat = gate_reshaped.reshape((b * num_heads * n_chunks, chunk_size))?;
            let gate_cumsum_flat = gate_flat.cumsum(1)?;
            gate_cumsum_flat
                .reshape((b, num_heads, n_chunks, chunk_size))?
                .reshape((b, num_heads, padded_len))?
        };

        // Pre-compute loop-invariant gate_cumsum_exp, kbeta_gexp, q_g_exp for ALL chunks.
        // These depend only on gate_cumsum (already computed) and k_beta/q_t (fixed),
        // so we compute them once and narrow per-chunk instead of recomputing each iteration.
        let gate_cumsum_exp_full = gate_cumsum.unsqueeze(3)?.exp()?;
        let kbeta_gexp_full = k_beta.broadcast_mul(&gate_cumsum_exp_full)?;
        let q_g_exp_full = q_t.broadcast_mul(&gate_cumsum_exp_full)?;
        log_shape("chunked.gate_cumsum", &gate_cumsum);
        log_shape("chunked.kbeta_gexp_full", &kbeta_gexp_full);
        log_shape("chunked.q_g_exp_full", &q_g_exp_full);

        // Process chunks and collect outputs
        let mut chunk_outputs: Vec<Tensor> = Vec::with_capacity(n_chunks);

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;

            // Extract chunk tensors
            let q_chunk = q_t.narrow(2, start, chunk_size)?;
            let k_chunk = k_t.narrow(2, start, chunk_size)?;
            let v_beta_chunk = v_beta.narrow(2, start, chunk_size)?;
            let k_beta_chunk = k_beta.narrow(2, start, chunk_size)?;
            let gate_cumsum_chunk = gate_cumsum.narrow(2, start, chunk_size)?;

            // Narrow pre-computed loop-invariant tensors for this chunk
            let kbeta_gexp_chunk = kbeta_gexp_full.narrow(2, start, chunk_size)?;
            let q_g_exp_chunk = q_g_exp_full.narrow(2, start, chunk_size)?;

            // Compute decay mask in F32, then convert to input dtype for matmul
            // First mask zeros the upper triangle before exp() to prevent overflow.
            // After exp(), upper triangle has exp(0)=1.0 but downstream ops (causal_mask,
            // causal_diag_mask multiplies) zero them, so a second masking is unnecessary.
            let gate_i = gate_cumsum_chunk.unsqueeze(3)?;
            let gate_j = gate_cumsum_chunk.unsqueeze(2)?;
            let decay_diff = gate_i.broadcast_sub(&gate_j)?;
            let decay_mask = decay_diff.broadcast_mul(chunk_causal_diag_mask.inner())?;
            let decay_mask = decay_mask.exp()?;

            // === Intra-chunk attention ===
            // kmulkbeta = k_beta @ k^T
            let kmulkbeta = k_beta_chunk.matmul(&k_chunk.transpose(2, 3)?)?;
            let k_decay = kmulkbeta.broadcast_mul(&decay_mask)?;
            let attn1 = k_decay.broadcast_mul(chunk_causal_mask.inner())?.neg()?;

            // Triangular solve
            let attn1_solved = solve_lower_triangular(&attn1, chunk_causal_mask.inner())?;

            // value = attn1_solved @ v_beta
            let value = attn1_solved.matmul(&v_beta_chunk)?;

            // For state interaction, use LOCAL cumsum (not total cumsum with offset).
            // The state already encodes decay from all previous chunks, so we only need
            // the decay from the START of this chunk to each position within the chunk.
            // Adding the offset would double-count the decay from previous chunks,
            // causing the state contribution to be incorrectly scaled down (context forgetting).
            // Reference: llama.cpp build_delta_net_chunking uses local cumsum for gexp_chunk.

            // k_cumdecay = attn1_solved @ (k_beta * exp(gate_cumsum))
            let k_cumdecay = attn1_solved.matmul(&kbeta_gexp_chunk)?;

            // q @ k^T attention (strictly lower triangular)
            let attn2 = q_chunk.matmul(&k_chunk.transpose(2, 3)?)?;
            let attn2 = attn2.broadcast_mul(&decay_mask)?;
            let attn2 = attn2.broadcast_mul(chunk_causal_diag_mask.inner())?;

            // === Inter-chunk (state) contribution ===
            // v_prime[t, i] = sum_j(k_cumdecay[t, j] * state[i, j])
            // In llama.cpp: v_prime = state^T @ k_cumdecay (via ggml_mul_mat(state_t, k_cumdecay))
            // state_t stays transposed across chunks to avoid repeated transposes.
            let v_prime = k_cumdecay.matmul(&state_t)?;

            // v_new = value - v_prime
            let v_new = (value - v_prime)?;

            // attn_inter[t, i] = sum_j(q[t, j] * g[t] * state[i, j])
            // In llama.cpp: attn_inter = state^T @ q_g_exp
            let attn_inter = q_g_exp_chunk.matmul(&state_t)?;

            // core_attn_out = attn_inter + attn2 @ v_new
            let v_attn = attn2.matmul(&v_new)?;
            let core_attn_out_chunk = (attn_inter + v_attn)?;

            chunk_outputs.push(core_attn_out_chunk);

            // === Update state for next chunk ===
            let gate_last = gate_cumsum_chunk.narrow(2, chunk_size - 1, 1)?;
            let g_diff = gate_last.broadcast_sub(&gate_cumsum_chunk)?;
            let g_diff_exp = g_diff.unsqueeze(3)?.exp()?;

            // kgdmulvnew_t[j, i] = sum_t(k[t, j] * g_diff[t] * v_new[t, i])
            // key_gdiff^T @ v_new: [b,h,d,seq] @ [b,h,seq,d] = [b,h,d,d]
            let key_gdiff = k_chunk.broadcast_mul(&g_diff_exp)?;
            let kgdmulvnew_t = key_gdiff.transpose(2, 3)?.matmul(&v_new)?;

            let g_last_exp = gate_last.unsqueeze(3)?.exp()?;
            state_t = (state_t.broadcast_mul(&g_last_exp)? + kgdmulvnew_t)?;

            // Update running gate offset: add this chunk's total gate
            let chunk_gate_sum = gate_last.squeeze(2)?; // [b, num_heads]
            running_gate_offset = Some(if let Some(offset) = running_gate_offset {
                (&offset + &chunk_gate_sum)?
            } else {
                chunk_gate_sum
            });
        }

        // Concatenate all chunk outputs
        let core_attn_out = Tensor::cat(&chunk_outputs, 2)?;
        log_shape("chunked.core_attn_out", &core_attn_out);

        // Remove padding if we added any
        let core_attn_out = if pad > 0 {
            core_attn_out.narrow(2, 0, l)?
        } else {
            core_attn_out
        };

        let state = state_t.transpose(2, 3)?;
        log_shape("chunked.final_state", &state);

        // Store final state for autoregressive phase, preserving backup tensors
        if let Some(ref mut rs) = self.recurrent_state {
            rs.set_ssm_state(state)?;
            rs.set_gate_offset(running_gate_offset)?;
            // conv_state is already correct from apply_conv, don't overwrite
        } else {
            let conv_dim = self.d_inner + 2 * self.n_groups * self.d_state;
            let conv_state = Tensor::zeros(
                (b, conv_dim, self.conv_kernel_size - 1),
                state.dtype(),
                state.device(),
            )?;
            let mut rs = RecurrentState::new(state, conv_state)?;
            rs.set_gate_offset(running_gate_offset)?;
            self.recurrent_state = Some(rs);
        }

        // Transpose back to [b, l, num_heads, head_v_dim]
        // Output is already in input dtype
        core_attn_out.transpose(1, 2)
    }

    /// Single-chunk delta net computation (for sequences <= CHUNK_SIZE)
    /// This avoids chunking overhead for short sequences
    pub(super) fn delta_net_single_chunk(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
        l: usize,
    ) -> Result<Tensor> {
        let (b, _, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            paramecia_core::bail!(
                "delta_net_single_chunk expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // NOTE: q/k are already L2-normalized and beta is already sigmoided by the caller.

        // Transpose to [b, num_heads, l, head_dim] for batch matmul
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        log_shape("single_chunk.q_t", &q_t);
        log_shape("single_chunk.k_t", &k_t);
        log_shape("single_chunk.v_t", &v_t);

        // beta: [b, l, num_heads] -> [b, num_heads, l, 1] (already sigmoided)
        let beta_expanded = beta.transpose(1, 2)?.unsqueeze(3)?;

        // v_beta = v * beta, k_beta = k * beta
        let v_beta = v_t.broadcast_mul(&beta_expanded)?;
        let k_beta = k_t.broadcast_mul(&beta_expanded)?;

        // Get or create cached masks (F32 matches model dtype)
        let (causal_mask, causal_diag_mask) = {
            let needs_new = match &self.cached_masks {
                Some(cached) => cached.seq_len != l,
                None => true,
            };

            if needs_new {
                let cm: TCausalMask = create_causal_mask(l, q.device(), DType::F32)?.try_into()?;
                let identity: TCausalMask =
                    create_identity_mask(l, q.device(), DType::F32)?.try_into()?;
                let cdm: TCausalMask = (cm.inner() + identity.inner())?.try_into()?;

                self.cached_masks = Some(CachedMasks {
                    seq_len: l,
                    causal_mask: cm.clone(),
                    causal_diag_mask: cdm.clone(),
                });
                (cm.inner().clone(), cdm.inner().clone())
            } else {
                let cached = self.cached_masks.as_ref().ok_or_else(|| {
                    paramecia_core::Error::Msg("missing cached masks".to_string())
                })?;
                (
                    cached.causal_mask.inner().clone(),
                    cached.causal_diag_mask.inner().clone(),
                )
            }
        };

        // gate: [b, l, num_heads] -> [b, num_heads, l]
        let gate_t = gate.transpose(1, 2)?.contiguous()?;
        let gate_cumsum = gate_t.cumsum(2)?;

        // Track gate offset for state updates, but do NOT add it to gate_cumsum.
        // The state already encodes decay from all previous tokens, so we only need
        // the LOCAL cumsum (decay within this chunk) for state interaction.
        // Adding the offset would double-count the decay, causing context forgetting.
        // Reference: llama.cpp build_delta_net_chunking uses local cumsum for gexp_chunk.
        let gate_offset_for_state = {
            let local_last = gate_cumsum.narrow(2, l - 1, 1)?.squeeze(2)?;
            if let Some(ref rs) = self.recurrent_state {
                if let Some(offset) = rs.gate_offset_ref() {
                    // Accumulate offset for next continuation
                    Some((offset + &local_last)?)
                } else {
                    Some(local_last)
                }
            } else {
                Some(local_last)
            }
        };

        // Compute decay mask (within-chunk decay is still based on relative positions)
        // First mask zeros the upper triangle before exp() to prevent overflow.
        // After exp(), upper triangle has exp(0)=1.0 but downstream ops (causal_mask,
        // causal_diag_mask multiplies) zero them, so a second masking is unnecessary.
        let gate_i = gate_cumsum.unsqueeze(3)?;
        let gate_j = gate_cumsum.unsqueeze(2)?;
        let decay_diff = gate_i.broadcast_sub(&gate_j)?;
        let decay_mask = decay_diff.broadcast_mul(&causal_diag_mask)?;
        let decay_mask = decay_mask.exp()?;

        // kmulkbeta = k_beta @ k^T
        let kmulkbeta = k_beta.matmul(&k_t.transpose(2, 3)?)?;
        let k_decay = kmulkbeta.broadcast_mul(&decay_mask)?;
        let attn1 = k_decay.broadcast_mul(&causal_mask)?.neg()?;

        // Triangular solve
        let attn1_solved = solve_lower_triangular(&attn1, &causal_mask)?;

        // value = attn1_solved @ v_beta
        let value = attn1_solved.matmul(&v_beta)?;

        // k_cumdecay = attn1_solved @ (k_beta * exp(gate_cumsum))
        let gate_cumsum_exp = gate_cumsum.unsqueeze(3)?.exp()?;
        let kbeta_gexp = k_beta.broadcast_mul(&gate_cumsum_exp)?;
        let k_cumdecay = attn1_solved.matmul(&kbeta_gexp)?;

        // q @ k^T attention (strictly lower triangular)
        let attn2 = q_t.matmul(&k_t.transpose(2, 3)?)?;
        let attn2 = attn2.broadcast_mul(&decay_mask)?;
        let attn2 = attn2.broadcast_mul(&causal_diag_mask)?;

        // Get or initialize state (in input dtype)
        let state = if let Some(ref rs) = self.recurrent_state {
            rs.ssm_state_ref().clone()
        } else {
            Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?
        };

        // v_prime[t, i] = sum_j(k_cumdecay[t, j] * state[i, j])
        // Need transposed state for correct matrix-vector multiply
        let state_t = state.transpose(2, 3)?;
        let v_prime = k_cumdecay.matmul(&state_t)?;
        let v_new = (value - v_prime)?;

        // attn_inter[t, i] = sum_j(q[t, j] * g[t] * state[i, j])
        let q_g_exp = q_t.broadcast_mul(&gate_cumsum_exp)?;
        let attn_inter = q_g_exp.matmul(&state_t)?;

        // core_attn_out = attn_inter + attn2 @ v_new
        let v_attn = attn2.matmul(&v_new)?;
        let core_attn_out = (attn_inter + v_attn)?;
        log_shape("single_chunk.core_attn_out", &core_attn_out);

        // Update state
        let gate_last = gate_cumsum.narrow(2, l - 1, 1)?;
        let g_diff = gate_last.broadcast_sub(&gate_cumsum)?;
        let g_diff_exp = g_diff.unsqueeze(3)?.exp()?;

        // kgdmulvnew[i, j] = sum_t(v_new[t, i] * k[t, j] * g_diff[t])
        // v_new^T @ key_gdiff: [b,h,d,seq] @ [b,h,seq,d] = [b,h,d,d]
        let key_gdiff = k_t.broadcast_mul(&g_diff_exp)?;
        let kgdmulvnew = v_new.transpose(2, 3)?.matmul(&key_gdiff)?;

        let g_last_exp = gate_last.unsqueeze(3)?.exp()?;
        let new_state = (state.broadcast_mul(&g_last_exp)? + kgdmulvnew)?;
        log_shape("single_chunk.new_state", &new_state);

        // Store state while preserving backup tensors
        if let Some(ref mut rs) = self.recurrent_state {
            rs.set_ssm_state(new_state)?;
            rs.set_gate_offset(gate_offset_for_state)?;
            // conv_state is already correct from apply_conv, don't overwrite
        } else {
            let conv_dim = self.d_inner + 2 * self.n_groups * self.d_state;
            let conv_state = Tensor::zeros(
                (b, conv_dim, self.conv_kernel_size - 1),
                new_state.dtype(),
                new_state.device(),
            )?;
            let mut rs = RecurrentState::new(new_state, conv_state)?;
            rs.set_gate_offset(gate_offset_for_state)?;
            self.recurrent_state = Some(rs);
        }

        // Output is already in input dtype
        core_attn_out.transpose(1, 2)
    }

    /// Single-chunk delta net with PARALLEL intermediate state materialization.
    ///
    /// Uses a fused CUDA kernel that computes DeltaNet outputs AND materializes
    /// intermediate states at each position in a single kernel launch, enabling
    /// O(1) state slicing for speculative decoding verification.
    pub(super) fn delta_net_single_chunk_with_states(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
        l: usize,
    ) -> Result<Tensor> {
        let (b, _, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            paramecia_core::bail!(
                "delta_net_single_chunk_with_states expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // NOTE: q/k are already L2-normalized and beta is already sigmoided by the caller.

        // Transpose to [b, num_heads, l, head_dim] for the CUDA kernel
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        // gate and beta: [b, l, num_heads] -> [b, num_heads, l]
        let gate_t = gate.transpose(1, 2)?.contiguous()?;
        let beta_t = beta.transpose(1, 2)?.contiguous()?;

        // Get or initialize state
        let state = if let Some(ref rs) = self.recurrent_state {
            rs.ssm_state_ref().clone()
        } else {
            Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?
        };

        // Call the fused CUDA kernel that computes outputs AND intermediate states
        let (output, final_state, all_states) =
            crate::ops::delta_net_parallel_with_states(&q_t, &k_t, &v_t, &gate_t, &beta_t, &state)?;

        // Store final state and intermediate states
        // all_states: [b, h, l, d, d]
        if let Some(ref mut rs) = self.recurrent_state {
            rs.set_ssm_state(final_state)?;
            // Extract individual states for O(1) access
            let mut intermediate_states = Vec::with_capacity(l);
            for t in 0..l {
                let state_t = all_states.narrow(2, t, 1)?.squeeze(2)?;
                intermediate_states.push(state_t.try_into()?);
            }
            rs.intermediate_states = Some(intermediate_states);
        } else {
            let conv_dim = self.d_inner + 2 * self.n_groups * self.d_state;
            let conv_state = Tensor::zeros(
                (b, conv_dim, self.conv_kernel_size - 1),
                final_state.dtype(),
                final_state.device(),
            )?;
            let mut rs = RecurrentState::new(final_state, conv_state)?;
            let mut intermediate_states = Vec::with_capacity(l);
            for t in 0..l {
                let state_t = all_states.narrow(2, t, 1)?.squeeze(2)?;
                intermediate_states.push(state_t.try_into()?);
            }
            rs.intermediate_states = Some(intermediate_states);
            self.recurrent_state = Some(rs);
        }

        // output is [b, h, l, d], transpose to [b, l, h, d]
        output.transpose(1, 2)
    }

    pub(super) fn clear_state(&mut self) {
        self.recurrent_state = None;
    }

    /// Save a snapshot of the current recurrent state for later rollback.
    /// Uses in-place copy to backup buffers (no allocation after first call).
    pub(super) fn snapshot_state(&mut self) -> Result<()> {
        if let Some(ref mut state) = self.recurrent_state {
            state.snapshot()?;
        }
        Ok(())
    }

    /// Restore the recurrent state from the saved snapshot.
    /// Uses deep copy to ensure state independence.
    pub(super) fn restore_state(&mut self) {
        if let Some(ref mut state) = self.recurrent_state {
            if let Err(e) = state.restore() {
                warn!(error = %e, "Failed to restore LinearAttention state");
            }
        }
    }

    /// Initialize intermediate states buffer for verification with state slicing.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    pub(super) fn init_intermediate_states(&mut self, seq_len: usize) {
        if let Some(ref mut state) = self.recurrent_state {
            state.init_intermediate_states(seq_len);
        }
    }

    /// Clear intermediate states buffer.
    pub(super) fn clear_intermediate_states(&mut self) {
        if let Some(ref mut state) = self.recurrent_state {
            state.clear_intermediate_states();
        }
    }

    /// Restore to a specific intermediate state by index.
    /// This is O(1) - just swaps tensor references.
    pub(super) fn restore_to_intermediate_state(&mut self, index: usize) -> bool {
        if let Some(ref mut state) = self.recurrent_state {
            state.restore_to_intermediate(index)
        } else {
            false
        }
    }

    /// Get the number of stored intermediate states.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    pub(super) fn num_intermediate_states(&self) -> usize {
        self.recurrent_state
            .as_ref()
            .map(|s| s.num_intermediate_states())
            .unwrap_or(0)
    }

    /// Save the current recurrent state for prefix caching.
    /// Returns a deep copy of the SSM state, conv state, and gate offset.
    pub(super) fn save_state_for_prefix(&self) -> PrefixCacheEntry {
        if let Some(ref state) = self.recurrent_state {
            let ssm = state.ssm_state.clone();
            let conv = state.conv_state.clone();
            let gate_offset = match state.gate_cumsum_offset.clone() {
                Some(offset) => offset,
                None => {
                    let (b, num_heads, _, _) = ssm.inner().dims4().unwrap_or((1, 1, 1, 1));
                    let zero = match Tensor::zeros(
                        (b, num_heads),
                        ssm.inner().dtype(),
                        ssm.inner().device(),
                    ) {
                        Ok(t) => t,
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                "Failed to create zero gate offset for prefix cache"
                            );
                            return PrefixCacheEntry::Empty;
                        }
                    };
                    match zero.try_into() {
                        Ok(t) => t,
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                "Failed to type zero gate offset for prefix cache"
                            );
                            return PrefixCacheEntry::Empty;
                        }
                    }
                }
            };
            return PrefixCacheEntry::LinearAttention {
                ssm_state: ssm,
                conv_state: conv,
                gate_cumsum_offset: gate_offset,
            };
        }
        PrefixCacheEntry::Empty
    }

    /// Restore recurrent state from a prefix cache entry.
    pub(super) fn restore_state_from_prefix(&mut self, entry: &PrefixCacheEntry) -> Result<()> {
        match entry {
            PrefixCacheEntry::LinearAttention {
                ssm_state,
                conv_state,
                gate_cumsum_offset,
            } => {
                // Restore full state including gate offset for correct prefix cache continuation
                self.recurrent_state = Some(RecurrentState::with_gate_offset(
                    ssm_state.inner().contiguous()?,
                    conv_state.inner().contiguous()?,
                    gate_cumsum_offset.inner().contiguous()?,
                )?);
                Ok(())
            }
            PrefixCacheEntry::Empty => {
                self.clear_state();
                Ok(())
            }
            PrefixCacheEntry::FullAttention { .. } => {
                // Wrong entry type - clear state
                self.clear_state();
                Ok(())
            }
        }
    }

    /// Fused multi-token delta net computation for small batches.
    ///
    /// Uses a fused CUDA kernel that processes K tokens with the same algorithm
    /// as K sequential autoregressive steps, ensuring identical state updates.
    /// This is used for verification where correctness is critical.
    ///
    pub(super) fn delta_net_fused_batch(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (b, _l, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            paramecia_core::bail!(
                "delta_net_fused_batch expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // L2 normalize Q (with scale) and K
        let q_norm = crate::ops::l2_normalize_scale(q, self.scale as f32, self.rms_norm_eps)?;
        let k_norm = l2_normalize(k, self.rms_norm_eps)?;

        // Apply sigmoid to beta: [b, l, num_heads]
        let beta_sig = paramecia_nn::ops::sigmoid(beta)?;

        // Transpose to [b, num_heads, l, head_dim] for the fused kernel
        let q_t = q_norm.transpose(1, 2)?.contiguous()?;
        let k_t = k_norm.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        let gate_t = gate.transpose(1, 2)?.contiguous()?;
        let beta_t = beta_sig.transpose(1, 2)?.contiguous()?;

        // Get or initialize state
        let state = if let Some(ref rs) = self.recurrent_state {
            rs.ssm_state_ref().clone()
        } else {
            Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?
        };

        // Use fused multi-token update kernel
        // Note: q_t is already L2-normalized with scale applied by l2_normalize_scale above
        let (output, new_state) =
            crate::ops::delta_net_multi_token_update(&q_t, &k_t, &v_t, &gate_t, &beta_t, &state)?;

        // Store final state, preserving backup tensors
        if let Some(ref mut rs) = self.recurrent_state {
            rs.set_ssm_state(new_state)?;
            // conv_state is already correct from apply_conv, don't overwrite
        } else {
            let conv_dim = self.d_inner + 2 * self.n_groups * self.d_state;
            let conv_state = Tensor::zeros(
                (b, conv_dim, self.conv_kernel_size - 1),
                new_state.dtype(),
                new_state.device(),
            )?;
            self.recurrent_state = Some(RecurrentState::new(new_state, conv_state)?);
        }

        // Transpose output back to [b, l, num_heads, head_v_dim]
        output.transpose(1, 2)
    }
}
