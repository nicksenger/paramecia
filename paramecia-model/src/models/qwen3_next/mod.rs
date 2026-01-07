//! Qwen3-Next implementation with quantization support.
//!
//! Based on Qwen3-Next architecture which is a hybrid model featuring:
//! - Full attention layers with gated Q projection
//! - Linear attention layers (Gated Delta Net) for efficient recurrence
//! - Mixture-of-Experts (MoE) FFN with shared experts
//! - YARN (Yet Another RoPE extensioN) for extended context up to 1M tokens
//!
//! References:
//! - llama.cpp qwen3next implementation
//! - Delta Net: https://arxiv.org/abs/2406.06484
//! - YARN: https://arxiv.org/abs/2309.00071
//!

mod config;
mod expert_cache;
pub mod full_attention;
mod gguf_loader;
mod kv_cache;
mod linear_attention;
mod model_weights;
mod moe;
mod mtp;
mod rope;
mod shape;
mod types;
mod utils;

// Public API re-exports (keeps lib.rs and all external consumers unchanged)
pub use config::{select_best_device, DeviceOffloadMode, KvCacheQuantization, LayerDeviceMap};
pub use kv_cache::{LayerSnapshot, PrefixCache, PrefixCacheEntry};
pub use model_weights::ModelWeights;
pub use mtp::MtpHead;
pub use rope::YarnConfig;
pub use types::{SpeculativeResult, VerificationResult};

use crate::quantized_nn::RmsNorm;
use inception::{primitive, Inception};
use paramecia_arrow::vis::{Graph, Vis, Visualize};
use paramecia_arrow::{Arrow, Combinator, CombinatorTraceExt, Fanout, LiftResult, WithUnitCtx};
use paramecia_core::{DType, Device, Result, Tensor};
use paramecia_nn::Module;
use paramecia_tensor::residual_add::ResidualAddOp;
use paramecia_tensor::Tensor as TypedTensor;

use full_attention::{FullAttention, FullAttentionForwardGraph};
use linear_attention::{LinearAttention, LinearAttentionForwardGraph};
use moe::{MoeBlock, MoeForwardGraph, ResidualAddHiddenFlow};
use shape::Hidden3;

/// Helper function to check if a device is a GPU (CUDA, Metal, or Vulkan)
fn is_gpu_device(device: &Device) -> bool {
    matches!(
        device,
        Device::Cuda(_) | Device::Metal(_) | Device::Vulkan(_)
    )
}

/// Free function: select the causal mask for a layer's device in multi-GPU mode.
fn layer_mask<'a>(
    layer_idx: usize,
    is_multi: bool,
    ldm: &LayerDeviceMap,
    per_device_masks: &'a [(String, Tensor)],
    original_mask: Option<&'a Tensor>,
) -> Option<&'a Tensor> {
    if is_multi && !per_device_masks.is_empty() {
        let dev = ldm.device_for_layer(layer_idx);
        let key = format!("{:?}", dev);
        per_device_masks
            .iter()
            .find(|(k, _)| k == &key)
            .map(|(_, m)| m)
    } else {
        original_mask
    }
}

/// Free function: transfer tensor to a specific layer's device if needed.
fn transfer_to(tensor: Tensor, target: &Device) -> Result<Tensor> {
    if !tensor.device().same_device(target) {
        tensor.to_device(target)
    } else {
        Ok(tensor)
    }
}

// ============================================================================
// Layer implementation
// ============================================================================

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum AttentionLayer {
    Full(FullAttention),
    Linear(LinearAttention),
}

/// Mode for layer-level forward dispatch.
/// Controls attention mode and whether to collect MoE router statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LayerForwardMode {
    /// Standard inference: basic attention + basic MoE (no stats)
    Normal,
    /// Training/profiling: basic attention + MoE with router stats + optional timing
    WithStats,
    /// Speculative verification: state-materializing linear attention + MoE (discards stats)
    WithStateMaterialization,
}

#[derive(Debug)]
struct LayerWeights {
    attn: AttentionLayer,
    moe_block: MoeBlock,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

#[derive(Debug)]
struct LayerBlocksCtx<'a> {
    attn: &'a mut AttentionLayer,
    attn_norm: &'a mut RmsNorm,
    ffn_norm: &'a mut RmsNorm,
    moe_block: &'a mut MoeBlock,
    mask: Option<&'a Tensor>,
    offset: usize,
    mode: LayerForwardMode,
    input_dtype: DType,
}
struct LayerAttnPrepared {
    residual: TypedTensor<Hidden3>,
    h_normed: TypedTensor<Hidden3>,
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerAttnPrepareOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<LayerBlocksCtx<'a>> for LayerAttnPrepareOp {
    type In = TypedTensor<Hidden3>;
    type Out = Result<LayerAttnPrepared>;

    fn forward(&mut self, ctx: &mut LayerBlocksCtx<'a>, input: Self::In) -> Self::Out {
        let (h, residual) = Fanout::default().traced_forward(&mut (), input);
        let h_normed: TypedTensor<Hidden3> = ctx.attn_norm.forward(h.inner())?.try_into()?;
        Ok(LayerAttnPrepared { residual, h_normed })
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerAttnPrepareOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "LayerAttnPrepare",
            Graph::sequence(
                <Fanout<TypedTensor<Hidden3>> as Vis>::visualize(),
                Graph::custom_leaf("RmsNormForward"),
            ),
        )
        .with_output_type::<Result<LayerAttnPrepared>>()
    }
}

#[allow(dead_code)]
pub(super) struct LayerAttentionDispatchGraph;
#[primitive(property = Visualize)]
impl Vis for LayerAttentionDispatchGraph {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "LayerAttentionDispatchGraph",
            Graph::zip_custom(
                "AttentionDispatchPaths",
                vec![
                    ("full", <FullAttentionForwardGraph as Vis>::visualize()),
                    ("linear", LinearAttentionForwardGraph::graph()),
                    (
                        "linear_with_state_materialization",
                        Graph::sequence(
                            LinearAttentionForwardGraph::graph(),
                            Graph::custom_leaf("StateMaterialization"),
                        ),
                    ),
                ],
            ),
        )
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerAttnDispatchOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<LayerBlocksCtx<'a>> for LayerAttnDispatchOp {
    type In = LayerAttnPrepared;
    type Out = Result<(TypedTensor<Hidden3>, TypedTensor<Hidden3>)>;

    fn forward(&mut self, ctx: &mut LayerBlocksCtx<'a>, input: Self::In) -> Self::Out {
        let LayerAttnPrepared { residual, h_normed } = input;
        let attn_out: TypedTensor<Hidden3> = match (&mut *ctx.attn, ctx.mode) {
            (AttentionLayer::Full(attn), _) => {
                let out = attn.forward_typed(&h_normed, ctx.mask, ctx.offset)?;
                if out.inner().dtype() != ctx.input_dtype {
                    out.inner().to_dtype(ctx.input_dtype)?.try_into()?
                } else {
                    out
                }
            }
            (AttentionLayer::Linear(attn), LayerForwardMode::WithStateMaterialization) => {
                attn.forward_with_state_materialization_typed(&h_normed)?
            }
            (AttentionLayer::Linear(attn), _) => attn.forward_typed(&h_normed)?,
        };
        Ok((attn_out, residual))
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerAttnDispatchOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "LayerAttnDispatch",
            Graph::sequence(
                <LayerAttentionDispatchGraph as Vis>::visualize(),
                Graph::custom_leaf("CastToInputDtypeIfNeeded"),
            ),
        )
        .with_output_type::<Result<(TypedTensor<Hidden3>, TypedTensor<Hidden3>)>>()
    }
}

type LayerAttnPrepareResult = Result<LayerAttnPrepared>;
type LayerAttnDispatchLift = LiftResult<
    LayerAttnDispatchOp,
    LayerAttnPrepareResult,
    (TypedTensor<Hidden3>, TypedTensor<Hidden3>),
>;
type LayerAttnDispatchResult = Result<(TypedTensor<Hidden3>, TypedTensor<Hidden3>)>;
type LayerAttnResidualLift =
    LiftResult<WithUnitCtx<ResidualAddHiddenFlow>, LayerAttnDispatchResult, TypedTensor<Hidden3>>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct LayerAttnBlock(
    LayerAttnPrepareOp,
    LayerAttnDispatchLift,
    LayerAttnResidualLift,
);
impl LayerAttnBlock {
    fn new() -> Self {
        Self(
            LayerAttnPrepareOp,
            LiftResult::new(LayerAttnDispatchOp),
            LiftResult::new(WithUnitCtx::new(ResidualAddHiddenFlow::new(
                ResidualAddOp::default(),
            ))),
        )
    }
}

struct LayerFfnPrepared {
    ffn_residual: TypedTensor<Hidden3>,
    h_normed: TypedTensor<Hidden3>,
}

struct LayerFfnMoeOut {
    ffn_residual: TypedTensor<Hidden3>,
    moe_out: TypedTensor<Hidden3>,
    stats: Option<(Tensor, Tensor)>,
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerFfnPrepareOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<LayerBlocksCtx<'a>> for LayerFfnPrepareOp {
    type In = TypedTensor<Hidden3>;
    type Out = Result<LayerFfnPrepared>;

    fn forward(&mut self, ctx: &mut LayerBlocksCtx<'a>, input: Self::In) -> Self::Out {
        let (h, ffn_residual) = Fanout::default().traced_forward(&mut (), input);
        let h_normed: TypedTensor<Hidden3> = ctx.ffn_norm.forward(h.inner())?.try_into()?;
        Ok(LayerFfnPrepared {
            ffn_residual,
            h_normed,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerFfnPrepareOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "LayerFfnPrepare",
            Graph::sequence(
                <Fanout<TypedTensor<Hidden3>> as Vis>::visualize(),
                Graph::custom_leaf("RmsNormForward"),
            ),
        )
        .with_output_type::<Result<LayerFfnPrepared>>()
    }
}

#[allow(dead_code)]
pub(super) struct LayerMoeForwardGraph;
#[primitive(property = Visualize)]
impl Vis for LayerMoeForwardGraph {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "LayerMoeForwardGraph",
            <MoeForwardGraph as Vis>::visualize(),
        )
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerFfnMoeOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<LayerBlocksCtx<'a>> for LayerFfnMoeOp {
    type In = LayerFfnPrepared;
    type Out = Result<LayerFfnMoeOut>;

    fn forward(&mut self, ctx: &mut LayerBlocksCtx<'a>, input: Self::In) -> Self::Out {
        let LayerFfnPrepared {
            ffn_residual,
            h_normed,
        } = input;
        let (moe_out, stats) = if ctx.mode == LayerForwardMode::Normal {
            (ctx.moe_block.forward_typed(&h_normed)?, None)
        } else {
            let (out, router_stats) = ctx.moe_block.forward_with_stats_typed(&h_normed)?;
            let stats = if ctx.mode == LayerForwardMode::WithStats {
                Some(router_stats)
            } else {
                None
            };
            (out, stats)
        };
        Ok(LayerFfnMoeOut {
            ffn_residual,
            moe_out,
            stats,
        })
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerFfnMoeOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "LayerFfnMoe",
            Graph::zip_custom(
                "MoeForwardPaths",
                vec![
                    ("normal", <LayerMoeForwardGraph as Vis>::visualize()),
                    (
                        "with_stats_or_state",
                        Graph::sequence(
                            <LayerMoeForwardGraph as Vis>::visualize(),
                            Graph::custom_leaf("MaybeCaptureRouterStats"),
                        ),
                    ),
                ],
            ),
        )
        .with_output_type::<Result<LayerFfnMoeOut>>()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct LayerFfnResidualOp;
#[primitive(property = Arrow)]
impl<'a> Combinator<LayerBlocksCtx<'a>> for LayerFfnResidualOp {
    type In = LayerFfnMoeOut;
    type Out = Result<(TypedTensor<Hidden3>, Option<(Tensor, Tensor)>)>;

    fn forward(&mut self, _ctx: &mut LayerBlocksCtx<'a>, input: Self::In) -> Self::Out {
        let LayerFfnMoeOut {
            ffn_residual,
            moe_out,
            stats,
        } = input;
        let h: TypedTensor<Hidden3> = if moe_out.inner().dtype() != ffn_residual.inner().dtype() {
            moe_out
                .inner()
                .to_dtype(ffn_residual.inner().dtype())?
                .try_into()?
        } else {
            moe_out
        };
        let output: TypedTensor<Hidden3> = ResidualAddHiddenFlow::new(ResidualAddOp::default())
            .traced_forward(&mut (), (h, ffn_residual))?;
        Ok((output, stats))
    }
}
#[primitive(property = Visualize)]
impl Vis for LayerFfnResidualOp {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph(
            "LayerFfnResidual",
            Graph::sequence(
                Graph::custom_leaf("CastToResidualDtypeIfNeeded"),
                <ResidualAddHiddenFlow as Vis>::visualize(),
            ),
        )
        .with_output_type::<Result<(TypedTensor<Hidden3>, Option<(Tensor, Tensor)>)>>()
    }
}

type LayerFfnPrepareResult = Result<LayerFfnPrepared>;
type LayerFfnMoeLift = LiftResult<LayerFfnMoeOp, LayerFfnPrepareResult, LayerFfnMoeOut>;
type LayerFfnMoeResult = Result<LayerFfnMoeOut>;
type LayerFfnResidualLift = LiftResult<
    LayerFfnResidualOp,
    LayerFfnMoeResult,
    (TypedTensor<Hidden3>, Option<(Tensor, Tensor)>),
>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct LayerFfnBlock(LayerFfnPrepareOp, LayerFfnMoeLift, LayerFfnResidualLift);
impl LayerFfnBlock {
    fn new() -> Self {
        Self(
            LayerFfnPrepareOp,
            LiftResult::new(LayerFfnMoeOp),
            LiftResult::new(LayerFfnResidualOp),
        )
    }
}

type LayerFfnBlockLift = LiftResult<
    LayerFfnBlock,
    Result<TypedTensor<Hidden3>>,
    (TypedTensor<Hidden3>, Option<(Tensor, Tensor)>),
>;

#[derive(Inception)]
#[inception(properties = [Arrow, Visualize])]
struct LayerBlocks(LayerAttnBlock, LayerFfnBlockLift);
impl LayerBlocks {
    fn new() -> Self {
        Self(LayerAttnBlock::new(), LiftResult::new(LayerFfnBlock::new()))
    }
}

#[allow(dead_code)]
pub(super) struct LayerForwardGraph;
#[primitive(property = Visualize)]
impl Vis for LayerForwardGraph {
    fn visualize() -> Graph {
        Graph::wrap_custom_subgraph("LayerForwardGraph", <LayerBlocks as Vis>::visualize())
    }
}

impl LayerWeights {
    fn forward_inner_typed(
        &mut self,
        x: TypedTensor<Hidden3>,
        mask: Option<&Tensor>,
        offset: usize,
        mode: LayerForwardMode,
    ) -> Result<(TypedTensor<Hidden3>, Option<(Tensor, Tensor)>)> {
        let input_dtype = x.inner().dtype();
        let t_start = if mode == LayerForwardMode::WithStats {
            use std::sync::OnceLock;
            static PROFILE: OnceLock<bool> = OnceLock::new();
            let profile = *PROFILE.get_or_init(|| std::env::var("PARAMECIA_PROFILE").is_ok());
            if profile {
                Some(std::time::Instant::now())
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(feature = "vulkan")]
        if mode == LayerForwardMode::WithStats {
            paramecia_core::vulkan_backend::device::set_transfer_label("attn");
        }

        let mut ctx = LayerBlocksCtx {
            attn: &mut self.attn,
            attn_norm: &mut self.attn_norm,
            ffn_norm: &mut self.ffn_norm,
            moe_block: &mut self.moe_block,
            mask,
            offset,
            mode,
            input_dtype,
        };
        let mut flow = LayerBlocks::new();
        let (output, stats) = flow.traced_forward(&mut ctx, x)?;

        if let Some(t) = t_start {
            use std::sync::atomic::{AtomicU32, Ordering};
            static LAYER_COUNTER: AtomicU32 = AtomicU32::new(0);
            let layer_n = LAYER_COUNTER.fetch_add(1, Ordering::Relaxed);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            tracing::trace!(
                layer = layer_n % 40,
                total_ms = format_args!("{:.1}", ms),
                "Layer profile"
            );
        }

        Ok((output, stats))
    }

    /// Typed forward for canonical hidden state shape [B, N, S].
    /// Keeps layer execution entirely on typed paths.
    fn forward_typed(
        &mut self,
        x: TypedTensor<Hidden3>,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<TypedTensor<Hidden3>> {
        let (result, _) = self.forward_inner_typed(x, mask, offset, LayerForwardMode::Normal)?;
        Ok(result)
    }

    fn clear_cache(&mut self) {
        match &mut self.attn {
            AttentionLayer::Full(attn) => attn.clear_kv_cache(),
            AttentionLayer::Linear(attn) => attn.clear_state(),
        }
    }

    /// Truncate KV cache to a given sequence length.
    /// For full attention layers, this truncates the cache.
    /// For linear attention layers, this clears the state (can't truncate recurrent state).
    fn truncate_cache(&mut self, new_len: usize) {
        match &mut self.attn {
            AttentionLayer::Full(attn) => attn.truncate_kv_cache(new_len),
            AttentionLayer::Linear(attn) => attn.clear_state(), // Can't truncate recurrent state
        }
    }

    /// Create a snapshot of the current cache/state for speculative decoding rollback.
    /// For full attention: just records the current cache length (O(1)).
    /// For linear attention: copies state to backup buffers (reuses allocation).
    fn snapshot_cache(&mut self) -> Result<kv_cache::LayerSnapshot> {
        match &mut self.attn {
            AttentionLayer::Full(attn) => {
                let len = attn
                    .preallocated_cache
                    .as_ref()
                    .map(|c| c.seq_len)
                    .unwrap_or(0);
                Ok(kv_cache::LayerSnapshot::FullAttention { seq_len: len })
            }
            AttentionLayer::Linear(attn) => {
                attn.snapshot_state()?;
                Ok(kv_cache::LayerSnapshot::LinearAttention)
            }
        }
    }

    /// Restore cache/state from a snapshot after speculative decoding rejection.
    /// For full attention: just updates the seq_len pointer (O(1)).
    /// For linear attention: swaps backup to primary (O(1) pointer swap).
    fn restore_cache(&mut self, snapshot: kv_cache::LayerSnapshot) {
        match (&mut self.attn, snapshot) {
            (AttentionLayer::Full(attn), kv_cache::LayerSnapshot::FullAttention { seq_len }) => {
                if let Some(ref mut cache) = attn.preallocated_cache {
                    cache.seq_len = seq_len;
                }
            }
            (AttentionLayer::Linear(attn), kv_cache::LayerSnapshot::LinearAttention) => {
                attn.restore_state();
            }
            _ => {} // Mismatched types - shouldn't happen
        }
    }

    /// Initialize intermediate states buffer for verification with state slicing.
    /// Only applies to linear attention layers.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    fn init_intermediate_states(&mut self, seq_len: usize) {
        if let AttentionLayer::Linear(attn) = &mut self.attn {
            attn.init_intermediate_states(seq_len);
        }
    }

    /// Clear intermediate states buffer.
    fn clear_intermediate_states(&mut self) {
        if let AttentionLayer::Linear(attn) = &mut self.attn {
            attn.clear_intermediate_states();
        }
    }

    /// Restore to a specific intermediate state by index.
    /// For full attention: truncates KV cache to (base_len + index + 1).
    /// For linear attention: restores to the saved intermediate state.
    fn restore_to_intermediate_state(&mut self, index: usize, base_kv_len: usize) -> bool {
        match &mut self.attn {
            AttentionLayer::Full(attn) => {
                // For full attention, truncate KV cache to the position after index
                let new_len = base_kv_len + index + 1;
                attn.truncate_kv_cache(new_len);
                true
            }
            AttentionLayer::Linear(attn) => attn.restore_to_intermediate_state(index),
        }
    }

    /// Save the current cache/state for prefix caching.
    /// Returns a deep copy that can be restored later.
    fn save_cache_for_prefix(&self) -> PrefixCacheEntry {
        match &self.attn {
            AttentionLayer::Full(attn) => attn.save_kv_state(),
            AttentionLayer::Linear(attn) => attn.save_state_for_prefix(),
        }
    }

    /// Restore cache/state from a prefix cache entry.
    fn restore_cache_from_prefix(&mut self, entry: &PrefixCacheEntry) -> Result<()> {
        match &mut self.attn {
            AttentionLayer::Full(attn) => attn.restore_kv_state(entry),
            AttentionLayer::Linear(attn) => attn.restore_state_from_prefix(entry),
        }
    }
}
