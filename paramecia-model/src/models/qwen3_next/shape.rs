#![allow(dead_code)]

use paramecia_tensor::glowstick::{dyndims, num, Shape1, Shape2, Shape3, Shape4};
use std::ops::{Add, Mul};

use num::*;

dyndims! {
    B: _B,
    N: _N,
    Sx2: ConcatHidden,
    Lq: QueryLength,
    Lk: KeyValueLength,
    P: PosCacheLength,
    R: RotaryPairDim,
    T: FlatTok,
    Tk: DynTopK,
    M: DynBlockCount,
    C: ChunkLength,
    Ch: ConvHistoryLength
}

type U248320 = <<U248 as Mul<U1000>>::Output as Add<U320>>::Output;
type U151936 = <<<<U151 as Mul<U1000>>::Output as Add<<U9 as Mul<U100>>::Output>>::Output as Add<
    <U3 as Mul<U10>>::Output,
>>::Output as Add<U6>>::Output;
type U2560 = <<U2 as Mul<U1024>>::Output as Add<U512>>::Output;
type U3072 = <U3 as Mul<U1024>>::Output;
type U3584 = <<U3 as Mul<U1024>>::Output as Add<U512>>::Output;
type U5120 = <U5 as Mul<U1024>>::Output;
type U6144 = <U6 as Mul<U1024>>::Output;
type U9216 = <U9 as Mul<U1024>>::Output;
type U12288 = <U12 as Mul<U1024>>::Output;
type U17408 = <U17 as Mul<U1024>>::Output;

// Selected architecture profile.
// If no architecture feature is selected, default to qwen3.5-0.8B.
#[cfg(feature = "qwen3next_80b_a3b")]
mod arch_cfg {
    use super::*;
    pub type V = U151936;
    pub type A = U16;
    pub type K = U2;
    pub type S = U2048;
    pub type I = U5120;
    pub type DInner = U4096;
    pub type DtRank = U32;
    pub type E = U512;
    pub type TopK = U10;
    pub type SI = U512;
}

#[cfg(feature = "qwen35moe_35b_a3b")]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U16;
    pub type K = U2;
    pub type S = U2048;
    pub type I = U5120;
    pub type DInner = U4096;
    pub type DtRank = U32;
    pub type E = U256;
    pub type TopK = U8;
    pub type SI = U512;
}

#[cfg(feature = "qwen35moe_122b_a10b")]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U32;
    pub type K = U2;
    pub type S = U3072;
    pub type I = U5120;
    pub type DInner = U8192;
    pub type DtRank = U64;
    pub type E = U256;
    pub type TopK = U8;
    pub type SI = U1024;
}

#[cfg(feature = "qwen35moe_397b_a17b")]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U32;
    pub type K = U2;
    pub type S = U4096;
    pub type I = U5120;
    pub type DInner = U8192;
    pub type DtRank = U64;
    pub type E = U512;
    pub type TopK = U10;
    pub type SI = U1024;
}

#[cfg(feature = "qwen35_4b")]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U16;
    pub type K = U4;
    pub type S = U2560;
    pub type I = U9216;
    pub type DInner = U4096;
    pub type DtRank = U32;
    pub type E = U1;
    pub type TopK = U1;
    pub type SI = U9216;
}

#[cfg(feature = "qwen35_2b")]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U8;
    pub type K = U2;
    pub type S = U2048;
    pub type I = U6144;
    pub type DInner = U2048;
    pub type DtRank = U16;
    pub type E = U1;
    pub type TopK = U1;
    pub type SI = U6144;
}

#[cfg(feature = "qwen35_9b")]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U16;
    pub type K = U4;
    pub type S = U4096;
    pub type I = U12288;
    pub type DInner = U4096;
    pub type DtRank = U32;
    pub type E = U1;
    pub type TopK = U1;
    pub type SI = U12288;
}

#[cfg(feature = "qwen35_27b")]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U24;
    pub type K = U4;
    pub type S = U5120;
    pub type I = U17408;
    pub type DInner = U6144;
    pub type DtRank = U48;
    pub type E = U1;
    pub type TopK = U1;
    pub type SI = U17408;
}

#[cfg(feature = "qwen35_0p8b")]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U8;
    pub type K = U2;
    pub type S = U1024;
    pub type I = U3584;
    pub type DInner = U2048;
    pub type DtRank = U16;
    pub type E = U1;
    pub type TopK = U1;
    pub type SI = U3584;
}

#[cfg(not(any(
    feature = "qwen3next_80b_a3b",
    feature = "qwen35moe_35b_a3b",
    feature = "qwen35moe_122b_a10b",
    feature = "qwen35moe_397b_a17b",
    feature = "qwen35_0p8b",
    feature = "qwen35_2b",
    feature = "qwen35_4b",
    feature = "qwen35_9b",
    feature = "qwen35_27b"
)))]
mod arch_cfg {
    use super::*;
    pub type V = U248320;
    pub type A = U8;
    pub type K = U2;
    pub type S = U1024;
    pub type I = U3584;
    pub type DInner = U2048;
    pub type DtRank = U16;
    pub type E = U1;
    pub type TopK = U1;
    pub type SI = U3584;
}

// Core model dimensions
pub type V = arch_cfg::V; // Vocabulary
pub type H = U256; // Head-dim
pub type A = arch_cfg::A; // Query Heads
pub type K = arch_cfg::K; // Key-Value Heads
pub type S = arch_cfg::S; // Hidden Size
pub type I = arch_cfg::I; // Intermediate Dim

// SSM / Linear Attention dimensions
pub type DInner = arch_cfg::DInner; // ssm_d_inner = num_v_heads * head_v_dim
pub type DState = U128; // ssm_d_state (= head_k_dim = head_v_dim)
pub type NGroups = U16; // ssm_n_groups (= num_k_heads for linear attn)
pub type DtRank = arch_cfg::DtRank; // ssm_dt_rank (= num_v_heads for linear attn)

// MoE dimensions
pub type E = arch_cfg::E; // num_experts (or 1 for dense)
pub type TopK = arch_cfg::TopK; // num_experts_per_tok
pub type SI = arch_cfg::SI; // Expert intermediate dim

// Derived products
pub type H2 = <H as Mul<U2>>::Output; // H*2 = 512 (gated Q head dim)
pub type AH = <A as Mul<H>>::Output; // A*H
pub type AH2 = <AH as Mul<U2>>::Output; // A*H*2 (gated Q output)
pub type S2 = <S as Mul<U2>>::Output; // S*2 (MTP fc input: embed + hidden concat)
pub type KH = <K as Mul<H>>::Output; // K*H
pub type BaDim = <DtRank as Mul<U2>>::Output; // DtRank*2 (beta+alpha)

// Fused linear-attention projection output dims.
// q = NGroups*DState, k = NGroups*DState, v = DInner (= DtRank*DState).
pub type QkDim = <<NGroups as Mul<DState>>::Output as Mul<U2>>::Output;
pub type QkvDim = <QkDim as Add<DInner>>::Output; // q+k+v
pub type QkvzDim = <QkvDim as Add<DInner>>::Output; // q+k+v+z

// RoPE
pub type NRot = U64; // partial RoPE dims

// Conv
pub type ConvKernel = U4; // conv_kernel_size

// ============================================================================
// Canonical shape aliases (cross-module)
// ============================================================================

// Hidden-state shapes
pub type Hidden3 = Shape3<B, N, S>;
pub type HiddenFlat2 = Shape2<T, S>;

// Full/MTP attention projection shapes
pub type QProj3 = Shape3<B, N, AH2>;
pub type KProj3 = Shape3<B, N, KH>;
pub type VProj3 = Shape3<B, N, KH>;

// Attention core shapes
pub type AttnQPre4 = Shape4<B, Lq, A, H>;
pub type AttnKPre4 = Shape4<B, Lk, K, H>;
pub type AttnVPre4 = Shape4<B, Lk, K, H>;
pub type AttnQ4 = Shape4<B, A, Lq, H>;
pub type AttnK4 = Shape4<B, K, Lk, H>;
pub type AttnV4 = Shape4<B, K, Lk, H>;
pub type AttnScores4 = Shape4<B, A, Lq, Lk>;
pub type AttnOut4 = Shape4<B, A, Lq, H>;

// Linear-attention core shapes
pub type LinearQkv3 = Shape3<B, N, QkvDim>;
pub type LinearQ4 = Shape4<B, Lq, NGroups, DState>;
pub type LinearK4 = Shape4<B, Lq, NGroups, DState>;
pub type LinearV4 = Shape4<B, Lq, DtRank, DState>;
pub type LinearGate3 = Shape3<B, Lq, DtRank>;
pub type LinearState4 = Shape4<B, DtRank, DState, DState>;
pub type LinearConvState3 = Shape3<B, QkvDim, Ch>;

// MoE routing/dispatch shapes
pub type Router3 = Shape3<B, N, E>;
pub type Router2 = Shape2<T, E>;
pub type TopWeights3 = Shape3<B, N, Tk>;
pub type TopIndices3 = Shape3<B, N, Tk>;
pub type TopWeights2 = Shape2<T, Tk>;
pub type TopIndices2 = Shape2<T, Tk>;
pub type BlockMults1 = Shape1<M>;

// RoPE cache shapes
pub type RopeCache2 = Shape2<P, R>;
