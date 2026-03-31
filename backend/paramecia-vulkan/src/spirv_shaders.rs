//! Pre-compiled SPIR-V shader binaries for subgroup-optimized operations.
//!
//! These shaders use GL_KHR_shader_subgroup_* for warp-level reductions,
//! packed integer operations for efficient quantized matmuls,
//! and register-based tiling.

// ============================================================================
// Flash Attention Q8
// ============================================================================

pub const FLASH_ATTN_Q8_D64_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_q8_d64.spv"));
pub const FLASH_ATTN_Q8_D128_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_q8_d128.spv"));
pub const FLASH_ATTN_Q8_D256_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/flash_attn_q8_d256.spv"));

pub fn get_flash_attn_q8_spirv(head_dim: usize) -> Option<&'static [u8]> {
    match head_dim {
        64 => Some(FLASH_ATTN_Q8_D64_SPV),
        128 => Some(FLASH_ATTN_Q8_D128_SPV),
        256 => Some(FLASH_ATTN_Q8_D256_SPV),
        _ => None,
    }
}

// ============================================================================
// Quantized MatMul Q8_0
// ============================================================================

pub const MATMUL_Q8_0_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_q8_0.spv"));
pub const MATMUL_Q8_0_DP4A_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_q8_0_dp4a.spv"));

pub fn get_matmul_q8_0_spirv() -> &'static [u8] {
    MATMUL_Q8_0_SPV
}

pub fn get_matmul_q8_0_dp4a_spirv() -> &'static [u8] {
    MATMUL_Q8_0_DP4A_SPV
}

// ============================================================================
// Hierarchical Tiled Quantized MatMul (Phase 1-2: Q8_0 and K-quants)
// ============================================================================

pub const MATMUL_TILED_Q8_0_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q8_0.spv"));

pub fn get_matmul_tiled_q8_0_spirv() -> &'static [u8] {
    MATMUL_TILED_Q8_0_SPV
}

pub const MATMUL_TILED_Q8_0_PREFILL_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q8_0_prefill.spv"));

pub fn get_matmul_tiled_q8_0_prefill_spirv() -> &'static [u8] {
    MATMUL_TILED_Q8_0_PREFILL_SPV
}

// Phase 4: Optimized prefill variant (K-loop unrolling)
pub const MATMUL_TILED_Q8_0_PREFILL_OPT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q8_0_prefill_opt.spv"
));

pub fn get_matmul_tiled_q8_0_prefill_opt_spirv() -> &'static [u8] {
    MATMUL_TILED_Q8_0_PREFILL_OPT_SPV
}

// Phase 4: Vectorized loading variant
pub const MATMUL_TILED_Q8_0_PREFILL_VEC_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q8_0_prefill_vec.spv"
));

pub fn get_matmul_tiled_q8_0_prefill_vec_spirv() -> &'static [u8] {
    MATMUL_TILED_Q8_0_PREFILL_VEC_SPV
}

// Phase 4: Double buffering variant
pub const MATMUL_TILED_Q8_0_PREFILL_DB_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q8_0_prefill_db.spv"
));

pub fn get_matmul_tiled_q8_0_prefill_db_spirv() -> &'static [u8] {
    MATMUL_TILED_Q8_0_PREFILL_DB_SPV
}

// Phase 6: Hierarchical tiled coopmat variant
pub const MATMUL_TILED_Q8_0_PREFILL_COOPMAT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q8_0_prefill_coopmat.spv"
));

pub fn get_matmul_tiled_q8_0_prefill_coopmat_spirv() -> &'static [u8] {
    MATMUL_TILED_Q8_0_PREFILL_COOPMAT_SPV
}

pub const MATMUL_TILED_Q2_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q2_k.spv"));

pub fn get_matmul_tiled_q2_k_spirv() -> &'static [u8] {
    MATMUL_TILED_Q2_K_SPV
}

pub const MATMUL_TILED_Q3_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q3_k.spv"));

pub fn get_matmul_tiled_q3_k_spirv() -> &'static [u8] {
    MATMUL_TILED_Q3_K_SPV
}

pub const MATMUL_TILED_Q4_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q4_k.spv"));

pub fn get_matmul_tiled_q4_k_spirv() -> &'static [u8] {
    MATMUL_TILED_Q4_K_SPV
}

pub const MATMUL_TILED_Q5_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q5_k.spv"));

pub fn get_matmul_tiled_q5_k_spirv() -> &'static [u8] {
    MATMUL_TILED_Q5_K_SPV
}

pub const MATMUL_TILED_Q6_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q6_k.spv"));

pub fn get_matmul_tiled_q6_k_spirv() -> &'static [u8] {
    MATMUL_TILED_Q6_K_SPV
}

// Phase 3: Prefill-optimized variants (BM=64, BN=64)
pub const MATMUL_TILED_Q2_K_PREFILL_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q2_k_prefill.spv"));

pub fn get_matmul_tiled_q2_k_prefill_spirv() -> &'static [u8] {
    MATMUL_TILED_Q2_K_PREFILL_SPV
}

pub const MATMUL_TILED_Q3_K_PREFILL_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q3_k_prefill.spv"));

pub fn get_matmul_tiled_q3_k_prefill_spirv() -> &'static [u8] {
    MATMUL_TILED_Q3_K_PREFILL_SPV
}

pub const MATMUL_TILED_Q4_K_PREFILL_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q4_k_prefill.spv"));

pub fn get_matmul_tiled_q4_k_prefill_spirv() -> &'static [u8] {
    MATMUL_TILED_Q4_K_PREFILL_SPV
}

pub const MATMUL_TILED_Q5_K_PREFILL_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q5_k_prefill.spv"));

pub fn get_matmul_tiled_q5_k_prefill_spirv() -> &'static [u8] {
    MATMUL_TILED_Q5_K_PREFILL_SPV
}

pub const MATMUL_TILED_Q6_K_PREFILL_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_tiled_q6_k_prefill.spv"));

pub fn get_matmul_tiled_q6_k_prefill_spirv() -> &'static [u8] {
    MATMUL_TILED_Q6_K_PREFILL_SPV
}

// Phase 6: Hierarchical tiled coopmat variants for K-quants
pub const MATMUL_TILED_Q2_K_PREFILL_COOPMAT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q2_k_prefill_coopmat.spv"
));

pub fn get_matmul_tiled_q2_k_prefill_coopmat_spirv() -> &'static [u8] {
    MATMUL_TILED_Q2_K_PREFILL_COOPMAT_SPV
}

pub const MATMUL_TILED_Q3_K_PREFILL_COOPMAT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q3_k_prefill_coopmat.spv"
));

pub fn get_matmul_tiled_q3_k_prefill_coopmat_spirv() -> &'static [u8] {
    MATMUL_TILED_Q3_K_PREFILL_COOPMAT_SPV
}

pub const MATMUL_TILED_Q4_K_PREFILL_COOPMAT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q4_k_prefill_coopmat.spv"
));

pub fn get_matmul_tiled_q4_k_prefill_coopmat_spirv() -> &'static [u8] {
    MATMUL_TILED_Q4_K_PREFILL_COOPMAT_SPV
}

pub const MATMUL_TILED_Q5_K_PREFILL_COOPMAT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q5_k_prefill_coopmat.spv"
));

pub fn get_matmul_tiled_q5_k_prefill_coopmat_spirv() -> &'static [u8] {
    MATMUL_TILED_Q5_K_PREFILL_COOPMAT_SPV
}

pub const MATMUL_TILED_Q6_K_PREFILL_COOPMAT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/matmul_tiled_q6_k_prefill_coopmat.spv"
));

pub fn get_matmul_tiled_q6_k_prefill_coopmat_spirv() -> &'static [u8] {
    MATMUL_TILED_Q6_K_PREFILL_COOPMAT_SPV
}

// ============================================================================
// Quantized MatMul Q4_K
// ============================================================================

pub const MATMUL_Q4_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_q4_k.spv"));

pub fn get_matmul_q4_k_spirv() -> &'static [u8] {
    MATMUL_Q4_K_SPV
}

// ============================================================================
// Quantized MatMul Q2_K
// ============================================================================

pub const MATMUL_Q2_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_q2_k.spv"));

pub fn get_matmul_q2_k_spirv() -> &'static [u8] {
    MATMUL_Q2_K_SPV
}

// ============================================================================
// Quantized MatMul Q5_K
// ============================================================================

pub const MATMUL_Q5_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_q5_k.spv"));

pub fn get_matmul_q5_k_spirv() -> &'static [u8] {
    MATMUL_Q5_K_SPV
}

// ============================================================================
// Quantized MatMul Q6_K
// ============================================================================

pub const MATMUL_Q6_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_q6_k.spv"));

pub fn get_matmul_q6_k_spirv() -> &'static [u8] {
    MATMUL_Q6_K_SPV
}

// ============================================================================
// Matrix-Vector Multiplication (Optimized for decode phase m=1)
// ============================================================================

pub const MATMUL_VEC_Q8_0_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_vec_q8_0.spv"));
pub const MATMUL_VEC_Q4_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_vec_q4_k.spv"));
pub const MATMUL_VEC_Q5_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_vec_q5_k.spv"));
pub const MATMUL_VEC_Q6_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_vec_q6_k.spv"));
pub const MATMUL_VEC_Q3_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_vec_q3_k.spv"));
pub const MATMUL_VEC_Q2_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_vec_q2_k.spv"));

pub fn get_matmul_vec_spirv(name: &str) -> Option<&'static [u8]> {
    match name {
        "matmul_vec_q8_0" => Some(MATMUL_VEC_Q8_0_SPV),
        "matmul_vec_q4_k" => Some(MATMUL_VEC_Q4_K_SPV),
        "matmul_vec_q5_k" => Some(MATMUL_VEC_Q5_K_SPV),
        "matmul_vec_q6_k" => Some(MATMUL_VEC_Q6_K_SPV),
        "matmul_vec_q3_k" => Some(MATMUL_VEC_Q3_K_SPV),
        "matmul_vec_q2_k" => Some(MATMUL_VEC_Q2_K_SPV),
        _ => None,
    }
}

// ============================================================================
// Fused Matmul+Perturb kernels
// ============================================================================

pub const FUSED_MATMUL_Q8_0_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_matmul_q8_0.spv"));
pub const FUSED_MATMUL_Q4_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_matmul_q4_k.spv"));
pub const FUSED_MATMUL_Q2_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_matmul_q2_k.spv"));
pub const FUSED_MATMUL_Q3_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_matmul_q3_k.spv"));
pub const FUSED_MATMUL_Q5_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_matmul_q5_k.spv"));
pub const FUSED_MATMUL_Q6_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_matmul_q6_k.spv"));
pub const FUSED_MATMUL_BF16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_matmul_bf16.spv"));

pub fn get_fused_matmul_spirv(name: &str) -> Option<&'static [u8]> {
    match name {
        "fused_matmul_q8_0" => Some(FUSED_MATMUL_Q8_0_SPV),
        "fused_matmul_q4_k" => Some(FUSED_MATMUL_Q4_K_SPV),
        "fused_matmul_q2_k" => Some(FUSED_MATMUL_Q2_K_SPV),
        "fused_matmul_q3_k" => Some(FUSED_MATMUL_Q3_K_SPV),
        "fused_matmul_q5_k" => Some(FUSED_MATMUL_Q5_K_SPV),
        "fused_matmul_q6_k" => Some(FUSED_MATMUL_Q6_K_SPV),
        "fused_matmul_bf16" => Some(FUSED_MATMUL_BF16_SPV),
        _ => None,
    }
}

// ============================================================================
// QuZO Perturbation kernels
// ============================================================================

pub const PERTURB_Q8_0_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/perturb_q8_0.spv"));
pub const PERTURB_Q4_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/perturb_q4_k.spv"));
pub const PERTURB_Q2_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/perturb_q2_k.spv"));
pub const PERTURB_Q3_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/perturb_q3_k.spv"));
pub const PERTURB_Q5_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/perturb_q5_k.spv"));
pub const PERTURB_Q6_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/perturb_q6_k.spv"));
pub const PERTURB_BF16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/perturb_bf16.spv"));

pub const RESTORE_UPDATE_Q8_0_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/restore_update_q8_0.spv"));
pub const RESTORE_UPDATE_Q4_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/restore_update_q4_k.spv"));
pub const RESTORE_UPDATE_Q2_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/restore_update_q2_k.spv"));
pub const RESTORE_UPDATE_Q3_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/restore_update_q3_k.spv"));
pub const RESTORE_UPDATE_Q5_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/restore_update_q5_k.spv"));
pub const RESTORE_UPDATE_Q6_K_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/restore_update_q6_k.spv"));
pub const RESTORE_UPDATE_BF16_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/restore_update_bf16.spv"));

pub fn get_perturb_spirv(name: &str) -> Option<&'static [u8]> {
    match name {
        "perturb_q8_0" => Some(PERTURB_Q8_0_SPV),
        "perturb_q4_k" => Some(PERTURB_Q4_K_SPV),
        "perturb_q2_k" => Some(PERTURB_Q2_K_SPV),
        "perturb_q3_k" => Some(PERTURB_Q3_K_SPV),
        "perturb_q5_k" => Some(PERTURB_Q5_K_SPV),
        "perturb_q6_k" => Some(PERTURB_Q6_K_SPV),
        "perturb_bf16" => Some(PERTURB_BF16_SPV),
        "restore_update_q8_0" => Some(RESTORE_UPDATE_Q8_0_SPV),
        "restore_update_q4_k" => Some(RESTORE_UPDATE_Q4_K_SPV),
        "restore_update_q2_k" => Some(RESTORE_UPDATE_Q2_K_SPV),
        "restore_update_q3_k" => Some(RESTORE_UPDATE_Q3_K_SPV),
        "restore_update_q5_k" => Some(RESTORE_UPDATE_Q5_K_SPV),
        "restore_update_q6_k" => Some(RESTORE_UPDATE_Q6_K_SPV),
        "restore_update_bf16" => Some(RESTORE_UPDATE_BF16_SPV),
        _ => None,
    }
}

// ============================================================================
// GLA Step (Gated Linear Attention single-token autoregressive step)
// ============================================================================

pub const GLA_STEP_D64_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gla_step_d64.spv"));
pub const GLA_STEP_D128_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gla_step_d128.spv"));
pub const GLA_STEP_D256_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gla_step_d256.spv"));

pub fn get_gla_step_spirv(head_dim: usize) -> Option<&'static [u8]> {
    match head_dim {
        64 => Some(GLA_STEP_D64_SPV),
        128 => Some(GLA_STEP_D128_SPV),
        256 => Some(GLA_STEP_D256_SPV),
        _ => None,
    }
}

// ============================================================================
// DeltaNet Parallel (multi-token prefill)
// ============================================================================

pub const DELTA_NET_PARALLEL_D64_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/delta_net_parallel_d64.spv"));
pub const DELTA_NET_PARALLEL_D128_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/delta_net_parallel_d128.spv"));
pub const DELTA_NET_PARALLEL_D256_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/delta_net_parallel_d256.spv"));

pub fn get_delta_net_parallel_spirv(head_dim: usize) -> Option<&'static [u8]> {
    match head_dim {
        64 => Some(DELTA_NET_PARALLEL_D64_SPV),
        128 => Some(DELTA_NET_PARALLEL_D128_SPV),
        256 => Some(DELTA_NET_PARALLEL_D256_SPV),
        _ => None,
    }
}
