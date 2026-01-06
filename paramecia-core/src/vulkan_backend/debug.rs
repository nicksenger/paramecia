//! Environment variable controls for Vulkan GPU vs CPU fallback.
//!
//! Set any of these env vars to "1" to force CPU fallback for that operation:
//!
//! - `VK_CPU_QMATMUL`      - All quantized matmul (forces CPU for QTensor::vulkan_fwd)
//! - `VK_CPU_QMATMUL_Q8`   - Q8_0 quantized matmul only
//! - `VK_CPU_QMATMUL_Q4K`  - Q4_K quantized matmul only
//! - `VK_CPU_QMATMUL_Q5K`  - Q5_K quantized matmul only
//! - `VK_CPU_QMATMUL_Q6K`  - Q6_K quantized matmul only
//! - `VK_CPU_QMATMUL_Q2K`  - Q2_K quantized matmul only
//! - `VK_CPU_GEMM`         - Standard (non-quantized) matrix multiply
//! - `VK_CPU_UNARY`        - Unary ops (neg, relu, silu, etc.)
//! - `VK_CPU_BINARY`       - Binary ops (add, mul, etc.)
//! - `VK_CPU_REDUCE`       - Reduce ops (sum, max, argmax, etc.)
//! - `VK_CPU_CAST`         - Dtype cast ops
//! - `VK_CPU_CUSTOMOP`     - All custom ops (RmsNorm, etc.)
//! - `VK_CPU_AFFINE`       - Affine, powf, elu ops
//! - `VK_CPU_COPY`         - Copy ops (copy_strided_src, copy2d)
//! - `VK_CPU_INDEX`        - Index ops (index_select, gather)
//! - `VK_CPU_WHERE`        - Where/conditional ops
//! - `VK_CPU_CONST`        - const_set (fill with scalar) and try_clone (buffer copy)
//! - `VK_CPU_DELTANET_STEP`    - DeltaNet autoregressive step (gla_step shader)
//! - `VK_CPU_DELTANET_PARALLEL`- DeltaNet parallel prefill (delta_net_parallel shader)
//! - `VK_CPU_DELTANET_MTP`     - DeltaNet multi-token update
//! - `VK_CPU_CONV1D`           - Depthwise conv1d
//! - `VK_CPU_L2NORM`           - L2 normalize + scale
//! - `VK_CPU_SWIGLU`           - Fused SwiGLU
//! - `VK_CPU_GATED_RMSNORM`    - Gated RMS norm
//! - `VK_CPU_ROPE`             - All RoPE variants (rope, rope_i, rope_thd)
//! - `VK_CPU_QUANTIZE`         - Q8_0 GPU quantize
//! - `VK_CPU_MOE`             - All indexed MoE forward (forces CPU for all dtypes)
//! - `VK_CPU_MOE_Q8`          - Q8_0 indexed MoE forward only
//! - `VK_CPU_MOE_Q4K`         - Q4_K indexed MoE forward only
//! - `VK_CPU_MOE_Q5K`         - Q5_K indexed MoE forward only
//! - `VK_CPU_MOE_Q6K`         - Q6_K indexed MoE forward only
//! - `VK_CPU_MOE_Q3K`         - Q3_K indexed MoE forward only
//! - `VK_CPU_MOE_Q2K`         - Q2_K indexed MoE forward only
//! - `VK_CPU_FLASH_Q8`        - Flash attention Q8 (fallback to dequant + standard attention)
//! - `VK_DISABLE_BATCH`        - Disable command batching (sync after every op)
//! - `VK_CPU_ALL`          - Force ALL ops to CPU fallback

use std::sync::OnceLock;
use tracing::debug;

struct VulkanDebugFlags {
    cpu_all: bool,
    cpu_qmatmul: bool,
    cpu_qmatmul_q8: bool,
    cpu_qmatmul_q4k: bool,
    cpu_qmatmul_q5k: bool,
    cpu_qmatmul_q6k: bool,
    cpu_qmatmul_q2k: bool,
    cpu_gemm: bool,
    cpu_unary: bool,
    cpu_binary: bool,
    cpu_reduce: bool,
    cpu_cast: bool,
    cpu_customop: bool,
    cpu_affine: bool,
    cpu_copy: bool,
    cpu_index: bool,
    cpu_where: bool,
    cpu_const: bool,
    cpu_deltanet_step: bool,
    cpu_deltanet_parallel: bool,
    cpu_deltanet_mtp: bool,
    cpu_conv1d: bool,
    cpu_l2norm: bool,
    cpu_swiglu: bool,
    cpu_gated_rmsnorm: bool,
    cpu_rope: bool,
    cpu_quantize: bool,
    cpu_moe: bool,
    cpu_moe_q8: bool,
    cpu_moe_q4k: bool,
    cpu_moe_q5k: bool,
    cpu_moe_q6k: bool,
    cpu_moe_q3k: bool,
    cpu_moe_q2k: bool,
    cpu_flash_q8: bool,
    disable_batch: bool,
    trace: bool,
    validate: u32, // number of ops to validate (0 = disabled)
}

fn env_is_set(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

static FLAGS: OnceLock<VulkanDebugFlags> = OnceLock::new();

fn flags() -> &'static VulkanDebugFlags {
    FLAGS.get_or_init(|| {
        let f = VulkanDebugFlags {
            cpu_all: env_is_set("VK_CPU_ALL"),
            cpu_qmatmul: env_is_set("VK_CPU_QMATMUL"),
            cpu_qmatmul_q8: env_is_set("VK_CPU_QMATMUL_Q8"),
            cpu_qmatmul_q4k: env_is_set("VK_CPU_QMATMUL_Q4K"),
            cpu_qmatmul_q5k: env_is_set("VK_CPU_QMATMUL_Q5K"),
            cpu_qmatmul_q6k: env_is_set("VK_CPU_QMATMUL_Q6K"),
            cpu_qmatmul_q2k: env_is_set("VK_CPU_QMATMUL_Q2K"),
            cpu_gemm: env_is_set("VK_CPU_GEMM"),
            cpu_unary: env_is_set("VK_CPU_UNARY"),
            cpu_binary: env_is_set("VK_CPU_BINARY"),
            cpu_reduce: env_is_set("VK_CPU_REDUCE"),
            cpu_cast: env_is_set("VK_CPU_CAST"),
            cpu_customop: env_is_set("VK_CPU_CUSTOMOP"),
            cpu_affine: env_is_set("VK_CPU_AFFINE"),
            cpu_copy: env_is_set("VK_CPU_COPY"),
            cpu_index: env_is_set("VK_CPU_INDEX"),
            cpu_where: env_is_set("VK_CPU_WHERE"),
            cpu_const: env_is_set("VK_CPU_CONST"),
            cpu_deltanet_step: env_is_set("VK_CPU_DELTANET_STEP"),
            cpu_deltanet_parallel: env_is_set("VK_CPU_DELTANET_PARALLEL"),
            cpu_deltanet_mtp: env_is_set("VK_CPU_DELTANET_MTP"),
            cpu_conv1d: env_is_set("VK_CPU_CONV1D"),
            cpu_l2norm: env_is_set("VK_CPU_L2NORM"),
            cpu_swiglu: env_is_set("VK_CPU_SWIGLU"),
            cpu_gated_rmsnorm: env_is_set("VK_CPU_GATED_RMSNORM"),
            cpu_rope: env_is_set("VK_CPU_ROPE"),
            cpu_quantize: env_is_set("VK_CPU_QUANTIZE"),
            cpu_moe: env_is_set("VK_CPU_MOE"),
            cpu_moe_q8: env_is_set("VK_CPU_MOE_Q8"),
            cpu_moe_q4k: env_is_set("VK_CPU_MOE_Q4K"),
            cpu_moe_q5k: env_is_set("VK_CPU_MOE_Q5K"),
            cpu_moe_q6k: env_is_set("VK_CPU_MOE_Q6K"),
            cpu_moe_q3k: env_is_set("VK_CPU_MOE_Q3K"),
            cpu_moe_q2k: env_is_set("VK_CPU_MOE_Q2K"),
            cpu_flash_q8: env_is_set("VK_CPU_FLASH_Q8"),
            disable_batch: env_is_set("VK_DISABLE_BATCH"),
            trace: env_is_set("VK_TRACE"),
            validate: std::env::var("VK_VALIDATE")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0),
        };
        // Log which overrides are active
        let mut active = Vec::new();
        if f.cpu_all {
            active.push("VK_CPU_ALL");
        }
        if f.cpu_qmatmul {
            active.push("VK_CPU_QMATMUL");
        }
        if f.cpu_qmatmul_q8 {
            active.push("VK_CPU_QMATMUL_Q8");
        }
        if f.cpu_qmatmul_q4k {
            active.push("VK_CPU_QMATMUL_Q4K");
        }
        if f.cpu_qmatmul_q5k {
            active.push("VK_CPU_QMATMUL_Q5K");
        }
        if f.cpu_qmatmul_q6k {
            active.push("VK_CPU_QMATMUL_Q6K");
        }
        if f.cpu_qmatmul_q2k {
            active.push("VK_CPU_QMATMUL_Q2K");
        }
        if f.cpu_gemm {
            active.push("VK_CPU_GEMM");
        }
        if f.cpu_unary {
            active.push("VK_CPU_UNARY");
        }
        if f.cpu_binary {
            active.push("VK_CPU_BINARY");
        }
        if f.cpu_reduce {
            active.push("VK_CPU_REDUCE");
        }
        if f.cpu_cast {
            active.push("VK_CPU_CAST");
        }
        if f.cpu_customop {
            active.push("VK_CPU_CUSTOMOP");
        }
        if f.cpu_affine {
            active.push("VK_CPU_AFFINE");
        }
        if f.cpu_copy {
            active.push("VK_CPU_COPY");
        }
        if f.cpu_index {
            active.push("VK_CPU_INDEX");
        }
        if f.cpu_where {
            active.push("VK_CPU_WHERE");
        }
        if f.cpu_const {
            active.push("VK_CPU_CONST");
        }
        if f.cpu_deltanet_step {
            active.push("VK_CPU_DELTANET_STEP");
        }
        if f.cpu_deltanet_parallel {
            active.push("VK_CPU_DELTANET_PARALLEL");
        }
        if f.cpu_deltanet_mtp {
            active.push("VK_CPU_DELTANET_MTP");
        }
        if f.cpu_conv1d {
            active.push("VK_CPU_CONV1D");
        }
        if f.cpu_l2norm {
            active.push("VK_CPU_L2NORM");
        }
        if f.cpu_swiglu {
            active.push("VK_CPU_SWIGLU");
        }
        if f.cpu_gated_rmsnorm {
            active.push("VK_CPU_GATED_RMSNORM");
        }
        if f.cpu_rope {
            active.push("VK_CPU_ROPE");
        }
        if f.cpu_quantize {
            active.push("VK_CPU_QUANTIZE");
        }
        if f.cpu_moe {
            active.push("VK_CPU_MOE");
        }
        if f.cpu_moe_q8 {
            active.push("VK_CPU_MOE_Q8");
        }
        if f.cpu_moe_q4k {
            active.push("VK_CPU_MOE_Q4K");
        }
        if f.cpu_moe_q5k {
            active.push("VK_CPU_MOE_Q5K");
        }
        if f.cpu_moe_q6k {
            active.push("VK_CPU_MOE_Q6K");
        }
        if f.cpu_moe_q3k {
            active.push("VK_CPU_MOE_Q3K");
        }
        if f.cpu_moe_q2k {
            active.push("VK_CPU_MOE_Q2K");
        }
        if f.cpu_flash_q8 {
            active.push("VK_CPU_FLASH_Q8");
        }
        if f.disable_batch {
            active.push("VK_DISABLE_BATCH");
        }
        if !active.is_empty() {
            debug!("[vulkan debug] CPU fallback active: {}", active.join(", "));
        }
        f
    })
}

/// Should quantized matmul for this dtype use CPU fallback?
pub fn force_cpu_qmatmul(dtype_name: &str) -> bool {
    let f = flags();
    if f.cpu_all || f.cpu_qmatmul {
        return true;
    }
    match dtype_name {
        "matmul_q8_0" => f.cpu_qmatmul_q8,
        "matmul_q4_k" => f.cpu_qmatmul_q4k,
        "matmul_q5_k" => f.cpu_qmatmul_q5k,
        "matmul_q6_k" => f.cpu_qmatmul_q6k,
        "matmul_q2_k" => f.cpu_qmatmul_q2k,
        _ => false,
    }
}

pub fn force_cpu_gemm() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_gemm
}

pub fn force_cpu_unary() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_unary
}

pub fn force_cpu_binary() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_binary
}

pub fn force_cpu_reduce() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_reduce
}

pub fn force_cpu_cast() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_cast
}

pub fn force_cpu_customop() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_customop
}

pub fn force_cpu_affine() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_affine
}

pub fn force_cpu_copy() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_copy
}

pub fn force_cpu_index() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_index
}

pub fn force_cpu_where() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_where
}

pub fn force_cpu_const() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_const
}

pub fn force_cpu_deltanet_step() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_deltanet_step
}

pub fn force_cpu_deltanet_parallel() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_deltanet_parallel
}

pub fn force_cpu_deltanet_mtp() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_deltanet_mtp
}

pub fn force_cpu_conv1d() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_conv1d
}

pub fn force_cpu_l2norm() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_l2norm
}

pub fn force_cpu_swiglu() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_swiglu
}

pub fn force_cpu_gated_rmsnorm() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_gated_rmsnorm
}

pub fn force_cpu_rope() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_rope
}

pub fn force_cpu_quantize() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_quantize
}

/// Should indexed MoE forward for this dtype use CPU fallback?
pub fn force_cpu_moe(dtype: crate::quantized::GgmlDType) -> bool {
    use crate::quantized::GgmlDType;
    let f = flags();
    if f.cpu_all || f.cpu_moe {
        return true;
    }
    match dtype {
        GgmlDType::Q8_0 => f.cpu_moe_q8,
        GgmlDType::Q4K => f.cpu_moe_q4k,
        GgmlDType::Q5K => f.cpu_moe_q5k,
        GgmlDType::Q6K => f.cpu_moe_q6k,
        GgmlDType::Q3K => f.cpu_moe_q3k,
        GgmlDType::Q2K => f.cpu_moe_q2k,
        _ => false,
    }
}

/// Should flash_attn_q8 use CPU fallback (dequant + standard attention)?
pub fn force_cpu_flash_q8() -> bool {
    let f = flags();
    f.cpu_all || f.cpu_flash_q8
}

pub fn disable_batch() -> bool {
    flags().disable_batch
}

pub fn trace_ops() -> bool {
    flags().trace
}

/// Returns the validation count (how many ops to validate GPU vs CPU). 0 = disabled.
pub fn validate_count() -> u32 {
    flags().validate
}
