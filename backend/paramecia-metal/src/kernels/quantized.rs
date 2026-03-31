use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
    BF16,
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q2K => {
            // Each thread group has N_SIMDGROUP=2 simd groups, each processing N_DST=4 rows
            // Total rows per thread group = 2 * 4 = 8, so align must be 8
            let nth0 = 2;
            let nth1 = 32;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => {
            let nth0 = 4;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 2;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
        GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mv_f32_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(lhs, MTLResourceUsage::Read);
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_f32_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            src0,
            (src1, src1_offset),
            (dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}

/// Indexed MoE forward pass for quantized expert weights.
///
/// This kernel performs batched matrix-vector multiplication with dynamic expert
/// selection, enabling efficient Mixture-of-Experts inference on Metal.
///
/// # Arguments
/// * `weights` - Quantized expert weights [num_experts, n, k]
/// * `input` - Input activations [batch, input_dim1, k]  
/// * `ids` - Expert indices [batch, topk]
/// * `output` - Output buffer [batch, topk, n]
/// * `n` - Output dimension (rows per expert)
/// * `k` - Input dimension
/// * `batch` - Batch size
/// * `topk` - Number of experts per token
/// * `input_dim1` - Second dimension of input (either 1 or topk)
#[allow(clippy::too_many_arguments)]
pub fn call_indexed_moe_forward(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    weights: &Buffer,
    input: &Buffer,
    input_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    output: &Buffer,
    n: usize,
    k: usize,
    batch: usize,
    topk: usize,
    input_dim1: usize,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2_K_f32",
        GgmlDType::Q3K => "indexed_moe_forward_q3_K_f32",
        GgmlDType::Q4K => "indexed_moe_forward_q4_K_f32",
        GgmlDType::Q5K => "indexed_moe_forward_q5_K_f32",
        GgmlDType::Q6K => "indexed_moe_forward_q6_K_f32",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_f32",
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                format!("{:?}", dtype).leak(),
                "indexed_moe_forward",
            ))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Thread group configuration depends on quantization type
    // Each format has different N_DST (rows per simd group) and N_SIMDGROUP values
    let (rows_per_threadgroup, threads_per_tg) = match dtype {
        // Q2_K: N_DST=4, N_SIMDGROUP=2 → 8 rows, 64 threads
        GgmlDType::Q2K => (8usize, 64usize),
        // Q3_K: N_DST=2, N_SIMDGROUP=2 → 4 rows, 64 threads
        GgmlDType::Q3K => (4usize, 64usize),
        // Q4_K: N_DST=4, N_SIMDGROUP=1 → 4 rows, 32 threads
        GgmlDType::Q4K => (4usize, 32usize),
        // Q5_K: N_DST=2, N_SIMDGROUP=2 → 4 rows, 64 threads
        GgmlDType::Q5K => (4usize, 64usize),
        // Q6_K: N_DST=1, N_SIMDGROUP=2 → 2 rows, 64 threads
        GgmlDType::Q6K => (2usize, 64usize),
        // Q8_0: N_DST=4, N_SIMDGROUP=2 → 8 rows, 64 threads
        GgmlDType::Q8_0 => (8usize, 64usize),
        _ => (8usize, 64usize), // fallback, shouldn't reach here
    };
    let row_blocks = divide(n, rows_per_threadgroup);

    let thread_groups_count = MTLSize {
        width: row_blocks,
        height: batch,
        depth: topk,
    };

    let threads_per_threadgroup = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };

    set_params!(
        encoder,
        (
            weights,
            (input, input_offset),
            (ids, ids_offset),
            output,
            n as i32,
            k as i32,
            batch as i32,
            topk as i32,
            input_dim1 as i32
        )
    );

    encoder.use_resource(weights, MTLResourceUsage::Read);
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(ids, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Fused gate+up indexed MoE forward pass for quantized expert weights.
///
/// This computes BOTH gate and up projections in a single kernel pass,
/// reading the input tensor only once. This halves memory bandwidth
/// compared to calling indexed_moe_forward twice.
///
/// # Arguments
/// * `gate_weights` - Gate projection weights [num_experts, n, k] quantized
/// * `up_weights` - Up projection weights [num_experts, n, k] quantized
/// * `input` - Input tensor [batch, 1, k]
/// * `ids` - Expert indices [batch, topk]
/// * `gate_output` - Gate output buffer [batch, topk, n]
/// * `up_output` - Up output buffer [batch, topk, n]
#[allow(clippy::too_many_arguments)]
pub fn call_indexed_moe_gate_up(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    gate_weights: &Buffer,
    up_weights: &Buffer,
    input: &Buffer,
    input_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    gate_output: &Buffer,
    up_output: &Buffer,
    n: usize,
    k: usize,
    batch: usize,
    topk: usize,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        GgmlDType::Q2K => "indexed_moe_gate_up_q2_K_f32",
        GgmlDType::Q3K => "indexed_moe_gate_up_q3_K_f32",
        GgmlDType::Q4K => "indexed_moe_gate_up_q4_K_f32",
        GgmlDType::Q5K => "indexed_moe_gate_up_q5_K_f32",
        GgmlDType::Q6K => "indexed_moe_gate_up_q6_K_f32",
        GgmlDType::Q8_0 => "indexed_moe_gate_up_q8_0_f32",
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                format!("{:?}", dtype).leak(),
                "indexed_moe_gate_up",
            ))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Thread group configuration depends on quantization type
    let (rows_per_threadgroup, threads_per_tg) = match dtype {
        GgmlDType::Q2K => (8usize, 64usize),
        GgmlDType::Q3K => (4usize, 64usize),
        GgmlDType::Q4K => (4usize, 32usize),
        GgmlDType::Q5K => (4usize, 64usize),
        GgmlDType::Q6K => (2usize, 64usize),
        GgmlDType::Q8_0 => (8usize, 64usize),
        _ => (8usize, 64usize),
    };
    let row_blocks = divide(n, rows_per_threadgroup);

    let thread_groups_count = MTLSize {
        width: row_blocks,
        height: batch,
        depth: topk,
    };

    let threads_per_threadgroup = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };

    set_params!(
        encoder,
        (
            gate_weights,
            up_weights,
            (input, input_offset),
            (ids, ids_offset),
            gate_output,
            up_output,
            n as i32,
            k as i32,
            batch as i32,
            topk as i32
        )
    );

    encoder.use_resource(gate_weights, MTLResourceUsage::Read);
    encoder.use_resource(up_weights, MTLResourceUsage::Read);
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(ids, MTLResourceUsage::Read);
    encoder.use_resource(gate_output, MTLResourceUsage::Write);
    encoder.use_resource(up_output, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}
