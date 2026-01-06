//! Metal kernel bindings for QuZO perturbation and scale operations.

use crate::utils::EncoderProvider;
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, GgmlDType, Kernels, MetalKernelError, Source,
};
use objc2_metal::{MTLResourceUsage, MTLSize};

/// Call the perturb_weights Metal kernel for quantized tensors.
///
/// Applies stochastic rounding perturbation: w' = Q(w ± ε * z)
///
/// # Arguments
/// * `dtype` - Quantization format (Q8_0, Q4K, Q2K, Q3K, Q5K, Q6K, BF16)
/// * `blocks_buf` - Mutable buffer containing quantized blocks
/// * `perturbation_buf` - f32 perturbation values
/// * `num_blocks` - Number of quantization blocks (or num_elements for BF16)
/// * `epsilon` - Perturbation magnitude
/// * `seed` - RNG seed for stochastic rounding
/// * `add` - 1 for add, 0 for subtract
#[allow(clippy::too_many_arguments)]
pub fn call_perturb_weights_metal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    blocks_buf: &Buffer,
    perturbation_buf: &Buffer,
    num_blocks: usize,
    epsilon: f32,
    seed: u64,
    add: bool,
) -> Result<(), MetalKernelError> {
    let (kernel_name, threads_per_group): (&str, usize) = match dtype {
        GgmlDType::Q8_0 => ("perturb_q8_0", 32),
        GgmlDType::Q4K => ("perturb_q4_K", 128),
        GgmlDType::Q2K => ("perturb_q2_K", 64),
        GgmlDType::Q3K => ("perturb_q3_K", 64),
        GgmlDType::Q5K => ("perturb_q5_K", 64),
        GgmlDType::Q6K => ("perturb_q6_K", 64),
        GgmlDType::F16 => ("perturb_bf16", 256), // BF16 uses same thread layout
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "perturb_weights not supported for {:?}",
                dtype
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::QuzoPerturb, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let num_blocks_i32 = num_blocks as i32;
    let add_i32: i32 = if add { 1 } else { 0 };

    set_params!(
        encoder,
        (
            blocks_buf,
            perturbation_buf,
            num_blocks_i32,
            epsilon,
            seed,
            add_i32
        )
    );

    let thread_group_count = MTLSize {
        width: num_blocks,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(
        blocks_buf,
        MTLResourceUsage::Read.union(MTLResourceUsage::Write),
    );
    encoder.use_resource(perturbation_buf, MTLResourceUsage::Read);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Call the restore_and_update Metal kernel for quantized tensors.
///
/// Fuses restore (from -ε to 0) and gradient update into a single kernel pass.
///
/// # Arguments
/// * `dtype` - Quantization format
/// * `blocks_buf` - Mutable buffer containing quantized blocks
/// * `perturbation_buf` - f32 perturbation values
/// * `num_blocks` - Number of quantization blocks (or num_elements for BF16)
/// * `restore_epsilon` - Epsilon for restore step
/// * `update_scale` - Scale for gradient update (η·μ/n)
/// * `restore_seed` - Seed for restore perturbation
/// * `update_seed` - Seed for update (must differ from restore_seed!)
#[allow(clippy::too_many_arguments)]
pub fn call_restore_and_update_metal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    blocks_buf: &Buffer,
    perturbation_buf: &Buffer,
    num_blocks: usize,
    restore_epsilon: f32,
    update_scale: f32,
    restore_seed: u64,
    update_seed: u64,
) -> Result<(), MetalKernelError> {
    let (kernel_name, threads_per_group): (&str, usize) = match dtype {
        GgmlDType::Q8_0 => ("restore_and_update_q8_0", 32),
        GgmlDType::Q4K => ("restore_and_update_q4_K", 128),
        GgmlDType::Q2K => ("restore_and_update_q2_K", 64),
        GgmlDType::Q3K => ("restore_and_update_q3_K", 64),
        GgmlDType::Q5K => ("restore_and_update_q5_K", 64),
        GgmlDType::Q6K => ("restore_and_update_q6_K", 64),
        GgmlDType::F16 => ("restore_and_update_bf16", 256),
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "restore_and_update not supported for {:?}",
                dtype
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::QuzoPerturb, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let num_blocks_i32 = num_blocks as i32;

    set_params!(
        encoder,
        (
            blocks_buf,
            perturbation_buf,
            num_blocks_i32,
            restore_epsilon,
            update_scale,
            restore_seed,
            update_seed
        )
    );

    let thread_group_count = MTLSize {
        width: num_blocks,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(
        blocks_buf,
        MTLResourceUsage::Read.union(MTLResourceUsage::Write),
    );
    encoder.use_resource(perturbation_buf, MTLResourceUsage::Read);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Call the extract_scales Metal kernel.
///
/// Extracts scale factors from quantized blocks into an f32 buffer.
///
/// # Arguments
/// * `dtype` - Quantization format
/// * `blocks_buf` - Buffer containing quantized blocks
/// * `output_buf` - Output f32 buffer for scales (one per block)
/// * `num_blocks` - Number of quantization blocks
#[allow(clippy::too_many_arguments)]
pub fn call_extract_scales_metal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    blocks_buf: &Buffer,
    output_buf: &Buffer,
    num_blocks: usize,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        GgmlDType::Q8_0 => "extract_scales_q8_0",
        GgmlDType::Q4K => "extract_scales_q4_K",
        GgmlDType::Q2K => "extract_scales_q2_K",
        GgmlDType::Q3K => "extract_scales_q3_K",
        GgmlDType::Q5K => "extract_scales_q5_K",
        GgmlDType::Q6K => "extract_scales_q6_K",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "extract_scales not supported for {:?}",
                dtype
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::QuzoPerturb, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let num_blocks_i32 = num_blocks as i32;

    set_params!(encoder, (blocks_buf, output_buf, num_blocks_i32));

    let threads_per_group = 256usize.min(pipeline.max_total_threads_per_threadgroup() as usize);
    let num_groups = (num_blocks + threads_per_group - 1) / threads_per_group;

    let thread_group_count = MTLSize {
        width: num_groups,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(blocks_buf, MTLResourceUsage::Read);
    encoder.use_resource(output_buf, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Call the modify_block_scales Metal kernel.
///
/// Multiplies scale factors in quantized blocks by per-block multipliers.
///
/// # Arguments
/// * `dtype` - Quantization format
/// * `blocks_buf` - Mutable buffer containing quantized blocks
/// * `multipliers_buf` - f32 buffer of per-block multipliers
/// * `num_blocks` - Number of quantization blocks
#[allow(clippy::too_many_arguments)]
pub fn call_modify_block_scales_metal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    blocks_buf: &Buffer,
    multipliers_buf: &Buffer,
    num_blocks: usize,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        GgmlDType::Q8_0 => "modify_block_scales_q8_0",
        GgmlDType::Q4K => "modify_block_scales_q4_K",
        GgmlDType::Q2K => "modify_block_scales_q2_K",
        GgmlDType::Q3K => "modify_block_scales_q3_K",
        GgmlDType::Q5K => "modify_block_scales_q5_K",
        GgmlDType::Q6K => "modify_block_scales_q6_K",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "modify_block_scales not supported for {:?}",
                dtype
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::QuzoPerturb, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let num_blocks_i32 = num_blocks as i32;

    set_params!(encoder, (blocks_buf, multipliers_buf, num_blocks_i32));

    let threads_per_group = 256usize.min(pipeline.max_total_threads_per_threadgroup() as usize);
    let num_groups = (num_blocks + threads_per_group - 1) / threads_per_group;

    let thread_group_count = MTLSize {
        width: num_groups,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(
        blocks_buf,
        MTLResourceUsage::Read.union(MTLResourceUsage::Write),
    );
    encoder.use_resource(multipliers_buf, MTLResourceUsage::Read);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Call the fused perturb+matmul-vec Metal kernel.
///
/// Computes y = (W + eps*z) @ x where z is generated on-the-fly from seed.
///
/// # Arguments
/// * `dtype` - Quantization format of weights
/// * `weights_buf` - Quantized weight blocks
/// * `x_buf` - Input vector (f32)
/// * `x_offset` - Byte offset into x_buf
/// * `output_buf` - Output buffer (f32)
/// * `output_offset` - Byte offset into output_buf
/// * `nrows` - Number of output rows
/// * `ncols` - Number of columns (input dimension)
/// * `seed` - RNG seed
/// * `epsilon` - Perturbation magnitude
/// * `tensor_id` - Unique per-tensor ID for RNG diversification
#[allow(clippy::too_many_arguments)]
pub fn call_fused_matmul_vec_metal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    weights_buf: &Buffer,
    x_buf: &Buffer,
    x_offset: usize,
    output_buf: &Buffer,
    output_offset: usize,
    nrows: usize,
    ncols: usize,
    seed: u64,
    epsilon: f32,
    tensor_id: u64,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        GgmlDType::Q8_0 => "fused_mul_mat_vec_q8_0_f32",
        GgmlDType::Q4K => "fused_mul_mat_vec_q4_K_f32",
        GgmlDType::Q2K => "fused_mul_mat_vec_q2_K_f32",
        GgmlDType::Q3K => "fused_mul_mat_vec_q3_K_f32",
        GgmlDType::Q5K => "fused_mul_mat_vec_q5_K_f32",
        GgmlDType::Q6K => "fused_mul_mat_vec_q6_K_f32",
        GgmlDType::BF16 => "fused_mul_mat_vec_bf16_f32",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "fused_matmul_vec not supported for {:?}",
                dtype
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::QuzoFused, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (weights_buf, 0usize),
            (x_buf, x_offset),
            (output_buf, output_offset),
            nrows as i32,
            ncols as i32,
            seed,
            epsilon,
            tensor_id
        )
    );

    let thread_group_count = MTLSize {
        width: nrows,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(weights_buf, MTLResourceUsage::Read);
    encoder.use_resource(x_buf, MTLResourceUsage::Read);
    encoder.use_resource(output_buf, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
