//! Metal kernel bindings for Q8 flash attention.

use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

/// Call the flash_attn_q8 Metal kernel.
///
/// Performs flash attention with Q8_0-quantized KV cache.
/// Q is in f16, K is Q8_0, V is f16, output is f16.
///
/// # Arguments
/// * `head_dim` - Head dimension (64, 128, or 256)
/// * `q_buf` - Query buffer (half)
/// * `k_buf` - Key buffer (block_q8_0)
/// * `v_buf` - Value buffer (half)
/// * `o_buf` - Output buffer (half)
/// * `scale` - Attention scale factor
/// * `seq_k` - Key/Value sequence length
/// * `seq_q` - Query sequence length
/// * `stride_q` - Q strides (b, h, s, d)
/// * `stride_k` - K strides (b, h, s) — in blocks
/// * `stride_v` - V strides (b, h, s, d)
/// * `stride_o` - O strides (b, h, s)
/// * `batch` - Batch size
/// * `num_heads` - Number of attention heads
#[allow(clippy::too_many_arguments)]
pub fn call_flash_attn_q8_metal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    head_dim: usize,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    o_buf: &Buffer,
    scale: f32,
    seq_k: usize,
    seq_q: usize,
    stride_q_b: usize,
    stride_q_h: usize,
    stride_q_s: usize,
    stride_q_d: usize,
    stride_k_b: usize,
    stride_k_h: usize,
    stride_k_s: usize,
    stride_v_b: usize,
    stride_v_h: usize,
    stride_v_s: usize,
    stride_v_d: usize,
    stride_o_b: usize,
    stride_o_h: usize,
    stride_o_s: usize,
    batch: usize,
    num_heads: usize,
) -> Result<(), MetalKernelError> {
    let kernel_name = match head_dim {
        64 => "flash_attn_q8_d64",
        128 => "flash_attn_q8_d128",
        256 => "flash_attn_q8_d256",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "flash_attn_q8 not supported for head_dim={}",
                head_dim
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::FlashAttnQ8, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            q_buf,
            k_buf,
            v_buf,
            o_buf,
            scale,
            seq_k as i32,
            seq_q as i32,
            stride_q_b as u64,
            stride_q_h as u64,
            stride_q_s as u64,
            stride_q_d as u64,
            stride_k_b as u64,
            stride_k_h as u64,
            stride_k_s as u64,
            stride_v_b as u64,
            stride_v_h as u64,
            stride_v_s as u64,
            stride_v_d as u64,
            stride_o_b as u64,
            stride_o_h as u64,
            stride_o_s as u64
        )
    );

    // Launch config: gridDim = (batch, num_heads, seq_q), blockDim = (32, 4, 1)
    let thread_group_count = MTLSize {
        width: batch,
        height: num_heads,
        depth: seq_q,
    };
    let thread_group_size = MTLSize {
        width: 32, // lanes per warp
        height: 4, // warps per block
        depth: 1,
    };

    encoder.use_resource(q_buf, MTLResourceUsage::Read);
    encoder.use_resource(k_buf, MTLResourceUsage::Read);
    encoder.use_resource(v_buf, MTLResourceUsage::Read);
    encoder.use_resource(o_buf, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
