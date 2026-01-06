//! Metal kernel bindings for Delta Net / Linear Attention operations.
//!
//! These kernels provide optimized Metal implementations for Qwen3-Next
//! and similar linear attention models.

use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

/// L2 normalize and scale in one fused operation.
///
/// Input: [..., dim]
/// Output: [..., dim] where output = input / ||input||_2 * scale
#[allow(clippy::too_many_arguments)]
pub fn call_l2_normalize_scale(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    batch: usize,
    dim: usize,
    scale: f32,
    eps: f32,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Cast dimensions to u32 to match Metal kernel's uint type
    set_params!(
        encoder,
        (&input, output, scale, eps, batch as u32, dim as u32)
    );

    let threads_per_group = dim.min(256).next_power_of_two();
    let num_simdgroups = (threads_per_group + 31) / 32;
    let shared_mem = num_simdgroups * std::mem::size_of::<f32>();

    let thread_group_count = MTLSize {
        width: batch,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.set_threadgroup_memory_length(0, shared_mem.max(128));
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Depthwise 1D convolution for SSM/Mamba-style models.
///
/// Input: [batch, channels, input_len] (with pre-padded input_len)
/// Weight: [channels, kernel_size]
/// Output: [batch, channels, output_len]
#[allow(clippy::too_many_arguments)]
pub fn call_depthwise_conv1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    batch: usize,
    channels: usize,
    input_len: usize,
    output_len: usize,
    kernel_size: usize,
    input: BufferOffset,
    weight: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Cast dimensions to u32 to match Metal kernel's uint type
    set_params!(
        encoder,
        (
            &input,
            &weight,
            output,
            batch as u32,
            channels as u32,
            input_len as u32,
            output_len as u32,
            kernel_size as u32
        )
    );

    let total_elems = batch * channels * output_len;
    let threads_per_group = 256usize.min(pipeline.max_total_threads_per_threadgroup() as usize);
    let num_groups = (total_elems + threads_per_group - 1) / threads_per_group;

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

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(weight.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Fused SwiGLU activation: silu(gate) * up
#[allow(clippy::too_many_arguments)]
pub fn call_fused_swiglu(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    total: usize,
    gate: BufferOffset,
    up: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Cast dimensions to u32 to match Metal kernel's uint type
    set_params!(encoder, (&gate, &up, output, total as u32));

    let threads_per_group = 256usize.min(pipeline.max_total_threads_per_threadgroup() as usize);
    let num_groups = (total + threads_per_group - 1) / threads_per_group;

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

    encoder.use_resource(gate.buffer, MTLResourceUsage::Read);
    encoder.use_resource(up.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Gated RMS norm: silu(z) * rms_norm(x, weight)
#[allow(clippy::too_many_arguments)]
pub fn call_gated_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    batch: usize,
    dim: usize,
    eps: f32,
    x: BufferOffset,
    z: BufferOffset,
    weight: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Cast dimensions to u32 to match Metal kernel's uint type
    set_params!(
        encoder,
        (&x, &z, &weight, output, eps, batch as u32, dim as u32)
    );

    let threads_per_group = dim.min(256).next_power_of_two();
    let num_simdgroups = (threads_per_group + 31) / 32;
    let shared_mem = num_simdgroups * std::mem::size_of::<f32>();

    let thread_group_count = MTLSize {
        width: batch,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.set_threadgroup_memory_length(0, shared_mem.max(128));
    encoder.use_resource(x.buffer, MTLResourceUsage::Read);
    encoder.use_resource(z.buffer, MTLResourceUsage::Read);
    encoder.use_resource(weight.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Lower triangular solve for (I - L) @ X = B
#[allow(clippy::too_many_arguments)]
pub fn call_solve_i_minus_lower_tri(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    batch_size: usize,
    n: usize,
    k: usize,
    l_matrix: BufferOffset,
    b_matrix: BufferOffset,
    x_output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Cast dimensions to u32 to match Metal kernel's uint type
    set_params!(
        encoder,
        (
            &l_matrix,
            &b_matrix,
            x_output,
            batch_size as u32,
            n as u32,
            k as u32
        )
    );

    let k_simdgroups = k.min(8);
    let threads_per_group = k_simdgroups * 32;

    let thread_group_count = MTLSize {
        width: batch_size,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(l_matrix.buffer, MTLResourceUsage::Read);
    encoder.use_resource(b_matrix.buffer, MTLResourceUsage::Read);
    encoder.use_resource(x_output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Fused Delta Net autoregressive step
///
/// q, k, v: [batch, num_heads, head_dim]
/// gate, beta: [batch, num_heads]
/// state: [batch, num_heads, head_dim, head_dim]
/// Returns: output [batch, num_heads, head_dim], new_state [batch, num_heads, head_dim, head_dim]
#[allow(clippy::too_many_arguments)]
pub fn call_delta_net_autoregressive_step(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    eps: f32,
    q: BufferOffset,
    k: BufferOffset,
    v: BufferOffset,
    gate: BufferOffset,
    beta: BufferOffset,
    state: BufferOffset,
    output: &Buffer,
    new_state: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Cast dimensions to u32 to match Metal kernel's uint type
    set_params!(
        encoder,
        (
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &state,
            output,
            new_state,
            scale,
            eps,
            batch as u32,
            num_heads as u32,
            head_dim as u32
        )
    );

    // Shared memory: 5 arrays of size 256 (s_q, s_k, s_v, s_kv_mem, s_delta) + warp sums
    // The kernel uses fixed-size 256-element arrays for flexibility across head dimensions
    let threads_per_group = head_dim.min(256);
    let _num_simdgroups = (threads_per_group + 31) / 32;
    // 5 * 256 = 1280 floats for main arrays + 64 for warp sums + 2 for norms
    let shared_mem = (5 * 256 + 64 + 2) * std::mem::size_of::<f32>();

    let thread_group_count = MTLSize {
        width: batch,
        height: num_heads,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.set_threadgroup_memory_length(0, shared_mem.max(1024));
    encoder.use_resource(q.buffer, MTLResourceUsage::Read);
    encoder.use_resource(k.buffer, MTLResourceUsage::Read);
    encoder.use_resource(v.buffer, MTLResourceUsage::Read);
    encoder.use_resource(gate.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(new_state, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Multi-token Delta Net state update
///
/// q, k, v: [batch, num_heads, num_tokens, head_dim]
/// gate, beta: [batch, num_heads, num_tokens]
/// state: [batch, num_heads, head_dim, head_dim]
#[allow(clippy::too_many_arguments)]
pub fn call_delta_net_multi_token_update(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    batch: usize,
    num_heads: usize,
    num_tokens: usize,
    head_dim: usize,
    q: BufferOffset,
    k: BufferOffset,
    v: BufferOffset,
    gate: BufferOffset,
    beta: BufferOffset,
    state: BufferOffset,
    output: &Buffer,
    new_state: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Cast dimensions to u32 to match Metal kernel's uint type
    set_params!(
        encoder,
        (
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &state,
            output,
            new_state,
            batch as u32,
            num_heads as u32,
            num_tokens as u32,
            head_dim as u32
        )
    );

    // Shared memory: 4 arrays of size 256 (s_k, s_v, s_delta, s_kv_mem)
    // The kernel uses fixed-size 256-element arrays
    let threads_per_group = head_dim.min(256);
    let shared_mem = 4 * 256 * std::mem::size_of::<f32>();

    let thread_group_count = MTLSize {
        width: batch,
        height: num_heads,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.set_threadgroup_memory_length(0, shared_mem.max(1024));
    encoder.use_resource(q.buffer, MTLResourceUsage::Read);
    encoder.use_resource(k.buffer, MTLResourceUsage::Read);
    encoder.use_resource(v.buffer, MTLResourceUsage::Read);
    encoder.use_resource(gate.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(new_state, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Delta Net parallel with intermediate state materialization
#[allow(clippy::too_many_arguments)]
pub fn call_delta_net_parallel_with_states(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    batch: usize,
    num_heads: usize,
    num_tokens: usize,
    head_dim: usize,
    q: BufferOffset,
    k: BufferOffset,
    v: BufferOffset,
    gate: BufferOffset,
    beta: BufferOffset,
    state: BufferOffset,
    output: &Buffer,
    new_state: &Buffer,
    intermediate_states: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &state,
            output,
            new_state,
            intermediate_states,
            batch as u32,
            num_heads as u32,
            num_tokens as u32,
            head_dim as u32
        )
    );

    // Shared memory: 4 arrays of size 256 (s_k, s_v, s_delta, s_kv_mem)
    // The kernel uses fixed-size 256-element arrays
    let threads_per_group = head_dim.min(256);
    let shared_mem = 4 * 256 * std::mem::size_of::<f32>();

    let thread_group_count = MTLSize {
        width: batch,
        height: num_heads,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };

    encoder.set_threadgroup_memory_length(0, shared_mem.max(1024));
    encoder.use_resource(q.buffer, MTLResourceUsage::Read);
    encoder.use_resource(k.buffer, MTLResourceUsage::Read);
    encoder.use_resource(v.buffer, MTLResourceUsage::Read);
    encoder.use_resource(gate.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(new_state, MTLResourceUsage::Write);
    encoder.use_resource(intermediate_states, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Top-K softmax for MoE routing
#[allow(clippy::too_many_arguments)]
pub fn call_topk_softmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    n_tokens: usize,
    n_experts: usize,
    topk: usize,
    logits: BufferOffset,
    weights: &Buffer,
    indices: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Deltanet, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Cast dimensions to u32 to match Metal kernel's uint type
    set_params!(
        encoder,
        (
            &logits,
            weights,
            indices,
            n_tokens as u32,
            n_experts as u32,
            topk as u32
        )
    );

    let thread_group_count = MTLSize {
        width: n_tokens,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: 32, // One simdgroup per token
        height: 1,
        depth: 1,
    };

    encoder.use_resource(logits.buffer, MTLResourceUsage::Read);
    encoder.use_resource(weights, MTLResourceUsage::Write);
    encoder.use_resource(indices, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
