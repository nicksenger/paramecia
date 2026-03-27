//! Delta Net / Linear Attention operations for Qwen3-Next and similar models.
//!
//! This module provides tensor-level operations for Delta Net computations,
//! using optimized CUDA kernels when available, with CPU fallback.

#[cfg(feature = "vulkan")]
use crate::quantized::QStorage;
#[cfg(feature = "vulkan")]
use crate::{DType, Storage};
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
use crate::{Device, Shape};
use crate::{Result, Tensor, D};

#[cfg(feature = "vulkan")]
use crate::vulkan_backend::VulkanStorage;

/// Extract the raw vk::Buffer from a Tensor known to be on Vulkan.
#[cfg(feature = "vulkan")]
fn vulkan_buffer(t: &Tensor) -> Result<ash::vk::Buffer> {
    match &*t.storage() {
        Storage::Vulkan(s) => s.vk_buffer(),
        _ => crate::bail!("Expected Vulkan storage"),
    }
}

/// Extract both raw vk::Buffer and Arc<VulkanBuffer> so callers can keep it alive across batched dispatches.
#[cfg(feature = "vulkan")]
fn vulkan_buffer_with_arc(
    t: &Tensor,
) -> Result<(
    ash::vk::Buffer,
    std::sync::Arc<crate::vulkan_backend::device::VulkanBuffer>,
)> {
    match &*t.storage() {
        Storage::Vulkan(s) => {
            let arc = s.vk_buffer_arc()?;
            Ok((arc.buffer, arc))
        }
        _ => crate::bail!("Expected Vulkan storage"),
    }
}

#[cfg(feature = "vulkan")]
fn flash_attn_q8_split_temp_bytes(
    b: usize,
    h: usize,
    d: usize,
    seq_q: usize,
    split_k: u32,
) -> Option<usize> {
    let rows = b.checked_mul(seq_q)?.checked_mul(h)?;
    let split = split_k as usize;
    let partial_bytes = split.checked_mul(rows)?.checked_mul(d)?.checked_mul(4)?;
    let stats_bytes = split.checked_mul(rows)?.checked_mul(2)?.checked_mul(4)?;
    partial_bytes.checked_add(stats_bytes)
}

#[cfg(feature = "vulkan")]
fn flash_attn_q8_split_max_temp_bytes() -> usize {
    const DEFAULT_MB: usize = 256;
    let mb = std::env::var("PARAMECIA_VULKAN_FLASH_Q8_SPLIT_MAX_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_MB);
    mb.saturating_mul(1024 * 1024)
}

#[cfg(feature = "vulkan")]
fn choose_flash_attn_q8_split_k(b: usize, h: usize, d: usize, seq_q: usize, seq_k: usize) -> u32 {
    if std::env::var("PARAMECIA_DISABLE_VULKAN_FLASH_Q8_SPLIT").is_ok() {
        return 1;
    }
    if seq_k < 2048 || seq_q == 0 {
        return 1;
    }

    let forced = std::env::var("PARAMECIA_VULKAN_FLASH_Q8_SPLIT_K")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|v| *v >= 1)
        .unwrap_or(0);

    let mut split_k = if forced > 0 {
        forced
    } else if seq_k >= 16384 {
        8
    } else if seq_k >= 8192 {
        4
    } else if seq_k >= 4096 {
        3
    } else {
        2
    };

    // Vulkan minimum guarantees for maxComputeWorkGroupCount are >= 65535 per dimension.
    // Keep x-dispatch within this bound: x = batch * split_k.
    let max_dispatch_split = if b == 0 { 1 } else { (65535 / b as u32).max(1) };
    split_k = split_k.min(max_dispatch_split);
    if split_k <= 1 {
        return 1;
    }

    let max_temp_bytes = flash_attn_q8_split_max_temp_bytes();
    while split_k > 1 {
        if let Some(bytes) = flash_attn_q8_split_temp_bytes(b, h, d, seq_q, split_k) {
            if bytes <= max_temp_bytes {
                break;
            }
        }
        split_k -= 1;
    }

    split_k.max(1)
}

/// Check if cooperative matrix is disabled for delta_net_parallel operations via environment variables.
/// Checks in order:
/// 1. PARAMECIA_DISABLE_COOPMAT_DELTA_NET_PARALLEL (all head dims)
/// 2. PARAMECIA_DISABLE_COOPMAT_D<HEAD_DIM>_DELTA_NET_PARALLEL (specific head dim)
#[cfg(feature = "vulkan")]
fn is_deltanet_parallel_coopmat_disabled(head_dim: usize) -> bool {
    // Check global enable flag first - coopmat is opt-in
    if std::env::var("PARAMECIA_ENABLE_COOPMAT").is_err() {
        return true; // Disabled by default
    }

    // Check operation-wide env var
    if std::env::var("PARAMECIA_DISABLE_COOPMAT_DELTA_NET_PARALLEL").is_ok() {
        return true;
    }

    // Check head_dim-specific env var
    let head_dim_specific = format!("PARAMECIA_DISABLE_COOPMAT_D{}_DELTA_NET_PARALLEL", head_dim);
    std::env::var(&head_dim_specific).is_ok()
}

/// Check if cooperative matrix is disabled for gla_step operations via environment variables.
/// Checks in order:
/// 1. PARAMECIA_DISABLE_COOPMAT_GLA_STEP (all head dims)
/// 2. PARAMECIA_DISABLE_COOPMAT_D<HEAD_DIM>_GLA_STEP (specific head dim)
#[cfg(feature = "vulkan")]
fn is_gla_step_coopmat_disabled(head_dim: usize) -> bool {
    // Check global enable flag first - coopmat is opt-in
    if std::env::var("PARAMECIA_ENABLE_COOPMAT").is_err() {
        return true; // Disabled by default
    }

    // Check operation-wide env var
    if std::env::var("PARAMECIA_DISABLE_COOPMAT_GLA_STEP").is_ok() {
        return true;
    }

    // Check head_dim-specific env var
    let head_dim_specific = format!("PARAMECIA_DISABLE_COOPMAT_D{}_GLA_STEP", head_dim);
    std::env::var(&head_dim_specific).is_ok()
}

/// Sigmoid activation: 1 / (1 + exp(-x))
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let one_plus_exp = (exp_neg_x + 1.0)?;
    one_plus_exp.recip()
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    let sig = sigmoid(x)?;
    x.mul(&sig)
}

/// Fused Delta Net autoregressive step for single-token generation.
///
/// This is the critical path for token generation speed in linear attention models.
/// The fused CUDA kernel combines L2 normalization, state decay, kv_mem computation,
/// delta rule update, and output computation into a single kernel launch.
///
/// # Arguments
/// * `q` - Query tensor [batch, num_heads, head_dim]
/// * `k` - Key tensor [batch, num_heads, head_dim]
/// * `v` - Value tensor [batch, num_heads, head_dim]
/// * `gate` - Log decay values [batch, num_heads]
/// * `beta` - Gate values (pre-sigmoid) [batch, num_heads]
/// * `state` - Recurrent state [batch, num_heads, head_dim, head_dim]
/// * `scale` - Q scaling factor (typically 1/sqrt(head_dim))
/// * `eps` - Epsilon for L2 normalization
///
/// # Returns
/// Tuple of (output, new_state) where:
/// * `output` - [batch, num_heads, head_dim]
/// * `new_state` - [batch, num_heads, head_dim, head_dim]
#[allow(clippy::too_many_arguments)]
pub fn delta_net_autoregressive_step(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    // Validate shapes
    let q_dims = q.dims();
    let k_dims = k.dims();
    let v_dims = v.dims();
    let gate_dims = gate.dims();
    let beta_dims = beta.dims();
    let state_dims = state.dims();

    if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
        crate::bail!(
            "delta_net_autoregressive_step: q, k, v must be 3D [batch, heads, dim], got {:?}, {:?}, {:?}",
            q_dims,
            k_dims,
            v_dims
        );
    }

    if gate_dims.len() != 2 || beta_dims.len() != 2 {
        crate::bail!(
            "delta_net_autoregressive_step: gate, beta must be 2D [batch, heads], got {:?}, {:?}",
            gate_dims,
            beta_dims
        );
    }

    if state_dims.len() != 4 {
        crate::bail!(
            "delta_net_autoregressive_step: state must be 4D [batch, heads, dim, dim], got {:?}",
            state_dims
        );
    }

    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let head_dim = q_dims[2];

    // Ensure all tensors are contiguous
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let gate = gate.contiguous()?;
    let beta = beta.contiguous()?;
    let state = state.contiguous()?;

    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = q.device() {
            return delta_net_autoregressive_step_cuda(
                &q, &k, &v, &gate, &beta, &state, scale, eps,
            );
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Device::Metal(_) = q.device() {
            return delta_net_autoregressive_step_metal(
                &q, &k, &v, &gate, &beta, &state, scale, eps,
            );
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = q.device() {
            if !crate::vulkan_backend::debug::force_cpu_deltanet_step() {
                return delta_net_autoregressive_step_vulkan(
                    &q, &k, &v, &gate, &beta, &state, scale, eps,
                );
            }
        }
    }

    // CPU fallback
    delta_net_autoregressive_step_cpu(
        &q, &k, &v, &gate, &beta, &state, scale, eps, batch, num_heads, head_dim,
    )
}

#[cfg(feature = "cuda")]
fn delta_net_autoregressive_step_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use crate::Storage;

    // Extract storage and layout from each tensor
    let q_storage = q.storage();
    let k_storage = k.storage();
    let v_storage = v.storage();
    let gate_storage = gate.storage();
    let beta_storage = beta.storage();
    let state_storage = state.storage();

    let q_layout = q.layout();
    let k_layout = k.layout();
    let v_layout = v.layout();
    let gate_layout = gate.layout();
    let beta_layout = beta.layout();
    let state_layout = state.layout();

    // Get CUDA storage for each tensor
    let (q_cuda, k_cuda, v_cuda, gate_cuda, beta_cuda, state_cuda) = match (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*gate_storage,
        &*beta_storage,
        &*state_storage,
    ) {
        (
            Storage::Cuda(q_s),
            Storage::Cuda(k_s),
            Storage::Cuda(v_s),
            Storage::Cuda(gate_s),
            Storage::Cuda(beta_s),
            Storage::Cuda(state_s),
        ) => (q_s, k_s, v_s, gate_s, beta_s, state_s),
        _ => crate::bail!("delta_net_autoregressive_step_cuda: all tensors must be on CUDA"),
    };

    // Call the CUDA kernel
    let (output_storage, new_state_storage) =
        crate::cuda_backend::deltanet::delta_net_autoregressive_step(
            q_cuda,
            q_layout,
            k_cuda,
            k_layout,
            v_cuda,
            v_layout,
            gate_cuda,
            gate_layout,
            beta_cuda,
            beta_layout,
            state_cuda,
            state_layout,
            scale,
            eps,
        )?;

    // Create output tensors
    let q_dims = q.dims();
    let output_shape = Shape::from((q_dims[0], q_dims[1], q_dims[2]));
    let state_shape = Shape::from((q_dims[0], q_dims[1], q_dims[2], q_dims[2]));

    let output = Tensor::from_storage(Storage::Cuda(output_storage), output_shape);

    let new_state = Tensor::from_storage(Storage::Cuda(new_state_storage), state_shape);

    Ok((output, new_state))
}

#[cfg(feature = "metal")]
fn delta_net_autoregressive_step_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use crate::metal_backend::MetalStorage;
    use crate::{Layout, Shape, Storage};

    let q_storage = q.storage();
    let k_storage = k.storage();
    let v_storage = v.storage();
    let gate_storage = gate.storage();
    let beta_storage = beta.storage();
    let state_storage = state.storage();

    let q_layout = q.layout();
    let k_layout = k.layout();
    let v_layout = v.layout();
    let gate_layout = gate.layout();
    let beta_layout = beta.layout();
    let state_layout = state.layout();

    let (q_metal, k_metal, v_metal, gate_metal, beta_metal, state_metal) = match (
        &*q_storage,
        &*k_storage,
        &*v_storage,
        &*gate_storage,
        &*beta_storage,
        &*state_storage,
    ) {
        (
            Storage::Metal(q_s),
            Storage::Metal(k_s),
            Storage::Metal(v_s),
            Storage::Metal(gate_s),
            Storage::Metal(beta_s),
            Storage::Metal(state_s),
        ) => (q_s, k_s, v_s, gate_s, beta_s, state_s),
        _ => crate::bail!("delta_net_autoregressive_step_metal: all tensors must be on Metal"),
    };

    let (output_storage, new_state_storage) =
        crate::metal_backend::deltanet::delta_net_autoregressive_step(
            q_metal,
            q_layout,
            k_metal,
            k_layout,
            v_metal,
            v_layout,
            gate_metal,
            gate_layout,
            beta_metal,
            beta_layout,
            state_metal,
            state_layout,
            scale,
            eps,
        )?;

    let q_dims = q.dims();
    let output_shape = Shape::from((q_dims[0], q_dims[1], q_dims[2]));
    let state_shape = Shape::from((q_dims[0], q_dims[1], q_dims[2], q_dims[2]));

    let output = Tensor::from_storage(Storage::Metal(output_storage), output_shape);

    let new_state = Tensor::from_storage(Storage::Metal(new_state_storage), state_shape);

    Ok((output, new_state))
}

#[cfg(feature = "vulkan")]
fn delta_net_autoregressive_step_vulkan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use std::sync::Arc;

    let q_dims = q.dims();
    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let head_dim = q_dims[2];
    let batch_heads = batch * num_heads;

    let device = match q.device() {
        Device::Vulkan(d) => d.clone(),
        _ => crate::bail!("delta_net_autoregressive_step_vulkan: expected Vulkan device"),
    };

    // Ensure f32, contiguous, and offset 0 for direct buffer access.
    // contiguous() handles non-contiguous strides; we additionally check for
    // non-zero storage offset which contiguous() doesn't fix.
    fn ensure_offset_zero(t: Tensor) -> Result<Tensor> {
        if t.layout().start_offset() != 0 {
            t.force_contiguous()
        } else {
            Ok(t)
        }
    }
    let q = ensure_offset_zero(q.to_dtype(DType::F32)?.contiguous()?)?;
    let k = ensure_offset_zero(k.to_dtype(DType::F32)?.contiguous()?)?;
    let v = ensure_offset_zero(v.to_dtype(DType::F32)?.contiguous()?)?;
    let gate = ensure_offset_zero(gate.to_dtype(DType::F32)?.contiguous()?)?;
    let beta = ensure_offset_zero(beta.to_dtype(DType::F32)?.contiguous()?)?;
    let state = ensure_offset_zero(state.to_dtype(DType::F32)?.contiguous()?)?;

    // Get GPU buffers from tensors (all guaranteed offset 0)
    let q_buf = vulkan_buffer(&q)?;
    let k_buf = vulkan_buffer(&k)?;
    let v_buf = vulkan_buffer(&v)?;
    let gate_buf = vulkan_buffer(&gate)?;
    let beta_buf = vulkan_buffer(&beta)?;
    let state_buf = vulkan_buffer(&state)?;

    // Allocate output buffers
    let new_state_elems = batch_heads * head_dim * head_dim;
    let new_state_gpu = device.allocate_buffer((new_state_elems * 4) as u64)?;
    let output_elems = batch_heads * head_dim;
    let output_gpu = device.allocate_buffer((output_elems * 4) as u64)?;

    // Check if cooperative matrix is available and not disabled
    let has_coopmat = device.has_cooperative_matrix();
    let use_coopmat = has_coopmat && !is_gla_step_coopmat_disabled(head_dim);

    // Select shader variant by head_dim
    let shader_name = if use_coopmat {
        format!("gla_step_d{}_coopmat", head_dim)
    } else {
        format!("gla_step_d{}", head_dim)
    };

    // Get tile sizes and specialization constants for coopmat
    let specialization_constants = if use_coopmat {
        let (tile_m, tile_n, tile_k) = device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
        Some(vec![tile_m, tile_n, tile_k])
    } else {
        None
    };

    let pc_size = 16u32; // 4 x u32/f32 = 16 bytes
    let num_buffers = 8u32;

    // Try to load the coopmat pipeline, fall back to standard on failure
    let pipeline = if use_coopmat {
        match device.kernels().load_pipeline(
            device.device(),
            &shader_name,
            None,
            pc_size,
            num_buffers,
            specialization_constants.as_ref().map(|v| v.as_slice()),
            false,
        ) {
            Ok(p) => p,
            Err(_) => {
                // Fall back to standard shader
                let fallback_name = format!("gla_step_d{}", head_dim);
                device
                    .kernels()
                    .load_pipeline(
                        device.device(),
                        &fallback_name,
                        None,
                        pc_size,
                        num_buffers,
                        None,
                        false,
                    )
                    .map_err(|e| {
                        crate::Error::Msg(format!("Failed to load gla_step pipeline: {}", e))
                    })?
            }
        }
    } else {
        device
            .kernels()
            .load_pipeline(
                device.device(),
                &shader_name,
                None,
                pc_size,
                num_buffers,
                None,
                false,
            )
            .map_err(|e| crate::Error::Msg(format!("Failed to load gla_step pipeline: {}", e)))?
    };

    // Push constants: batch_heads (u32), d (u32), scale (f32), eps (f32)
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GlaStepPC {
        batch_heads: u32,
        d: u32,
        scale: f32,
        eps: f32,
    }
    let pc = GlaStepPC {
        batch_heads: batch_heads as u32,
        d: head_dim as u32,
        scale,
        eps,
    };

    // Dispatch: each buffer bound separately, no packing
    // bindings: Q, K, V, Gate, Beta, State, NewState, Output
    device.record_compute_with_write_mask(
        &pipeline,
        &[
            q_buf,
            k_buf,
            v_buf,
            gate_buf,
            beta_buf,
            state_buf,
            new_state_gpu.buffer,
            output_gpu.buffer,
        ],
        Some((1u64 << 6) | (1u64 << 7)),
        Some(bytemuck::bytes_of(&pc)),
        [batch_heads as u32, 1, 1],
    )?;

    // Create output tensors directly from GPU buffers
    let output_shape = Shape::from((batch, num_heads, head_dim));
    let state_shape = Shape::from((batch, num_heads, head_dim, head_dim));

    let output = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(output_gpu)),
            device.clone(),
            output_elems,
            DType::F32,
        )),
        output_shape,
    );
    let new_state = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(new_state_gpu)),
            device.clone(),
            new_state_elems,
            DType::F32,
        )),
        state_shape,
    );

    // No explicit flush needed: subsequent GPU operations have proper
    // compute→compute barriers, and downloads will flush when needed.

    Ok((output, new_state))
}

#[allow(clippy::too_many_arguments)]
fn delta_net_autoregressive_step_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    scale: f32,
    eps: f32,
    _batch: usize,
    _num_heads: usize,
    _head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    // L2 normalize Q (with scale) and K
    let q_sq = q.sqr()?;
    let q_norm_sq = q_sq.sum_keepdim(D::Minus1)?;
    let q_norm = (q_norm_sq + eps as f64)?.sqrt()?;
    let q_normalized = (q.broadcast_div(&q_norm)? * (scale as f64))?;

    let k_sq = k.sqr()?;
    let k_norm_sq = k_sq.sum_keepdim(D::Minus1)?;
    let k_norm = (k_norm_sq + eps as f64)?.sqrt()?;
    let k_normalized = k.broadcast_div(&k_norm)?;

    // Apply sigmoid to beta
    let beta_sig = sigmoid(beta)?;

    // g_t = exp(gate).unsqueeze(-1).unsqueeze(-1)
    let g_t = gate.exp()?.unsqueeze(2)?.unsqueeze(3)?;

    // Decay the state: state shape is [b, h, d, d]
    let decayed_state = state.broadcast_mul(&g_t)?;

    // kv_mem = sum_j(state[i, j] * k[j])
    // k shape [b, h, d], unsqueeze(2) -> [b, h, 1, d] broadcasts along dim 2
    // state [b, h, d, d] * k_expanded [b, h, 1, d] -> result[i, j] = state[i, j] * k[j]
    // sum over last dim (j) -> [b, h, d]
    let k_expanded = k_normalized.unsqueeze(2)?;
    let kv_mem = decayed_state.broadcast_mul(&k_expanded)?.sum(D::Minus1)?;

    // delta = (v - kv_mem) * beta.unsqueeze(-1)
    let beta_t = beta_sig.unsqueeze(2)?;
    let delta = (v - kv_mem)?.broadcast_mul(&beta_t)?;

    // new_state[i, j] = decayed_state[i, j] + k[j] * delta[i]
    // delta_col: [b, h, d, 1], k_row: [b, h, 1, d]
    // delta_col @ k_row = [b, h, d, d] where result[i, j] = delta[i] * k[j]
    let delta_col = delta.unsqueeze(3)?;
    let k_row = k_normalized.unsqueeze(2)?;
    let k_delta_outer = delta_col.matmul(&k_row)?;
    let new_state = (decayed_state + k_delta_outer)?;

    // output[i] = sum_j(new_state[i, j] * q[j])
    // q_expanded: [b, h, 1, d] broadcasts along dim 2
    let q_expanded = q_normalized.unsqueeze(2)?;
    let output = new_state.broadcast_mul(&q_expanded)?.sum(D::Minus1)?;

    Ok((output, new_state))
}

/// L2 normalize and scale in one fused operation.
///
/// Computes: x / ||x||_2 * scale
///
/// # Arguments
/// * `x` - Input tensor [..., dim]
/// * `scale` - Scale factor to apply after normalization
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// Normalized and scaled tensor with same shape as input
pub fn l2_normalize_scale(x: &Tensor, scale: f32, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = x.device() {
            return l2_normalize_scale_cuda(x, scale, eps);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Device::Metal(_) = x.device() {
            return l2_normalize_scale_metal(x, scale, eps);
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = x.device() {
            if !crate::vulkan_backend::debug::force_cpu_l2norm() {
                return l2_normalize_scale_vulkan(x, scale, eps);
            }
        }
    }

    // CPU fallback
    l2_normalize_scale_cpu(x, scale, eps)
}

#[cfg(feature = "cuda")]
fn l2_normalize_scale_cuda(x: &Tensor, scale: f32, eps: f64) -> Result<Tensor> {
    use crate::{DType, Storage};

    let original_dtype = x.dtype();

    // CUDA kernel supports f32 and f16 natively, convert other dtypes to f32
    let needs_conversion = !matches!(original_dtype, DType::F32 | DType::F16);
    let x_work = if needs_conversion {
        x.to_dtype(DType::F32)?
    } else {
        x.clone()
    };

    let x_work = x_work.contiguous()?;
    let x_storage = x_work.storage();
    let x_layout = x_work.layout();

    let x_cuda = match &*x_storage {
        Storage::Cuda(s) => s,
        _ => crate::bail!("l2_normalize_scale_cuda: tensor must be on CUDA"),
    };

    let output_storage =
        crate::cuda_backend::deltanet::l2_normalize_scale(x_cuda, x_layout, scale, eps as f32)?;

    let output = Tensor::from_storage(Storage::Cuda(output_storage), x.shape().clone());

    // Convert back to original dtype if needed
    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

#[cfg(feature = "metal")]
fn l2_normalize_scale_metal(x: &Tensor, scale: f32, eps: f64) -> Result<Tensor> {
    use crate::{DType, Storage};

    let original_dtype = x.dtype();

    // Metal kernel supports f32 and f16 natively
    let needs_conversion = !matches!(original_dtype, DType::F32 | DType::F16);
    let x_work = if needs_conversion {
        x.to_dtype(DType::F32)?
    } else {
        x.clone()
    };

    let x_work = x_work.contiguous()?;
    let x_storage = x_work.storage();
    let x_layout = x_work.layout();

    let x_metal = match &*x_storage {
        Storage::Metal(s) => s,
        _ => crate::bail!("l2_normalize_scale_metal: tensor must be on Metal"),
    };

    let output_storage =
        crate::metal_backend::deltanet::l2_normalize_scale(x_metal, x_layout, scale, eps as f32)?;

    let output = Tensor::from_storage(Storage::Metal(output_storage), x.shape().clone());

    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

#[cfg(feature = "vulkan")]
fn l2_normalize_scale_vulkan(x: &Tensor, scale: f32, eps: f64) -> Result<Tensor> {
    use ash::vk;
    use std::sync::Arc;

    let device = match x.device() {
        Device::Vulkan(d) => d.clone(),
        _ => crate::bail!("l2_normalize_scale_vulkan: expected Vulkan device"),
    };

    let original_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let x_buf = vulkan_buffer(&x)?;

    let x_dims = x.dims();
    let dim = *x_dims.last().unwrap();
    let num_rows = x.elem_count() / dim;

    // Push constants: [num_rows, dim, scale, eps] = 16 bytes
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct L2NormPC {
        num_rows: u32,
        dim: u32,
        scale: f32,
        eps: f32,
    }
    let pc = L2NormPC {
        num_rows: num_rows as u32,
        dim: dim as u32,
        scale,
        eps: eps as f32,
    };

    // Output buffer
    let output_elems = x.elem_count();
    let output_gpu = device.allocate_buffer((output_elems * 4) as vk::DeviceSize)?;

    let pipeline = device
        .kernels()
        .load_pipeline(
            device.device(),
            "l2_normalize_scale",
            None,
            16,
            2,
            None,
            false,
        )
        .map_err(|e| {
            crate::Error::Msg(format!("Failed to load l2_normalize_scale pipeline: {}", e))
        })?;

    device.record_compute_with_write_mask(
        &pipeline,
        &[x_buf, output_gpu.buffer],
        Some(0b10),
        Some(bytemuck::bytes_of(&pc)),
        [num_rows as u32, 1, 1],
    )?;

    let output = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(output_gpu)),
            device,
            output_elems,
            DType::F32,
        )),
        x.shape().clone(),
    );

    if original_dtype != DType::F32 {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

fn l2_normalize_scale_cpu(x: &Tensor, scale: f32, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let norm_sq = x_sq.sum_keepdim(D::Minus1)?;
    let norm = (norm_sq + eps)?.sqrt()?;
    let normalized = x.broadcast_div(&norm)?;
    normalized * scale as f64
}

/// Depthwise 1D convolution for SSM/Mamba-style models.
///
/// This is more efficient than unfold + broadcast_mul + sum.
///
/// # Arguments
/// * `input` - Input tensor [batch, channels, input_len] (with pre-padded input_len)
/// * `weight` - Convolution weights [channels, kernel_size]
///
/// # Returns
/// Output tensor [batch, channels, output_len] where output_len = input_len - kernel_size + 1
pub fn depthwise_conv1d(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (_batch, channels, input_len) = input.dims3()?;
    let (weight_channels, kernel_size) = weight.dims2()?;

    if channels != weight_channels {
        crate::bail!(
            "depthwise_conv1d: channel mismatch {} vs {}",
            channels,
            weight_channels
        );
    }

    let output_len = input_len - kernel_size + 1;

    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = input.device() {
            return depthwise_conv1d_cuda(input, weight, output_len);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Device::Metal(_) = input.device() {
            return depthwise_conv1d_metal(input, weight, output_len);
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = input.device() {
            if !crate::vulkan_backend::debug::force_cpu_conv1d() {
                return depthwise_conv1d_vulkan(input, weight, output_len);
            }
        }
    }

    // CPU fallback using unfold
    depthwise_conv1d_cpu(input, weight, output_len)
}

#[cfg(feature = "cuda")]
fn depthwise_conv1d_cuda(input: &Tensor, weight: &Tensor, output_len: usize) -> Result<Tensor> {
    use crate::{DType, Storage};

    let original_dtype = input.dtype();

    // Convert to supported dtype if needed
    let needs_conversion = !matches!(original_dtype, DType::F32 | DType::F16 | DType::BF16);
    let (input_work, weight_work) = if needs_conversion {
        (input.to_dtype(DType::F32)?, weight.to_dtype(DType::F32)?)
    } else {
        (input.clone(), weight.clone())
    };

    let input_work = input_work.contiguous()?;
    let weight_work = weight_work.contiguous()?;
    let input_storage = input_work.storage();
    let weight_storage = weight_work.storage();
    let input_layout = input_work.layout();
    let weight_layout = weight_work.layout();

    let (input_cuda, weight_cuda) = match (&*input_storage, &*weight_storage) {
        (Storage::Cuda(i), Storage::Cuda(w)) => (i, w),
        _ => crate::bail!("depthwise_conv1d_cuda: tensors must be on CUDA"),
    };

    let output_storage = crate::cuda_backend::deltanet::depthwise_conv1d(
        input_cuda,
        input_layout,
        weight_cuda,
        weight_layout,
        output_len,
    )?;

    let (batch, channels, _) = input.dims3()?;
    let output = Tensor::from_storage(
        Storage::Cuda(output_storage),
        Shape::from((batch, channels, output_len)),
    );

    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

#[cfg(feature = "metal")]
fn depthwise_conv1d_metal(input: &Tensor, weight: &Tensor, output_len: usize) -> Result<Tensor> {
    use crate::{DType, Storage};

    let original_dtype = input.dtype();

    let needs_conversion = !matches!(original_dtype, DType::F32 | DType::F16);
    let (input_work, weight_work) = if needs_conversion {
        (input.to_dtype(DType::F32)?, weight.to_dtype(DType::F32)?)
    } else {
        (input.clone(), weight.clone())
    };

    let input_work = input_work.contiguous()?;
    let weight_work = weight_work.contiguous()?;
    let input_storage = input_work.storage();
    let weight_storage = weight_work.storage();
    let input_layout = input_work.layout();
    let weight_layout = weight_work.layout();

    let (input_metal, weight_metal) = match (&*input_storage, &*weight_storage) {
        (Storage::Metal(i), Storage::Metal(w)) => (i, w),
        _ => crate::bail!("depthwise_conv1d_metal: tensors must be on Metal"),
    };

    let output_storage = crate::metal_backend::deltanet::depthwise_conv1d(
        input_metal,
        input_layout,
        weight_metal,
        weight_layout,
        output_len,
    )?;

    let (batch, channels, _) = input.dims3()?;
    let output = Tensor::from_storage(
        Storage::Metal(output_storage),
        Shape::from((batch, channels, output_len)),
    );

    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

#[cfg(feature = "vulkan")]
fn depthwise_conv1d_vulkan(input: &Tensor, weight: &Tensor, output_len: usize) -> Result<Tensor> {
    use ash::vk;
    use std::sync::Arc;

    let device = match input.device() {
        Device::Vulkan(d) => d.clone(),
        _ => crate::bail!("depthwise_conv1d_vulkan: expected Vulkan device"),
    };

    let original_dtype = input.dtype();
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let weight = weight.to_dtype(DType::F32)?.contiguous()?;

    let (batch, channels, input_len) = input.dims3()?;
    let (_, kernel_size) = weight.dims2()?;

    let input_buf = vulkan_buffer(&input)?;
    let weight_buf = vulkan_buffer(&weight)?;

    // Push constants: 32 bytes
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ConvPC {
        batch: u32,
        channels: u32,
        input_len: u32,
        kernel_size: u32,
        output_len: u32,
        _pad: [u32; 3],
    }
    let pc = ConvPC {
        batch: batch as u32,
        channels: channels as u32,
        input_len: input_len as u32,
        kernel_size: kernel_size as u32,
        output_len: output_len as u32,
        _pad: [0; 3],
    };

    let output_elems = batch * channels * output_len;
    let output_gpu = device.allocate_buffer((output_elems * 4) as vk::DeviceSize)?;

    let pipeline = device
        .kernels()
        .load_pipeline(
            device.device(),
            "depthwise_conv1d",
            None,
            32,
            3,
            None,
            false,
        )
        .map_err(|e| {
            crate::Error::Msg(format!("Failed to load depthwise_conv1d pipeline: {}", e))
        })?;

    device.record_compute_with_write_mask(
        &pipeline,
        &[input_buf, weight_buf, output_gpu.buffer],
        Some(0b100),
        Some(bytemuck::bytes_of(&pc)),
        [
            ((output_len as u32) + 255) / 256,
            channels as u32,
            batch as u32,
        ],
    )?;

    let output = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(output_gpu)),
            device,
            output_elems,
            DType::F32,
        )),
        Shape::from((batch, channels, output_len)),
    );

    if original_dtype != DType::F32 {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

fn depthwise_conv1d_cpu(input: &Tensor, weight: &Tensor, _output_len: usize) -> Result<Tensor> {
    // input: [b, channels, input_len]
    // weight: [channels, kernel_size]
    let kernel_size = weight.dims()[1];

    // Use unfold to extract sliding windows
    let input_contiguous = input.contiguous()?;
    let windows = input_contiguous.unfold(2, kernel_size, 1)?; // [b, channels, output_len, kernel_size]

    // weight is [channels, kernel_size], need to broadcast to [1, channels, 1, kernel_size]
    let kernel = weight.unsqueeze(0)?.unsqueeze(2)?;

    // Broadcast multiply and sum over kernel dimension
    let out = windows.broadcast_mul(&kernel)?.sum(3)?;

    Ok(out)
}

/// Fused SwiGLU activation: silu(gate) * up
///
/// This fuses the SiLU activation and element-wise multiply into a single kernel,
/// reducing memory traffic by avoiding intermediate tensor writes.
///
/// # Arguments
/// * `gate` - Gate projection output
/// * `up` - Up projection output (same shape as gate)
///
/// # Returns
/// silu(gate) * up with same shape as inputs
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = gate.device() {
            return fused_swiglu_cuda(gate, up);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Device::Metal(_) = gate.device() {
            return fused_swiglu_metal(gate, up);
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = gate.device() {
            if !crate::vulkan_backend::debug::force_cpu_swiglu() {
                return fused_swiglu_vulkan(gate, up);
            }
        }
    }

    // CPU fallback - silu(x) = x * sigmoid(x)
    let gate_sigmoid = sigmoid(gate)?;
    let silu_gate = gate.mul(&gate_sigmoid)?;
    silu_gate.mul(up)
}

#[cfg(feature = "vulkan")]
fn fused_swiglu_vulkan(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    use ash::vk;
    use std::sync::Arc;

    let device = match gate.device() {
        Device::Vulkan(d) => d.clone(),
        _ => crate::bail!("fused_swiglu_vulkan: expected Vulkan device"),
    };

    let gate = gate.to_dtype(DType::F32)?.contiguous()?;
    let up = up.to_dtype(DType::F32)?.contiguous()?;

    let gate_buf = vulkan_buffer(&gate)?;
    let up_buf = vulkan_buffer(&up)?;

    let total_elems = gate.elem_count();

    // Push constants: 4 bytes (just total_elements)
    let pc: [u32; 1] = [total_elems as u32];

    let output_gpu = device.allocate_buffer((total_elems * 4) as vk::DeviceSize)?;

    let pipeline = device
        .kernels()
        .load_pipeline(device.device(), "fused_swiglu", None, 4, 3, None, false)
        .map_err(|e| crate::Error::Msg(format!("Failed to load fused_swiglu pipeline: {}", e)))?;

    device.record_compute_with_write_mask(
        &pipeline,
        &[gate_buf, up_buf, output_gpu.buffer],
        Some(0b100),
        Some(bytemuck::cast_slice(&pc)),
        [((total_elems as u32) + 255) / 256, 1, 1],
    )?;

    Ok(Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(output_gpu)),
            device,
            total_elems,
            DType::F32,
        )),
        gate.shape().clone(),
    ))
}

#[cfg(feature = "metal")]
fn fused_swiglu_metal(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    use crate::Storage;

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let gate_storage = gate.storage();
    let up_storage = up.storage();
    let gate_layout = gate.layout();
    let up_layout = up.layout();

    let (gate_metal, up_metal) = match (&*gate_storage, &*up_storage) {
        (Storage::Metal(g), Storage::Metal(u)) => (g, u),
        _ => crate::bail!("fused_swiglu_metal: tensors must be on Metal"),
    };

    let output_storage =
        crate::metal_backend::deltanet::fused_swiglu(gate_metal, gate_layout, up_metal, up_layout)?;

    let output = Tensor::from_storage(Storage::Metal(output_storage), gate.shape().clone());

    Ok(output)
}

#[cfg(feature = "cuda")]
fn fused_swiglu_cuda(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    use crate::Storage;

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let gate_storage = gate.storage();
    let up_storage = up.storage();
    let gate_layout = gate.layout();
    let up_layout = up.layout();

    let (gate_cuda, up_cuda) = match (&*gate_storage, &*up_storage) {
        (Storage::Cuda(g), Storage::Cuda(u)) => (g, u),
        _ => crate::bail!("fused_swiglu_cuda: tensors must be on CUDA"),
    };

    let output_storage =
        crate::cuda_backend::deltanet::fused_swiglu(gate_cuda, gate_layout, up_cuda, up_layout)?;

    let output = Tensor::from_storage(Storage::Cuda(output_storage), gate.shape().clone());

    Ok(output)
}

/// Fused gated RMS norm: silu(z) * rms_norm(x, weight)
///
/// This combines SiLU gate with RMS normalization into a single kernel,
/// reducing memory traffic.
///
/// # Arguments
/// * `x` - Input to normalize [batch, dim]
/// * `z` - Gate input [batch, dim] (same shape as x)
/// * `weight` - RMS norm weights [dim]
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// silu(z) * rms_norm(x, weight) with same shape as x
pub fn gated_rms_norm(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = x.device() {
            return gated_rms_norm_cuda(x, z, weight, eps);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Device::Metal(_) = x.device() {
            return gated_rms_norm_metal(x, z, weight, eps);
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = x.device() {
            if !crate::vulkan_backend::debug::force_cpu_gated_rmsnorm() {
                return gated_rms_norm_vulkan(x, z, weight, eps);
            }
        }
    }

    // CPU fallback
    gated_rms_norm_cpu(x, z, weight, eps)
}

#[cfg(feature = "vulkan")]
fn gated_rms_norm_vulkan(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use ash::vk;
    use std::sync::Arc;

    let device = match x.device() {
        Device::Vulkan(d) => d.clone(),
        _ => crate::bail!("gated_rms_norm_vulkan: expected Vulkan device"),
    };

    let original_dtype = x.dtype();

    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let z = z.to_dtype(DType::F32)?.contiguous()?;
    let weight = weight.to_dtype(DType::F32)?.contiguous()?;

    let x_buf = vulkan_buffer(&x)?;
    let z_buf = vulkan_buffer(&z)?;
    let weight_buf = vulkan_buffer(&weight)?;

    let x_dims = x.dims();
    let dim = *x_dims.last().unwrap();
    let num_rows = x.elem_count() / dim;

    // Push constants: 16 bytes
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GatedRmsNormPC {
        num_rows: u32,
        dim: u32,
        eps: f32,
        _pad: u32,
    }
    let pc = GatedRmsNormPC {
        num_rows: num_rows as u32,
        dim: dim as u32,
        eps,
        _pad: 0,
    };

    let output_elems = x.elem_count();
    let output_gpu = device.allocate_buffer((output_elems * 4) as vk::DeviceSize)?;

    let pipeline = device
        .kernels()
        .load_pipeline(device.device(), "gated_rms_norm", None, 16, 4, None, false)
        .map_err(|e| crate::Error::Msg(format!("Failed to load gated_rms_norm pipeline: {}", e)))?;

    device.record_compute_with_write_mask(
        &pipeline,
        &[x_buf, z_buf, weight_buf, output_gpu.buffer],
        Some(0b1000),
        Some(bytemuck::bytes_of(&pc)),
        [num_rows as u32, 1, 1],
    )?;

    let output = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(output_gpu)),
            device,
            output_elems,
            DType::F32,
        )),
        x.shape().clone(),
    );

    if original_dtype != DType::F32 {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

#[cfg(feature = "metal")]
fn gated_rms_norm_metal(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use crate::{DType, Storage};

    let original_dtype = x.dtype();

    // Metal kernel only supports f32
    let needs_conversion = original_dtype != DType::F32;
    let (x_work, z_work, weight_work) = if needs_conversion {
        (
            x.to_dtype(DType::F32)?,
            z.to_dtype(DType::F32)?,
            weight.to_dtype(DType::F32)?,
        )
    } else {
        (x.clone(), z.clone(), weight.clone())
    };

    let x_work = x_work.contiguous()?;
    let z_work = z_work.contiguous()?;
    let weight_work = weight_work.contiguous()?;
    let x_storage = x_work.storage();
    let z_storage = z_work.storage();
    let weight_storage = weight_work.storage();
    let x_layout = x_work.layout();
    let z_layout = z_work.layout();
    let weight_layout = weight_work.layout();

    let (x_metal, z_metal, weight_metal) = match (&*x_storage, &*z_storage, &*weight_storage) {
        (Storage::Metal(xs), Storage::Metal(zs), Storage::Metal(ws)) => (xs, zs, ws),
        _ => crate::bail!("gated_rms_norm_metal: tensors must be on Metal"),
    };

    let output_storage = crate::metal_backend::deltanet::gated_rms_norm(
        x_metal,
        x_layout,
        z_metal,
        z_layout,
        weight_metal,
        weight_layout,
        eps,
    )?;

    let output = Tensor::from_storage(Storage::Metal(output_storage), x.shape().clone());

    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

#[cfg(feature = "cuda")]
fn gated_rms_norm_cuda(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use crate::{DType, Storage};

    let original_dtype = x.dtype();

    // CUDA kernel only supports f32
    let needs_conversion = original_dtype != DType::F32;
    let (x_work, z_work, weight_work) = if needs_conversion {
        (
            x.to_dtype(DType::F32)?,
            z.to_dtype(DType::F32)?,
            weight.to_dtype(DType::F32)?,
        )
    } else {
        (x.clone(), z.clone(), weight.clone())
    };

    let x_work = x_work.contiguous()?;
    let z_work = z_work.contiguous()?;
    let weight_work = weight_work.contiguous()?;
    let x_storage = x_work.storage();
    let z_storage = z_work.storage();
    let weight_storage = weight_work.storage();
    let x_layout = x_work.layout();
    let z_layout = z_work.layout();
    let weight_layout = weight_work.layout();

    let (x_cuda, z_cuda, weight_cuda) = match (&*x_storage, &*z_storage, &*weight_storage) {
        (Storage::Cuda(xs), Storage::Cuda(zs), Storage::Cuda(ws)) => (xs, zs, ws),
        _ => crate::bail!("gated_rms_norm_cuda: tensors must be on CUDA"),
    };

    let output_storage = crate::cuda_backend::deltanet::gated_rms_norm(
        x_cuda,
        x_layout,
        z_cuda,
        z_layout,
        weight_cuda,
        weight_layout,
        eps,
    )?;

    let output = Tensor::from_storage(Storage::Cuda(output_storage), x.shape().clone());

    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

/// Multi-token Delta Net state update for MTP/speculative decoding.
///
/// Processes K tokens in parallel, mathematically equivalent to K sequential
/// autoregressive steps. This is used after MTP verification to update the
/// DeltaNet recurrent state for all accepted tokens in one efficient pass.
///
/// # Arguments
/// * `q` - Query tensor [batch, num_heads, num_tokens, head_dim] (L2 normalized WITH scale applied)
/// * `k` - Key tensor [batch, num_heads, num_tokens, head_dim] (L2 normalized)
/// * `v` - Value tensor [batch, num_heads, num_tokens, head_dim]
/// * `gate` - Log decay values [batch, num_heads, num_tokens]
/// * `beta` - Gate values [batch, num_heads, num_tokens] (already sigmoided)
/// * `state` - Initial recurrent state [batch, num_heads, head_dim, head_dim]
///
/// # Returns
/// Tuple of (output, new_state) where:
/// * `output` - [batch, num_heads, num_tokens, head_dim]
/// * `new_state` - [batch, num_heads, head_dim, head_dim]
pub fn delta_net_multi_token_update(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Validate shapes
    let q_dims = q.dims();
    if q_dims.len() != 4 {
        crate::bail!(
            "delta_net_multi_token_update: q must be 4D [batch, heads, tokens, dim], got {:?}",
            q_dims
        );
    }

    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let num_tokens = q_dims[2];
    let head_dim = q_dims[3];

    // Ensure all tensors are contiguous
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let gate = gate.contiguous()?;
    let beta = beta.contiguous()?;
    let state = state.contiguous()?;

    #[cfg(feature = "cuda")]
    {
        use crate::{Shape, Storage};

        if let Device::Cuda(_) = q.device() {
            let q_storage = q.storage();
            let k_storage = k.storage();
            let v_storage = v.storage();
            let gate_storage = gate.storage();
            let beta_storage = beta.storage();
            let state_storage = state.storage();

            let q_layout = q.layout();
            let k_layout = k.layout();
            let v_layout = v.layout();
            let gate_layout = gate.layout();
            let beta_layout = beta.layout();
            let state_layout = state.layout();

            let (q_cuda, k_cuda, v_cuda, gate_cuda, beta_cuda, state_cuda) = match (
                &*q_storage,
                &*k_storage,
                &*v_storage,
                &*gate_storage,
                &*beta_storage,
                &*state_storage,
            ) {
                (
                    Storage::Cuda(q_s),
                    Storage::Cuda(k_s),
                    Storage::Cuda(v_s),
                    Storage::Cuda(gate_s),
                    Storage::Cuda(beta_s),
                    Storage::Cuda(state_s),
                ) => (q_s, k_s, v_s, gate_s, beta_s, state_s),
                _ => crate::bail!("delta_net_multi_token_update: all tensors must be on CUDA"),
            };

            let (output_storage, new_state_storage) =
                crate::cuda_backend::deltanet::delta_net_multi_token_update(
                    q_cuda,
                    q_layout,
                    k_cuda,
                    k_layout,
                    v_cuda,
                    v_layout,
                    gate_cuda,
                    gate_layout,
                    beta_cuda,
                    beta_layout,
                    state_cuda,
                    state_layout,
                )?;

            let output_shape = Shape::from((batch, num_heads, num_tokens, head_dim));
            let state_shape = Shape::from((batch, num_heads, head_dim, head_dim));

            let output = Tensor::from_storage(Storage::Cuda(output_storage), output_shape);

            let new_state = Tensor::from_storage(Storage::Cuda(new_state_storage), state_shape);

            return Ok((output, new_state));
        }
    }

    #[cfg(feature = "metal")]
    {
        use crate::{Shape, Storage};

        if let Device::Metal(_) = q.device() {
            let q_storage = q.storage();
            let k_storage = k.storage();
            let v_storage = v.storage();
            let gate_storage = gate.storage();
            let beta_storage = beta.storage();
            let state_storage = state.storage();

            let q_layout = q.layout();
            let k_layout = k.layout();
            let v_layout = v.layout();
            let gate_layout = gate.layout();
            let beta_layout = beta.layout();
            let state_layout = state.layout();

            let (q_metal, k_metal, v_metal, gate_metal, beta_metal, state_metal) = match (
                &*q_storage,
                &*k_storage,
                &*v_storage,
                &*gate_storage,
                &*beta_storage,
                &*state_storage,
            ) {
                (
                    Storage::Metal(q_s),
                    Storage::Metal(k_s),
                    Storage::Metal(v_s),
                    Storage::Metal(gate_s),
                    Storage::Metal(beta_s),
                    Storage::Metal(state_s),
                ) => (q_s, k_s, v_s, gate_s, beta_s, state_s),
                _ => crate::bail!("delta_net_multi_token_update: all tensors must be on Metal"),
            };

            let (output_storage, new_state_storage) =
                crate::metal_backend::deltanet::delta_net_multi_token_update(
                    q_metal,
                    q_layout,
                    k_metal,
                    k_layout,
                    v_metal,
                    v_layout,
                    gate_metal,
                    gate_layout,
                    beta_metal,
                    beta_layout,
                    state_metal,
                    state_layout,
                )?;

            let output_shape = Shape::from((batch, num_heads, num_tokens, head_dim));
            let state_shape = Shape::from((batch, num_heads, head_dim, head_dim));

            let output = Tensor::from_storage(Storage::Metal(output_storage), output_shape);

            let new_state = Tensor::from_storage(Storage::Metal(new_state_storage), state_shape);

            return Ok((output, new_state));
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = q.device() {
            if !crate::vulkan_backend::debug::force_cpu_deltanet_mtp() {
                // Reuse the parallel_with_states implementation and discard intermediate states
                let (output, new_state, _inter_states) = delta_net_parallel_with_states_vulkan(
                    &q, &k, &v, &gate, &beta, &state, batch, num_heads, num_tokens, head_dim,
                )?;
                return Ok((output, new_state));
            }
        }
    }

    // CPU fallback: process tokens sequentially
    delta_net_multi_token_update_cpu(
        &q, &k, &v, &gate, &beta, &state, batch, num_heads, num_tokens, head_dim,
    )
}

#[allow(clippy::too_many_arguments)]
fn delta_net_multi_token_update_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    _batch: usize,
    _num_heads: usize,
    num_tokens: usize,
    _head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    // Process tokens sequentially on CPU
    let mut current_state = state.clone();
    let mut outputs = Vec::with_capacity(num_tokens);

    for t in 0..num_tokens {
        // Extract token t
        let q_t = q.narrow(2, t, 1)?.squeeze(2)?; // [batch, heads, dim]
        let k_t = k.narrow(2, t, 1)?.squeeze(2)?;
        let v_t = v.narrow(2, t, 1)?.squeeze(2)?;
        let gate_t = gate.narrow(2, t, 1)?.squeeze(2)?; // [batch, heads]
        let beta_t = beta.narrow(2, t, 1)?.squeeze(2)?;

        // Use the autoregressive step logic
        let g_t = gate_t.exp()?.unsqueeze(2)?.unsqueeze(3)?;
        let decayed_state = current_state.broadcast_mul(&g_t)?;

        let k_expanded = k_t.unsqueeze(2)?;
        let kv_mem = decayed_state.broadcast_mul(&k_expanded)?.sum(D::Minus1)?;

        let beta_expanded = beta_t.unsqueeze(2)?;
        let delta = (v_t.clone() - kv_mem)?.broadcast_mul(&beta_expanded)?;

        let delta_col = delta.unsqueeze(3)?;
        let k_row = k_t.unsqueeze(2)?;
        let k_delta_outer = delta_col.matmul(&k_row)?;
        current_state = (decayed_state + k_delta_outer)?;

        // q is already normalized and scaled, so just use it directly
        let q_expanded = q_t.unsqueeze(2)?;
        let output_t = current_state.broadcast_mul(&q_expanded)?.sum(D::Minus1)?;
        outputs.push(output_t.unsqueeze(2)?);
    }

    let output = Tensor::cat(&outputs, 2)?;
    Ok((output, current_state))
}

/// Multi-token Delta Net with PARALLEL intermediate state materialization.
///
/// Same as `delta_net_multi_token_update` but also materializes and returns
/// the state after each position, enabling O(1) state slicing for speculative
/// decoding verification.
///
/// # Arguments
/// * `q` - Query tensor [batch, num_heads, num_tokens, head_dim] (L2 normalized WITH scale applied)
/// * `k` - Key tensor [batch, num_heads, num_tokens, head_dim] (L2 normalized)
/// * `v` - Value tensor [batch, num_heads, num_tokens, head_dim]
/// * `gate` - Log decay values [batch, num_heads, num_tokens]
/// * `beta` - Gate values [batch, num_heads, num_tokens] (already sigmoided)
/// * `state` - Initial recurrent state [batch, num_heads, head_dim, head_dim]
///
/// # Returns
/// Tuple of (output, new_state, intermediate_states) where:
/// * `output` - [batch, num_heads, num_tokens, head_dim]
/// * `new_state` - [batch, num_heads, head_dim, head_dim] (final state)
/// * `intermediate_states` - [batch, num_heads, num_tokens, head_dim, head_dim] (state after each token)
///
/// # State Slicing
/// On partial rejection at position i, simply use:
/// `intermediate_states.narrow(2, i, 1).squeeze(2)` as the new state.
pub fn delta_net_parallel_with_states(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Validate shapes
    let q_dims = q.dims();
    if q_dims.len() != 4 {
        crate::bail!(
            "delta_net_parallel_with_states: q must be 4D [batch, heads, tokens, dim], got {:?}",
            q_dims
        );
    }

    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let num_tokens = q_dims[2];
    let head_dim = q_dims[3];

    // Ensure all tensors are contiguous
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let gate = gate.contiguous()?;
    let beta = beta.contiguous()?;
    let state = state.contiguous()?;

    #[cfg(feature = "cuda")]
    {
        use crate::{Shape, Storage};

        if let Device::Cuda(_) = q.device() {
            let q_storage = q.storage();
            let k_storage = k.storage();
            let v_storage = v.storage();
            let gate_storage = gate.storage();
            let beta_storage = beta.storage();
            let state_storage = state.storage();

            let q_layout = q.layout();
            let k_layout = k.layout();
            let v_layout = v.layout();
            let gate_layout = gate.layout();
            let beta_layout = beta.layout();
            let state_layout = state.layout();

            let (q_cuda, k_cuda, v_cuda, gate_cuda, beta_cuda, state_cuda) = match (
                &*q_storage,
                &*k_storage,
                &*v_storage,
                &*gate_storage,
                &*beta_storage,
                &*state_storage,
            ) {
                (
                    Storage::Cuda(q_s),
                    Storage::Cuda(k_s),
                    Storage::Cuda(v_s),
                    Storage::Cuda(gate_s),
                    Storage::Cuda(beta_s),
                    Storage::Cuda(state_s),
                ) => (q_s, k_s, v_s, gate_s, beta_s, state_s),
                _ => crate::bail!("delta_net_parallel_with_states: all tensors must be on CUDA"),
            };

            let (output_storage, new_state_storage, inter_states_storage) =
                crate::cuda_backend::deltanet::delta_net_parallel_with_states(
                    q_cuda,
                    q_layout,
                    k_cuda,
                    k_layout,
                    v_cuda,
                    v_layout,
                    gate_cuda,
                    gate_layout,
                    beta_cuda,
                    beta_layout,
                    state_cuda,
                    state_layout,
                )?;

            let output_shape = Shape::from((batch, num_heads, num_tokens, head_dim));
            let state_shape = Shape::from((batch, num_heads, head_dim, head_dim));
            let inter_state_shape = Shape::from((batch, num_heads, num_tokens, head_dim, head_dim));

            let output = Tensor::from_storage(Storage::Cuda(output_storage), output_shape);

            let new_state = Tensor::from_storage(Storage::Cuda(new_state_storage), state_shape);

            let intermediate_states =
                Tensor::from_storage(Storage::Cuda(inter_states_storage), inter_state_shape);

            return Ok((output, new_state, intermediate_states));
        }
    }

    #[cfg(feature = "metal")]
    {
        use crate::{Shape, Storage};

        if let Device::Metal(_) = q.device() {
            let q_storage = q.storage();
            let k_storage = k.storage();
            let v_storage = v.storage();
            let gate_storage = gate.storage();
            let beta_storage = beta.storage();
            let state_storage = state.storage();

            let q_layout = q.layout();
            let k_layout = k.layout();
            let v_layout = v.layout();
            let gate_layout = gate.layout();
            let beta_layout = beta.layout();
            let state_layout = state.layout();

            let (q_metal, k_metal, v_metal, gate_metal, beta_metal, state_metal) = match (
                &*q_storage,
                &*k_storage,
                &*v_storage,
                &*gate_storage,
                &*beta_storage,
                &*state_storage,
            ) {
                (
                    Storage::Metal(q_s),
                    Storage::Metal(k_s),
                    Storage::Metal(v_s),
                    Storage::Metal(gate_s),
                    Storage::Metal(beta_s),
                    Storage::Metal(state_s),
                ) => (q_s, k_s, v_s, gate_s, beta_s, state_s),
                _ => crate::bail!("delta_net_parallel_with_states: all tensors must be on Metal"),
            };

            let (output_storage, new_state_storage, inter_states_storage) =
                crate::metal_backend::deltanet::delta_net_parallel_with_states(
                    q_metal,
                    q_layout,
                    k_metal,
                    k_layout,
                    v_metal,
                    v_layout,
                    gate_metal,
                    gate_layout,
                    beta_metal,
                    beta_layout,
                    state_metal,
                    state_layout,
                )?;

            let output_shape = Shape::from((batch, num_heads, num_tokens, head_dim));
            let state_shape = Shape::from((batch, num_heads, head_dim, head_dim));
            let inter_state_shape = Shape::from((batch, num_heads, num_tokens, head_dim, head_dim));

            let output = Tensor::from_storage(Storage::Metal(output_storage), output_shape);

            let new_state = Tensor::from_storage(Storage::Metal(new_state_storage), state_shape);

            let intermediate_states =
                Tensor::from_storage(Storage::Metal(inter_states_storage), inter_state_shape);

            return Ok((output, new_state, intermediate_states));
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = q.device() {
            if !crate::vulkan_backend::debug::force_cpu_deltanet_parallel() {
                return delta_net_parallel_with_states_vulkan(
                    &q, &k, &v, &gate, &beta, &state, batch, num_heads, num_tokens, head_dim,
                );
            }
        }
    }

    // CPU fallback: process tokens sequentially and save states
    delta_net_parallel_with_states_cpu(
        &q, &k, &v, &gate, &beta, &state, batch, num_heads, num_tokens, head_dim,
    )
}

#[cfg(feature = "vulkan")]
fn delta_net_parallel_with_states_vulkan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    batch: usize,
    num_heads: usize,
    num_tokens: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    use std::sync::Arc;

    let device = match q.device() {
        Device::Vulkan(d) => d.clone(),
        _ => crate::bail!("delta_net_parallel_with_states_vulkan: expected Vulkan device"),
    };

    // Ensure f32 and contiguous
    let q = q.to_dtype(DType::F32)?.contiguous()?;
    let k = k.to_dtype(DType::F32)?.contiguous()?;
    let v = v.to_dtype(DType::F32)?.contiguous()?;
    let gate = gate.to_dtype(DType::F32)?.contiguous()?;
    let beta = beta.to_dtype(DType::F32)?.contiguous()?;
    let state = state.to_dtype(DType::F32)?.contiguous()?;

    let (q_buf, q_buf_arc) = vulkan_buffer_with_arc(&q)?;
    let (k_buf, k_buf_arc) = vulkan_buffer_with_arc(&k)?;
    let (v_buf, v_buf_arc) = vulkan_buffer_with_arc(&v)?;
    let (gate_buf, gate_buf_arc) = vulkan_buffer_with_arc(&gate)?;
    let (beta_buf, beta_buf_arc) = vulkan_buffer_with_arc(&beta)?;
    let (state_buf, state_buf_arc) = vulkan_buffer_with_arc(&state)?;

    // Push constants: 16 bytes
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct DeltaNetParallelPC {
        batch: u32,
        num_heads: u32,
        num_tokens: u32,
        head_dim: u32,
    }
    let pc = DeltaNetParallelPC {
        batch: batch as u32,
        num_heads: num_heads as u32,
        num_tokens: num_tokens as u32,
        head_dim: head_dim as u32,
    };

    // Allocate dedicated output buffers directly (no post-dispatch split copy).
    let output_elems = batch * num_heads * num_tokens * head_dim;
    let state_elems = batch * num_heads * head_dim * head_dim;
    let output_gpu = Arc::new(device.allocate_buffer((output_elems * 4) as u64)?);
    let new_state_gpu = Arc::new(device.allocate_buffer((state_elems * 4) as u64)?);

    // Intermediate states buffer: [batch, heads, tokens, dim, dim]
    let inter_state_elems = batch * num_heads * num_tokens * head_dim * head_dim;
    let inter_states_gpu = Arc::new(device.allocate_buffer((inter_state_elems * 4) as u64)?);

    // Check if cooperative matrix is available and not disabled
    let has_coopmat = device.has_cooperative_matrix();
    let use_coopmat = has_coopmat && !is_deltanet_parallel_coopmat_disabled(head_dim);

    // Select shader
    let shader_name = if use_coopmat {
        format!("delta_net_parallel_d{}_coopmat", head_dim)
    } else {
        format!("delta_net_parallel_d{}", head_dim)
    };

    // Get tile sizes and specialization constants for coopmat
    let specialization_constants = if use_coopmat {
        let (tile_m, tile_n, tile_k) = device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
        Some(vec![tile_m, tile_n, tile_k])
    } else {
        None
    };

    // Try to load the coopmat pipeline, fall back to standard on failure
    let pipeline = if use_coopmat {
        match device.kernels().load_pipeline(
            device.device(),
            &shader_name,
            None,
            16,
            9,
            specialization_constants.as_ref().map(|v| v.as_slice()),
            false,
        ) {
            Ok(p) => p,
            Err(_) => {
                // Fall back to standard shader
                let fallback_name = format!("delta_net_parallel_d{}", head_dim);
                device
                    .kernels()
                    .load_pipeline(device.device(), &fallback_name, None, 16, 9, None, false)
                    .map_err(|e| {
                        crate::Error::Msg(format!(
                            "Failed to load delta_net_parallel pipeline: {}",
                            e
                        ))
                    })?
            }
        }
    } else {
        device
            .kernels()
            .load_pipeline(device.device(), &shader_name, None, 16, 9, None, false)
            .map_err(|e| {
                crate::Error::Msg(format!("Failed to load delta_net_parallel pipeline: {}", e))
            })?
    };

    // Dispatch with push constants
    device.record_compute_with_write_mask(
        &pipeline,
        &[
            q_buf,
            k_buf,
            v_buf,
            gate_buf,
            beta_buf,
            state_buf,
            output_gpu.buffer,
            new_state_gpu.buffer,
            inter_states_gpu.buffer,
        ],
        Some((1u64 << 6) | (1u64 << 7) | (1u64 << 8)),
        Some(bytemuck::bytes_of(&pc)),
        [batch as u32, num_heads as u32, 1],
    )?;

    // Keep source buffers alive until this batch is submitted.
    device.keep_buffer_alive(q_buf_arc)?;
    device.keep_buffer_alive(k_buf_arc)?;
    device.keep_buffer_alive(v_buf_arc)?;
    device.keep_buffer_alive(gate_buf_arc)?;
    device.keep_buffer_alive(beta_buf_arc)?;
    device.keep_buffer_alive(state_buf_arc)?;

    // Create output tensors
    let output_shape = Shape::from((batch, num_heads, num_tokens, head_dim));
    let state_shape = Shape::from((batch, num_heads, head_dim, head_dim));
    let inter_state_shape = Shape::from((batch, num_heads, num_tokens, head_dim, head_dim));

    let output = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(output_gpu),
            device.clone(),
            output_elems,
            DType::F32,
        )),
        output_shape,
    );
    let new_state = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(new_state_gpu),
            device.clone(),
            state_elems,
            DType::F32,
        )),
        state_shape,
    );
    let intermediate_states = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(inter_states_gpu),
            device.clone(),
            inter_state_elems,
            DType::F32,
        )),
        inter_state_shape,
    );

    Ok((output, new_state, intermediate_states))
}

#[allow(clippy::too_many_arguments)]
fn delta_net_parallel_with_states_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    _batch: usize,
    _num_heads: usize,
    num_tokens: usize,
    _head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Process tokens sequentially on CPU, saving intermediate states
    let mut current_state = state.clone();
    let mut outputs = Vec::with_capacity(num_tokens);
    let mut intermediate_states = Vec::with_capacity(num_tokens);

    for t in 0..num_tokens {
        // Extract token t
        let q_t = q.narrow(2, t, 1)?.squeeze(2)?; // [batch, heads, dim]
        let k_t = k.narrow(2, t, 1)?.squeeze(2)?;
        let v_t = v.narrow(2, t, 1)?.squeeze(2)?;
        let gate_t = gate.narrow(2, t, 1)?.squeeze(2)?; // [batch, heads]
        let beta_t = beta.narrow(2, t, 1)?.squeeze(2)?;

        // Autoregressive step
        let g_t = gate_t.exp()?.unsqueeze(2)?.unsqueeze(3)?;
        let decayed_state = current_state.broadcast_mul(&g_t)?;

        let k_expanded = k_t.unsqueeze(2)?;
        let kv_mem = decayed_state.broadcast_mul(&k_expanded)?.sum(D::Minus1)?;

        let beta_expanded = beta_t.unsqueeze(2)?;
        let delta = (v_t.clone() - kv_mem)?.broadcast_mul(&beta_expanded)?;

        let delta_col = delta.unsqueeze(3)?;
        let k_row = k_t.unsqueeze(2)?;
        let k_delta_outer = delta_col.matmul(&k_row)?;
        current_state = (decayed_state + k_delta_outer)?;

        // Save intermediate state
        intermediate_states.push(current_state.unsqueeze(2)?);

        // Compute output
        let q_expanded = q_t.unsqueeze(2)?;
        let output_t = current_state.broadcast_mul(&q_expanded)?.sum(D::Minus1)?;
        outputs.push(output_t.unsqueeze(2)?);
    }

    let output = Tensor::cat(&outputs, 2)?;
    let inter_states = Tensor::cat(&intermediate_states, 2)?;
    Ok((output, current_state, inter_states))
}

/// Top-k routing softmax for Mixture-of-Experts routing.
///
/// Input logits shape: `[n_tokens, n_experts]`.
/// Returns `(weights, indices)` with shapes `[n_tokens, topk]` where weights are
/// normalized across the selected top-k experts.
pub fn topk_softmax(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let (n_tokens, n_experts) = logits.dims2()?;

    if topk == 0 {
        crate::bail!("topk_softmax: topk must be > 0");
    }
    if topk > n_experts {
        crate::bail!(
            "topk_softmax: topk {} exceeds number of experts {}",
            topk,
            n_experts
        );
    }
    if n_tokens == 0 {
        let weights = Tensor::zeros((0, topk), crate::DType::F32, logits.device())?;
        let indices = Tensor::zeros((0, topk), crate::DType::U32, logits.device())?;
        return Ok((weights, indices));
    }

    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = logits.device() {
            return topk_softmax_cuda(logits, topk);
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = logits.device() {
            return topk_softmax_vulkan(logits, topk);
        }
    }

    topk_softmax_cpu(logits, topk)
}

fn topk_softmax_cpu(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let logits = logits.to_dtype(crate::DType::F32)?.contiguous()?;
    let max_vals = logits.max_keepdim(D::Minus1)?;
    let exp_vals = logits.broadcast_sub(&max_vals)?.exp()?;
    let sum_exp = exp_vals.sum_keepdim(D::Minus1)?;
    let probs = exp_vals.broadcast_div(&sum_exp)?;

    let (sorted_vals, sorted_idx) = probs.sort_last_dim(false)?;
    let top_weights = sorted_vals.narrow(1, 0, topk)?;
    let top_indices = sorted_idx.narrow(1, 0, topk)?.to_dtype(crate::DType::U32)?;

    let selected_sum = top_weights.sum_keepdim(D::Minus1)?;
    let normalized_weights = top_weights.broadcast_div(&selected_sum)?;
    Ok((normalized_weights, top_indices))
}

#[cfg(feature = "cuda")]
fn topk_softmax_cuda(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    use crate::Storage;

    let logits = logits.to_dtype(crate::DType::F32)?.contiguous()?;
    let logits_storage = logits.storage();
    let logits_layout = logits.layout();

    let logits_cuda = match &*logits_storage {
        Storage::Cuda(s) => s,
        _ => crate::bail!("topk_softmax_cuda: tensor must be on CUDA"),
    };

    let (weights_storage, indices_storage, weights_shape, indices_shape) =
        crate::cuda_backend::deltanet::topk_softmax(logits_cuda, logits_layout, topk)?;

    let weights = Tensor::from_storage(Storage::Cuda(weights_storage), weights_shape);
    let indices_i32 = Tensor::from_storage(Storage::Cuda(indices_storage), indices_shape);
    let indices = indices_i32.to_dtype(crate::DType::U32)?;
    Ok((weights, indices))
}

#[cfg(feature = "vulkan")]
fn topk_softmax_vulkan(logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    use std::sync::Arc;

    let device = match logits.device() {
        Device::Vulkan(d) => d.clone(),
        _ => crate::bail!("topk_softmax_vulkan: expected Vulkan device"),
    };

    let logits_contiguous = logits.contiguous()?;
    let mut logits_input = logits_contiguous.clone();
    let mut kernel_name = match logits_input.dtype() {
        DType::F16 => "topk_softmax_f16",
        DType::BF16 => "topk_softmax_bf16",
        DType::F32 => "topk_softmax_f32",
        _ => {
            logits_input = logits_input.to_dtype(DType::F32)?.contiguous()?;
            "topk_softmax_f32"
        }
    };

    let load_pipeline = |name: &str| {
        device.kernels().load_pipeline(
            device.device(),
            name,
            None,
            (std::mem::size_of::<u32>() * 5) as u32,
            3,
            None,
            false,
        )
    };

    let mut pipeline = load_pipeline(kernel_name).map_err(|e| {
        crate::Error::Msg(format!(
            "Failed to load topk_softmax pipeline '{}': {}",
            kernel_name, e
        ))
    });

    if pipeline.is_err() && kernel_name != "topk_softmax_f32" {
        logits_input = logits_contiguous.to_dtype(DType::F32)?.contiguous()?;
        kernel_name = "topk_softmax_f32";
        pipeline = load_pipeline(kernel_name).map_err(|e| {
            crate::Error::Msg(format!(
                "Failed to load topk_softmax fallback pipeline '{}': {}",
                kernel_name, e
            ))
        });
    }

    let pipeline = pipeline?;

    let logits_layout = logits_input.layout();
    let (n_tokens, n_experts) = logits.dims2()?;

    if topk > 32 {
        crate::bail!("topk_softmax_vulkan: topk {} exceeds kernel limit 32", topk);
    }

    let logits_base = u32::try_from(logits_layout.start_offset()).map_err(|_| {
        crate::Error::Msg("topk_softmax_vulkan: logits start offset exceeds u32".into())
    })?;
    let n_tokens_u32 = u32::try_from(n_tokens)
        .map_err(|_| crate::Error::Msg("topk_softmax_vulkan: n_tokens exceeds u32".into()))?;
    let n_experts_u32 = u32::try_from(n_experts)
        .map_err(|_| crate::Error::Msg("topk_softmax_vulkan: n_experts exceeds u32".into()))?;
    let topk_u32 = u32::try_from(topk)
        .map_err(|_| crate::Error::Msg("topk_softmax_vulkan: topk exceeds u32".into()))?;

    let out_elems = n_tokens
        .checked_mul(topk)
        .ok_or_else(|| crate::Error::Msg("topk_softmax_vulkan: output size overflow".into()))?;

    let weights_buf = device.allocate_buffer((out_elems * std::mem::size_of::<f32>()) as u64)?;
    let indices_buf = device.allocate_buffer((out_elems * std::mem::size_of::<u32>()) as u64)?;
    let logits_buf = vulkan_buffer(&logits_input)?;

    let x_groups = n_tokens_u32.min(65535);
    let y_groups = if x_groups == 0 {
        1
    } else {
        n_tokens_u32.div_ceil(x_groups)
    };

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct TopkSoftmaxPC {
        logits_base: u32,
        n_tokens: u32,
        n_experts: u32,
        topk: u32,
        x_groups: u32,
    }

    let pc = TopkSoftmaxPC {
        logits_base,
        n_tokens: n_tokens_u32,
        n_experts: n_experts_u32,
        topk: topk_u32,
        x_groups,
    };

    device.record_compute_with_write_mask(
        &pipeline,
        &[logits_buf, weights_buf.buffer, indices_buf.buffer],
        Some((1u64 << 1) | (1u64 << 2)),
        Some(bytemuck::bytes_of(&pc)),
        [x_groups, y_groups, 1],
    )?;

    let weights = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(weights_buf)),
            device.clone(),
            out_elems,
            DType::F32,
        )),
        Shape::from((n_tokens, topk)),
    );
    let indices = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(indices_buf)),
            device,
            out_elems,
            DType::U32,
        )),
        Shape::from((n_tokens, topk)),
    );
    Ok((weights, indices))
}

/// GPU-side computation of P(x) and Q(x) for draft tokens.
///
/// Computes the probability of each draft token under both target and draft
/// distributions without moving full vocabulary logits to CPU.
///
/// # Arguments
/// * `target_logits` - Target model logits [num_drafts, vocab_size]
/// * `draft_logits` - Draft model logits [num_drafts, vocab_size]  
/// * `draft_tokens` - Drafted token IDs [num_drafts] (i32)
/// * `temperature` - Temperature for softmax
///
/// # Returns
/// (p_values, q_values) - P(x) and Q(x) for each draft token
pub fn compute_draft_probs(
    _target_logits: &Tensor,
    _draft_logits: &Tensor,
    draft_tokens: &Tensor,
    _temperature: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    #[cfg(feature = "cuda")]
    {
        use crate::Storage;

        if let Device::Cuda(_) = _target_logits.device() {
            let target_logits = _target_logits.contiguous()?;
            let draft_logits = _draft_logits.contiguous()?;
            let draft_tokens = draft_tokens.contiguous()?;

            let target_storage = target_logits.storage();
            let draft_storage = draft_logits.storage();
            let tokens_storage = draft_tokens.storage();

            let (target_cuda, draft_cuda, tokens_cuda) =
                match (&*target_storage, &*draft_storage, &*tokens_storage) {
                    (Storage::Cuda(t), Storage::Cuda(d), Storage::Cuda(tok)) => (t, d, tok),
                    _ => crate::bail!("compute_draft_probs: all tensors must be on CUDA"),
                };

            return crate::cuda_backend::deltanet::compute_draft_probs(
                target_cuda,
                target_logits.layout(),
                draft_cuda,
                draft_logits.layout(),
                tokens_cuda,
                draft_tokens.layout(),
                _temperature,
            );
        }
    }

    // CPU fallback - return 1.0 for all (accept all)
    let num_drafts = draft_tokens.dims()[0];
    Ok((vec![1.0; num_drafts], vec![1.0; num_drafts]))
}

fn gated_rms_norm_cpu(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // RMS norm: x / sqrt(mean(x^2) + eps) * weight
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps as f64)?.sqrt()?;
    let normalized = x.broadcast_div(&rms)?;
    let scaled = normalized.broadcast_mul(weight)?;

    // SiLU(z) * scaled
    let z_silu = silu(z)?;
    z_silu.mul(&scaled)
}

/// Lower triangular solve for (I - L) @ X = B (DeltaNet chunked algorithm)
///
/// Solves (I - L) @ X = B where L is strictly lower triangular.
/// Since (I - L) has 1s on the diagonal, no division is needed.
/// Formula: X[i] = B[i] + sum_{j<i} L[i,j] * X[j]
///
/// This uses a GPU kernel that matches the precision characteristics of
/// the autoregressive DeltaNet kernel for consistent state updates.
///
/// # Arguments
/// * `l_matrix` - Strictly lower triangular matrix [batch, n, n]
/// * `b_matrix` - Right-hand side matrix [batch, n, k]
///
/// # Returns
/// Solution X [batch, n, k]
pub fn solve_i_minus_lower_triangular(l_matrix: &Tensor, b_matrix: &Tensor) -> Result<Tensor> {
    let l_dims = l_matrix.dims();
    let b_dims = b_matrix.dims();

    if l_dims.len() < 2 || b_dims.len() < 2 {
        crate::bail!(
            "solve_i_minus_lower_triangular: need at least 2D, got {:?} and {:?}",
            l_dims,
            b_dims
        );
    }

    // Ensure contiguous for CUDA kernel
    let l_matrix = l_matrix.contiguous()?;
    let b_matrix = b_matrix.contiguous()?;

    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = l_matrix.device() {
            return solve_i_minus_lower_triangular_cuda(&l_matrix, &b_matrix);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Device::Metal(_) = l_matrix.device() {
            return solve_i_minus_lower_triangular_metal(&l_matrix, &b_matrix);
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let Device::Vulkan(_) = l_matrix.device() {
            return solve_i_minus_lower_triangular_vulkan(&l_matrix, &b_matrix);
        }
    }

    // CPU fallback
    solve_i_minus_lower_triangular_cpu(&l_matrix, &b_matrix)
}

#[cfg(feature = "vulkan")]
fn solve_i_minus_lower_triangular_vulkan(l_matrix: &Tensor, b_matrix: &Tensor) -> Result<Tensor> {
    use ash::vk;
    use std::sync::Arc;

    // Vulkan shader currently operates on f32.
    let original_dtype = l_matrix.dtype();
    let needs_conversion = !matches!(original_dtype, DType::F32);

    let l_work = if needs_conversion {
        l_matrix.to_dtype(DType::F32)?
    } else {
        l_matrix.clone()
    };
    let b_work = if needs_conversion {
        b_matrix.to_dtype(DType::F32)?
    } else {
        b_matrix.clone()
    };

    let device = match l_work.device() {
        Device::Vulkan(d) => d.clone(),
        _ => crate::bail!("solve_i_minus_lower_triangular_vulkan: expected Vulkan device"),
    };

    let l_dims = l_work.dims();
    let b_dims = b_work.dims();
    let n = l_dims[l_dims.len() - 1];
    let k = b_dims[b_dims.len() - 1];
    let batch: usize = l_dims[..l_dims.len() - 2].iter().product::<usize>().max(1);
    let output_elems = b_work.elem_count();
    let output_bytes = (output_elems * std::mem::size_of::<f32>()) as vk::DeviceSize;

    let l_buf = vulkan_buffer(&l_work)?;
    let b_buf = vulkan_buffer(&b_work)?;
    let x_buf = device.allocate_buffer(output_bytes)?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct SolveTriPC {
        batch_size: u32,
        n: u32,
        k: u32,
        row: u32,
    }

    let pipeline = device
        .kernels()
        .load_pipeline(
            device.device(),
            "solve_triangular",
            None,
            16,
            3,
            None,
            false,
        )
        .map_err(|e| {
            crate::Error::Msg(format!("Failed to load solve_triangular pipeline: {}", e))
        })?;

    let work_items = batch.checked_mul(k).ok_or_else(|| {
        crate::Error::Msg("solve_i_minus_lower_triangular_vulkan: dispatch size overflow".into())
    })?;
    let groups_x = ((work_items + 255) / 256) as u32;
    let groups_x = groups_x.max(1);

    for row in 0..n {
        let pc = SolveTriPC {
            batch_size: batch as u32,
            n: n as u32,
            k: k as u32,
            row: row as u32,
        };

        // Binding 2 (X) is read+write across row steps; mark it writable so
        // the batched recorder inserts compute barriers between dispatches.
        device.record_compute_with_write_mask(
            &pipeline,
            &[l_buf, b_buf, x_buf.buffer],
            Some(1u64 << 2),
            Some(bytemuck::bytes_of(&pc)),
            [groups_x, 1, 1],
        )?;
    }

    device.flush()?;

    let output = Tensor::from_storage(
        Storage::Vulkan(VulkanStorage::new(
            Some(Arc::new(x_buf)),
            device,
            output_elems,
            DType::F32,
        )),
        b_work.shape().clone(),
    );

    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

#[cfg(feature = "cuda")]
fn solve_i_minus_lower_triangular_cuda(l_matrix: &Tensor, b_matrix: &Tensor) -> Result<Tensor> {
    use crate::{DType, Storage};

    // CUDA kernel only supports F32
    let original_dtype = l_matrix.dtype();
    let needs_conversion = !matches!(original_dtype, DType::F32);

    let l_work = if needs_conversion {
        l_matrix.to_dtype(DType::F32)?
    } else {
        l_matrix.clone()
    };
    let b_work = if needs_conversion {
        b_matrix.to_dtype(DType::F32)?
    } else {
        b_matrix.clone()
    };

    let l_storage = l_work.storage();
    let b_storage = b_work.storage();
    let l_layout = l_work.layout();
    let b_layout = b_work.layout();

    let (l_cuda, b_cuda) = match (&*l_storage, &*b_storage) {
        (Storage::Cuda(l_s), Storage::Cuda(b_s)) => (l_s, b_s),
        _ => crate::bail!("solve_i_minus_lower_triangular_cuda: tensors must be on CUDA"),
    };

    let output_storage = crate::cuda_backend::deltanet::solve_i_minus_lower_triangular(
        l_cuda, l_layout, b_cuda, b_layout,
    )?;

    let output = Tensor::from_storage(Storage::Cuda(output_storage), b_matrix.shape().clone());

    // Convert back to original dtype if needed
    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

#[cfg(feature = "metal")]
fn solve_i_minus_lower_triangular_metal(l_matrix: &Tensor, b_matrix: &Tensor) -> Result<Tensor> {
    use crate::{DType, Storage};

    // Metal kernel only supports F32
    let original_dtype = l_matrix.dtype();
    let needs_conversion = !matches!(original_dtype, DType::F32);

    let l_work = if needs_conversion {
        l_matrix.to_dtype(DType::F32)?
    } else {
        l_matrix.clone()
    };
    let b_work = if needs_conversion {
        b_matrix.to_dtype(DType::F32)?
    } else {
        b_matrix.clone()
    };

    let l_storage = l_work.storage();
    let b_storage = b_work.storage();
    let l_layout = l_work.layout();
    let b_layout = b_work.layout();

    let (l_metal, b_metal) = match (&*l_storage, &*b_storage) {
        (Storage::Metal(l_s), Storage::Metal(b_s)) => (l_s, b_s),
        _ => crate::bail!("solve_i_minus_lower_triangular_metal: tensors must be on Metal"),
    };

    let output_storage = crate::metal_backend::deltanet::solve_i_minus_lower_triangular(
        l_metal, l_layout, b_metal, b_layout,
    )?;

    let output = Tensor::from_storage(Storage::Metal(output_storage), b_matrix.shape().clone());

    // Convert back to original dtype if needed
    if needs_conversion {
        output.to_dtype(original_dtype)
    } else {
        Ok(output)
    }
}

fn solve_i_minus_lower_triangular_cpu(l_matrix: &Tensor, b_matrix: &Tensor) -> Result<Tensor> {
    use crate::DType;

    let l_dims = l_matrix.dims();
    let b_dims = b_matrix.dims();

    let n = l_dims[l_dims.len() - 1];
    let k = b_dims[b_dims.len() - 1];
    let batch: usize = l_dims[..l_dims.len() - 2].iter().product();
    let batch = batch.max(1);

    // Convert to F32 for computation
    let l_f32 = l_matrix.to_dtype(DType::F32)?;
    let b_f32 = b_matrix.to_dtype(DType::F32)?;

    let l_data: Vec<f32> = l_f32.flatten_all()?.to_vec1()?;
    let b_data: Vec<f32> = b_f32.flatten_all()?.to_vec1()?;

    let mut x_data = b_data.clone();

    // Forward substitution: X[i] = B[i] + sum_{j<i} L[i,j] * X[j]
    let l_stride_batch = n * n;
    let x_stride_batch = n * k;

    for b_idx in 0..batch {
        for i in 1..n {
            for j in 0..i {
                let l_ij = l_data[b_idx * l_stride_batch + i * n + j];
                if l_ij.abs() > 1e-10 {
                    for ki in 0..k {
                        let x_j = x_data[b_idx * x_stride_batch + j * k + ki];
                        x_data[b_idx * x_stride_batch + i * k + ki] += l_ij * x_j;
                    }
                }
            }
        }
    }

    let output = Tensor::from_vec(x_data, b_matrix.shape(), &crate::Device::Cpu)?;
    output.to_dtype(l_matrix.dtype())
}

/// Flash attention with Q8_0 quantized K and V.
///
/// Uses optimized CUDA kernel that dequantizes K and V on-the-fly during
/// attention computation, avoiding the memory overhead of dequantizing the
/// entire KV cache before flash attention.
///
/// # Memory Savings
///
/// Current approach (dequantize before flash attention):
///   - Quantized K/V: ~1 GB
///   - Dequantized K/V: ~4 GB
///   - Working memory: ~0.5 GB
///   - Total: ~5.5 GB
///
/// With Q8_0 flash attention:
///   - Quantized K/V: ~1 GB
///   - Output: ~2 GB
///   - Working memory: ~0.5 GB
///   - Total: ~3.5 GB
///
/// # Usage Example
///
/// ```rust,ignore
/// let output = paramecia_core::deltanet_ops::flash_attn_q8(
///     &q_f32,
///     &k_storage,
///     &v_storage,
///     &q_layout,
///     &k_layout,
///     &v_layout,
///     scale,
///     batch, num_heads, num_kv_heads, head_dim,
///     seq_q, seq_k,
///     causal,
/// )?;
/// ```
///
/// # Arguments
///
/// * `q` - Query tensor: [batch, seq_q, num_heads, head_dim] - F32
/// * `k_storage` - Key cache as QStorage (Q8_0 quantized)
/// * `v_storage` - Value cache as QStorage (Q8_0 quantized)
/// * `scale` - Attention scaling factor (1 / sqrt(head_dim))
/// * `b` - Batch size
/// * `h` - Number of query heads
/// * `h_k` - Number of KV heads (for GQA - h_k may be < h)
/// * `d` - Head dimension (must be 64, 128, or 256)
/// * `seq_q` - Query sequence length
/// * `seq_k` - KV cache sequence length
/// * `causal` - Whether to apply causal masking
///
/// # Returns
///
/// Output tensor: [batch, seq_q, num_heads, head_dim] - F32
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_q8(
    q: &Tensor,
    k_storage: &crate::quantized::QStorage,
    v_storage: &crate::quantized::QStorage,
    q_l: &crate::Layout,
    k_l: &crate::Layout,
    v_l: &crate::Layout,
    scale: f32,
    b: usize,
    h: usize,
    h_k: usize,
    d: usize,
    seq_q: usize,
    seq_k: usize,
    q_offset: usize,
    causal: bool,
    prefer_mma: bool,
) -> Result<Tensor> {
    #[cfg(not(feature = "cuda"))]
    let _ = prefer_mma;

    // Validate shapes
    let q_dims = q.dims();
    let k_dims = k_l.shape().dims();
    let v_dims = v_l.shape().dims();

    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        crate::bail!(
            "flash_attn_q8: all tensors must be 4D, got q: {:?}, k: {:?}, v: {:?}",
            q_dims,
            k_dims,
            v_dims
        );
    }

    let (q_b, q_sq, q_h, q_d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (k_b, k_sk, k_hk, k_d) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
    let (v_b, v_sk, v_hk, v_d) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);

    if q_b != b || q_h != h || q_d != d {
        crate::bail!(
            "flash_attn_q8: Q shape mismatch, got [{}, {}, {}, {}] expected [{}, {}, {}, {}]",
            q_b,
            q_sq,
            q_h,
            q_d,
            b,
            seq_q,
            h,
            d
        );
    }

    if k_b != b || k_sk != seq_k || k_hk != h_k || k_d != d {
        crate::bail!(
            "flash_attn_q8: K shape mismatch, got [{}, {}, {}, {}] expected [{}, {}, {}, {}]",
            k_b,
            k_sk,
            k_hk,
            k_d,
            b,
            seq_k,
            h_k,
            d
        );
    }

    if v_b != b || v_sk != seq_k || v_hk != h_k || v_d != d {
        crate::bail!(
            "flash_attn_q8: V shape mismatch, got [{}, {}, {}, {}] expected [{}, {}, {}, {}]",
            v_b,
            v_sk,
            v_hk,
            v_d,
            b,
            seq_k,
            h_k,
            d
        );
    }

    // Validate head dimension
    if d != 64 && d != 128 && d != 256 {
        crate::bail!(
            "flash_attn_q8: unsupported head dimension {}, must be 64, 128, or 256",
            d
        );
    }

    #[cfg(feature = "cuda")]
    {
        if let crate::Device::Cuda(_) = q.device() {
            return crate::cuda_backend::deltanet::flash_attn_q8(
                q, k_storage, v_storage, q_l, k_l, v_l, scale, b, h, h_k, d, seq_q, seq_k,
                q_offset, causal, prefer_mma,
            );
        }
    }

    #[cfg(feature = "metal")]
    {
        if let crate::Device::Metal(_) = q.device() {
            return crate::metal_backend::deltanet::flash_attn_q8_metal(
                q, k_storage, v_storage, q_l, k_l, v_l, scale, b, h, h_k, d, seq_q, seq_k,
                q_offset, causal,
            );
        }
    }

    #[cfg(feature = "vulkan")]
    {
        if let crate::Device::Vulkan(_) = q.device() {
            // Vulkan flash attention shader enabled by default.
            // Disable with PARAMECIA_DISABLE_VULKAN_FLASH_Q8=1 to use CPU fallback.
            if std::env::var("PARAMECIA_DISABLE_VULKAN_FLASH_Q8").is_err() {
                return flash_attn_q8_vulkan(
                    q, k_storage, v_storage, q_l, k_l, v_l, scale, b, h, h_k, d, seq_q, seq_k,
                    q_offset, causal,
                );
            }
        }
    }

    // Fallback: dequantize KV and compute attention with basic ops
    flash_attn_q8_fallback(
        q, k_storage, v_storage, q_l, k_l, v_l, scale, b, h, h_k, d, seq_q, seq_k, q_offset, causal,
    )
}

#[cfg(feature = "vulkan")]
fn flash_attn_q8_vulkan(
    q: &Tensor,
    k_storage: &crate::quantized::QStorage,
    v_storage: &crate::quantized::QStorage,
    q_l: &crate::Layout,
    k_l: &crate::Layout,
    v_l: &crate::Layout,
    scale: f32,
    b: usize,
    h: usize,
    h_k: usize,
    d: usize,
    seq_q: usize,
    seq_k: usize,
    q_offset: usize,
    causal: bool,
) -> Result<Tensor> {
    use std::sync::Arc;

    let device = q.device().as_vulkan_device()?;

    // Get K and V GPU buffers from QStorage — keep Arc alive until batch completes
    let (k_buf, v_buf, k_buf_arc, v_buf_arc) = match (k_storage, v_storage) {
        (QStorage::Vulkan(k_vk), QStorage::Vulkan(v_vk)) => {
            let k_arc = k_vk.gpu_buffer_arc().ok_or_else(|| {
                crate::Error::Msg("flash_attn_q8_vulkan: K has no GPU buffer".into())
            })?;
            let v_arc = v_vk.gpu_buffer_arc().ok_or_else(|| {
                crate::Error::Msg("flash_attn_q8_vulkan: V has no GPU buffer".into())
            })?;
            let kb = k_arc.buffer;
            let vb = v_arc.buffer;
            (kb, vb, k_arc, v_arc)
        }
        _ => {
            return flash_attn_q8_fallback(
                q, k_storage, v_storage, q_l, k_l, v_l, scale, b, h, h_k, d, seq_q, seq_k,
                q_offset, causal,
            );
        }
    };

    // Convert Q to F16 (shader expects F16 input)
    let q_f16 = match q.dtype() {
        DType::F16 => q.clone(),
        _ => q.to_dtype(DType::F16)?,
    }
    .contiguous()?;
    let q_storage_guard = q_f16.storage();
    let q_vk = match &*q_storage_guard {
        Storage::Vulkan(s) => s,
        _ => crate::bail!("flash_attn_q8_vulkan: Q is not Vulkan storage"),
    };
    let q_buf = q_vk.vk_buffer()?;

    // Compute contiguous strides for Q (shape: [b, seq_q, h, d])
    // Q was made .contiguous() above, so q_l.stride() may not match the actual layout.
    let q_stride = &[seq_q * h * d, h * d, d, 1];
    let k_stride = k_l.stride();
    let v_stride = v_l.stride();

    // Output: [b, seq_q, h, d] F16
    let out_shape = Shape::from((b, seq_q, h, d));
    let out_layout = crate::Layout::contiguous(&out_shape);
    let out_stride = out_layout.stride();
    let out_elems = out_shape.elem_count();
    let out_buf = device.allocate_buffer((out_elems * 2) as u64)?;
    let out_storage = VulkanStorage::new(
        Some(Arc::new(out_buf)),
        device.clone(),
        out_elems,
        DType::F16,
    );
    let use_flash_idp = device.has_integer_dot_product()
        && std::env::var("PARAMECIA_DISABLE_FLASH_Q8_IDP").is_err();

    let split_k = choose_flash_attn_q8_split_k(b, h, d, seq_q, seq_k);
    if split_k > 1 {
        let split_kv = ((seq_k as u32) + split_k - 1) / split_k;
        let temp_bytes =
            flash_attn_q8_split_temp_bytes(b, h, d, seq_q, split_k).ok_or_else(|| {
                crate::Error::Msg("flash_attn_q8_vulkan split-k temporary size overflow".into())
            })?;

        let rows = b
            .checked_mul(seq_q)
            .and_then(|x| x.checked_mul(h))
            .ok_or_else(|| crate::Error::Msg("flash_attn_q8_vulkan rows overflow".into()))?;
        let partial_bytes = (split_k as usize)
            .checked_mul(rows)
            .and_then(|x| x.checked_mul(d))
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| {
                crate::Error::Msg("flash_attn_q8_vulkan partial bytes overflow".into())
            })?;
        let stats_bytes = temp_bytes.checked_sub(partial_bytes).ok_or_else(|| {
            crate::Error::Msg("flash_attn_q8_vulkan stats bytes underflow".into())
        })?;

        let partial_buf = device.acquire_scratch_buffer(partial_bytes as u64)?;
        let stats_buf = device.acquire_scratch_buffer(stats_bytes as u64)?;

        let params: [u32; 27] = [
            scale.to_bits(),                  // [0] scale as u32 bits
            b as u32,                         // [1] batch
            h as u32,                         // [2] num_heads
            h_k as u32,                       // [3] num_kv_heads
            d as u32,                         // [4] head_dim
            seq_q as u32,                     // [5] seq_q
            seq_k as u32,                     // [6] seq_k
            q_offset as u32,                  // [7] q_offset
            if causal { 1u32 } else { 0u32 }, // [8] causal
            q_stride[0] as u32,               // [9] q_stride_b
            q_stride[1] as u32,               // [10] q_stride_seq
            q_stride[2] as u32,               // [11] q_stride_h
            q_stride[3] as u32,               // [12] q_stride_d
            k_stride[0] as u32,               // [13] k_stride_b
            k_stride[1] as u32,               // [14] k_stride_seq
            k_stride[2] as u32,               // [15] k_stride_h
            k_stride[3] as u32,               // [16] k_stride_d
            v_stride[0] as u32,               // [17] v_stride_b
            v_stride[1] as u32,               // [18] v_stride_seq
            v_stride[2] as u32,               // [19] v_stride_h
            v_stride[3] as u32,               // [20] v_stride_d
            out_stride[0] as u32,             // [21] o_stride_b (unused in split pass)
            out_stride[1] as u32,             // [22] o_stride_seq (unused in split pass)
            out_stride[2] as u32,             // [23] o_stride_h (unused in split pass)
            out_stride[3] as u32,             // [24] o_stride_d (unused in split pass)
            split_k,                          // [25] split_k
            split_kv,                         // [26] split_kv
        ];
        let split_shader_name = if use_flash_idp {
            format!("flash_attn_q8_split_idp_d{}", d)
        } else {
            format!("flash_attn_q8_split_d{}", d)
        };
        let split_pipeline = device
            .kernels()
            .load_pipeline(
                device.device(),
                &split_shader_name,
                None,
                (params.len() * std::mem::size_of::<u32>()) as u32,
                5,
                None,
                false,
            )
            .or_else(|e| {
                if use_flash_idp {
                    let fallback = format!("flash_attn_q8_split_d{}", d);
                    device
                        .kernels()
                        .load_pipeline(
                            device.device(),
                            &fallback,
                            None,
                            (params.len() * std::mem::size_of::<u32>()) as u32,
                            5,
                            None,
                            false,
                        )
                        .map_err(|e2| {
                            crate::Error::Msg(format!(
                                "Failed to load flash_attn_q8 split pipeline '{}' (err: {}) and fallback '{}' (err: {})",
                                split_shader_name, e, fallback, e2
                            ))
                        })
                } else {
                    Err(crate::Error::Msg(format!(
                        "Failed to load flash_attn_q8 split pipeline {}: {}",
                        split_shader_name, e
                    )))
                }
            })?;

        let split_dispatch = [
            (b as u32).checked_mul(split_k).ok_or_else(|| {
                crate::Error::Msg("flash_attn_q8_vulkan split dispatch overflow".into())
            })?,
            h as u32,
            seq_q as u32,
        ];

        // Bindings: [0]=partials, [1]=stats, [2]=Q, [3]=K, [4]=V
        device.record_compute_with_write_mask(
            &split_pipeline,
            &[partial_buf.buffer, stats_buf.buffer, q_buf, k_buf, v_buf],
            Some((1u64 << 0) | (1u64 << 1)),
            Some(bytemuck::cast_slice(&params)),
            split_dispatch,
        )?;

        let reduce_pipeline = device
            .kernels()
            .load_pipeline(
                device.device(),
                "flash_attn_q8_split_k_reduce",
                None,
                (5 * std::mem::size_of::<u32>()) as u32,
                3,
                None,
                false,
            )
            .map_err(|e| {
                crate::Error::Msg(format!(
                    "Failed to load flash_attn_q8 split reduce pipeline: {}",
                    e
                ))
            })?;

        let reduce_pc: [u32; 5] = [b as u32, h as u32, seq_q as u32, d as u32, split_k];
        let reduce_dispatch = [((d as u32) / 2 + 63) / 64, h as u32, (b * seq_q) as u32];
        device.record_compute_with_write_mask(
            &reduce_pipeline,
            &[
                partial_buf.buffer,
                stats_buf.buffer,
                out_storage.vk_buffer()?,
            ],
            Some(1u64 << 2),
            Some(bytemuck::cast_slice(&reduce_pc)),
            reduce_dispatch,
        )?;

        device.keep_buffer_alive(k_buf_arc)?;
        device.keep_buffer_alive(v_buf_arc)?;
        device.release_scratch_buffer(partial_buf);
        device.release_scratch_buffer(stats_buf);
    } else {
        // Push constants: 25 x u32
        let params: [u32; 25] = [
            scale.to_bits(),                  // [0] scale as u32 bits
            b as u32,                         // [1] batch
            h as u32,                         // [2] num_heads
            h_k as u32,                       // [3] num_kv_heads
            d as u32,                         // [4] head_dim
            seq_q as u32,                     // [5] seq_q
            seq_k as u32,                     // [6] seq_k
            q_offset as u32,                  // [7] q_offset
            if causal { 1u32 } else { 0u32 }, // [8] causal
            // Q strides (f16 element indices)
            q_stride[0] as u32, // [9] q_stride_b
            q_stride[1] as u32, // [10] q_stride_seq
            q_stride[2] as u32, // [11] q_stride_h
            q_stride[3] as u32, // [12] q_stride_d
            // K strides (bytes - Q8_0 blocks)
            k_stride[0] as u32, // [13] k_stride_b
            k_stride[1] as u32, // [14] k_stride_seq
            k_stride[2] as u32, // [15] k_stride_h
            k_stride[3] as u32, // [16] k_stride_d
            // V strides (bytes - Q8_0 blocks)
            v_stride[0] as u32, // [17] v_stride_b
            v_stride[1] as u32, // [18] v_stride_seq
            v_stride[2] as u32, // [19] v_stride_h
            v_stride[3] as u32, // [20] v_stride_d
            // Output strides (f16 element indices)
            out_stride[0] as u32, // [21] o_stride_b
            out_stride[1] as u32, // [22] o_stride_seq
            out_stride[2] as u32, // [23] o_stride_h
            out_stride[3] as u32, // [24] o_stride_d
        ];
        let shader_name = if use_flash_idp {
            format!("flash_attn_q8_idp_d{}", d)
        } else {
            format!("flash_attn_q8_d{}", d)
        };
        let pipeline = device
            .kernels()
            .load_pipeline(
                device.device(),
                &shader_name,
                None,
                (params.len() * std::mem::size_of::<u32>()) as u32,
                4,
                None,
                false,
            )
            .or_else(|e| {
                if use_flash_idp {
                    let fallback = format!("flash_attn_q8_d{}", d);
                    device
                        .kernels()
                        .load_pipeline(
                            device.device(),
                            &fallback,
                            None,
                            (params.len() * std::mem::size_of::<u32>()) as u32,
                            4,
                            None,
                            false,
                        )
                        .map_err(|e2| {
                            crate::Error::Msg(format!(
                                "Failed to load flash_attn_q8 pipeline '{}' (err: {}) and fallback '{}' (err: {})",
                                shader_name, e, fallback, e2
                            ))
                        })
                } else {
                    Err(crate::Error::Msg(format!(
                        "Failed to load flash_attn_q8 pipeline: {}",
                        e
                    )))
                }
            })?;

        let dispatch = [b as u32, h as u32, seq_q as u32];
        device.record_compute_with_write_mask(
            &pipeline,
            &[out_storage.vk_buffer()?, q_buf, k_buf, v_buf],
            Some(1u64 << 0),
            Some(bytemuck::cast_slice(&params)),
            dispatch,
        )?;

        device.keep_buffer_alive(k_buf_arc)?;
        device.keep_buffer_alive(v_buf_arc)?;
    }

    // Create output tensor [b, seq_q, h, d] in F16, cast back to input dtype
    let out_tensor = Tensor::from_storage(Storage::Vulkan(out_storage), out_shape);

    if q.dtype() != DType::F16 {
        out_tensor.to_dtype(q.dtype())
    } else {
        Ok(out_tensor)
    }
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_q8_fallback(
    q: &Tensor,
    k_storage: &crate::quantized::QStorage,
    v_storage: &crate::quantized::QStorage,
    _q_l: &crate::Layout,
    k_l: &crate::Layout,
    v_l: &crate::Layout,
    scale: f32,
    b: usize,
    h: usize,
    h_k: usize,
    d: usize,
    seq_q: usize,
    seq_k: usize,
    _q_offset: usize,
    causal: bool,
) -> Result<Tensor> {
    // Dequantize K and V to CPU, run SDPA on CPU, then move result back.
    // This avoids potential issues with Vulkan strided tensor operations.
    let orig_device = q.device().clone();
    let orig_dtype = q.dtype();

    let k_elem_count = k_l.shape().elem_count();
    let v_elem_count = v_l.shape().elem_count();

    let k_deq_storage = k_storage.dequantize(k_elem_count)?;
    let v_deq_storage = v_storage.dequantize(v_elem_count)?;

    let k_shape = crate::Shape::from_dims(&[b, seq_k, h_k, d]);
    let v_shape = crate::Shape::from_dims(&[b, seq_k, h_k, d]);

    let k = crate::tensor::from_storage(k_deq_storage, k_shape);
    let v = crate::tensor::from_storage(v_deq_storage, v_shape);

    // Move everything to CPU for computation
    let cpu = crate::Device::Cpu;
    let k = k.to_device(&cpu)?.to_dtype(crate::DType::F32)?;
    let v = v.to_device(&cpu)?.to_dtype(crate::DType::F32)?;
    let q_cpu = q.to_device(&cpu)?.to_dtype(crate::DType::F32)?;

    // q: [b, seq_q, h, d] -> [b, h, seq_q, d]
    let q_t = q_cpu.transpose(1, 2)?.contiguous()?;
    // k: [b, seq_k, h_k, d] -> [b, h_k, seq_k, d]
    let k = k.transpose(1, 2)?.contiguous()?;
    // v: [b, seq_k, h_k, d] -> [b, h_k, seq_k, d]
    let v = v.transpose(1, 2)?.contiguous()?;

    // GQA: repeat KV heads
    let repeats = h / h_k;
    let k = if repeats > 1 {
        k.unsqueeze(2)?
            .expand(&[b, h_k, repeats, seq_k, d])?
            .reshape(&[b, h, seq_k, d])?
            .contiguous()?
    } else {
        k
    };
    let v = if repeats > 1 {
        v.unsqueeze(2)?
            .expand(&[b, h_k, repeats, seq_k, d])?
            .reshape(&[b, h, seq_k, d])?
            .contiguous()?
    } else {
        v
    };

    // Attention: q @ k^T * scale
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let attn = (q_t.matmul(&k_t)? * (scale as f64))?;

    // Causal mask: additive approach to avoid NaN from -inf * 0
    let attn = if causal && seq_q > 1 {
        let mask = Tensor::tril2(seq_k, crate::DType::F32, &cpu)?;
        let mask = if seq_q < seq_k {
            mask.narrow(0, seq_k - seq_q, seq_q)?
        } else {
            mask
        };
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq_q, seq_k]
                                                     // Create additive causal mask: 0 for attended, -1e9 for masked
        let causal_mask = ((mask - 1.0)? * 1e9)?;
        attn.broadcast_add(&causal_mask)?
    } else {
        attn
    };

    // Softmax along last dim
    let max_vals = attn.max_keepdim(crate::D::Minus1)?;
    let attn = attn.broadcast_sub(&max_vals)?;
    let attn = attn.exp()?;
    let sum_vals = attn.sum_keepdim(crate::D::Minus1)?;
    let attn = attn.broadcast_div(&sum_vals)?;

    // attn @ v -> [b, h, seq_q, d]
    let output = attn.matmul(&v.contiguous()?)?;

    // [b, h, seq_q, d] -> [b, seq_q, h, d]
    let output = output.transpose(1, 2)?.contiguous()?;

    // Move result back to original device and dtype
    output.to_dtype(orig_dtype)?.to_device(&orig_device)
}
