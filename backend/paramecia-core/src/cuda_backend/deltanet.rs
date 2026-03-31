//! CUDA kernels for Delta Net / Mamba-style operations
//!
//! These kernels are optimized for linear attention models like Qwen3-Next.

use super::{CudaStorage, CudaStorageSlice, WrapErr};
use crate::quantized::QStorage;
use crate::{builder_arg as barg, DType, Device, Error, Layout, Result, Shape, Storage, Tensor};
use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
use half::f16;
use paramecia_cuda as kernels;

/// Depthwise 1D convolution for SSM/Mamba-style models
///
/// Input: [batch, channels, input_len] (with pre-padded input_len)
/// Weight: [channels, kernel_size]
/// Output: [batch, channels, output_len]
pub fn depthwise_conv1d(
    input: &CudaStorage,
    input_l: &Layout,
    weight: &CudaStorage,
    weight_l: &Layout,
    output_len: usize,
) -> Result<CudaStorage> {
    let dev = &input.device;
    let (batch, channels, input_len) = input_l.shape().dims3()?;
    let (weight_channels, kernel_size) = weight_l.shape().dims2()?;

    if channels != weight_channels {
        return Err(Error::Msg(format!(
            "depthwise_conv1d: channel mismatch {} vs {}",
            channels, weight_channels
        )));
    }

    let out_elem_count = batch * channels * output_len;
    let cfg = LaunchConfig::for_num_elems(out_elem_count as u32);

    match (&input.slice, &weight.slice) {
        (CudaStorageSlice::F32(input_slice), CudaStorageSlice::F32(weight_slice)) => {
            let func = dev.get_or_load_func("depthwise_conv1d_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(out_elem_count)? };
            let input_slice = input_slice.slice(input_l.start_offset()..);
            let weight_slice = weight_slice.slice(weight_l.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&input_slice);
            builder.arg(&weight_slice);
            builder.arg(&out);
            barg!(builder, batch);
            barg!(builder, channels);
            barg!(builder, input_len);
            barg!(builder, output_len);
            barg!(builder, kernel_size);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            })
        }
        _ => Err(Error::Msg(
            "depthwise_conv1d: only f32 supported".to_string(),
        )),
    }
}

/// Cumulative sum along the last dimension (optimized with warp-level prefix sum)
///
/// Input: [..., len]
/// Output: [..., len] where output[..., i] = sum(input[..., 0:i+1])
pub fn cumsum(input: &CudaStorage, input_l: &Layout) -> Result<CudaStorage> {
    let dev = &input.device;
    let shape = input_l.shape();
    let dims = shape.dims();

    if dims.is_empty() {
        return Err(Error::Msg("cumsum: empty shape".to_string()));
    }

    let len = dims[dims.len() - 1];
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let batch = batch.max(1);

    // Use 256 threads for the optimized kernel
    let block_size = 256.min(len);
    let num_warps = (block_size + 31) / 32;
    let shared_mem = (num_warps + 1) * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    match &input.slice {
        CudaStorageSlice::F32(input_slice) => {
            let func = dev.get_or_load_func("cumsum_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(shape.elem_count())? };
            let input_slice = input_slice.slice(input_l.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&input_slice);
            builder.arg(&out);
            barg!(builder, batch);
            barg!(builder, len);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            })
        }
        _ => Err(Error::Msg("cumsum: only f32 supported".to_string())),
    }
}

/// L2 normalize and scale in one fused operation
///
/// Input: [..., dim]
/// Output: [..., dim] where output = input / ||input||_2 * scale
pub fn l2_normalize_scale(
    input: &CudaStorage,
    input_l: &Layout,
    scale: f32,
    eps: f32,
) -> Result<CudaStorage> {
    let dev = &input.device;
    let shape = input_l.shape();
    let dims = shape.dims();

    if dims.is_empty() {
        return Err(Error::Msg("l2_normalize_scale: empty shape".to_string()));
    }

    let dim = dims[dims.len() - 1];
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let batch = batch.max(1);

    let threads = dim.min(256);
    let num_warps = (threads + 31) / 32;
    let shared_mem = num_warps * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    match &input.slice {
        CudaStorageSlice::F32(input_slice) => {
            let func = dev.get_or_load_func("l2_normalize_scale_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(shape.elem_count())? };
            let input_slice = input_slice.slice(input_l.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&input_slice);
            builder.arg(&out);
            barg!(builder, scale);
            barg!(builder, eps);
            barg!(builder, batch);
            barg!(builder, dim);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            })
        }
        _ => Err(Error::Msg(
            "l2_normalize_scale: only f32 supported".to_string(),
        )),
    }
}

/// Gated Linear Attention single step (autoregressive mode)
///
/// q, k, v: [batch * heads, d]
/// gate, beta: [batch * heads]
/// state: [batch * heads, d, d] (in/out)
/// output: [batch * heads, d]
pub fn gla_step(
    q: &CudaStorage,
    k: &CudaStorage,
    v: &CudaStorage,
    gate: &CudaStorage,
    beta: &CudaStorage,
    state: &mut CudaStorage,
    scale: f32,
    batch_heads: usize,
    d: usize,
) -> Result<CudaStorage> {
    let dev = &q.device;

    let shared_mem = 2 * d * std::mem::size_of::<f32>();

    let cfg = LaunchConfig {
        grid_dim: (batch_heads as u32, 1, 1),
        block_dim: (d as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    match (
        &q.slice,
        &k.slice,
        &v.slice,
        &gate.slice,
        &beta.slice,
        &mut state.slice,
    ) {
        (
            CudaStorageSlice::F32(q_slice),
            CudaStorageSlice::F32(k_slice),
            CudaStorageSlice::F32(v_slice),
            CudaStorageSlice::F32(gate_slice),
            CudaStorageSlice::F32(beta_slice),
            CudaStorageSlice::F32(state_slice),
        ) => {
            let func = dev.get_or_load_func("gla_step_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(batch_heads * d)? };
            let mut builder = func.builder();
            builder.arg(q_slice);
            builder.arg(k_slice);
            builder.arg(v_slice);
            builder.arg(gate_slice);
            builder.arg(beta_slice);
            builder.arg(state_slice);
            builder.arg(&out);
            barg!(builder, scale);
            barg!(builder, batch_heads);
            barg!(builder, d);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            })
        }
        _ => Err(Error::Msg("gla_step: only f32 supported".to_string())),
    }
}

/// Top-K MoE routing with softmax on GPU
///
/// Logits: [n_tokens, n_experts]
/// Returns: (weights: [n_tokens, topk], indices: [n_tokens, topk])
pub fn topk_softmax(
    logits: &CudaStorage,
    logits_l: &Layout,
    topk: usize,
) -> Result<(CudaStorage, CudaStorage, Shape, Shape)> {
    let dev = &logits.device;
    let (n_tokens, n_experts) = logits_l.shape().dims2()?;

    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    match &logits.slice {
        CudaStorageSlice::F32(logits_slice) => {
            let func = dev.get_or_load_func("topk_softmax_f32", &kernels::DELTANET)?;
            let weights_out = unsafe { dev.alloc::<f32>(n_tokens * topk)? };
            let indices_out = unsafe { dev.alloc::<i32>(n_tokens * topk)? };
            let logits_slice = logits_slice.slice(logits_l.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&logits_slice);
            builder.arg(&weights_out);
            builder.arg(&indices_out);
            barg!(builder, n_tokens);
            barg!(builder, n_experts);
            barg!(builder, topk);
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::F32(weights_out),
                    device: dev.clone(),
                },
                CudaStorage {
                    slice: CudaStorageSlice::I32(indices_out),
                    device: dev.clone(),
                },
                Shape::from((n_tokens, topk)),
                Shape::from((n_tokens, topk)),
            ))
        }
        _ => Err(Error::Msg("topk_softmax: only f32 supported".to_string())),
    }
}

/// Lower triangular solve (forward substitution)
///
/// Solves L @ X = B for X, where L is lower triangular
/// L: [batch, n, n]
/// B: [batch, n, k]
/// X: [batch, n, k]
pub fn solve_lower_triangular(
    l_matrix: &CudaStorage,
    l_layout: &Layout,
    b_matrix: &CudaStorage,
    b_layout: &Layout,
) -> Result<CudaStorage> {
    let dev = &l_matrix.device;
    let l_dims = l_layout.shape().dims();
    let b_dims = b_layout.shape().dims();

    if l_dims.len() < 2 || b_dims.len() < 2 {
        return Err(Error::Msg(
            "solve_lower_triangular: need at least 2D".to_string(),
        ));
    }

    let n = l_dims[l_dims.len() - 1];
    let k = b_dims[b_dims.len() - 1];
    let batch: usize = l_dims[..l_dims.len() - 2].iter().product();
    let batch = batch.max(1);

    let shared_mem = n * n * std::mem::size_of::<f32>();
    let k_threads = k.min(32);

    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (32, k_threads as u32, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    match (&l_matrix.slice, &b_matrix.slice) {
        (CudaStorageSlice::F32(l_slice), CudaStorageSlice::F32(b_slice)) => {
            let func = dev.get_or_load_func("solve_lower_tri_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(batch * n * k)? };
            let l_slice = l_slice.slice(l_layout.start_offset()..);
            let b_slice = b_slice.slice(b_layout.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&l_slice);
            builder.arg(&b_slice);
            builder.arg(&out);
            barg!(builder, batch);
            barg!(builder, n);
            barg!(builder, k);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            })
        }
        _ => Err(Error::Msg(
            "solve_lower_triangular: only f32 supported".to_string(),
        )),
    }
}

/// Lower triangular solve for (I - L) @ X = B (DeltaNet chunked algorithm)
///
/// Solves (I - L) @ X = B where L is strictly lower triangular.
/// Since (I - L) has 1s on the diagonal, no division is needed.
/// Formula: X[i] = B[i] + sum_{j<i} L[i,j] * X[j]
///
/// This matches the precision characteristics of the autoregressive kernel
/// by running entirely on GPU with the same FMA behavior.
///
/// L: [batch, n, n] - strictly lower triangular
/// B: [batch, n, k] - right-hand side matrix
/// Returns: X [batch, n, k] - solution
pub fn solve_i_minus_lower_triangular(
    l_matrix: &CudaStorage,
    l_layout: &Layout,
    b_matrix: &CudaStorage,
    b_layout: &Layout,
) -> Result<CudaStorage> {
    let dev = &l_matrix.device;
    let l_dims = l_layout.shape().dims();
    let b_dims = b_layout.shape().dims();

    if l_dims.len() < 2 || b_dims.len() < 2 {
        return Err(Error::Msg(
            "solve_i_minus_lower_triangular: need at least 2D".to_string(),
        ));
    }

    let n = l_dims[l_dims.len() - 1];
    let k = b_dims[b_dims.len() - 1];
    let batch: usize = l_dims[..l_dims.len() - 2].iter().product();
    let batch = batch.max(1);

    // Shared memory for L matrix
    let shared_mem = n * n * std::mem::size_of::<f32>();
    let k_threads = k.min(32);

    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (32, k_threads as u32, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    match (&l_matrix.slice, &b_matrix.slice) {
        (CudaStorageSlice::F32(l_slice), CudaStorageSlice::F32(b_slice)) => {
            let func = dev.get_or_load_func("solve_i_minus_lower_tri_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(batch * n * k)? };
            let l_slice = l_slice.slice(l_layout.start_offset()..);
            let b_slice = b_slice.slice(b_layout.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&l_slice);
            builder.arg(&b_slice);
            builder.arg(&out);
            barg!(builder, batch);
            barg!(builder, n);
            barg!(builder, k);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            })
        }
        _ => Err(Error::Msg(
            "solve_i_minus_lower_triangular: only f32 supported".to_string(),
        )),
    }
}

/// Fused SwiGLU activation: silu(gate) * up
///
/// gate: [...] - gate projection output
/// up: [...] - up projection output (same shape as gate)
/// Returns: silu(gate) * up
pub fn fused_swiglu(
    gate: &CudaStorage,
    gate_l: &Layout,
    up: &CudaStorage,
    up_l: &Layout,
) -> Result<CudaStorage> {
    let dev = &gate.device;
    let total = gate_l.shape().elem_count();
    let cfg = LaunchConfig::for_num_elems(total as u32);

    match (&gate.slice, &up.slice) {
        (CudaStorageSlice::F32(gate_slice), CudaStorageSlice::F32(up_slice)) => {
            let func = dev.get_or_load_func("fused_swiglu_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(total)? };
            let gate_slice = gate_slice.slice(gate_l.start_offset()..);
            let up_slice = up_slice.slice(up_l.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&gate_slice);
            builder.arg(&up_slice);
            builder.arg(&out);
            barg!(builder, total);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            })
        }
        _ => Err(Error::Msg("fused_swiglu: only f32 supported".to_string())),
    }
}

/// Expert-parallel matmul for MoE
///
/// input: [n_tokens, k] - input hidden states
/// weights: [n_active_experts, n, k] - gathered weights for active experts
/// expert_map: [n_tokens, top_k] - maps (token, k) to expert index in weights (i32)
/// Returns: [n_tokens, top_k, n]
pub fn expert_parallel_matmul(
    input: &CudaStorage,
    input_l: &Layout,
    weights: &CudaStorage,
    weights_l: &Layout,
    expert_map: &CudaStorage,
    expert_map_l: &Layout,
    top_k: usize,
) -> Result<(CudaStorage, Shape)> {
    let dev = &input.device;

    let input_dims = input_l.shape().dims();
    let weights_dims = weights_l.shape().dims();

    if input_dims.len() != 2 || weights_dims.len() != 3 {
        return Err(Error::Msg(
            "expert_parallel_matmul: invalid dimensions".to_string(),
        ));
    }

    let n_tokens = input_dims[0];
    let k = input_dims[1];
    let n = weights_dims[1];

    let out_shape = Shape::from((n_tokens, top_k, n));
    let out_elem_count = n_tokens * top_k * n;

    match (&input.slice, &weights.slice, &expert_map.slice) {
        (
            CudaStorageSlice::F32(input_slice),
            CudaStorageSlice::F32(weights_slice),
            CudaStorageSlice::I32(expert_map_slice),
        ) => {
            // Use shared memory version for better performance
            let func =
                dev.get_or_load_func("expert_parallel_matmul_shared_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(out_elem_count)? };

            let input_slice = input_slice.slice(input_l.start_offset()..);
            let weights_slice = weights_slice.slice(weights_l.start_offset()..);
            let expert_map_slice = expert_map_slice.slice(expert_map_l.start_offset()..);

            // Launch config: one block per token, threads handle (expert_k, out_idx) pairs
            let threads_per_block = 256u32;
            let shared_mem = (k * std::mem::size_of::<f32>()) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n_tokens as u32, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: shared_mem,
            };

            let mut builder = func.builder();
            builder.arg(&input_slice);
            builder.arg(&weights_slice);
            builder.arg(&expert_map_slice);
            builder.arg(&out);
            barg!(builder, n_tokens);
            barg!(builder, top_k);
            barg!(builder, n);
            barg!(builder, k);
            unsafe { builder.launch(cfg) }.w()?;

            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::F32(out),
                    device: dev.clone(),
                },
                out_shape,
            ))
        }
        _ => Err(Error::Msg(
            "expert_parallel_matmul: unsupported dtype".to_string(),
        )),
    }
}

/// Scatter-add expert outputs with weights
///
/// expert_outputs: [n_tokens, top_k, hidden_dim]
/// expert_weights: [n_tokens, top_k]
/// Returns: [n_tokens, hidden_dim]
pub fn scatter_add_experts(
    expert_outputs: &CudaStorage,
    outputs_l: &Layout,
    expert_weights: &CudaStorage,
    weights_l: &Layout,
) -> Result<(CudaStorage, Shape)> {
    let dev = &expert_outputs.device;
    let outputs_dims = outputs_l.shape().dims();

    if outputs_dims.len() != 3 {
        return Err(Error::Msg(
            "scatter_add_experts: outputs must be 3D".to_string(),
        ));
    }

    let n_tokens = outputs_dims[0];
    let top_k = outputs_dims[1];
    let hidden_dim = outputs_dims[2];

    let out_shape = Shape::from((n_tokens, hidden_dim));
    let out_elem_count = n_tokens * hidden_dim;
    let cfg = LaunchConfig::for_num_elems(out_elem_count as u32);

    match (&expert_outputs.slice, &expert_weights.slice) {
        (CudaStorageSlice::F32(outputs_slice), CudaStorageSlice::F32(weights_slice)) => {
            let func = dev.get_or_load_func("scatter_add_experts_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(out_elem_count)? };
            let outputs_slice = outputs_slice.slice(outputs_l.start_offset()..);
            let weights_slice = weights_slice.slice(weights_l.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&outputs_slice);
            builder.arg(&weights_slice);
            builder.arg(&out);
            barg!(builder, n_tokens);
            barg!(builder, top_k);
            barg!(builder, hidden_dim);
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::F32(out),
                    device: dev.clone(),
                },
                out_shape,
            ))
        }
        _ => Err(Error::Msg(
            "scatter_add_experts: only f32 supported".to_string(),
        )),
    }
}

/// Fused Delta Net autoregressive step
///
/// Combines L2 normalization, state decay, kv_mem, delta, state update,
/// and output computation into a single kernel.
///
/// q, k, v: [batch, num_heads, head_dim] - input vectors (squeezed)
/// gate: [batch, num_heads] - log decay values
/// beta: [batch, num_heads] - gate values (pre-sigmoid)
/// state: [batch, num_heads, head_dim, head_dim] - recurrent state
///
/// Returns: (output, new_state) where:
///   output: [batch, num_heads, head_dim]
///   new_state: [batch, num_heads, head_dim, head_dim]
pub fn delta_net_autoregressive_step(
    q: &CudaStorage,
    q_l: &Layout,
    k: &CudaStorage,
    k_l: &Layout,
    v: &CudaStorage,
    v_l: &Layout,
    gate: &CudaStorage,
    gate_l: &Layout,
    beta: &CudaStorage,
    beta_l: &Layout,
    state: &CudaStorage,
    state_l: &Layout,
    scale: f32,
    eps: f32,
) -> Result<(CudaStorage, CudaStorage)> {
    let dev = &q.device;
    let q_dims = q_l.shape().dims();

    if q_dims.len() != 3 {
        return Err(Error::Msg(
            "delta_net_autoregressive_step: q must be 3D [batch, heads, dim]".to_string(),
        ));
    }

    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let head_dim = q_dims[2];

    let output_elems = batch * num_heads * head_dim;
    let state_elems = batch * num_heads * head_dim * head_dim;

    // Shared memory: 6 * head_dim floats (q, k, v, kv_mem, delta, out)
    let shared_mem = (6 * head_dim * std::mem::size_of::<f32>()) as u32;
    let threads = 256.min(head_dim) as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (batch as u32, num_heads as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    match (
        &q.slice,
        &k.slice,
        &v.slice,
        &gate.slice,
        &beta.slice,
        &state.slice,
    ) {
        (
            CudaStorageSlice::F32(q_slice),
            CudaStorageSlice::F32(k_slice),
            CudaStorageSlice::F32(v_slice),
            CudaStorageSlice::F32(gate_slice),
            CudaStorageSlice::F32(beta_slice),
            CudaStorageSlice::F32(state_slice),
        ) => {
            let func =
                dev.get_or_load_func("delta_net_autoregressive_step_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(output_elems)? };
            let new_state = unsafe { dev.alloc::<f32>(state_elems)? };

            let q_slice = q_slice.slice(q_l.start_offset()..);
            let k_slice = k_slice.slice(k_l.start_offset()..);
            let v_slice = v_slice.slice(v_l.start_offset()..);
            let gate_slice = gate_slice.slice(gate_l.start_offset()..);
            let beta_slice = beta_slice.slice(beta_l.start_offset()..);
            let state_slice = state_slice.slice(state_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&q_slice);
            builder.arg(&k_slice);
            builder.arg(&v_slice);
            builder.arg(&gate_slice);
            builder.arg(&beta_slice);
            builder.arg(&state_slice);
            builder.arg(&out);
            builder.arg(&new_state);
            builder.arg(&scale);
            builder.arg(&eps);
            barg!(builder, batch);
            barg!(builder, num_heads);
            barg!(builder, head_dim);
            unsafe { builder.launch(cfg) }.w()?;

            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::F32(out),
                    device: dev.clone(),
                },
                CudaStorage {
                    slice: CudaStorageSlice::F32(new_state),
                    device: dev.clone(),
                },
            ))
        }
        _ => Err(Error::Msg(
            "delta_net_autoregressive_step: only f32 supported".to_string(),
        )),
    }
}

/// Multi-token Delta Net state update for MTP/speculative decoding
///
/// Processes K tokens in parallel, mathematically equivalent to K sequential
/// autoregressive steps but more efficient.
///
/// q, k, v: [batch, num_heads, num_tokens, head_dim]
/// gate: [batch, num_heads, num_tokens] - log decay values
/// beta: [batch, num_heads, num_tokens] - already sigmoided
/// state: [batch, num_heads, head_dim, head_dim]
///
/// Returns: (output, new_state) where:
///   output: [batch, num_heads, num_tokens, head_dim]
///   new_state: [batch, num_heads, head_dim, head_dim]
pub fn delta_net_multi_token_update(
    q: &CudaStorage,
    q_l: &Layout,
    k: &CudaStorage,
    k_l: &Layout,
    v: &CudaStorage,
    v_l: &Layout,
    gate: &CudaStorage,
    gate_l: &Layout,
    beta: &CudaStorage,
    beta_l: &Layout,
    state: &CudaStorage,
    state_l: &Layout,
) -> Result<(CudaStorage, CudaStorage)> {
    let dev = &q.device;
    let q_dims = q_l.shape().dims();

    if q_dims.len() != 4 {
        return Err(Error::Msg(
            "delta_net_multi_token_update: q must be 4D [batch, heads, tokens, dim]".to_string(),
        ));
    }

    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let num_tokens = q_dims[2];
    let head_dim = q_dims[3];

    let output_elems = batch * num_heads * num_tokens * head_dim;
    let state_elems = batch * num_heads * head_dim * head_dim;

    // Shared memory: 4 temp vectors only (k, v, delta, kv_mem)
    // State matrix stays in global memory to avoid exceeding shared mem limits
    let shared_mem = (4 * head_dim) * std::mem::size_of::<f32>();
    let threads = 256.min(head_dim) as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (batch as u32, num_heads as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    match (
        &q.slice,
        &k.slice,
        &v.slice,
        &gate.slice,
        &beta.slice,
        &state.slice,
    ) {
        (
            CudaStorageSlice::F32(q_slice),
            CudaStorageSlice::F32(k_slice),
            CudaStorageSlice::F32(v_slice),
            CudaStorageSlice::F32(gate_slice),
            CudaStorageSlice::F32(beta_slice),
            CudaStorageSlice::F32(state_slice),
        ) => {
            // Always use the general multi-token kernel (handles any num_tokens)
            let func =
                dev.get_or_load_func("delta_net_multi_token_update_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(output_elems)? };
            let new_state = unsafe { dev.alloc::<f32>(state_elems)? };

            let q_slice = q_slice.slice(q_l.start_offset()..);
            let k_slice = k_slice.slice(k_l.start_offset()..);
            let v_slice = v_slice.slice(v_l.start_offset()..);
            let gate_slice = gate_slice.slice(gate_l.start_offset()..);
            let beta_slice = beta_slice.slice(beta_l.start_offset()..);
            let state_slice = state_slice.slice(state_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&q_slice);
            builder.arg(&k_slice);
            builder.arg(&v_slice);
            builder.arg(&gate_slice);
            builder.arg(&beta_slice);
            builder.arg(&state_slice);
            builder.arg(&out);
            builder.arg(&new_state);
            // Note: scale not passed - q is already L2-normalized with scale applied
            barg!(builder, batch);
            barg!(builder, num_heads);
            barg!(builder, num_tokens);
            barg!(builder, head_dim);
            unsafe { builder.launch(cfg) }.w()?;

            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::F32(out),
                    device: dev.clone(),
                },
                CudaStorage {
                    slice: CudaStorageSlice::F32(new_state),
                    device: dev.clone(),
                },
            ))
        }
        _ => Err(Error::Msg(
            "delta_net_multi_token_update: only f32 supported".to_string(),
        )),
    }
}

/// Multi-token Delta Net with PARALLEL intermediate state materialization
///
/// Same as delta_net_multi_token_update but also materializes and returns
/// the state after each position, enabling O(1) state slicing for speculative
/// decoding verification.
///
/// q, k, v: [batch, num_heads, num_tokens, head_dim]
/// gate: [batch, num_heads, num_tokens] - log decay values
/// beta: [batch, num_heads, num_tokens] - already sigmoided
/// state: [batch, num_heads, head_dim, head_dim]
///
/// Returns: (output, new_state, intermediate_states) where:
///   output: [batch, num_heads, num_tokens, head_dim]
///   new_state: [batch, num_heads, head_dim, head_dim]
///   intermediate_states: [batch, num_heads, num_tokens, head_dim, head_dim]
pub fn delta_net_parallel_with_states(
    q: &CudaStorage,
    q_l: &Layout,
    k: &CudaStorage,
    k_l: &Layout,
    v: &CudaStorage,
    v_l: &Layout,
    gate: &CudaStorage,
    gate_l: &Layout,
    beta: &CudaStorage,
    beta_l: &Layout,
    state: &CudaStorage,
    state_l: &Layout,
) -> Result<(CudaStorage, CudaStorage, CudaStorage)> {
    let dev = &q.device;
    let q_dims = q_l.shape().dims();

    if q_dims.len() != 4 {
        return Err(Error::Msg(
            "delta_net_parallel_with_states: q must be 4D [batch, heads, tokens, dim]".to_string(),
        ));
    }

    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let num_tokens = q_dims[2];
    let head_dim = q_dims[3];

    let output_elems = batch * num_heads * num_tokens * head_dim;
    let state_elems = batch * num_heads * head_dim * head_dim;
    let inter_state_elems = batch * num_heads * num_tokens * head_dim * head_dim;

    // Shared memory: 4 temp vectors (k, v, delta, kv_mem)
    let shared_mem = (4 * head_dim) * std::mem::size_of::<f32>();
    let threads = 256.min(head_dim) as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (batch as u32, num_heads as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    match (
        &q.slice,
        &k.slice,
        &v.slice,
        &gate.slice,
        &beta.slice,
        &state.slice,
    ) {
        (
            CudaStorageSlice::F32(q_slice),
            CudaStorageSlice::F32(k_slice),
            CudaStorageSlice::F32(v_slice),
            CudaStorageSlice::F32(gate_slice),
            CudaStorageSlice::F32(beta_slice),
            CudaStorageSlice::F32(state_slice),
        ) => {
            let func =
                dev.get_or_load_func("delta_net_parallel_with_states_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(output_elems)? };
            let new_state = unsafe { dev.alloc::<f32>(state_elems)? };
            let intermediate_states = unsafe { dev.alloc::<f32>(inter_state_elems)? };

            let q_slice = q_slice.slice(q_l.start_offset()..);
            let k_slice = k_slice.slice(k_l.start_offset()..);
            let v_slice = v_slice.slice(v_l.start_offset()..);
            let gate_slice = gate_slice.slice(gate_l.start_offset()..);
            let beta_slice = beta_slice.slice(beta_l.start_offset()..);
            let state_slice = state_slice.slice(state_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&q_slice);
            builder.arg(&k_slice);
            builder.arg(&v_slice);
            builder.arg(&gate_slice);
            builder.arg(&beta_slice);
            builder.arg(&state_slice);
            builder.arg(&out);
            builder.arg(&new_state);
            builder.arg(&intermediate_states);
            barg!(builder, batch);
            barg!(builder, num_heads);
            barg!(builder, num_tokens);
            barg!(builder, head_dim);
            unsafe { builder.launch(cfg) }.w()?;

            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::F32(out),
                    device: dev.clone(),
                },
                CudaStorage {
                    slice: CudaStorageSlice::F32(new_state),
                    device: dev.clone(),
                },
                CudaStorage {
                    slice: CudaStorageSlice::F32(intermediate_states),
                    device: dev.clone(),
                },
            ))
        }
        _ => Err(Error::Msg(
            "delta_net_parallel_with_states: only f32 supported".to_string(),
        )),
    }
}

/// GPU-side computation of P(x) and Q(x) for draft tokens
///
/// Computes the probability of each draft token under both target and draft
/// distributions. Returns vectors of P(x) and Q(x) values.
pub fn compute_draft_probs(
    target_logits: &CudaStorage,
    target_l: &Layout,
    draft_logits: &CudaStorage,
    draft_l: &Layout,
    draft_tokens: &CudaStorage,
    draft_tokens_l: &Layout,
    temperature: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let dev = &target_logits.device;
    let target_dims = target_l.shape().dims();

    if target_dims.len() != 2 {
        return Err(Error::Msg(
            "compute_draft_probs: logits must be 2D".to_string(),
        ));
    }

    let num_drafts = target_dims[0];
    let vocab_size = target_dims[1];

    // Shared memory for warp reductions (3 arrays of 32 floats)
    let shared_mem = (32 * 3 * std::mem::size_of::<f32>()) as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_drafts as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    match (
        &target_logits.slice,
        &draft_logits.slice,
        &draft_tokens.slice,
    ) {
        (
            CudaStorageSlice::F32(target_slice),
            CudaStorageSlice::F32(draft_slice),
            CudaStorageSlice::I32(tokens_slice),
        ) => {
            let func = dev.get_or_load_func("compute_draft_probs_f32", &kernels::DELTANET)?;
            let p_values = unsafe { dev.alloc::<f32>(num_drafts)? };
            let q_values = unsafe { dev.alloc::<f32>(num_drafts)? };

            let target_slice = target_slice.slice(target_l.start_offset()..);
            let draft_slice = draft_slice.slice(draft_l.start_offset()..);
            let tokens_slice = tokens_slice.slice(draft_tokens_l.start_offset()..);

            let mut builder = func.builder();
            builder.arg(&target_slice);
            builder.arg(&draft_slice);
            builder.arg(&tokens_slice);
            builder.arg(&p_values);
            builder.arg(&q_values);
            builder.arg(&temperature);
            barg!(builder, num_drafts);
            barg!(builder, vocab_size);
            unsafe { builder.launch(cfg) }.w()?;

            // Read results back (just num_drafts floats, not full vocab)
            let p_vec: Vec<f32> = dev.clone_dtoh(&p_values)?;
            let q_vec: Vec<f32> = dev.clone_dtoh(&q_values)?;

            Ok((p_vec, q_vec))
        }
        _ => Err(Error::Msg(
            "compute_draft_probs: unsupported dtypes".to_string(),
        )),
    }
}

/// Fused gated RMS norm
///
/// x: [batch, dim] - input to normalize
/// z: [batch, dim] - gate input
/// weight: [dim] - RMS norm weights
/// Returns: silu(z) * rms_norm(x, weight)
pub fn gated_rms_norm(
    x: &CudaStorage,
    x_l: &Layout,
    z: &CudaStorage,
    z_l: &Layout,
    weight: &CudaStorage,
    weight_l: &Layout,
    eps: f32,
) -> Result<CudaStorage> {
    let dev = &x.device;
    let x_dims = x_l.shape().dims();

    if x_dims.len() != 2 {
        return Err(Error::Msg(
            "gated_rms_norm: x must be 2D [batch, dim]".to_string(),
        ));
    }

    let batch = x_dims[0];
    let dim = x_dims[1];
    let total_elems = batch * dim;

    let shared_mem =
        (2 * dim * std::mem::size_of::<f32>() + 32 * std::mem::size_of::<f32>()) as u32;
    let threads = 256.min(dim) as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: shared_mem,
    };

    match (&x.slice, &z.slice, &weight.slice) {
        (
            CudaStorageSlice::F32(x_slice),
            CudaStorageSlice::F32(z_slice),
            CudaStorageSlice::F32(weight_slice),
        ) => {
            let func = dev.get_or_load_func("gated_rms_norm_f32", &kernels::DELTANET)?;
            let out = unsafe { dev.alloc::<f32>(total_elems)? };
            let x_slice = x_slice.slice(x_l.start_offset()..);
            let z_slice = z_slice.slice(z_l.start_offset()..);
            let weight_slice = weight_slice.slice(weight_l.start_offset()..);
            let mut builder = func.builder();
            builder.arg(&x_slice);
            builder.arg(&z_slice);
            builder.arg(&weight_slice);
            builder.arg(&out);
            builder.arg(&eps);
            barg!(builder, batch);
            barg!(builder, dim);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage {
                slice: CudaStorageSlice::F32(out),
                device: dev.clone(),
            })
        }
        _ => Err(Error::Msg("gated_rms_norm: only f32 supported".to_string())),
    }
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
pub fn flash_attn_q8(
    q: &Tensor,
    k_storage: &crate::quantized::QStorage,
    v_storage: &crate::quantized::QStorage,
    q_l: &Layout,
    k_l: &Layout,
    v_l: &Layout,
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
    // Validate shapes
    let q_dims = q.dims();
    let k_dims = k_l.shape().dims();
    let v_dims = v_l.shape().dims();

    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::Msg(format!(
            "flash_attn_q8: all tensors must be 4D, got q: {:?}, k: {:?}, v: {:?}",
            q_dims, k_dims, v_dims
        )));
    }

    let (q_b, q_sq, q_h, q_d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (k_b, k_sk, k_hk, k_d) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
    let (v_b, v_sk, v_hk, v_d) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);

    if q_b != b || q_h != h || q_d != d {
        return Err(Error::Msg(format!(
            "flash_attn_q8: Q shape mismatch, got [{}, {}, {}, {}] expected [{}, {}, {}, {}]",
            q_b, q_sq, q_h, q_d, b, seq_q, h, d
        )));
    }

    if k_b != b || k_sk != seq_k || k_hk != h_k || k_d != d {
        return Err(Error::Msg(format!(
            "flash_attn_q8: K shape mismatch, got [{}, {}, {}, {}] expected [{}, {}, {}, {}]",
            k_b, k_sk, k_hk, k_d, b, seq_k, h_k, d
        )));
    }

    if v_b != b || v_sk != seq_k || v_hk != h_k || v_d != d {
        return Err(Error::Msg(format!(
            "flash_attn_q8: V shape mismatch, got [{}, {}, {}, {}] expected [{}, {}, {}, {}]",
            v_b, v_sk, v_hk, v_d, b, seq_k, h_k, d
        )));
    }

    // Validate head dimension
    if d != 64 && d != 128 && d != 256 {
        return Err(Error::Msg(format!(
            "flash_attn_q8: unsupported head dimension {}, must be 64, 128, or 256",
            d
        )));
    }

    // Ensure all tensors are contiguous
    let q = q.contiguous()?;

    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = q.device() {
            return flash_attn_q8_cuda(
                &q, k_storage, v_storage, q_l, k_l, v_l, scale, b, h, h_k, d, seq_q, seq_k,
                q_offset, causal, prefer_mma,
            );
        }
    }

    // CPU fallback (would require dequantization)
    Err(Error::Msg(
        "flash_attn_q8: CUDA backend required for Q8_0 quantized flash attention".to_string(),
    ))
}

#[cfg(feature = "cuda")]
fn flash_attn_q8_cuda(
    q: &Tensor,
    k_storage: &crate::quantized::QStorage,
    v_storage: &crate::quantized::QStorage,
    q_l: &Layout,
    k_l: &Layout,
    v_l: &Layout,
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
    // Extract storage and device
    let dev = match &*q.storage() {
        Storage::Cuda(storage) => storage.device.clone(),
        _ => {
            return Err(Error::Msg(
                "flash_attn_q8 requires CUDA storage".to_string(),
            ))
        }
    };

    // Get QStorage CUDA pointers
    let (k_ptr, v_ptr) = match (k_storage, v_storage) {
        (QStorage::Cuda(k_storage), QStorage::Cuda(v_storage)) => {
            let kp = k_storage
                .device_ptr()
                .map_err(|e| Error::Msg(format!("Failed to get K pointer: {}", e)))?;
            let vp = v_storage
                .device_ptr()
                .map_err(|e| Error::Msg(format!("Failed to get V pointer: {}", e)))?;
            (kp, vp)
        }
        _ => {
            return Err(Error::Msg(
                "flash_attn_q8 requires CUDA quantized storage".to_string(),
            ));
        }
    };

    // Keep Q in F16 for speed (Q is typically F16/BF16).
    let q_f16 = match q.dtype() {
        DType::F16 => q.clone(),
        DType::BF16 | DType::F32 => q.to_dtype(DType::F16)?,
        dt => {
            return Err(Error::Msg(format!(
                "flash_attn_q8: unsupported Q dtype {dt:?}, expected f16/bf16/f32"
            )));
        }
    }
    .contiguous()?;

    let q_storage_guard = q_f16.storage();
    let q_storage = match &*q_storage_guard {
        Storage::Cuda(s) => s,
        _ => {
            return Err(Error::Msg(
                "flash_attn_q8 requires CUDA storage".to_string(),
            ))
        }
    };

    let q_slice = q_storage.as_cuda_slice::<f16>()?;
    let stream = dev.cuda_stream();
    let (q_ptr, _q_sync) = q_slice.device_ptr(stream.as_ref());

    // Allocate output buffer
    let out_shape = Shape::from((b, seq_q, h, d));
    let out_layout = Layout::contiguous(&out_shape);
    let out_size = out_shape.elem_count();
    let out = unsafe { dev.alloc::<f16>(out_size)? };

    // Compute strides
    let q_stride = q_l.stride();
    let k_stride = k_l.stride();
    let v_stride = v_l.stride();
    let out_stride = out_layout.stride();

    // Use MMA tensor core kernel for prefill (seq_q > 1) on Ampere+ GPUs
    // PARAMECIA_NO_MMA_FA=1 disables for benchmarking
    let mma_fa = prefer_mma && std::env::var("PARAMECIA_NO_MMA_FA").as_deref() != Ok("1");
    if seq_q > 1 && mma_fa && crate::quantized::cuda::has_mma_support() {
        use core::ffi::c_void;
        let stream_i64 = stream.cu_stream() as i64;
        let (out_ptr, out_sync) = out.device_ptr(stream.as_ref());

        unsafe {
            kernels::ffi::flash_attn_q8_mma_launch(
                q_ptr as *const c_void,
                k_ptr as *const c_void,
                v_ptr as *const c_void,
                out_ptr as *mut c_void,
                scale,
                b as i32,
                h as i32,
                h_k as i32,
                d as i32,
                seq_q as i32,
                seq_k as i32,
                q_offset as i32,
                i32::from(causal),
                q_stride[0] as u64,
                q_stride[1] as u64,
                q_stride[2] as u64,
                q_stride[3] as u64,
                k_stride[0] as u64,
                k_stride[1] as u64,
                k_stride[2] as u64,
                k_stride[3] as u64,
                v_stride[0] as u64,
                v_stride[1] as u64,
                v_stride[2] as u64,
                v_stride[3] as u64,
                out_stride[0] as u64,
                out_stride[1] as u64,
                out_stride[2] as u64,
                out_stride[3] as u64,
                stream_i64,
            );
        }
        drop(out_sync);

        let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
        return Ok(Tensor::from_storage(Storage::Cuda(out_storage), out_shape));
    }

    // Fallback: existing DP4A kernel (decode path or pre-Turing GPUs)
    let nwarps = if seq_q <= 2 {
        if seq_k >= 4096 {
            8u32
        } else if seq_k <= 512 {
            2u32
        } else {
            4u32
        }
    } else {
        4u32
    };
    let kernel_name = match (d, nwarps) {
        (64, 2) => "flash_attn_q8_kernel_64_w2",
        (64, 4) => "flash_attn_q8_kernel_64_w4",
        (64, 8) => "flash_attn_q8_kernel_64_w8",
        (128, 2) => "flash_attn_q8_kernel_128_w2",
        (128, 4) => "flash_attn_q8_kernel_128_w4",
        (128, 8) => "flash_attn_q8_kernel_128_w8",
        (256, 2) => "flash_attn_q8_kernel_256_w2",
        (256, 4) => "flash_attn_q8_kernel_256_w4",
        (256, 8) => "flash_attn_q8_kernel_256_w8",
        _ => {
            return Err(Error::Msg(format!(
                "flash_attn_q8: unsupported head dimension {}",
                d
            )))
        }
    };
    let func = dev.get_or_load_func(kernel_name, &kernels::FLASH_ATTN_Q8)?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (b as u32, h as u32, seq_q as u32),
        block_dim: (32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();

    // Q pointer (F32)
    builder.arg(&q_ptr);

    // K and V pointers (Q8_0 quantized)
    let k_ptr = k_ptr as u64;
    let v_ptr = v_ptr as u64;
    builder.arg(&k_ptr);
    builder.arg(&v_ptr);

    // Output pointer
    let (out_ptr, out_sync) = out.device_ptr(stream.as_ref());
    builder.arg(&out_ptr);

    // Attention parameters
    let b_u32 = b as u32;
    let h_u32 = h as u32;
    let h_k_u32 = h_k as u32;
    let d_u32 = d as u32;
    let seq_q_u32 = seq_q as u32;
    let seq_k_u32 = seq_k as u32;
    let q_offset_u32 = q_offset as u32;
    let causal_i32 = i32::from(causal);
    builder.arg(&scale);
    builder.arg(&b_u32);
    builder.arg(&h_u32);
    builder.arg(&h_k_u32);
    builder.arg(&d_u32);
    builder.arg(&seq_q_u32);
    builder.arg(&seq_k_u32);
    builder.arg(&q_offset_u32);
    builder.arg(&causal_i32);

    // Strides for Q
    let q_stride0 = q_stride[0] as u64;
    let q_stride1 = q_stride[1] as u64;
    let q_stride2 = q_stride[2] as u64;
    let q_stride3 = q_stride[3] as u64;
    builder.arg(&q_stride0);
    builder.arg(&q_stride1);
    builder.arg(&q_stride2);
    builder.arg(&q_stride3);

    // Strides for K
    let k_stride0 = k_stride[0] as u64;
    let k_stride1 = k_stride[1] as u64;
    let k_stride2 = k_stride[2] as u64;
    let k_stride3 = k_stride[3] as u64;
    builder.arg(&k_stride0);
    builder.arg(&k_stride1);
    builder.arg(&k_stride2);
    builder.arg(&k_stride3);

    // Strides for V
    let v_stride0 = v_stride[0] as u64;
    let v_stride1 = v_stride[1] as u64;
    let v_stride2 = v_stride[2] as u64;
    let v_stride3 = v_stride[3] as u64;
    builder.arg(&v_stride0);
    builder.arg(&v_stride1);
    builder.arg(&v_stride2);
    builder.arg(&v_stride3);

    // Strides for output
    let out_stride0 = out_stride[0] as u64;
    let out_stride1 = out_stride[1] as u64;
    let out_stride2 = out_stride[2] as u64;
    let out_stride3 = out_stride[3] as u64;
    builder.arg(&out_stride0);
    builder.arg(&out_stride1);
    builder.arg(&out_stride2);
    builder.arg(&out_stride3);

    unsafe { builder.launch(cfg) }.w()?;
    drop(out_sync);

    let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from_storage(Storage::Cuda(out_storage), out_shape))
}
