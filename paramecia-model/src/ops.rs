//! Accelerated operations for Qwen3-Next
//!
//! This module provides optimized tensor operations that use hardware-specific kernels
//! when available (CUDA on NVIDIA, Metal on Apple Silicon), with CPU fallback.
//!
//! All operations delegate to candle-core's deltanet_ops module which handles
//! automatic device dispatch to the appropriate backend.

use candle::{DType, Result, Tensor, D};

/// Depthwise 1D convolution for SSM/linear attention layers.
///
/// Uses optimized CUDA kernel when available.
///
/// Input: [batch, channels, input_len] (with pre-padded input_len)
/// Weight: [channels, kernel_size]
/// Output: [batch, channels, output_len]
pub fn depthwise_conv1d(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    candle::deltanet_ops::depthwise_conv1d(input, weight)
}

/// GPU-accelerated Top-K selection with softmax for MoE routing
///
/// Input: routing_weights [batch, seq_len, n_experts] (after softmax)
/// Returns: (top_weights, top_indices) each of shape [batch, seq_len, topk]
///
/// This avoids the CPU-GPU synchronization that occurs when using to_vec1() per token.
pub fn topk_moe_routing(routing_weights: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    // The CUDA kernel for topk_softmax expects pre-softmaxed logits, but we receive
    // post-softmax weights. For now, use the optimized CPU path which uses GPU sort.
    //
    // TODO: When CUDA feature is enabled, we could use the deltanet::topk_softmax
    // kernel directly on pre-softmax logits (would require API change in MoeBlock)
    topk_moe_routing_optimized(routing_weights, topk)
}

/// Optimized top-k MoE routing using GPU-based sort
fn topk_moe_routing_optimized(routing_weights: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let (batch_size, seq_len, n_experts) = routing_weights.dims3()?;

    // Use sort_last_dim to get sorted values and indices on GPU if available
    // This is much faster than pulling each token to CPU
    let weights_flat = routing_weights.reshape((batch_size * seq_len, n_experts))?;
    let (sorted_vals, sorted_idx) = weights_flat.sort_last_dim(false)?; // descending

    // Take top-k
    let top_weights = sorted_vals.narrow(1, 0, topk)?;
    let top_indices = sorted_idx.narrow(1, 0, topk)?;

    // Normalize top-k weights
    let weight_sum = top_weights.sum_keepdim(D::Minus1)?;
    let normalized_weights = top_weights.broadcast_div(&weight_sum)?;

    // Reshape
    let normalized_weights = normalized_weights.reshape((batch_size, seq_len, topk))?;
    let top_indices = top_indices.reshape((batch_size, seq_len, topk))?;

    Ok((normalized_weights, top_indices))
}

/// GPU-side expert grouping for efficient CPU-offloaded MoE processing
///
/// This function sorts (token, expert) pairs by expert ID on GPU, so when
/// transferred to CPU, tokens for each expert are contiguous.
///
/// Input:
///   - expert_indices: [n_tokens, top_k] expert assignments
///
/// Returns:
///   - sorted_token_ids: [n_tokens * top_k] original token indices in sorted order
///   - sorted_k_ids: [n_tokens * top_k] original k indices in sorted order  
///   - sorted_expert_ids: [n_tokens * top_k] expert IDs in sorted order
///   - segment_offsets: [n_experts + 1] start/end indices for each expert's segment
pub fn group_tokens_by_expert(
    expert_indices: &Tensor,
    num_experts: usize,
) -> Result<(Tensor, Tensor, Tensor, Vec<usize>)> {
    let (n_tokens, top_k) = expert_indices.dims2()?;
    let total_pairs = n_tokens * top_k;
    let device = expert_indices.device();

    // Create flat expert_id tensor for sorting
    let expert_flat = expert_indices.flatten_all()?.to_dtype(DType::U32)?;

    // Create token_id and k_id tensors
    // token_ids: [0, 0, ..., 0, 1, 1, ..., 1, ...] (each token repeated top_k times)
    // k_ids: [0, 1, 2, ..., top_k-1, 0, 1, 2, ..., top_k-1, ...]
    let token_ids: Vec<u32> = (0..n_tokens as u32)
        .flat_map(|t| std::iter::repeat_n(t, top_k))
        .collect();
    let k_ids: Vec<u32> = (0..n_tokens).flat_map(|_| 0..top_k as u32).collect();

    let token_id_tensor = Tensor::from_vec(token_ids.clone(), total_pairs, device)?;
    let k_id_tensor = Tensor::from_vec(k_ids.clone(), total_pairs, device)?;

    // Sort by expert_id
    // CUDA arg_sort uses bitonic sort with ncols_pad threads per block.
    // For sizes > 1024, ncols_pad exceeds CUDA's 1024 threads/block limit.
    // Fall back to CPU sorting for large inputs.
    let (sorted_token_ids, sorted_k_ids, sorted_expert_ids) = if total_pairs > 1024 {
        // CPU fallback for large inputs
        let expert_cpu: Vec<u32> = expert_flat.to_device(&candle::Device::Cpu)?.to_vec1()?;

        // Create indices and sort
        let mut indices: Vec<usize> = (0..total_pairs).collect();
        indices.sort_by_key(|&i| expert_cpu[i]);

        // Apply sort permutation
        let sorted_token_ids: Vec<u32> = indices.iter().map(|&i| token_ids[i]).collect();
        let sorted_k_ids: Vec<u32> = indices.iter().map(|&i| k_ids[i]).collect();
        let sorted_expert_ids: Vec<u32> = indices.iter().map(|&i| expert_cpu[i]).collect();

        (
            Tensor::from_vec(sorted_token_ids, total_pairs, device)?,
            Tensor::from_vec(sorted_k_ids, total_pairs, device)?,
            Tensor::from_vec(sorted_expert_ids, total_pairs, device)?,
        )
    } else {
        // GPU sort for small inputs (fast path)
        let sort_indices = expert_flat.arg_sort_last_dim(true)?; // ascending

        // Gather in sorted order
        let sorted_expert_ids = expert_flat.index_select(&sort_indices, 0)?;
        let sorted_token_ids = token_id_tensor.index_select(&sort_indices, 0)?;
        let sorted_k_ids = k_id_tensor.index_select(&sort_indices, 0)?;

        (sorted_token_ids, sorted_k_ids, sorted_expert_ids)
    };

    // Find segment boundaries (where expert ID changes)
    // Transfer sorted expert IDs to CPU for boundary detection
    let sorted_experts_cpu: Vec<u32> = sorted_expert_ids.to_vec1()?;

    // Build segment offsets: segment_offsets[e] = start index for expert e
    let mut segment_offsets = vec![0usize; num_experts + 1];
    let mut current_expert = 0usize;

    for (i, &expert_id) in sorted_experts_cpu.iter().enumerate() {
        let expert = expert_id as usize;
        while current_expert <= expert && current_expert < num_experts {
            segment_offsets[current_expert] = i;
            current_expert += 1;
        }
    }
    // Fill remaining experts that have no tokens
    while current_expert <= num_experts {
        segment_offsets[current_expert] = total_pairs;
        current_expert += 1;
    }

    Ok((
        sorted_token_ids,
        sorted_k_ids,
        sorted_expert_ids,
        segment_offsets,
    ))
}

/// Process MoE with GPU-side grouping for CPU-offloaded experts
///
/// This is the optimized path that:
/// 1. Groups tokens by expert on GPU (no per-expert sync)
/// 2. Transfers grouped data to CPU in one operation
/// 3. Processes each expert's batch using direct quantized matmul (no dequantization overhead)
/// 4. Parallelizes expert processing with Rayon
pub fn process_moe_gpu_grouped(
    hidden_states: &Tensor,
    expert_indices: &Tensor,
    expert_weights: &Tensor,
    gate_exps: &std::sync::Arc<candle::quantized::QTensor>,
    up_exps: &std::sync::Arc<candle::quantized::QTensor>,
    down_exps: &std::sync::Arc<candle::quantized::QTensor>,
    num_experts: usize,
) -> Result<Tensor> {
    use candle::Device;
    use rayon::prelude::*;

    let (n_tokens, hidden_dim) = hidden_states.dims2()?;
    let (_, top_k) = expert_indices.dims2()?;
    let original_device = hidden_states.device().clone();
    let dtype = hidden_states.dtype();

    // Step 1: Group tokens by expert on GPU
    let (sorted_token_ids, sorted_k_ids, _sorted_expert_ids, segment_offsets) =
        group_tokens_by_expert(expert_indices, num_experts)?;

    // Step 2: Transfer everything to CPU in one batch
    let hidden_cpu = hidden_states
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?;
    let weights_cpu = expert_weights
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?;
    let sorted_tokens_cpu: Vec<u32> = sorted_token_ids.to_vec1()?;
    let sorted_k_cpu: Vec<u32> = sorted_k_ids.to_vec1()?;

    let hidden_data: Vec<f32> = hidden_cpu.flatten_all()?.to_vec1()?;
    let weights_flat: Vec<f32> = weights_cpu.flatten_all()?.to_vec1()?;

    // Step 3: Build expert assignments (which tokens go to which expert)
    type ExpertAssignment = Vec<(usize, usize, usize)>;
    let expert_assignments: Vec<(usize, ExpertAssignment)> = (0..num_experts)
        .filter_map(|expert_id| {
            let start = segment_offsets[expert_id];
            let end = segment_offsets[expert_id + 1];
            if start >= end {
                return None;
            }
            // Collect (position_in_segment, token_id, k_id) for this expert
            let assignments: Vec<(usize, usize, usize)> = (start..end)
                .enumerate()
                .map(|(i, idx)| {
                    let token_id = sorted_tokens_cpu[idx] as usize;
                    let k_id = sorted_k_cpu[idx] as usize;
                    (i, token_id, k_id)
                })
                .collect();
            Some((expert_id, assignments))
        })
        .collect();

    // Step 4: Process each expert in parallel using Rayon
    // Uses apply_op1_no_bwd which leverages optimized quantized matmul internally
    type ExpertResult = Vec<(usize, f32, Vec<f32>)>;
    let expert_results: Vec<(usize, ExpertResult)> = expert_assignments
        .into_par_iter()
        .filter_map(|(expert_id, assignments)| {
            let segment_size = assignments.len();

            // Get expert weights (quantized)
            let gate = gate_exps.slice_first_dim(expert_id).ok()?;
            let up = up_exps.slice_first_dim(expert_id).ok()?;
            let down = down_exps.slice_first_dim(expert_id).ok()?;

            // Gather input for this expert's tokens
            let mut expert_input = vec![0.0f32; segment_size * hidden_dim];
            let mut token_info: Vec<(usize, f32)> = Vec::with_capacity(segment_size);

            for (i, token_id, k_id) in &assignments {
                expert_input[*i * hidden_dim..(*i + 1) * hidden_dim].copy_from_slice(
                    &hidden_data[*token_id * hidden_dim..(*token_id + 1) * hidden_dim],
                );
                let routing_weight = weights_flat[*token_id * top_k + *k_id];
                token_info.push((*token_id, routing_weight));
            }

            // Create input tensor - batched for this expert
            let input =
                Tensor::from_vec(expert_input, (segment_size, hidden_dim), &Device::Cpu).ok()?;

            // SwiGLU forward using quantized matmul via apply_op1_no_bwd
            // gate_out = input @ gate.T, up_out = input @ up.T
            let gate_out = input.apply_op1_no_bwd(&gate).ok()?;
            let up_out = input.apply_op1_no_bwd(&up).ok()?;

            // silu(gate) * up
            let activated = candle_nn::ops::silu(&gate_out).ok()?.mul(&up_out).ok()?;

            // down projection
            let output = activated.apply_op1_no_bwd(&down).ok()?;

            // Extract results
            let output_vec: Vec<f32> = output.flatten_all().ok()?.to_vec1().ok()?;

            // Prepare results: (token_id, weight, output_row)
            let results: Vec<(usize, f32, Vec<f32>)> = token_info
                .into_iter()
                .enumerate()
                .map(|(i, (token_id, weight))| {
                    let output_row = output_vec[i * hidden_dim..(i + 1) * hidden_dim].to_vec();
                    (token_id, weight, output_row)
                })
                .collect();

            Some((expert_id, results))
        })
        .collect();

    // Step 5: Scatter results back to output (sequential to avoid race conditions)
    let mut output_data = vec![0.0f32; n_tokens * hidden_dim];

    for (_expert_id, results) in expert_results {
        for (token_id, weight, output_row) in results {
            for j in 0..hidden_dim {
                output_data[token_id * hidden_dim + j] += output_row[j] * weight;
            }
        }
    }

    // Step 6: Transfer back to GPU
    let output = Tensor::from_vec(output_data, (n_tokens, hidden_dim), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(&original_device)?;

    Ok(output)
}

/// GPU-accelerated lower triangular solve
///
/// Solves (I - L) @ X = B for X where L is strictly lower triangular.
/// Uses GPU forward substitution kernel for precision matching with autoregressive path.
/// Returns X masked to lower triangular with identity added.
///
/// L: [batch, heads, n, n] (strictly lower triangular coefficients)
/// B: [batch, heads, n, k] (the matrix to solve for)
/// Returns: [batch, heads, n, k]
pub fn solve_lower_triangular_batched(
    l_matrix: &Tensor,
    b_matrix: &Tensor,
    causal_mask: &Tensor,
) -> Result<Tensor> {
    let l_dims = l_matrix.dims();
    let b_dims = b_matrix.dims();

    if l_dims.len() != 4 || b_dims.len() != 4 {
        candle::bail!("solve_lower_triangular_batched expects 4D tensors");
    }

    let (batch, heads, n, n2) = (l_dims[0], l_dims[1], l_dims[2], l_dims[3]);
    let k = b_dims[3];

    if n != n2 {
        candle::bail!("L matrix must be square in last two dimensions");
    }

    let original_device = l_matrix.device().clone();
    let dtype = l_matrix.dtype();

    // Handle trivial cases
    if n <= 1 {
        let identity = Tensor::eye(n, dtype, &original_device)?;
        let identity = identity.unsqueeze(0)?.unsqueeze(0)?;
        return b_matrix
            .broadcast_mul(causal_mask)?
            .broadcast_add(&identity);
    }

    // Extract strictly lower triangular part
    let l = l_matrix.broadcast_mul(causal_mask)?;

    // Flatten batch and heads dimensions for the solve kernel
    // [batch, heads, n, n] -> [batch*heads, n, n]
    let l_flat = l.reshape((batch * heads, n, n))?.contiguous()?;
    let b_flat = b_matrix.reshape((batch * heads, n, k))?.contiguous()?;

    // Use GPU kernel for forward substitution (matches autoregressive precision)
    let x_flat = candle::deltanet_ops::solve_i_minus_lower_triangular(&l_flat, &b_flat)?;

    // Reshape back to [batch, heads, n, k]
    let x = x_flat.reshape((batch, heads, n, k))?;

    // Create identity for output shape [batch, heads, n, n]
    let output_identity = Tensor::eye(n, dtype, &original_device)?
        .unsqueeze(0)?
        .unsqueeze(0)?;

    // Apply causal mask and add identity (matching llama.cpp behavior)
    x.broadcast_mul(causal_mask)?
        .broadcast_add(&output_identity)
}

/// Fused L2 normalize and scale
///
/// Computes: x / ||x||_2 * scale
/// This fuses the normalization and scaling into a single operation.
pub fn l2_normalize_scale(x: &Tensor, scale: f32, eps: f64) -> Result<Tensor> {
    // Delegate to candle-core's fused CUDA kernel implementation
    candle::deltanet_ops::l2_normalize_scale(x, scale, eps)
}

/// GPU-accelerated cumulative sum
pub fn cumsum_last_dim(x: &Tensor) -> Result<Tensor> {
    // Use candle's built-in cumsum which handles GPU/CPU automatically
    x.cumsum(x.rank() - 1)
}

/// Fused Delta Net autoregressive step
///
/// Combines L2 normalization, state decay, kv_mem, delta, state update,
/// and output computation. Uses optimized CUDA kernel when available.
///
/// q, k, v: [batch, num_heads, head_dim] - input vectors (squeezed from [b, 1, ...])
/// gate: [batch, num_heads] - log decay values  
/// beta: [batch, num_heads] - gate values (pre-sigmoid)
/// state: [batch, num_heads, head_dim, head_dim] - recurrent state
///
/// Returns: (output, new_state) where:
///   output: [batch, num_heads, head_dim]
///   new_state: [batch, num_heads, head_dim, head_dim]
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
    // Delegate to candle-core's fused CUDA kernel implementation
    candle::deltanet_ops::delta_net_autoregressive_step(q, k, v, gate, beta, state, scale, eps)
}

/// Multi-token Delta Net state update for MTP/speculative decoding
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
    // Delegate to candle-core's fused CUDA kernel implementation
    candle::deltanet_ops::delta_net_multi_token_update(q, k, v, gate, beta, state)
}

/// Multi-token Delta Net with PARALLEL intermediate state materialization
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
pub fn delta_net_parallel_with_states(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Delegate to candle-core's fused CUDA kernel implementation
    candle::deltanet_ops::delta_net_parallel_with_states(q, k, v, gate, beta, state)
}

/// GPU-side computation of P(x) and Q(x) for draft tokens
///
/// Computes probabilities without moving full vocabulary to CPU.
/// Returns (p_values, q_values) for each draft token.
pub fn compute_draft_probs(
    target_logits: &Tensor,
    draft_logits: &Tensor,
    draft_tokens: &Tensor,
    temperature: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    candle::deltanet_ops::compute_draft_probs(
        target_logits,
        draft_logits,
        draft_tokens,
        temperature,
    )
}

/// Fused SwiGLU activation: silu(gate) * up
///
/// This fuses the SiLU activation and multiply into a single kernel,
/// reducing memory traffic by avoiding intermediate tensor writes.
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    // Delegate to candle-core's fused CUDA kernel implementation
    candle::deltanet_ops::fused_swiglu(gate, up)
}

/// Expert-parallel forward pass for MoE
///
/// This processes all (token, expert) pairs in parallel, avoiding the
/// sequential loop over experts.
///
/// hidden_states: [n_tokens, hidden_dim]
/// gate_weights: [num_experts, n_ff, hidden_dim] - all expert weights
/// up_weights: [num_experts, n_ff, hidden_dim]
/// down_weights: [num_experts, hidden_dim, n_ff]
/// expert_indices: [n_tokens, top_k] - which experts each token uses (u32)
/// expert_weights: [n_tokens, top_k] - routing weights
///
/// Returns: [n_tokens, hidden_dim]
pub fn expert_parallel_forward(
    hidden_states: &Tensor,
    gate_weights: &Tensor,
    up_weights: &Tensor,
    down_weights: &Tensor,
    expert_indices: &Tensor,
    expert_weights: &Tensor,
) -> Result<Tensor> {
    let (n_tokens, hidden_dim) = hidden_states.dims2()?;
    let (_, top_k) = expert_indices.dims2()?;
    let n_ff = gate_weights.dims()[1];

    // Get unique experts used in this batch
    let indices_flat = expert_indices.flatten_all()?.to_vec1::<u32>()?;
    let mut unique_experts: Vec<usize> = indices_flat.iter().map(|&x| x as usize).collect();
    unique_experts.sort_unstable();
    unique_experts.dedup();
    let n_active = unique_experts.len();

    // Create mapping from global expert index to local (gathered) index
    let mut expert_to_local = vec![-1i32; gate_weights.dims()[0]];
    for (local_idx, &global_idx) in unique_experts.iter().enumerate() {
        expert_to_local[global_idx] = local_idx as i32;
    }

    // Remap expert_indices to local indices
    let local_indices: Vec<i32> = indices_flat
        .iter()
        .map(|&x| expert_to_local[x as usize])
        .collect();
    let local_indices = Tensor::from_vec(local_indices, (n_tokens, top_k), hidden_states.device())?;

    // Gather active expert weights
    let expert_idx_tensor = Tensor::from_vec(
        unique_experts.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        (n_active,),
        gate_weights.device(),
    )?;

    // gate_gathered: [n_active, n_ff, hidden_dim]
    let gate_gathered = gate_weights.index_select(&expert_idx_tensor, 0)?;
    let up_gathered = up_weights.index_select(&expert_idx_tensor, 0)?;
    let down_gathered = down_weights.index_select(&expert_idx_tensor, 0)?;

    // Gate projection: [n_tokens, hidden_dim] @ [n_active, n_ff, hidden_dim]^T
    // -> [n_tokens, top_k, n_ff]
    let gate_out =
        parallel_indexed_matmul(hidden_states, &gate_gathered, &local_indices, top_k, n_ff)?;

    // Up projection
    let up_out = parallel_indexed_matmul(hidden_states, &up_gathered, &local_indices, top_k, n_ff)?;

    // Fused SwiGLU
    let activated = fused_swiglu(&gate_out, &up_out)?;

    // Down projection: [n_tokens, top_k, n_ff] @ [n_active, hidden_dim, n_ff]^T
    // -> [n_tokens, top_k, hidden_dim]
    // Reshape for matmul: [n_tokens * top_k, n_ff]
    let activated_flat = activated.reshape((n_tokens * top_k, n_ff))?;

    // Expand local indices for down projection
    let local_indices_flat = local_indices.reshape((n_tokens * top_k,))?;
    let local_indices_expanded = local_indices_flat.unsqueeze(1)?;

    let down_out = parallel_indexed_matmul(
        &activated_flat,
        &down_gathered,
        &local_indices_expanded,
        1,
        hidden_dim,
    )?;
    let down_out = down_out.reshape((n_tokens, top_k, hidden_dim))?;

    // Weighted sum
    let weights_expanded = expert_weights.unsqueeze(2)?;
    let weighted = down_out.broadcast_mul(&weights_expanded)?;
    weighted.sum(1)
}

/// Parallel indexed matmul for expert-parallel MoE
///
/// Uses batched operations to enable GPU parallelism:
/// 1. Expands input to [n_tokens, top_k, k] by repeating each token
/// 2. Gathers expert weights into [n_tokens, top_k, n, k]
/// 3. Performs batched matmul: [n_tokens, top_k, 1, k] @ [n_tokens, top_k, k, n]
///    = [n_tokens, top_k, 1, n] -> [n_tokens, top_k, n]
fn parallel_indexed_matmul(
    input: &Tensor,          // [n_tokens, k]
    weights: &Tensor,        // [n_active_experts, n, k]
    expert_indices: &Tensor, // [n_tokens, top_k]
    _top_k: usize,
    n: usize,
) -> Result<Tensor> {
    let (n_tokens, k) = input.dims2()?;
    let _n_active = weights.dims()[0];

    // Ensure indices are 2D [n_tokens, top_k]
    let indices = if expert_indices.dims().len() == 1 {
        expert_indices.unsqueeze(1)?
    } else {
        expert_indices.clone()
    };
    let actual_top_k = indices.dims()[1];

    // Flatten indices for gather: [n_tokens * top_k]
    let indices_flat = indices.flatten_all()?;

    // Clamp indices to valid range
    let indices_u32 = indices_flat.to_dtype(candle::DType::U32)?;

    // Gather expert weights: weights[indices] -> [n_tokens * top_k, n, k]
    let gathered_weights = weights.index_select(&indices_u32, 0)?;
    // Reshape to [n_tokens, top_k, n, k]
    let gathered_weights = gathered_weights.reshape((n_tokens, actual_top_k, n, k))?;

    // Expand input: [n_tokens, k] -> [n_tokens, top_k, 1, k]
    let input_expanded = input
        .unsqueeze(1)? // [n_tokens, 1, k]
        .unsqueeze(2)? // [n_tokens, 1, 1, k]
        .broadcast_as((n_tokens, actual_top_k, 1, k))?
        .contiguous()?;

    // Transpose weights for matmul: [n_tokens, top_k, n, k] -> [n_tokens, top_k, k, n]
    let weights_t = gathered_weights.transpose(2, 3)?;

    // Batched matmul: [n_tokens, top_k, 1, k] @ [n_tokens, top_k, k, n]
    //              = [n_tokens, top_k, 1, n]
    let result = input_expanded.matmul(&weights_t)?;

    // Squeeze: [n_tokens, top_k, 1, n] -> [n_tokens, top_k, n]
    result.squeeze(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_topk_moe_routing_cpu() -> Result<()> {
        let device = Device::Cpu;
        let weights = Tensor::randn(0f32, 1., (2, 4, 8), &device)?;
        let weights = candle_nn::ops::softmax_last_dim(&weights)?;

        let (top_weights, top_indices) = topk_moe_routing(&weights, 2)?;

        assert_eq!(top_weights.dims(), &[2, 4, 2]);
        assert_eq!(top_indices.dims(), &[2, 4, 2]);

        // Check normalization
        let sums = top_weights.sum_keepdim(D::Minus1)?;
        let sums_vec = sums.flatten_all()?.to_vec1::<f32>()?;
        for sum in sums_vec {
            assert!((sum - 1.0).abs() < 1e-5, "Sum should be 1.0, got {}", sum);
        }

        Ok(())
    }

    #[test]
    fn test_l2_normalize_scale_cpu() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device)?;

        let result = l2_normalize_scale(&x, 0.5, 1e-6)?;

        // Check output shape
        assert_eq!(result.dims(), &[2, 3]);

        // Check that norms are scaled correctly
        let result_sq = result.sqr()?;
        let norms = result_sq.sum_keepdim(D::Minus1)?.sqrt()?;
        let norms_vec = norms.flatten_all()?.to_vec1::<f32>()?;
        for norm in norms_vec {
            assert!(
                (norm - 0.5).abs() < 1e-5,
                "Norm should be 0.5 (scale), got {}",
                norm
            );
        }

        Ok(())
    }
}
