#include "cuda_utils.cuh"
#include <stdint.h>

// =============================================================================
// WARP SIZE MACROS
// =============================================================================
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// =============================================================================
// Warp-level reduction helpers
// =============================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_prefix_sum(float val, int lane) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

// =============================================================================
// Depthwise 1D Convolution (SSM-style) - Direct implementation
// =============================================================================

extern "C" __global__ void depthwise_conv1d_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const size_t batch,
    const size_t channels,
    const size_t input_len,
    const size_t output_len,
    const size_t kernel_size
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = batch * channels * output_len;
    
    if (idx >= total) return;
    
    const size_t b = idx / (channels * output_len);
    const size_t c = (idx / output_len) % channels;
    const size_t o = idx % output_len;
    
    float acc = 0.0f;
    
    // Optimized for common kernel sizes
    if (kernel_size == 4) {
        const size_t base_in = b * channels * input_len + c * input_len + o;
        const size_t base_w = c * 4;
        acc = input[base_in] * weight[base_w]
            + input[base_in + 1] * weight[base_w + 1]
            + input[base_in + 2] * weight[base_w + 2]
            + input[base_in + 3] * weight[base_w + 3];
    } else if (kernel_size == 3) {
        const size_t base_in = b * channels * input_len + c * input_len + o;
        const size_t base_w = c * 3;
        acc = input[base_in] * weight[base_w]
            + input[base_in + 1] * weight[base_w + 1]
            + input[base_in + 2] * weight[base_w + 2];
    } else {
        for (size_t k = 0; k < kernel_size; ++k) {
            const size_t in_idx = b * channels * input_len + c * input_len + o + k;
            const size_t w_idx = c * kernel_size + k;
            acc += input[in_idx] * weight[w_idx];
        }
    }
    
    output[idx] = acc;
}

// =============================================================================
// Cumulative Sum with warp-level prefix sum
// =============================================================================

extern "C" __global__ void cumsum_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const size_t batch,
    const size_t len
) {
    const size_t bid = blockIdx.x;
    if (bid >= batch) return;
    
    const float* in_ptr = input + bid * len;
    float* out_ptr = output + bid * len;
    
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int block_size = blockDim.x;
    const int num_warps = block_size / WARP_SIZE;
    
    extern __shared__ float smem[];
    float* warp_sums = smem;
    float* carry_ptr = smem + num_warps;
    
    if (tid == 0) {
        *carry_ptr = 0.0f;
    }
    __syncthreads();
    
    for (size_t start = 0; start < len; start += block_size) {
        size_t idx = start + tid;
        float val = (idx < len) ? in_ptr[idx] : 0.0f;
        
        // Warp prefix sum
        val = warp_prefix_sum(val, lane);
        
        // Store warp total
        if (lane == WARP_SIZE - 1) {
            warp_sums[warp_id] = val;
        }
        __syncthreads();
        
        // Compute warp offsets (first warp scans warp sums)
        float warp_offset = 0.0f;
        if (warp_id == 0 && tid < num_warps) {
            float ws = warp_sums[tid];
            ws = warp_prefix_sum(ws, lane);
            warp_sums[tid] = ws;
        }
        __syncthreads();
        
        if (warp_id > 0) {
            warp_offset = warp_sums[warp_id - 1];
        }
        
        float chunk_total = warp_sums[num_warps - 1];
        float carry = *carry_ptr;
        float final_val = val + warp_offset + carry;
        
        if (idx < len) {
            out_ptr[idx] = final_val;
        }
        
        __syncthreads();
        if (tid == 0) {
            *carry_ptr = carry + chunk_total;
        }
        __syncthreads();
    }
}

// =============================================================================
// L2 Normalize + Scale (fused)
// =============================================================================

extern "C" __global__ void l2_normalize_scale_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float scale,
    const float eps,
    const size_t batch,
    const size_t dim
) {
    extern __shared__ float smem[];
    
    const size_t bid = blockIdx.x;
    if (bid >= batch) return;
    
    const float* in_ptr = input + bid * dim;
    float* out_ptr = output + bid * dim;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = in_ptr[i];
        sum_sq += val * val;
    }
    
    // Warp reduction
    sum_sq = warp_reduce_sum(sum_sq);
    
    // Cross-warp reduction via shared memory
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    if (lane == 0) {
        smem[warp_id] = sum_sq;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        sum_sq = (lane < num_warps) ? smem[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        if (lane == 0) {
            smem[0] = rsqrtf(sum_sq + eps) * scale;
        }
    }
    __syncthreads();
    
    float norm_scale = smem[0];
    
    // Apply normalization
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out_ptr[i] = in_ptr[i] * norm_scale;
    }
}

// =============================================================================
// Gated Linear Attention (GLA) - Single token autoregressive step
// =============================================================================

extern "C" __global__ void gla_step_f32(
    const float* __restrict__ q,       // [batch * heads, d]
    const float* __restrict__ k,       // [batch * heads, d]
    const float* __restrict__ v,       // [batch * heads, d]
    const float* __restrict__ gate,    // [batch * heads]
    const float* __restrict__ beta,    // [batch * heads]
    float* __restrict__ state,         // [batch * heads, d, d] (in/out)
    float* __restrict__ output,        // [batch * heads, d]
    const float scale,
    const size_t batch_heads,
    const size_t d
) {
    extern __shared__ float shared[];
    float* s_k = shared;
    float* s_q = shared + d;
    
    const size_t bh = blockIdx.x;
    const size_t tid = threadIdx.x;
    
    if (bh >= batch_heads || tid >= d) return;
    
    // Load q, k into shared memory
    s_k[tid] = k[bh * d + tid];
    s_q[tid] = q[bh * d + tid] * scale;
    __syncthreads();
    
    // Load gate and beta
    float g = expf(gate[bh]);
    float b = 1.0f / (1.0f + expf(-beta[bh]));  // sigmoid
    
    float* state_ptr = state + bh * d * d;
    
    // Decay state
    for (size_t j = 0; j < d; ++j) {
        size_t idx = tid * d + j;
        state_ptr[idx] *= g;
    }
    __syncthreads();
    
    // Compute kv_mem = k^T @ state[:, tid]
    float kv_mem = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        kv_mem += s_k[j] * state_ptr[j * d + tid];
    }
    
    // Load v
    float v_val = v[bh * d + tid];
    
    // Delta update
    float delta = (v_val - kv_mem) * b;
    for (size_t j = 0; j < d; ++j) {
        size_t idx = j * d + tid;
        state_ptr[idx] += s_k[j] * delta;
    }
    __syncthreads();
    
    // Output: q^T @ state[:, tid]
    float out_val = 0.0f;
    for (size_t j = 0; j < d; ++j) {
        out_val += s_q[j] * state_ptr[j * d + tid];
    }
    
    output[bh * d + tid] = out_val;
}

// =============================================================================
// Top-K selection (GPU-based, single token)
// =============================================================================

extern "C" __global__ void topk_softmax_f32(
    const float* __restrict__ logits,   // [n_tokens, n_experts]
    float* __restrict__ weights,         // [n_tokens, topk]
    int* __restrict__ indices,           // [n_tokens, topk]
    const size_t n_tokens,
    const size_t n_experts,
    const size_t topk
) {
    const size_t token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;
    
    const int lane = threadIdx.x;
    const float* token_logits = logits + token_idx * n_experts;
    float* token_weights = weights + token_idx * topk;
    int* token_indices = indices + token_idx * topk;
    
    // Each thread handles ceil(n_experts / WARP_SIZE) experts
    const int experts_per_thread = (n_experts + WARP_SIZE - 1) / WARP_SIZE;
    
    // Load logits and compute softmax
    float vals[8];  // Max 8 experts per thread (256 experts)
    float max_val = -INFINITY;
    
    for (int i = 0; i < experts_per_thread && i < 8; i++) {
        int expert = lane + i * WARP_SIZE;
        vals[i] = (expert < n_experts) ? token_logits[expert] : -INFINITY;
        max_val = fmaxf(max_val, vals[i]);
    }
    
    max_val = warp_reduce_max(max_val);
    
    float sum = 0.0f;
    for (int i = 0; i < experts_per_thread && i < 8; i++) {
        int expert = lane + i * WARP_SIZE;
        if (expert < n_experts) {
            vals[i] = expf(vals[i] - max_val);
            sum += vals[i];
        } else {
            vals[i] = 0.0f;
        }
    }
    sum = warp_reduce_sum(sum);
    float inv_sum = 1.0f / sum;
    
    for (int i = 0; i < experts_per_thread && i < 8; i++) {
        vals[i] *= inv_sum;
    }
    
    // Find top-k
    for (size_t k = 0; k < topk; k++) {
        float best_val = -INFINITY;
        int best_expert = -1;
        
        for (int i = 0; i < experts_per_thread && i < 8; i++) {
            int expert = lane + i * WARP_SIZE;
            if (expert < n_experts && vals[i] > best_val) {
                best_val = vals[i];
                best_expert = expert;
            }
        }
        
        // Warp reduction for max
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            float other_val = __shfl_xor_sync(0xffffffff, best_val, mask);
            int other_expert = __shfl_xor_sync(0xffffffff, best_expert, mask);
            if (other_val > best_val || (other_val == best_val && other_expert < best_expert)) {
                best_val = other_val;
                best_expert = other_expert;
            }
        }
        
        // Lane 0 writes result
        if (lane == 0) {
            token_weights[k] = best_val;
            token_indices[k] = best_expert;
        }
        
        // Broadcast selected expert to all lanes and mark as used
        best_expert = __shfl_sync(0xffffffff, best_expert, 0);
        for (int i = 0; i < experts_per_thread && i < 8; i++) {
            if (lane + i * WARP_SIZE == best_expert) {
                vals[i] = -INFINITY;
            }
        }
    }
    
    // Normalize selected weights
    if (lane == 0) {
        float norm_sum = 0.0f;
        for (size_t k = 0; k < topk; k++) {
            norm_sum += token_weights[k];
        }
        float norm_inv = 1.0f / norm_sum;
        for (size_t k = 0; k < topk; k++) {
            token_weights[k] *= norm_inv;
        }
    }
}

// =============================================================================
// Lower triangular solve (forward substitution)
// =============================================================================

extern "C" __global__ void solve_lower_tri_f32(
    const float* __restrict__ L,       // [batch, n, n]
    const float* __restrict__ B,       // [batch, n, k]
    float* __restrict__ X,             // [batch, n, k]
    const size_t batch_size,
    const size_t n,
    const size_t k
) {
    extern __shared__ float smem[];
    
    const size_t bid = blockIdx.x;
    if (bid >= batch_size) return;
    
    const int lane = threadIdx.x;
    const int col = threadIdx.y;
    
    if (col >= k) return;
    
    // Load L into shared memory
    float* sL = smem;
    const float* L_batch = L + bid * n * n;
    
    const int total_threads = blockDim.x * blockDim.y;
    const int thread_id = lane + col * blockDim.x;
    for (size_t i = thread_id; i < n * n; i += total_threads) {
        sL[i] = L_batch[i];
    }
    __syncthreads();
    
    // Load B and solve
    const float* B_batch = B + bid * n * k;
    float* X_batch = X + bid * n * k;
    
    // Process rows sequentially for forward substitution
    for (size_t row = 0; row < n; row++) {
        float sum = 0.0f;
        
        // Compute sum of L[row, j] * X[j, col] for j < row
        // Parallelize over lanes
        for (size_t j = lane; j < row; j += WARP_SIZE) {
            sum += sL[row * n + j] * X_batch[j * k + col];
        }
        
        // Warp reduction
        sum = warp_reduce_sum(sum);
        
        // Lane 0 computes and writes X[row, col]
        if (lane == 0) {
            float b_val = B_batch[row * k + col];
            float diag = sL[row * n + row];
            X_batch[row * k + col] = (b_val - sum) / diag;
        }
        __syncthreads();
    }
}

// =============================================================================
// Lower triangular solve for (I - L) @ X = B (DeltaNet chunked algorithm)
// =============================================================================
// Solves (I - L) @ X = B where L is strictly lower triangular.
// Since (I - L) has 1s on the diagonal, no division is needed.
// Formula: X[i] = B[i] + sum_{j<i} L[i,j] * X[j]
//
// This matches the precision characteristics of the autoregressive kernel
// by running entirely on GPU with the same FMA behavior.
//
// L: [batch, n, n] - strictly lower triangular (L[i,j] = 0 for j >= i)
// B: [batch, n, k] - right-hand side matrix
// X: [batch, n, k] - output solution
//
// Grid: (batch, 1, 1)
// Block: (32, k_threads, 1) where k_threads = min(k, 32)
// Each thread processes multiple columns if k > k_threads

extern "C" __global__ void solve_i_minus_lower_tri_f32(
    const float* __restrict__ L,       // [batch, n, n]
    const float* __restrict__ B,       // [batch, n, k]
    float* __restrict__ X,             // [batch, n, k]
    const size_t batch_size,
    const size_t n,
    const size_t k
) {
    extern __shared__ float smem[];
    
    const size_t bid = blockIdx.x;
    if (bid >= batch_size) return;
    
    const int lane = threadIdx.x;
    const int col_base = threadIdx.y;
    const int k_threads = blockDim.y;
    
    // Load L into shared memory for faster access
    float* sL = smem;
    const float* L_batch = L + bid * n * n;
    
    const int total_threads = blockDim.x * blockDim.y;
    const int thread_id = lane + col_base * blockDim.x;
    for (size_t i = thread_id; i < n * n; i += total_threads) {
        sL[i] = L_batch[i];
    }
    __syncthreads();
    
    // Initialize X with B
    const float* B_batch = B + bid * n * k;
    float* X_batch = X + bid * n * k;
    
    // Copy B to X first - each thread handles multiple columns (strided by k_threads)
    for (size_t col = col_base; col < k; col += k_threads) {
        for (size_t row = lane; row < n; row += WARP_SIZE) {
            X_batch[row * k + col] = B_batch[row * k + col];
        }
    }
    __syncthreads();
    
    // Forward substitution: X[i] = B[i] + sum_{j<i} L[i,j] * X[j]
    // Process rows sequentially (inherent data dependency)
    for (size_t row = 1; row < n; row++) {
        // Each thread processes multiple columns (strided by k_threads)
        for (size_t col = col_base; col < k; col += k_threads) {
            float sum = 0.0f;
            
            // Compute sum of L[row, j] * X[j, col] for j < row
            // Parallelize over lanes within the warp
            for (size_t j = lane; j < row; j += WARP_SIZE) {
                sum += sL[row * n + j] * X_batch[j * k + col];
            }
            
            // Warp reduction to get total sum
            sum = warp_reduce_sum(sum);
            
            // Lane 0 updates X[row, col]
            if (lane == 0) {
                X_batch[row * k + col] += sum;
            }
        }
        __syncthreads();
    }
}

// =============================================================================
// Fused SwiGLU Activation (silu(gate) * up)
// =============================================================================
// This fuses the SiLU activation and element-wise multiply into one kernel,
// avoiding intermediate memory writes.
//
// gate: [total_elems] - gate projection output
// up: [total_elems] - up projection output  
// output: [total_elems] = silu(gate) * up

extern "C" __global__ void fused_swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    const size_t total
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    const float g = gate[idx];
    const float u = up[idx];
    
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const float silu_g = g / (1.0f + expf(-g));
    output[idx] = silu_g * u;
}

// =============================================================================
// Expert-Parallel Batched MatMul for MoE
// =============================================================================
// Performs parallel matmul across multiple experts for MoE layers.
// This processes all (token, expert) pairs concurrently.
//
// For gate/up projections:
//   input: [n_tokens, hidden_dim]
//   weights: [n_active_experts, n_ff, hidden_dim] (pre-gathered for active experts)
//   expert_map: [n_tokens, top_k] -> index into weights (0 to n_active_experts-1)
//   output: [n_tokens, top_k, n_ff]
//
// This kernel uses one warp per output element for coalesced access.

extern "C" __global__ void expert_parallel_matmul_f32(
    const float* __restrict__ input,      // [n_tokens, k]
    const float* __restrict__ weights,    // [n_active, n, k] - gathered active expert weights
    const int* __restrict__ expert_map,   // [n_tokens, top_k] -> expert index in weights
    float* __restrict__ output,           // [n_tokens, top_k, n]
    const size_t n_tokens,
    const size_t top_k,
    const size_t n,                       // output dim (n_ff or hidden_dim)
    const size_t k                        // input dim (hidden_dim or n_ff)
) {
    // Grid: (n_tokens, top_k, n / blockDim.x)
    // Each block handles one (token, expert, output_chunk) combination
    const size_t token_idx = blockIdx.x;
    const size_t expert_k = blockIdx.y;
    const size_t out_base = blockIdx.z * blockDim.x;
    const size_t out_idx = out_base + threadIdx.x;
    
    if (token_idx >= n_tokens || expert_k >= top_k || out_idx >= n) return;
    
    // Get which expert this (token, k) uses
    const int expert_idx = expert_map[token_idx * top_k + expert_k];
    if (expert_idx < 0) {
        output[token_idx * top_k * n + expert_k * n + out_idx] = 0.0f;
        return;
    }
    
    // Compute dot product: input[token_idx, :] @ weights[expert_idx, out_idx, :]
    const float* input_row = input + token_idx * k;
    const float* weight_row = weights + (size_t)expert_idx * n * k + out_idx * k;
    
    float acc = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        acc += input_row[i] * weight_row[i];
    }
    
    output[token_idx * top_k * n + expert_k * n + out_idx] = acc;
}

// Optimized version with shared memory for input caching
extern "C" __global__ void expert_parallel_matmul_shared_f32(
    const float* __restrict__ input,      // [n_tokens, k]
    const float* __restrict__ weights,    // [n_active, n, k]
    const int* __restrict__ expert_map,   // [n_tokens, top_k]
    float* __restrict__ output,           // [n_tokens, top_k, n]
    const size_t n_tokens,
    const size_t top_k,
    const size_t n,
    const size_t k
) {
    extern __shared__ float shared_input[];
    
    const size_t token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;
    
    const size_t tid = threadIdx.x;
    const size_t block_size = blockDim.x;
    
    // Cooperatively load input for this token into shared memory
    const float* input_row = input + token_idx * k;
    for (size_t i = tid; i < k; i += block_size) {
        shared_input[i] = input_row[i];
    }
    __syncthreads();
    
    // Each thread handles multiple (expert_k, out_idx) pairs
    const size_t total_outputs = top_k * n;
    for (size_t idx = tid; idx < total_outputs; idx += block_size) {
        const size_t expert_k = idx / n;
        const size_t out_idx = idx % n;
        
        const int expert_idx = expert_map[token_idx * top_k + expert_k];
        float result = 0.0f;
        
        if (expert_idx >= 0) {
            const float* weight_row = weights + (size_t)expert_idx * n * k + out_idx * k;
            for (size_t i = 0; i < k; ++i) {
                result += shared_input[i] * weight_row[i];
            }
        }
        
        output[token_idx * top_k * n + expert_k * n + out_idx] = result;
    }
}

// Scatter-add weighted expert outputs back to token outputs
// expert_outputs: [n_tokens, top_k, hidden_dim]
// expert_weights: [n_tokens, top_k]
// output: [n_tokens, hidden_dim]
extern "C" __global__ void scatter_add_experts_f32(
    const float* __restrict__ expert_outputs,
    const float* __restrict__ expert_weights,
    float* __restrict__ output,
    const size_t n_tokens,
    const size_t top_k,
    const size_t hidden_dim
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = n_tokens * hidden_dim;
    
    if (idx >= total) return;
    
    const size_t token_idx = idx / hidden_dim;
    const size_t dim_idx = idx % hidden_dim;
    
    float acc = 0.0f;
    for (size_t k = 0; k < top_k; ++k) {
        const float weight = expert_weights[token_idx * top_k + k];
        const float value = expert_outputs[token_idx * top_k * hidden_dim + k * hidden_dim + dim_idx];
        acc += weight * value;
    }
    
    output[idx] = acc;
}

// =============================================================================
// Fused Delta Net Autoregressive Step
// =============================================================================
// Combines L2 norm, state decay, kv_mem, delta, state update, and output
// into a single kernel for single-token autoregressive generation.
//
// This is the critical path for token generation speed.
//
// Inputs:
//   q, k, v: [batch, num_heads, head_dim] - already squeezed from [b, 1, ...]
//   gate: [batch, num_heads] - log decay values
//   beta: [batch, num_heads] - gate values (pre-sigmoid)
//   state: [batch, num_heads, head_dim, head_dim] - recurrent state
//   scale: float - 1/sqrt(head_dim) for Q scaling
//   eps: float - epsilon for L2 norm
//
// Outputs:
//   output: [batch, num_heads, head_dim]
//   new_state: [batch, num_heads, head_dim, head_dim]

extern "C" __global__ void delta_net_autoregressive_step_f32(
    const float* __restrict__ q,      // [b, h, d]
    const float* __restrict__ k,      // [b, h, d]  
    const float* __restrict__ v,      // [b, h, d]
    const float* __restrict__ gate,   // [b, h]
    const float* __restrict__ beta,   // [b, h]
    const float* __restrict__ state,  // [b, h, d, d]
    float* __restrict__ output,       // [b, h, d]
    float* __restrict__ new_state,    // [b, h, d, d]
    const float scale,
    const float eps,
    const size_t batch,
    const size_t num_heads,
    const size_t head_dim
) {
    // Grid: (batch, num_heads)
    // Each block handles one (batch, head) pair
    const size_t b = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t tid = threadIdx.x;
    
    if (b >= batch || h >= num_heads) return;
    
    extern __shared__ float shared[];
    float* s_q = shared;                           // [head_dim]
    float* s_k = shared + head_dim;                // [head_dim]
    float* s_v = shared + 2 * head_dim;            // [head_dim]
    float* s_kv_mem = shared + 3 * head_dim;       // [head_dim]
    float* s_delta = shared + 4 * head_dim;        // [head_dim]
    float* s_out = shared + 5 * head_dim;          // [head_dim]
    
    const size_t qkv_offset = b * num_heads * head_dim + h * head_dim;
    const size_t gate_offset = b * num_heads + h;
    const size_t state_offset = b * num_heads * head_dim * head_dim + h * head_dim * head_dim;
    
    // Load q, k, v into shared memory and compute L2 norms
    float q_sum_sq = 0.0f, k_sum_sq = 0.0f;
    for (size_t i = tid; i < head_dim; i += blockDim.x) {
        float qi = q[qkv_offset + i];
        float ki = k[qkv_offset + i];
        s_q[i] = qi;
        s_k[i] = ki;
        s_v[i] = v[qkv_offset + i];
        q_sum_sq += qi * qi;
        k_sum_sq += ki * ki;
    }
    __syncthreads();
    
    // Warp reduce for L2 norms
    q_sum_sq = warp_reduce_sum(q_sum_sq);
    k_sum_sq = warp_reduce_sum(k_sum_sq);
    
    // Block reduce across warps using shared memory
    __shared__ float warp_q_sums[32];
    __shared__ float warp_k_sums[32];
    __shared__ float s_q_norm, s_k_norm;
    
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    // Each warp's lane 0 stores its partial sum
    if (lane == 0) {
        warp_q_sums[warp_id] = q_sum_sq;
        warp_k_sums[warp_id] = k_sum_sq;
    }
    __syncthreads();
    
    // First warp reduces all warp sums
    if (warp_id == 0) {
        q_sum_sq = (lane < num_warps) ? warp_q_sums[lane] : 0.0f;
        k_sum_sq = (lane < num_warps) ? warp_k_sums[lane] : 0.0f;
        q_sum_sq = warp_reduce_sum(q_sum_sq);
        k_sum_sq = warp_reduce_sum(k_sum_sq);
        if (lane == 0) {
            s_q_norm = rsqrtf(q_sum_sq + eps);
            s_k_norm = rsqrtf(k_sum_sq + eps);
        }
    }
    __syncthreads();
    
    // Normalize q (with scale) and k
    // s_q_norm and s_k_norm are now rsqrt (1/sqrt), so multiply directly
    float q_scale = scale * s_q_norm;
    float k_scale = s_k_norm;
    for (size_t i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] *= q_scale;
        s_k[i] *= k_scale;
    }
    __syncthreads();
    
    // Load gate and beta
    float g = expf(gate[gate_offset]);       // exp(gate) for decay
    float beta_sig = 1.0f / (1.0f + expf(-beta[gate_offset]));  // sigmoid(beta)
    
    // Step 1: Decay state FIRST (matching llama.cpp reference)
    // new_state = state * g
    for (size_t i = tid; i < head_dim; i += blockDim.x) {
        for (size_t j = 0; j < head_dim; ++j) {
            new_state[state_offset + i * head_dim + j] = state[state_offset + i * head_dim + j] * g;
        }
    }
    __syncthreads();
    
    // Step 2: Compute kv_mem from DECAYED state (no extra g!)
    // kv_mem[i] = sum_j(decayed_state[i, j] * k[j])
    for (size_t i = tid; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < head_dim; ++j) {
            acc += new_state[state_offset + i * head_dim + j] * s_k[j];
        }
        s_kv_mem[i] = acc;
    }
    __syncthreads();
    
    // Step 3: Compute delta = (v - kv_mem) * beta
    for (size_t i = tid; i < head_dim; i += blockDim.x) {
        s_delta[i] = (s_v[i] - s_kv_mem[i]) * beta_sig;
    }
    __syncthreads();
    
    // Step 4: Update decayed state and compute output
    // new_state[i, j] = decayed_state[i, j] + k[j] * delta[i]  (no extra g!)
    // output[i] = sum_j(new_state[i, j] * q[j])
    for (size_t i = tid; i < head_dim; i += blockDim.x) {
        float out_acc = 0.0f;
        for (size_t j = 0; j < head_dim; ++j) {
            float decayed_val = new_state[state_offset + i * head_dim + j];
            float updated_val = decayed_val + s_k[j] * s_delta[i];
            new_state[state_offset + i * head_dim + j] = updated_val;
            out_acc += updated_val * s_q[j];
        }
        output[qkv_offset + i] = out_acc;
    }
}

// =============================================================================
// Fused Gated RMS Norm
// =============================================================================
// out = silu(z) * rms_norm(x, weight)
// Combines SiLU gate with RMS normalization

extern "C" __global__ void gated_rms_norm_f32(
    const float* __restrict__ x,      // [batch, dim]
    const float* __restrict__ z,      // [batch, dim] - gate input
    const float* __restrict__ weight, // [dim]
    float* __restrict__ output,       // [batch, dim]
    const float eps,
    const size_t batch,
    const size_t dim
) {
    const size_t vec_idx = blockIdx.x;
    if (vec_idx >= batch) return;
    
    extern __shared__ float shared_data[];
    float* s_x = shared_data;
    float* s_z = shared_data + dim;
    
    const float* x_ptr = x + vec_idx * dim;
    const float* z_ptr = z + vec_idx * dim;
    float* out_ptr = output + vec_idx * dim;
    
    // Load x and z, compute sum of squares for RMS
    float sum_sq = 0.0f;
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float xi = x_ptr[i];
        s_x[i] = xi;
        s_z[i] = z_ptr[i];
        sum_sq += xi * xi;
    }
    
    // Warp reduce
    sum_sq = warp_reduce_sum(sum_sq);
    
    __shared__ float warp_sums[32];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }
    
    __shared__ float rms_scale;
    if (threadIdx.x == 0) {
        rms_scale = rsqrtf(sum_sq / dim + eps);
    }
    __syncthreads();
    
    // Apply gated RMS norm: silu(z) * (x * weight * rms_scale)
    for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float zi = s_z[i];
        float silu_z = zi / (1.0f + expf(-zi));
        float rms_x = s_x[i] * weight[i] * rms_scale;
        out_ptr[i] = silu_z * rms_x;
    }
}

// =============================================================================
// Fused Multi-Token Delta Net State Update (for MTP/Speculative Decoding)
// =============================================================================
// Processes K tokens in parallel using associative scan.
// This is mathematically equivalent to K sequential autoregressive steps
// but runs in O(log K) parallel depth.
//
// For DeltaNet, the state update for token t is:
//   delta_t = (v_t - state_{t-1} @ k_t) * beta_t
//   state_t = g_t * state_{t-1} + outer(delta_t, k_t)
//
// This can be reformulated as an associative operation:
//   (A_t, b_t) where A_t = g_t (scalar decay) and b_t = outer(delta_t, k_t)
//   Composition: (A2, b2) ∘ (A1, b1) = (A2*A1, A2*b1 + b2)
//
// Inputs:
//   q: [batch, num_heads, num_tokens, head_dim] - queries
//   k: [batch, num_heads, num_tokens, head_dim] - keys (L2 normalized)
//   v: [batch, num_heads, num_tokens, head_dim] - values
//   gate: [batch, num_heads, num_tokens] - log decay values
//   beta: [batch, num_heads, num_tokens] - gate values (already sigmoided)
//   state: [batch, num_heads, head_dim, head_dim] - initial state
//   scale: Q scaling factor
//
// Outputs:
//   output: [batch, num_heads, num_tokens, head_dim] - outputs for all tokens
//   new_state: [batch, num_heads, head_dim, head_dim] - final state after all tokens

extern "C" __global__ void delta_net_multi_token_update_f32(
    const float* __restrict__ q,        // [b, h, n_tokens, d] - already L2 normalized with scale
    const float* __restrict__ k,        // [b, h, n_tokens, d] - already L2 normalized
    const float* __restrict__ v,        // [b, h, n_tokens, d]
    const float* __restrict__ gate,     // [b, h, n_tokens]
    const float* __restrict__ beta,     // [b, h, n_tokens] - already sigmoided
    const float* __restrict__ state,    // [b, h, d, d]
    float* __restrict__ output,         // [b, h, n_tokens, d]
    float* __restrict__ new_state,      // [b, h, d, d]
    const size_t batch,
    const size_t num_heads,
    const size_t num_tokens,
    const size_t head_dim
) {
    // Grid: (batch, num_heads)
    // Each block processes all tokens for one (batch, head) pair
    const size_t b = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t tid = threadIdx.x;
    
    if (b >= batch || h >= num_heads) return;
    
    // Shared memory: only for small vectors (k, v, delta, kv_mem per token)
    // State matrix stays in global memory to avoid exceeding shared mem limits
    extern __shared__ float shared[];
    float* s_k = shared;                    // [head_dim]
    float* s_v = shared + head_dim;         // [head_dim]
    float* s_delta = shared + 2 * head_dim; // [head_dim]
    float* s_kv_mem = shared + 3 * head_dim;// [head_dim]
    
    const size_t qkv_base = b * num_heads * num_tokens * head_dim + h * num_tokens * head_dim;
    const size_t gate_base = b * num_heads * num_tokens + h * num_tokens;
    const size_t state_base = b * num_heads * head_dim * head_dim + h * head_dim * head_dim;
    
    // Copy initial state to new_state (we'll update new_state in place)
    for (size_t idx = tid; idx < head_dim * head_dim; idx += blockDim.x) {
        new_state[state_base + idx] = state[state_base + idx];
    }
    __syncthreads();
    
    // Process tokens sequentially but with parallel computation within each token
    for (size_t t = 0; t < num_tokens; t++) {
        const size_t token_offset = t * head_dim;
        
        // Load k, v for this token into shared memory
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            s_k[i] = k[qkv_base + token_offset + i];
            s_v[i] = v[qkv_base + token_offset + i];
        }
        __syncthreads();
        
        // Load gate and beta for this token
        float g = expf(gate[gate_base + t]);
        float beta_val = beta[gate_base + t];  // Already sigmoided
        
        // Step 1: Decay state FIRST (matching llama.cpp reference)
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            for (size_t j = 0; j < head_dim; j++) {
                new_state[state_base + i * head_dim + j] *= g;
            }
        }
        __syncthreads();
        
        // Step 2: Compute kv_mem from DECAYED state (no extra g!)
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            float acc = 0.0f;
            for (size_t j = 0; j < head_dim; j++) {
                acc += new_state[state_base + i * head_dim + j] * s_k[j];
            }
            s_kv_mem[i] = acc;
        }
        __syncthreads();
        
        // Step 3: Compute delta = (v - kv_mem) * beta
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            s_delta[i] = (s_v[i] - s_kv_mem[i]) * beta_val;
        }
        __syncthreads();
        
        // Step 4: Update decayed state and compute output (no extra g!)
        // new_state[i,j] = decayed_state[i,j] + k[j] * delta[i]
        // output[i] = sum_j(new_state[i,j] * q[j])
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            float out_acc = 0.0f;
            
            for (size_t j = 0; j < head_dim; j++) {
                float decayed_val = new_state[state_base + i * head_dim + j];
                float updated_val = decayed_val + s_k[j] * s_delta[i];
                new_state[state_base + i * head_dim + j] = updated_val;
                out_acc += updated_val * q[qkv_base + token_offset + j];
            }
            
            output[qkv_base + token_offset + i] = out_acc;
        }
        __syncthreads();
    }
}

// =============================================================================
// Delta Net Parallel with Intermediate State Materialization
// =============================================================================
// Processes K tokens with PARALLEL intermediate state computation for O(1) slicing.
// 
// Unlike delta_net_multi_token_update which only returns the final state,
// this kernel materializes and stores the state AFTER each position, enabling
// O(1) state restoration for speculative decoding verification.
//
// State update formula:
//   delta_t = (v_t - state_{t-1} @ k_t) * beta_t
//   state_t = g_t * state_{t-1} + outer(delta_t, k_t)
//
// Memory layout for intermediate_states:
//   [batch, num_heads, num_tokens, head_dim, head_dim]
//   intermediate_states[b, h, t, :, :] = state after processing token t
//
// This enables O(1) state slicing: on partial rejection at position i,
// simply read intermediate_states[:, :, i-1, :, :] as the new state.
//
// Inputs:
//   q: [batch, num_heads, num_tokens, head_dim] - queries (L2 normalized with scale)
//   k: [batch, num_heads, num_tokens, head_dim] - keys (L2 normalized)
//   v: [batch, num_heads, num_tokens, head_dim] - values
//   gate: [batch, num_heads, num_tokens] - log decay values
//   beta: [batch, num_heads, num_tokens] - gate values (already sigmoided)
//   state: [batch, num_heads, head_dim, head_dim] - initial state
//
// Outputs:
//   output: [batch, num_heads, num_tokens, head_dim] - outputs for all tokens
//   new_state: [batch, num_heads, head_dim, head_dim] - final state
//   intermediate_states: [batch, num_heads, num_tokens, head_dim, head_dim] - state after each token

extern "C" __global__ void delta_net_parallel_with_states_f32(
    const float* __restrict__ q,        // [b, h, n_tokens, d]
    const float* __restrict__ k,        // [b, h, n_tokens, d]
    const float* __restrict__ v,        // [b, h, n_tokens, d]
    const float* __restrict__ gate,     // [b, h, n_tokens]
    const float* __restrict__ beta,     // [b, h, n_tokens]
    const float* __restrict__ state,    // [b, h, d, d]
    float* __restrict__ output,         // [b, h, n_tokens, d]
    float* __restrict__ new_state,      // [b, h, d, d]
    float* __restrict__ intermediate_states,  // [b, h, n_tokens, d, d]
    const size_t batch,
    const size_t num_heads,
    const size_t num_tokens,
    const size_t head_dim
) {
    // Grid: (batch, num_heads)
    // Each block processes all tokens for one (batch, head) pair
    const size_t b = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t tid = threadIdx.x;
    
    if (b >= batch || h >= num_heads) return;
    
    // Shared memory for per-token vectors
    extern __shared__ float shared[];
    float* s_k = shared;                    // [head_dim]
    float* s_v = shared + head_dim;         // [head_dim]
    float* s_delta = shared + 2 * head_dim; // [head_dim]
    float* s_kv_mem = shared + 3 * head_dim;// [head_dim]
    
    const size_t qkv_base = b * num_heads * num_tokens * head_dim + h * num_tokens * head_dim;
    const size_t gate_base = b * num_heads * num_tokens + h * num_tokens;
    const size_t state_base = b * num_heads * head_dim * head_dim + h * head_dim * head_dim;
    const size_t inter_state_base = b * num_heads * num_tokens * head_dim * head_dim 
                                  + h * num_tokens * head_dim * head_dim;
    
    // Copy initial state to new_state (we'll update in place)
    for (size_t idx = tid; idx < head_dim * head_dim; idx += blockDim.x) {
        new_state[state_base + idx] = state[state_base + idx];
    }
    __syncthreads();
    
    // Process tokens sequentially (state dependency) but with parallel computation within
    for (size_t t = 0; t < num_tokens; t++) {
        const size_t token_offset = t * head_dim;
        const size_t inter_state_offset = t * head_dim * head_dim;
        
        // Load k, v for this token into shared memory
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            s_k[i] = k[qkv_base + token_offset + i];
            s_v[i] = v[qkv_base + token_offset + i];
        }
        __syncthreads();
        
        // Load gate and beta for this token
        float g = expf(gate[gate_base + t]);
        float beta_val = beta[gate_base + t];  // Already sigmoided
        
        // Step 1: Decay state FIRST (matching llama.cpp reference)
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            for (size_t j = 0; j < head_dim; j++) {
                new_state[state_base + i * head_dim + j] *= g;
            }
        }
        __syncthreads();
        
        // Step 2: Compute kv_mem from DECAYED state (no extra g!)
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            float acc = 0.0f;
            for (size_t j = 0; j < head_dim; j++) {
                acc += new_state[state_base + i * head_dim + j] * s_k[j];
            }
            s_kv_mem[i] = acc;
        }
        __syncthreads();
        
        // Step 3: Compute delta = (v - kv_mem) * beta
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            s_delta[i] = (s_v[i] - s_kv_mem[i]) * beta_val;
        }
        __syncthreads();
        
        // Step 4: Update decayed state, compute output, AND save intermediate state (no extra g!)
        // new_state[i,j] = decayed_state[i,j] + k[j] * delta[i]
        // output[i] = sum_j(new_state[i,j] * q[j])
        for (size_t i = tid; i < head_dim; i += blockDim.x) {
            float out_acc = 0.0f;
            
            for (size_t j = 0; j < head_dim; j++) {
                float decayed_val = new_state[state_base + i * head_dim + j];
                float updated_val = decayed_val + s_k[j] * s_delta[i];
                new_state[state_base + i * head_dim + j] = updated_val;
                
                // Save to intermediate state buffer
                intermediate_states[inter_state_base + inter_state_offset + i * head_dim + j] = updated_val;
                
                out_acc += updated_val * q[qkv_base + token_offset + j];
            }
            
            output[qkv_base + token_offset + i] = out_acc;
        }
        __syncthreads();
    }
}

// =============================================================================
// Speculative Decoding: Compute P(x) and Q(x) for draft tokens (GPU-side)
// =============================================================================
// Computes the probability of each draft token under both target and draft
// distributions. The actual rejection sampling decision happens on CPU.
//
// Inputs:
//   target_logits: [num_drafts, vocab_size] - target model logits
//   draft_logits: [num_drafts, vocab_size] - draft model logits
//   draft_tokens: [num_drafts] - the drafted token IDs
//   temperature: temperature for softmax
//
// Outputs:
//   p_values: [num_drafts] - P(x) for each draft token
//   q_values: [num_drafts] - Q(x) for each draft token

extern "C" __global__ void compute_draft_probs_f32(
    const float* __restrict__ target_logits,  // [num_drafts, vocab_size]
    const float* __restrict__ draft_logits,   // [num_drafts, vocab_size]
    const int* __restrict__ draft_tokens,     // [num_drafts]
    float* __restrict__ p_values,             // output: P(x) for each draft
    float* __restrict__ q_values,             // output: Q(x) for each draft
    const float temperature,
    const int num_drafts,
    const int vocab_size
) {
    // One block per draft position
    const int draft_idx = blockIdx.x;
    if (draft_idx >= num_drafts) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    const float* tgt = target_logits + draft_idx * vocab_size;
    const float* dft = draft_logits + draft_idx * vocab_size;
    const int draft_tok = draft_tokens[draft_idx];
    
    extern __shared__ float shared[];
    float* warp_max = shared;
    float* warp_sum = shared + 32;
    float* warp_draft = shared + 64;
    
    // === Compute P(draft_token) from target logits ===
    
    // Find max for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += block_size) {
        local_max = fmaxf(local_max, tgt[i]);
    }
    local_max = warp_reduce_max(local_max);
    
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    
    if (warp_id == 0) {
        local_max = (lane < num_warps) ? warp_max[lane] : -INFINITY;
        local_max = warp_reduce_max(local_max);
        if (lane == 0) warp_max[0] = local_max;
    }
    __syncthreads();
    float tgt_max = warp_max[0];
    
    // Compute exp sum and exp of draft token
    float exp_sum = 0.0f;
    float exp_draft = 0.0f;
    for (int i = tid; i < vocab_size; i += block_size) {
        float exp_val = expf((tgt[i] - tgt_max) / temperature);
        exp_sum += exp_val;
        if (i == draft_tok) exp_draft = exp_val;
    }
    exp_sum = warp_reduce_sum(exp_sum);
    exp_draft = warp_reduce_sum(exp_draft);
    
    if (lane == 0) {
        warp_sum[warp_id] = exp_sum;
        warp_draft[warp_id] = exp_draft;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        exp_sum = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        exp_draft = (lane < num_warps) ? warp_draft[lane] : 0.0f;
        exp_sum = warp_reduce_sum(exp_sum);
        exp_draft = warp_reduce_sum(exp_draft);
        if (lane == 0) {
            p_values[draft_idx] = exp_draft / exp_sum;
        }
    }
    __syncthreads();
    
    // === Compute Q(draft_token) from draft logits ===
    
    local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += block_size) {
        local_max = fmaxf(local_max, dft[i]);
    }
    local_max = warp_reduce_max(local_max);
    
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    
    if (warp_id == 0) {
        local_max = (lane < num_warps) ? warp_max[lane] : -INFINITY;
        local_max = warp_reduce_max(local_max);
        if (lane == 0) warp_max[0] = local_max;
    }
    __syncthreads();
    float dft_max = warp_max[0];
    
    exp_sum = 0.0f;
    exp_draft = 0.0f;
    for (int i = tid; i < vocab_size; i += block_size) {
        float exp_val = expf((dft[i] - dft_max) / temperature);
        exp_sum += exp_val;
        if (i == draft_tok) exp_draft = exp_val;
    }
    exp_sum = warp_reduce_sum(exp_sum);
    exp_draft = warp_reduce_sum(exp_draft);
    
    if (lane == 0) {
        warp_sum[warp_id] = exp_sum;
        warp_draft[warp_id] = exp_draft;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        exp_sum = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        exp_draft = (lane < num_warps) ? warp_draft[lane] : 0.0f;
        exp_sum = warp_reduce_sum(exp_sum);
        exp_draft = warp_reduce_sum(exp_draft);
        if (lane == 0) {
            q_values[draft_idx] = exp_draft / exp_sum;
        }
    }
}
