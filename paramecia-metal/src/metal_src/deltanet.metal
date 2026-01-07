// =============================================================================
// Delta Net / Linear Attention Metal Kernels
// =============================================================================
// Based on the CUDA implementation from candle-kernels/src/deltanet.cu

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// L2 Normalize + Scale (fused)
// =============================================================================
// Computes: output = input / ||input||_2 * scale

kernel void l2_normalize_scale_f32(
    device const float* input      [[buffer(0)]],
    device float* output           [[buffer(1)]],
    constant float& scale          [[buffer(2)]],
    constant float& eps            [[buffer(3)]],
    constant uint& batch           [[buffer(4)]],
    constant uint& dim             [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tpitg [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const uint bid = tgpig.x;
    if (bid >= batch) return;
    
    device const float* in_ptr = input + bid * dim;
    device float* out_ptr = output + bid * dim;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = tpitg; i < dim; i += tptg.x) {
        float val = in_ptr[i];
        sum_sq += val * val;
    }
    
    // Simdgroup reduction
    sum_sq = simd_sum(sum_sq);
    
    // Cross-simdgroup reduction
    threadgroup float warp_sums[32];
    uint num_simdgroups = (tptg.x + 31) / 32;
    
    if (tiisg == 0) {
        warp_sums[sgitg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sgitg == 0) {
        sum_sq = (tiisg < num_simdgroups) ? warp_sums[tiisg] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (tiisg == 0) {
            warp_sums[0] = rsqrt(sum_sq + eps) * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float norm_scale = warp_sums[0];
    
    // Apply normalization
    for (uint i = tpitg; i < dim; i += tptg.x) {
        out_ptr[i] = in_ptr[i] * norm_scale;
    }
}

kernel void l2_normalize_scale_f16(
    device const half* input       [[buffer(0)]],
    device half* output            [[buffer(1)]],
    constant float& scale          [[buffer(2)]],
    constant float& eps            [[buffer(3)]],
    constant uint& batch           [[buffer(4)]],
    constant uint& dim             [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tpitg [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const uint bid = tgpig.x;
    if (bid >= batch) return;
    
    device const half* in_ptr = input + bid * dim;
    device half* out_ptr = output + bid * dim;
    
    float sum_sq = 0.0f;
    for (uint i = tpitg; i < dim; i += tptg.x) {
        float val = static_cast<float>(in_ptr[i]);
        sum_sq += val * val;
    }
    
    sum_sq = simd_sum(sum_sq);
    
    threadgroup float warp_sums[32];
    uint num_simdgroups = (tptg.x + 31) / 32;
    
    if (tiisg == 0) {
        warp_sums[sgitg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sgitg == 0) {
        sum_sq = (tiisg < num_simdgroups) ? warp_sums[tiisg] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (tiisg == 0) {
            warp_sums[0] = rsqrt(sum_sq + eps) * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float norm_scale = warp_sums[0];
    
    for (uint i = tpitg; i < dim; i += tptg.x) {
        out_ptr[i] = static_cast<half>(static_cast<float>(in_ptr[i]) * norm_scale);
    }
}

// =============================================================================
// Depthwise 1D Convolution
// =============================================================================

kernel void depthwise_conv1d_f32(
    device const float* input      [[buffer(0)]],
    device const float* weight     [[buffer(1)]],
    device float* output           [[buffer(2)]],
    constant uint& batch           [[buffer(3)]],
    constant uint& channels        [[buffer(4)]],
    constant uint& input_len       [[buffer(5)]],
    constant uint& output_len      [[buffer(6)]],
    constant uint& kernel_size     [[buffer(7)]],
    uint idx [[thread_position_in_grid]]
) {
    const uint total = batch * channels * output_len;
    if (idx >= total) return;
    
    const uint b = idx / (channels * output_len);
    const uint c = (idx / output_len) % channels;
    const uint o = idx % output_len;
    
    float acc = 0.0f;
    
    // Optimized for common kernel sizes
    if (kernel_size == 4) {
        const uint base_in = b * channels * input_len + c * input_len + o;
        const uint base_w = c * 4;
        acc = input[base_in] * weight[base_w]
            + input[base_in + 1] * weight[base_w + 1]
            + input[base_in + 2] * weight[base_w + 2]
            + input[base_in + 3] * weight[base_w + 3];
    } else if (kernel_size == 3) {
        const uint base_in = b * channels * input_len + c * input_len + o;
        const uint base_w = c * 3;
        acc = input[base_in] * weight[base_w]
            + input[base_in + 1] * weight[base_w + 1]
            + input[base_in + 2] * weight[base_w + 2];
    } else {
        for (uint k = 0; k < kernel_size; ++k) {
            const uint in_idx = b * channels * input_len + c * input_len + o + k;
            const uint w_idx = c * kernel_size + k;
            acc += input[in_idx] * weight[w_idx];
        }
    }
    
    output[idx] = acc;
}

kernel void depthwise_conv1d_f16(
    device const half* input       [[buffer(0)]],
    device const half* weight      [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& batch           [[buffer(3)]],
    constant uint& channels        [[buffer(4)]],
    constant uint& input_len       [[buffer(5)]],
    constant uint& output_len      [[buffer(6)]],
    constant uint& kernel_size     [[buffer(7)]],
    uint idx [[thread_position_in_grid]]
) {
    const uint total = batch * channels * output_len;
    if (idx >= total) return;
    
    const uint b = idx / (channels * output_len);
    const uint c = (idx / output_len) % channels;
    const uint o = idx % output_len;
    
    float acc = 0.0f;
    
    // Optimized for common kernel sizes
    if (kernel_size == 4) {
        const uint base_in = b * channels * input_len + c * input_len + o;
        const uint base_w = c * 4;
        acc = float(input[base_in]) * float(weight[base_w])
            + float(input[base_in + 1]) * float(weight[base_w + 1])
            + float(input[base_in + 2]) * float(weight[base_w + 2])
            + float(input[base_in + 3]) * float(weight[base_w + 3]);
    } else if (kernel_size == 3) {
        const uint base_in = b * channels * input_len + c * input_len + o;
        const uint base_w = c * 3;
        acc = float(input[base_in]) * float(weight[base_w])
            + float(input[base_in + 1]) * float(weight[base_w + 1])
            + float(input[base_in + 2]) * float(weight[base_w + 2]);
    } else {
        for (uint k = 0; k < kernel_size; ++k) {
            const uint in_idx = b * channels * input_len + c * input_len + o + k;
            const uint w_idx = c * kernel_size + k;
            acc += float(input[in_idx]) * float(weight[w_idx]);
        }
    }
    
    output[idx] = half(acc);
}

// =============================================================================
// Fused SwiGLU Activation (silu(gate) * up)
// =============================================================================

kernel void fused_swiglu_f32(
    device const float* gate       [[buffer(0)]],
    device const float* up         [[buffer(1)]],
    device float* output           [[buffer(2)]],
    constant uint& total           [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= total) return;
    
    const float g = gate[idx];
    const float u = up[idx];
    
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const float silu_g = g / (1.0f + exp(-g));
    output[idx] = silu_g * u;
}

kernel void fused_swiglu_f16(
    device const half* gate        [[buffer(0)]],
    device const half* up          [[buffer(1)]],
    device half* output            [[buffer(2)]],
    constant uint& total           [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= total) return;
    
    const float g = float(gate[idx]);
    const float u = float(up[idx]);
    
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const float silu_g = g / (1.0f + exp(-g));
    output[idx] = half(silu_g * u);
}

// =============================================================================
// Fused Gated RMS Norm
// =============================================================================
// out = silu(z) * rms_norm(x, weight)

kernel void gated_rms_norm_f32(
    device const float* x          [[buffer(0)]],
    device const float* z          [[buffer(1)]],
    device const float* weight     [[buffer(2)]],
    device float* output           [[buffer(3)]],
    constant float& eps            [[buffer(4)]],
    constant uint& batch           [[buffer(5)]],
    constant uint& dim             [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tpitg [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const uint vec_idx = tgpig.x;
    if (vec_idx >= batch) return;
    
    device const float* x_ptr = x + vec_idx * dim;
    device const float* z_ptr = z + vec_idx * dim;
    device float* out_ptr = output + vec_idx * dim;
    
    // Compute sum of squares for RMS norm
    float sum_sq = 0.0f;
    for (uint i = tpitg; i < dim; i += tptg.x) {
        float xi = x_ptr[i];
        sum_sq += xi * xi;
    }
    
    // Simdgroup reduction
    sum_sq = simd_sum(sum_sq);
    
    // Cross-simdgroup reduction
    threadgroup float warp_sums[32];
    uint num_simdgroups = (tptg.x + 31) / 32;
    
    if (tiisg == 0) {
        warp_sums[sgitg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sgitg == 0) {
        sum_sq = (tiisg < num_simdgroups) ? warp_sums[tiisg] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (tiisg == 0) {
            warp_sums[0] = rsqrt(sum_sq / float(dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float rms_scale = warp_sums[0];
    
    // Apply gated RMS norm: silu(z) * (x * weight * rms_scale)
    for (uint i = tpitg; i < dim; i += tptg.x) {
        float zi = z_ptr[i];
        float silu_z = zi / (1.0f + exp(-zi));
        float rms_x = x_ptr[i] * weight[i] * rms_scale;
        out_ptr[i] = silu_z * rms_x;
    }
}

// =============================================================================
// Lower triangular solve for (I - L) @ X = B
// =============================================================================
// Solves (I - L) @ X = B where L is strictly lower triangular.
// Formula: X[i] = B[i] + sum_{j<i} L[i,j] * X[j]

kernel void solve_i_minus_lower_tri_f32(
    device const float* L          [[buffer(0)]],
    device const float* B          [[buffer(1)]],
    device float* X                [[buffer(2)]],
    constant uint& batch_size      [[buffer(3)]],
    constant uint& n               [[buffer(4)]],
    constant uint& k               [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tpitg [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const uint bid = tgpig.x;
    if (bid >= batch_size) return;
    
    const uint lane = tpitg % 32;
    const uint col_base = sgitg;
    const uint k_simdgroups = (tptg.x + 31) / 32;
    
    device const float* L_batch = L + bid * n * n;
    device const float* B_batch = B + bid * n * k;
    device float* X_batch = X + bid * n * k;
    
    // Copy B to X first
    for (uint col = col_base; col < k; col += k_simdgroups) {
        for (uint row = lane; row < n; row += 32) {
            X_batch[row * k + col] = B_batch[row * k + col];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Forward substitution: X[i] = B[i] + sum_{j<i} L[i,j] * X[j]
    for (uint row = 1; row < n; row++) {
        for (uint col = col_base; col < k; col += k_simdgroups) {
            float sum = 0.0f;
            
            // Parallelize over lanes within the simdgroup
            for (uint j = lane; j < row; j += 32) {
                sum += L_batch[row * n + j] * X_batch[j * k + col];
            }
            
            // Simdgroup reduction
            sum = simd_sum(sum);
            
            // Lane 0 updates X[row, col]
            if (lane == 0) {
                X_batch[row * k + col] += sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// =============================================================================
// Fused Delta Net Autoregressive Step
// =============================================================================
// Combines L2 norm, state decay, kv_mem, delta, state update, and output
// into a single kernel for single-token autoregressive generation.
//
// This closely follows the CUDA implementation.
//
// Inputs:
//   q, k, v: [batch, num_heads, head_dim] - squeezed from [b, 1, ...]
//   gate: [batch, num_heads] - log decay values
//   beta: [batch, num_heads] - gate values (pre-sigmoid)
//   state: [batch, num_heads, head_dim, head_dim] - recurrent state
//   scale: float - 1/sqrt(head_dim) for Q scaling
//   eps: float - epsilon for L2 norm
//
// Outputs:
//   output: [batch, num_heads, head_dim]
//   new_state: [batch, num_heads, head_dim, head_dim]
//
// Grid: (batch, num_heads, 1)
// Threadgroup: (min(head_dim, 256), 1, 1)

kernel void delta_net_autoregressive_step_f32(
    device const float* q          [[buffer(0)]],
    device const float* k          [[buffer(1)]],
    device const float* v          [[buffer(2)]],
    device const float* gate       [[buffer(3)]],
    device const float* beta       [[buffer(4)]],
    device const float* state      [[buffer(5)]],
    device float* output           [[buffer(6)]],
    device float* new_state        [[buffer(7)]],
    constant float& scale          [[buffer(8)]],
    constant float& eps            [[buffer(9)]],
    constant uint& batch           [[buffer(10)]],
    constant uint& num_heads       [[buffer(11)]],
    constant uint& head_dim        [[buffer(12)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tpitg [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    // Grid: (batch, num_heads)
    const uint b = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg;
    const uint num_threads = tptg.x;
    
    if (b >= batch || h >= num_heads) return;
    
    // Calculate offsets
    const uint qkv_offset = b * num_heads * head_dim + h * head_dim;
    const uint gate_offset = b * num_heads + h;
    const uint state_offset = b * num_heads * head_dim * head_dim + h * head_dim * head_dim;
    
    // Shared memory - use 256 max for head_dim, supporting Qwen3-Next and similar models
    // Qwen3-Next 80B has head_dim = d_inner/dt_rank = 8192/32 = 256
    threadgroup float s_q[256];
    threadgroup float s_k[256];
    threadgroup float s_v[256];
    threadgroup float s_kv_mem[256];
    threadgroup float s_delta[256];
    threadgroup float warp_sums[32];
    threadgroup float s_q_norm;
    threadgroup float s_k_norm;
    
    // Initialize shared memory (important for correctness)
    for (uint i = tid; i < 256; i += num_threads) {
        s_q[i] = 0.0f;
        s_k[i] = 0.0f;
        s_v[i] = 0.0f;
        s_kv_mem[i] = 0.0f;
        s_delta[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load q, k, v into shared memory and compute L2 norms
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint i = tid; i < head_dim; i += num_threads) {
        float qi = q[qkv_offset + i];
        float ki = k[qkv_offset + i];
        s_q[i] = qi;
        s_k[i] = ki;
        s_v[i] = v[qkv_offset + i];
        q_sum_sq += qi * qi;
        k_sum_sq += ki * ki;
    }
    
    // Simdgroup reduction for L2 norms
    q_sum_sq = simd_sum(q_sum_sq);
    k_sum_sq = simd_sum(k_sum_sq);
    
    // Cross-simdgroup reduction for q norm
    uint num_simdgroups = (num_threads + 31) / 32;
    if (tiisg == 0) {
        warp_sums[sgitg] = q_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sgitg == 0) {
        float sum = (tiisg < num_simdgroups) ? warp_sums[tiisg] : 0.0f;
        sum = simd_sum(sum);
        if (tiisg == 0) {
            s_q_norm = rsqrt(sum + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cross-simdgroup reduction for k norm
    if (tiisg == 0) {
        warp_sums[sgitg] = k_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sgitg == 0) {
        float sum = (tiisg < num_simdgroups) ? warp_sums[tiisg] : 0.0f;
        sum = simd_sum(sum);
        if (tiisg == 0) {
            s_k_norm = rsqrt(sum + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize q (with scale) and k
    // Guard against potential NaN/Inf in norms
    float q_scale_val = scale * s_q_norm;
    float k_scale_val = s_k_norm;
    if (!isfinite(q_scale_val)) q_scale_val = scale;
    if (!isfinite(k_scale_val)) k_scale_val = 1.0f;
    
    for (uint i = tid; i < head_dim; i += num_threads) {
        s_q[i] *= q_scale_val;
        s_k[i] *= k_scale_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load gate and beta with clamping for numerical stability
    float gate_val = gate[gate_offset];
    // Clamp gate to [-50, 50] to prevent exp overflow/underflow
    gate_val = clamp(gate_val, -50.0f, 50.0f);
    float g = exp(gate_val);
    
    float beta_val = beta[gate_offset];
    // Clamp beta to [-50, 50] to prevent sigmoid overflow
    beta_val = clamp(beta_val, -50.0f, 50.0f);
    float beta_sig = 1.0f / (1.0f + exp(-beta_val));
    
    // Step 1: Decay state FIRST
    // new_state[i,j] = state[i,j] * g
    for (uint i = tid; i < head_dim; i += num_threads) {
        for (uint j = 0; j < head_dim; ++j) {
            uint idx = state_offset + i * head_dim + j;
            new_state[idx] = state[idx] * g;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Compute kv_mem from DECAYED state
    // kv_mem[i] = sum_j(new_state[i, j] * k[j])
    for (uint i = tid; i < head_dim; i += num_threads) {
        float acc = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            acc += new_state[state_offset + i * head_dim + j] * s_k[j];
        }
        s_kv_mem[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 3: Compute delta = (v - kv_mem) * beta
    for (uint i = tid; i < head_dim; i += num_threads) {
        s_delta[i] = (s_v[i] - s_kv_mem[i]) * beta_sig;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 4: Update state and compute output
    // new_state[i,j] += k[j] * delta[i]
    // output[i] = sum_j(new_state[i,j] * q[j])
    for (uint i = tid; i < head_dim; i += num_threads) {
        float delta_i = s_delta[i];
        // Clamp delta to prevent extreme values
        delta_i = clamp(delta_i, -1e6f, 1e6f);
        
        float out_acc = 0.0f;
        for (uint j = 0; j < head_dim; ++j) {
            uint idx = state_offset + i * head_dim + j;
            float state_val = new_state[idx] + s_k[j] * delta_i;
            // Clamp state values to prevent accumulation of extreme values
            state_val = clamp(state_val, -1e6f, 1e6f);
            new_state[idx] = state_val;
            out_acc += state_val * s_q[j];
        }
        // Clamp output to prevent extreme values
        output[qkv_offset + i] = clamp(out_acc, -1e6f, 1e6f);
    }
}

// =============================================================================
// Multi-Token Delta Net State Update (for MTP/Speculative Decoding)
// =============================================================================
// Processes K tokens sequentially with parallel computation within each token.
//
// Inputs:
//   q: [batch, num_heads, num_tokens, head_dim] - already L2 normalized with scale
//   k: [batch, num_heads, num_tokens, head_dim] - already L2 normalized
//   v: [batch, num_heads, num_tokens, head_dim]
//   gate: [batch, num_heads, num_tokens] - log decay values
//   beta: [batch, num_heads, num_tokens] - already sigmoided
//   state: [batch, num_heads, head_dim, head_dim]
//
// Outputs:
//   output: [batch, num_heads, num_tokens, head_dim]
//   new_state: [batch, num_heads, head_dim, head_dim]

kernel void delta_net_multi_token_update_f32(
    device const float* q          [[buffer(0)]],
    device const float* k          [[buffer(1)]],
    device const float* v          [[buffer(2)]],
    device const float* gate       [[buffer(3)]],
    device const float* beta       [[buffer(4)]],
    device const float* state      [[buffer(5)]],
    device float* output           [[buffer(6)]],
    device float* new_state        [[buffer(7)]],
    constant uint& batch           [[buffer(8)]],
    constant uint& num_heads       [[buffer(9)]],
    constant uint& num_tokens      [[buffer(10)]],
    constant uint& head_dim        [[buffer(11)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tpitg [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const uint b = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg;
    
    if (b >= batch || h >= num_heads) return;
    
    // Shared memory for per-token vectors
    threadgroup float shared_mem[256 * 4];
    threadgroup float* s_k = shared_mem;
    threadgroup float* s_v = shared_mem + 256;
    threadgroup float* s_delta = shared_mem + 256 * 2;
    threadgroup float* s_kv_mem = shared_mem + 256 * 3;
    
    const uint qkv_base = b * num_heads * num_tokens * head_dim + h * num_tokens * head_dim;
    const uint gate_base = b * num_heads * num_tokens + h * num_tokens;
    const uint state_base = b * num_heads * head_dim * head_dim + h * head_dim * head_dim;
    
    // Copy initial state to new_state
    for (uint idx = tid; idx < head_dim * head_dim; idx += tptg.x) {
        new_state[state_base + idx] = state[state_base + idx];
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    // Process tokens sequentially
    for (uint t = 0; t < num_tokens; t++) {
        const uint token_offset = t * head_dim;
        
        // Load k, v for this token
        for (uint i = tid; i < head_dim; i += tptg.x) {
            s_k[i] = k[qkv_base + token_offset + i];
            s_v[i] = v[qkv_base + token_offset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float g = exp(gate[gate_base + t]);
        float beta_val = beta[gate_base + t];
        
        // Decay state
        for (uint i = tid; i < head_dim; i += tptg.x) {
            for (uint j = 0; j < head_dim; j++) {
                new_state[state_base + i * head_dim + j] *= g;
            }
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
        
        // Compute kv_mem
        for (uint i = tid; i < head_dim; i += tptg.x) {
            float acc = 0.0f;
            for (uint j = 0; j < head_dim; j++) {
                acc += new_state[state_base + i * head_dim + j] * s_k[j];
            }
            s_kv_mem[i] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute delta
        for (uint i = tid; i < head_dim; i += tptg.x) {
            s_delta[i] = (s_v[i] - s_kv_mem[i]) * beta_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Update state and compute output
        for (uint i = tid; i < head_dim; i += tptg.x) {
            float out_acc = 0.0f;
            
            for (uint j = 0; j < head_dim; j++) {
                float decayed_val = new_state[state_base + i * head_dim + j];
                float updated_val = decayed_val + s_k[j] * s_delta[i];
                new_state[state_base + i * head_dim + j] = updated_val;
                out_acc += updated_val * q[qkv_base + token_offset + j];
            }
            
            output[qkv_base + token_offset + i] = out_acc;
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    }
}

// =============================================================================
// Delta Net Parallel with Intermediate State Materialization
// =============================================================================
// Same as multi_token_update but also stores intermediate states.

kernel void delta_net_parallel_with_states_f32(
    device const float* q          [[buffer(0)]],
    device const float* k          [[buffer(1)]],
    device const float* v          [[buffer(2)]],
    device const float* gate       [[buffer(3)]],
    device const float* beta       [[buffer(4)]],
    device const float* state      [[buffer(5)]],
    device float* output           [[buffer(6)]],
    device float* new_state        [[buffer(7)]],
    device float* intermediate_states [[buffer(8)]],
    constant uint& batch           [[buffer(9)]],
    constant uint& num_heads       [[buffer(10)]],
    constant uint& num_tokens      [[buffer(11)]],
    constant uint& head_dim        [[buffer(12)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tpitg [[thread_index_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const uint b = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg;
    
    if (b >= batch || h >= num_heads) return;
    
    threadgroup float shared_mem[256 * 4];
    threadgroup float* s_k = shared_mem;
    threadgroup float* s_v = shared_mem + 256;
    threadgroup float* s_delta = shared_mem + 256 * 2;
    threadgroup float* s_kv_mem = shared_mem + 256 * 3;
    
    const uint qkv_base = b * num_heads * num_tokens * head_dim + h * num_tokens * head_dim;
    const uint gate_base = b * num_heads * num_tokens + h * num_tokens;
    const uint state_base = b * num_heads * head_dim * head_dim + h * head_dim * head_dim;
    const uint inter_state_base = b * num_heads * num_tokens * head_dim * head_dim 
                                + h * num_tokens * head_dim * head_dim;
    
    // Copy initial state
    for (uint idx = tid; idx < head_dim * head_dim; idx += tptg.x) {
        new_state[state_base + idx] = state[state_base + idx];
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    
    // Process tokens
    for (uint t = 0; t < num_tokens; t++) {
        const uint token_offset = t * head_dim;
        const uint inter_state_offset = t * head_dim * head_dim;
        
        for (uint i = tid; i < head_dim; i += tptg.x) {
            s_k[i] = k[qkv_base + token_offset + i];
            s_v[i] = v[qkv_base + token_offset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float g = exp(gate[gate_base + t]);
        float beta_val = beta[gate_base + t];
        
        // Decay state
        for (uint i = tid; i < head_dim; i += tptg.x) {
            for (uint j = 0; j < head_dim; j++) {
                new_state[state_base + i * head_dim + j] *= g;
            }
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
        
        // Compute kv_mem
        for (uint i = tid; i < head_dim; i += tptg.x) {
            float acc = 0.0f;
            for (uint j = 0; j < head_dim; j++) {
                acc += new_state[state_base + i * head_dim + j] * s_k[j];
            }
            s_kv_mem[i] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute delta
        for (uint i = tid; i < head_dim; i += tptg.x) {
            s_delta[i] = (s_v[i] - s_kv_mem[i]) * beta_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Update state, compute output, AND save intermediate state
        for (uint i = tid; i < head_dim; i += tptg.x) {
            float out_acc = 0.0f;
            
            for (uint j = 0; j < head_dim; j++) {
                float decayed_val = new_state[state_base + i * head_dim + j];
                float updated_val = decayed_val + s_k[j] * s_delta[i];
                new_state[state_base + i * head_dim + j] = updated_val;
                
                // Save to intermediate state buffer
                intermediate_states[inter_state_base + inter_state_offset + i * head_dim + j] = updated_val;
                
                out_acc += updated_val * q[qkv_base + token_offset + j];
            }
            
            output[qkv_base + token_offset + i] = out_acc;
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    }
}

// =============================================================================
// Top-K Softmax for MoE Routing
// =============================================================================

kernel void topk_softmax_f32(
    device const float* logits     [[buffer(0)]],
    device float* weights          [[buffer(1)]],
    device int* indices            [[buffer(2)]],
    constant uint& n_tokens        [[buffer(3)]],
    constant uint& n_experts       [[buffer(4)]],
    constant uint& topk            [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    const uint token_idx = tgpig.x;
    if (token_idx >= n_tokens) return;
    
    const uint lane = tiisg;
    device const float* token_logits = logits + token_idx * n_experts;
    device float* token_weights = weights + token_idx * topk;
    device int* token_indices = indices + token_idx * topk;
    
    // Each thread handles ceil(n_experts / 32) experts
    const uint experts_per_thread = (n_experts + 31) / 32;
    
    float vals[8];  // Max 8 experts per thread (256 experts)
    float max_val = -INFINITY;
    
    for (uint i = 0; i < experts_per_thread && i < 8; i++) {
        uint expert = lane + i * 32;
        vals[i] = (expert < n_experts) ? token_logits[expert] : -INFINITY;
        max_val = max(max_val, vals[i]);
    }
    
    max_val = simd_max(max_val);
    
    float sum = 0.0f;
    for (uint i = 0; i < experts_per_thread && i < 8; i++) {
        uint expert = lane + i * 32;
        if (expert < n_experts) {
            vals[i] = exp(vals[i] - max_val);
            sum += vals[i];
        } else {
            vals[i] = 0.0f;
        }
    }
    sum = simd_sum(sum);
    float inv_sum = 1.0f / sum;
    
    for (uint i = 0; i < experts_per_thread && i < 8; i++) {
        vals[i] *= inv_sum;
    }
    
    // Find top-k
    for (uint k_idx = 0; k_idx < topk; k_idx++) {
        float best_val = -INFINITY;
        int best_expert = -1;
        
        for (uint i = 0; i < experts_per_thread && i < 8; i++) {
            int expert = lane + i * 32;
            if (expert < (int)n_experts && vals[i] > best_val) {
                best_val = vals[i];
                best_expert = expert;
            }
        }
        
        // Simdgroup reduction for max
        for (int mask = 16; mask > 0; mask /= 2) {
            float other_val = simd_shuffle_xor(best_val, mask);
            int other_expert = simd_shuffle_xor(best_expert, mask);
            if (other_val > best_val || (other_val == best_val && other_expert < best_expert)) {
                best_val = other_val;
                best_expert = other_expert;
            }
        }
        
        // Lane 0 writes result
        if (lane == 0) {
            token_weights[k_idx] = best_val;
            token_indices[k_idx] = best_expert;
        }
        
        // Broadcast selected expert to all lanes and mark as used
        best_expert = simd_broadcast_first(best_expert);
        for (uint i = 0; i < experts_per_thread && i < 8; i++) {
            if ((int)(lane + i * 32) == best_expert) {
                vals[i] = -INFINITY;
            }
        }
    }
    
    // Normalize selected weights
    if (lane == 0) {
        float norm_sum = 0.0f;
        for (uint k_idx = 0; k_idx < topk; k_idx++) {
            norm_sum += token_weights[k_idx];
        }
        float norm_inv = 1.0f / norm_sum;
        for (uint k_idx = 0; k_idx < topk; k_idx++) {
            token_weights[k_idx] *= norm_inv;
        }
    }
}
