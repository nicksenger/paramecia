// QuZO Fused Perturbation Kernels
// CUDA kernels that perform perturbation during matmul dequantization
//
// Instead of modifying weights beforehand, these kernels generate perturbations
// on-the-fly using deterministic RNG (Philox) during the forward pass.
//
// Key advantages:
// - No memory overhead for storing perturbed weights
// - Works seamlessly with CPU-offloaded experts
// - Perturbation is computed as weights stream through PCIe
//
// Per-tensor uniqueness:
// Each tensor gets a unique perturbation by XORing the seed with the weight
// tensor's device pointer. This ensures different tensors get different
// perturbation patterns even with the same global seed.

#include "cuda_fp16.h"
#include <stdint.h>

// Block sizes for GGML quantization formats
#define QK8_0 32
#define QK_K 256
#define QI8_0 (QK8_0 / 4)
#define QI4_K (QK_K / 32)
#define QR4_K 2

// Warp size
#define WARP_SIZE 32

// ============================================================================
// Philox 4x32 Counter-Based RNG
// ============================================================================
// Philox is ideal for GPU because:
// - Stateless: generate random(seed, index) without maintaining state
// - Deterministic: same (seed, index) always produces same value
// - High quality: passes BigCrush statistical tests
// - Fast: simple arithmetic operations, no transcendentals

// Philox constants (from the original paper)
#define PHILOX_M0 0xD2511F53
#define PHILOX_M1 0xCD9E8D57
#define PHILOX_W0 0x9E3779B9
#define PHILOX_W1 0xBB67AE85

// Single Philox round
__device__ __forceinline__ void philox_round(uint32_t& c0, uint32_t& c1,
                                              uint32_t& c2, uint32_t& c3,
                                              uint32_t k0, uint32_t k1) {
    uint32_t hi0, lo0, hi1, lo1;

    // Multiply and get high/low parts
    lo0 = c0 * PHILOX_M0;
    hi0 = __umulhi(c0, PHILOX_M0);
    lo1 = c2 * PHILOX_M1;
    hi1 = __umulhi(c2, PHILOX_M1);

    // Update counters with permutation
    uint32_t new_c0 = hi1 ^ c1 ^ k0;
    uint32_t new_c1 = lo1;
    uint32_t new_c2 = hi0 ^ c3 ^ k1;
    uint32_t new_c3 = lo0;

    c0 = new_c0;
    c1 = new_c1;
    c2 = new_c2;
    c3 = new_c3;
}

// Generate 4 uint32 random values from seed and index using Philox-4x32-10
__device__ __forceinline__ void philox4x32(uint64_t seed, uint64_t index,
                                            uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3) {
    // Initialize counter from index
    uint32_t c0 = (uint32_t)index;
    uint32_t c1 = (uint32_t)(index >> 32);
    uint32_t c2 = 0;
    uint32_t c3 = 0;

    // Initialize key from seed
    uint32_t k0 = (uint32_t)seed;
    uint32_t k1 = (uint32_t)(seed >> 32);

    // 10 rounds of Philox
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        philox_round(c0, c1, c2, c3, k0, k1);
        k0 += PHILOX_W0;
        k1 += PHILOX_W1;
    }

    r0 = c0;
    r1 = c1;
    r2 = c2;
    r3 = c3;
}

// Generate a single float in [-1, 1] from seed and index
__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t index) {
    uint32_t r0, r1, r2, r3;
    philox4x32(seed, index, r0, r1, r2, r3);
    // Convert to float in [0, 1) then shift to [-1, 1]
    return (float)(r0 >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

// Generate Gaussian-distributed float using Box-Muller (uses 2 uniforms)
__device__ __forceinline__ float philox_gaussian(uint64_t seed, uint64_t index) {
    uint32_t r0, r1, r2, r3;
    philox4x32(seed, index, r0, r1, r2, r3);

    // Box-Muller transform
    float u1 = (float)(r0 | 1) * (1.0f / 4294967296.0f);  // (0, 1]
    float u2 = (float)r1 * (1.0f / 4294967296.0f);        // [0, 1)

    float radius = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;

    return radius * cosf(theta);
}

// ============================================================================
// Q8_0 Block Structure
// ============================================================================

typedef struct {
    half    d;              // scale (delta)
    int8_t  qs[QK8_0];      // quantized values
} block_q8_0;

// ============================================================================
// Q4K Block Structure
// ============================================================================

typedef struct {
    half2 dm;                  // super-block scale (d) and min (m)
    uint8_t scales[12];        // sub-block scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4-bit quantized values (packed)
} block_q4_K;

// Helper to decode sub-block scale and min (matches GGML get_scale_min_k4)
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* scales,
                                                   uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q8_0 (Vector-Matrix)
// ============================================================================
// This kernel performs: y = (W + ε*z) @ x
// where z is generated on-the-fly from the seed and weight index.
//
// The perturbation is applied during dequantization:
//   dequant_value = quant_value * delta
//   perturbed_value = dequant_value + epsilon * z[weight_index]

extern "C" __global__ void fused_mul_mat_vec_q8_0_f32(
    const block_q8_0* __restrict__ vx,  // Quantized weights [nrows, ncols/32 blocks]
    const float* __restrict__ vy,        // Input vector [ncols]
    float* __restrict__ dst,             // Output vector [nrows]
    const int ncols,                     // Number of columns (K dimension)
    const int nrows,                     // Number of rows (output dimension)
    const uint64_t seed,                 // Random seed for perturbation
    const float epsilon,                 // Perturbation magnitude
    const uint64_t weight_ptr            // Weight tensor device pointer (for per-tensor uniqueness)
) {
    // XOR seed with weight pointer to get unique seed per tensor
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;  // Thread within warp (0-31)
    const int num_blocks = ncols / QK8_0;

    float sum = 0.0f;

    // Each warp processes one row
    // Threads cooperate to process blocks
    for (int block_idx = tid; block_idx < num_blocks; block_idx += WARP_SIZE) {
        const block_q8_0* block = &vx[row * num_blocks + block_idx];
        const float d = __half2float(block->d);

        // Process 32 elements in this block
        #pragma unroll
        for (int i = 0; i < QK8_0; i++) {
            const int col = block_idx * QK8_0 + i;
            const int weight_idx = row * ncols + col;

            // Dequantize
            float w = d * (float)block->qs[i];

            // Add perturbation using deterministic RNG
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            // Accumulate dot product
            sum += w * vy[col];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in warp writes result
    if (tid == 0) {
        dst[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q4K (Vector-Matrix)
// ============================================================================
// Handles the more complex Q4K dequantization with sub-block scales and mins.
// Dequantization formula: value = dall * sc * q_nibble - dmin * m

extern "C" __global__ void fused_mul_mat_vec_q4_K_f32(
    const block_q4_K* __restrict__ vx,  // Quantized weights
    const float* __restrict__ vy,        // Input vector
    float* __restrict__ dst,             // Output vector
    const int ncols,                     // Number of columns
    const int nrows,                     // Number of rows
    const uint64_t seed,                 // Random seed
    const float epsilon,                 // Perturbation magnitude
    const uint64_t weight_ptr            // Weight tensor device pointer
) {
    // XOR seed with weight pointer to get unique seed per tensor
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q4_K* block = &vx[row * num_blocks + block_idx];

        const float dall = __low2float(block->dm);
        const float dmin = __high2float(block->dm);

        // Process elements assigned to this thread
        // Q4K has 256 elements per block, 128 bytes of packed nibbles
        // Each thread processes 256/32 = 8 elements
        for (int elem_offset = tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Determine sub-block and get scales
            const int il = elem_offset / 64;
            const int is = 2 * il;
            const int within_il = elem_offset % 64;

            uint8_t sc, m;
            if (within_il < 32) {
                get_scale_min_k4(is + 0, block->scales, sc, m);
            } else {
                get_scale_min_k4(is + 1, block->scales, sc, m);
            }

            const float d_eff = dall * (float)sc;
            const float m_eff = dmin * (float)m;

            // Get quantized value (4-bit packed)
            // Q4K uses a specific packing layout:
            // - Byte at 32*il + r contains:
            //   - Low nibble (bits 0-3): element 64*il + r
            //   - High nibble (bits 4-7): element 64*il + r + 32
            int byte_idx, is_high_nibble;
            if (within_il < 32) {
                byte_idx = 32 * il + within_il;
                is_high_nibble = 0;
            } else {
                byte_idx = 32 * il + (within_il - 32);
                is_high_nibble = 1;
            }
            uint8_t packed = block->qs[byte_idx];
            uint8_t q4 = is_high_nibble ? (packed >> 4) : (packed & 0x0F);

            // Dequantize
            float w = d_eff * (float)q4 - m_eff;

            // Add perturbation
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            // Accumulate
            sum += w * vy[col];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        dst[row] = sum;
    }
}

// ============================================================================
// Batch Matrix Multiply Variants
// ============================================================================
// For larger batch sizes, we use tiled matrix multiplication with shared memory.
// The perturbation is applied when loading tiles from global memory.

// Shared memory tile dimensions
#define TILE_SIZE_M 64
#define TILE_SIZE_N 64
#define TILE_SIZE_K 32

extern "C" __global__ void fused_mul_mat_q8_0_f32(
    const block_q8_0* __restrict__ vx,  // Quantized weights [M, K/32 blocks]
    const float* __restrict__ vy,        // Input matrix [K, N]
    float* __restrict__ dst,             // Output matrix [M, N]
    const int M,                         // Weight rows
    const int N,                         // Input columns (batch)
    const int K,                         // Weight columns / Input rows
    const uint64_t seed,
    const float epsilon,
    const uint64_t weight_ptr
) {
    const uint64_t effective_seed = seed ^ weight_ptr;

    // Simplified tiled implementation
    // Each block computes a TILE_SIZE_M x TILE_SIZE_N tile of the output

    const int row = blockIdx.y * TILE_SIZE_M + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    const int num_blocks_k = K / QK8_0;
    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q8_0* block = &vx[row * num_blocks_k + kb];
        const float d = __half2float(block->d);

        #pragma unroll 4
        for (int i = 0; i < QK8_0; i++) {
            const int k = kb * QK8_0 + i;
            const int weight_idx = row * K + k;

            // Dequantize with perturbation
            float w = d * (float)block->qs[i];
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            sum += w * vy[k * N + col];
        }
    }

    dst[row * N + col] = sum;
}

extern "C" __global__ void fused_mul_mat_q4_K_f32(
    const block_q4_K* __restrict__ vx,
    const float* __restrict__ vy,
    float* __restrict__ dst,
    const int M,
    const int N,
    const int K,
    const uint64_t seed,
    const float epsilon,
    const uint64_t weight_ptr
) {
    const uint64_t effective_seed = seed ^ weight_ptr;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    const int num_blocks_k = K / QK_K;
    float sum = 0.0f;

    for (int kb = 0; kb < num_blocks_k; kb++) {
        const block_q4_K* block = &vx[row * num_blocks_k + kb];

        const float dall = __low2float(block->dm);
        const float dmin = __high2float(block->dm);

        for (int elem = 0; elem < QK_K; elem++) {
            const int k = kb * QK_K + elem;
            const int weight_idx = row * K + k;

            // Get sub-block scales
            const int il = elem / 64;
            const int is = 2 * il;
            const int within_il = elem % 64;

            uint8_t sc, m;
            if (within_il < 32) {
                get_scale_min_k4(is + 0, block->scales, sc, m);
            } else {
                get_scale_min_k4(is + 1, block->scales, sc, m);
            }

            const float d_eff = dall * (float)sc;
            const float m_eff = dmin * (float)m;

            // Get quantized nibble (Q4K layout)
            int byte_idx, is_high_nibble;
            if (within_il < 32) {
                byte_idx = 32 * il + within_il;
                is_high_nibble = 0;
            } else {
                byte_idx = 32 * il + (within_il - 32);
                is_high_nibble = 1;
            }
            uint8_t packed = block->qs[byte_idx];
            uint8_t q4 = is_high_nibble ? (packed >> 4) : (packed & 0x0F);

            // Dequantize with perturbation
            float w = d_eff * (float)q4 - m_eff;
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            sum += w * vy[k * N + col];
        }
    }

    dst[row * N + col] = sum;
}

// ============================================================================
// Half-precision output variants
// ============================================================================

extern "C" __global__ void fused_mul_mat_vec_q8_0_f16(
    const block_q8_0* __restrict__ vx,
    const half* __restrict__ vy,
    half* __restrict__ dst,
    const int ncols,
    const int nrows,
    const uint64_t seed,
    const float epsilon,
    const uint64_t weight_ptr
) {
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int num_blocks = ncols / QK8_0;

    float sum = 0.0f;

    for (int block_idx = tid; block_idx < num_blocks; block_idx += WARP_SIZE) {
        const block_q8_0* block = &vx[row * num_blocks + block_idx];
        const float d = __half2float(block->d);

        #pragma unroll
        for (int i = 0; i < QK8_0; i++) {
            const int col = block_idx * QK8_0 + i;
            const int weight_idx = row * ncols + col;

            float w = d * (float)block->qs[i];
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            sum += w * __half2float(vy[col]);
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        dst[row] = __float2half(sum);
    }
}

extern "C" __global__ void fused_mul_mat_vec_q4_K_f16(
    const block_q4_K* __restrict__ vx,
    const half* __restrict__ vy,
    half* __restrict__ dst,
    const int ncols,
    const int nrows,
    const uint64_t seed,
    const float epsilon,
    const uint64_t weight_ptr
) {
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q4_K* block = &vx[row * num_blocks + block_idx];

        const float dall = __low2float(block->dm);
        const float dmin = __high2float(block->dm);

        for (int elem_offset = tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            const int il = elem_offset / 64;
            const int is = 2 * il;
            const int within_il = elem_offset % 64;

            uint8_t sc, m;
            if (within_il < 32) {
                get_scale_min_k4(is + 0, block->scales, sc, m);
            } else {
                get_scale_min_k4(is + 1, block->scales, sc, m);
            }

            const float d_eff = dall * (float)sc;
            const float m_eff = dmin * (float)m;

            // Get quantized nibble (Q4K layout)
            int byte_idx, is_high_nibble;
            if (within_il < 32) {
                byte_idx = 32 * il + within_il;
                is_high_nibble = 0;
            } else {
                byte_idx = 32 * il + (within_il - 32);
                is_high_nibble = 1;
            }
            uint8_t packed = block->qs[byte_idx];
            uint8_t q4 = is_high_nibble ? (packed >> 4) : (packed & 0x0F);

            float w = d_eff * (float)q4 - m_eff;
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            sum += w * __half2float(vy[col]);
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        dst[row] = __float2half(sum);
    }
}

// ============================================================================
// BF16 Block Structures and Fused Kernels
// ============================================================================

#include "cuda_bf16.h"

// ============================================================================
// Fused Dequantize + Perturb + MatMul for BF16 (Vector-Matrix)
// ============================================================================
// BF16 doesn't need dequantization, just load -> perturb -> matmul
// Kernel: y = (W + ε*z) @ x where z is generated on-the-fly

extern "C" __global__ void fused_mul_mat_vec_bf16_f32(
    const __nv_bfloat16* __restrict__ vx,  // BF16 weights [nrows, ncols]
    const float* __restrict__ vy,           // Input vector [ncols]
    float* __restrict__ dst,                // Output vector [nrows]
    const int ncols,                        // Number of columns (K dimension)
    const int nrows,                        // Number of rows (output dimension)
    const uint64_t seed,                    // Random seed for perturbation
    const float epsilon,                    // Perturbation magnitude
    const uint64_t weight_ptr               // Weight tensor device pointer
) {
    // XOR seed with weight pointer to get unique seed per tensor
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;  // Thread within warp (0-31)

    float sum = 0.0f;

    // Each warp processes one row
    // Threads cooperate to process columns
    for (int col = tid; col < ncols; col += WARP_SIZE) {
        const int weight_idx = row * ncols + col;

        // Load BF16 weight and convert to float
        float w = __bfloat162float(vx[weight_idx]);

        // Add perturbation using deterministic RNG
        float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
        w += epsilon * z;

        // Accumulate dot product
        sum += w * vy[col];
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in warp writes result
    if (tid == 0) {
        dst[row] = sum;
    }
}

// ============================================================================
// K-Quant Block Structures
// ============================================================================

typedef struct {
    uint8_t scales[QK_K/16];   // scales and mins, quantized with 4 bits (16 bytes)
    uint8_t qs[QK_K/4];        // 2-bit quants (64 bytes)
    half2 dm;                  // super-block scale for quantized scales/mins
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K/8];     // high bit of quants (32 bytes)
    uint8_t qs[QK_K/4];        // low 2 bits of quants (64 bytes)
    uint8_t scales[12];        // scales, quantized with 6 bits
    half d;                    // super-block scale
} block_q3_K;

typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[12];           // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit (32 bytes)
    uint8_t qs[QK_K/2];           // quants, low 4 bits (128 bytes)
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];        // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];        // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16];   // scales (16 bytes, signed!)
    half    d;                 // delta
} block_q6_K;

// ============================================================================
// Q3K Helper: Get scale from 6-bit packed format
// ============================================================================

__device__ __forceinline__ int8_t get_q3k_scale_fused(int is, const uint8_t* scales) {
    int8_t us;
    if (is < 4) {
        us = (scales[is] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4);
    } else if (is < 8) {
        us = (scales[is] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4);
    } else if (is < 12) {
        us = (scales[is - 8] >> 4) | (((scales[is] >> 4) & 3) << 4);
    } else {
        us = (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
    }
    return us - 32;
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q6K (Vector-Matrix)
// ============================================================================

extern "C" __global__ void fused_mul_mat_vec_q6_K_f32(
    const block_q6_K* __restrict__ vx,
    const float* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const uint64_t seed,
    const float epsilon,
    const uint64_t weight_ptr
) {
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q6_K* block = &vx[row * num_blocks + block_idx];
        const float d = __half2float(block->d);

        // Process elements assigned to this thread
        for (int elem_offset = tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Q6K dequantization (matching GGML)
            const int ip = elem_offset / 128;
            const int il = (elem_offset - 128 * ip) % 32;
            const int is = 8 * ip + il / 16;
            const int elem_in_half = elem_offset % 128;

            int8_t sc;
            int q6_val;

            if (elem_in_half < 32) {
                sc = block->scales[is + 0];
                const uint8_t ql = block->ql[64 * ip + il];
                const uint8_t qh = block->qh[32 * ip + il];
                q6_val = (int)(ql & 0xF) | (((qh >> 0) & 3) << 4);
            } else if (elem_in_half < 64) {
                sc = block->scales[is + 2];
                const uint8_t ql = block->ql[64 * ip + il + 32];
                const uint8_t qh = block->qh[32 * ip + il];
                q6_val = (int)(ql & 0xF) | (((qh >> 2) & 3) << 4);
            } else if (elem_in_half < 96) {
                sc = block->scales[is + 4];
                const uint8_t ql = block->ql[64 * ip + il];
                const uint8_t qh = block->qh[32 * ip + il];
                q6_val = (int)(ql >> 4) | (((qh >> 4) & 3) << 4);
            } else {
                sc = block->scales[is + 6];
                const uint8_t ql = block->ql[64 * ip + il + 32];
                const uint8_t qh = block->qh[32 * ip + il];
                q6_val = (int)(ql >> 4) | (((qh >> 6) & 3) << 4);
            }

            float w = d * (float)sc * (float)(q6_val - 32);

            // Add perturbation
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            sum += w * vy[col];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        dst[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q5K (Vector-Matrix)
// ============================================================================

extern "C" __global__ void fused_mul_mat_vec_q5_K_f32(
    const block_q5_K* __restrict__ vx,
    const float* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const uint64_t seed,
    const float epsilon,
    const uint64_t weight_ptr
) {
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q5_K* block = &vx[row * num_blocks + block_idx];

        const float dall = __low2float(block->dm);
        const float dmin = __high2float(block->dm);

        for (int elem_offset = tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Q5K dequantization
            const int il = elem_offset / 64;   // 0..3
            const int within_64 = elem_offset % 64;
            const int is = 2 * il;

            uint8_t sc, m;
            int q5_val;

            if (within_64 < 32) {
                // Low nibble elements
                get_scale_min_k4(is + 0, block->scales, sc, m);
                const int byte_idx = 32 * il + within_64;
                const uint8_t ql = block->qs[byte_idx];
                const uint8_t qh = block->qh[within_64];
                const uint8_t hm = 1 << (2 * il);
                q5_val = (ql & 0xF) + ((qh & hm) ? 16 : 0);
            } else {
                // High nibble elements
                get_scale_min_k4(is + 1, block->scales, sc, m);
                const int within_32 = within_64 - 32;
                const int byte_idx = 32 * il + within_32;
                const uint8_t ql = block->qs[byte_idx];
                const uint8_t qh = block->qh[within_32];
                const uint8_t hm = 1 << (2 * il + 1);
                q5_val = (ql >> 4) + ((qh & hm) ? 16 : 0);
            }

            float w = dall * (float)sc * (float)q5_val - dmin * (float)m;

            // Add perturbation
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            sum += w * vy[col];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        dst[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q2K (Vector-Matrix)
// ============================================================================

extern "C" __global__ void fused_mul_mat_vec_q2_K_f32(
    const block_q2_K* __restrict__ vx,
    const float* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const uint64_t seed,
    const float epsilon,
    const uint64_t weight_ptr
) {
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q2_K* block = &vx[row * num_blocks + block_idx];

        const float dall = __low2float(block->dm);
        const float dmin = __high2float(block->dm);

        for (int elem_offset = tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Q2K dequantization
            const int n = elem_offset / 128;
            const int elem_in_n = elem_offset % 128;
            const int l = elem_in_n % 32;
            const int shift_idx = elem_in_n / 32;  // 0, 1, 2, 3

            const int is = 8 * n + l / 16;
            const int qs_idx = 32 * n + l;
            const uint8_t q_byte = block->qs[qs_idx];
            const uint8_t q2 = (q_byte >> (2 * shift_idx)) & 3;

            const float d_sc = dall * (float)(block->scales[is + 2 * shift_idx] & 0xF);
            const float d_min = dmin * (float)(block->scales[is + 2 * shift_idx] >> 4);

            float w = d_sc * (float)q2 - d_min;

            // Add perturbation
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            sum += w * vy[col];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        dst[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q3K (Vector-Matrix)
// ============================================================================

extern "C" __global__ void fused_mul_mat_vec_q3_K_f32(
    const block_q3_K* __restrict__ vx,
    const float* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const uint64_t seed,
    const float epsilon,
    const uint64_t weight_ptr
) {
    const uint64_t effective_seed = seed ^ weight_ptr;
    vy += (size_t)blockIdx.y * ncols;
    dst += (size_t)blockIdx.y * nrows;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const block_q3_K* block = &vx[row * num_blocks + block_idx];
        const float d_all = __half2float(block->d);

        for (int elem_offset = tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Q3K dequantization (matching GGML layout)
            const int n = elem_offset / 128;       // 0 or 1
            const int j = (elem_offset % 128) / 32; // 0, 1, 2, 3
            const int l = elem_offset % 32;

            const uint8_t m = 1 << (4 * n + j);
            const int is0 = l / 16;
            const int is = 8 * n + 2 * j + is0;
            const int shift = 2 * j;

            const int8_t scale = get_q3k_scale_fused(is, block->scales);
            const float dl = d_all * (float)scale;

            const uint8_t q_byte = block->qs[32 * n + l];
            const uint8_t q2_val = (q_byte >> shift) & 3;
            const int q3_val = q2_val - ((block->hmask[l] & m) ? 0 : 4);

            float w = dl * (float)q3_val;

            // Add perturbation
            float z = philox_gaussian(effective_seed, (uint64_t)weight_idx);
            w += epsilon * z;

            sum += w * vy[col];
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        dst[row] = sum;
    }
}
