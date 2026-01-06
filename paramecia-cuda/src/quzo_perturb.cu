// QuZO Perturbation Kernels
// CUDA kernels for in-place perturbation of quantized tensors using stochastic rounding
//
// These kernels implement the core QuZO operation:
//   w' = Q(w + ε * z)
// where Q is stochastic rounding to the quantized format.

#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <stdint.h>

// Block sizes for GGML quantization formats
#define QK8_0 32
#define QK_K 256

// ============================================================================
// Random Number Generation (xorshift64* for stochastic rounding)
// ============================================================================

// Fast GPU-friendly RNG based on xorshift64*
// Each thread gets a unique sequence based on global_seed + thread_id
__device__ __forceinline__ uint64_t xorshift64star(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// Returns uniform float in [0, 1)
__device__ __forceinline__ float rand_uniform(uint64_t* state) {
    uint64_t r = xorshift64star(state);
    // Use upper 23 bits for mantissa (float precision)
    return (r >> 40) * (1.0f / 16777216.0f);  // 2^-24
}

// Stochastic rounding: round to nearest with probability proportional to fractional part
__device__ __forceinline__ int8_t stochastic_round_i8(float x, uint64_t* rng_state) {
    // Clamp to int8 range
    x = fmaxf(-127.0f, fminf(127.0f, x));

    float floor_x = floorf(x);
    float frac = x - floor_x;

    // Round up with probability = frac, down with probability = 1-frac
    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    // Final clamp to int8
    return (int8_t)max(-127, min(127, result));
}

// Stochastic rounding for 4-bit values (0-15 range)
__device__ __forceinline__ uint8_t stochastic_round_u4(float x, uint64_t* rng_state) {
    x = fmaxf(0.0f, fminf(15.0f, x));

    float floor_x = floorf(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (uint8_t)max(0, min(15, result));
}

// Stochastic rounding for 2-bit values (0-3 range)
__device__ __forceinline__ uint8_t stochastic_round_u2(float x, uint64_t* rng_state) {
    x = fmaxf(0.0f, fminf(3.0f, x));

    float floor_x = floorf(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (uint8_t)max(0, min(3, result));
}

// Stochastic rounding for 3-bit signed values (-4 to 3 range)
__device__ __forceinline__ int8_t stochastic_round_i3(float x, uint64_t* rng_state) {
    x = fmaxf(-4.0f, fminf(3.0f, x));

    float floor_x = floorf(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (int8_t)max(-4, min(3, result));
}

// Stochastic rounding for 5-bit values (0-31 range)
__device__ __forceinline__ uint8_t stochastic_round_u5(float x, uint64_t* rng_state) {
    x = fmaxf(0.0f, fminf(31.0f, x));

    float floor_x = floorf(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (uint8_t)max(0, min(31, result));
}

// Stochastic rounding for 6-bit signed values (-32 to 31 range)
__device__ __forceinline__ int8_t stochastic_round_i6(float x, uint64_t* rng_state) {
    x = fmaxf(-32.0f, fminf(31.0f, x));

    float floor_x = floorf(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (int8_t)max(-32, min(31, result));
}

// ============================================================================
// Q8_0 Block Structure
// ============================================================================

typedef struct {
    half    d;              // scale (delta)
    int8_t  qs[QK8_0];      // quantized values
} block_q8_0;

// ============================================================================
// Q4K Block Structure (QK_K = 256)
// ============================================================================

typedef struct {
    half2 dm;                  // super-block scale (d) and min (m)
    uint8_t scales[12];        // sub-block scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4-bit quantized values (packed)
} block_q4_K;

// ============================================================================
// Q2K Block Structure (QK_K = 256)
// ============================================================================

typedef struct {
    uint8_t scales[QK_K/16];   // scales and mins, quantized with 4 bits (16 bytes)
    uint8_t qs[QK_K/4];        // 2-bit quants (64 bytes)
    half2 dm;                  // super-block scale for quantized scales/mins
} block_q2_K;

// ============================================================================
// Q3K Block Structure (QK_K = 256)
// ============================================================================

typedef struct {
    uint8_t hmask[QK_K/8];     // high bit of quants (32 bytes)
    uint8_t qs[QK_K/4];        // low 2 bits of quants (64 bytes)
    uint8_t scales[12];        // scales, quantized with 6 bits
    half d;                    // super-block scale
} block_q3_K;

// ============================================================================
// Q5K Block Structure (QK_K = 256)
// ============================================================================

typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[12];           // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit (32 bytes)
    uint8_t qs[QK_K/2];           // quants, low 4 bits (128 bytes)
} block_q5_K;

// ============================================================================
// Q6K Block Structure (QK_K = 256)
// ============================================================================

typedef struct {
    uint8_t ql[QK_K/2];        // quants, lower 4 bits (128 bytes)
    uint8_t qh[QK_K/4];        // quants, upper 2 bits (64 bytes)
    int8_t  scales[QK_K/16];   // scales (16 bytes, signed!)
    half    d;                 // delta
} block_q6_K;

// ============================================================================
// Q8_0 Perturbation Kernel
// ============================================================================
// Each block handles one Q8_0 block (32 elements)
// blockDim.x = 32 (one thread per element)

extern "C" __global__ void perturb_q8_0(
    block_q8_0* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float epsilon,
    const uint64_t seed,
    const int add  // 1 = add, 0 = subtract
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= QK8_0) return;

    // Initialize RNG state unique to this element
    uint64_t rng_state = seed ^ ((uint64_t)block_idx * QK8_0 + tid) * 0x9E3779B97F4A7C15ULL;
    // Warm up RNG
    xorshift64star(&rng_state);

    block_q8_0* block = &blocks[block_idx];
    const float d = __half2float(block->d);

    // Handle zero scale (skip perturbation)
    if (fabsf(d) < 1e-10f) return;

    const int global_idx = block_idx * QK8_0 + tid;
    const float perturb = perturbation[global_idx];

    // Current dequantized value
    float val = d * (float)block->qs[tid];

    // Apply perturbation
    if (add) {
        val += epsilon * perturb;
    } else {
        val -= epsilon * perturb;
    }

    // Re-quantize with stochastic rounding
    float q_val = val / d;
    int8_t new_q = stochastic_round_i8(q_val, &rng_state);

    // Write back
    block->qs[tid] = new_q;
}

// ============================================================================
// Q4K Perturbation Kernel
// ============================================================================
// Q4K has 256 elements per block, packed as 4-bit values (128 bytes)
// Each block_q4_K contains:
//   - dm: half2 with super-block scale (dall) and min (dmin)
//   - scales[12]: sub-block scales and mins (6-bit packed)
//   - qs[128]: 4-bit quantized values (two per byte)
//
// Dequantization formula (from GGML):
//   value = dall * sc * q_nibble - dmin * m
// where sc and m are 6-bit sub-block scale and min from get_scale_min_k4()
//
// Re-quantization:
//   q_nibble = (value + dmin * m) / (dall * sc)
//
// We use 128 threads per CUDA block, one per byte (handles 2 elements)

// Helper to decode sub-block scale and min (matches GGML get_scale_min_k4)
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* scales, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

extern "C" __global__ void perturb_q4_K(
    block_q4_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float epsilon,
    const uint64_t seed,
    const int add
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= QK_K / 2) return;  // 128 threads, each handles 2 elements (one byte)

    // Initialize RNG
    uint64_t rng_state = seed ^ ((uint64_t)block_idx * QK_K + tid * 2) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    block_q4_K* block = &blocks[block_idx];

    // Extract super-block scale (dall) and min (dmin)
    const float dall = __low2float(block->dm);
    const float dmin = __high2float(block->dm);

    if (fabsf(dall) < 1e-10f) return;

    // Get packed byte containing two 4-bit values
    uint8_t packed = block->qs[tid];
    uint8_t q4_lo = packed & 0x0F;
    uint8_t q4_hi = (packed >> 4) & 0x0F;

    // Determine which sub-block this byte belongs to
    // Q4K has 8 sub-blocks of 32 elements each
    // Byte tid contains elements tid*2 and tid*2+1
    // Sub-block for low nibble: (tid * 2) / 32 = tid / 16
    // But the qs[] layout is: first 64 bytes contain low nibbles of all 256 elements
    // Actually looking at dequantize_block_q4_K more carefully...
    //
    // The layout is:
    // - Thread with il = tid/8 (0-3), ir = tid%8 (0-7)
    // - is = 2*il (sub-block pair index: 0, 2, 4, 6)
    // - q = qs + 32*il + n*ir where n=4
    // - y[l+0] uses scale from is+0, y[l+32] uses scale from is+1
    //
    // So each thread (in original 32-thread version) processes 4 low nibbles and 4 high nibbles
    // For 128-thread version where each thread handles 1 byte:
    // - Byte at index tid: low nibble element is at some position, high nibble at position+32

    // Let me match the GGML layout exactly:
    // In GGML, with 32 threads:
    //   il = tid/8 (0-3), ir = tid%8 (0-7), is = 2*il
    //   q = qs + 32*il + 4*ir (reads 4 consecutive bytes)
    //   For each of 4 values l in 0..3:
    //     y[64*il + 4*ir + l] = d1 * (q[l] & 0xF) - m1  (low nibble)
    //     y[64*il + 4*ir + l + 32] = d2 * (q[l] >> 4) - m2  (high nibble)
    //
    // With 128 threads where thread tid handles byte qs[tid]:
    //   il = tid / 32 (0-3), within_il = tid % 32
    //   is = 2 * il
    //   Low nibble element index: 64*il + within_il
    //   High nibble element index: 64*il + within_il + 32
    //   Low nibble uses sub-block scale is+0
    //   High nibble uses sub-block scale is+1

    const int il = tid / 32;  // 0-3
    const int within_il = tid % 32;
    const int is = 2 * il;  // sub-block pair: 0, 2, 4, 6

    // Element indices
    const int elem_lo = 64 * il + within_il;
    const int elem_hi = 64 * il + within_il + 32;

    // Get sub-block scales and mins
    uint8_t sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(is + 0, block->scales, sc_lo, m_lo);
    get_scale_min_k4(is + 1, block->scales, sc_hi, m_hi);

    const float d_lo = dall * (float)sc_lo;
    const float m_lo_f = dmin * (float)m_lo;
    const float d_hi = dall * (float)sc_hi;
    const float m_hi_f = dmin * (float)m_hi;

    // Process low nibble
    {
        // Dequantize: value = dall * sc * q - dmin * m
        float val = d_lo * (float)q4_lo - m_lo_f;

        const int global_idx = block_idx * QK_K + elem_lo;
        const float perturb = perturbation[global_idx];

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        // Re-quantize: q = (value + dmin * m) / (dall * sc)
        float q_float = (val + m_lo_f) / d_lo;
        q4_lo = stochastic_round_u4(q_float, &rng_state);
    }

    // Process high nibble
    {
        float val = d_hi * (float)q4_hi - m_hi_f;

        const int global_idx = block_idx * QK_K + elem_hi;
        const float perturb = perturbation[global_idx];

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m_hi_f) / d_hi;
        q4_hi = stochastic_round_u4(q_float, &rng_state);
    }

    // Pack both nibbles back into the byte
    block->qs[tid] = q4_lo | (q4_hi << 4);
}

// ============================================================================
// BF16 Perturbation Kernel
// ============================================================================
// BF16 is a simple floating point format without quantization blocks.
// Each element is 16 bits, so perturbation is just: w' = w ± ε * z
// Uses 256 threads per CUDA block for coalesced memory access.

extern "C" __global__ void perturb_bf16(
    __nv_bfloat16* __restrict__ data,
    const float* __restrict__ perturbation,
    const int num_elements,
    const float epsilon,
    const uint64_t seed,  // unused for BF16, but kept for API consistency
    const int add
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Load BF16 and convert to float
    float val = __bfloat162float(data[idx]);

    // Apply perturbation
    const float perturb = perturbation[idx];
    if (add) {
        val += epsilon * perturb;
    } else {
        val -= epsilon * perturb;
    }

    // Convert back to BF16 and store (truncation rounding)
    data[idx] = __float2bfloat16(val);
}

// ============================================================================
// Combined Restore + Update Kernel (BF16)
// ============================================================================
// Performs both restore from -ε to 0 AND applies update in one pass

extern "C" __global__ void restore_and_update_bf16(
    __nv_bfloat16* __restrict__ data,
    const float* __restrict__ perturbation,
    const int num_elements,
    const float restore_epsilon,
    const float update_scale,
    const uint64_t restore_seed,  // unused for BF16
    const uint64_t update_seed    // unused for BF16
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Load BF16 and convert to float
    float val = __bfloat162float(data[idx]);
    const float perturb = perturbation[idx];

    // Restore: add ε to go from -ε to 0
    val += restore_epsilon * perturb;

    // Apply update: subtract η*μ*z (gradient descent)
    val -= update_scale * perturb;

    // Convert back to BF16 and store
    data[idx] = __float2bfloat16(val);
}

// ============================================================================
// Combined Restore + Update Kernel (Q8_0)
// ============================================================================
// Performs both restore from -ε to 0 AND applies update in one pass
// This saves one full pass over the data

extern "C" __global__ void restore_and_update_q8_0(
    block_q8_0* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float restore_epsilon,   // ε to restore from -ε to 0
    const float update_scale,      // η * μ / n (learning rate * gradient * 1/num_samples)
    const uint64_t restore_seed,
    const uint64_t update_seed
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= QK8_0) return;

    block_q8_0* block = &blocks[block_idx];
    const float d = __half2float(block->d);

    if (fabsf(d) < 1e-10f) return;

    const int global_idx = block_idx * QK8_0 + tid;
    const float perturb = perturbation[global_idx];

    // Current value (at -ε perturbation)
    float val = d * (float)block->qs[tid];

    // Restore: add ε to go from -ε to 0
    // Use restore_seed for stochastic rounding consistency
    uint64_t rng_restore = restore_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_restore);
    val += restore_epsilon * perturb;

    // Apply update: subtract η*μ*z (gradient descent)
    // Use different seed for update
    uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_update);
    val -= update_scale * perturb;

    // Final re-quantization
    float q_val = val / d;
    int8_t new_q = stochastic_round_i8(q_val, &rng_update);

    block->qs[tid] = new_q;
}

// ============================================================================
// Combined Restore + Update Kernel (Q4K)
// ============================================================================
// Performs both restore from -ε to 0 AND applies update in one pass

extern "C" __global__ void restore_and_update_q4_K(
    block_q4_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float restore_epsilon,
    const float update_scale,
    const uint64_t restore_seed,
    const uint64_t update_seed
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= QK_K / 2) return;  // 128 threads

    block_q4_K* block = &blocks[block_idx];

    const float dall = __low2float(block->dm);
    const float dmin = __high2float(block->dm);

    if (fabsf(dall) < 1e-10f) return;

    uint8_t packed = block->qs[tid];
    uint8_t q4_lo = packed & 0x0F;
    uint8_t q4_hi = (packed >> 4) & 0x0F;

    const int il = tid / 32;
    const int within_il = tid % 32;
    const int is = 2 * il;

    const int elem_lo = 64 * il + within_il;
    const int elem_hi = 64 * il + within_il + 32;

    uint8_t sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(is + 0, block->scales, sc_lo, m_lo);
    get_scale_min_k4(is + 1, block->scales, sc_hi, m_hi);

    const float d_lo = dall * (float)sc_lo;
    const float m_lo_f = dmin * (float)m_lo;
    const float d_hi = dall * (float)sc_hi;
    const float m_hi_f = dmin * (float)m_hi;

    // Process low nibble
    {
        const int global_idx = block_idx * QK_K + elem_lo;
        const float perturb = perturbation[global_idx];

        // Initialize RNG for restore
        uint64_t rng_restore = restore_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_restore);

        // Dequantize current value (at -ε perturbation)
        float val = d_lo * (float)q4_lo - m_lo_f;

        // Restore: add ε to go from -ε to 0
        val += restore_epsilon * perturb;

        // Initialize RNG for update
        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        // Apply update
        val -= update_scale * perturb;

        // Re-quantize
        float q_float = (val + m_lo_f) / d_lo;
        q4_lo = stochastic_round_u4(q_float, &rng_update);
    }

    // Process high nibble
    {
        const int global_idx = block_idx * QK_K + elem_hi;
        const float perturb = perturbation[global_idx];

        uint64_t rng_restore = restore_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_restore);

        float val = d_hi * (float)q4_hi - m_hi_f;
        val += restore_epsilon * perturb;

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        val -= update_scale * perturb;

        float q_float = (val + m_hi_f) / d_hi;
        q4_hi = stochastic_round_u4(q_float, &rng_update);
    }

    block->qs[tid] = q4_lo | (q4_hi << 4);
}

// ============================================================================
// Q6K Perturbation Kernel
// ============================================================================
// Q6K has 256 elements per block
// Layout:
//   - ql[128]: lower 4 bits of 6-bit quants
//   - qh[64]: upper 2 bits of 6-bit quants
//   - scales[16]: signed 8-bit scales
//   - d: super-block scale
//
// Dequantization formula (from GGML):
//   value = d * scale * (q6 - 32)
// where q6 = (ql_nibble | ((qh_bits & 3) << 4))
//
// Thread layout: 64 threads per block
// Each thread handles 4 elements (matching GGML dequantize pattern)

extern "C" __global__ void perturb_q6_K(
    block_q6_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float epsilon,
    const uint64_t seed,
    const int add
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= 64) return;

    uint64_t rng_state = seed ^ ((uint64_t)block_idx * QK_K + tid * 4) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    block_q6_K* block = &blocks[block_idx];

    const float d = __half2float(block->d);
    if (fabsf(d) < 1e-10f) return;

    // Thread mapping: ip = tid/32 (0 or 1), il = tid - 32*ip (0..31)
    const int ip = tid / 32;
    const int il = tid - 32 * ip;
    const int is = 8 * ip + il / 16;

    // Each thread processes 4 elements: y[0], y[32], y[64], y[96]
    // Element offsets from block start: 128*ip + il + {0, 32, 64, 96}
    const int base_elem = 128 * ip + il;

    // Element 0: ql[ip*64+il] low nibble, qh[ip*32+il] bits 0-1
    {
        const int elem_idx = base_elem + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const int8_t sc = block->scales[is + 0];
        const uint8_t ql_byte = block->ql[64 * ip + il];
        const uint8_t qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte & 0xF) | (((qh_byte >> 0) & 3) << 4);
        const int q6_signed = q6_val - 32;

        float val = d * (float)sc * (float)q6_signed;
        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        // Re-quantize
        float q_float = val / (d * (float)sc) + 32.0f;
        int8_t q6_new = stochastic_round_i6(q_float - 32.0f, &rng_state) + 32;
        q6_new = max(0, min(63, (int)q6_new));

        // Update ql and qh
        block->ql[64 * ip + il] = (block->ql[64 * ip + il] & 0xF0) | (q6_new & 0x0F);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x03) | ((q6_new >> 4) & 0x03);
    }

    // Element 1: ql[ip*64+il+32] low nibble, qh[ip*32+il] bits 2-3
    {
        const int elem_idx = base_elem + 32;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const int8_t sc = block->scales[is + 2];
        const uint8_t ql_byte = block->ql[64 * ip + il + 32];
        const uint8_t qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte & 0xF) | (((qh_byte >> 2) & 3) << 4);
        const int q6_signed = q6_val - 32;

        float val = d * (float)sc * (float)q6_signed;
        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = val / (d * (float)sc) + 32.0f;
        int8_t q6_new = stochastic_round_i6(q_float - 32.0f, &rng_state) + 32;
        q6_new = max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il + 32] = (block->ql[64 * ip + il + 32] & 0xF0) | (q6_new & 0x0F);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x0C) | (((q6_new >> 4) & 0x03) << 2);
    }

    // Element 2: ql[ip*64+il] high nibble, qh[ip*32+il] bits 4-5
    {
        const int elem_idx = base_elem + 64;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const int8_t sc = block->scales[is + 4];
        const uint8_t ql_byte = block->ql[64 * ip + il];
        const uint8_t qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte >> 4) | (((qh_byte >> 4) & 3) << 4);
        const int q6_signed = q6_val - 32;

        float val = d * (float)sc * (float)q6_signed;
        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = val / (d * (float)sc) + 32.0f;
        int8_t q6_new = stochastic_round_i6(q_float - 32.0f, &rng_state) + 32;
        q6_new = max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il] = (block->ql[64 * ip + il] & 0x0F) | ((q6_new & 0x0F) << 4);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x30) | (((q6_new >> 4) & 0x03) << 4);
    }

    // Element 3: ql[ip*64+il+32] high nibble, qh[ip*32+il] bits 6-7
    {
        const int elem_idx = base_elem + 96;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const int8_t sc = block->scales[is + 6];
        const uint8_t ql_byte = block->ql[64 * ip + il + 32];
        const uint8_t qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte >> 4) | (((qh_byte >> 6) & 3) << 4);
        const int q6_signed = q6_val - 32;

        float val = d * (float)sc * (float)q6_signed;
        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = val / (d * (float)sc) + 32.0f;
        int8_t q6_new = stochastic_round_i6(q_float - 32.0f, &rng_state) + 32;
        q6_new = max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il + 32] = (block->ql[64 * ip + il + 32] & 0x0F) | ((q6_new & 0x0F) << 4);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0xC0) | (((q6_new >> 4) & 0x03) << 6);
    }
}

// ============================================================================
// Q6K Combined Restore + Update Kernel
// ============================================================================

extern "C" __global__ void restore_and_update_q6_K(
    block_q6_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float restore_epsilon,
    const float update_scale,
    const uint64_t restore_seed,
    const uint64_t update_seed
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= 64) return;

    block_q6_K* block = &blocks[block_idx];

    const float d = __half2float(block->d);
    if (fabsf(d) < 1e-10f) return;

    const int ip = tid / 32;
    const int il = tid - 32 * ip;
    const int is = 8 * ip + il / 16;
    const int base_elem = 128 * ip + il;

    // Element 0
    {
        const int elem_idx = base_elem + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const int8_t sc = block->scales[is + 0];
        const uint8_t ql_byte = block->ql[64 * ip + il];
        const uint8_t qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte & 0xF) | (((qh_byte >> 0) & 3) << 4);
        float val = d * (float)sc * (float)(q6_val - 32);

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / (d * (float)sc) + 32.0f;
        int8_t q6_new = stochastic_round_i6(q_float - 32.0f, &rng_update) + 32;
        q6_new = max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il] = (block->ql[64 * ip + il] & 0xF0) | (q6_new & 0x0F);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x03) | ((q6_new >> 4) & 0x03);
    }

    // Element 1
    {
        const int elem_idx = base_elem + 32;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const int8_t sc = block->scales[is + 2];
        const uint8_t ql_byte = block->ql[64 * ip + il + 32];
        const uint8_t qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte & 0xF) | (((qh_byte >> 2) & 3) << 4);
        float val = d * (float)sc * (float)(q6_val - 32);

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / (d * (float)sc) + 32.0f;
        int8_t q6_new = stochastic_round_i6(q_float - 32.0f, &rng_update) + 32;
        q6_new = max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il + 32] = (block->ql[64 * ip + il + 32] & 0xF0) | (q6_new & 0x0F);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x0C) | (((q6_new >> 4) & 0x03) << 2);
    }

    // Element 2
    {
        const int elem_idx = base_elem + 64;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const int8_t sc = block->scales[is + 4];
        const uint8_t ql_byte = block->ql[64 * ip + il];
        const uint8_t qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte >> 4) | (((qh_byte >> 4) & 3) << 4);
        float val = d * (float)sc * (float)(q6_val - 32);

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / (d * (float)sc) + 32.0f;
        int8_t q6_new = stochastic_round_i6(q_float - 32.0f, &rng_update) + 32;
        q6_new = max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il] = (block->ql[64 * ip + il] & 0x0F) | ((q6_new & 0x0F) << 4);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x30) | (((q6_new >> 4) & 0x03) << 4);
    }

    // Element 3
    {
        const int elem_idx = base_elem + 96;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const int8_t sc = block->scales[is + 6];
        const uint8_t ql_byte = block->ql[64 * ip + il + 32];
        const uint8_t qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte >> 4) | (((qh_byte >> 6) & 3) << 4);
        float val = d * (float)sc * (float)(q6_val - 32);

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / (d * (float)sc) + 32.0f;
        int8_t q6_new = stochastic_round_i6(q_float - 32.0f, &rng_update) + 32;
        q6_new = max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il + 32] = (block->ql[64 * ip + il + 32] & 0x0F) | ((q6_new & 0x0F) << 4);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0xC0) | (((q6_new >> 4) & 0x03) << 6);
    }
}

// ============================================================================
// Q5K Perturbation Kernel
// ============================================================================
// Q5K has 256 elements per block
// Layout:
//   - dm: half2 (dall, dmin)
//   - scales[12]: sub-block scales (6-bit packed)
//   - qh[32]: high bits (1 bit per element)
//   - qs[128]: low 4 bits (packed)
//
// Dequantization formula:
//   value = dall * sc * (q4 | (qh_bit << 4)) - dmin * m
//
// Thread layout: 64 threads, each handles 4 elements

extern "C" __global__ void perturb_q5_K(
    block_q5_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float epsilon,
    const uint64_t seed,
    const int add
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= 64) return;

    uint64_t rng_state = seed ^ ((uint64_t)block_idx * QK_K + tid * 4) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    block_q5_K* block = &blocks[block_idx];

    const float dall = __low2float(block->dm);
    const float dmin = __high2float(block->dm);

    if (fabsf(dall) < 1e-10f) return;

    // Thread mapping (from GGML dequantize_block_q5_K with 64 threads)
    const int il = tid / 16;   // 0..3
    const int ir = tid % 16;   // 0..15
    const int is = 2 * il;     // sub-block pair

    // Each thread processes elements at: 64*il + 2*ir + {0, 1, 32, 33}
    const int base_elem = 64 * il + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, block->scales, sc, m);
    const float d1 = dall * (float)sc;
    const float m1 = dmin * (float)m;
    get_scale_min_k4(is + 1, block->scales, sc, m);
    const float d2 = dall * (float)sc;
    const float m2 = dmin * (float)m;

    uint8_t hm = 1 << (2 * il);

    // Element 0: ql[32*il + 2*ir] low nibble, qh[2*ir] bit hm
    {
        const int elem_idx = base_elem + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const uint8_t ql_byte = block->qs[32 * il + 2 * ir];
        const uint8_t qh_byte = block->qh[2 * ir];
        int q5_val = (ql_byte & 0xF) + ((qh_byte & hm) ? 16 : 0);
        float val = d1 * (float)q5_val - m1;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m1) / d1;
        uint8_t q5_new = stochastic_round_u5(q_float, &rng_state);

        block->qs[32 * il + 2 * ir] = (block->qs[32 * il + 2 * ir] & 0xF0) | (q5_new & 0x0F);
        if (q5_new >= 16) {
            block->qh[2 * ir] |= hm;
        } else {
            block->qh[2 * ir] &= ~hm;
        }
    }

    // Element 1: ql[32*il + 2*ir + 1] low nibble, qh[2*ir + 1] bit hm
    {
        const int elem_idx = base_elem + 1;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const uint8_t ql_byte = block->qs[32 * il + 2 * ir + 1];
        const uint8_t qh_byte = block->qh[2 * ir + 1];
        int q5_val = (ql_byte & 0xF) + ((qh_byte & hm) ? 16 : 0);
        float val = d1 * (float)q5_val - m1;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m1) / d1;
        uint8_t q5_new = stochastic_round_u5(q_float, &rng_state);

        block->qs[32 * il + 2 * ir + 1] = (block->qs[32 * il + 2 * ir + 1] & 0xF0) | (q5_new & 0x0F);
        if (q5_new >= 16) {
            block->qh[2 * ir + 1] |= hm;
        } else {
            block->qh[2 * ir + 1] &= ~hm;
        }
    }

    hm <<= 1;  // Next bit for high nibble elements

    // Element 2: ql[32*il + 2*ir] high nibble, qh[2*ir] bit hm (shifted)
    {
        const int elem_idx = base_elem + 32;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const uint8_t ql_byte = block->qs[32 * il + 2 * ir];
        const uint8_t qh_byte = block->qh[2 * ir];
        int q5_val = (ql_byte >> 4) + ((qh_byte & hm) ? 16 : 0);
        float val = d2 * (float)q5_val - m2;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m2) / d2;
        uint8_t q5_new = stochastic_round_u5(q_float, &rng_state);

        block->qs[32 * il + 2 * ir] = (block->qs[32 * il + 2 * ir] & 0x0F) | ((q5_new & 0x0F) << 4);
        if (q5_new >= 16) {
            block->qh[2 * ir] |= hm;
        } else {
            block->qh[2 * ir] &= ~hm;
        }
    }

    // Element 3: ql[32*il + 2*ir + 1] high nibble, qh[2*ir + 1] bit hm
    {
        const int elem_idx = base_elem + 33;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const uint8_t ql_byte = block->qs[32 * il + 2 * ir + 1];
        const uint8_t qh_byte = block->qh[2 * ir + 1];
        int q5_val = (ql_byte >> 4) + ((qh_byte & hm) ? 16 : 0);
        float val = d2 * (float)q5_val - m2;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m2) / d2;
        uint8_t q5_new = stochastic_round_u5(q_float, &rng_state);

        block->qs[32 * il + 2 * ir + 1] = (block->qs[32 * il + 2 * ir + 1] & 0x0F) | ((q5_new & 0x0F) << 4);
        if (q5_new >= 16) {
            block->qh[2 * ir + 1] |= hm;
        } else {
            block->qh[2 * ir + 1] &= ~hm;
        }
    }
}

// ============================================================================
// Q5K Combined Restore + Update Kernel
// ============================================================================

extern "C" __global__ void restore_and_update_q5_K(
    block_q5_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float restore_epsilon,
    const float update_scale,
    const uint64_t restore_seed,
    const uint64_t update_seed
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= 64) return;

    block_q5_K* block = &blocks[block_idx];

    const float dall = __low2float(block->dm);
    const float dmin = __high2float(block->dm);

    if (fabsf(dall) < 1e-10f) return;

    const int il = tid / 16;
    const int ir = tid % 16;
    const int is = 2 * il;
    const int base_elem = 64 * il + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, block->scales, sc, m);
    const float d1 = dall * (float)sc;
    const float m1 = dmin * (float)m;
    get_scale_min_k4(is + 1, block->scales, sc, m);
    const float d2 = dall * (float)sc;
    const float m2 = dmin * (float)m;

    uint8_t hm = 1 << (2 * il);

    // Element 0
    {
        const int elem_idx = base_elem + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const uint8_t ql_byte = block->qs[32 * il + 2 * ir];
        const uint8_t qh_byte = block->qh[2 * ir];
        int q5_val = (ql_byte & 0xF) + ((qh_byte & hm) ? 16 : 0);
        float val = d1 * (float)q5_val - m1;

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + m1) / d1;
        uint8_t q5_new = stochastic_round_u5(q_float, &rng_update);

        block->qs[32 * il + 2 * ir] = (block->qs[32 * il + 2 * ir] & 0xF0) | (q5_new & 0x0F);
        if (q5_new >= 16) {
            block->qh[2 * ir] |= hm;
        } else {
            block->qh[2 * ir] &= ~hm;
        }
    }

    // Element 1
    {
        const int elem_idx = base_elem + 1;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const uint8_t ql_byte = block->qs[32 * il + 2 * ir + 1];
        const uint8_t qh_byte = block->qh[2 * ir + 1];
        int q5_val = (ql_byte & 0xF) + ((qh_byte & hm) ? 16 : 0);
        float val = d1 * (float)q5_val - m1;

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + m1) / d1;
        uint8_t q5_new = stochastic_round_u5(q_float, &rng_update);

        block->qs[32 * il + 2 * ir + 1] = (block->qs[32 * il + 2 * ir + 1] & 0xF0) | (q5_new & 0x0F);
        if (q5_new >= 16) {
            block->qh[2 * ir + 1] |= hm;
        } else {
            block->qh[2 * ir + 1] &= ~hm;
        }
    }

    hm <<= 1;

    // Element 2
    {
        const int elem_idx = base_elem + 32;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const uint8_t ql_byte = block->qs[32 * il + 2 * ir];
        const uint8_t qh_byte = block->qh[2 * ir];
        int q5_val = (ql_byte >> 4) + ((qh_byte & hm) ? 16 : 0);
        float val = d2 * (float)q5_val - m2;

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + m2) / d2;
        uint8_t q5_new = stochastic_round_u5(q_float, &rng_update);

        block->qs[32 * il + 2 * ir] = (block->qs[32 * il + 2 * ir] & 0x0F) | ((q5_new & 0x0F) << 4);
        if (q5_new >= 16) {
            block->qh[2 * ir] |= hm;
        } else {
            block->qh[2 * ir] &= ~hm;
        }
    }

    // Element 3
    {
        const int elem_idx = base_elem + 33;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const uint8_t ql_byte = block->qs[32 * il + 2 * ir + 1];
        const uint8_t qh_byte = block->qh[2 * ir + 1];
        int q5_val = (ql_byte >> 4) + ((qh_byte & hm) ? 16 : 0);
        float val = d2 * (float)q5_val - m2;

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + m2) / d2;
        uint8_t q5_new = stochastic_round_u5(q_float, &rng_update);

        block->qs[32 * il + 2 * ir + 1] = (block->qs[32 * il + 2 * ir + 1] & 0x0F) | ((q5_new & 0x0F) << 4);
        if (q5_new >= 16) {
            block->qh[2 * ir + 1] |= hm;
        } else {
            block->qh[2 * ir + 1] &= ~hm;
        }
    }
}

// ============================================================================
// Q2K Perturbation Kernel
// ============================================================================
// Q2K has 256 elements per block
// Layout:
//   - scales[16]: 4-bit scale and 4-bit min per 16-element group
//   - qs[64]: 2-bit quants (4 per byte)
//   - dm: half2 (dall, dmin)
//
// Dequantization formula:
//   value = dall * (scales[j] & 0xF) * q2 - dmin * (scales[j] >> 4)
//
// Thread layout: 64 threads, each handles 4 elements (matching GGML)

extern "C" __global__ void perturb_q2_K(
    block_q2_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float epsilon,
    const uint64_t seed,
    const int add
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= 64) return;

    uint64_t rng_state = seed ^ ((uint64_t)block_idx * QK_K + tid * 4) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    block_q2_K* block = &blocks[block_idx];

    const float dall = __low2float(block->dm);
    const float dmin = __high2float(block->dm);

    if (fabsf(dall) < 1e-10f) return;

    // Thread mapping from GGML dequantize_block_q2_K (QK_K=256, 64 threads)
    const int n = tid / 32;       // 0 or 1
    const int l = tid - 32 * n;   // 0..31
    const int is = 8 * n + l / 16;

    // Byte index in qs
    const int qs_idx = 32 * n + l;
    uint8_t q_byte = block->qs[qs_idx];

    // Each byte contains 4 2-bit values
    // y[l+0], y[l+32], y[l+64], y[l+96] use scales is+0, is+2, is+4, is+6
    const int base_elem = 128 * n;

    // Extract 2-bit values
    uint8_t q0 = (q_byte >> 0) & 3;
    uint8_t q1 = (q_byte >> 2) & 3;
    uint8_t q2 = (q_byte >> 4) & 3;
    uint8_t q3 = (q_byte >> 6) & 3;

    // Element 0: uses scales[is+0]
    {
        const int elem_idx = base_elem + l + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const float d_sc = dall * (float)(block->scales[is + 0] & 0xF);
        const float d_min = dmin * (float)(block->scales[is + 0] >> 4);

        float val = d_sc * (float)q0 - d_min;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + d_min) / d_sc;
        q0 = stochastic_round_u2(q_float, &rng_state);
    }

    // Element 1: uses scales[is+2]
    {
        const int elem_idx = base_elem + l + 32;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const float d_sc = dall * (float)(block->scales[is + 2] & 0xF);
        const float d_min = dmin * (float)(block->scales[is + 2] >> 4);

        float val = d_sc * (float)q1 - d_min;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + d_min) / d_sc;
        q1 = stochastic_round_u2(q_float, &rng_state);
    }

    // Element 2: uses scales[is+4]
    {
        const int elem_idx = base_elem + l + 64;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const float d_sc = dall * (float)(block->scales[is + 4] & 0xF);
        const float d_min = dmin * (float)(block->scales[is + 4] >> 4);

        float val = d_sc * (float)q2 - d_min;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + d_min) / d_sc;
        q2 = stochastic_round_u2(q_float, &rng_state);
    }

    // Element 3: uses scales[is+6]
    {
        const int elem_idx = base_elem + l + 96;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const float d_sc = dall * (float)(block->scales[is + 6] & 0xF);
        const float d_min = dmin * (float)(block->scales[is + 6] >> 4);

        float val = d_sc * (float)q3 - d_min;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + d_min) / d_sc;
        q3 = stochastic_round_u2(q_float, &rng_state);
    }

    // Pack back into byte
    block->qs[qs_idx] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
}

// ============================================================================
// Q2K Combined Restore + Update Kernel
// ============================================================================

extern "C" __global__ void restore_and_update_q2_K(
    block_q2_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float restore_epsilon,
    const float update_scale,
    const uint64_t restore_seed,
    const uint64_t update_seed
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= 64) return;

    block_q2_K* block = &blocks[block_idx];

    const float dall = __low2float(block->dm);
    const float dmin = __high2float(block->dm);

    if (fabsf(dall) < 1e-10f) return;

    const int n = tid / 32;
    const int l = tid - 32 * n;
    const int is = 8 * n + l / 16;
    const int qs_idx = 32 * n + l;
    uint8_t q_byte = block->qs[qs_idx];
    const int base_elem = 128 * n;

    uint8_t q0 = (q_byte >> 0) & 3;
    uint8_t q1 = (q_byte >> 2) & 3;
    uint8_t q2 = (q_byte >> 4) & 3;
    uint8_t q3 = (q_byte >> 6) & 3;

    // Element 0
    {
        const int elem_idx = base_elem + l + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const float d_sc = dall * (float)(block->scales[is + 0] & 0xF);
        const float d_min = dmin * (float)(block->scales[is + 0] >> 4);

        float val = d_sc * (float)q0 - d_min;
        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + d_min) / d_sc;
        q0 = stochastic_round_u2(q_float, &rng_update);
    }

    // Element 1
    {
        const int elem_idx = base_elem + l + 32;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const float d_sc = dall * (float)(block->scales[is + 2] & 0xF);
        const float d_min = dmin * (float)(block->scales[is + 2] >> 4);

        float val = d_sc * (float)q1 - d_min;
        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + d_min) / d_sc;
        q1 = stochastic_round_u2(q_float, &rng_update);
    }

    // Element 2
    {
        const int elem_idx = base_elem + l + 64;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const float d_sc = dall * (float)(block->scales[is + 4] & 0xF);
        const float d_min = dmin * (float)(block->scales[is + 4] >> 4);

        float val = d_sc * (float)q2 - d_min;
        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + d_min) / d_sc;
        q2 = stochastic_round_u2(q_float, &rng_update);
    }

    // Element 3
    {
        const int elem_idx = base_elem + l + 96;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const float d_sc = dall * (float)(block->scales[is + 6] & 0xF);
        const float d_min = dmin * (float)(block->scales[is + 6] >> 4);

        float val = d_sc * (float)q3 - d_min;
        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + d_min) / d_sc;
        q3 = stochastic_round_u2(q_float, &rng_update);
    }

    block->qs[qs_idx] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
}

// ============================================================================
// Q3K Helper: Get scale from 6-bit packed format
// ============================================================================

__device__ __forceinline__ int8_t get_q3k_scale(int is, const uint8_t* scales) {
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
    return us - 32;  // Q3K scales are centered at 32
}

// ============================================================================
// Q3K Perturbation Kernel
// ============================================================================
// Q3K has 256 elements per block
// Layout:
//   - hmask[32]: high bits of 3-bit quants
//   - qs[64]: low 2 bits of 3-bit quants
//   - scales[12]: 6-bit packed scales
//   - d: super-block scale
//
// Dequantization formula (complex due to hmask):
//   q3 = ((qs >> shift) & 3) - (hmask_bit ? 0 : 4)
//   value = d * scale * q3
// where scale = us - 32 (6-bit packed scale centered at 32)
//
// Thread layout: 64 threads

extern "C" __global__ void perturb_q3_K(
    block_q3_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float epsilon,
    const uint64_t seed,
    const int add
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= 64) return;

    uint64_t rng_state = seed ^ ((uint64_t)block_idx * QK_K + tid * 4) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    block_q3_K* block = &blocks[block_idx];

    const float d_all = __half2float(block->d);
    if (fabsf(d_all) < 1e-10f) return;

    // Thread mapping (from GGML dequantize_block_q3_K with 64 threads)
    const int r = tid / 4;
    const int tid_mod = tid % 4;
    const int is0 = r % 2;
    const int l0 = 16 * is0 + 4 * tid_mod;
    const int n = r / 4;
    const int j = (r / 2) - 2 * n;

    const uint8_t m = 1 << (4 * n + j);
    const int is = 8 * n + 2 * j + is0;
    const int shift = 2 * j;

    const int8_t scale = get_q3k_scale(is, block->scales);
    const float dl = d_all * (float)scale;

    // Pointer to qs for this group
    const uint8_t* q = block->qs + 32 * n;
    const uint8_t* hm = block->hmask;

    // Process 4 elements: l0, l0+1, l0+2, l0+3
    for (int l_off = 0; l_off < 4; l_off++) {
        const int l = l0 + l_off;
        const int elem_idx = 128 * n + 32 * j + l;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        // Extract 3-bit value
        int q2_val = (q[l] >> shift) & 3;
        int q3_val = q2_val - ((hm[l] & m) ? 0 : 4);

        float val = dl * (float)q3_val;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        // Re-quantize
        float q_float = val / dl;
        int8_t q3_new = stochastic_round_i3(q_float, &rng_state);

        // Pack back: low 2 bits to qs, high bit to hmask
        int q2_new = (q3_new >= 0) ? q3_new : (q3_new + 4);
        bool h_bit = (q3_new >= 0);

        // Update qs: clear and set the 2-bit field
        uint8_t qs_byte = block->qs[32 * n + l];
        qs_byte &= ~(3 << shift);
        qs_byte |= (q2_new & 3) << shift;
        block->qs[32 * n + l] = qs_byte;

        // Update hmask
        if (h_bit) {
            block->hmask[l] |= m;
        } else {
            block->hmask[l] &= ~m;
        }
    }
}

// ============================================================================
// Q3K Combined Restore + Update Kernel
// ============================================================================

extern "C" __global__ void restore_and_update_q3_K(
    block_q3_K* __restrict__ blocks,
    const float* __restrict__ perturbation,
    const int num_blocks,
    const float restore_epsilon,
    const float update_scale,
    const uint64_t restore_seed,
    const uint64_t update_seed
) {
    const int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    const int tid = threadIdx.x;
    if (tid >= 64) return;

    block_q3_K* block = &blocks[block_idx];

    const float d_all = __half2float(block->d);
    if (fabsf(d_all) < 1e-10f) return;

    const int r = tid / 4;
    const int tid_mod = tid % 4;
    const int is0 = r % 2;
    const int l0 = 16 * is0 + 4 * tid_mod;
    const int n = r / 4;
    const int j = (r / 2) - 2 * n;

    const uint8_t m = 1 << (4 * n + j);
    const int is = 8 * n + 2 * j + is0;
    const int shift = 2 * j;

    const int8_t scale = get_q3k_scale(is, block->scales);
    const float dl = d_all * (float)scale;

    const uint8_t* q = block->qs + 32 * n;
    const uint8_t* hm = block->hmask;

    for (int l_off = 0; l_off < 4; l_off++) {
        const int l = l0 + l_off;
        const int elem_idx = 128 * n + 32 * j + l;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        uint64_t rng_update = update_seed ^ ((uint64_t)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        int q2_val = (q[l] >> shift) & 3;
        int q3_val = q2_val - ((hm[l] & m) ? 0 : 4);

        float val = dl * (float)q3_val;
        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / dl;
        int8_t q3_new = stochastic_round_i3(q_float, &rng_update);

        int q2_new = (q3_new >= 0) ? q3_new : (q3_new + 4);
        bool h_bit = (q3_new >= 0);

        uint8_t qs_byte = block->qs[32 * n + l];
        qs_byte &= ~(3 << shift);
        qs_byte |= (q2_new & 3) << shift;
        block->qs[32 * n + l] = qs_byte;

        if (h_bit) {
            block->hmask[l] |= m;
        } else {
            block->hmask[l] &= ~m;
        }
    }
}

