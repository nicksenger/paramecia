// QuZO Perturbation Kernels (Metal Shading Language)
// Metal port of CUDA kernels for in-place perturbation of quantized tensors
// using stochastic rounding.
//
// These kernels implement the core QuZO operation:
//   w' = Q(w + epsilon * z)
// where Q is stochastic rounding to the quantized format.

#include <metal_stdlib>
using namespace metal;

// Block sizes for GGML quantization formats
#define QK8_0 32
#define QK_K 256

// ============================================================================
// Block Struct Definitions (matching GGML layout)
// ============================================================================

struct block_q8_0 {
    half d;
    char qs[QK8_0];
};

struct block_q4_K {
    half2 dm;
    uchar scales[12];
    uchar qs[QK_K/2];
};

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half2 dm;
};

struct block_q3_K {
    uchar hmask[QK_K/8];
    uchar qs[QK_K/4];
    uchar scales[12];
    half d;
};

struct block_q5_K {
    half2 dm;
    uchar scales[12];
    uchar qh[QK_K/8];
    uchar qs[QK_K/2];
};

struct block_q6_K {
    uchar ql[QK_K/2];
    uchar qh[QK_K/4];
    char scales[QK_K/16];
    half d;
};

// ============================================================================
// Random Number Generation (xorshift64* for stochastic rounding)
// ============================================================================

// Fast GPU-friendly RNG based on xorshift64*
// Each thread gets a unique sequence based on global_seed + thread_id
inline ulong xorshift64star(thread ulong* state) {
    ulong x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// Returns uniform float in [0, 1)
inline float rand_uniform(thread ulong* state) {
    ulong r = xorshift64star(state);
    // Use upper 23 bits for mantissa (float precision)
    return (r >> 40) * (1.0f / 16777216.0f);  // 2^-24
}

// Stochastic rounding: round to nearest with probability proportional to fractional part
inline char stochastic_round_i8(float x, thread ulong* rng_state) {
    // Clamp to int8 range
    x = max(-127.0f, min(127.0f, x));

    float floor_x = floor(x);
    float frac = x - floor_x;

    // Round up with probability = frac, down with probability = 1-frac
    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    // Final clamp to int8
    return (char)max(-127, min(127, result));
}

// Stochastic rounding for 4-bit values (0-15 range)
inline uchar stochastic_round_u4(float x, thread ulong* rng_state) {
    x = max(0.0f, min(15.0f, x));

    float floor_x = floor(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (uchar)max(0, min(15, result));
}

// Stochastic rounding for 2-bit values (0-3 range)
inline uchar stochastic_round_u2(float x, thread ulong* rng_state) {
    x = max(0.0f, min(3.0f, x));

    float floor_x = floor(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (uchar)max(0, min(3, result));
}

// Stochastic rounding for 3-bit signed values (-4 to 3 range)
inline char stochastic_round_i3(float x, thread ulong* rng_state) {
    x = max(-4.0f, min(3.0f, x));

    float floor_x = floor(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (char)max(-4, min(3, result));
}

// Stochastic rounding for 5-bit values (0-31 range)
inline uchar stochastic_round_u5(float x, thread ulong* rng_state) {
    x = max(0.0f, min(31.0f, x));

    float floor_x = floor(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (uchar)max(0, min(31, result));
}

// Stochastic rounding for 6-bit signed values (-32 to 31 range)
inline char stochastic_round_i6(float x, thread ulong* rng_state) {
    x = max(-32.0f, min(31.0f, x));

    float floor_x = floor(x);
    float frac = x - floor_x;

    float r = rand_uniform(rng_state);
    int result = (r < frac) ? (int)(floor_x + 1) : (int)floor_x;

    return (char)max(-32, min(31, result));
}

// ============================================================================
// Helper Functions
// ============================================================================

// Helper to decode sub-block scale and min (matches GGML get_scale_min_k4)
inline void get_scale_min_k4(int j, const device uchar* scales, thread uchar& sc, thread uchar& m) {
    if (j < 4) {
        sc = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

// Q3K helper: Get scale from 6-bit packed format
inline char get_q3k_scale(int is, const device uchar* scales) {
    char us;
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
// Q8_0 Perturbation Kernel
// ============================================================================
// Each threadgroup handles one Q8_0 block (32 elements)
// threads_per_threadgroup = 32 (one thread per element)

kernel void perturb_q8_0(
    device block_q8_0* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    constant int& add [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= QK8_0) return;

    // Initialize RNG state unique to this element
    ulong rng_state = seed ^ ((ulong)block_idx * QK8_0 + tid) * 0x9E3779B97F4A7C15ULL;
    // Warm up RNG
    xorshift64star(&rng_state);

    device block_q8_0* block = &blocks[block_idx];
    const float d = float(block->d);

    // Handle zero scale (skip perturbation)
    if (abs(d) < 1e-10f) return;

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
    char new_q = stochastic_round_i8(q_val, &rng_state);

    // Write back
    block->qs[tid] = new_q;
}

// ============================================================================
// Q4K Perturbation Kernel
// ============================================================================
// Q4K has 256 elements per block, packed as 4-bit values (128 bytes)
// We use 128 threads per threadgroup, one per byte (handles 2 elements)

kernel void perturb_q4_K(
    device block_q4_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    constant int& add [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= QK_K / 2) return;  // 128 threads, each handles 2 elements (one byte)

    // Initialize RNG
    ulong rng_state = seed ^ ((ulong)block_idx * QK_K + tid * 2) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    device block_q4_K* block = &blocks[block_idx];

    // Extract super-block scale (dall) and min (dmin)
    const float dall = float(block->dm.x);
    const float dmin = float(block->dm.y);

    if (abs(dall) < 1e-10f) return;

    // Get packed byte containing two 4-bit values
    uchar packed = block->qs[tid];
    uchar q4_lo = packed & 0x0F;
    uchar q4_hi = (packed >> 4) & 0x0F;

    const int il = tid / 32;  // 0-3
    const int within_il = tid % 32;
    const int is = 2 * il;  // sub-block pair: 0, 2, 4, 6

    // Element indices
    const int elem_lo = 64 * il + within_il;
    const int elem_hi = 64 * il + within_il + 32;

    // Get sub-block scales and mins
    uchar sc_lo, m_lo, sc_hi, m_hi;
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
// Q2K Perturbation Kernel
// ============================================================================
// Q2K has 256 elements per block
// Thread layout: 64 threads, each handles 4 elements

kernel void perturb_q2_K(
    device block_q2_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    constant int& add [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= 64) return;

    ulong rng_state = seed ^ ((ulong)block_idx * QK_K + tid * 4) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    device block_q2_K* block = &blocks[block_idx];

    const float dall = float(block->dm.x);
    const float dmin = float(block->dm.y);

    if (abs(dall) < 1e-10f) return;

    // Thread mapping from GGML dequantize_block_q2_K (QK_K=256, 64 threads)
    const int n = tid / 32;       // 0 or 1
    const int l = tid - 32 * n;   // 0..31
    const int is = 8 * n + l / 16;

    // Byte index in qs
    const int qs_idx = 32 * n + l;
    uchar q_byte = block->qs[qs_idx];

    // Each byte contains 4 2-bit values
    const int base_elem = 128 * n;

    // Extract 2-bit values
    uchar q0 = (q_byte >> 0) & 3;
    uchar q1 = (q_byte >> 2) & 3;
    uchar q2 = (q_byte >> 4) & 3;
    uchar q3 = (q_byte >> 6) & 3;

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
// Q3K Perturbation Kernel
// ============================================================================
// Q3K has 256 elements per block
// Thread layout: 64 threads

kernel void perturb_q3_K(
    device block_q3_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    constant int& add [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= 64) return;

    ulong rng_state = seed ^ ((ulong)block_idx * QK_K + tid * 4) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    device block_q3_K* block = &blocks[block_idx];

    const float d_all = float(block->d);
    if (abs(d_all) < 1e-10f) return;

    // Thread mapping (from GGML dequantize_block_q3_K with 64 threads)
    const int r = tid / 4;
    const int tid_mod = tid % 4;
    const int is0 = r % 2;
    const int l0 = 16 * is0 + 4 * tid_mod;
    const int n = r / 4;
    const int j = (r / 2) - 2 * n;

    const uchar m = 1 << (4 * n + j);
    const int is = 8 * n + 2 * j + is0;
    const int shift = 2 * j;

    const char scale = get_q3k_scale(is, block->scales);
    const float dl = d_all * (float)scale;

    // Pointer to qs for this group
    device const uchar* q = block->qs + 32 * n;
    device const uchar* hm = block->hmask;

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
        char q3_new = stochastic_round_i3(q_float, &rng_state);

        // Pack back: low 2 bits to qs, high bit to hmask
        int q2_new = (q3_new >= 0) ? q3_new : (q3_new + 4);
        bool h_bit = (q3_new >= 0);

        // Update qs: clear and set the 2-bit field
        uchar qs_byte = block->qs[32 * n + l];
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
// Q5K Perturbation Kernel
// ============================================================================
// Q5K has 256 elements per block
// Thread layout: 64 threads, each handles 4 elements

kernel void perturb_q5_K(
    device block_q5_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    constant int& add [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= 64) return;

    ulong rng_state = seed ^ ((ulong)block_idx * QK_K + tid * 4) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    device block_q5_K* block = &blocks[block_idx];

    const float dall = float(block->dm.x);
    const float dmin = float(block->dm.y);

    if (abs(dall) < 1e-10f) return;

    // Thread mapping (from GGML dequantize_block_q5_K with 64 threads)
    const int il = tid / 16;   // 0..3
    const int ir = tid % 16;   // 0..15
    const int is = 2 * il;     // sub-block pair

    // Each thread processes elements at: 64*il + 2*ir + {0, 1, 32, 33}
    const int base_elem = 64 * il + 2 * ir;

    uchar sc, m_val;
    get_scale_min_k4(is + 0, block->scales, sc, m_val);
    const float d1 = dall * (float)sc;
    const float m1 = dmin * (float)m_val;
    get_scale_min_k4(is + 1, block->scales, sc, m_val);
    const float d2 = dall * (float)sc;
    const float m2 = dmin * (float)m_val;

    uchar hm = 1 << (2 * il);

    // Element 0: ql[32*il + 2*ir] low nibble, qh[2*ir] bit hm
    {
        const int elem_idx = base_elem + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const uchar ql_byte = block->qs[32 * il + 2 * ir];
        const uchar qh_byte = block->qh[2 * ir];
        int q5_val = (ql_byte & 0xF) + ((qh_byte & hm) ? 16 : 0);
        float val = d1 * (float)q5_val - m1;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m1) / d1;
        uchar q5_new = stochastic_round_u5(q_float, &rng_state);

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

        const uchar ql_byte = block->qs[32 * il + 2 * ir + 1];
        const uchar qh_byte = block->qh[2 * ir + 1];
        int q5_val = (ql_byte & 0xF) + ((qh_byte & hm) ? 16 : 0);
        float val = d1 * (float)q5_val - m1;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m1) / d1;
        uchar q5_new = stochastic_round_u5(q_float, &rng_state);

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

        const uchar ql_byte = block->qs[32 * il + 2 * ir];
        const uchar qh_byte = block->qh[2 * ir];
        int q5_val = (ql_byte >> 4) + ((qh_byte & hm) ? 16 : 0);
        float val = d2 * (float)q5_val - m2;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m2) / d2;
        uchar q5_new = stochastic_round_u5(q_float, &rng_state);

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

        const uchar ql_byte = block->qs[32 * il + 2 * ir + 1];
        const uchar qh_byte = block->qh[2 * ir + 1];
        int q5_val = (ql_byte >> 4) + ((qh_byte & hm) ? 16 : 0);
        float val = d2 * (float)q5_val - m2;

        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = (val + m2) / d2;
        uchar q5_new = stochastic_round_u5(q_float, &rng_state);

        block->qs[32 * il + 2 * ir + 1] = (block->qs[32 * il + 2 * ir + 1] & 0x0F) | ((q5_new & 0x0F) << 4);
        if (q5_new >= 16) {
            block->qh[2 * ir + 1] |= hm;
        } else {
            block->qh[2 * ir + 1] &= ~hm;
        }
    }
}

// ============================================================================
// Q6K Perturbation Kernel
// ============================================================================
// Q6K has 256 elements per block
// Thread layout: 64 threads per block
// Each thread handles 4 elements (matching GGML dequantize pattern)

kernel void perturb_q6_K(
    device block_q6_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    constant int& add [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= 64) return;

    ulong rng_state = seed ^ ((ulong)block_idx * QK_K + tid * 4) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_state);

    device block_q6_K* block = &blocks[block_idx];

    const float d = float(block->d);
    if (abs(d) < 1e-10f) return;

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

        const char sc = block->scales[is + 0];
        const uchar ql_byte = block->ql[64 * ip + il];
        const uchar qh_byte = block->qh[32 * ip + il];
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
        char q6_new = stochastic_round_i6(q_float - 32.0f, &rng_state) + 32;
        q6_new = (char)max(0, min(63, (int)q6_new));

        // Update ql and qh
        block->ql[64 * ip + il] = (block->ql[64 * ip + il] & 0xF0) | (q6_new & 0x0F);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x03) | ((q6_new >> 4) & 0x03);
    }

    // Element 1: ql[ip*64+il+32] low nibble, qh[ip*32+il] bits 2-3
    {
        const int elem_idx = base_elem + 32;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const char sc = block->scales[is + 2];
        const uchar ql_byte = block->ql[64 * ip + il + 32];
        const uchar qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte & 0xF) | (((qh_byte >> 2) & 3) << 4);
        const int q6_signed = q6_val - 32;

        float val = d * (float)sc * (float)q6_signed;
        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = val / (d * (float)sc) + 32.0f;
        char q6_new = stochastic_round_i6(q_float - 32.0f, &rng_state) + 32;
        q6_new = (char)max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il + 32] = (block->ql[64 * ip + il + 32] & 0xF0) | (q6_new & 0x0F);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x0C) | (((q6_new >> 4) & 0x03) << 2);
    }

    // Element 2: ql[ip*64+il] high nibble, qh[ip*32+il] bits 4-5
    {
        const int elem_idx = base_elem + 64;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const char sc = block->scales[is + 4];
        const uchar ql_byte = block->ql[64 * ip + il];
        const uchar qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte >> 4) | (((qh_byte >> 4) & 3) << 4);
        const int q6_signed = q6_val - 32;

        float val = d * (float)sc * (float)q6_signed;
        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = val / (d * (float)sc) + 32.0f;
        char q6_new = stochastic_round_i6(q_float - 32.0f, &rng_state) + 32;
        q6_new = (char)max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il] = (block->ql[64 * ip + il] & 0x0F) | ((q6_new & 0x0F) << 4);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x30) | (((q6_new >> 4) & 0x03) << 4);
    }

    // Element 3: ql[ip*64+il+32] high nibble, qh[ip*32+il] bits 6-7
    {
        const int elem_idx = base_elem + 96;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        const char sc = block->scales[is + 6];
        const uchar ql_byte = block->ql[64 * ip + il + 32];
        const uchar qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte >> 4) | (((qh_byte >> 6) & 3) << 4);
        const int q6_signed = q6_val - 32;

        float val = d * (float)sc * (float)q6_signed;
        if (add) {
            val += epsilon * perturb;
        } else {
            val -= epsilon * perturb;
        }

        float q_float = val / (d * (float)sc) + 32.0f;
        char q6_new = stochastic_round_i6(q_float - 32.0f, &rng_state) + 32;
        q6_new = (char)max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il + 32] = (block->ql[64 * ip + il + 32] & 0x0F) | ((q6_new & 0x0F) << 4);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0xC0) | (((q6_new >> 4) & 0x03) << 6);
    }
}

// ============================================================================
// BF16 Perturbation Kernel
// ============================================================================
// BF16 is a simple floating point format without quantization blocks.
// Each element is 16 bits, so perturbation is just: w' = w +/- epsilon * z
// Uses 256 threads per threadgroup for coalesced memory access.

kernel void perturb_bf16(
    device bfloat* data [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_elements [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant ulong& seed [[buffer(4)]],
    constant int& add [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    const int idx = tgid * tgs + tid;
    if (idx >= num_elements) return;

    // Load BF16 and convert to float
    float val = float(data[idx]);

    // Apply perturbation
    const float perturb = perturbation[idx];
    if (add) {
        val += epsilon * perturb;
    } else {
        val -= epsilon * perturb;
    }

    // Convert back to BF16 and store (truncation rounding)
    data[idx] = bfloat(val);
}

// ============================================================================
// Combined Restore + Update Kernel (BF16)
// ============================================================================
// Performs both restore from -epsilon to 0 AND applies update in one pass

kernel void restore_and_update_bf16(
    device bfloat* data [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_elements [[buffer(2)]],
    constant float& restore_epsilon [[buffer(3)]],
    constant float& update_scale [[buffer(4)]],
    constant ulong& restore_seed [[buffer(5)]],
    constant ulong& update_seed [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    const int idx = tgid * tgs + tid;
    if (idx >= num_elements) return;

    // Load BF16 and convert to float
    float val = float(data[idx]);
    const float perturb = perturbation[idx];

    // Restore: add epsilon to go from -epsilon to 0
    val += restore_epsilon * perturb;

    // Apply update: subtract eta*mu*z (gradient descent)
    val -= update_scale * perturb;

    // Convert back to BF16 and store
    data[idx] = bfloat(val);
}

// ============================================================================
// Combined Restore + Update Kernel (Q8_0)
// ============================================================================
// Performs both restore from -epsilon to 0 AND applies update in one pass

kernel void restore_and_update_q8_0(
    device block_q8_0* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& restore_epsilon [[buffer(3)]],
    constant float& update_scale [[buffer(4)]],
    constant ulong& restore_seed [[buffer(5)]],
    constant ulong& update_seed [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= QK8_0) return;

    device block_q8_0* block = &blocks[block_idx];
    const float d = float(block->d);

    if (abs(d) < 1e-10f) return;

    const int global_idx = block_idx * QK8_0 + tid;
    const float perturb = perturbation[global_idx];

    // Current value (at -epsilon perturbation)
    float val = d * (float)block->qs[tid];

    // Restore: add epsilon to go from -epsilon to 0
    // Use restore_seed for stochastic rounding consistency
    ulong rng_restore = restore_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_restore);
    val += restore_epsilon * perturb;

    // Apply update: subtract eta*mu*z (gradient descent)
    // Use different seed for update
    ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
    xorshift64star(&rng_update);
    val -= update_scale * perturb;

    // Final re-quantization
    float q_val = val / d;
    char new_q = stochastic_round_i8(q_val, &rng_update);

    block->qs[tid] = new_q;
}

// ============================================================================
// Combined Restore + Update Kernel (Q4K)
// ============================================================================
// Performs both restore from -epsilon to 0 AND applies update in one pass

kernel void restore_and_update_q4_K(
    device block_q4_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& restore_epsilon [[buffer(3)]],
    constant float& update_scale [[buffer(4)]],
    constant ulong& restore_seed [[buffer(5)]],
    constant ulong& update_seed [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= QK_K / 2) return;  // 128 threads

    device block_q4_K* block = &blocks[block_idx];

    const float dall = float(block->dm.x);
    const float dmin = float(block->dm.y);

    if (abs(dall) < 1e-10f) return;

    uchar packed = block->qs[tid];
    uchar q4_lo = packed & 0x0F;
    uchar q4_hi = (packed >> 4) & 0x0F;

    const int il = tid / 32;
    const int within_il = tid % 32;
    const int is = 2 * il;

    const int elem_lo = 64 * il + within_il;
    const int elem_hi = 64 * il + within_il + 32;

    uchar sc_lo, m_lo, sc_hi, m_hi;
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
        ulong rng_restore = restore_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_restore);

        // Dequantize current value (at -epsilon perturbation)
        float val = d_lo * (float)q4_lo - m_lo_f;

        // Restore: add epsilon to go from -epsilon to 0
        val += restore_epsilon * perturb;

        // Initialize RNG for update
        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
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

        ulong rng_restore = restore_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_restore);

        float val = d_hi * (float)q4_hi - m_hi_f;
        val += restore_epsilon * perturb;

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        val -= update_scale * perturb;

        float q_float = (val + m_hi_f) / d_hi;
        q4_hi = stochastic_round_u4(q_float, &rng_update);
    }

    block->qs[tid] = q4_lo | (q4_hi << 4);
}

// ============================================================================
// Combined Restore + Update Kernel (Q2K)
// ============================================================================

kernel void restore_and_update_q2_K(
    device block_q2_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& restore_epsilon [[buffer(3)]],
    constant float& update_scale [[buffer(4)]],
    constant ulong& restore_seed [[buffer(5)]],
    constant ulong& update_seed [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= 64) return;

    device block_q2_K* block = &blocks[block_idx];

    const float dall = float(block->dm.x);
    const float dmin = float(block->dm.y);

    if (abs(dall) < 1e-10f) return;

    const int n = tid / 32;
    const int l = tid - 32 * n;
    const int is = 8 * n + l / 16;
    const int qs_idx = 32 * n + l;
    uchar q_byte = block->qs[qs_idx];
    const int base_elem = 128 * n;

    uchar q0 = (q_byte >> 0) & 3;
    uchar q1 = (q_byte >> 2) & 3;
    uchar q2 = (q_byte >> 4) & 3;
    uchar q3 = (q_byte >> 6) & 3;

    // Element 0
    {
        const int elem_idx = base_elem + l + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
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

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
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

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
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

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
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
// Combined Restore + Update Kernel (Q3K)
// ============================================================================

kernel void restore_and_update_q3_K(
    device block_q3_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& restore_epsilon [[buffer(3)]],
    constant float& update_scale [[buffer(4)]],
    constant ulong& restore_seed [[buffer(5)]],
    constant ulong& update_seed [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= 64) return;

    device block_q3_K* block = &blocks[block_idx];

    const float d_all = float(block->d);
    if (abs(d_all) < 1e-10f) return;

    const int r = tid / 4;
    const int tid_mod = tid % 4;
    const int is0 = r % 2;
    const int l0 = 16 * is0 + 4 * tid_mod;
    const int n = r / 4;
    const int j = (r / 2) - 2 * n;

    const uchar m = 1 << (4 * n + j);
    const int is = 8 * n + 2 * j + is0;
    const int shift = 2 * j;

    const char scale = get_q3k_scale(is, block->scales);
    const float dl = d_all * (float)scale;

    device const uchar* q = block->qs + 32 * n;
    device const uchar* hm = block->hmask;

    for (int l_off = 0; l_off < 4; l_off++) {
        const int l = l0 + l_off;
        const int elem_idx = 128 * n + 32 * j + l;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        int q2_val = (q[l] >> shift) & 3;
        int q3_val = q2_val - ((hm[l] & m) ? 0 : 4);

        float val = dl * (float)q3_val;
        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / dl;
        char q3_new = stochastic_round_i3(q_float, &rng_update);

        int q2_new = (q3_new >= 0) ? q3_new : (q3_new + 4);
        bool h_bit = (q3_new >= 0);

        uchar qs_byte = block->qs[32 * n + l];
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

// ============================================================================
// Combined Restore + Update Kernel (Q5K)
// ============================================================================

kernel void restore_and_update_q5_K(
    device block_q5_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& restore_epsilon [[buffer(3)]],
    constant float& update_scale [[buffer(4)]],
    constant ulong& restore_seed [[buffer(5)]],
    constant ulong& update_seed [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= 64) return;

    device block_q5_K* block = &blocks[block_idx];

    const float dall = float(block->dm.x);
    const float dmin = float(block->dm.y);

    if (abs(dall) < 1e-10f) return;

    const int il = tid / 16;
    const int ir = tid % 16;
    const int is = 2 * il;
    const int base_elem = 64 * il + 2 * ir;

    uchar sc, m_val;
    get_scale_min_k4(is + 0, block->scales, sc, m_val);
    const float d1 = dall * (float)sc;
    const float m1 = dmin * (float)m_val;
    get_scale_min_k4(is + 1, block->scales, sc, m_val);
    const float d2 = dall * (float)sc;
    const float m2 = dmin * (float)m_val;

    uchar hm = 1 << (2 * il);

    // Element 0
    {
        const int elem_idx = base_elem + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const uchar ql_byte = block->qs[32 * il + 2 * ir];
        const uchar qh_byte = block->qh[2 * ir];
        int q5_val = (ql_byte & 0xF) + ((qh_byte & hm) ? 16 : 0);
        float val = d1 * (float)q5_val - m1;

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + m1) / d1;
        uchar q5_new = stochastic_round_u5(q_float, &rng_update);

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

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const uchar ql_byte = block->qs[32 * il + 2 * ir + 1];
        const uchar qh_byte = block->qh[2 * ir + 1];
        int q5_val = (ql_byte & 0xF) + ((qh_byte & hm) ? 16 : 0);
        float val = d1 * (float)q5_val - m1;

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + m1) / d1;
        uchar q5_new = stochastic_round_u5(q_float, &rng_update);

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

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const uchar ql_byte = block->qs[32 * il + 2 * ir];
        const uchar qh_byte = block->qh[2 * ir];
        int q5_val = (ql_byte >> 4) + ((qh_byte & hm) ? 16 : 0);
        float val = d2 * (float)q5_val - m2;

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + m2) / d2;
        uchar q5_new = stochastic_round_u5(q_float, &rng_update);

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

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const uchar ql_byte = block->qs[32 * il + 2 * ir + 1];
        const uchar qh_byte = block->qh[2 * ir + 1];
        int q5_val = (ql_byte >> 4) + ((qh_byte & hm) ? 16 : 0);
        float val = d2 * (float)q5_val - m2;

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = (val + m2) / d2;
        uchar q5_new = stochastic_round_u5(q_float, &rng_update);

        block->qs[32 * il + 2 * ir + 1] = (block->qs[32 * il + 2 * ir + 1] & 0x0F) | ((q5_new & 0x0F) << 4);
        if (q5_new >= 16) {
            block->qh[2 * ir + 1] |= hm;
        } else {
            block->qh[2 * ir + 1] &= ~hm;
        }
    }
}

// ============================================================================
// Combined Restore + Update Kernel (Q6K)
// ============================================================================

kernel void restore_and_update_q6_K(
    device block_q6_K* blocks [[buffer(0)]],
    device const float* perturbation [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    constant float& restore_epsilon [[buffer(3)]],
    constant float& update_scale [[buffer(4)]],
    constant ulong& restore_seed [[buffer(5)]],
    constant ulong& update_seed [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;

    if (tid >= 64) return;

    device block_q6_K* block = &blocks[block_idx];

    const float d = float(block->d);
    if (abs(d) < 1e-10f) return;

    const int ip = tid / 32;
    const int il = tid - 32 * ip;
    const int is = 8 * ip + il / 16;
    const int base_elem = 128 * ip + il;

    // Element 0
    {
        const int elem_idx = base_elem + 0;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const char sc = block->scales[is + 0];
        const uchar ql_byte = block->ql[64 * ip + il];
        const uchar qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte & 0xF) | (((qh_byte >> 0) & 3) << 4);
        float val = d * (float)sc * (float)(q6_val - 32);

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / (d * (float)sc) + 32.0f;
        char q6_new = stochastic_round_i6(q_float - 32.0f, &rng_update) + 32;
        q6_new = (char)max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il] = (block->ql[64 * ip + il] & 0xF0) | (q6_new & 0x0F);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x03) | ((q6_new >> 4) & 0x03);
    }

    // Element 1
    {
        const int elem_idx = base_elem + 32;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const char sc = block->scales[is + 2];
        const uchar ql_byte = block->ql[64 * ip + il + 32];
        const uchar qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte & 0xF) | (((qh_byte >> 2) & 3) << 4);
        float val = d * (float)sc * (float)(q6_val - 32);

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / (d * (float)sc) + 32.0f;
        char q6_new = stochastic_round_i6(q_float - 32.0f, &rng_update) + 32;
        q6_new = (char)max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il + 32] = (block->ql[64 * ip + il + 32] & 0xF0) | (q6_new & 0x0F);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x0C) | (((q6_new >> 4) & 0x03) << 2);
    }

    // Element 2
    {
        const int elem_idx = base_elem + 64;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const char sc = block->scales[is + 4];
        const uchar ql_byte = block->ql[64 * ip + il];
        const uchar qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte >> 4) | (((qh_byte >> 4) & 3) << 4);
        float val = d * (float)sc * (float)(q6_val - 32);

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / (d * (float)sc) + 32.0f;
        char q6_new = stochastic_round_i6(q_float - 32.0f, &rng_update) + 32;
        q6_new = (char)max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il] = (block->ql[64 * ip + il] & 0x0F) | ((q6_new & 0x0F) << 4);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0x30) | (((q6_new >> 4) & 0x03) << 4);
    }

    // Element 3
    {
        const int elem_idx = base_elem + 96;
        const int global_idx = block_idx * QK_K + elem_idx;
        const float perturb = perturbation[global_idx];

        ulong rng_update = update_seed ^ ((ulong)global_idx) * 0x9E3779B97F4A7C15ULL;
        xorshift64star(&rng_update);

        const char sc = block->scales[is + 6];
        const uchar ql_byte = block->ql[64 * ip + il + 32];
        const uchar qh_byte = block->qh[32 * ip + il];
        const int q6_val = (int)(ql_byte >> 4) | (((qh_byte >> 6) & 3) << 4);
        float val = d * (float)sc * (float)(q6_val - 32);

        val += restore_epsilon * perturb;
        val -= update_scale * perturb;

        float q_float = val / (d * (float)sc) + 32.0f;
        char q6_new = stochastic_round_i6(q_float - 32.0f, &rng_update) + 32;
        q6_new = (char)max(0, min(63, (int)q6_new));

        block->ql[64 * ip + il + 32] = (block->ql[64 * ip + il + 32] & 0x0F) | ((q6_new & 0x0F) << 4);
        block->qh[32 * ip + il] = (block->qh[32 * ip + il] & ~0xC0) | (((q6_new >> 4) & 0x03) << 6);
    }
}

// ============================================================================
// Phase 2: Extract Scales Kernels
// ============================================================================
// 1 thread per block, reads the scale field, writes to f32 output buffer

kernel void extract_scales_q8_0(
    device const block_q8_0* blocks [[buffer(0)]],
    device float* scales_out [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    scales_out[block_idx] = float(blocks[block_idx].d);
}

kernel void extract_scales_q4_K(
    device const block_q4_K* blocks [[buffer(0)]],
    device float* scales_out [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    // Extract dall (super-block scale) from dm.x
    scales_out[block_idx] = float(blocks[block_idx].dm.x);
}

kernel void extract_scales_q2_K(
    device const block_q2_K* blocks [[buffer(0)]],
    device float* scales_out [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    // Extract dall (super-block scale) from dm.x
    scales_out[block_idx] = float(blocks[block_idx].dm.x);
}

kernel void extract_scales_q3_K(
    device const block_q3_K* blocks [[buffer(0)]],
    device float* scales_out [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    scales_out[block_idx] = float(blocks[block_idx].d);
}

kernel void extract_scales_q5_K(
    device const block_q5_K* blocks [[buffer(0)]],
    device float* scales_out [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    // Extract dall (super-block scale) from dm.x
    scales_out[block_idx] = float(blocks[block_idx].dm.x);
}

kernel void extract_scales_q6_K(
    device const block_q6_K* blocks [[buffer(0)]],
    device float* scales_out [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    scales_out[block_idx] = float(blocks[block_idx].d);
}

// ============================================================================
// Phase 2: Modify Block Scales Kernels
// ============================================================================
// 1 thread per block, multiplies scale field by f32 multiplier

kernel void modify_block_scales_q8_0(
    device block_q8_0* blocks [[buffer(0)]],
    device const float* multipliers [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    float d = float(blocks[block_idx].d);
    d *= multipliers[block_idx];
    blocks[block_idx].d = half(d);
}

kernel void modify_block_scales_q4_K(
    device block_q4_K* blocks [[buffer(0)]],
    device const float* multipliers [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    float dall = float(blocks[block_idx].dm.x);
    dall *= multipliers[block_idx];
    blocks[block_idx].dm.x = half(dall);
}

kernel void modify_block_scales_q2_K(
    device block_q2_K* blocks [[buffer(0)]],
    device const float* multipliers [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    float dall = float(blocks[block_idx].dm.x);
    dall *= multipliers[block_idx];
    blocks[block_idx].dm.x = half(dall);
}

kernel void modify_block_scales_q3_K(
    device block_q3_K* blocks [[buffer(0)]],
    device const float* multipliers [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    float d = float(blocks[block_idx].d);
    d *= multipliers[block_idx];
    blocks[block_idx].d = half(d);
}

kernel void modify_block_scales_q5_K(
    device block_q5_K* blocks [[buffer(0)]],
    device const float* multipliers [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    float dall = float(blocks[block_idx].dm.x);
    dall *= multipliers[block_idx];
    blocks[block_idx].dm.x = half(dall);
}

kernel void modify_block_scales_q6_K(
    device block_q6_K* blocks [[buffer(0)]],
    device const float* multipliers [[buffer(1)]],
    constant int& num_blocks [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int block_idx = tgid;
    if (block_idx >= num_blocks) return;
    if (tid != 0) return;

    float d = float(blocks[block_idx].d);
    d *= multipliers[block_idx];
    blocks[block_idx].d = half(d);
}
