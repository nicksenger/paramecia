// QuZO Fused Perturbation Kernels (Metal Shading Language)
// Metal port of CUDA kernels that perform perturbation during matmul dequantization
//
// Instead of modifying weights beforehand, these kernels generate perturbations
// on-the-fly using deterministic RNG (Philox) during the forward pass.
//
// Key advantages:
// - No memory overhead for storing perturbed weights
// - Works seamlessly with CPU-offloaded experts
// - Perturbation is computed as weights stream through memory
//
// Per-tensor uniqueness:
// Each tensor gets a unique perturbation by XORing the seed with a
// tensor_id passed as a kernel argument. This ensures different tensors
// get different perturbation patterns even with the same global seed.

#include <metal_stdlib>
using namespace metal;

// Block sizes for GGML quantization formats
#define QK8_0 32
#define QK_K 256
#define WARP_SIZE 32

// ============================================================================
// Block Structures for GGML Quantization Formats
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
// Philox 4x32 Counter-Based RNG
// ============================================================================
// Philox is ideal for GPU because:
// - Stateless: generate random(seed, index) without maintaining state
// - Deterministic: same (seed, index) always produces same value
// - High quality: passes BigCrush statistical tests
// - Fast: simple arithmetic operations, no transcendentals

// Philox constants (from the original paper)
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

// Single Philox round
inline void philox_round(thread uint& c0, thread uint& c1,
                         thread uint& c2, thread uint& c3,
                         uint k0, uint k1) {
    uint hi0, lo0, hi1, lo1;

    // Multiply and get high/low parts
    lo0 = c0 * PHILOX_M0;
    hi0 = mulhi(c0, PHILOX_M0);
    lo1 = c2 * PHILOX_M1;
    hi1 = mulhi(c2, PHILOX_M1);

    // Update counters with permutation
    uint new_c0 = hi1 ^ c1 ^ k0;
    uint new_c1 = lo1;
    uint new_c2 = hi0 ^ c3 ^ k1;
    uint new_c3 = lo0;

    c0 = new_c0;
    c1 = new_c1;
    c2 = new_c2;
    c3 = new_c3;
}

// Generate 4 uint random values from seed and index using Philox-4x32-10
inline void philox4x32(ulong seed, ulong index,
                       thread uint& r0, thread uint& r1,
                       thread uint& r2, thread uint& r3) {
    // Initialize counter from index
    uint c0 = (uint)index;
    uint c1 = (uint)(index >> 32);
    uint c2 = 0;
    uint c3 = 0;

    // Initialize key from seed
    uint k0 = (uint)seed;
    uint k1 = (uint)(seed >> 32);

    // 10 rounds of Philox
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
inline float philox_uniform(ulong seed, ulong index) {
    uint r0, r1, r2, r3;
    philox4x32(seed, index, r0, r1, r2, r3);
    // Convert to float in [0, 1) then shift to [-1, 1]
    return (float)(r0 >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

// Generate Gaussian-distributed float using Box-Muller (uses 2 uniforms)
inline float philox_gaussian(ulong seed, ulong index) {
    uint r0, r1, r2, r3;
    philox4x32(seed, index, r0, r1, r2, r3);

    // Box-Muller transform
    float u1 = (float)(r0 | 1) * (1.0f / 4294967296.0f);  // (0, 1]
    float u2 = (float)r1 * (1.0f / 4294967296.0f);         // [0, 1)

    float radius = sqrt(-2.0f * log(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;

    return radius * cos(theta);
}

// ============================================================================
// Q4K Helper: Decode sub-block scale and min (matches GGML get_scale_min_k4)
// ============================================================================

inline void get_scale_min_k4(int j, device const uchar* scales,
                             thread uchar& sc, thread uchar& m) {
    if (j < 4) {
        sc = scales[j] & 63;
        m = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

// ============================================================================
// Q3K Helper: Get scale from 6-bit packed format
// ============================================================================

inline char get_q3k_scale_fused(int is, device const uchar* scales) {
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
    return us - 32;
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q8_0 (Vector-Matrix)
// ============================================================================
// This kernel performs: y = (W + eps*z) @ x
// where z is generated on-the-fly from the seed and weight index.
//
// The perturbation is applied during dequantization:
//   dequant_value = quant_value * delta
//   perturbed_value = dequant_value + epsilon * z[weight_index]

kernel void fused_mul_mat_vec_q8_0_f32(
    device const block_q8_0* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& nrows [[buffer(3)]],
    constant int& ncols [[buffer(4)]],
    constant ulong& seed [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant ulong& tensor_id [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    // XOR seed with tensor_id to get unique seed per tensor
    const ulong effective_seed = seed ^ tensor_id;

    const int row = tgid;
    if (row >= nrows) return;

    const int num_blocks = ncols / QK8_0;

    float sum = 0.0f;

    // Each simdgroup processes one row
    // Threads cooperate to process blocks
    for (int block_idx = (int)tid; block_idx < num_blocks; block_idx += WARP_SIZE) {
        device const block_q8_0* blk = &weights[row * num_blocks + block_idx];
        const float d = float(blk->d);

        // Process 32 elements in this block
        for (int i = 0; i < QK8_0; i++) {
            const int col = block_idx * QK8_0 + i;
            const int weight_idx = row * ncols + col;

            // Dequantize
            float w = d * (float)blk->qs[i];

            // Add perturbation using deterministic RNG
            float z = philox_gaussian(effective_seed, (ulong)weight_idx);
            w += epsilon * z;

            // Accumulate dot product
            sum += w * x[col];
        }
    }

    // Warp reduction using simd_shuffle_down
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }

    // First thread in simdgroup writes result
    if (tid == 0) {
        output[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q4K (Vector-Matrix)
// ============================================================================
// Handles the more complex Q4K dequantization with sub-block scales and mins.
// Dequantization formula: value = dall * sc * q_nibble - dmin * m

kernel void fused_mul_mat_vec_q4_K_f32(
    device const block_q4_K* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& nrows [[buffer(3)]],
    constant int& ncols [[buffer(4)]],
    constant ulong& seed [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant ulong& tensor_id [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    // XOR seed with tensor_id to get unique seed per tensor
    const ulong effective_seed = seed ^ tensor_id;

    const int row = tgid;
    if (row >= nrows) return;

    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        device const block_q4_K* blk = &weights[row * num_blocks + block_idx];

        const float dall = float(blk->dm.x);
        const float dmin = float(blk->dm.y);

        // Process elements assigned to this thread
        // Q4K has 256 elements per block
        // Each thread processes 256/32 = 8 elements
        for (int elem_offset = (int)tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Determine sub-block and get scales
            const int il = elem_offset / 64;
            const int is = 2 * il;
            const int within_il = elem_offset % 64;

            uchar sc, m;
            if (within_il < 32) {
                get_scale_min_k4(is + 0, blk->scales, sc, m);
            } else {
                get_scale_min_k4(is + 1, blk->scales, sc, m);
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
            uchar packed = blk->qs[byte_idx];
            uchar q4 = is_high_nibble ? (packed >> 4) : (packed & 0x0F);

            // Dequantize
            float w = d_eff * (float)q4 - m_eff;

            // Add perturbation
            float z = philox_gaussian(effective_seed, (ulong)weight_idx);
            w += epsilon * z;

            // Accumulate
            sum += w * x[col];
        }
    }

    // Warp reduction
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }

    if (tid == 0) {
        output[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q2K (Vector-Matrix)
// ============================================================================

kernel void fused_mul_mat_vec_q2_K_f32(
    device const block_q2_K* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& nrows [[buffer(3)]],
    constant int& ncols [[buffer(4)]],
    constant ulong& seed [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant ulong& tensor_id [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const ulong effective_seed = seed ^ tensor_id;

    const int row = tgid;
    if (row >= nrows) return;

    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        device const block_q2_K* blk = &weights[row * num_blocks + block_idx];

        const float dall = float(blk->dm.x);
        const float dmin = float(blk->dm.y);

        for (int elem_offset = (int)tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Q2K dequantization
            const int n = elem_offset / 128;
            const int elem_in_n = elem_offset % 128;
            const int l = elem_in_n % 32;
            const int shift_idx = elem_in_n / 32;  // 0, 1, 2, 3

            const int is = 8 * n + l / 16;
            const int qs_idx = 32 * n + l;
            const uchar q_byte = blk->qs[qs_idx];
            const uchar q2 = (q_byte >> (2 * shift_idx)) & 3;

            const float d_sc = dall * (float)(blk->scales[is + 2 * shift_idx] & 0xF);
            const float d_min = dmin * (float)(blk->scales[is + 2 * shift_idx] >> 4);

            float w = d_sc * (float)q2 - d_min;

            // Add perturbation
            float z = philox_gaussian(effective_seed, (ulong)weight_idx);
            w += epsilon * z;

            sum += w * x[col];
        }
    }

    // Warp reduction
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }

    if (tid == 0) {
        output[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q3K (Vector-Matrix)
// ============================================================================

kernel void fused_mul_mat_vec_q3_K_f32(
    device const block_q3_K* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& nrows [[buffer(3)]],
    constant int& ncols [[buffer(4)]],
    constant ulong& seed [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant ulong& tensor_id [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const ulong effective_seed = seed ^ tensor_id;

    const int row = tgid;
    if (row >= nrows) return;

    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        device const block_q3_K* blk = &weights[row * num_blocks + block_idx];
        const float d_all = float(blk->d);

        for (int elem_offset = (int)tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Q3K dequantization (matching GGML layout)
            const int n = elem_offset / 128;        // 0 or 1
            const int j = (elem_offset % 128) / 32;  // 0, 1, 2, 3
            const int l = elem_offset % 32;

            const uchar mask = 1 << (4 * n + j);
            const int is0 = l / 16;
            const int is = 8 * n + 2 * j + is0;
            const int shift = 2 * j;

            const char scale = get_q3k_scale_fused(is, blk->scales);
            const float dl = d_all * (float)scale;

            const uchar q_byte = blk->qs[32 * n + l];
            const uchar q2_val = (q_byte >> shift) & 3;
            const int q3_val = (int)q2_val - ((blk->hmask[l] & mask) ? 0 : 4);

            float w = dl * (float)q3_val;

            // Add perturbation
            float z = philox_gaussian(effective_seed, (ulong)weight_idx);
            w += epsilon * z;

            sum += w * x[col];
        }
    }

    // Warp reduction
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }

    if (tid == 0) {
        output[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q5K (Vector-Matrix)
// ============================================================================

kernel void fused_mul_mat_vec_q5_K_f32(
    device const block_q5_K* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& nrows [[buffer(3)]],
    constant int& ncols [[buffer(4)]],
    constant ulong& seed [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant ulong& tensor_id [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const ulong effective_seed = seed ^ tensor_id;

    const int row = tgid;
    if (row >= nrows) return;

    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        device const block_q5_K* blk = &weights[row * num_blocks + block_idx];

        const float dall = float(blk->dm.x);
        const float dmin = float(blk->dm.y);

        for (int elem_offset = (int)tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Q5K dequantization
            const int il = elem_offset / 64;    // 0..3
            const int within_64 = elem_offset % 64;
            const int is = 2 * il;

            uchar sc, m;
            int q5_val;

            if (within_64 < 32) {
                // Low nibble elements
                get_scale_min_k4(is + 0, blk->scales, sc, m);
                const int byte_idx = 32 * il + within_64;
                const uchar ql = blk->qs[byte_idx];
                const uchar qh = blk->qh[within_64];
                const uchar hm = 1 << (2 * il);
                q5_val = (ql & 0xF) + ((qh & hm) ? 16 : 0);
            } else {
                // High nibble elements
                get_scale_min_k4(is + 1, blk->scales, sc, m);
                const int within_32 = within_64 - 32;
                const int byte_idx = 32 * il + within_32;
                const uchar ql = blk->qs[byte_idx];
                const uchar qh = blk->qh[within_32];
                const uchar hm = 1 << (2 * il + 1);
                q5_val = (ql >> 4) + ((qh & hm) ? 16 : 0);
            }

            float w = dall * (float)sc * (float)q5_val - dmin * (float)m;

            // Add perturbation
            float z = philox_gaussian(effective_seed, (ulong)weight_idx);
            w += epsilon * z;

            sum += w * x[col];
        }
    }

    // Warp reduction
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }

    if (tid == 0) {
        output[row] = sum;
    }
}

// ============================================================================
// Fused Dequantize + Perturb + MatMul for Q6K (Vector-Matrix)
// ============================================================================

kernel void fused_mul_mat_vec_q6_K_f32(
    device const block_q6_K* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& nrows [[buffer(3)]],
    constant int& ncols [[buffer(4)]],
    constant ulong& seed [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant ulong& tensor_id [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const ulong effective_seed = seed ^ tensor_id;

    const int row = tgid;
    if (row >= nrows) return;

    const int num_blocks = ncols / QK_K;

    float sum = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        device const block_q6_K* blk = &weights[row * num_blocks + block_idx];
        const float d = float(blk->d);

        // Process elements assigned to this thread
        for (int elem_offset = (int)tid; elem_offset < QK_K; elem_offset += WARP_SIZE) {
            const int col = block_idx * QK_K + elem_offset;
            const int weight_idx = row * ncols + col;

            // Q6K dequantization (matching GGML)
            const int ip = elem_offset / 128;
            const int il = (elem_offset - 128 * ip) % 32;
            const int is = 8 * ip + il / 16;
            const int elem_in_half = elem_offset % 128;

            char sc;
            int q6_val;

            if (elem_in_half < 32) {
                sc = blk->scales[is + 0];
                const uchar ql_val = blk->ql[64 * ip + il];
                const uchar qh_val = blk->qh[32 * ip + il];
                q6_val = (int)(ql_val & 0xF) | (((qh_val >> 0) & 3) << 4);
            } else if (elem_in_half < 64) {
                sc = blk->scales[is + 2];
                const uchar ql_val = blk->ql[64 * ip + il + 32];
                const uchar qh_val = blk->qh[32 * ip + il];
                q6_val = (int)(ql_val & 0xF) | (((qh_val >> 2) & 3) << 4);
            } else if (elem_in_half < 96) {
                sc = blk->scales[is + 4];
                const uchar ql_val = blk->ql[64 * ip + il];
                const uchar qh_val = blk->qh[32 * ip + il];
                q6_val = (int)(ql_val >> 4) | (((qh_val >> 4) & 3) << 4);
            } else {
                sc = blk->scales[is + 6];
                const uchar ql_val = blk->ql[64 * ip + il + 32];
                const uchar qh_val = blk->qh[32 * ip + il];
                q6_val = (int)(ql_val >> 4) | (((qh_val >> 6) & 3) << 4);
            }

            float w = d * (float)sc * (float)(q6_val - 32);

            // Add perturbation
            float z = philox_gaussian(effective_seed, (ulong)weight_idx);
            w += epsilon * z;

            sum += w * x[col];
        }
    }

    // Warp reduction
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }

    if (tid == 0) {
        output[row] = sum;
    }
}

// ============================================================================
// Fused Perturb + MatMul for BF16 (Vector-Matrix)
// ============================================================================
// BF16 doesn't need dequantization, just load -> perturb -> matmul
// Kernel: y = (W + eps*z) @ x where z is generated on-the-fly

kernel void fused_mul_mat_vec_bf16_f32(
    device const bfloat* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& nrows [[buffer(3)]],
    constant int& ncols [[buffer(4)]],
    constant ulong& seed [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant ulong& tensor_id [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    // XOR seed with tensor_id to get unique seed per tensor
    const ulong effective_seed = seed ^ tensor_id;

    const int row = tgid;
    if (row >= nrows) return;

    float sum = 0.0f;

    // Each simdgroup processes one row
    // Threads cooperate to process columns
    for (int col = (int)tid; col < ncols; col += WARP_SIZE) {
        const int weight_idx = row * ncols + col;

        // Load BF16 weight and convert to float
        float w = float(weights[weight_idx]);

        // Add perturbation using deterministic RNG
        float z = philox_gaussian(effective_seed, (ulong)weight_idx);
        w += epsilon * z;

        // Accumulate dot product
        sum += w * x[col];
    }

    // Warp reduction
    for (ushort offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += simd_shuffle_down(sum, offset);
    }

    // First thread in simdgroup writes result
    if (tid == 0) {
        output[row] = sum;
    }
}
