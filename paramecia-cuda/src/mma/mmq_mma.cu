// MMA tensor core matrix multiply for quantized weights (MMQ).
//
// Uses m16n8k16 s8×s8→s32 MMA instructions for Ampere+ (sm_80+).
// Weights are quantized (Q8_0, Q4_K, Q6_K), activations are Q8_1.
//
// Each thread block computes a (mmq_y × mmq_x) output tile.
// K dimension is iterated in tiles, with cooperative loading of both
// weight and activation data into shared memory.
//
// The key difference from DP4A-based MMQ:
// - DP4A: each thread computes one element via sequential dp4a calls
// - MMA: warp-level matrix multiply accumulates 16×8 tiles at once
//
// Layout:
//   weights (vx): [nrows_x, ncols_x] quantized (row-major blocks)
//   activations (vy): [ncols_y, nrows_y] Q8_1 (column-major, each column = one activation vector)
//   output (dst): [nrows_x, ncols_y] float (row-major)

#include "mma_utils.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// Configuration
// ============================================================================

// Output tile dimensions
#define MMQ_Y 64    // rows of output per block (weight dimension)
#define MMQ_X 64    // cols of output per block (activation dimension)

// K tile size: process this many K elements per iteration
// Must be multiple of QK8_0 (32) and QK_K (256)
#define K_TILE_Q8_0 128  // 4 Q8_0 blocks per row
#define K_TILE_Q4_K 256  // 1 Q4_K block per row
#define K_TILE_Q6_K 256  // 1 Q6_K block per row

// Thread block: 4 warps × 32 threads = 128 threads
#define NWARPS_MMQ 4

// Shared memory layout for Q8_1 activations:
// We load mmq_x columns, each K_TILE elements deep.
// Each Q8_1 block = 32 int8 values + scale(f32) + sum(f32)
// For MMA, we need the int8 values in shared memory laid out for ldmatrix.

// ============================================================================
// Shared memory tile structures
// ============================================================================

// For MMA i8 path, we need weights and activations as int8 in shared memory,
// with scales stored separately.
// The i8 MMA m16n8k16 expects:
//   A (weight): row-major 16×16 int8 → 2 int32 regs per thread
//   B (activation): col-major 16×8 int8 → 1 int32 reg per thread

// ============================================================================
// Q8_0 × Q8_1 MMA kernel
// ============================================================================

// Q8_0 weights: 32 int8 values + 1 f16 scale per block (34 bytes)
// Q8_1 activations: 32 int8 values + 1 f32 scale + 1 f32 sum per block (40 bytes)
//
// For each K_TILE iteration, we load weight rows and activation columns,
// dequantize to int8, perform MMA i8, then scale the i32 results by
// (weight_d × activation_d) and accumulate to float.

template<int mmq_y, int mmq_x>
__device__ void mul_mat_q8_0_mma(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tid = warp_id * WARP_SIZE + lane_id;

    const int row_dst_0 = blockIdx.x * mmq_y;
    const int col_dst_0 = blockIdx.y * mmq_x;

    const int blocks_per_row_x = ncols_x / QK8_0;

    // Each warp handles a 16×8 MMA tile.
    // With 4 warps and mmq_y=64, mmq_x=64:
    //   Layout: 4 rows of 16 × 8 cols of 8 = 64×64
    //   Warp (wy, wx): wy = warp_id / 2, wx = warp_id % 2
    //   But we iterate: each warp does multiple 16×8 tiles to fill mmq_y × mmq_x

    // Output accumulator: each warp computes multiple 16×8 tiles
    // With 4 warps: warp grid is (mmq_y/16) × (mmq_x/8) = 4×8 = 32 tiles
    // 32 tiles / 4 warps = 8 tiles per warp
    // Assign: warp handles (mmq_y/16 / 2) rows × (mmq_x/8 / 1) cols of tiles
    // Actually simpler: warp `w` handles tiles at row offsets [w*16, w*16+15]
    //   and iterates over all mmq_x/8 column tiles

    constexpr int n_mma_y = mmq_y / MMA_M;  // 4 (for mmq_y=64)
    constexpr int n_mma_x = mmq_x / MMA_N;  // 8 (for mmq_x=64)
    constexpr int tiles_per_warp_y = n_mma_y / NWARPS_MMQ; // 1
    constexpr int tiles_per_warp = tiles_per_warp_y * n_mma_x; // 8

    float acc[tiles_per_warp][4]; // MMA accumulator: 4 floats per 16×8 tile
#pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            acc[t][i] = 0.0f;
        }
    }

    // Shared memory for weight and activation tiles
    // Weight (A): mmq_y rows × K_TILE_Q8_0 int8 values + scales
    // Activation (B): mmq_x columns × K_TILE_Q8_0 int8 values + scales
    constexpr int K_TILE = K_TILE_Q8_0;
    constexpr int k_blocks = K_TILE / QK8_0; // 4

    extern __shared__ char shmem[];
    // Layout in shared memory:
    // [0, mmq_y * K_TILE): weight int8 values (row-major, padded)
    // [mmq_y * K_TILE, mmq_y * K_TILE + mmq_y * k_blocks * 4): weight scales (float)
    // [next, next + mmq_x * K_TILE): activation int8 values (col-major, padded)
    // [next, next + mmq_x * k_blocks * 4): activation d scales (float)
    // [next, next + mmq_x * k_blocks * 4): activation s sums (float)

    constexpr int w_qs_stride = K_TILE + 16; // padding for bank conflicts
    int8_t  *tile_w_qs = (int8_t *)shmem;
    float   *tile_w_d  = (float *)(tile_w_qs + mmq_y * w_qs_stride);

    constexpr int a_qs_stride = K_TILE + 16;
    int8_t  *tile_a_qs = (int8_t *)(tile_w_d + mmq_y * k_blocks);
    float   *tile_a_d  = (float *)(tile_a_qs + mmq_x * a_qs_stride);

    const block_q8_0 *x = (const block_q8_0 *)vx;
    const block_q8_1 *y = (const block_q8_1 *)vy;

    // Iterate over K dimension
    for (int k0 = 0; k0 < blocks_per_row_x; k0 += k_blocks) {

        // === Load weight tile (Q8_0) ===
        // 128 threads load mmq_y × k_blocks blocks
        // Each block = 32 int8 values + 1 f16 scale
        {
            const int loads_per_thread = (mmq_y * k_blocks + 128 - 1) / 128;
            for (int l = 0; l < loads_per_thread; l++) {
                int idx = tid + l * 128;
                if (idx >= mmq_y * k_blocks) break;

                int row = idx / k_blocks;
                int kb  = idx % k_blocks;

                int abs_row = row_dst_0 + row;
                if (abs_row < nrows_x) {
                    const block_q8_0 *blk = &x[abs_row * blocks_per_row_x + k0 + kb];
                    float d = __half2float(*reinterpret_cast<const __half *>(&blk->d));
                    tile_w_d[row * k_blocks + kb] = d;

                    // Copy int8 values
                    int8_t *dst_qs = &tile_w_qs[row * w_qs_stride + kb * QK8_0];
                    const int8_t *src_qs = blk->qs;
                    // Copy 32 bytes = 8 int32s
                    for (int i = 0; i < QK8_0; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = load_i32_aligned2(src_qs + i);
                    }
                } else {
                    tile_w_d[row * k_blocks + kb] = 0.0f;
                    int8_t *dst_qs = &tile_w_qs[row * w_qs_stride + kb * QK8_0];
                    for (int i = 0; i < QK8_0; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = 0;
                    }
                }
            }
        }

        // === Load activation tile (Q8_1) ===
        {
            const int blocks_per_col_y = nrows_y / QK8_1;
            const int loads_per_thread = (mmq_x * k_blocks + 128 - 1) / 128;
            for (int l = 0; l < loads_per_thread; l++) {
                int idx = tid + l * 128;
                if (idx >= mmq_x * k_blocks) break;

                int col = idx / k_blocks;
                int kb  = idx % k_blocks;

                int abs_col = col_dst_0 + col;
                if (abs_col < ncols_y) {
                    const block_q8_1 *blk = &y[abs_col * blocks_per_col_y + k0 + kb];
                    tile_a_d[col * k_blocks + kb] = __half2float(__low2half(blk->ds));

                    int8_t *dst_qs = &tile_a_qs[col * a_qs_stride + kb * QK8_1];
                    const int8_t *src_qs = blk->qs;
                    for (int i = 0; i < QK8_1; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = *reinterpret_cast<const int *>(src_qs + i);
                    }
                } else {
                    tile_a_d[col * k_blocks + kb] = 0.0f;
                    int8_t *dst_qs = &tile_a_qs[col * a_qs_stride + kb * QK8_1];
                    for (int i = 0; i < QK8_1; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = 0;
                    }
                }
            }
        }

        __syncthreads();

        // === MMA computation ===
        // Warp's row offset in the output tile
        const int warp_row_offset = warp_id * tiles_per_warp_y * MMA_M;

        for (int k_sub = 0; k_sub < K_TILE; k_sub += MMA_K) {
            // Determine which Q8_0/Q8_1 block this sub-tile falls into
            int kb = k_sub / QK8_0;
            int k_in_block = k_sub % QK8_0;

            // Load A fragments (weight, row-major) for this warp's rows
            // MMA A: m16×k16 int8, row-major → 2 int32 regs per thread
            // Thread mapping: threads 0-15 contribute row 0-15, each holds 16 bytes (= 16 int8)
            // packed into 2 int32 regs (a[0] = bytes 0..3, a[1] = bytes 8..11 with specific mapping)
            //
            // For ldmatrix with int8, we treat it as loading b16 elements (2 bytes per element)
            // m8n8.x2 loads 2 matrices of 8×8 b16 = 16×8 b16 = 16×16 b8 (int8)

            for (int tile_y = 0; tile_y < tiles_per_warp_y; tile_y++) {
                int mma_row = warp_row_offset + tile_y * MMA_M;

                // A fragment (m16n8k16, s8, row-major):
                //   groupID = lane_id / 4 (0..7) → row 0..7
                //   tid_in_group = lane_id % 4 (0..3) → col group
                //   a[0]: A[groupID][tid_in_group*4 .. tid_in_group*4+3]
                //   a[1]: A[groupID+8][tid_in_group*4 .. tid_in_group*4+3]
                int a[2];
                {
                    int groupID = lane_id / 4;
                    int tid_in_group = lane_id % 4;
                    int col_offset = tid_in_group * 4;
                    a[0] = *reinterpret_cast<int *>(&tile_w_qs[(mma_row + groupID) * w_qs_stride + k_sub + col_offset]);
                    a[1] = *reinterpret_cast<int *>(&tile_w_qs[(mma_row + groupID + 8) * w_qs_stride + k_sub + col_offset]);
                }

                for (int tile_x = 0; tile_x < n_mma_x; tile_x++) {
                    int mma_col = tile_x * MMA_N;

                    // B fragment (m16n8k16, s8, col-major):
                    //   groupID = lane_id / 4 (0..7) → n column
                    //   tid_in_group = lane_id % 4 (0..3) → k row group
                    //   b[0]: B[tid_in_group*4 .. tid_in_group*4+3][groupID]
                    int b[1];
                    {
                        int groupID = lane_id / 4;
                        int tid_in_group = lane_id % 4;
                        int col_abs = mma_col + groupID;
                        int k_offset = tid_in_group * 4;
                        b[0] = *reinterpret_cast<int *>(&tile_a_qs[col_abs * a_qs_stride + k_sub + k_offset]);
                    }

                    int c[4] = {0, 0, 0, 0};
                    int d[4];
                    int tile_idx = tile_y * n_mma_x + tile_x;
                    mma_i8_m16n8k16(d, a, b, c);

                    // Scale: multiply by weight_d × activation_d for the relevant Q8 blocks
                    // Since K sub-tile of 16 may span parts of a Q8 block (32 values),
                    // and we're processing 16 consecutive int8 values, the scale is
                    // uniform within each 32-element Q8 block.
                    //
                    // For the MMA result, each thread's d[i] is a partial sum of
                    // 16 int8×int8 products. We need to scale by w_d * a_d.
                    //
                    // For m16n8k16 D fragment layout (same for s32 and f32):
                    // d[0] -> row = groupID,     col = 2*tid_in_group     (rows 0..7, even cols)
                    // d[1] -> row = groupID,     col = 2*tid_in_group + 1 (rows 0..7, odd cols)
                    // d[2] -> row = groupID + 8, col = 2*tid_in_group     (rows 8..15, even cols)
                    // d[3] -> row = groupID + 8, col = 2*tid_in_group + 1 (rows 8..15, odd cols)

                    // Weight scale for each of the 16 rows covered by this MMA:
                    // MMA rows 0..15 map to weight rows mma_row..mma_row+15
                    // The K position k_sub..k_sub+15 falls in Q8_0 block kb, position k_in_block..k_in_block+15

                    // For simplicity and correctness: since each MMA m16n8k16 processes
                    // exactly 16 K elements from the same Q8_0 block half (or spanning two
                    // blocks at boundary), we track scale per row and col.

                    int lane_row_0 = lane_id / 4;             // 0..7
                    int lane_row_1 = lane_row_0 + 8;          // 8..15
                    int lane_col_0 = 2 * (lane_id % 4);       // 0,2,4,6
                    int lane_col_1 = lane_col_0 + 1;          // 1,3,5,7

                    float w_d_0 = tile_w_d[(mma_row + lane_row_0) * k_blocks + kb];
                    float w_d_1 = tile_w_d[(mma_row + lane_row_1) * k_blocks + kb];
                    float a_d_0 = tile_a_d[(mma_col + lane_col_0) * k_blocks + kb];
                    float a_d_1 = tile_a_d[(mma_col + lane_col_1) * k_blocks + kb];

                    // If k_sub crosses a Q8_0 block boundary (k_in_block + 16 > 32),
                    // the second half uses the next block's scale. But since K_TILE is
                    // a multiple of QK8_0 and MMA_K=16 < QK8_0=32, k_sub is always
                    // aligned within a block (k_in_block is 0 or 16).
                    // For k_in_block=16, we're still in the same Q8_0 block, so scale is the same.

                    // d[0] → acc[0] (rows 0-7, even col): w_d_0 * a_d_0
                    // d[1] → acc[2] (rows 0-7, odd col):  w_d_0 * a_d_1
                    // d[2] → acc[1] (rows 8-15, even col): w_d_1 * a_d_0
                    // d[3] → acc[3] (rows 8-15, odd col):  w_d_1 * a_d_1
                    acc[tile_idx][0] += (float)d[0] * w_d_0 * a_d_0;
                    acc[tile_idx][1] += (float)d[2] * w_d_1 * a_d_0;
                    acc[tile_idx][2] += (float)d[1] * w_d_0 * a_d_1;
                    acc[tile_idx][3] += (float)d[3] * w_d_1 * a_d_1;
                }
            }
        }

        __syncthreads();
    }

    // === Write output ===
    const int warp_row_offset = warp_id * tiles_per_warp_y * MMA_M;

    for (int tile_y = 0; tile_y < tiles_per_warp_y; tile_y++) {
        int mma_row = warp_row_offset + tile_y * MMA_M;
        for (int tile_x = 0; tile_x < n_mma_x; tile_x++) {
            int mma_col = tile_x * MMA_N;
            int tile_idx = tile_y * n_mma_x + tile_x;

            int lane_row_0 = lane_id / 4;
            int lane_row_1 = lane_row_0 + 8;
            int lane_col_0 = 2 * (lane_id % 4);
            int lane_col_1 = lane_col_0 + 1;

            int r0 = row_dst_0 + mma_row + lane_row_0;
            int r1 = row_dst_0 + mma_row + lane_row_1;
            int c0 = col_dst_0 + mma_col + lane_col_0;
            int c1 = col_dst_0 + mma_col + lane_col_1;

            // Output column-major: acc[0]=(r0,c0), acc[1]=(r1,c0), acc[2]=(r0,c1), acc[3]=(r1,c1)
            if (r0 < nrows_dst && c0 < ncols_y) dst[c0 * nrows_dst + r0] = acc[tile_idx][0];
            if (r1 < nrows_dst && c0 < ncols_y) dst[c0 * nrows_dst + r1] = acc[tile_idx][1];
            if (r0 < nrows_dst && c1 < ncols_y) dst[c1 * nrows_dst + r0] = acc[tile_idx][2];
            if (r1 < nrows_dst && c1 < ncols_y) dst[c1 * nrows_dst + r1] = acc[tile_idx][3];
        }
    }
}

// ============================================================================
// Q4_K × Q8_1 MMA kernel
// ============================================================================

// Q4_K: 256 elements per block (144 bytes)
//   - d (f16), dmin (f16), scales[12], qs[128]
//   - 4-bit quants, 8 sub-blocks of 32 values each
//   - Each sub-block has its own 6-bit scale and min
//
// Strategy: For each K_TILE=256 iteration (= 1 Q4_K block per weight row):
//   1. Load weight block, dequantize 4-bit values to int8 in shared memory
//   2. Load activation Q8_1 blocks (256/32 = 8 blocks per column)
//   3. Run MMA i8 sub-tiles, scale appropriately

template<int mmq_y, int mmq_x>
__device__ void mul_mat_q4_K_mma(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tid = warp_id * WARP_SIZE + lane_id;

    const int row_dst_0 = blockIdx.x * mmq_y;
    const int col_dst_0 = blockIdx.y * mmq_x;

    const int blocks_per_row_x = ncols_x / QK_K;

    constexpr int n_mma_y = mmq_y / MMA_M;
    constexpr int n_mma_x = mmq_x / MMA_N;
    constexpr int tiles_per_warp_y = n_mma_y / NWARPS_MMQ;
    constexpr int tiles_per_warp = tiles_per_warp_y * n_mma_x;

    float acc[tiles_per_warp][4];
#pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
#pragma unroll
        for (int i = 0; i < 4; i++) acc[t][i] = 0.0f;
    }

    constexpr int K_TILE = K_TILE_Q4_K; // 256
    constexpr int k_q8_blocks = K_TILE / QK8_1; // 8

    extern __shared__ char shmem[];
    constexpr int w_qs_stride = K_TILE + 16;
    int8_t *tile_w_qs = (int8_t *)shmem;
    // Per-sub-block scales: 8 sub-blocks per weight row × mmq_y rows
    float  *tile_w_sc = (float *)(tile_w_qs + mmq_y * w_qs_stride);
    float  *tile_w_mn = tile_w_sc + mmq_y * 8;

    constexpr int a_qs_stride = K_TILE + 16;
    int8_t *tile_a_qs = (int8_t *)(tile_w_mn + mmq_y * 8);
    float  *tile_a_d  = (float *)(tile_a_qs + mmq_x * a_qs_stride);
    float  *tile_a_s  = tile_a_d + mmq_x * k_q8_blocks;

    const block_q4_K *x = (const block_q4_K *)vx;
    const block_q8_1 *y = (const block_q8_1 *)vy;

    for (int k0 = 0; k0 < blocks_per_row_x; k0++) {

        // === Load and dequantize Q4_K weight tile ===
        {
            // Each of 128 threads helps load mmq_y Q4_K blocks
            // Each block has 256 4-bit values = 128 bytes of qs
            // We dequantize to int8: q4_val - 8 (to center around 0)
            // Actually for Q4_K with scales/mins, the dequantized value is:
            //   val = scale * q4_val - min
            // But for MMA i8, we want to keep it as integer and apply scale after MMA.
            // Strategy: store raw (q4_val - 8) as int8, store scale and min*8 separately.
            // After MMA: result = sum(w_qs[i] * a_qs[i])
            //   = sum((q4_val-8) * a_qs[i])
            //   = sum(q4_val * a_qs) - 8 * sum(a_qs)
            // Dequantized result = w_scale * (sum(q4_val * a_qs) - 8*sum(a_qs)) - w_min * sum(a_qs)
            //                    = w_scale * MMA_result - (8*w_scale + w_min) * a_sum
            // The a_sum is the Q8_1 block's `s` field.

            const int loads = (mmq_y + 128 - 1) / 128;
            for (int l = 0; l < loads; l++) {
                int row = tid + l * 128;
                if (row >= mmq_y) break;

                int abs_row = row_dst_0 + row;
                if (abs_row < nrows_x) {
                    const block_q4_K *blk = &x[abs_row * blocks_per_row_x + k0];
                    float d_super = __half2float(*reinterpret_cast<const __half *>(&blk->d));
                    float dmin_super = __half2float(*reinterpret_cast<const __half *>(&blk->dmin));

                    // Decode scales and mins for 8 sub-blocks
                    const uint8_t *sc = blk->scales;
                    for (int sb = 0; sb < 8; sb++) {
                        uint8_t sc_val, m_val;
                        if (sb < 4) {
                            sc_val = sc[sb] & 0x3F;
                            m_val  = sc[sb + 4] & 0x3F;
                        } else {
                            sc_val = ((sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4));
                            m_val  = ((sc[sb + 4] >> 4)  | ((sc[sb    ] >> 6) << 4));
                        }
                        tile_w_sc[row * 8 + sb] = d_super * (float)sc_val;
                        tile_w_mn[row * 8 + sb] = dmin_super * (float)m_val;
                    }

                    // Unpack 4-bit values to int8 (raw 0-15, fits in signed int8)
                    // Q4_K packs 256 values into 128 bytes. Each group of 32 bytes
                    // encodes 64 values: low nibbles → first 32, high nibbles → next 32.
                    int8_t *dst_qs = &tile_w_qs[row * w_qs_stride];
                    const uint8_t *src_qs = blk->qs;
                    for (int j = 0; j < QK_K; j += 64) {
                        const uint8_t *q = &src_qs[j / 2];
                        for (int l = 0; l < 32; l++) {
                            dst_qs[j + l]      = (int8_t)(q[l] & 0xF);
                            dst_qs[j + l + 32] = (int8_t)(q[l] >> 4);
                        }
                    }
                } else {
                    for (int sb = 0; sb < 8; sb++) {
                        tile_w_sc[row * 8 + sb] = 0.0f;
                        tile_w_mn[row * 8 + sb] = 0.0f;
                    }
                    int8_t *dst_qs = &tile_w_qs[row * w_qs_stride];
                    for (int i = 0; i < K_TILE; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = 0;
                    }
                }
            }
        }

        // === Load activation tile (Q8_1) ===
        {
            const int blocks_per_col_y = nrows_y / QK8_1;
            const int loads = (mmq_x * k_q8_blocks + 128 - 1) / 128;
            for (int l = 0; l < loads; l++) {
                int idx = tid + l * 128;
                if (idx >= mmq_x * k_q8_blocks) break;

                int col = idx / k_q8_blocks;
                int kb  = idx % k_q8_blocks;
                int abs_col = col_dst_0 + col;

                if (abs_col < ncols_y) {
                    const block_q8_1 *blk = &y[abs_col * blocks_per_col_y + k0 * k_q8_blocks + kb];
                    tile_a_d[col * k_q8_blocks + kb] = __half2float(__low2half(blk->ds));
                    tile_a_s[col * k_q8_blocks + kb] = __half2float(__high2half(blk->ds));

                    int8_t *dst_qs = &tile_a_qs[col * a_qs_stride + kb * QK8_1];
                    const int8_t *src_qs = blk->qs;
                    for (int i = 0; i < QK8_1; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = *reinterpret_cast<const int *>(src_qs + i);
                    }
                } else {
                    tile_a_d[col * k_q8_blocks + kb] = 0.0f;
                    tile_a_s[col * k_q8_blocks + kb] = 0.0f;
                    int8_t *dst_qs = &tile_a_qs[col * a_qs_stride + kb * QK8_1];
                    for (int i = 0; i < QK8_1; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = 0;
                    }
                }
            }
        }

        __syncthreads();

        // === MMA computation ===
        const int warp_row_offset = warp_id * tiles_per_warp_y * MMA_M;

        // Process 16 K elements at a time with MMA
        for (int k_sub = 0; k_sub < K_TILE; k_sub += MMA_K) {
            int q8_block = k_sub / QK8_1;   // which Q8_1 block (0..7)
            int sub_block = k_sub / 32;      // which Q4_K sub-block (0..7)

            for (int tile_y = 0; tile_y < tiles_per_warp_y; tile_y++) {
                int mma_row = warp_row_offset + tile_y * MMA_M;

                int a[2];
                {
                    int groupID = lane_id / 4;
                    int tid_in_group = lane_id % 4;
                    int col_offset = tid_in_group * 4;
                    a[0] = *reinterpret_cast<int *>(&tile_w_qs[(mma_row + groupID) * w_qs_stride + k_sub + col_offset]);
                    a[1] = *reinterpret_cast<int *>(&tile_w_qs[(mma_row + groupID + 8) * w_qs_stride + k_sub + col_offset]);
                }

                for (int tile_x = 0; tile_x < n_mma_x; tile_x++) {
                    int mma_col = tile_x * MMA_N;

                    int b[1];
                    {
                        int groupID = lane_id / 4;
                        int tid_in_group = lane_id % 4;
                        int col_abs = mma_col + groupID;
                        int k_offset = tid_in_group * 4;
                        b[0] = *reinterpret_cast<int *>(&tile_a_qs[col_abs * a_qs_stride + k_sub + k_offset]);
                    }

                    int c[4] = {0, 0, 0, 0};
                    int d[4];
                    mma_i8_m16n8k16(d, a, b, c);

                    int tile_idx = tile_y * n_mma_x + tile_x;

                    int lane_row_0 = lane_id / 4;
                    int lane_row_1 = lane_row_0 + 8;
                    int lane_col_0 = 2 * (lane_id % 4);
                    int lane_col_1 = lane_col_0 + 1;

                    // Q4_K scaling: raw q4 values (0-15), no centering
                    // MMA_dot = sum(q4_raw * a_qs) over 16 elements
                    // main term = w_scale * a_d * MMA_dot
                    // min correction = -w_min * a_s (once per Q8_1 block of 32 elements)
                    float w_sc_0 = tile_w_sc[(mma_row + lane_row_0) * 8 + sub_block];
                    float w_sc_1 = tile_w_sc[(mma_row + lane_row_1) * 8 + sub_block];

                    float a_d_0 = tile_a_d[(mma_col + lane_col_0) * k_q8_blocks + q8_block];
                    float a_d_1 = tile_a_d[(mma_col + lane_col_1) * k_q8_blocks + q8_block];

                    // d[1] and d[2] swapped: d[1]→(rows 0-7, odd col), d[2]→(rows 8-15, even col)
                    acc[tile_idx][0] += w_sc_0 * a_d_0 * (float)d[0];
                    acc[tile_idx][1] += w_sc_1 * a_d_0 * (float)d[2];
                    acc[tile_idx][2] += w_sc_0 * a_d_1 * (float)d[1];
                    acc[tile_idx][3] += w_sc_1 * a_d_1 * (float)d[3];

                    // Min correction: apply once per Q8_1 block (32 elements)
                    if (k_sub % QK8_1 == 0) {
                        float w_mn_0 = tile_w_mn[(mma_row + lane_row_0) * 8 + sub_block];
                        float w_mn_1 = tile_w_mn[(mma_row + lane_row_1) * 8 + sub_block];
                        float a_s_0 = tile_a_s[(mma_col + lane_col_0) * k_q8_blocks + q8_block];
                        float a_s_1 = tile_a_s[(mma_col + lane_col_1) * k_q8_blocks + q8_block];

                        acc[tile_idx][0] -= w_mn_0 * a_s_0;
                        acc[tile_idx][1] -= w_mn_1 * a_s_0;
                        acc[tile_idx][2] -= w_mn_0 * a_s_1;
                        acc[tile_idx][3] -= w_mn_1 * a_s_1;
                    }
                }
            }
        }

        __syncthreads();
    }

    // === Write output ===
    const int warp_row_out = warp_id * tiles_per_warp_y * MMA_M;
    for (int tile_y = 0; tile_y < tiles_per_warp_y; tile_y++) {
        int mma_row = warp_row_out + tile_y * MMA_M;
        for (int tile_x = 0; tile_x < n_mma_x; tile_x++) {
            int mma_col = tile_x * MMA_N;
            int tile_idx = tile_y * n_mma_x + tile_x;

            int lane_row_0 = lane_id / 4;
            int lane_row_1 = lane_row_0 + 8;
            int lane_col_0 = 2 * (lane_id % 4);
            int lane_col_1 = lane_col_0 + 1;

            int r0 = row_dst_0 + mma_row + lane_row_0;
            int r1 = row_dst_0 + mma_row + lane_row_1;
            int c0 = col_dst_0 + mma_col + lane_col_0;
            int c1 = col_dst_0 + mma_col + lane_col_1;

            // Output is column-major: dst[col * nrows_dst + row]
            if (r0 < nrows_dst && c0 < ncols_y) dst[c0 * nrows_dst + r0] = acc[tile_idx][0];
            if (r1 < nrows_dst && c0 < ncols_y) dst[c0 * nrows_dst + r1] = acc[tile_idx][1];
            if (r0 < nrows_dst && c1 < ncols_y) dst[c1 * nrows_dst + r0] = acc[tile_idx][2];
            if (r1 < nrows_dst && c1 < ncols_y) dst[c1 * nrows_dst + r1] = acc[tile_idx][3];
        }
    }
}

// ============================================================================
// Q6_K × Q8_1 MMA kernel
// ============================================================================

// Q6_K: 256 elements per block (210 bytes)
//   - ql[128]: low 4 bits of 6-bit quants
//   - qh[64]:  high 2 bits of 6-bit quants
//   - scales[16]: int8 block scales
//   - d (f16): super-block scale
//
// Dequantized value = d * scales[i/16] * (q6_val - 32)
// where q6_val = ql_low4 | (qh_2bits << 4), range [0, 63]
//
// For MMA i8: store (q6_val - 32) as int8, scale = d * scales[sb]

template<int mmq_y, int mmq_x>
__device__ void mul_mat_q6_K_mma(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst
) {
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tid = warp_id * WARP_SIZE + lane_id;

    const int row_dst_0 = blockIdx.x * mmq_y;
    const int col_dst_0 = blockIdx.y * mmq_x;

    const int blocks_per_row_x = ncols_x / QK_K;

    constexpr int n_mma_y = mmq_y / MMA_M;
    constexpr int n_mma_x = mmq_x / MMA_N;
    constexpr int tiles_per_warp_y = n_mma_y / NWARPS_MMQ;
    constexpr int tiles_per_warp = tiles_per_warp_y * n_mma_x;

    float acc[tiles_per_warp][4];
#pragma unroll
    for (int t = 0; t < tiles_per_warp; t++) {
#pragma unroll
        for (int i = 0; i < 4; i++) acc[t][i] = 0.0f;
    }

    constexpr int K_TILE = K_TILE_Q6_K;
    constexpr int k_q8_blocks = K_TILE / QK8_1;
    constexpr int n_sub_blocks = 16; // QK_K / 16 = 16 sub-blocks of 16 values

    extern __shared__ char shmem[];
    constexpr int w_qs_stride = K_TILE + 16;
    int8_t *tile_w_qs = (int8_t *)shmem;
    // Scales: 16 sub-blocks per row
    float  *tile_w_sc = (float *)(tile_w_qs + mmq_y * w_qs_stride);

    constexpr int a_qs_stride = K_TILE + 16;
    int8_t *tile_a_qs = (int8_t *)(tile_w_sc + mmq_y * n_sub_blocks);
    float  *tile_a_d  = (float *)(tile_a_qs + mmq_x * a_qs_stride);

    const block_q6_K *x = (const block_q6_K *)vx;
    const block_q8_1 *y = (const block_q8_1 *)vy;

    for (int k0 = 0; k0 < blocks_per_row_x; k0++) {

        // === Load and dequantize Q6_K weight tile ===
        {
            const int loads = (mmq_y + 128 - 1) / 128;
            for (int l = 0; l < loads; l++) {
                int row = tid + l * 128;
                if (row >= mmq_y) break;

                int abs_row = row_dst_0 + row;
                if (abs_row < nrows_x) {
                    const block_q6_K *blk = &x[abs_row * blocks_per_row_x + k0];
                    float d_super = __half2float(*reinterpret_cast<const __half *>(&blk->d));

                    // Store per-sub-block scales
                    for (int sb = 0; sb < n_sub_blocks; sb++) {
                        tile_w_sc[row * n_sub_blocks + sb] = d_super * (float)blk->scales[sb];
                    }

                    // Dequantize 6-bit values to int8 (q6_val - 32)
                    // Q6_K stores 256 values in groups of 128. Within each 128-value group:
                    //   values 0..31:   ql[l] low nibble  + qh[l] bits 0-1
                    //   values 32..63:  ql[32+l] low nibble + qh[l] bits 2-3
                    //   values 64..95:  ql[l] high nibble + qh[l] bits 4-5
                    //   values 96..127: ql[32+l] high nibble + qh[l] bits 6-7
                    // (l = 0..31 within each group)
                    int8_t *dst_qs = &tile_w_qs[row * w_qs_stride];
                    const uint8_t *ql = blk->ql;
                    const uint8_t *qh = blk->qh;

                    for (int n = 0; n < QK_K; n += 128) {
                        const uint8_t *ql_n = &ql[n / 2];
                        const uint8_t *qh_n = &qh[n / 4];
                        for (int l = 0; l < 32; l++) {
                            dst_qs[n + l +  0] = (int8_t)(((ql_n[l]      & 0xF) | (((qh_n[l] >> 0) & 3) << 4)) - 32);
                            dst_qs[n + l + 32] = (int8_t)(((ql_n[l + 32] & 0xF) | (((qh_n[l] >> 2) & 3) << 4)) - 32);
                            dst_qs[n + l + 64] = (int8_t)(((ql_n[l]      >> 4)  | (((qh_n[l] >> 4) & 3) << 4)) - 32);
                            dst_qs[n + l + 96] = (int8_t)(((ql_n[l + 32] >> 4)  | (((qh_n[l] >> 6) & 3) << 4)) - 32);
                        }
                    }
                } else {
                    for (int sb = 0; sb < n_sub_blocks; sb++) {
                        tile_w_sc[row * n_sub_blocks + sb] = 0.0f;
                    }
                    int8_t *dst_qs = &tile_w_qs[row * w_qs_stride];
                    for (int i = 0; i < K_TILE; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = 0;
                    }
                }
            }
        }

        // === Load activation tile (Q8_1) ===
        {
            const int blocks_per_col_y = nrows_y / QK8_1;
            const int loads = (mmq_x * k_q8_blocks + 128 - 1) / 128;
            for (int l = 0; l < loads; l++) {
                int idx = tid + l * 128;
                if (idx >= mmq_x * k_q8_blocks) break;

                int col = idx / k_q8_blocks;
                int kb  = idx % k_q8_blocks;
                int abs_col = col_dst_0 + col;

                if (abs_col < ncols_y) {
                    const block_q8_1 *blk = &y[abs_col * blocks_per_col_y + k0 * k_q8_blocks + kb];
                    tile_a_d[col * k_q8_blocks + kb] = __half2float(__low2half(blk->ds));

                    int8_t *dst_qs = &tile_a_qs[col * a_qs_stride + kb * QK8_1];
                    const int8_t *src_qs = blk->qs;
                    for (int i = 0; i < QK8_1; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = *reinterpret_cast<const int *>(src_qs + i);
                    }
                } else {
                    tile_a_d[col * k_q8_blocks + kb] = 0.0f;
                    int8_t *dst_qs = &tile_a_qs[col * a_qs_stride + kb * QK8_1];
                    for (int i = 0; i < QK8_1; i += 4) {
                        *reinterpret_cast<int *>(dst_qs + i) = 0;
                    }
                }
            }
        }

        __syncthreads();

        // === MMA computation ===
        const int warp_row_offset = warp_id * tiles_per_warp_y * MMA_M;

        for (int k_sub = 0; k_sub < K_TILE; k_sub += MMA_K) {
            int q8_block = k_sub / QK8_1;
            int sub_block = k_sub / 16; // Q6_K has 16-element sub-blocks

            for (int tile_y = 0; tile_y < tiles_per_warp_y; tile_y++) {
                int mma_row = warp_row_offset + tile_y * MMA_M;

                int a[2];
                {
                    int groupID = lane_id / 4;
                    int tid_in_group = lane_id % 4;
                    int col_offset = tid_in_group * 4;
                    a[0] = *reinterpret_cast<int *>(&tile_w_qs[(mma_row + groupID) * w_qs_stride + k_sub + col_offset]);
                    a[1] = *reinterpret_cast<int *>(&tile_w_qs[(mma_row + groupID + 8) * w_qs_stride + k_sub + col_offset]);
                }

                for (int tile_x = 0; tile_x < n_mma_x; tile_x++) {
                    int mma_col = tile_x * MMA_N;

                    int b[1];
                    {
                        int groupID = lane_id / 4;
                        int tid_in_group = lane_id % 4;
                        int col_abs = mma_col + groupID;
                        int k_offset = tid_in_group * 4;
                        b[0] = *reinterpret_cast<int *>(&tile_a_qs[col_abs * a_qs_stride + k_sub + k_offset]);
                    }

                    int c[4] = {0, 0, 0, 0};
                    int d[4];
                    mma_i8_m16n8k16(d, a, b, c);

                    int tile_idx = tile_y * n_mma_x + tile_x;

                    int lane_row_0 = lane_id / 4;
                    int lane_row_1 = lane_row_0 + 8;
                    int lane_col_0 = 2 * (lane_id % 4);
                    int lane_col_1 = lane_col_0 + 1;

                    // Q6_K: result = w_scale * a_d * MMA_dot
                    // where MMA_dot = sum((q6_val-32) * a_qs) over 16 elements
                    float w_sc_0 = tile_w_sc[(mma_row + lane_row_0) * n_sub_blocks + sub_block];
                    float w_sc_1 = tile_w_sc[(mma_row + lane_row_1) * n_sub_blocks + sub_block];
                    float a_d_0 = tile_a_d[(mma_col + lane_col_0) * k_q8_blocks + q8_block];
                    float a_d_1 = tile_a_d[(mma_col + lane_col_1) * k_q8_blocks + q8_block];

                    // d[1] and d[2] swapped: d[1]→(rows 0-7, odd col), d[2]→(rows 8-15, even col)
                    acc[tile_idx][0] += w_sc_0 * a_d_0 * (float)d[0];
                    acc[tile_idx][1] += w_sc_1 * a_d_0 * (float)d[2];
                    acc[tile_idx][2] += w_sc_0 * a_d_1 * (float)d[1];
                    acc[tile_idx][3] += w_sc_1 * a_d_1 * (float)d[3];
                }
            }
        }

        __syncthreads();
    }

    // === Write output ===
    const int warp_row_out = warp_id * tiles_per_warp_y * MMA_M;
    for (int tile_y = 0; tile_y < tiles_per_warp_y; tile_y++) {
        int mma_row = warp_row_out + tile_y * MMA_M;
        for (int tile_x = 0; tile_x < n_mma_x; tile_x++) {
            int mma_col = tile_x * MMA_N;
            int tile_idx = tile_y * n_mma_x + tile_x;

            int lane_row_0 = lane_id / 4;
            int lane_row_1 = lane_row_0 + 8;
            int lane_col_0 = 2 * (lane_id % 4);
            int lane_col_1 = lane_col_0 + 1;

            int r0 = row_dst_0 + mma_row + lane_row_0;
            int r1 = row_dst_0 + mma_row + lane_row_1;
            int c0 = col_dst_0 + mma_col + lane_col_0;
            int c1 = col_dst_0 + mma_col + lane_col_1;

            // Output is column-major: dst[col * nrows_dst + row]
            if (r0 < nrows_dst && c0 < ncols_y) dst[c0 * nrows_dst + r0] = acc[tile_idx][0];
            if (r1 < nrows_dst && c0 < ncols_y) dst[c0 * nrows_dst + r1] = acc[tile_idx][1];
            if (r0 < nrows_dst && c1 < ncols_y) dst[c1 * nrows_dst + r0] = acc[tile_idx][2];
            if (r1 < nrows_dst && c1 < ncols_y) dst[c1 * nrows_dst + r1] = acc[tile_idx][3];
        }
    }
}

// ============================================================================
// Kernel entry points (extern "C")
// ============================================================================

extern "C" __global__ void __launch_bounds__(NWARPS_MMQ * WARP_SIZE, 2)
mul_mat_q8_0_mma_kernel(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst
) {
    mul_mat_q8_0_mma<MMQ_Y, MMQ_X>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void __launch_bounds__(NWARPS_MMQ * WARP_SIZE, 2)
mul_mat_q4_K_mma_kernel(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst
) {
    mul_mat_q4_K_mma<MMQ_Y, MMQ_X>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void __launch_bounds__(NWARPS_MMQ * WARP_SIZE, 2)
mul_mat_q6_K_mma_kernel(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst
) {
    mul_mat_q6_K_mma<MMQ_Y, MMQ_X>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

// ============================================================================
// Launch wrappers (extern "C" for FFI)
// ============================================================================

static int calc_shmem_q8_0() {
    constexpr int w_qs = MMQ_Y * (K_TILE_Q8_0 + 16);
    constexpr int w_d  = MMQ_Y * (K_TILE_Q8_0 / QK8_0) * sizeof(float);
    constexpr int a_qs = MMQ_X * (K_TILE_Q8_0 + 16);
    constexpr int a_d  = MMQ_X * (K_TILE_Q8_0 / QK8_1) * sizeof(float);
    return w_qs + w_d + a_qs + a_d;
}

static int calc_shmem_q4_K() {
    constexpr int w_qs = MMQ_Y * (K_TILE_Q4_K + 16);
    constexpr int w_sc = MMQ_Y * 8 * sizeof(float);
    constexpr int w_mn = MMQ_Y * 8 * sizeof(float);
    constexpr int a_qs = MMQ_X * (K_TILE_Q4_K + 16);
    constexpr int a_d  = MMQ_X * (K_TILE_Q4_K / QK8_1) * sizeof(float);
    constexpr int a_s  = MMQ_X * (K_TILE_Q4_K / QK8_1) * sizeof(float);
    return w_qs + w_sc + w_mn + a_qs + a_d + a_s;
}

static int calc_shmem_q6_K() {
    constexpr int w_qs = MMQ_Y * (K_TILE_Q6_K + 16);
    constexpr int w_sc = MMQ_Y * 16 * sizeof(float);
    constexpr int a_qs = MMQ_X * (K_TILE_Q6_K + 16);
    constexpr int a_d  = MMQ_X * (K_TILE_Q6_K / QK8_1) * sizeof(float);
    return w_qs + w_sc + a_qs + a_d;
}

static bool mma_debug_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char *env = getenv("PARAMECIA_MMA_DEBUG");
        enabled = (env && env[0] == '1') ? 1 : 0;
    }
    return enabled == 1;
}

#define MMA_CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "MMA CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

extern "C" void mul_mat_q8_0_mma_launch(
    const void *vx, const void *vy, float *dst,
    int ncols_x, int nrows_x, int ncols_y, int nrows_y, int nrows_dst,
    cudaStream_t stream
) {
    const dim3 block(WARP_SIZE, NWARPS_MMQ, 1);
    const dim3 grid(
        (nrows_x + MMQ_Y - 1) / MMQ_Y,
        (ncols_y + MMQ_X - 1) / MMQ_X,
        1
    );
    int shmem = calc_shmem_q8_0();
    MMA_CHECK_CUDA(cudaFuncSetAttribute(mul_mat_q8_0_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    mul_mat_q8_0_mma_kernel<<<grid, block, shmem, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst
    );
    MMA_CHECK_CUDA(cudaGetLastError());
    if (mma_debug_enabled()) {
        MMA_CHECK_CUDA(cudaStreamSynchronize(stream));
        fprintf(stderr, "MMA Q8_0: grid(%d,%d) block(%d,%d) shmem=%d ncols_x=%d nrows_x=%d ncols_y=%d nrows_y=%d nrows_dst=%d\n",
                grid.x, grid.y, block.x, block.y, shmem, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

extern "C" void mul_mat_q4_K_mma_launch(
    const void *vx, const void *vy, float *dst,
    int ncols_x, int nrows_x, int ncols_y, int nrows_y, int nrows_dst,
    cudaStream_t stream
) {
    const dim3 block(WARP_SIZE, NWARPS_MMQ, 1);
    const dim3 grid(
        (nrows_x + MMQ_Y - 1) / MMQ_Y,
        (ncols_y + MMQ_X - 1) / MMQ_X,
        1
    );
    int shmem = calc_shmem_q4_K();
    MMA_CHECK_CUDA(cudaFuncSetAttribute(mul_mat_q4_K_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    mul_mat_q4_K_mma_kernel<<<grid, block, shmem, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst
    );
    MMA_CHECK_CUDA(cudaGetLastError());
    if (mma_debug_enabled()) {
        MMA_CHECK_CUDA(cudaStreamSynchronize(stream));
        fprintf(stderr, "MMA Q4_K: grid(%d,%d) block(%d,%d) shmem=%d ncols_x=%d nrows_x=%d ncols_y=%d nrows_y=%d nrows_dst=%d\n",
                grid.x, grid.y, block.x, block.y, shmem, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}

extern "C" void mul_mat_q6_K_mma_launch(
    const void *vx, const void *vy, float *dst,
    int ncols_x, int nrows_x, int ncols_y, int nrows_y, int nrows_dst,
    cudaStream_t stream
) {
    const dim3 block(WARP_SIZE, NWARPS_MMQ, 1);
    const dim3 grid(
        (nrows_x + MMQ_Y - 1) / MMQ_Y,
        (ncols_y + MMQ_X - 1) / MMQ_X,
        1
    );
    int shmem = calc_shmem_q6_K();
    MMA_CHECK_CUDA(cudaFuncSetAttribute(mul_mat_q6_K_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    mul_mat_q6_K_mma_kernel<<<grid, block, shmem, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst
    );
    MMA_CHECK_CUDA(cudaGetLastError());
    if (mma_debug_enabled()) {
        MMA_CHECK_CUDA(cudaStreamSynchronize(stream));
        fprintf(stderr, "MMA Q6_K: grid(%d,%d) block(%d,%d) shmem=%d ncols_x=%d nrows_x=%d ncols_y=%d nrows_y=%d nrows_dst=%d\n",
                grid.x, grid.y, block.x, block.y, shmem, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
    }
}
