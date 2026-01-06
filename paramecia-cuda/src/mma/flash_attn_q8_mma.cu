// MMA tensor core flash attention for Q8_0 quantized KV cache.
//
// Uses m16n8k16 f16×f16→f32 MMA for both Q×K^T and softmax×V products.
// Targets prefill (seq_q > 1) on Ampere+ (sm_80+).
//
// For single-token decode (seq_q=1), the existing DP4A kernel is preferred
// since decode is memory-bound and MMA doesn't help.
//
// Algorithm per block:
// 1. Load FA_NCOLS=16 query rows into shared memory (F16, pre-scaled)
// 2. For each KV tile of FA_KV_TILE=32 positions:
//    a. Dequantize Q8_0 K tile → F16 in shared memory
//    b. MMA Q×K^T → F32 attention scores (16×32)
//    c. Online softmax: update row max/sum, rescale output accumulator
//    d. Dequantize Q8_0 V tile → F16 in shared memory
//    e. MMA P×V → accumulate to output (16×D)
// 3. Normalize by softmax sum, write F16 output

#include "mma_utils.cuh"
#include <math.h>
#include <float.h>
#include <stdio.h>

// ============================================================================
// Configuration
// ============================================================================

#define FA_NCOLS   16   // Query rows per block (= MMA_M)
#define FA_KV_TILE 32   // KV positions per tile
#define FA_NWARPS  4    // Warps per block

// ============================================================================
// Pack two half values into a uint32
// ============================================================================

static __device__ __forceinline__ uint32_t pack_half2(half a, half b) {
    uint32_t result;
    unsigned short ha = __half_as_ushort(a);
    unsigned short hb = __half_as_ushort(b);
    asm("{mov.b32 %0, {%1, %2};}" : "=r"(result) : "h"(ha), "h"(hb));
    return result;
}

// ============================================================================
// MMA fragment loading helpers
// ============================================================================

// Load A operand (row-major 16×16 half) from shared memory.
// A is stored as A[row][col] at shmem[row * stride + col].
// Thread lane_id loads its fragment according to m16n8k16 A layout.
//
// Fragment layout (m16n8k16, A row-major, f16):
//   groupID = lane_id / 4   (0..7)
//   tid_in_group = lane_id % 4  (0..3)
//   a[0]: A[groupID][tid_in_group*2],     A[groupID][tid_in_group*2+1]
//   a[1]: A[groupID][tid_in_group*2+8],   A[groupID][tid_in_group*2+9]
//   a[2]: A[groupID+8][tid_in_group*2],   A[groupID+8][tid_in_group*2+1]
//   a[3]: A[groupID+8][tid_in_group*2+8], A[groupID+8][tid_in_group*2+9]
static __device__ __forceinline__ void load_A_f16(
    uint32_t a[4],
    const half *shmem, int stride, // A[row * stride + col]
    int lane_id
) {
    int groupID = lane_id / 4;
    int tid_in_group = lane_id % 4;
    int col_base = tid_in_group * 2;

    a[0] = *reinterpret_cast<const uint32_t *>(&shmem[groupID * stride + col_base]);
    a[1] = *reinterpret_cast<const uint32_t *>(&shmem[groupID * stride + col_base + 8]);
    a[2] = *reinterpret_cast<const uint32_t *>(&shmem[(groupID + 8) * stride + col_base]);
    a[3] = *reinterpret_cast<const uint32_t *>(&shmem[(groupID + 8) * stride + col_base + 8]);
}

// Load B operand (col-major 16×8 half) for Q × K^T.
// K is stored row-major in kv_tile[kv_pos * kv_stride + d].
// B_colmaj[k][n] = K^T[k][n] = K[n][k].
// n = kv position (columns of B), k = D position (rows of B).
//
// Fragment layout (m16n8k16, B col-major, f16):
//   groupID = lane_id / 4   (0..7) → n column
//   tid_in_group = lane_id % 4  (0..3) → k row group
//   b[0]: B[tid_in_group*2][groupID],   B[tid_in_group*2+1][groupID]
//   b[1]: B[tid_in_group*2+8][groupID], B[tid_in_group*2+9][groupID]
static __device__ __forceinline__ void load_B_KT_f16(
    uint32_t b[2],
    const half *kv_tile, int kv_stride,
    int kv_n_base,  // kv position base for this MMA N-tile
    int d_k_base,   // D position base for this MMA K-step
    int lane_id
) {
    int groupID = lane_id / 4;       // 0..7: which kv position (N column)
    int tid_in_group = lane_id % 4;  // 0..3: which k-row group

    int kv_pos = kv_n_base + groupID;
    int k0 = d_k_base + tid_in_group * 2;
    int k1 = k0 + 8;

    // B[k][n] = K^T[k][n] = K[n][k] = kv_tile[n * kv_stride + k]
    // Read 2 consecutive D values for b[0], and 2 more offset by 8 for b[1]
    b[0] = *reinterpret_cast<const uint32_t *>(&kv_tile[kv_pos * kv_stride + k0]);
    b[1] = *reinterpret_cast<const uint32_t *>(&kv_tile[kv_pos * kv_stride + k1]);
}

// Load B operand (col-major 16×8 half) for P × V.
// V is stored row-major in kv_tile[kv_pos * kv_stride + d].
// B_colmaj[k][n] = V[k][n].
// k = kv position (rows of B, reduction dim), n = D position (cols of B).
//
// Fragment layout (m16n8k16, B col-major, f16):
//   groupID = lane_id / 4   (0..7) → n column (D position)
//   tid_in_group = lane_id % 4  (0..3) → k row group
//   b[0]: V[kv_k_base + tid_in_group*2][d_n_base + groupID],
//         V[kv_k_base + tid_in_group*2+1][d_n_base + groupID]
//   b[1]: V[kv_k_base + tid_in_group*2+8][d_n_base + groupID],
//         V[kv_k_base + tid_in_group*2+9][d_n_base + groupID]
static __device__ __forceinline__ void load_B_V_f16(
    uint32_t b[2],
    const half *kv_tile, int kv_stride,
    int kv_k_base,  // kv position base for this MMA K-step
    int d_n_base,   // D position base for this MMA N-tile
    int lane_id
) {
    int groupID = lane_id / 4;       // 0..7: which D position (N column)
    int tid_in_group = lane_id % 4;  // 0..3: which kv-row group

    int d_pos = d_n_base + groupID;
    int k0 = kv_k_base + tid_in_group * 2;
    int k1 = k0 + 8;

    // B[k][n] = V[k][n] = kv_tile[k * kv_stride + n]
    // Strided reads across KV rows
    half h0 = kv_tile[k0 * kv_stride + d_pos];
    half h1 = kv_tile[(k0 + 1) * kv_stride + d_pos];
    half h2 = kv_tile[k1 * kv_stride + d_pos];
    half h3 = kv_tile[(k1 + 1) * kv_stride + d_pos];
    b[0] = pack_half2(h0, h1);
    b[1] = pack_half2(h2, h3);
}

// ============================================================================
// MMA C/D output element mapping
// ============================================================================

// For m16n8k16 f32 accumulator, thread lane_id's d[0..3] map to:
//   d[0] → row = lane/4,     col = 2*(lane%4)       (rows 0-7, even col)
//   d[1] → row = lane/4,     col = 2*(lane%4) + 1   (rows 0-7, odd col)
//   d[2] → row = lane/4 + 8, col = 2*(lane%4)       (rows 8-15, even col)
//   d[3] → row = lane/4 + 8, col = 2*(lane%4) + 1   (rows 8-15, odd col)

static __device__ __forceinline__ void mma_output_indices(
    int lane_id,
    int &r0, int &r1, int &c0, int &c1
) {
    r0 = lane_id / 4;
    r1 = r0 + 8;
    c0 = 2 * (lane_id % 4);
    c1 = c0 + 1;
}

// ============================================================================
// Flash attention kernel
// ============================================================================

template<int D>
__device__ void flash_attn_q8_mma_impl(
    const half * __restrict__ Q,
    const void * __restrict__ K,
    const void * __restrict__ V,
    half * __restrict__ output,
    const float scale,
    const uint32_t b,
    const uint32_t h,
    const uint32_t h_k,
    const uint32_t seq_q,
    const uint32_t seq_k,
    const uint32_t q_offset,
    const bool causal,
    const uint64_t q_stride_b, const uint64_t q_stride_seq,
    const uint64_t q_stride_h, const uint64_t q_stride_d,
    const uint64_t k_stride_b, const uint64_t k_stride_seq,
    const uint64_t k_stride_h,
    const uint64_t v_stride_b, const uint64_t v_stride_seq,
    const uint64_t v_stride_h,
    const uint64_t o_stride_b, const uint64_t o_stride_seq,
    const uint64_t o_stride_h, const uint64_t o_stride_d
) {
    const uint32_t ib = blockIdx.x;
    const uint32_t ih = blockIdx.y;
    const uint32_t iq_base = blockIdx.z * FA_NCOLS;

    if (ib >= b || ih >= h) return;

    const uint32_t lane_id = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t tid = warp_id * WARP_SIZE + lane_id;
    const uint32_t nthreads = FA_NWARPS * WARP_SIZE;

    const uint32_t kv_head = (h_k == h) ? ih : (ih * h_k) / h;

    // ================================================================
    // Shared memory layout
    // ================================================================
    extern __shared__ char shmem[];

    half  *q_tile    = (half *)shmem;                               // [FA_NCOLS][D]
    half  *kv_tile   = q_tile + FA_NCOLS * D;                       // [FA_KV_TILE][D]
    float *scores    = (float *)(kv_tile + FA_KV_TILE * D);         // [FA_NCOLS][FA_KV_TILE]
    float *row_max   = scores + FA_NCOLS * FA_KV_TILE;              // [FA_NCOLS]
    float *row_sum   = row_max + FA_NCOLS;                          // [FA_NCOLS]
    float *row_scale = row_sum + FA_NCOLS;                          // [FA_NCOLS] rescale factors
    half  *scores_h  = (half *)(row_scale + FA_NCOLS);              // [FA_NCOLS][FA_KV_TILE]
    float *out_acc   = (float *)(scores_h + FA_NCOLS * FA_KV_TILE); // [FA_NCOLS][D]

    // ================================================================
    // Step 1: Load Q tile and initialize state
    // ================================================================
    for (int idx = tid; idx < FA_NCOLS * D; idx += nthreads) {
        int qr = idx / D;
        int qd = idx % D;
        uint32_t abs_q_row = iq_base + qr;
        if (abs_q_row < seq_q) {
            const half *q_row = Q + ib * q_stride_b + abs_q_row * q_stride_seq + ih * q_stride_h;
            float qv = __half2float(q_row[qd * q_stride_d]) * scale;
            q_tile[qr * D + qd] = __float2half_rn(qv);
        } else {
            q_tile[qr * D + qd] = __float2half_rn(0.0f);
        }
        out_acc[idx] = 0.0f;
    }

    if (tid < FA_NCOLS) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }

    __syncthreads();

    // ================================================================
    // Step 2: KV tile loop
    // ================================================================
    for (uint32_t kv_start = 0; kv_start < seq_k; kv_start += FA_KV_TILE) {

        // ---- 2a: Load and dequantize K tile → kv_tile ----
        for (int idx = tid; idx < FA_KV_TILE * D; idx += nthreads) {
            int kr = idx / D;
            int kd = idx % D;
            uint32_t abs_k = kv_start + kr;
            if (abs_k < seq_k) {
                const char *k_row_ptr = reinterpret_cast<const char *>(K)
                    + (uint64_t)ib * k_stride_b
                    + (uint64_t)abs_k * k_stride_seq
                    + (uint64_t)kv_head * k_stride_h;
                const block_q8_0 *k_blk = reinterpret_cast<const block_q8_0 *>(k_row_ptr);
                int bi = kd / QK8_0;
                int bo = kd % QK8_0;
                float d = __half2float(*reinterpret_cast<const __half *>(&k_blk[bi].d));
                kv_tile[kr * D + kd] = __float2half_rn(d * (float)k_blk[bi].qs[bo]);
            } else {
                kv_tile[kr * D + kd] = __float2half_rn(0.0f);
            }
        }
        __syncthreads();

        // ---- 2b: Compute S = Q × K^T via MMA ----
        // S[16×32] = Q[16×D] × K^T[D×32]
        // MMA tiles: 1 M-tile × 4 N-tiles (32/8=4), each warp handles 1 N-tile
        // Reduce over D in steps of MMA_K=16

        constexpr int n_score_tiles = FA_KV_TILE / MMA_N; // 4
        if (warp_id < n_score_tiles) {
            int kv_n_base = warp_id * MMA_N;

            // MMA accumulator for this 16×8 tile
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int dk = 0; dk < D; dk += MMA_K) {
                uint32_t a[4];
                load_A_f16(a, &q_tile[dk], D, lane_id);

                uint32_t b_reg[2];
                load_B_KT_f16(b_reg, kv_tile, D, kv_n_base, dk, lane_id);

                float d_out[4];
                mma_f16_m16n8k16(d_out, a, b_reg, c);
                c[0] = d_out[0]; c[1] = d_out[1];
                c[2] = d_out[2]; c[3] = d_out[3];
            }

            // Write scores to shared memory
            int r0, r1, c0, c1;
            mma_output_indices(lane_id, r0, r1, c0, c1);
            c0 += kv_n_base;
            c1 += kv_n_base;
            // d[0]→(r0,c0), d[1]→(r0,c1), d[2]→(r1,c0), d[3]→(r1,c1)
            scores[r0 * FA_KV_TILE + c0] = c[0];
            scores[r0 * FA_KV_TILE + c1] = c[1];
            scores[r1 * FA_KV_TILE + c0] = c[2];
            scores[r1 * FA_KV_TILE + c1] = c[3];
        }
        __syncthreads();

        // ---- 2c: Online softmax + rescale output accumulator ----
        // Process each query row. With 128 threads and 16 rows, most threads
        // cooperate on rescaling out_acc while a few handle softmax per row.

        // Step c1: Compute new row max and rescale factor
        // Each thread processes ceil(FA_NCOLS / nthreads) rows for softmax
        // (With nthreads=128 and FA_NCOLS=16, only first 16 threads)
        {
            // Small enough that single-thread-per-row is fine
            if (tid < FA_NCOLS) {
                int qr = tid;
                uint32_t abs_q = q_offset + iq_base + qr;
                float old_max = row_max[qr];
                float new_max = old_max;

                // Apply causal mask and find new max
                int effective_kv = min((int)FA_KV_TILE, (int)(seq_k - kv_start));
                for (int kc = 0; kc < effective_kv; kc++) {
                    if (causal && ((kv_start + kc) > abs_q)) {
                        scores[qr * FA_KV_TILE + kc] = -INFINITY;
                    } else {
                        new_max = fmaxf(new_max, scores[qr * FA_KV_TILE + kc]);
                    }
                }
                for (int kc = effective_kv; kc < FA_KV_TILE; kc++) {
                    scores[qr * FA_KV_TILE + kc] = -INFINITY;
                }

                float rescale = (isfinite(old_max) && old_max != new_max)
                    ? expf(old_max - new_max)
                    : (isfinite(old_max) ? 1.0f : 0.0f);

                // Compute softmax weights and tile sum
                float tile_sum = 0.0f;
                for (int kc = 0; kc < FA_KV_TILE; kc++) {
                    float s = scores[qr * FA_KV_TILE + kc];
                    float w = isfinite(s) ? expf(s - new_max) : 0.0f;
                    scores[qr * FA_KV_TILE + kc] = w;
                    tile_sum += w;
                }

                row_max[qr] = new_max;
                row_sum[qr] = row_sum[qr] * rescale + tile_sum;

                // Store rescale factor for cooperative out_acc update
                row_scale[qr] = rescale;
            }
        }
        __syncthreads();

        // Step c2: Cooperatively rescale out_acc
        for (int idx = tid; idx < FA_NCOLS * D; idx += nthreads) {
            int qr = idx / D;
            out_acc[idx] *= row_scale[qr];
        }
        __syncthreads();

        // ---- 2d: Convert scores (softmax weights) to half ----
        for (int idx = tid; idx < FA_NCOLS * FA_KV_TILE; idx += nthreads) {
            scores_h[idx] = __float2half_rn(scores[idx]);
        }

        // ---- 2e: Load and dequantize V tile → kv_tile ----
        for (int idx = tid; idx < FA_KV_TILE * D; idx += nthreads) {
            int vr = idx / D;
            int vd = idx % D;
            uint32_t abs_v = kv_start + vr;
            if (abs_v < seq_k) {
                const char *v_row_ptr = reinterpret_cast<const char *>(V)
                    + (uint64_t)ib * v_stride_b
                    + (uint64_t)abs_v * v_stride_seq
                    + (uint64_t)kv_head * v_stride_h;
                const block_q8_0 *v_blk = reinterpret_cast<const block_q8_0 *>(v_row_ptr);
                int bi = vd / QK8_0;
                int bo = vd % QK8_0;
                float d = __half2float(*reinterpret_cast<const __half *>(&v_blk[bi].d));
                kv_tile[vr * D + vd] = __float2half_rn(d * (float)v_blk[bi].qs[bo]);
            } else {
                kv_tile[vr * D + vd] = __float2half_rn(0.0f);
            }
        }
        __syncthreads();

        // ---- 2f: Accumulate O += P × V via MMA ----
        // P[16×32] × V[32×D] = O[16×D]
        // MMA: A=P[16×16], B=V[16×8]
        // Iterate: K reduction over FA_KV_TILE=32 (2 steps of 16)
        //          N tiles over D/8
        // Each warp handles D/(8*FA_NWARPS) N-tiles

        constexpr int n_out_tiles = D / MMA_N;  // D/8
        constexpr int tiles_per_warp = (n_out_tiles + FA_NWARPS - 1) / FA_NWARPS;

        for (int t = 0; t < tiles_per_warp; t++) {
            int tile_idx = warp_id * tiles_per_warp + t;
            if (tile_idx >= n_out_tiles) break;

            int d_n_base = tile_idx * MMA_N;

            // MMA accumulator
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            // Reduce over FA_KV_TILE in steps of MMA_K=16
            for (int kk = 0; kk < FA_KV_TILE; kk += MMA_K) {
                // A operand: P[0..15][kk..kk+15] from scores_h
                uint32_t a[4];
                load_A_f16(a, &scores_h[kk], FA_KV_TILE, lane_id);

                // B operand: V[kk..kk+15][d_n_base..d_n_base+7]
                uint32_t b_reg[2];
                load_B_V_f16(b_reg, kv_tile, D, kk, d_n_base, lane_id);

                float d_out[4];
                mma_f16_m16n8k16(d_out, a, b_reg, c);
                c[0] = d_out[0]; c[1] = d_out[1];
                c[2] = d_out[2]; c[3] = d_out[3];
            }

            // Add MMA result to out_acc
            int r0, r1, c0_off, c1_off;
            mma_output_indices(lane_id, r0, r1, c0_off, c1_off);
            int d0 = d_n_base + c0_off;
            int d1 = d_n_base + c1_off;

            // d[0]→(r0,d0), d[1]→(r0,d1), d[2]→(r1,d0), d[3]→(r1,d1)
            out_acc[r0 * D + d0] += c[0];
            out_acc[r0 * D + d1] += c[1];
            out_acc[r1 * D + d0] += c[2];
            out_acc[r1 * D + d1] += c[3];
        }

        __syncthreads();
    }

    // ================================================================
    // Step 3: Normalize and write output
    // ================================================================
    for (int idx = tid; idx < FA_NCOLS * D; idx += nthreads) {
        int qr = idx / D;
        int qd = idx % D;
        uint32_t abs_q_row = iq_base + qr;
        if (abs_q_row < seq_q) {
            float inv_sum = (row_sum[qr] > 0.0f) ? 1.0f / row_sum[qr] : 0.0f;
            float val = out_acc[idx] * inv_sum;
            half *out_row = output + ib * o_stride_b + abs_q_row * o_stride_seq + ih * o_stride_h;
            out_row[qd * o_stride_d] = __float2half_rn(val);
        }
    }
}

// ============================================================================
// Kernel entry points
// ============================================================================

extern "C" __global__ void __launch_bounds__(FA_NWARPS * WARP_SIZE, 1)
flash_attn_q8_mma_kernel_64(
    const half *Q, const void *K, const void *V, half *Out,
    const float scale,
    const uint32_t b, const uint32_t h, const uint32_t h_k, const uint32_t d,
    const uint32_t seq_q, const uint32_t seq_k, const uint32_t q_offset, const int causal,
    const uint64_t q_stride_b, const uint64_t q_stride_seq, const uint64_t q_stride_h, const uint64_t q_stride_d,
    const uint64_t k_stride_b, const uint64_t k_stride_seq, const uint64_t k_stride_h, const uint64_t k_stride_d,
    const uint64_t v_stride_b, const uint64_t v_stride_seq, const uint64_t v_stride_h, const uint64_t v_stride_d,
    const uint64_t o_stride_b, const uint64_t o_stride_seq, const uint64_t o_stride_h, const uint64_t o_stride_d
) {
    flash_attn_q8_mma_impl<64>(
        Q, K, V, Out, scale, b, h, h_k, seq_q, seq_k, q_offset, causal != 0,
        q_stride_b, q_stride_seq, q_stride_h, q_stride_d,
        k_stride_b, k_stride_seq, k_stride_h,
        v_stride_b, v_stride_seq, v_stride_h,
        o_stride_b, o_stride_seq, o_stride_h, o_stride_d
    );
}

extern "C" __global__ void __launch_bounds__(FA_NWARPS * WARP_SIZE, 1)
flash_attn_q8_mma_kernel_128(
    const half *Q, const void *K, const void *V, half *Out,
    const float scale,
    const uint32_t b, const uint32_t h, const uint32_t h_k, const uint32_t d,
    const uint32_t seq_q, const uint32_t seq_k, const uint32_t q_offset, const int causal,
    const uint64_t q_stride_b, const uint64_t q_stride_seq, const uint64_t q_stride_h, const uint64_t q_stride_d,
    const uint64_t k_stride_b, const uint64_t k_stride_seq, const uint64_t k_stride_h, const uint64_t k_stride_d,
    const uint64_t v_stride_b, const uint64_t v_stride_seq, const uint64_t v_stride_h, const uint64_t v_stride_d,
    const uint64_t o_stride_b, const uint64_t o_stride_seq, const uint64_t o_stride_h, const uint64_t o_stride_d
) {
    flash_attn_q8_mma_impl<128>(
        Q, K, V, Out, scale, b, h, h_k, seq_q, seq_k, q_offset, causal != 0,
        q_stride_b, q_stride_seq, q_stride_h, q_stride_d,
        k_stride_b, k_stride_seq, k_stride_h,
        v_stride_b, v_stride_seq, v_stride_h,
        o_stride_b, o_stride_seq, o_stride_h, o_stride_d
    );
}

extern "C" __global__ void __launch_bounds__(FA_NWARPS * WARP_SIZE, 1)
flash_attn_q8_mma_kernel_256(
    const half *Q, const void *K, const void *V, half *Out,
    const float scale,
    const uint32_t b, const uint32_t h, const uint32_t h_k, const uint32_t d,
    const uint32_t seq_q, const uint32_t seq_k, const uint32_t q_offset, const int causal,
    const uint64_t q_stride_b, const uint64_t q_stride_seq, const uint64_t q_stride_h, const uint64_t q_stride_d,
    const uint64_t k_stride_b, const uint64_t k_stride_seq, const uint64_t k_stride_h, const uint64_t k_stride_d,
    const uint64_t v_stride_b, const uint64_t v_stride_seq, const uint64_t v_stride_h, const uint64_t v_stride_d,
    const uint64_t o_stride_b, const uint64_t o_stride_seq, const uint64_t o_stride_h, const uint64_t o_stride_d
) {
    flash_attn_q8_mma_impl<256>(
        Q, K, V, Out, scale, b, h, h_k, seq_q, seq_k, q_offset, causal != 0,
        q_stride_b, q_stride_seq, q_stride_h, q_stride_d,
        k_stride_b, k_stride_seq, k_stride_h,
        v_stride_b, v_stride_seq, v_stride_h,
        o_stride_b, o_stride_seq, o_stride_h, o_stride_d
    );
}

// ============================================================================
// Launch wrapper
// ============================================================================

static int calc_shmem_flash_attn(int d) {
    int q       = FA_NCOLS * d * sizeof(half);
    int kv      = FA_KV_TILE * d * sizeof(half);
    int sc_f    = FA_NCOLS * FA_KV_TILE * sizeof(float);
    int rowstate = FA_NCOLS * 3 * sizeof(float);  // row_max + row_sum + row_scale
    int sc_h    = FA_NCOLS * FA_KV_TILE * sizeof(half);
    int out     = FA_NCOLS * d * sizeof(float);
    return q + kv + sc_f + rowstate + sc_h + out;
}

extern "C" void flash_attn_q8_mma_launch(
    const void *Q, const void *K, const void *V, void *Out,
    float scale,
    int b, int h, int h_k, int d, int seq_q, int seq_k,
    int q_offset, int causal,
    uint64_t q_stride_b, uint64_t q_stride_seq, uint64_t q_stride_h, uint64_t q_stride_d,
    uint64_t k_stride_b, uint64_t k_stride_seq, uint64_t k_stride_h, uint64_t k_stride_d,
    uint64_t v_stride_b, uint64_t v_stride_seq, uint64_t v_stride_h, uint64_t v_stride_d,
    uint64_t o_stride_b, uint64_t o_stride_seq, uint64_t o_stride_h, uint64_t o_stride_d,
    cudaStream_t stream
) {
    const dim3 block(WARP_SIZE, FA_NWARPS, 1);
    const dim3 grid(b, h, (seq_q + FA_NCOLS - 1) / FA_NCOLS);

    int shmem = calc_shmem_flash_attn(d);

    void (*kernel)(
        const half*, const void*, const void*, half*,
        float, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, int,
        uint64_t, uint64_t, uint64_t, uint64_t,
        uint64_t, uint64_t, uint64_t, uint64_t,
        uint64_t, uint64_t, uint64_t, uint64_t,
        uint64_t, uint64_t, uint64_t, uint64_t
    ) = nullptr;

    switch (d) {
        case 64:  kernel = flash_attn_q8_mma_kernel_64;  break;
        case 128: kernel = flash_attn_q8_mma_kernel_128; break;
        case 256: kernel = flash_attn_q8_mma_kernel_256; break;
        default: return;
    }

    cudaError_t attr_err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q8_mma: cudaFuncSetAttribute failed: %s (shmem=%d)\n",
                cudaGetErrorString(attr_err), shmem);
        return;
    }

    kernel<<<grid, block, shmem, stream>>>(
        (const half *)Q, K, V, (half *)Out,
        scale, b, h, h_k, d, seq_q, seq_k, q_offset, causal,
        q_stride_b, q_stride_seq, q_stride_h, q_stride_d,
        k_stride_b, k_stride_seq, k_stride_h, k_stride_d,
        v_stride_b, v_stride_seq, v_stride_h, v_stride_d,
        o_stride_b, o_stride_seq, o_stride_h, o_stride_d
    );

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q8_mma: kernel launch failed: %s "
                "(grid=(%d,%d,%d) block=(%d,%d,%d) shmem=%d d=%d)\n",
                cudaGetErrorString(launch_err),
                grid.x, grid.y, grid.z, block.x, block.y, block.z, shmem, d);
        return;
    }
}
