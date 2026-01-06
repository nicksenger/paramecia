// Q8_0 flash attention kernel (fast path).
//
// Key ideas adapted from llama.cpp's "fattn-vec" kernels:
// - Quantize Q on-the-fly to q8_1 (scale-only) and use dp4a for K·Q.
// - Use multiple warps per block and tile over the KV sequence.
// - Online softmax (max/sum rescaling) to avoid extra passes.
//
// Supported head dims: 64, 128, 256.
// Launch config: blockDim = (32 lanes, 4 warps, 1), gridDim = (b, h, seq_q).

#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#define WARP_SIZE 32
#define QK8_0 32
#define QI8_0 (QK8_0 / (int)sizeof(int))  // 8 packed int32 per 32 values
#define MIN_CC_DP4A 610

struct block_q8_0 {
    uint16_t d;      // f16 scale
    int8_t qs[QK8_0];
};

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, mask, WARP_SIZE);
    }
    return v;
}

static __device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, mask, WARP_SIZE));
    }
    return v;
}

// `qs` in block_q8_0 is only 2-byte aligned (due to the leading f16 scale).
// Load 4 consecutive int8 as a packed int32 using two 16-bit loads (requires 2-byte alignment).
static __device__ __forceinline__ int load_i32_aligned2(const int8_t * __restrict__ p) {
    const uint16_t *p16 = reinterpret_cast<const uint16_t *>(p);
    const uint32_t lo = (uint32_t) p16[0];
    const uint32_t hi = (uint32_t) p16[1];
    return (int) (lo | (hi << 16));
}

template <int D>
static __device__ __forceinline__ void quantize_q_to_q8_1_scale_only(
    const half * __restrict__ q_row,
    const float scale,
    const uint64_t q_stride_d,
    int * __restrict__ q_q8,
    float * __restrict__ q_d
) {
    // This is modeled after llama.cpp's quantize_q8_1_to_shared (scale-only usage).
    const int lane = (int) threadIdx.x;
    constexpr int ints_per_row = D / (int)sizeof(int); // D/4
    constexpr int nthreads_quantize = (ints_per_row < WARP_SIZE ? ints_per_row : WARP_SIZE);

#pragma unroll
    for (int i0 = 0; i0 < ints_per_row; i0 += nthreads_quantize) {
        float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (lane < nthreads_quantize) {
            const int base = 4*(i0 + lane);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float qv = __half2float(q_row[(base + l) * (int)q_stride_d]);
                vals[l] = scale * qv;
            }
        }

        float amax = fabsf(vals[0]);
#pragma unroll
        for (int l = 1; l < 4; ++l) {
            amax = fmaxf(amax, fabsf(vals[l]));
        }

        // Reduce within 8-lane groups (QI8_0 == 8) so each group quantizes one 32-value block.
#pragma unroll
        for (int mask = QI8_0/2; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const float d = amax / 127.0f;
        int q32 = 0;
        int8_t *q8 = (int8_t *) &q32;
        if (lane < nthreads_quantize) {
            if (d != 0.0f) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    q8[l] = (int8_t) nearbyintf(vals[l] / d);
                }
            } else {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    q8[l] = 0;
                }
            }

            q_q8[i0 + lane] = q32;
            if ((lane % QI8_0) == 0) {
                q_d[(i0 + lane) / QI8_0] = d;
            }
        }
    }
}

template <int D, int NWARPS>
__device__ void flash_attn_q8_kernel_fast(
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
    // Q strides (elements)
    const uint64_t q_stride_b, const uint64_t q_stride_seq, const uint64_t q_stride_h, const uint64_t q_stride_d,
    // K/V strides (bytes; rows are packed Q8_0 blocks)
    const uint64_t k_stride_b, const uint64_t k_stride_seq, const uint64_t k_stride_h, const uint64_t /*k_stride_d_unused*/,
    const uint64_t v_stride_b, const uint64_t v_stride_seq, const uint64_t v_stride_h, const uint64_t /*v_stride_d_unused*/,
    // Output strides (elements)
    const uint64_t o_stride_b, const uint64_t o_stride_seq, const uint64_t o_stride_h, const uint64_t o_stride_d
) {
    const uint32_t ib = blockIdx.x;
    const uint32_t ih = blockIdx.y;
    const uint32_t iq_seq = blockIdx.z;
    if (ib >= b || ih >= h || iq_seq >= seq_q) return;

    const uint32_t lane = (uint32_t) threadIdx.x;
    const uint32_t warp = (uint32_t) threadIdx.y;
    const uint32_t tid = warp * WARP_SIZE + lane;

    const uint32_t kv_head = (h_k == h) ? ih : (ih * h_k) / h;

    const half *q_row = Q + ib * q_stride_b + iq_seq * q_stride_seq + ih * q_stride_h;
    half *out_row = output + ib * o_stride_b + iq_seq * o_stride_seq + ih * o_stride_h;

    constexpr int ints_per_row = D / (int)sizeof(int);
    constexpr int blocks_per_row = D / QK8_0;
    constexpr int nthreads_V = (D/4 < WARP_SIZE ? D/4 : WARP_SIZE);
    constexpr int acc_per_lane = (D/2) / nthreads_V; // number of float2 per lane

    __shared__ int   q_q8[ints_per_row];
    __shared__ float q_d[blocks_per_row];

    if (warp == 0) {
        quantize_q_to_q8_1_scale_only<D>(q_row, scale, q_stride_d, q_q8, q_d);
    }
    __syncthreads();

    // Online softmax state.
    float kq_max = -INFINITY;
    float kq_sum = 0.0f;

    float2 acc[acc_per_lane];
#pragma unroll
    for (int i = 0; i < acc_per_lane; ++i) {
        acc[i] = make_float2(0.0f, 0.0f);
    }

    __shared__ float tile_max_s;
    __shared__ float tile_sum_s;
    __shared__ float warp_red_s[NWARPS];
    __shared__ float kq_w[NWARPS * WARP_SIZE];
    __shared__ float2 acc_s[NWARPS][WARP_SIZE][acc_per_lane];

    const uint32_t q_abs = q_offset + iq_seq;

    constexpr uint32_t keys_per_tile = NWARPS * WARP_SIZE;
    for (uint32_t tile0 = 0; tile0 < seq_k; tile0 += keys_per_tile) {
        // Each thread will end up holding the score for its "lane key":
        // key = tile0 + tid (one key per thread), but scores are computed with collaborative dot products.
        float score_reg = -INFINITY;

        float warp_max = -INFINITY;

        // Each warp computes 32 keys for this tile; for each key, the whole warp participates
        // in the dp4a dot-product reduction.
#pragma unroll
        for (int i_key = 0; i_key < (int)WARP_SIZE; ++i_key) {
            const uint32_t k_idx = tile0 + warp * WARP_SIZE + (uint32_t)i_key;
            float score = -INFINITY;
            const bool key_in_range = k_idx < seq_k;
            const bool key_allowed = !causal || (k_idx <= q_abs);
            if (key_in_range && key_allowed) {
                const char *k_row = reinterpret_cast<const char *>(K) + (uint64_t)ib * k_stride_b + (uint64_t)k_idx * k_stride_seq + (uint64_t)kv_head * k_stride_h;
                const block_q8_0 *k_blocks = reinterpret_cast<const block_q8_0 *>(k_row);

                float sum = 0.0f;
#pragma unroll
                for (int k_int_0 = 0; k_int_0 < ints_per_row; k_int_0 += WARP_SIZE) {
                    const int k_int = k_int_0 + (int)lane;
                    if (k_int < ints_per_row) {
                        const int iblk = k_int / QI8_0;
                        const int iqs  = k_int % QI8_0;
                        const int v = load_i32_aligned2(k_blocks[iblk].qs + 4*iqs);
                        const int u = q_q8[k_int];
                        const float d0 = __half2float(*reinterpret_cast<const __half *>(&k_blocks[iblk].d));
                        const float d1 = q_d[iblk];
                        sum += (d0 * d1) * (float) ggml_cuda_dp4a(v, u, 0);
                    }
                }

                sum = warp_reduce_sum(sum);
                score = sum;
            }

            warp_max = fmaxf(warp_max, score);
            if (lane == (uint32_t)i_key) {
                score_reg = score;
            }
        }

        warp_max = warp_reduce_max(warp_max);
        if (lane == 0) {
            warp_red_s[warp] = warp_max;
        }
        __syncthreads();

        if (warp == 0 && lane == 0) {
            float tm = warp_red_s[0];
#pragma unroll
            for (int w = 1; w < NWARPS; ++w) {
                tm = fmaxf(tm, warp_red_s[w]);
            }
            tile_max_s = tm;
        }
        __syncthreads();

        const float kq_max_new = fmaxf(kq_max, tile_max_s);
        const float kq_scale_old = isfinite(kq_max) ? expf(kq_max - kq_max_new) : 0.0f;

        kq_sum *= kq_scale_old;
#pragma unroll
        for (int i = 0; i < acc_per_lane; ++i) {
            acc[i].x *= kq_scale_old;
            acc[i].y *= kq_scale_old;
        }
        kq_max = kq_max_new;

        // Compute exp(score - kq_max) for this thread's key.
        float w = 0.0f;
        if (isfinite(score_reg)) {
            w = expf(score_reg - kq_max);
        }
        kq_w[tid] = w;

        float warp_sum = warp_reduce_sum(w);
        if (lane == 0) {
            warp_red_s[warp] = warp_sum;
        }
        __syncthreads();

        if (warp == 0 && lane == 0) {
            float ts = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                ts += warp_red_s[w];
            }
            tile_sum_s = ts;
        }
        __syncthreads();

        kq_sum += tile_sum_s;

        // Accumulate V * w for all keys in this tile.
        // Dequantize 4 V values per lane (2 float2).
#pragma unroll
        for (int k0 = 0; k0 < (int)WARP_SIZE; ++k0) {
            const uint32_t k_idx = tile0 + warp * WARP_SIZE + (uint32_t)k0;
            const float wk = kq_w[warp * WARP_SIZE + (uint32_t)k0];
            if (wk == 0.0f) {
                continue;
            }
            if (k_idx >= seq_k) {
                continue;
            }

            const char *v_row = reinterpret_cast<const char *>(V) + (uint64_t)ib * v_stride_b + (uint64_t)k_idx * v_stride_seq + (uint64_t)kv_head * v_stride_h;
            const block_q8_0 *v_blocks = reinterpret_cast<const block_q8_0 *>(v_row);

            if ((int)lane >= nthreads_V) {
                continue;
            }

#pragma unroll
            for (int base_el = 0; base_el < D; base_el += nthreads_V * 4) {
                const int i_el = base_el + (int)lane * 4;

                const int iblk = i_el / QK8_0;
                const int iqs  = i_el % QK8_0;

                const float d = __half2float(*reinterpret_cast<const __half *>(&v_blocks[iblk].d));
                const int packed = load_i32_aligned2(v_blocks[iblk].qs + iqs);
                const int8_t *qv = (const int8_t *) &packed;

                const float2 v0 = make_float2(d * (float)qv[0], d * (float)qv[1]);
                const float2 v1 = make_float2(d * (float)qv[2], d * (float)qv[3]);

                const int out_i0 = base_el / (nthreads_V * 2);
                acc[out_i0 + 0].x += wk * v0.x;
                acc[out_i0 + 0].y += wk * v0.y;
                acc[out_i0 + 1].x += wk * v1.x;
                acc[out_i0 + 1].y += wk * v1.y;
            }
        }

        __syncthreads();
    }

    // Store warp accumulators to shared and reduce across warps.
#pragma unroll
    for (int i = 0; i < acc_per_lane; ++i) {
        acc_s[warp][lane][i] = acc[i];
    }
    __syncthreads();

    if (warp == 0) {
        const float inv_sum = kq_sum > 0.0f ? 1.0f / kq_sum : 0.0f;
        if ((int)lane >= nthreads_V) {
            return;
        }
#pragma unroll
        for (int i = 0; i < acc_per_lane; ++i) {
            float2 s = acc_s[0][lane][i];
#pragma unroll
            for (int w = 1; w < NWARPS; ++w) {
                const float2 t = acc_s[w][lane][i];
                s.x += t.x;
                s.y += t.y;
            }
            s.x *= inv_sum;
            s.y *= inv_sum;

            const int base_el = (i / 2) * (nthreads_V * 4) + (int)lane * 4 + (i % 2) * 2;
            out_row[(base_el + 0) * (int)o_stride_d] = __float2half_rn(s.x);
            out_row[(base_el + 1) * (int)o_stride_d] = __float2half_rn(s.y);
        }
    }
}

#define DEFINE_FLASH_ATTN_Q8_KERNEL(DIM, NW) \
extern "C" __global__ void flash_attn_q8_kernel_##DIM##_w##NW( \
    const half * __restrict__ q_ptr, \
    const void * __restrict__ k_ptr, \
    const void * __restrict__ v_ptr, \
    half * __restrict__ o_ptr, \
    const float scale, \
    const uint32_t b, \
    const uint32_t h, \
    const uint32_t h_k, \
    const uint32_t d, \
    const uint32_t seq_q, \
    const uint32_t seq_k, \
    const uint32_t q_offset, \
    const int causal, \
    const uint64_t q_stride_b, const uint64_t q_stride_seq, const uint64_t q_stride_h, const uint64_t q_stride_d, \
    const uint64_t k_stride_b, const uint64_t k_stride_seq, const uint64_t k_stride_h, const uint64_t k_stride_d, \
    const uint64_t v_stride_b, const uint64_t v_stride_seq, const uint64_t v_stride_h, const uint64_t v_stride_d, \
    const uint64_t o_stride_b, const uint64_t o_stride_seq, const uint64_t o_stride_h, const uint64_t o_stride_d) { \
    flash_attn_q8_kernel_fast<DIM, NW>( \
        q_ptr, k_ptr, v_ptr, o_ptr, scale, b, h, h_k, seq_q, seq_k, q_offset, causal, \
        q_stride_b, q_stride_seq, q_stride_h, q_stride_d, \
        k_stride_b, k_stride_seq, k_stride_h, k_stride_d, \
        v_stride_b, v_stride_seq, v_stride_h, v_stride_d, \
        o_stride_b, o_stride_seq, o_stride_h, o_stride_d); \
}

DEFINE_FLASH_ATTN_Q8_KERNEL(64, 2)
DEFINE_FLASH_ATTN_Q8_KERNEL(64, 4)
DEFINE_FLASH_ATTN_Q8_KERNEL(64, 8)
DEFINE_FLASH_ATTN_Q8_KERNEL(128, 2)
DEFINE_FLASH_ATTN_Q8_KERNEL(128, 4)
DEFINE_FLASH_ATTN_Q8_KERNEL(128, 8)
DEFINE_FLASH_ATTN_Q8_KERNEL(256, 2)
DEFINE_FLASH_ATTN_Q8_KERNEL(256, 4)
DEFINE_FLASH_ATTN_Q8_KERNEL(256, 8)
