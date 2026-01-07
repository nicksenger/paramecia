// Q8_0 flash attention kernel (fast path) -- Metal port.
//
// Key ideas adapted from llama.cpp's "fattn-vec" kernels:
// - Quantize Q on-the-fly to q8_1 (scale-only) and use dp4a for K·Q.
// - Use multiple warps (simdgroups) per threadgroup and tile over the KV sequence.
// - Online softmax (max/sum rescaling) to avoid extra passes.
//
// Supported head dims: 64, 128, 256.
// Launch config: threads_per_threadgroup = (32 lanes, 4 warps, 1),
//                threadgroups_per_grid   = (b, h, seq_q).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_math>

using namespace metal;

#define WARP_SIZE 32
#define QK8_0 32
#define QI8_0 (QK8_0 / 4)  // 8 packed int32 per 32 values
#define NWARPS 4

struct block_q8_0 {
    ushort d;           // f16 scale stored as uint16
    char qs[QK8_0];     // quantized values
};

// dp4a equivalent: dot product of 4 packed int8 values
inline int dp4a(int a, int b, int c) {
    char4 a8 = as_type<char4>(a);
    char4 b8 = as_type<char4>(b);
    return c + int(a8.x)*int(b8.x) + int(a8.y)*int(b8.y) + int(a8.z)*int(b8.z) + int(a8.w)*int(b8.w);
}

// Load 4 consecutive int8 as a packed int32 using two 16-bit loads (2-byte aligned).
inline int load_i32_aligned2(const device char* p) {
    const device ushort* p16 = reinterpret_cast<const device ushort*>(p);
    uint lo = uint(p16[0]);
    uint hi = uint(p16[1]);
    return int(lo | (hi << 16));
}

inline float warp_reduce_sum(float v) {
    for (ushort mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        v += simd_shuffle_xor(v, mask);
    }
    return v;
}

inline float warp_reduce_max(float v) {
    for (ushort mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
        v = max(v, simd_shuffle_xor(v, mask));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Quantize Q row to q8_1 (scale-only variant)
// ---------------------------------------------------------------------------

template <int D>
inline void quantize_q_to_q8_1_scale_only(
    const device half* q_row,
    const float scale,
    const ulong q_stride_d,
    threadgroup int* q_q8,
    threadgroup float* q_d,
    uint lane
) {
    constexpr int ints_per_row = D / 4;
    constexpr int nthreads_quantize = (ints_per_row < WARP_SIZE ? ints_per_row : WARP_SIZE);

    for (int i0 = 0; i0 < ints_per_row; i0 += nthreads_quantize) {
        float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if ((int)lane < nthreads_quantize) {
            const int base = 4 * (i0 + (int)lane);
            for (int l = 0; l < 4; ++l) {
                const float qv = float(q_row[(base + l) * (int)q_stride_d]);
                vals[l] = scale * qv;
            }
        }

        float amax = abs(vals[0]);
        for (int l = 1; l < 4; ++l) {
            amax = max(amax, abs(vals[l]));
        }

        // Reduce within 8-lane groups (QI8_0 == 8) so each group quantizes one 32-value block.
        for (ushort mask = QI8_0/2; mask > 0; mask >>= 1) {
            amax = max(amax, simd_shuffle_xor(amax, mask));
        }

        const float d = amax / 127.0f;
        int q32 = 0;
        char4 q8;
        if ((int)lane < nthreads_quantize) {
            if (d != 0.0f) {
                for (int l = 0; l < 4; ++l) {
                    // Use rint for nearest-integer rounding (same as nearbyintf)
                    ((thread char*)&q32)[l] = char(rint(vals[l] / d));
                }
            } else {
                q32 = 0;
            }

            q_q8[i0 + (int)lane] = q32;
            if (((int)lane % QI8_0) == 0) {
                q_d[(i0 + (int)lane) / QI8_0] = d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main flash attention kernel body (templated on head dim D)
// ---------------------------------------------------------------------------

template <int D>
void flash_attn_q8_impl(
    const device half* Q,
    const device block_q8_0* K,
    const device half* V_half,
    device half* output,
    constant float& scale,
    constant int& seq_k_param,
    constant int& seq_q_param,
    constant ulong& q_stride_b,
    constant ulong& q_stride_h,
    constant ulong& q_stride_s,
    constant ulong& q_stride_d,
    constant ulong& k_stride_b,
    constant ulong& k_stride_h,
    constant ulong& k_stride_s,
    constant ulong& v_stride_b,
    constant ulong& v_stride_h,
    constant ulong& v_stride_s,
    constant ulong& v_stride_d,
    constant ulong& o_stride_b,
    constant ulong& o_stride_h,
    constant ulong& o_stride_s,
    uint3 tgid,
    uint2 tid
) {
    const uint ib     = tgid.x;
    const uint ih     = tgid.y;
    const uint iq_seq = tgid.z;

    const uint seq_k_val = uint(seq_k_param);
    const uint seq_q_val = uint(seq_q_param);

    const uint lane = tid.x;   // 0..31
    const uint warp = tid.y;   // 0..3
    const uint thread_id = warp * WARP_SIZE + lane;

    // Compute pointer to Q row and output row.
    const device half* q_row = Q + ib * q_stride_b + iq_seq * q_stride_s + ih * q_stride_h;
    device half* out_row     = output + ib * o_stride_b + iq_seq * o_stride_s + ih * o_stride_h;

    constexpr int ints_per_row  = D / 4;
    constexpr int blocks_per_row = D / QK8_0;
    constexpr int nthreads_V    = (D/4 < WARP_SIZE ? D/4 : WARP_SIZE);
    constexpr int acc_per_lane  = (D/2) / nthreads_V;

    // Shared memory declarations.
    threadgroup int   q_q8[ints_per_row];
    threadgroup float q_d_shared[blocks_per_row];
    threadgroup float tile_max_s;
    threadgroup float tile_sum_s;
    threadgroup float warp_red_s[NWARPS];
    threadgroup float kq_w[NWARPS * WARP_SIZE];
    threadgroup float2 acc_s[NWARPS][WARP_SIZE][acc_per_lane];

    // Quantize Q to q8_1 (only warp 0).
    if (warp == 0) {
        quantize_q_to_q8_1_scale_only<D>(q_row, scale, q_stride_d, q_q8, q_d_shared, lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state.
    float kq_max = -INFINITY;
    float kq_sum = 0.0f;

    float2 acc[acc_per_lane];
    for (int i = 0; i < acc_per_lane; ++i) {
        acc[i] = float2(0.0f, 0.0f);
    }

    // K and V are passed as device pointers; we cast to char* for byte-offset arithmetic.
    const device char* K_bytes = reinterpret_cast<const device char*>(K);
    const device char* V_bytes = reinterpret_cast<const device char*>(V_half);

    constexpr uint keys_per_tile = NWARPS * WARP_SIZE;
    for (uint tile0 = 0; tile0 < seq_k_val; tile0 += keys_per_tile) {
        float score_reg = -INFINITY;
        float warp_max_val = -INFINITY;

        for (int i_key = 0; i_key < (int)WARP_SIZE; ++i_key) {
            const uint k_idx = tile0 + warp * WARP_SIZE + uint(i_key);
            float score_val = -INFINITY;
            const bool key_in_range = k_idx < seq_k_val;
            if (key_in_range) {
                const device char* k_row = K_bytes + ulong(ib) * k_stride_b + ulong(k_idx) * k_stride_s + ulong(ih) * k_stride_h;
                const device block_q8_0* k_blocks = reinterpret_cast<const device block_q8_0*>(k_row);

                float sum = 0.0f;
                for (int k_int_0 = 0; k_int_0 < ints_per_row; k_int_0 += WARP_SIZE) {
                    const int k_int = k_int_0 + (int)lane;
                    if (k_int < ints_per_row) {
                        const int iblk = k_int / QI8_0;
                        const int iqs  = k_int % QI8_0;
                        const int v = load_i32_aligned2(k_blocks[iblk].qs + 4*iqs);
                        const int u = q_q8[k_int];
                        const float d0 = float(as_type<half>(k_blocks[iblk].d));
                        const float d1 = q_d_shared[iblk];
                        sum += (d0 * d1) * float(dp4a(v, u, 0));
                    }
                }

                sum = warp_reduce_sum(sum);
                score_val = sum;
            }

            warp_max_val = max(warp_max_val, score_val);
            if (lane == uint(i_key)) {
                score_reg = score_val;
            }
        }

        warp_max_val = warp_reduce_max(warp_max_val);
        if (lane == 0) {
            warp_red_s[warp] = warp_max_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (warp == 0 && lane == 0) {
            float tm = warp_red_s[0];
            for (int w = 1; w < NWARPS; ++w) {
                tm = max(tm, warp_red_s[w]);
            }
            tile_max_s = tm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float kq_max_new = max(kq_max, tile_max_s);
        const float kq_scale_old = isinf(kq_max) && kq_max < 0 ? 0.0f : exp(kq_max - kq_max_new);

        kq_sum *= kq_scale_old;
        for (int i = 0; i < acc_per_lane; ++i) {
            acc[i].x *= kq_scale_old;
            acc[i].y *= kq_scale_old;
        }
        kq_max = kq_max_new;

        // Compute exp(score - kq_max) for this thread's key.
        float w_val = 0.0f;
        if (!isinf(score_reg) || score_reg > 0) {
            w_val = exp(score_reg - kq_max);
        }
        kq_w[thread_id] = w_val;

        float warp_sum_val = warp_reduce_sum(w_val);
        if (lane == 0) {
            warp_red_s[warp] = warp_sum_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (warp == 0 && lane == 0) {
            float ts = 0.0f;
            for (int w = 0; w < NWARPS; ++w) {
                ts += warp_red_s[w];
            }
            tile_sum_s = ts;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        kq_sum += tile_sum_s;

        // Accumulate V * w for all keys in this tile.
        for (int k0 = 0; k0 < (int)WARP_SIZE; ++k0) {
            const uint k_idx = tile0 + warp * WARP_SIZE + uint(k0);
            const float wk = kq_w[warp * WARP_SIZE + uint(k0)];
            if (wk == 0.0f) {
                continue;
            }
            if (k_idx >= seq_k_val) {
                continue;
            }

            const device char* v_row = V_bytes + ulong(ib) * v_stride_b + ulong(k_idx) * v_stride_s + ulong(ih) * v_stride_h;
            const device block_q8_0* v_blocks = reinterpret_cast<const device block_q8_0*>(v_row);

            if ((int)lane >= nthreads_V) {
                continue;
            }

            for (int base_el = 0; base_el < D; base_el += nthreads_V * 4) {
                const int i_el = base_el + (int)lane * 4;

                const int iblk = i_el / QK8_0;
                const int iqs  = i_el % QK8_0;

                const float dv = float(as_type<half>(v_blocks[iblk].d));
                const int packed = load_i32_aligned2(v_blocks[iblk].qs + iqs);
                const char4 qv = as_type<char4>(packed);

                const float2 v0 = float2(dv * float(qv.x), dv * float(qv.y));
                const float2 v1 = float2(dv * float(qv.z), dv * float(qv.w));

                const int out_i0 = base_el / (nthreads_V * 2);
                acc[out_i0 + 0].x += wk * v0.x;
                acc[out_i0 + 0].y += wk * v0.y;
                acc[out_i0 + 1].x += wk * v1.x;
                acc[out_i0 + 1].y += wk * v1.y;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store warp accumulators to shared and reduce across warps.
    for (int i = 0; i < acc_per_lane; ++i) {
        acc_s[warp][lane][i] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0) {
        const float inv_sum = kq_sum > 0.0f ? 1.0f / kq_sum : 0.0f;
        if ((int)lane >= nthreads_V) {
            return;
        }
        for (int i = 0; i < acc_per_lane; ++i) {
            float2 s = acc_s[0][lane][i];
            for (int w = 1; w < NWARPS; ++w) {
                const float2 t = acc_s[w][lane][i];
                s.x += t.x;
                s.y += t.y;
            }
            s.x *= inv_sum;
            s.y *= inv_sum;

            const int base_el = (i / 2) * (nthreads_V * 4) + (int)lane * 4 + (i % 2) * 2;
            out_row[base_el + 0] = half(s.x);
            out_row[base_el + 1] = half(s.y);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel entry points for each supported head dimension
// ---------------------------------------------------------------------------

kernel void flash_attn_q8_d64(
    device const half* Q [[buffer(0)]],
    device const block_q8_0* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& seq_k [[buffer(5)]],
    constant int& seq_q [[buffer(6)]],
    constant ulong& stride_q_b [[buffer(7)]],
    constant ulong& stride_q_h [[buffer(8)]],
    constant ulong& stride_q_s [[buffer(9)]],
    constant ulong& stride_q_d [[buffer(10)]],
    constant ulong& stride_k_b [[buffer(11)]],
    constant ulong& stride_k_h [[buffer(12)]],
    constant ulong& stride_k_s [[buffer(13)]],
    constant ulong& stride_v_b [[buffer(14)]],
    constant ulong& stride_v_h [[buffer(15)]],
    constant ulong& stride_v_s [[buffer(16)]],
    constant ulong& stride_v_d [[buffer(17)]],
    constant ulong& stride_o_b [[buffer(18)]],
    constant ulong& stride_o_h [[buffer(19)]],
    constant ulong& stride_o_s [[buffer(20)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    flash_attn_q8_impl<64>(
        Q, K, V, O, scale, seq_k, seq_q,
        stride_q_b, stride_q_h, stride_q_s, stride_q_d,
        stride_k_b, stride_k_h, stride_k_s,
        stride_v_b, stride_v_h, stride_v_s, stride_v_d,
        stride_o_b, stride_o_h, stride_o_s,
        tgid, tid);
}

kernel void flash_attn_q8_d128(
    device const half* Q [[buffer(0)]],
    device const block_q8_0* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& seq_k [[buffer(5)]],
    constant int& seq_q [[buffer(6)]],
    constant ulong& stride_q_b [[buffer(7)]],
    constant ulong& stride_q_h [[buffer(8)]],
    constant ulong& stride_q_s [[buffer(9)]],
    constant ulong& stride_q_d [[buffer(10)]],
    constant ulong& stride_k_b [[buffer(11)]],
    constant ulong& stride_k_h [[buffer(12)]],
    constant ulong& stride_k_s [[buffer(13)]],
    constant ulong& stride_v_b [[buffer(14)]],
    constant ulong& stride_v_h [[buffer(15)]],
    constant ulong& stride_v_s [[buffer(16)]],
    constant ulong& stride_v_d [[buffer(17)]],
    constant ulong& stride_o_b [[buffer(18)]],
    constant ulong& stride_o_h [[buffer(19)]],
    constant ulong& stride_o_s [[buffer(20)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    flash_attn_q8_impl<128>(
        Q, K, V, O, scale, seq_k, seq_q,
        stride_q_b, stride_q_h, stride_q_s, stride_q_d,
        stride_k_b, stride_k_h, stride_k_s,
        stride_v_b, stride_v_h, stride_v_s, stride_v_d,
        stride_o_b, stride_o_h, stride_o_s,
        tgid, tid);
}

kernel void flash_attn_q8_d256(
    device const half* Q [[buffer(0)]],
    device const block_q8_0* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant int& seq_k [[buffer(5)]],
    constant int& seq_q [[buffer(6)]],
    constant ulong& stride_q_b [[buffer(7)]],
    constant ulong& stride_q_h [[buffer(8)]],
    constant ulong& stride_q_s [[buffer(9)]],
    constant ulong& stride_q_d [[buffer(10)]],
    constant ulong& stride_k_b [[buffer(11)]],
    constant ulong& stride_k_h [[buffer(12)]],
    constant ulong& stride_k_s [[buffer(13)]],
    constant ulong& stride_v_b [[buffer(14)]],
    constant ulong& stride_v_h [[buffer(15)]],
    constant ulong& stride_v_s [[buffer(16)]],
    constant ulong& stride_v_d [[buffer(17)]],
    constant ulong& stride_o_b [[buffer(18)]],
    constant ulong& stride_o_h [[buffer(19)]],
    constant ulong& stride_o_s [[buffer(20)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    flash_attn_q8_impl<256>(
        Q, K, V, O, scale, seq_k, seq_q,
        stride_q_b, stride_q_h, stride_q_s, stride_q_d,
        stride_k_b, stride_k_h, stride_k_s,
        stride_v_b, stride_v_h, stride_v_s, stride_v_d,
        stride_o_b, stride_o_h, stride_o_s,
        tgid, tid);
}
