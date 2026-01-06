// MMA tensor core utilities for Ampere+ (sm_80+).
//
// Provides:
// - m16n8k16 MMA PTX wrappers for f16×f16→f32 and s8×s8→s32
// - ldmatrix for efficient shared→register transfers
// - Tile load/store helpers
// - Common reduction and dequantization utilities
//
// All operations target 32-thread warps using inline PTX.

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define WARP_SIZE 32

// ============================================================================
// Quantized block types (matching ggml)
// ============================================================================

#define QK8_0 32
#define QK8_1 32
#define QK4_0 32
#define QK_K  256
#define K_SCALE_SIZE 12

struct block_q8_0 {
    uint16_t d;        // f16 scale
    int8_t   qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");

struct block_q8_1 {
    half2   ds;        // ds.x = delta (f16), ds.y = d * sum(qs[i]) (f16)
    int8_t  qs[QK8_1]; // quants
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 size mismatch");

struct block_q4_0 {
    uint16_t d;        // f16 scale
    uint8_t  qs[QK4_0 / 2];
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

struct block_q4_K {
    uint16_t d;        // f16 super-block scale
    uint16_t dmin;     // f16 super-block min
    uint8_t  scales[K_SCALE_SIZE]; // quantized scales/mins
    uint8_t  qs[QK_K / 2];        // 4-bit quants
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch");

struct block_q6_K {
    uint8_t  ql[QK_K / 2]; // low 4 bits of quants
    uint8_t  qh[QK_K / 4]; // high 2 bits of quants
    int8_t   scales[QK_K / 16]; // scales (quantized with 8 bits)
    uint16_t d;             // f16 super-block scale
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size mismatch");

// ============================================================================
// Warp reductions
// ============================================================================

static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, mask, WARP_SIZE);
    }
    return v;
}

static __device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, mask, WARP_SIZE));
    }
    return v;
}

// ============================================================================
// DP4A intrinsic (fallback for pre-sm_61)
// ============================================================================

static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t *a8 = (const int8_t *)&a;
    const int8_t *b8 = (const int8_t *)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}

// ============================================================================
// Safe int32 load from 2-byte aligned int8 array
// ============================================================================

static __device__ __forceinline__ int load_i32_aligned2(const int8_t *p) {
    const uint16_t *p16 = reinterpret_cast<const uint16_t *>(p);
    return (int)((uint32_t)p16[0] | ((uint32_t)p16[1] << 16));
}

// ============================================================================
// MMA m16n8k16 PTX wrappers
// ============================================================================

// f16 × f16 → f32 accumulate
// A: 4 registers (row-major, m16×k16 across warp)
// B: 2 registers (col-major, k16×n8 across warp)
// C/D: 4 registers (m16×n8 accumulator)
static __device__ __forceinline__ void mma_f16_m16n8k16(
    float       d[4],        // output/accumulator
    const uint32_t a[4],     // A operand (f16 packed as u32)
    const uint32_t b[2],     // B operand (f16 packed as u32)
    const float    c[4]      // input accumulator
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// s8 × s8 → s32 accumulate
// A: 2 registers (row-major, m16×k16 packed int8)
// B: 1 register  (col-major, k16×n8 packed int8)
// C/D: 4 registers (m16×n8 accumulator)
static __device__ __forceinline__ void mma_i8_m16n8k16(
    int         d[4],        // output/accumulator
    const int   a[2],        // A operand (s8 packed)
    const int   b[1],        // B operand (s8 packed)
    const int   c[4]         // input accumulator
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// ============================================================================
// ldmatrix: Shared memory → register file (efficient warp-wide load)
// ============================================================================

// Load 4 x 8×8 matrices (trans=false: row-major, trans=true: col-major)
// Each thread provides its own shared memory address via `ptr`.
// Returns 4 uint32_t in `dst`.
static __device__ __forceinline__ void ldmatrix_x4(uint32_t dst[4], const void *ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 addr64; cvta.to.shared.u64 addr64, %1; cvt.u32.u64 %0, addr64; }\n"
                 : "=r"(addr)
                 : "l"(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(addr)
    );
}

// Load 2 x 8×8 matrices
static __device__ __forceinline__ void ldmatrix_x2(uint32_t dst[2], const void *ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 addr64; cvta.to.shared.u64 addr64, %1; cvt.u32.u64 %0, addr64; }\n"
                 : "=r"(addr)
                 : "l"(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(addr)
    );
}

// Load 1 x 8×8 matrix
static __device__ __forceinline__ void ldmatrix_x1(uint32_t *dst, const void *ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 addr64; cvta.to.shared.u64 addr64, %1; cvt.u32.u64 %0, addr64; }\n"
                 : "=r"(addr)
                 : "l"(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(dst[0])
        : "r"(addr)
    );
}

// Transposed load variants (needed for B operands in some cases)
static __device__ __forceinline__ void ldmatrix_x4_trans(uint32_t dst[4], const void *ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 addr64; cvta.to.shared.u64 addr64, %1; cvt.u32.u64 %0, addr64; }\n"
                 : "=r"(addr)
                 : "l"(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(addr)
    );
}

static __device__ __forceinline__ void ldmatrix_x2_trans(uint32_t dst[2], const void *ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 addr64; cvta.to.shared.u64 addr64, %1; cvt.u32.u64 %0, addr64; }\n"
                 : "=r"(addr)
                 : "l"(ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(addr)
    );
}

// ============================================================================
// Shared memory helpers
// ============================================================================

// Bank-conflict-free index for shared memory (16 banks, 4-byte words)
// Pads by 1 element per 8 elements to avoid 4-way bank conflicts
static __device__ __forceinline__ int shmem_idx_bc(int row, int col, int stride) {
    return row * (stride + 8) + col;  // +8 padding per row
}

// ============================================================================
// Dequantize Q8_0 block to half2 (2 consecutive values)
// ============================================================================

static __device__ __forceinline__ half2 dequant_q8_0_pair(const block_q8_0 *blk, int offset) {
    float d = __half2float(*reinterpret_cast<const __half *>(&blk->d));
    float v0 = d * (float)blk->qs[offset];
    float v1 = d * (float)blk->qs[offset + 1];
    return __halves2half2(__float2half_rn(v0), __float2half_rn(v1));
}

// Dequantize 4 consecutive Q8_0 values to 2 half2
static __device__ __forceinline__ void dequant_q8_0_quad(
    const block_q8_0 *blk, int offset,
    half2 *out0, half2 *out1
) {
    float d = __half2float(*reinterpret_cast<const __half *>(&blk->d));
    float v0 = d * (float)blk->qs[offset + 0];
    float v1 = d * (float)blk->qs[offset + 1];
    float v2 = d * (float)blk->qs[offset + 2];
    float v3 = d * (float)blk->qs[offset + 3];
    *out0 = __halves2half2(__float2half_rn(v0), __float2half_rn(v1));
    *out1 = __halves2half2(__float2half_rn(v2), __float2half_rn(v3));
}

// ============================================================================
// Q8_0 → int8 extraction helpers for MMA i8 path
// ============================================================================

// Extract packed int32 (4 x int8) from Q8_0 block at given offset
static __device__ __forceinline__ int q8_0_get_i32(const block_q8_0 *blk, int i32_idx) {
    return load_i32_aligned2(blk->qs + 4 * i32_idx);
}

// Get scale from Q8_0 block
static __device__ __forceinline__ float q8_0_get_d(const block_q8_0 *blk) {
    return __half2float(*reinterpret_cast<const __half *>(&blk->d));
}

// ============================================================================
// Q4_K helpers
// ============================================================================

// Decode 6-bit scales from Q4_K block
// Returns scale and min for a given sub-block (0..7)
static __device__ __forceinline__ void q4_k_get_scale_min(
    const block_q4_K *blk, int sub_block,
    float *scale, float *min
) {
    const float d = __half2float(*reinterpret_cast<const __half *>(&blk->d));
    const float dmin = __half2float(*reinterpret_cast<const __half *>(&blk->dmin));
    const uint8_t *sc = blk->scales;

    uint8_t sc_val, m_val;
    if (sub_block < 4) {
        sc_val = sc[sub_block] & 0x3F;
        m_val  = sc[sub_block + 4] & 0x3F;
    } else {
        sc_val = ((sc[sub_block + 4] & 0xF) | ((sc[sub_block - 4] >> 6) << 4));
        m_val  = ((sc[sub_block + 4] >> 4)  | ((sc[sub_block    ] >> 6) << 4));
    }
    *scale = d * (float)sc_val;
    *min   = dmin * (float)m_val;
}

// ============================================================================
// Q6_K helpers
// ============================================================================

// Decode a single Q6_K value at position `idx` (0..255)
static __device__ __forceinline__ int8_t q6_k_get_val(const block_q6_K *blk, int idx) {
    int ql_byte = blk->ql[idx / 2];
    int ql_val;
    if (idx % 2 == 0) {
        ql_val = ql_byte & 0xF;
    } else {
        ql_val = (ql_byte >> 4) & 0xF;
    }
    int qh_byte = blk->qh[idx / 4];
    int qh_shift = (idx % 4) * 2;
    int qh_val = (qh_byte >> qh_shift) & 3;
    return (int8_t)((ql_val | (qh_val << 4)) - 32);
}

// ============================================================================
// MMQ tile configuration constants
// ============================================================================

// MMA tile: m16×n8×k16
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// MMQ block configuration
// Each block computes a MMQ_Y × MMQ_X output tile
#define MMQ_MMA_TILE_X_K_Q8_0  128   // activation tile width per iteration
