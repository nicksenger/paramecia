// Shared dequantization helpers for K-quant tiled matmul kernels
// This file contains dequantization functions for Q2_K through Q6_K
// optimized for hierarchical tiling with cooperative loading

#ifndef DEQUANT_KQUANTS_TILED_GLSL
#define DEQUANT_KQUANTS_TILED_GLSL

#include "dequant_helpers.glsl"

// Load u32 from unaligned byte offset
uint load_u32_unaligned(uint byte_offset) {
    uint word_idx = byte_offset / 4;
    uint byte_in_word = byte_offset % 4;
    if (byte_in_word == 0) {
        return v_input_b[word_idx];
    } else {
        uint w0 = v_input_b[word_idx];
        uint w1 = v_input_b[word_idx + 1];
        return (w0 >> (byte_in_word * 8)) | (w1 << ((4 - byte_in_word) * 8));
    }
}

// ============================================================================
// Q4_K Dequantization (256 elements/block, 144 bytes)
// ============================================================================

// Extract scale and minimum for a given sub-block (is = 0-7)
void get_q4k_scale_min(uint block_byte, uint is, out float d_scaled, out float m_scaled) {
    float d = load_f16(block_byte + 0);
    float dmin = load_f16(block_byte + 2);

    uint sc_offset = block_byte + 4;
    uint b0 = load_byte(sc_offset + 0);
    uint b1 = load_byte(sc_offset + 1);
    uint b2 = load_byte(sc_offset + 2);
    uint b3 = load_byte(sc_offset + 3);
    uint b4 = load_byte(sc_offset + 4);
    uint b5 = load_byte(sc_offset + 5);
    uint b6 = load_byte(sc_offset + 6);
    uint b7 = load_byte(sc_offset + 7);
    uint b8 = load_byte(sc_offset + 8);
    uint b9 = load_byte(sc_offset + 9);
    uint b10 = load_byte(sc_offset + 10);
    uint b11 = load_byte(sc_offset + 11);

    uint sc, mn;
    if (is < 4) {
        uint[4] b_lo = uint[](b0, b1, b2, b3);
        uint[4] b_hi = uint[](b4, b5, b6, b7);
        sc = b_lo[is] & 0x3F;
        mn = b_hi[is] & 0x3F;
    } else {
        uint[4] b_hi2 = uint[](b8, b9, b10, b11);
        uint[4] b_lo = uint[](b0, b1, b2, b3);
        uint[4] b_mid = uint[](b4, b5, b6, b7);
        uint idx = is - 4;
        uint q_is4 = b_hi2[idx];
        uint q_is4n = b_lo[idx];
        uint q_is = b_mid[idx];
        sc = (q_is4 & 0xF) | (((q_is4n >> 6) & 0x3) << 4);
        mn = ((q_is4 >> 4) & 0xF) | (((q_is >> 6) & 0x3) << 4);
    }

    d_scaled = d * float(sc);
    m_scaled = dmin * float(mn);
}

// Dequantize single Q4_K element at k_idx within block
float dequantize_q4k_elem(uint block_byte, uint k_idx) {
    // k_idx is 0-255 within the block
    // Determine which of 8 groups (32 elements each)
    uint chunk = k_idx / 64;
    uint off_in_chunk = k_idx % 64;
    uint qs_idx;
    bool use_upper_nibble;
    uint scale_idx;

    if (off_in_chunk < 32) {
        qs_idx = chunk * 32 + off_in_chunk;
        use_upper_nibble = false;
        scale_idx = chunk * 2;
    } else {
        qs_idx = chunk * 32 + (off_in_chunk - 32);
        use_upper_nibble = true;
        scale_idx = chunk * 2 + 1;
    }

    // Load qs nibble
    uint qs_byte = load_byte(block_byte + 16 + qs_idx);
    uint q4 = use_upper_nibble ? (qs_byte >> 4) : (qs_byte & 0xF);

    // Get scale and min
    float d_scaled, m_scaled;
    get_q4k_scale_min(block_byte, scale_idx, d_scaled, m_scaled);

    return d_scaled * float(q4) - m_scaled;
}

// ============================================================================
// Q6_K Dequantization (256 elements/block, 210 bytes)
// ============================================================================

// Get scale for Q6_K sub-block (is = 0-15, 16 groups of 16 elements)
// Correct Q6_K layout: [ql:128, qh:64, scales:16, d:2]
float get_q6k_scale(uint block_byte, uint is) {
    float d = load_f16(block_byte + 208);  // d is at bytes 208-209
    uint scale_offset = block_byte + 192;  // scales at bytes 192-207
    int8_t sc = int8_t(load_byte(scale_offset + is));
    return d * float(sc);
}

// Dequantize single Q6_K element
// Correct Q6_K layout: [ql:128, qh:64, scales:16, d:2]
float dequantize_q6k_elem(uint block_byte, uint k_idx) {
    // k_idx is 0-255 within block
    // 16 groups of 16 elements each
    uint scale_idx = k_idx / 16;
    float scale = get_q6k_scale(block_byte, scale_idx);

    // Q6K stores: 128 bytes ql (lower 4 bits), 64 bytes qh (upper 2 bits)
    uint ql_offset = block_byte + 0;    // ql at bytes 0-127
    uint qh_offset = block_byte + 128;  // qh at bytes 128-191

    uint ql_byte = load_byte(ql_offset + k_idx / 2);
    uint ql = (k_idx % 2 == 0) ? (ql_byte & 0xF) : (ql_byte >> 4);

    uint qh_byte = load_byte(qh_offset + k_idx / 4);
    uint qh_shift = ((k_idx % 4) * 2);
    uint qh = (qh_byte >> qh_shift) & 0x3;

    // Combine: 6-bit value = (qh << 4) | ql
    uint q6 = (qh << 4) | ql;

    return scale * (float(q6) - 32.0);
}

// ============================================================================
// Q5_K Dequantization (256 elements/block, 176 bytes)
// Correct layout: [d:2, dmin:2, scales:12, qh:32, qs:128]
// ============================================================================

// Get scale and min for Q5_K sub-block (is = 0-7)
// Q5_K scales are packed similarly to Q4_K
void get_q5k_scale_min(uint block_byte, uint is, out float d_scaled, out float m_scaled) {
    float d = load_f16(block_byte + 0);
    float dmin = load_f16(block_byte + 2);

    uint sc_offset = block_byte + 4;
    uint b0 = load_byte(sc_offset + 0);
    uint b1 = load_byte(sc_offset + 1);
    uint b2 = load_byte(sc_offset + 2);
    uint b3 = load_byte(sc_offset + 3);
    uint b4 = load_byte(sc_offset + 4);
    uint b5 = load_byte(sc_offset + 5);
    uint b6 = load_byte(sc_offset + 6);
    uint b7 = load_byte(sc_offset + 7);
    uint b8 = load_byte(sc_offset + 8);
    uint b9 = load_byte(sc_offset + 9);
    uint b10 = load_byte(sc_offset + 10);
    uint b11 = load_byte(sc_offset + 11);

    // Same encoding as Q4_K
    uint sc, mn;
    if (is < 4) {
        uint[4] b_lo = uint[](b0, b1, b2, b3);
        uint[4] b_hi = uint[](b4, b5, b6, b7);
        sc = b_lo[is] & 0x3F;
        mn = b_hi[is] & 0x3F;
    } else {
        uint[4] b_hi2 = uint[](b8, b9, b10, b11);
        uint[4] b_lo = uint[](b0, b1, b2, b3);
        uint[4] b_mid = uint[](b4, b5, b6, b7);
        uint idx = is - 4;
        uint q_is4 = b_hi2[idx];
        uint q_is4n = b_lo[idx];
        uint q_is = b_mid[idx];
        sc = (q_is4 & 0xF) | (((q_is4n >> 6) & 0x3) << 4);
        mn = ((q_is4 >> 4) & 0xF) | (((q_is >> 6) & 0x3) << 4);
    }

    d_scaled = d * float(sc);
    m_scaled = dmin * float(mn);
}

// Dequantize single Q5_K element
float dequantize_q5k_elem(uint block_byte, uint k_idx) {
    // 8 groups of 32 elements
    uint scale_idx = k_idx / 32;
    float d_scaled, m_scaled;
    get_q5k_scale_min(block_byte, scale_idx, d_scaled, m_scaled);

    // Q5K: 128 bytes qs (lower 4 bits), 32 bytes qh (upper 1 bit)
    uint qs_offset = block_byte + 48;  // qs at bytes 48-175
    uint qh_offset = block_byte + 16;  // qh at bytes 16-47

    uint qs_byte = load_byte(qs_offset + k_idx / 2);
    uint qs = (k_idx % 2 == 0) ? (qs_byte & 0xF) : (qs_byte >> 4);

    uint qh_byte = load_byte(qh_offset + k_idx / 8);
    uint qh_bit = (qh_byte >> (k_idx % 8)) & 0x1;

    // Combine: 5-bit value = (qh << 4) | qs
    uint q5 = (qh_bit << 4) | qs;

    return d_scaled * float(q5) - m_scaled;
}

// ============================================================================
// Q3_K Dequantization (256 elements/block, 110 bytes)
// Correct layout: [hmask:32, qs:64, scales:12, d:2]
// Q3_K encoding: 2-bit values in qs + 1-bit in hmask = 3-bit range [-4, 3]
// 256 elements = 2 half-blocks of 128, each with 4 groups (shift 0,2,4,6)
// ============================================================================

// Get scale for Q3_K sub-block (is = 0-15)
// Scales are packed in 12 bytes, decoded using KMASK bit manipulation
float get_q3k_scale(uint block_byte, uint is) {
    float d = load_f16(block_byte + 108);  // d at bytes 108-109

    // Load 12 scale bytes as 3 u32s
    uint scales_offset = block_byte + 96;
    uint a0 = load_u32_unaligned(scales_offset);
    uint a1 = load_u32_unaligned(scales_offset + 4);
    uint a2 = load_u32_unaligned(scales_offset + 8);

    // Decode 16 scales from 12 bytes using KMASK pattern
    const uint KMASK1 = 0x03030303u;
    const uint KMASK2 = 0x0f0f0f0fu;

    uint d0 = (a0 & KMASK2) | (((a2) & KMASK1) << 4);
    uint d1 = (a1 & KMASK2) | (((a2 >> 2) & KMASK1) << 4);
    uint d2 = ((a0 >> 4) & KMASK2) | (((a2 >> 4) & KMASK1) << 4);
    uint d3 = ((a1 >> 4) & KMASK2) | (((a2 >> 6) & KMASK1) << 4);

    // Pick the right u32 and byte based on scale index
    uint word;
    uint idx = is;
    if (idx < 4) {
        word = d0;
    } else if (idx < 8) {
        word = d1;
        idx -= 4;
    } else if (idx < 12) {
        word = d2;
        idx -= 8;
    } else {
        word = d3;
        idx -= 12;
    }

    // Extract byte and convert to signed (scale - 32)
    int scale_byte = int((word >> (idx * 8)) & 0xFF);
    return d * float(scale_byte - 32);
}

// Dequantize single Q3_K element
// Q3_K uses 2-bit qs values + 1-bit hmask to create 3-bit signed values
float dequantize_q3k_elem(uint block_byte, uint k_idx) {
    // Determine position in structure
    uint half_blk = k_idx / 128;        // 0 or 1
    uint local_idx = k_idx % 128;       // 0-127
    uint shift_group = local_idx / 32;  // 0-3 (determines shift: 0, 2, 4, 6)
    uint subgroup = (local_idx % 32) / 16;  // 0 or 1
    uint lane = local_idx % 16;         // 0-15

    // Scale index: 16 scales total (2 subgroups × 4 shift_groups × 2 half_blocks)
    uint scale_idx = half_blk * 8 + shift_group * 2 + subgroup;
    float scale = get_q3k_scale(block_byte, scale_idx);

    // qs byte index
    uint qs_offset = block_byte + 32;  // qs at bytes 32-95
    uint qs_byte_idx = half_blk * 32 + subgroup * 16 + lane;
    uint qs_byte = load_byte(qs_offset + qs_byte_idx);

    // Extract 2-bit value using shift
    uint shift = shift_group * 2;
    uint q2 = (qs_byte >> shift) & 3;

    // Get hmask bit
    uint hmask_offset = block_byte + 0;  // hmask at bytes 0-31
    uint hmask_byte_idx = subgroup * 16 + lane;
    uint hmask_byte = load_byte(hmask_offset + hmask_byte_idx);
    uint m = 1u << shift_group;  // Bit position for this shift group

    // Combine: q3 = q2 - (hmask_bit ? 0 : 4)
    int q3 = int(q2) - ((hmask_byte & m) != 0u ? 0 : 4);

    return scale * float(q3);
}

// ============================================================================
// Q2_K Dequantization (256 elements/block, 84 bytes)
// Correct layout: [scales:16, qs:64, d:2, dmin:2]
// Q2_K encoding: 2-bit values, organized in 2 halves with 4 shift groups each
// 256 elements = 2 halves (qs 0-31, 32-63), each with 4 groups (shift 0,2,4,6)
// Each scale byte: low 4 bits = scale, high 4 bits = min
// ============================================================================

// Dequantize single Q2_K element
float dequantize_q2k_elem(uint block_byte, uint k_idx) {
    float d = load_f16(block_byte + 80);     // d at bytes 80-81
    float dmin = load_f16(block_byte + 82);  // dmin at bytes 82-83

    // Determine position in structure
    uint qs_half = k_idx / 128;         // 0 or 1 (which half of qs array)
    uint local_idx = k_idx % 128;       // 0-127
    uint shift_group = local_idx / 32;  // 0-3 (determines shift: 0, 2, 4, 6)
    uint subgroup = (local_idx % 32) / 16;  // 0 or 1
    uint lane = local_idx % 16;         // 0-15

    // Scale index: 16 scales total (2 subgroups × 4 shift_groups × 2 halves)
    uint scale_idx = qs_half * 8 + shift_group * 2 + subgroup;

    // Load packed scale/min byte
    uint scales_offset = block_byte + 0;  // scales at bytes 0-15
    uint sc_byte = load_byte(scales_offset + scale_idx);
    uint sc = sc_byte & 0xF;   // Low 4 bits
    uint mn = sc_byte >> 4;    // High 4 bits

    float dl = d * float(sc);
    float ml = dmin * float(mn);

    // qs byte index
    uint qs_offset = block_byte + 16;  // qs at bytes 16-79
    uint qs_byte_idx = qs_half * 32 + subgroup * 16 + lane;
    uint qs_byte = load_byte(qs_offset + qs_byte_idx);

    // Extract 2-bit value using shift
    uint shift = shift_group * 2;
    uint q2 = (qs_byte >> shift) & 3;

    return dl * float(q2) - ml;
}

#endif // DEQUANT_KQUANTS_TILED_GLSL
