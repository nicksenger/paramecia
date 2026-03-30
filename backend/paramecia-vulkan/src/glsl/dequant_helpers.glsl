// Shared helper functions for quantized matrix-vector multiplication kernels
// This file contains common utility functions used across Q8_0 and K-quant kernels

#ifndef DEQUANT_HELPERS_GLSL
#define DEQUANT_HELPERS_GLSL

// ============================================================================
// f16 Conversion Helpers
// ============================================================================

// Convert packed f16 to f32 (lower half)
float half_to_float_lo(uint packed) {
    return unpackHalf2x16(packed).x;
}

// Convert packed f16 to f32 (upper half)
float half_to_float_hi(uint packed) {
    return unpackHalf2x16(packed).y;
}

// Load f16 from byte offset, handling alignment
// Note: Assumes v_input_b[] buffer is available in scope
float load_f16(uint byte_offset) {
    uint word_idx = byte_offset / 4;
    uint byte_in_word = byte_offset % 4;
    uint word = v_input_b[word_idx];

    if (byte_in_word == 0) {
        return half_to_float_lo(word);
    } else if (byte_in_word == 2) {
        return half_to_float_hi(word);
    } else {
        // Unaligned - straddle two words
        uint word2 = v_input_b[word_idx + 1];
        uint h = (byte_in_word == 1)
            ? ((word >> 8) | (word2 << 24))
            : ((word >> 24) | (word2 << 8));
        return half_to_float_lo(h);
    }
}

// ============================================================================
// Byte Loading Helpers
// ============================================================================

// Load single byte from byte offset, handling alignment
// Note: Assumes v_input_b[] buffer is available in scope
uint load_byte(uint byte_offset) {
    uint word_idx = byte_offset / 4;
    uint byte_in_word = byte_offset % 4;
    uint word = v_input_b[word_idx];
    return (word >> (byte_in_word * 8)) & 0xFF;
}

// ============================================================================
// Warp Reduction (for K-quants with warp-per-output architecture)
// ============================================================================

// Fast warp-level reduction using shuffle operations (32 threads -> 1 value)
float warp_reduce_sum(float v) {
    v += subgroupShuffleXor(v, 16);
    v += subgroupShuffleXor(v, 8);
    v += subgroupShuffleXor(v, 4);
    v += subgroupShuffleXor(v, 2);
    v += subgroupShuffleXor(v, 1);
    return v;
}

// ============================================================================
// FMA Helper Macros
// ============================================================================

// Explicit FMA for 2-element vectors
// Usage: FMA2(sum, weights, activations)
#define FMA2(sum, w, a) sum = fma(w.x, a.x, sum); sum = fma(w.y, a.y, sum)

// Explicit FMA for 4-element vectors
#define FMA4(sum, w, a) sum = fma(w.x, a.x, sum); sum = fma(w.y, a.y, sum); sum = fma(w.z, a.z, sum); sum = fma(w.w, a.w, sum)

#endif // DEQUANT_HELPERS_GLSL
