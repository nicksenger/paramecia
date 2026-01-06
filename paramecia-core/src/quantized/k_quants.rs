use super::utils::{
    get_scale_min_k4, group_for_dequantization, group_for_quantization, make_q3_quants,
    make_qkx1_quants, make_qx_quants, nearest_int,
};
use super::GgmlDType;
use crate::quantized::utils::{make_qkx3_quants, make_qp_quants};
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};
use half::{bf16, f16, slice::HalfFloatSliceExt};
use rayon::prelude::*;

// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

/// Stochastic rounding utilities for QuZO (Quantized Zeroth-Order optimization).
///
/// These functions implement unbiased stochastic rounding where E[round(x)] = x,
/// enabling gradient-free optimization of discrete quantized weights.
pub mod stochastic {
    /// Simple linear congruential generator for fast, reproducible random numbers.
    /// Uses the same constants as glibc.
    #[inline]
    pub fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }

    /// Generate a uniform random float in [0, 1) from LCG state.
    #[inline]
    pub fn lcg_uniform(state: &mut u64) -> f32 {
        let val = lcg_next(state);
        // Use upper 23 bits for mantissa
        (val >> 41) as f32 / (1u64 << 23) as f32
    }

    /// Stochastic rounding for 4-bit unsigned values (0-15 range).
    ///
    /// Given a float x, returns floor(x) with probability (ceil(x) - x),
    /// and ceil(x) with probability (x - floor(x)).
    /// This ensures E[stochastic_round(x)] = x.
    #[inline]
    pub fn stochastic_round_u4(x: f32, rng_state: &mut u64) -> u8 {
        let floor = x.floor();
        let frac = x - floor;
        let rounded = if lcg_uniform(rng_state) < frac {
            floor as i32 + 1
        } else {
            floor as i32
        };
        rounded.clamp(0, 15) as u8
    }

    /// Stochastic rounding for 4-bit signed values (-8 to 7 range, stored as 0-15).
    #[inline]
    pub fn stochastic_round_i4(x: f32, rng_state: &mut u64) -> i8 {
        let floor = x.floor();
        let frac = x - floor;
        let rounded = if lcg_uniform(rng_state) < frac {
            floor as i32 + 1
        } else {
            floor as i32
        };
        rounded.clamp(-8, 7) as i8
    }

    /// Stochastic rounding for 8-bit signed values (-128 to 127 range).
    #[inline]
    pub fn stochastic_round_i8(x: f32, rng_state: &mut u64) -> i8 {
        let floor = x.floor();
        let frac = x - floor;
        let rounded = if lcg_uniform(rng_state) < frac {
            floor as i32 + 1
        } else {
            floor as i32
        };
        rounded.clamp(-128, 127) as i8
    }

    /// Stochastic rounding for 2-bit values (0-3 range).
    #[inline]
    pub fn stochastic_round_u2(x: f32, rng_state: &mut u64) -> u8 {
        let floor = x.floor();
        let frac = x - floor;
        let rounded = if lcg_uniform(rng_state) < frac {
            floor as i32 + 1
        } else {
            floor as i32
        };
        rounded.clamp(0, 3) as u8
    }

    /// Stochastic rounding for 6-bit values (0-63 range).
    #[inline]
    pub fn stochastic_round_u6(x: f32, rng_state: &mut u64) -> u8 {
        let floor = x.floor();
        let frac = x - floor;
        let rounded = if lcg_uniform(rng_state) < frac {
            floor as i32 + 1
        } else {
            floor as i32
        };
        rounded.clamp(0, 63) as u8
    }

    // ========================================================================
    // Philox 4x32-10 Counter-Based RNG (matches CUDA kernel)
    // ========================================================================
    // Philox is ideal for fused perturbation because:
    // - Stateless: generate random(seed, index) without maintaining state
    // - Deterministic: same (seed, index) always produces same value
    // - High quality: passes BigCrush statistical tests

    const PHILOX_M0: u32 = 0xD2511F53;
    const PHILOX_M1: u32 = 0xCD9E8D57;
    const PHILOX_W0: u32 = 0x9E3779B9;
    const PHILOX_W1: u32 = 0xBB67AE85;

    #[inline]
    fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
        let product = (a as u64) * (b as u64);
        ((product >> 32) as u32, product as u32)
    }

    #[inline]
    fn philox_round(c: &mut [u32; 4], k0: u32, k1: u32) {
        let (hi0, lo0) = mulhilo32(c[0], PHILOX_M0);
        let (hi1, lo1) = mulhilo32(c[2], PHILOX_M1);

        let new_c0 = hi1 ^ c[1] ^ k0;
        let new_c1 = lo1;
        let new_c2 = hi0 ^ c[3] ^ k1;
        let new_c3 = lo0;

        c[0] = new_c0;
        c[1] = new_c1;
        c[2] = new_c2;
        c[3] = new_c3;
    }

    /// Generate 4 uint32 random values from seed and index using Philox-4x32-10.
    #[inline]
    pub fn philox4x32(seed: u64, index: u64) -> [u32; 4] {
        let mut c = [index as u32, (index >> 32) as u32, 0u32, 0u32];

        let mut k0 = seed as u32;
        let mut k1 = (seed >> 32) as u32;

        for _ in 0..10 {
            philox_round(&mut c, k0, k1);
            k0 = k0.wrapping_add(PHILOX_W0);
            k1 = k1.wrapping_add(PHILOX_W1);
        }

        c
    }

    /// Generate Gaussian-distributed float using Box-Muller (matches CUDA kernel).
    #[inline]
    pub fn philox_gaussian(seed: u64, index: u64) -> f32 {
        let r = philox4x32(seed, index);

        // Box-Muller transform (matches CUDA kernel)
        let u1 = ((r[0] | 1) as f64) * (1.0 / 4294967296.0); // (0, 1]
        let u2 = (r[1] as f64) * (1.0 / 4294967296.0); // [0, 1)

        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;

        (radius * theta.cos()) as f32
    }

    // ========================================================================
    // Noise Lookup Table (LUT) for Fast Perturbation
    // ========================================================================
    // Pre-generated Gaussian noise table for fast perturbation lookups.
    // Much faster than Philox: ~3 ops vs ~30 ops per sample.
    // 64K entries = 256KB, fits comfortably in L2 cache.

    /// Size of the noise lookup table (power of 2 for fast modulo)
    pub const NOISE_TABLE_SIZE: usize = 65536;
    const NOISE_TABLE_MASK: usize = NOISE_TABLE_SIZE - 1;

    /// Pre-computed Gaussian noise table, generated lazily on first access.
    /// Uses Philox to generate high-quality initial values.
    pub fn get_noise_table() -> &'static [f32; NOISE_TABLE_SIZE] {
        use std::sync::OnceLock;
        static TABLE: OnceLock<Box<[f32; NOISE_TABLE_SIZE]>> = OnceLock::new();

        TABLE.get_or_init(|| {
            let mut table = Box::new([0.0f32; NOISE_TABLE_SIZE]);
            // Use a fixed seed for reproducibility
            const TABLE_SEED: u64 = 0xDEADBEEF_CAFEBABE;
            for i in 0..NOISE_TABLE_SIZE {
                table[i] = philox_gaussian(TABLE_SEED, i as u64);
            }
            table
        })
    }

    /// Fast Gaussian noise lookup using precomputed table.
    /// Uses a simple hash to combine seed and index for table lookup.
    #[inline]
    pub fn noise_lut(seed: u64, index: u64) -> f32 {
        // Fast hash combining seed and index
        // Using golden ratio constant for good mixing
        let hash = (seed ^ index).wrapping_mul(0x9E3779B97F4A7C15);
        let table_idx = (hash as usize) & NOISE_TABLE_MASK;
        get_noise_table()[table_idx]
    }

    /// Batch noise lookup - returns 8 consecutive noise values for SIMD processing.
    /// More efficient than 8 individual lookups due to better cache behavior.
    #[inline]
    pub fn noise_lut_batch_8(seed: u64, base_index: u64) -> [f32; 8] {
        let table = get_noise_table();
        let mut result = [0.0f32; 8];
        for (i, result) in result.iter_mut().enumerate() {
            let hash = (seed ^ (base_index + i as u64)).wrapping_mul(0x9E3779B97F4A7C15);
            let table_idx = (hash as usize) & NOISE_TABLE_MASK;
            *result = table[table_idx];
        }
        result
    }

    /// AVX2-optimized perturbation dot product: Σ(noise[i] * lhs[i])
    /// Uses gather instructions for vectorized LUT lookups and FMA for accumulation.
    #[cfg(target_feature = "avx2")]
    #[inline]
    pub fn perturb_dot_avx2(lhs_f32: &[f32], n: usize, seed: u64, col_offset: usize) -> f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        const HASH_MULT: u64 = 0x9E3779B97F4A7C15;

        unsafe {
            let table = get_noise_table();
            let table_ptr = table.as_ptr();

            let mut acc = _mm256_setzero_ps();
            let chunks = n / 8;
            let remainder = n % 8;

            // Process 8 elements at a time with AVX2
            for chunk in 0..chunks {
                let base_i = chunk * 8;
                let base_idx = (col_offset + base_i) as u64;

                // Compute 8 table indices
                // idx[i] = ((seed ^ (base_idx + i)) * HASH_MULT) & MASK
                let mut indices = [0i32; 8];
                for (i, index) in indices.iter_mut().enumerate() {
                    let hash = (seed ^ (base_idx + i as u64)).wrapping_mul(HASH_MULT);
                    *index = (hash as usize & NOISE_TABLE_MASK) as i32;
                }

                // Load indices into AVX register
                let idx_vec = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);

                // Gather 8 noise values from LUT using indices
                // _mm256_i32gather_ps(base, indices, scale) - scale=4 for f32
                let noise = _mm256_i32gather_ps(table_ptr, idx_vec, 4);

                // Load 8 lhs values
                let lhs = _mm256_loadu_ps(lhs_f32.as_ptr().add(base_i));

                // Multiply-accumulate: acc += noise * lhs
                acc = _mm256_fmadd_ps(noise, lhs, acc);
            }

            // Horizontal sum of accumulator
            // acc = [a0, a1, a2, a3, a4, a5, a6, a7]
            let hi = _mm256_extractf128_ps(acc, 1);
            let lo = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(lo, hi);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_movehdup_ps(sum64));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remaining elements with scalar code
            let start = chunks * 8;
            for i in 0..remainder {
                let idx = start + i;
                let hash = (seed ^ ((col_offset + idx) as u64)).wrapping_mul(HASH_MULT);
                let table_idx = (hash as usize) & NOISE_TABLE_MASK;
                result += table[table_idx] * lhs_f32[idx];
            }

            result
        }
    }

    /// Fallback for non-AVX2 platforms
    #[cfg(not(target_feature = "avx2"))]
    #[inline]
    pub fn perturb_dot_avx2(lhs_f32: &[f32], n: usize, seed: u64, col_offset: usize) -> f32 {
        let mut sum = 0f32;
        for i in 0..n {
            sum += noise_lut(seed, (col_offset + i) as u64) * lhs_f32[i];
        }
        sum
    }
}

// Re-export at module level for use in trait default impl and other modules
pub use stochastic::{noise_lut, perturb_dot_avx2, philox_gaussian};

pub trait GgmlType: Sized + Clone + Send + Sync {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;
    const DIRECT_COPY: bool = false;
    type VecDotType: GgmlType;

    // This is only safe for types that include immediate values such as float/int/...
    fn zeros() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
    fn to_float(xs: &[Self], ys: &mut [f32]);
    fn from_float(xs: &[f32], ys: &mut [Self]);
    fn from_float_imatrix(
        _xs: &[f32],
        _ys: &mut [Self],
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) {
        panic!(
            "`from_float_imatrix` is unimplemented for {:?}",
            Self::DTYPE
        );
    }

    fn direct_copy(_xs: &[f32], _ys: &mut [Self]) {}

    /// Dot product used as a building block for quantized mat-mul.
    /// n is the number of elements to be considered.
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32;

    /// Generic implementation of the dot product without simd optimizations.
    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32;

    /// Perturbed dot product for fused QuZO training.
    ///
    /// Computes: vec_dot(xs, ys) + ε * Σ(z[i] * lhs_f32[i])
    /// where z[i] is looked up from a precomputed noise table.
    ///
    /// This preserves the SIMD benefits of quantized vec_dot while adding the
    /// perturbation correction term. Uses a noise LUT for fast random access
    /// with AVX2 SIMD acceleration when available.
    ///
    /// # Arguments
    /// * `n` - Number of elements
    /// * `xs` - Quantized weight blocks
    /// * `ys` - Quantized input blocks (VecDotType)
    /// * `lhs_f32` - Original f32 input values (length n)
    /// * `seed` - Seed for noise lookup (should be XORed with tensor pointer for uniqueness)
    /// * `col_offset` - Column offset for noise indexing (col_idx * k)
    /// * `epsilon` - Perturbation magnitude
    #[inline]
    #[allow(unreachable_code)]
    fn vec_dot_perturbed(
        n: usize,
        xs: &[Self],
        ys: &[Self::VecDotType],
        lhs_f32: &[f32],
        seed: u64,
        col_offset: usize,
        epsilon: f32,
    ) -> f32 {
        // Regular quantized dot product (SIMD-optimized)
        let base = Self::vec_dot(n, xs, ys);

        // Perturbation correction: ε * Σ(z[i] * lhs_f32[i])
        // Use AVX2 SIMD when available for ~4x speedup
        #[cfg(target_feature = "avx2")]
        {
            let perturb_sum = stochastic::perturb_dot_avx2(lhs_f32, n, seed, col_offset);
            return base + epsilon * perturb_sum;
        }

        // Fallback: scalar with unrolling
        use stochastic::noise_lut;
        let mut perturb_sum = 0f32;
        let chunks = n / 8;
        let remainder = n % 8;

        for chunk in 0..chunks {
            let base_i = chunk * 8;
            let base_idx = (col_offset + base_i) as u64;
            let lhs_chunk = &lhs_f32[base_i..base_i + 8];

            perturb_sum += noise_lut(seed, base_idx) * lhs_chunk[0];
            perturb_sum += noise_lut(seed, base_idx + 1) * lhs_chunk[1];
            perturb_sum += noise_lut(seed, base_idx + 2) * lhs_chunk[2];
            perturb_sum += noise_lut(seed, base_idx + 3) * lhs_chunk[3];
            perturb_sum += noise_lut(seed, base_idx + 4) * lhs_chunk[4];
            perturb_sum += noise_lut(seed, base_idx + 5) * lhs_chunk[5];
            perturb_sum += noise_lut(seed, base_idx + 6) * lhs_chunk[6];
            perturb_sum += noise_lut(seed, base_idx + 7) * lhs_chunk[7];
        }

        let start = chunks * 8;
        for i in 0..remainder {
            let idx = start + i;
            perturb_sum += noise_lut(seed, (col_offset + idx) as u64) * lhs_f32[idx];
        }

        base + epsilon * perturb_sum
    }

    /// Extract the primary scaling factors from all blocks.
    fn extract_scales(xs: &[Self]) -> Vec<f32>;

    /// Dequantize using custom scaling factors.
    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]);

    /// Fused dequantize + perturb operation for QuZO.
    ///
    /// Dequantizes and applies perturbation in a single pass using Philox RNG.
    /// The perturbation z is generated on-the-fly: value = dequant + epsilon * z
    /// where z = philox_gaussian(seed ^ tensor_ptr, element_index).
    ///
    /// This is more efficient than separate dequantize + perturb for CPU tensors
    /// during fused QuZO training.
    ///
    /// # Arguments
    /// * `xs` - Quantized blocks
    /// * `ys` - Output buffer for dequantized+perturbed values
    /// * `seed` - Philox seed (should be XORed with tensor pointer for uniqueness)
    /// * `epsilon` - Perturbation magnitude (can be negative for -ε direction)
    fn to_float_perturbed(xs: &[Self], ys: &mut [f32], seed: u64, epsilon: f32) {
        // Default: dequantize then perturb separately (less efficient)
        Self::to_float(xs, ys);
        for (i, y) in ys.iter_mut().enumerate() {
            let z = philox_gaussian(seed, i as u64);
            *y += epsilon * z;
        }
    }

    /// Apply stochastic perturbation to discrete quantized weights (QuZO).
    ///
    /// This method perturbs the quantized values directly using stochastic rounding,
    /// enabling gradient-free optimization of discrete weights as described in the
    /// QuZO paper (arXiv:2502.12346).
    ///
    /// # Arguments
    /// * `xs` - Mutable slice of quantized blocks to perturb
    /// * `perturbation` - Continuous perturbation values (one per element, not per block)
    /// * `epsilon` - Perturbation magnitude
    /// * `seed` - Random seed for stochastic rounding (use different seeds for u_{i,1} and u_{i,2})
    /// * `add` - If true, add perturbation; if false, subtract
    fn apply_perturbation_stochastic(
        _xs: &mut [Self],
        _perturbation: &[f32],
        _epsilon: f32,
        _seed: u64,
        _add: bool,
    ) {
        // Default implementation does nothing - override for supported types
    }

    /// Apply a quantized update to discrete weights using stochastic rounding (QuZO).
    ///
    /// This is used in the update step: w̄ ← w̄ - Q(η·μ/n · u_{i,2})
    /// The seed should be different from the one used in forward perturbation.
    ///
    /// # Arguments
    /// * `xs` - Mutable slice of quantized blocks to update
    /// * `perturbation` - Continuous perturbation direction u_i
    /// * `scale` - Update scale (typically η·μ/n where η=lr, μ=directional derivative, n=num_perturbations)
    /// * `seed` - Random seed for stochastic rounding (must differ from forward pass seed!)
    fn apply_quantized_update(_xs: &mut [Self], _perturbation: &[f32], _scale: f32, _seed: u64) {
        // Default implementation does nothing - override for supported types
    }

    /// Check if this type supports QuZO-style discrete weight perturbation.
    fn supports_quzo() -> bool {
        false
    }

    /// Apply a quantized update with accumulated error feedback (QES-style).
    ///
    /// The `gain` parameter controls how much the accumulated residual influences rounding:
    /// - gain=0.0: purely stochastic rounding (residual tracked but doesn't affect rounding)
    /// - gain=1.0: full error feedback (QES behavior)
    /// - intermediate: tunable hybrid
    ///
    /// The residual update always tracks the true error regardless of gain:
    /// `residuals[j] = raw_desired + residuals[j] - actual_delta`
    fn apply_quantized_update_with_residual(
        _xs: &mut [Self],
        _perturbation: &[f32],
        _scale: f32,
        _seed: u64,
        _residuals: &mut [f16],
        _gain: f32,
    ) {
        // Default: delegate to non-residual version (ignores residuals)
        Self::apply_quantized_update(_xs, _perturbation, _scale, _seed);
    }

    /// Simulate a quantized update with residual tracking (read-only weights).
    ///
    /// Same logic as `apply_quantized_update_with_residual` but does not modify weights.
    /// Used for replay reconstruction — simulates what the rounding would produce
    /// using current weights as proxy for historical weights.
    fn simulate_update_with_residual(
        _xs: &[Self],
        _perturbation: &[f32],
        _scale: f32,
        _seed: u64,
        _residuals: &mut [f16],
        _gain: f32,
    ) {
        // Default: no-op for unsupported types
    }

    /// Combined restore-and-update operation for QuZO (more efficient than separate calls).
    ///
    /// This performs two operations in a single pass:
    /// 1. Restore: w̄ ← w̄ + ε·Q_1(u) (go from -ε to 0)
    /// 2. Update: w̄ ← w̄ - Q_2(scale·u) (apply gradient update)
    ///
    /// # Arguments
    /// * `xs` - Mutable slice of quantized blocks
    /// * `perturbation` - Continuous perturbation values
    /// * `restore_epsilon` - Epsilon for restore step
    /// * `update_scale` - Scale for gradient update
    /// * `restore_seed` - Seed for restore rounding
    /// * `update_seed` - Seed for update rounding (must differ from restore_seed!)
    fn apply_restore_and_update(
        xs: &mut [Self],
        perturbation: &[f32],
        restore_epsilon: f32,
        update_scale: f32,
        restore_seed: u64,
        update_seed: u64,
    ) {
        // Default: just call the two operations sequentially
        Self::apply_perturbation_stochastic(xs, perturbation, restore_epsilon, restore_seed, true);
        Self::apply_quantized_update(xs, perturbation, update_scale, update_seed);
    }

    /// Combined restore-and-update with error feedback.
    #[allow(clippy::too_many_arguments)]
    fn apply_restore_and_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        restore_epsilon: f32,
        update_scale: f32,
        restore_seed: u64,
        update_seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        Self::apply_perturbation_stochastic(xs, perturbation, restore_epsilon, restore_seed, true);
        Self::apply_quantized_update_with_residual(
            xs,
            perturbation,
            update_scale,
            update_seed,
            residuals,
            gain,
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub(crate) d: f16,
    pub(crate) s: f16,
    pub(crate) qs: [i8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ2K {
    pub(crate) scales: [u8; QK_K / 16],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
    pub(crate) dmin: f16,
}
const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ3K {
    pub(crate) hmask: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) scales: [u8; 12],
    pub(crate) d: f16,
}
const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

#[derive(Debug, Clone, PartialEq)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub struct BlockQ4K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ5K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qh: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () =
    assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ6K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
const _: () = assert!(4 + QK_K + QK_K / 16 * 2 == std::mem::size_of::<BlockQ8K>());

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK4_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        debug_assert!(
            k.is_multiple_of(qk),
            "dequantize_row_q4_0: {k} is not divisible by {qk}"
        );

        let nb = k / qk;
        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..(qk / 2) {
                let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (xs[i].qs[j] >> 4) as i16 - 8;

                ys[i * qk + j] = (x0 as f32) * d;
                ys[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q4_0
        let qk = Self::BLCK_SIZE;
        let k = xs.len();
        debug_assert!(k.is_multiple_of(qk), "{k} is not divisible by {qk}");
        debug_assert_eq!(
            ys.len(),
            k / qk,
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;

            let xs = &xs[i * qk..(i + 1) * qk];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);

            for (j, q) in ys.qs.iter_mut().enumerate() {
                let x0 = xs[j] * id;
                let x1 = xs[qk / 2 + j] * id;
                let xi0 = u8::min(15, (x0 + 8.5) as u8);
                let xi1 = u8::min(15, (x1 + 8.5) as u8);
                *q = xi0 | (xi1 << 4)
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L2361C10-L2361C122
    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q4_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q4_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q4_0_q8_0(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK8_0),
            "vec_dot_q4_0_q8_0: {n} is not divisible by {QK8_0}"
        );
        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sum_i = 0;
            for j in 0..QK8_0 / 2 {
                let v0 = (xs.qs[j] & 0x0F) as i32 - 8;
                let v1 = (xs.qs[j] >> 4) as i32 - 8;
                sum_i += v0 * ys.qs[j] as i32 + v1 * ys.qs[j + QK8_0 / 2] as i32
            }
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        debug_assert!(
            k.is_multiple_of(qk),
            "dequantize_row_q4_0: {k} is not divisible by {qk}"
        );
        let nb = k / qk;
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );

        for i in 0..nb {
            let d = scales[i];
            for j in 0..(qk / 2) {
                let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (xs[i].qs[j] >> 4) as i16 - 8;
                ys[i * qk + j] = (x0 as f32) * d;
                ys[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
    }

    fn apply_perturbation_stochastic(
        xs: &mut [Self],
        perturbation: &[f32],
        epsilon: f32,
        seed: u64,
        add: bool,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i4};
        let qk = Self::BLCK_SIZE;
        let mut rng_state = seed;
        // Warm up RNG
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * qk;

            for j in 0..(qk / 2) {
                // Extract current 4-bit values (stored as 0-15, represent -8 to 7)
                let q0 = (block.qs[j] & 0x0F) as i8 - 8;
                let q1 = (block.qs[j] >> 4) as i8 - 8;

                // Get perturbation values for these positions
                let p0_idx = perturb_offset + j;
                let p1_idx = perturb_offset + j + qk / 2;

                if p0_idx < perturbation.len() && p1_idx < perturbation.len() {
                    let p0 = perturbation[p0_idx] * epsilon;
                    let p1 = perturbation[p1_idx] * epsilon;

                    // Stochastically round the perturbation
                    let delta0 = stochastic_round_i4(p0, &mut rng_state);
                    let delta1 = stochastic_round_i4(p1, &mut rng_state);

                    // Apply perturbation
                    let new_q0 = if add {
                        (q0 as i16 + delta0 as i16).clamp(-8, 7) as i8
                    } else {
                        (q0 as i16 - delta0 as i16).clamp(-8, 7) as i8
                    };
                    let new_q1 = if add {
                        (q1 as i16 + delta1 as i16).clamp(-8, 7) as i8
                    } else {
                        (q1 as i16 - delta1 as i16).clamp(-8, 7) as i8
                    };

                    // Pack back (convert from -8..7 to 0..15)
                    let packed0 = (new_q0 + 8) as u8;
                    let packed1 = (new_q1 + 8) as u8;
                    block.qs[j] = (packed0 & 0x0F) | ((packed1 & 0x0F) << 4);
                }
            }
        }
    }

    fn apply_quantized_update(xs: &mut [Self], perturbation: &[f32], scale: f32, seed: u64) {
        Self::apply_perturbation_stochastic(xs, perturbation, -scale, seed, true);
    }

    fn apply_quantized_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i4};
        let qk = Self::BLCK_SIZE;
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * qk;

            for j in 0..(qk / 2) {
                let q0 = (block.qs[j] & 0x0F) as i8 - 8;
                let q1 = (block.qs[j] >> 4) as i8 - 8;

                let p0_idx = perturb_offset + j;
                let p1_idx = perturb_offset + j + qk / 2;

                if p0_idx < perturbation.len() && p1_idx < perturbation.len() {
                    // Element 0
                    let raw0 = perturbation[p0_idx] * (-scale);
                    let desired0 = raw0 + gain * residuals[p0_idx].to_f32();
                    let delta0 = stochastic_round_i4(desired0, &mut rng_state);
                    let new_q0 = (q0 as i16 + delta0 as i16).clamp(-8, 7) as i8;
                    let actual0 = (new_q0 - q0) as f32;
                    residuals[p0_idx] = f16::from_f32(raw0 + residuals[p0_idx].to_f32() - actual0);

                    // Element 1
                    let raw1 = perturbation[p1_idx] * (-scale);
                    let desired1 = raw1 + gain * residuals[p1_idx].to_f32();
                    let delta1 = stochastic_round_i4(desired1, &mut rng_state);
                    let new_q1 = (q1 as i16 + delta1 as i16).clamp(-8, 7) as i8;
                    let actual1 = (new_q1 - q1) as f32;
                    residuals[p1_idx] = f16::from_f32(raw1 + residuals[p1_idx].to_f32() - actual1);

                    let packed0 = (new_q0 + 8) as u8;
                    let packed1 = (new_q1 + 8) as u8;
                    block.qs[j] = (packed0 & 0x0F) | ((packed1 & 0x0F) << 4);
                }
            }
        }
    }

    fn simulate_update_with_residual(
        xs: &[Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i4};
        let qk = Self::BLCK_SIZE;
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter().enumerate() {
            let perturb_offset = block_idx * qk;

            for j in 0..(qk / 2) {
                let q0 = (block.qs[j] & 0x0F) as i8 - 8;
                let q1 = (block.qs[j] >> 4) as i8 - 8;

                let p0_idx = perturb_offset + j;
                let p1_idx = perturb_offset + j + qk / 2;

                if p0_idx < perturbation.len() && p1_idx < perturbation.len() {
                    let raw0 = perturbation[p0_idx] * (-scale);
                    let desired0 = raw0 + gain * residuals[p0_idx].to_f32();
                    let delta0 = stochastic_round_i4(desired0, &mut rng_state);
                    let new_q0 = (q0 as i16 + delta0 as i16).clamp(-8, 7) as i8;
                    let actual0 = (new_q0 - q0) as f32;
                    residuals[p0_idx] = f16::from_f32(raw0 + residuals[p0_idx].to_f32() - actual0);

                    let raw1 = perturbation[p1_idx] * (-scale);
                    let desired1 = raw1 + gain * residuals[p1_idx].to_f32();
                    let delta1 = stochastic_round_i4(desired1, &mut rng_state);
                    let new_q1 = (q1 as i16 + delta1 as i16).clamp(-8, 7) as i8;
                    let actual1 = (new_q1 - q1) as f32;
                    residuals[p1_idx] = f16::from_f32(raw1 + residuals[p1_idx].to_f32() - actual1);
                }
            }
        }
    }

    fn supports_quzo() -> bool {
        true
    }
}

impl GgmlType for BlockQ4_1 {
    const DTYPE: GgmlDType = GgmlDType::Q4_1;
    const BLCK_SIZE: usize = QK4_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        // ggml_vec_dot_q4_1_q8_1
        let qk = QK8_1;
        debug_assert!(
            n.is_multiple_of(qk),
            "vec_dot_q4_1_q8_1: {n} is not divisible by {qk}"
        );
        debug_assert!(
            (n / qk).is_multiple_of(2),
            "vec_dot_q4_1_q8_1: {n}, nb is not divisible by 2"
        );

        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sumi = 0i32;

            for j in 0..qk / 2 {
                let v0 = xs.qs[j] as i32 & 0x0F;
                let v1 = xs.qs[j] as i32 >> 4;
                sumi += (v0 * ys.qs[j] as i32) + (v1 * ys.qs[j + qk / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
                + f16::to_f32(xs.m) * f16::to_f32(ys.s)
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q4_1
        let qk = Self::BLCK_SIZE;

        debug_assert_eq!(
            ys.len() * qk,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * qk..(i + 1) * qk];

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &x in xs.iter() {
                min = f32::min(x, min);
                max = f32::max(x, max);
            }
            let d = (max - min) / ((1 << 4) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            ys.m = f16::from_f32(min);

            for (j, q) in ys.qs.iter_mut().take(qk / 2).enumerate() {
                let x0 = (xs[j] - min) * id;
                let x1 = (xs[qk / 2 + j] - min) * id;

                let xi0 = u8::min(15, (x0 + 0.5) as u8);
                let xi1 = u8::min(15, (x1 + 0.5) as u8);

                *q = xi0 | (xi1 << 4);
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1545
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK4_1),
            "dequantize_row_q4_1: {k} is not divisible by {QK4_1}"
        );

        let nb = k / QK4_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();

            for j in 0..(QK4_1 / 2) {
                let x0 = xs[i].qs[j] & 0x0F;
                let x1 = xs[i].qs[j] >> 4;

                ys[i * QK4_1 + j] = (x0 as f32) * d + m;
                ys[i * QK4_1 + j + QK4_1 / 2] = (x1 as f32) * d + m;
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK4_1),
            "dequantize_row_q4_1: {k} is not divisible by {QK4_1}"
        );
        let nb = k / QK4_1;
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );

        for i in 0..nb {
            let d = scales[i];
            let m = xs[i].m.to_f32();
            for j in 0..(QK4_1 / 2) {
                let x0 = xs[i].qs[j] & 0x0F;
                let x1 = xs[i].qs[j] >> 4;
                ys[i * QK4_1 + j] = (x0 as f32) * d + m;
                ys[i * QK4_1 + j + QK4_1 / 2] = (x1 as f32) * d + m;
            }
        }
    }
}

impl GgmlType for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLCK_SIZE: usize = QK5_0;
    type VecDotType = BlockQ8_0;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        let qk = Self::BLCK_SIZE;

        debug_assert!(
            n.is_multiple_of(qk),
            "vec_dot_q5_0_q8_0: {n} is not divisible by {qk}"
        );
        debug_assert!(
            (n / qk).is_multiple_of(2),
            "vec_dot_q5_0_q8_0: {n}, nb is not divisible by 2"
        );
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(_n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let qh = LittleEndian::read_u32(&xs.qh);
            let mut sumi = 0i32;

            for j in 0..Self::BLCK_SIZE / 2 {
                let xh_0 = (((qh & (1u32 << j)) >> j) << 4) as u8;
                let xh_1 = ((qh & (1u32 << (j + 16))) >> (j + 12)) as u8;

                let x0 = ((xs.qs[j] & 0x0F) as i32 | xh_0 as i32) - 16;
                let x1 = ((xs.qs[j] >> 4) as i32 | xh_1 as i32) - 16;

                sumi += (x0 * ys.qs[j] as i32) + (x1 * ys.qs[j + Self::BLCK_SIZE / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q5_0
        debug_assert_eq!(
            ys.len() * Self::BLCK_SIZE,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];

            let mut amax = 0f32;
            let mut max = 0f32;
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -16.;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            let mut qh = 0u32;
            for j in 0..Self::BLCK_SIZE / 2 {
                let x0 = xs[j] * id;
                let x1 = xs[j + Self::BLCK_SIZE / 2] * id;
                let xi0 = ((x0 + 16.5) as i8).min(31) as u8;
                let xi1 = ((x1 + 16.5) as i8).min(31) as u8;
                ys.qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
                qh |= ((xi0 as u32 & 0x10) >> 4) << j;
                qh |= ((xi1 as u32 & 0x10) >> 4) << (j + Self::BLCK_SIZE / 2);
            }
            LittleEndian::write_u32(&mut ys.qh, qh)
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1566
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK5_0),
            "dequantize_row_q5_0: {k} is not divisible by {QK5_0}"
        );
        let nb = k / QK5_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);

            for j in 0..(QK5_0 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = ((xs[i].qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((xs[i].qs[j] >> 4) | xh_1) as i32 - 16;

                ys[i * QK5_0 + j] = (x0 as f32) * d;
                ys[i * QK5_0 + j + QK5_0 / 2] = (x1 as f32) * d;
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK5_0),
            "dequantize_row_q5_0: {k} is not divisible by {QK5_0}"
        );
        let nb = k / QK5_0;
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );

        for i in 0..nb {
            let d = scales[i];
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);
            for j in 0..(QK5_0 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
                let x0 = ((xs[i].qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((xs[i].qs[j] >> 4) | xh_1) as i32 - 16;
                ys[i * QK5_0 + j] = (x0 as f32) * d;
                ys[i * QK5_0 + j + QK5_0 / 2] = (x1 as f32) * d;
            }
        }
    }
}

impl GgmlType for BlockQ5_1 {
    const DTYPE: GgmlDType = GgmlDType::Q5_1;
    const BLCK_SIZE: usize = QK5_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        let qk = Self::BLCK_SIZE;
        debug_assert!(
            n.is_multiple_of(qk),
            "vec_dot_q5_1_q8_1: {n} is not divisible by {qk}"
        );
        debug_assert!(
            (n / qk).is_multiple_of(2),
            "vec_dot_q5_1_q8_1: {n}, nb is not divisible by 2"
        );

        // Generic implementation.
        let mut sumf = 0f32;

        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let qh = LittleEndian::read_u32(&xs.qh);
            let mut sumi = 0i32;

            for j in 0..Self::BLCK_SIZE / 2 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;

                let x0 = (xs.qs[j] as i32 & 0xF) | xh_0 as i32;
                let x1 = (xs.qs[j] as i32 >> 4) | xh_1 as i32;

                sumi += (x0 * ys.qs[j] as i32) + (x1 * ys.qs[j + Self::BLCK_SIZE / 2] as i32);
            }

            sumf += sumi as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
                + f16::to_f32(xs.m) * f16::to_f32(ys.s)
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q5_1
        let qk = Self::BLCK_SIZE;
        debug_assert_eq!(
            ys.len() * qk,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * qk..(i + 1) * qk];

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &x in xs.iter() {
                min = f32::min(x, min);
                max = f32::max(x, max);
            }
            let d = (max - min) / ((1 << 5) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            ys.m = f16::from_f32(min);

            let mut qh = 0u32;
            for (j, q) in ys.qs.iter_mut().take(qk / 2).enumerate() {
                let x0 = (xs[j] - min) * id;
                let x1 = (xs[qk / 2 + j] - min) * id;

                let xi0 = (x0 + 0.5) as u8;
                let xi1 = (x1 + 0.5) as u8;

                *q = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
                // get the 5-th bit and store it in qh at the right position
                qh |= ((xi0 as u32 & 0x10) >> 4) << j;
                qh |= ((xi1 as u32 & 0x10) >> 4) << (j + qk / 2);
            }
            LittleEndian::write_u32(&mut ys.qh, qh);
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1592
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK5_1),
            "dequantize_row_q5_1: {k} is not divisible by {QK5_1}"
        );

        let nb = k / QK5_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);

            for j in 0..(QK5_1 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = (xs[i].qs[j] & 0x0F) | xh_0;
                let x1 = (xs[i].qs[j] >> 4) | xh_1;

                ys[i * QK5_1 + j] = (x0 as f32) * d + m;
                ys[i * QK5_1 + j + QK5_1 / 2] = (x1 as f32) * d + m;
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK5_1),
            "dequantize_row_q5_1: {k} is not divisible by {QK5_1}"
        );
        let nb = k / QK5_1;
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );

        for i in 0..nb {
            let d = scales[i];
            let m = xs[i].m.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);
            for j in 0..(QK5_1 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
                let x0 = (xs[i].qs[j] & 0x0F) | xh_0;
                let x1 = (xs[i].qs[j] >> 4) | xh_1;
                ys[i * QK5_1 + j] = (x0 as f32) * d + m;
                ys[i * QK5_1 + j + QK5_1 / 2] = (x1 as f32) * d + m;
            }
        }
    }
}

impl GgmlType for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0;
    type VecDotType = BlockQ8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1619
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK8_0),
            "dequantize_row_q8_0: {k} is not divisible by {QK8_0}"
        );

        let nb = k / QK8_0;

        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..QK8_0 {
                ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q8_0
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(Self::BLCK_SIZE),
            "{k} is not divisible by {}",
            Self::BLCK_SIZE
        );
        debug_assert_eq!(
            ys.len(),
            k / Self::BLCK_SIZE,
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f32::round(x * id) as i8
            }
        }
    }

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q8_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q8_0_q8_0(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q8_0_q8_0(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK8_0),
            "vec_dot_q8_0_q8_0: {n} is not divisible by {QK8_0}"
        );

        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let sum_i = xs
                .qs
                .iter()
                .zip(ys.qs.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>();
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK8_0),
            "dequantize_row_q8_0: {k} is not divisible by {QK8_0}"
        );
        let nb = k / QK8_0;
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );

        for i in 0..nb {
            let d = scales[i];
            for j in 0..QK8_0 {
                ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
            }
        }
    }

    fn apply_perturbation_stochastic(
        xs: &mut [Self],
        perturbation: &[f32],
        epsilon: f32,
        seed: u64,
        add: bool,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i8};

        let mut rng_state = seed;
        // Warm up RNG
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK8_0;

            for j in 0..QK8_0 {
                let idx = perturb_offset + j;
                if idx < perturbation.len() {
                    let p = perturbation[idx] * epsilon;
                    let delta = stochastic_round_i8(p, &mut rng_state);

                    let current = block.qs[j];
                    let new_val = if add {
                        (current as i16 + delta as i16).clamp(-128, 127) as i8
                    } else {
                        (current as i16 - delta as i16).clamp(-128, 127) as i8
                    };
                    block.qs[j] = new_val;
                }
            }
        }
    }

    fn apply_quantized_update(xs: &mut [Self], perturbation: &[f32], scale: f32, seed: u64) {
        Self::apply_perturbation_stochastic(xs, perturbation, -scale, seed, true);
    }

    fn apply_quantized_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i8};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK8_0;
            for j in 0..QK8_0 {
                let idx = perturb_offset + j;
                if idx < perturbation.len() {
                    let current = block.qs[j];
                    let raw = perturbation[idx] * (-scale);
                    let desired = raw + gain * residuals[idx].to_f32();
                    let delta = stochastic_round_i8(desired, &mut rng_state);
                    let new_val = (current as i16 + delta as i16).clamp(-128, 127) as i8;
                    let actual = (new_val - current) as f32;
                    residuals[idx] = f16::from_f32(raw + residuals[idx].to_f32() - actual);
                    block.qs[j] = new_val;
                }
            }
        }
    }

    fn simulate_update_with_residual(
        xs: &[Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i8};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter().enumerate() {
            let perturb_offset = block_idx * QK8_0;
            for j in 0..QK8_0 {
                let idx = perturb_offset + j;
                if idx < perturbation.len() {
                    let current = block.qs[j];
                    let raw = perturbation[idx] * (-scale);
                    let desired = raw + gain * residuals[idx].to_f32();
                    let delta = stochastic_round_i8(desired, &mut rng_state);
                    let new_val = (current as i16 + delta as i16).clamp(-128, 127) as i8;
                    let actual = (new_val - current) as f32;
                    residuals[idx] = f16::from_f32(raw + residuals[idx].to_f32() - actual);
                }
            }
        }
    }

    fn supports_quzo() -> bool {
        true
    }
}

impl GgmlType for BlockQ8_1 {
    const DTYPE: GgmlDType = GgmlDType::Q8_1;
    const BLCK_SIZE: usize = QK8_1;
    type VecDotType = BlockQ8_1;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK8_1),
            "vec_dot_q8_1_q8_1: {n} is not divisible by {QK8_1}"
        );

        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let sum_i = xs
                .qs
                .iter()
                .zip(ys.qs.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>();
            sumf += sum_i as f32 * f16::to_f32(xs.d) * f16::to_f32(ys.d)
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q8_1
        debug_assert_eq!(
            ys.len() * Self::BLCK_SIZE,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            let mut sum = 0i32;
            for j in 0..Self::BLCK_SIZE / 2 {
                let v0 = xs[j] * id;
                let v1 = xs[j + Self::BLCK_SIZE / 2] * id;
                ys.qs[j] = f32::round(v0) as i8;
                ys.qs[j + Self::BLCK_SIZE / 2] = f32::round(v1) as i8;
                sum += ys.qs[j] as i32 + ys.qs[j + Self::BLCK_SIZE / 2] as i32;
            }
            ys.s = f16::from_f32(sum as f32) * ys.d;
        }
    }

    fn to_float(_xs: &[Self], _ys: &mut [f32]) {
        unimplemented!("no support for vec-dot on Q8_1")
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(_xs: &[Self], _ys: &mut [f32], _scales: &[f32]) {
        unimplemented!("no support for to_float on Q8_1")
    }
}

impl GgmlType for BlockQ2K {
    const DTYPE: GgmlDType = GgmlDType::Q2K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q2k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q2k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q2k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q2k_q8k: {n} is not divisible by {QK_K}"
        );

        let mut sumf = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut q2: &[_] = &x.qs;
            let mut q8: &[_] = &y.qs;
            let sc = &x.scales;

            let mut summs = 0;
            for (bsum, scale) in y.bsums.iter().zip(sc) {
                summs += *bsum as i32 * ((scale >> 4) as i32);
            }

            let dall = y.d * x.d.to_f32();
            let dmin = y.d * x.dmin.to_f32();

            let mut isum = 0;
            let mut is = 0;
            for _ in 0..(QK_K / 128) {
                let mut shift = 0;
                for _ in 0..4 {
                    let d = (sc[is] & 0xF) as i32;
                    is += 1;
                    let mut isuml = 0;
                    for l in 0..16 {
                        isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                    }
                    isum += d * isuml;
                    let d = (sc[is] & 0xF) as i32;
                    is += 1;
                    isuml = 0;
                    for l in 16..32 {
                        isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                    }
                    isum += d * isuml;
                    shift += 2;
                    // adjust the indexing
                    q8 = &q8[32..];
                }
                // adjust the indexing
                q2 = &q2[32..];
            }
            sumf += dall * isum as f32 - dmin * summs as f32;
        }

        sumf
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L279
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        const Q4SCALE: f32 = 15.0;

        for (block, x) in group_for_quantization(xs, ys) {
            //calculate scales and mins
            let mut mins: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];

            for (j, x_scale_slice) in x.chunks(16).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(3, 5, x_scale_slice);
            }
            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            if max_scale > 0.0 {
                let iscale = Q4SCALE / max_scale;
                for (j, scale) in scales.iter().enumerate().take(QK_K / 16) {
                    block.scales[j] = nearest_int(iscale * scale) as u8;
                }
                block.d = f16::from_f32(max_scale / Q4SCALE);
            } else {
                for j in 0..QK_K / 16 {
                    block.scales[j] = 0;
                }
                block.d = f16::from_f32(0.0);
            }

            if max_min > 0.0 {
                let iscale = Q4SCALE / max_min;
                for (j, scale) in block.scales.iter_mut().enumerate() {
                    let l = nearest_int(iscale * mins[j]) as u8;
                    *scale |= l << 4;
                }
                block.dmin = f16::from_f32(max_min / Q4SCALE);
            } else {
                block.dmin = f16::from_f32(0.0);
            }

            let mut big_l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let d = block.d.to_f32() * (block.scales[j] & 0xF) as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * (block.scales[j] >> 4) as f32;
                for ii in 0..16 {
                    let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                    big_l[16 * j + ii] = ll as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for ll in 0..32 {
                    block.qs[j / 4 + ll] = big_l[j + ll]
                        | (big_l[j + ll + 32] << 2)
                        | (big_l[j + ll + 64] << 4)
                        | (big_l[j + ll + 96] << 6);
                }
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        for (sblk_idx, (block, x)) in group_for_quantization(xs, ys).into_iter().enumerate() {
            let mut mins: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut weights: [f32; 16] = [0.0; 16];
            let mut sw: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut ls: [u8; QK_K / 16] = [0; QK_K / 16];
            let mut lm: [u8; QK_K / 16] = [0; QK_K / 16];

            let sum_x2 = x.iter().map(|x| x * x).sum::<f32>();
            let sigma2 = sum_x2 / QK_K as f32;
            for (j, x_scale_slice) in x.chunks_exact(16).enumerate() {
                for (l, (w_elem, x_elem)) in weights.iter_mut().zip(x_scale_slice).enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let imatrix_w = imatrix_weights[imatrix_row * QK_K + 16 * j + l];
                    *w_elem = imatrix_w * (sigma2 + x_elem * x_elem).sqrt();
                }
                let sumw = weights.iter().sum::<f32>();
                sw[j] = sumw;
                (scales[j], mins[j]) =
                    make_qkx3_quants(3, x_scale_slice, Some(&weights), -0.9, 0.05, 36, false);
            }

            let d_block = make_qp_quants(QK_K / 16, 15, &scales, &mut ls, &sw);
            let m_block = make_qp_quants(QK_K / 16, 15, &mins, &mut lm, &sw);

            block.d = f16::from_f32(d_block);
            block.dmin = f16::from_f32(m_block);

            for j in 0..QK_K / 16 {
                block.scales[j] = ls[j] | (lm[j] << 4);
            }

            let mut big_l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let d = block.d.to_f32() * (block.scales[j] & 0xF) as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * (block.scales[j] >> 4) as f32;
                for ii in 0..16 {
                    let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                    big_l[16 * j + ii] = ll as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for ll in 0..32 {
                    block.qs[j / 4 + ll] = big_l[j + ll]
                        | (big_l[j + ll + 32] << 2)
                        | (big_l[j + ll + 64] << 4)
                        | (big_l[j + ll + 96] << 6);
                }
            }
        }
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L354
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();

            let mut is = 0;

            for (y_block, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
                // Step by 32 over q.
                let mut shift = 0;
                let mut y_block_index = 0;
                for _j in 0..4 {
                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[..16] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[16..] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    shift += 2;
                }
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        // Simplified implementation: multiply entire block by custom scale
        let orig_ys = vec![0.0f32; ys.len()];
        let mut orig_ys_slice = orig_ys.clone();
        Self::to_float(xs, &mut orig_ys_slice);
        let nb = xs.len();
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );
        for (i, block_ys) in orig_ys_slice.chunks(Self::BLCK_SIZE).enumerate() {
            let orig_scale = xs[i].d.to_f32();
            let scale_ratio = if orig_scale != 0.0 {
                scales[i] / orig_scale
            } else {
                0.0
            };
            for (j, &y) in block_ys.iter().enumerate() {
                ys[i * Self::BLCK_SIZE + j] = y * scale_ratio;
            }
        }
    }

    fn apply_perturbation_stochastic(
        xs: &mut [Self],
        perturbation: &[f32],
        epsilon: f32,
        seed: u64,
        add: bool,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u2};

        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            // Q2K: 2-bit values (0-3 range) packed 4 per byte in qs[64]
            // 256 elements per block, organized in 128-element chunks

            for (chunk_idx, qs_chunk) in block.qs.chunks_exact_mut(32).enumerate() {
                let base_elem = chunk_idx * 128;

                for shift in (0..8).step_by(2) {
                    // Process 32 elements at this shift position
                    for i in 0..32 {
                        // Calculate the element index within the block
                        // Each shift covers 32 elements
                        let elem_in_chunk = (shift / 2) * 32 + i;
                        let idx = perturb_offset + base_elem + elem_in_chunk;

                        if idx < perturbation.len() {
                            // Extract current 2-bit value
                            let qs_idx = i;
                            let q = (qs_chunk[qs_idx] >> shift) & 3;

                            let p = perturbation[idx] * epsilon;
                            let delta = stochastic_round_u2(p.abs(), &mut rng_state);
                            let delta = if p >= 0.0 {
                                delta as i8
                            } else {
                                -(delta as i8)
                            };

                            let new_q = if add {
                                (q as i8 + delta).clamp(0, 3) as u8
                            } else {
                                (q as i8 - delta).clamp(0, 3) as u8
                            };

                            // Update the 2-bit value
                            let mask = !(3u8 << shift);
                            qs_chunk[qs_idx] = (qs_chunk[qs_idx] & mask) | (new_q << shift);
                        }
                    }
                }
            }
        }
    }

    fn apply_quantized_update(xs: &mut [Self], perturbation: &[f32], scale: f32, seed: u64) {
        Self::apply_perturbation_stochastic(xs, perturbation, -scale, seed, true);
    }

    fn apply_quantized_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u2};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            for (chunk_idx, qs_chunk) in block.qs.chunks_exact_mut(32).enumerate() {
                let base_elem = chunk_idx * 128;

                for shift in (0..8).step_by(2) {
                    for i in 0..32 {
                        let elem_in_chunk = (shift / 2) * 32 + i;
                        let idx = perturb_offset + base_elem + elem_in_chunk;

                        if idx < perturbation.len() {
                            let qs_idx = i;
                            let q = (qs_chunk[qs_idx] >> shift) & 3;

                            let raw = perturbation[idx] * (-scale);
                            let desired = raw + gain * residuals[idx].to_f32();
                            // For unsigned formats, round abs and apply sign
                            let delta = stochastic_round_u2(desired.abs(), &mut rng_state);
                            let delta = if desired >= 0.0 {
                                delta as i8
                            } else {
                                -(delta as i8)
                            };

                            let new_q = (q as i8 + delta).clamp(0, 3) as u8;
                            let actual = (new_q as i8 - q as i8) as f32;
                            residuals[idx] = f16::from_f32(raw + residuals[idx].to_f32() - actual);

                            let mask = !(3u8 << shift);
                            qs_chunk[qs_idx] = (qs_chunk[qs_idx] & mask) | (new_q << shift);
                        }
                    }
                }
            }
        }
    }

    fn simulate_update_with_residual(
        xs: &[Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u2};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter().enumerate() {
            let perturb_offset = block_idx * QK_K;

            for (chunk_idx, qs_chunk) in block.qs.chunks_exact(32).enumerate() {
                let base_elem = chunk_idx * 128;

                for shift in (0..8).step_by(2) {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..32 {
                        let elem_in_chunk = (shift / 2) * 32 + i;
                        let idx = perturb_offset + base_elem + elem_in_chunk;

                        if idx < perturbation.len() {
                            let q = (qs_chunk[i] >> shift) & 3;

                            let raw = perturbation[idx] * (-scale);
                            let desired = raw + gain * residuals[idx].to_f32();
                            let delta = stochastic_round_u2(desired.abs(), &mut rng_state);
                            let delta = if desired >= 0.0 {
                                delta as i8
                            } else {
                                -(delta as i8)
                            };

                            let new_q = (q as i8 + delta).clamp(0, 3) as u8;
                            let actual = (new_q as i8 - q as i8) as f32;
                            residuals[idx] = f16::from_f32(raw + residuals[idx].to_f32() - actual);
                        }
                    }
                }
            }
        }
    }

    fn supports_quzo() -> bool {
        true
    }
}

impl GgmlType for BlockQ3K {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q3k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q3k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q3k_q8k: {n} is not divisible by {QK_K}"
        );

        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut auxs: [u32; 4] = [0; 4];

        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut q3: &[u8] = &x.qs;
            let hmask: &[u8] = &x.hmask;
            let mut q8: &[i8] = &y.qs;

            aux32.fill(0);
            let mut a = &mut aux8[..];

            let mut m = 1;
            //Like the GGML original this is written this way to enable the compiler to vectorize it.
            for _ in 0..QK_K / 128 {
                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = (q3_val & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 2) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 4) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;

                a.iter_mut()
                    .take(32)
                    .zip(q3)
                    .for_each(|(a_val, q3_val)| *a_val = ((q3_val >> 6) & 3) as i8);
                a.iter_mut()
                    .take(32)
                    .zip(hmask)
                    .for_each(|(a_val, hmask_val)| {
                        *a_val -= if hmask_val & m != 0 { 0 } else { 4 }
                    });
                a = &mut a[32..];
                m <<= 1;
                q3 = &q3[32..];
            }

            a = &mut aux8[..];

            LittleEndian::read_u32_into(&x.scales, &mut auxs[0..3]);

            let tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            auxs[3] = ((auxs[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            auxs[0] = (auxs[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            auxs[1] = (auxs[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

            for aux in auxs {
                for scale in aux.to_le_bytes() {
                    let scale = i8::from_be_bytes([scale]);
                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += (scale as i32 - 32) * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];

                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += (scale as i32 - 32) * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let d = x.d.to_f32() * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
        }

        sums.iter().sum()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            for (j, x_scale_slice) in x.chunks_exact(16).enumerate() {
                scales[j] = make_q3_quants(x_scale_slice, 4, true);
            }

            // Get max scale by absolute value.
            let mut max_scale: f32 = 0.0;
            for &scale in scales.iter() {
                if scale.abs() > max_scale.abs() {
                    max_scale = scale;
                }
            }

            block.scales.fill(0);

            if max_scale != 0.0 {
                let iscale = -32.0 / max_scale;
                for (j, scale) in scales.iter().enumerate() {
                    let l_val = nearest_int(iscale * scale);
                    let l_val = l_val.clamp(-32, 31) + 32;
                    if j < 8 {
                        block.scales[j] = (l_val & 0xF) as u8;
                    } else {
                        block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                    }
                    let l_val = l_val >> 4;
                    block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
                }
                block.d = f16::from_f32(1.0 / iscale);
            } else {
                block.d = f16::from_f32(0.0);
            }

            let mut l: [i8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let sc = if j < 8 {
                    block.scales[j] & 0xF
                } else {
                    block.scales[j - 8] >> 4
                };
                let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    for ii in 0..16 {
                        let l_val = nearest_int(x[16 * j + ii] / d);
                        l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                    }
                }
            }

            block.hmask.fill(0);
            let mut m = 0;
            let mut hm = 1;

            for ll in l.iter_mut() {
                if *ll > 3 {
                    block.hmask[m] |= hm;
                    *ll -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for l_val in 0..32 {
                    block.qs[j / 4 + l_val] = (l[j + l_val]
                        | (l[j + l_val + 32] << 2)
                        | (l[j + l_val + 64] << 4)
                        | (l[j + l_val + 96] << 6))
                        as u8;
                }
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        for (sblk_idx, (block, x)) in group_for_quantization(xs, ys).into_iter().enumerate() {
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut weights: [f32; 16] = [0.0; 16];
            let mut sw: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut ls: [i8; QK_K / 16] = [0; QK_K / 16];
            let mut l: [i8; QK_K] = [0; QK_K];

            let sum_x2 = x.iter().map(|x| x * x).sum::<f32>();
            let sigma2 = 2. * sum_x2 / QK_K as f32;

            for (j, x_scale_slice) in x.chunks_exact(16).enumerate() {
                for (l_idx, (w_elem, x_elem)) in weights.iter_mut().zip(x_scale_slice).enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let imatrix_w = imatrix_weights[imatrix_row * QK_K + 16 * j + l_idx];
                    *w_elem = imatrix_w * (sigma2 + x_elem * x_elem).sqrt();
                }
                let sumw = weights.iter().sum::<f32>();
                sw[j] = sumw;
                scales[j] = unsafe {
                    make_qx_quants(
                        16,
                        4,
                        x_scale_slice.as_ptr(),
                        l.as_mut_ptr().add(16 * j),
                        1,
                        weights.as_ptr(),
                    )
                };
            }

            block.scales.fill(0);
            let d_block = unsafe {
                make_qx_quants(
                    QK_K / 16,
                    32,
                    scales.as_ptr(),
                    ls.as_mut_ptr(),
                    1,
                    sw.as_ptr(),
                )
            };
            block.d = f16::from_f32(d_block);
            for (j, l_val) in ls.iter().enumerate().take(QK_K / 16) {
                if j < 8 {
                    block.scales[j] = (l_val & 0xF) as u8;
                } else {
                    block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                }
                let l_val = l_val >> 4;
                block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
            }

            for j in 0..QK_K / 16 {
                let sc = if j < 8 {
                    block.scales[j] & 0xF
                } else {
                    block.scales[j - 8] >> 4
                };
                let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    for ii in 0..16 {
                        let l_val = nearest_int(x[16 * j + ii] / d);
                        l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                    }
                }
            }

            block.hmask.fill(0);
            let mut m = 0;
            let mut hm = 1;

            for ll in l.iter_mut() {
                if *ll > 3 {
                    block.hmask[m] |= hm;
                    *ll -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for l_val in 0..32 {
                    block.qs[j / 4 + l_val] = (l[j + l_val]
                        | (l[j + l_val + 32] << 2)
                        | (l[j + l_val + 64] << 4)
                        | (l[j + l_val + 96] << 6))
                        as u8;
                }
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        for (block, y) in group_for_dequantization(xs, ys) {
            //Reconstruct the scales
            let mut aux = [0; 4];
            LittleEndian::read_u32_into(&block.scales, &mut aux[0..3]);

            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

            //Transfer the scales into an i8 array
            let scales: &mut [i8] =
                unsafe { std::slice::from_raw_parts_mut(aux.as_mut_ptr() as *mut i8, 16) };

            let d_all = block.d.to_f32();
            let mut m = 1;
            let mut is = 0;

            // Dequantize both 128 long blocks
            // 32 qs values per 128 long block
            // Each 16 elements get a scale
            for (y, qs) in y.chunks_exact_mut(128).zip(block.qs.chunks_exact(32)) {
                let mut shift = 0;
                for shift_scoped_y in y.chunks_exact_mut(32) {
                    for (scale_index, scale_scoped_y) in
                        shift_scoped_y.chunks_exact_mut(16).enumerate()
                    {
                        let dl = d_all * (scales[is] as f32 - 32.0);
                        for (i, inner_y) in scale_scoped_y.iter_mut().enumerate() {
                            let new_y = dl
                                * (((qs[i + 16 * scale_index] >> shift) & 3) as i8
                                    - if (block.hmask[i + 16 * scale_index] & m) == 0 {
                                        4
                                    } else {
                                        0
                                    }) as f32;
                            *inner_y = new_y;
                        }
                        // 16 block finished => advance scale index
                        is += 1;
                    }
                    // 32 block finished => increase shift and m
                    shift += 2;
                    m <<= 1;
                }
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let orig_ys = vec![0.0f32; ys.len()];
        let mut orig_ys_slice = orig_ys.clone();
        Self::to_float(xs, &mut orig_ys_slice);
        let nb = xs.len();
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );
        for (i, block_ys) in orig_ys_slice.chunks(Self::BLCK_SIZE).enumerate() {
            let orig_scale = xs[i].d.to_f32();
            let scale_ratio = if orig_scale != 0.0 {
                scales[i] / orig_scale
            } else {
                0.0
            };
            for (j, &y) in block_ys.iter().enumerate() {
                ys[i * Self::BLCK_SIZE + j] = y * scale_ratio;
            }
        }
    }

    fn apply_perturbation_stochastic(
        xs: &mut [Self],
        perturbation: &[f32],
        epsilon: f32,
        seed: u64,
        add: bool,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u2};

        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            // Q3K: 3-bit values stored as:
            // - qs[64]: 2-bit values (4 per byte)
            // - hmask[32]: high bit flags
            // Value = (2-bit from qs) + (4 if hmask bit set) - 4
            // Range: -4 to 3

            let mut m: u8 = 1;
            let mut elem_idx = 0;

            for (chunk_idx, qs_chunk) in block.qs.chunks_exact_mut(32).enumerate() {
                let mut shift = 0;

                for _ in 0..4 {
                    // Process 32 elements per shift
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..32 {
                        let hmask_idx = i;
                        let h_bit = (block.hmask[hmask_idx] & m) != 0;

                        // Extract current 3-bit value
                        let q2 = (qs_chunk[i] >> shift) & 3;
                        let q3 = q2 as i8 + if h_bit { 0 } else { -4 };
                        // q3 is now in range -4 to 3

                        let idx = perturb_offset + chunk_idx * 128 + (shift as usize / 2) * 32 + i;
                        if idx < perturbation.len() {
                            let p = perturbation[idx] * epsilon;
                            let delta = stochastic_round_u2(p.abs(), &mut rng_state) as i8;
                            let delta = if p >= 0.0 { delta } else { -delta };

                            let new_q3 = if add {
                                (q3 + delta).clamp(-4, 3)
                            } else {
                                (q3 - delta).clamp(-4, 3)
                            };

                            // Re-encode: convert -4..3 to 2-bit + hmask
                            // If new_q3 >= 0: hmask bit set, q2 = new_q3
                            // If new_q3 < 0: hmask bit clear, q2 = new_q3 + 4
                            let (new_q2, new_h) = if new_q3 >= 0 {
                                (new_q3 as u8, true)
                            } else {
                                ((new_q3 + 4) as u8, false)
                            };

                            // Update qs
                            let mask = !(3u8 << shift);
                            qs_chunk[i] = (qs_chunk[i] & mask) | (new_q2 << shift);

                            // Update hmask
                            if new_h {
                                block.hmask[hmask_idx] |= m;
                            } else {
                                block.hmask[hmask_idx] &= !m;
                            }
                        }

                        elem_idx += 1;
                    }

                    shift += 2;
                    m <<= 1;
                }
            }
            let _ = elem_idx; // Silence unused warning
        }
    }

    fn apply_quantized_update(xs: &mut [Self], perturbation: &[f32], scale: f32, seed: u64) {
        Self::apply_perturbation_stochastic(xs, perturbation, -scale, seed, true);
    }

    fn apply_quantized_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u2};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;
            let mut m: u8 = 1;
            let mut elem_idx = 0;

            for (chunk_idx, qs_chunk) in block.qs.chunks_exact_mut(32).enumerate() {
                let mut shift = 0;

                for _ in 0..4 {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..32 {
                        let hmask_idx = i;
                        let h_bit = (block.hmask[hmask_idx] & m) != 0;
                        let q2 = (qs_chunk[i] >> shift) & 3;
                        let q3 = q2 as i8 + if h_bit { 0 } else { -4 };

                        let idx = perturb_offset + chunk_idx * 128 + (shift as usize / 2) * 32 + i;
                        if idx < perturbation.len() {
                            let raw = perturbation[idx] * (-scale);
                            let desired = raw + gain * residuals[idx].to_f32();
                            let delta = stochastic_round_u2(desired.abs(), &mut rng_state) as i8;
                            let delta = if desired >= 0.0 { delta } else { -delta };

                            let new_q3 = (q3 + delta).clamp(-4, 3);
                            let actual = (new_q3 - q3) as f32;
                            residuals[idx] = f16::from_f32(raw + residuals[idx].to_f32() - actual);

                            let (new_q2, new_h) = if new_q3 >= 0 {
                                (new_q3 as u8, true)
                            } else {
                                ((new_q3 + 4) as u8, false)
                            };

                            let mask = !(3u8 << shift);
                            qs_chunk[i] = (qs_chunk[i] & mask) | (new_q2 << shift);

                            if new_h {
                                block.hmask[hmask_idx] |= m;
                            } else {
                                block.hmask[hmask_idx] &= !m;
                            }
                        }

                        elem_idx += 1;
                    }

                    shift += 2;
                    m <<= 1;
                }
            }
            let _ = elem_idx;
        }
    }

    fn simulate_update_with_residual(
        xs: &[Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u2};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter().enumerate() {
            let perturb_offset = block_idx * QK_K;
            let mut m: u8 = 1;
            let mut elem_idx = 0;

            for (chunk_idx, qs_chunk) in block.qs.chunks_exact(32).enumerate() {
                let mut shift = 0;

                for _ in 0..4 {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..32 {
                        let h_bit = (block.hmask[i] & m) != 0;
                        let q2 = (qs_chunk[i] >> shift) & 3;
                        let q3 = q2 as i8 + if h_bit { 0 } else { -4 };

                        let idx = perturb_offset + chunk_idx * 128 + (shift as usize / 2) * 32 + i;
                        if idx < perturbation.len() {
                            let raw = perturbation[idx] * (-scale);
                            let desired = raw + gain * residuals[idx].to_f32();
                            let delta = stochastic_round_u2(desired.abs(), &mut rng_state) as i8;
                            let delta = if desired >= 0.0 { delta } else { -delta };

                            let new_q3 = (q3 + delta).clamp(-4, 3);
                            let actual = (new_q3 - q3) as f32;
                            residuals[idx] = f16::from_f32(raw + residuals[idx].to_f32() - actual);
                        }

                        elem_idx += 1;
                    }

                    shift += 2;
                    m <<= 1;
                }
            }
            let _ = elem_idx;
        }
    }

    fn supports_quzo() -> bool {
        true
    }
}

impl GgmlType for BlockQ4K {
    const DTYPE: GgmlDType = GgmlDType::Q4K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q4k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q4k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q4k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q4k_q8k: {n} is not divisible by {QK_K}"
        );

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        let mut scales: [u8; 8] = [0; 8];
        let mut mins: [u8; 8] = [0; 8];

        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut sumf = 0.0;
        for (y, x) in ys.iter().zip(xs.iter()) {
            let q4 = &x.qs;
            let q8 = &y.qs;
            aux32.fill(0);

            let mut a = &mut aux8[..];
            let mut q4 = &q4[..];
            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    a[l] = (q4[l] & 0xF) as i8;
                }
                a = &mut a[32..];
                for l in 0..32 {
                    a[l] = (q4[l] >> 4) as i8;
                }
                a = &mut a[32..];
                q4 = &q4[32..];
            }

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            //extract scales and mins
            LittleEndian::write_u32_into(&utmp[0..2], &mut scales);
            LittleEndian::write_u32_into(&utmp[2..4], &mut mins);

            let mut sumi = 0;
            for j in 0..QK_K / 16 {
                sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
            }

            let mut a = &mut aux8[..];
            let mut q8 = &q8[..];

            for scale in scales {
                let scale = scale as i32;
                for _ in 0..4 {
                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += scale * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let d = x.d.to_f32() * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
            let dmin = x.dmin.to_f32() * y.d;
            sumf -= dmin * sumi as f32;
        }
        sumf + sums.iter().sum::<f32>()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(15, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }

            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                    }
                }
            }

            let q = &mut block.qs;
            for j in (0..QK_K).step_by(64) {
                for l_val in 0..32 {
                    let offset_index = (j / 64) * 32 + l_val;
                    q[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
                }
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        for (sblk_idx, (block, x)) in group_for_quantization(xs, ys).into_iter().enumerate() {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut weights: [f32; 32] = [0.0; 32];
            let mut sw: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut ls: [u8; QK_K / 32] = [0; QK_K / 32];
            let mut lm: [u8; QK_K / 32] = [0; QK_K / 32];

            let sum_x2 = x.iter().map(|x| x * x).sum::<f32>();
            let sigma2 = 2. * sum_x2 / QK_K as f32;

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                for (l, (w_elem, x_elem)) in weights.iter_mut().zip(x_scale_slice).enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let imatrix_w = imatrix_weights[imatrix_row * QK_K + 32 * j + l];
                    *w_elem = imatrix_w * (sigma2 + x_elem * x_elem).sqrt();
                }
                let sumw = weights.iter().sum::<f32>();
                sw[j] = sumw;
                (scales[j], mins[j]) =
                    make_qkx3_quants(15, x_scale_slice, Some(&weights), -0.9, 0.05, 36, false);
            }

            let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls, &sw);
            let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm, &sw);
            for j in 0..QK_K / 32 {
                let ls_val = ls[j];
                let lm_val = lm[j];
                if j < 4 {
                    block.scales[j] = ls_val;
                    block.scales[j + 4] = lm_val;
                } else {
                    block.scales[j + 4] = (ls_val & 0xF) | ((lm_val & 0xF) << 4);
                    block.scales[j - 4] |= (ls_val >> 4) << 6;
                    block.scales[j] |= (lm_val >> 4) << 6;
                }
            }

            block.d = f16::from_f32(d_block);
            block.dmin = f16::from_f32(m_block);

            let mut l: [u8; QK_K] = [0; QK_K];
            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                    }
                }
            }

            let q = &mut block.qs;
            for j in (0..QK_K).step_by(64) {
                for l_val in 0..32 {
                    let offset_index = (j / 64) * 32 + l_val;
                    q[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
                }
            }
        }
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L735
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let q = &block.qs;
            let mut is = 0;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let q = &q[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for q in q {
                    y[ys_index] = d1 * (q & 0xF) as f32 - m1;
                    ys_index += 1;
                }
                for q in q {
                    y[ys_index] = d2 * (q >> 4) as f32 - m2;
                    ys_index += 1;
                }
                is += 2;
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let orig_ys = vec![0.0f32; ys.len()];
        let mut orig_ys_slice = orig_ys.clone();
        Self::to_float(xs, &mut orig_ys_slice);
        let nb = xs.len();
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );
        for (i, block_ys) in orig_ys_slice.chunks(Self::BLCK_SIZE).enumerate() {
            let orig_scale = xs[i].d.to_f32();
            let scale_ratio = if orig_scale != 0.0 {
                scales[i] / orig_scale
            } else {
                0.0
            };
            for (j, &y) in block_ys.iter().enumerate() {
                ys[i * Self::BLCK_SIZE + j] = y * scale_ratio;
            }
        }
    }

    fn apply_perturbation_stochastic(
        xs: &mut [Self],
        perturbation: &[f32],
        epsilon: f32,
        seed: u64,
        add: bool,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u4};

        let mut rng_state = seed;
        // Warm up RNG
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            // Q4K stores 256 elements per block in qs[128]
            // Each byte has two 4-bit values (low nibble, high nibble)
            // Layout: first 64 elements in first 32 bytes, next 64 in next 32 bytes, etc.
            for j in (0..QK_K).step_by(64) {
                for l in 0..32 {
                    let qs_idx = j / 2 + l;
                    if qs_idx >= block.qs.len() {
                        continue;
                    }

                    // Extract current 4-bit values (0-15 range)
                    let q_lo = block.qs[qs_idx] & 0x0F;
                    let q_hi = block.qs[qs_idx] >> 4;

                    // Element indices
                    let idx_lo = perturb_offset + j + l;
                    let idx_hi = perturb_offset + j + l + 32;

                    if idx_lo < perturbation.len() && idx_hi < perturbation.len() {
                        let p_lo = perturbation[idx_lo] * epsilon;
                        let p_hi = perturbation[idx_hi] * epsilon;

                        // Stochastically round perturbations
                        let delta_lo = stochastic_round_u4(p_lo.abs(), &mut rng_state);
                        let delta_hi = stochastic_round_u4(p_hi.abs(), &mut rng_state);

                        // Apply perturbation with sign
                        let new_lo = if add == (p_lo >= 0.0) {
                            (q_lo as i16 + delta_lo as i16).clamp(0, 15) as u8
                        } else {
                            (q_lo as i16 - delta_lo as i16).clamp(0, 15) as u8
                        };
                        let new_hi = if add == (p_hi >= 0.0) {
                            (q_hi as i16 + delta_hi as i16).clamp(0, 15) as u8
                        } else {
                            (q_hi as i16 - delta_hi as i16).clamp(0, 15) as u8
                        };

                        block.qs[qs_idx] = (new_lo & 0x0F) | ((new_hi & 0x0F) << 4);
                    }
                }
            }
        }
    }

    fn apply_quantized_update(xs: &mut [Self], perturbation: &[f32], scale: f32, seed: u64) {
        Self::apply_perturbation_stochastic(xs, perturbation, -scale, seed, true);
    }

    fn apply_quantized_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u4};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            for j in (0..QK_K).step_by(64) {
                for l in 0..32 {
                    let qs_idx = j / 2 + l;
                    if qs_idx >= block.qs.len() {
                        continue;
                    }

                    let q_lo = block.qs[qs_idx] & 0x0F;
                    let q_hi = block.qs[qs_idx] >> 4;

                    let idx_lo = perturb_offset + j + l;
                    let idx_hi = perturb_offset + j + l + 32;

                    if idx_lo < perturbation.len() && idx_hi < perturbation.len() {
                        // Low nibble
                        let raw_lo = perturbation[idx_lo] * (-scale);
                        let desired_lo = raw_lo + gain * residuals[idx_lo].to_f32();
                        let delta_lo = stochastic_round_u4(desired_lo.abs(), &mut rng_state);
                        let new_lo = if desired_lo >= 0.0 {
                            (q_lo as i16 + delta_lo as i16).clamp(0, 15) as u8
                        } else {
                            (q_lo as i16 - delta_lo as i16).clamp(0, 15) as u8
                        };
                        let actual_lo = (new_lo as i16 - q_lo as i16) as f32;
                        residuals[idx_lo] =
                            f16::from_f32(raw_lo + residuals[idx_lo].to_f32() - actual_lo);

                        // High nibble
                        let raw_hi = perturbation[idx_hi] * (-scale);
                        let desired_hi = raw_hi + gain * residuals[idx_hi].to_f32();
                        let delta_hi = stochastic_round_u4(desired_hi.abs(), &mut rng_state);
                        let new_hi = if desired_hi >= 0.0 {
                            (q_hi as i16 + delta_hi as i16).clamp(0, 15) as u8
                        } else {
                            (q_hi as i16 - delta_hi as i16).clamp(0, 15) as u8
                        };
                        let actual_hi = (new_hi as i16 - q_hi as i16) as f32;
                        residuals[idx_hi] =
                            f16::from_f32(raw_hi + residuals[idx_hi].to_f32() - actual_hi);

                        block.qs[qs_idx] = (new_lo & 0x0F) | ((new_hi & 0x0F) << 4);
                    }
                }
            }
        }
    }

    fn simulate_update_with_residual(
        xs: &[Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u4};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter().enumerate() {
            let perturb_offset = block_idx * QK_K;

            for j in (0..QK_K).step_by(64) {
                for l in 0..32 {
                    let qs_idx = j / 2 + l;
                    if qs_idx >= block.qs.len() {
                        continue;
                    }

                    let q_lo = block.qs[qs_idx] & 0x0F;
                    let q_hi = block.qs[qs_idx] >> 4;

                    let idx_lo = perturb_offset + j + l;
                    let idx_hi = perturb_offset + j + l + 32;

                    if idx_lo < perturbation.len() && idx_hi < perturbation.len() {
                        let raw_lo = perturbation[idx_lo] * (-scale);
                        let desired_lo = raw_lo + gain * residuals[idx_lo].to_f32();
                        let delta_lo = stochastic_round_u4(desired_lo.abs(), &mut rng_state);
                        let new_lo = if desired_lo >= 0.0 {
                            (q_lo as i16 + delta_lo as i16).clamp(0, 15) as u8
                        } else {
                            (q_lo as i16 - delta_lo as i16).clamp(0, 15) as u8
                        };
                        let actual_lo = (new_lo as i16 - q_lo as i16) as f32;
                        residuals[idx_lo] =
                            f16::from_f32(raw_lo + residuals[idx_lo].to_f32() - actual_lo);

                        let raw_hi = perturbation[idx_hi] * (-scale);
                        let desired_hi = raw_hi + gain * residuals[idx_hi].to_f32();
                        let delta_hi = stochastic_round_u4(desired_hi.abs(), &mut rng_state);
                        let new_hi = if desired_hi >= 0.0 {
                            (q_hi as i16 + delta_hi as i16).clamp(0, 15) as u8
                        } else {
                            (q_hi as i16 - delta_hi as i16).clamp(0, 15) as u8
                        };
                        let actual_hi = (new_hi as i16 - q_hi as i16) as f32;
                        residuals[idx_hi] =
                            f16::from_f32(raw_hi + residuals[idx_hi].to_f32() - actual_hi);
                    }
                }
            }
        }
    }

    fn supports_quzo() -> bool {
        true
    }
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
impl GgmlType for BlockQ5K {
    const DTYPE: GgmlDType = GgmlDType::Q5K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q5k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q5k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q5k_q8k: {n} is not divisible by {QK_K}"
        );

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        let mut scales: [u8; 8] = [0; 8];
        let mut mins: [u8; 8] = [0; 8];

        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut sumf = 0.0;
        for (y, x) in ys.iter().zip(xs.iter()) {
            let q5 = &x.qs;
            let hm = &x.qh;
            let q8 = &y.qs;
            aux32.fill(0);

            let mut a = &mut aux8[..];
            let mut q5 = &q5[..];
            let mut m = 1u8;

            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    a[l] = (q5[l] & 0xF) as i8;
                    a[l] += if hm[l] & m != 0 { 16 } else { 0 };
                }
                a = &mut a[32..];
                m <<= 1;
                for l in 0..32 {
                    a[l] = (q5[l] >> 4) as i8;
                    a[l] += if hm[l] & m != 0 { 16 } else { 0 };
                }
                a = &mut a[32..];
                m <<= 1;
                q5 = &q5[32..];
            }

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            //extract scales and mins
            LittleEndian::write_u32_into(&utmp[0..2], &mut scales);
            LittleEndian::write_u32_into(&utmp[2..4], &mut mins);

            let mut sumi = 0;
            for j in 0..QK_K / 16 {
                sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
            }

            let mut a = &mut aux8[..];
            let mut q8 = &q8[..];

            for scale in scales {
                let scale = scale as i32;
                for _ in 0..4 {
                    for l in 0..8 {
                        aux16[l] = q8[l] as i16 * a[l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += scale * aux16[l] as i32;
                    }
                    q8 = &q8[8..];
                    a = &mut a[8..];
                }
            }
            let d = x.d.to_f32() * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
            let dmin = x.dmin.to_f32() * y.d;
            sumf -= dmin * sumi as f32;
        }
        sumf + sums.iter().sum::<f32>()
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L793
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(31, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }
            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];
            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * m as f32;
                for ii in 0..32 {
                    let ll = nearest_int((x[32 * j + ii] + dm) / d);
                    l[32 * j + ii] = ll.clamp(0, 31) as u8;
                }
            }

            let qh = &mut block.qh;
            let ql = &mut block.qs;
            qh.fill(0);

            let mut m1 = 1;
            let mut m2 = 2;
            for n in (0..QK_K).step_by(64) {
                let offset = (n / 64) * 32;
                for j in 0..32 {
                    let mut l1 = l[n + j];
                    if l1 > 15 {
                        l1 -= 16;
                        qh[j] |= m1;
                    }
                    let mut l2 = l[n + j + 32];
                    if l2 > 15 {
                        l2 -= 16;
                        qh[j] |= m2;
                    }
                    ql[offset + j] = l1 | (l2 << 4);
                }
                m1 <<= 2;
                m2 <<= 2;
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        for (sblk_idx, (block, x)) in group_for_quantization(xs, ys).into_iter().enumerate() {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut weights: [f32; 32] = [0.0; 32];
            let mut sw: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut ls: [u8; QK_K / 32] = [0; QK_K / 32];
            let mut lm: [u8; QK_K / 32] = [0; QK_K / 32];

            let sum_x2 = x.iter().map(|x| x * x).sum::<f32>();
            let sigma2 = 2. * sum_x2 / QK_K as f32;

            for (j, x_scale_slice) in x.chunks_exact(32).enumerate() {
                for (l, (w_elem, x_elem)) in weights.iter_mut().zip(x_scale_slice).enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let imatrix_w = imatrix_weights[imatrix_row * QK_K + 32 * j + l];
                    *w_elem = imatrix_w * (sigma2 + x_elem * x_elem).sqrt();
                }
                let sumw = weights.iter().sum::<f32>();
                sw[j] = sumw;
                (scales[j], mins[j]) =
                    make_qkx3_quants(31, x_scale_slice, Some(&weights), -0.9, 0.05, 36, false);
            }

            let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls, &sw);
            let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm, &sw);
            for j in 0..QK_K / 32 {
                let ls_val = ls[j].min(63);
                let lm_val = lm[j].min(63);
                if j < 4 {
                    block.scales[j] = ls_val;
                    block.scales[j + 4] = lm_val;
                } else {
                    block.scales[j + 4] = (ls_val & 0xF) | ((lm_val & 0xF) << 4);
                    block.scales[j - 4] |= (ls_val >> 4) << 6;
                    block.scales[j] |= (lm_val >> 4) << 6;
                }
            }

            block.d = f16::from_f32(d_block);
            block.dmin = f16::from_f32(m_block);

            let mut l: [u8; QK_K] = [0; QK_K];
            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 31) as u8;
                    }
                }
            }

            let qh = &mut block.qh;
            let ql = &mut block.qs;
            qh.fill(0);

            let mut m1 = 1;
            let mut m2 = 2;
            for n in (0..QK_K).step_by(64) {
                let offset = (n / 64) * 32;
                for j in 0..32 {
                    let mut l1 = l[n + j];
                    if l1 > 15 {
                        l1 -= 16;
                        qh[j] |= m1;
                    }
                    let mut l2 = l[n + j + 32];
                    if l2 > 15 {
                        l2 -= 16;
                        qh[j] |= m2;
                    }
                    ql[offset + j] = l1 | (l2 << 4);
                }
                m1 <<= 2;
                m2 <<= 2;
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let ql = &block.qs;
            let qh = &block.qh;
            let mut is = 0;
            let mut u1 = 1;
            let mut u2 = 2;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let ql = &ql[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u1 != 0 { 16f32 } else { 0f32 };
                    y[ys_index] = d1 * ((ql & 0xF) as f32 + to_add) - m1;
                    ys_index += 1;
                }
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u2 != 0 { 16f32 } else { 0f32 };
                    y[ys_index] = d2 * ((ql >> 4) as f32 + to_add) - m2;
                    ys_index += 1;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let orig_ys = vec![0.0f32; ys.len()];
        let mut orig_ys_slice = orig_ys.clone();
        Self::to_float(xs, &mut orig_ys_slice);
        let nb = xs.len();
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );
        for (i, block_ys) in orig_ys_slice.chunks(Self::BLCK_SIZE).enumerate() {
            let orig_scale = xs[i].d.to_f32();
            let scale_ratio = if orig_scale != 0.0 {
                scales[i] / orig_scale
            } else {
                0.0
            };
            for (j, &y) in block_ys.iter().enumerate() {
                ys[i * Self::BLCK_SIZE + j] = y * scale_ratio;
            }
        }
    }

    fn apply_perturbation_stochastic(
        xs: &mut [Self],
        perturbation: &[f32],
        epsilon: f32,
        seed: u64,
        add: bool,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u4};

        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            // Q5K: 5-bit values (0-31)
            // qs[128] has low 4 bits (2 per byte)
            // qh[32] has high bit (8 elements per byte, 2 bits per element position)

            let mut u1: u8 = 1;
            let mut u2: u8 = 2;

            for j in (0..QK_K).step_by(64) {
                let ql = &mut block.qs[j / 2..j / 2 + 32];
                let qh = &mut block.qh[..];

                for l in 0..32 {
                    // Extract current 5-bit values
                    let qh_byte = qh[l];
                    let q_lo = (ql[l] & 0xF) + if qh_byte & u1 != 0 { 16 } else { 0 };
                    let q_hi = (ql[l] >> 4) + if qh_byte & u2 != 0 { 16 } else { 0 };

                    let idx_lo = perturb_offset + j + l;
                    let idx_hi = perturb_offset + j + l + 32;

                    // Apply perturbations (values are 0-31)
                    let new_q_lo = if idx_lo < perturbation.len() {
                        let p = perturbation[idx_lo] * epsilon;
                        let delta = stochastic_round_u4(p.abs(), &mut rng_state);
                        let delta = if p >= 0.0 {
                            delta as i16
                        } else {
                            -(delta as i16)
                        };
                        if add {
                            (q_lo as i16 + delta).clamp(0, 31) as u8
                        } else {
                            (q_lo as i16 - delta).clamp(0, 31) as u8
                        }
                    } else {
                        q_lo
                    };

                    let new_q_hi = if idx_hi < perturbation.len() {
                        let p = perturbation[idx_hi] * epsilon;
                        let delta = stochastic_round_u4(p.abs(), &mut rng_state);
                        let delta = if p >= 0.0 {
                            delta as i16
                        } else {
                            -(delta as i16)
                        };
                        if add {
                            (q_hi as i16 + delta).clamp(0, 31) as u8
                        } else {
                            (q_hi as i16 - delta).clamp(0, 31) as u8
                        }
                    } else {
                        q_hi
                    };

                    // Re-encode: low 4 bits to qs, high bit to qh
                    ql[l] = (new_q_lo & 0xF) | ((new_q_hi & 0xF) << 4);

                    // Update qh bits
                    if new_q_lo >= 16 {
                        qh[l] |= u1;
                    } else {
                        qh[l] &= !u1;
                    }
                    if new_q_hi >= 16 {
                        qh[l] |= u2;
                    } else {
                        qh[l] &= !u2;
                    }
                }

                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    fn apply_quantized_update(xs: &mut [Self], perturbation: &[f32], scale: f32, seed: u64) {
        Self::apply_perturbation_stochastic(xs, perturbation, -scale, seed, true);
    }

    fn apply_quantized_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u4};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            let mut u1: u8 = 1;
            let mut u2: u8 = 2;

            for j in (0..QK_K).step_by(64) {
                let ql = &mut block.qs[j / 2..j / 2 + 32];
                let qh = &mut block.qh[..];

                for l in 0..32 {
                    let qh_byte = qh[l];
                    let q_lo = (ql[l] & 0xF) + if qh_byte & u1 != 0 { 16 } else { 0 };
                    let q_hi = (ql[l] >> 4) + if qh_byte & u2 != 0 { 16 } else { 0 };

                    let idx_lo = perturb_offset + j + l;
                    let idx_hi = perturb_offset + j + l + 32;

                    // Low element
                    let new_q_lo = if idx_lo < perturbation.len() {
                        let raw = perturbation[idx_lo] * (-scale);
                        let desired = raw + gain * residuals[idx_lo].to_f32();
                        let delta = stochastic_round_u4(desired.abs(), &mut rng_state);
                        let delta = if desired >= 0.0 {
                            delta as i16
                        } else {
                            -(delta as i16)
                        };
                        let new_val = (q_lo as i16 + delta).clamp(0, 31) as u8;
                        let actual = (new_val as i16 - q_lo as i16) as f32;
                        residuals[idx_lo] =
                            f16::from_f32(raw + residuals[idx_lo].to_f32() - actual);
                        new_val
                    } else {
                        q_lo
                    };

                    // High element
                    let new_q_hi = if idx_hi < perturbation.len() {
                        let raw = perturbation[idx_hi] * (-scale);
                        let desired = raw + gain * residuals[idx_hi].to_f32();
                        let delta = stochastic_round_u4(desired.abs(), &mut rng_state);
                        let delta = if desired >= 0.0 {
                            delta as i16
                        } else {
                            -(delta as i16)
                        };
                        let new_val = (q_hi as i16 + delta).clamp(0, 31) as u8;
                        let actual = (new_val as i16 - q_hi as i16) as f32;
                        residuals[idx_hi] =
                            f16::from_f32(raw + residuals[idx_hi].to_f32() - actual);
                        new_val
                    } else {
                        q_hi
                    };

                    ql[l] = (new_q_lo & 0xF) | ((new_q_hi & 0xF) << 4);
                    if new_q_lo >= 16 {
                        qh[l] |= u1;
                    } else {
                        qh[l] &= !u1;
                    }
                    if new_q_hi >= 16 {
                        qh[l] |= u2;
                    } else {
                        qh[l] &= !u2;
                    }
                }

                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    fn simulate_update_with_residual(
        xs: &[Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u4};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter().enumerate() {
            let perturb_offset = block_idx * QK_K;

            let mut u1: u8 = 1;
            let mut u2: u8 = 2;

            for j in (0..QK_K).step_by(64) {
                let ql = &block.qs[j / 2..j / 2 + 32];
                let qh = &block.qh[..];

                for l in 0..32 {
                    let qh_byte = qh[l];
                    let q_lo = (ql[l] & 0xF) + if qh_byte & u1 != 0 { 16 } else { 0 };
                    let q_hi = (ql[l] >> 4) + if qh_byte & u2 != 0 { 16 } else { 0 };

                    let idx_lo = perturb_offset + j + l;
                    let idx_hi = perturb_offset + j + l + 32;

                    if idx_lo < perturbation.len() {
                        let raw = perturbation[idx_lo] * (-scale);
                        let desired = raw + gain * residuals[idx_lo].to_f32();
                        let delta = stochastic_round_u4(desired.abs(), &mut rng_state);
                        let delta = if desired >= 0.0 {
                            delta as i16
                        } else {
                            -(delta as i16)
                        };
                        let new_val = (q_lo as i16 + delta).clamp(0, 31) as u8;
                        let actual = (new_val as i16 - q_lo as i16) as f32;
                        residuals[idx_lo] =
                            f16::from_f32(raw + residuals[idx_lo].to_f32() - actual);
                    }

                    if idx_hi < perturbation.len() {
                        let raw = perturbation[idx_hi] * (-scale);
                        let desired = raw + gain * residuals[idx_hi].to_f32();
                        let delta = stochastic_round_u4(desired.abs(), &mut rng_state);
                        let delta = if desired >= 0.0 {
                            delta as i16
                        } else {
                            -(delta as i16)
                        };
                        let new_val = (q_hi as i16 + delta).clamp(0, 31) as u8;
                        let actual = (new_val as i16 - q_hi as i16) as f32;
                        residuals[idx_hi] =
                            f16::from_f32(raw + residuals[idx_hi].to_f32() - actual);
                    }
                }

                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    fn supports_quzo() -> bool {
        true
    }
}

impl GgmlType for BlockQ6K {
    const DTYPE: GgmlDType = GgmlDType::Q6K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q6k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q6k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q6k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q6k_q8k: {n} is not divisible by {QK_K}"
        );

        let mut aux8 = [0i8; QK_K];
        let mut aux16 = [0i16; 8];
        let mut sums = [0f32; 8];
        let mut aux32 = [0f32; 8];

        for (x, y) in xs.iter().zip(ys.iter()) {
            let q4 = &x.ql;
            let qh = &x.qh;
            let q8 = &y.qs;
            aux32.fill(0f32);

            for j in (0..QK_K).step_by(128) {
                let aux8 = &mut aux8[j..];
                let q4 = &q4[j / 2..];
                let qh = &qh[j / 4..];
                for l in 0..32 {
                    aux8[l] = (((q4[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 32] =
                        (((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 64] = (((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                    aux8[l + 96] =
                        (((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
                }
            }

            for (j, &scale) in x.scales.iter().enumerate() {
                let scale = scale as f32;
                let q8 = &q8[16 * j..];
                let aux8 = &aux8[16 * j..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
                let q8 = &q8[8..];
                let aux8 = &aux8[8..];
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * aux8[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as f32
                }
            }

            let d = x.d.to_f32() * y.d;
            for (sum, &a) in sums.iter_mut().zip(aux32.iter()) {
                *sum += a * d;
            }
        }
        sums.iter().sum()
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len() * Self::BLCK_SIZE,
            "quantize_row_q6k: size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = xs.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for y in ys.iter_mut() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let scale =
                        make_qx_quants(16, 32, x.add(16 * ib), l.add(16 * ib), 1, std::ptr::null());
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        debug_assert_eq!(
            xs.len(),
            ys.len() * Self::BLCK_SIZE,
            "quantize_row_q6k imatrix: size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = xs.as_ptr();
        let imatrix_weights = imatrix_weights.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for (sblk_idx, y) in ys.iter_mut().enumerate() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let scale = make_qx_quants(
                        16,
                        32,
                        x.add(16 * ib),
                        l.add(16 * ib),
                        1,
                        imatrix_weights.add(QK_K * imatrix_row + 16 * ib),
                    );
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L1067
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "dequantize_row_q6k: {k} is not divisible by {QK_K}"
        );

        for (idx_x, x) in xs.iter().enumerate() {
            let d = x.d.to_f32();
            let ql = &x.ql;
            let qh = &x.qh;
            let sc = &x.scales;
            for n in (0..QK_K).step_by(128) {
                let idx = n / 128;
                let ys = &mut ys[idx_x * QK_K + n..];
                let sc = &sc[8 * idx..];
                let ql = &ql[64 * idx..];
                let qh = &qh[32 * idx..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    ys[l] = d * sc[is] as f32 * q1 as f32;
                    ys[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                    ys[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                    ys[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                }
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d.to_f32()).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let orig_ys = vec![0.0f32; ys.len()];
        let mut orig_ys_slice = orig_ys.clone();
        Self::to_float(xs, &mut orig_ys_slice);
        let nb = xs.len();
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );
        for (i, block_ys) in orig_ys_slice.chunks(Self::BLCK_SIZE).enumerate() {
            let orig_scale = xs[i].d.to_f32();
            let scale_ratio = if orig_scale != 0.0 {
                scales[i] / orig_scale
            } else {
                0.0
            };
            for (j, &y) in block_ys.iter().enumerate() {
                ys[i * Self::BLCK_SIZE + j] = y * scale_ratio;
            }
        }
    }

    fn apply_perturbation_stochastic(
        xs: &mut [Self],
        perturbation: &[f32],
        epsilon: f32,
        seed: u64,
        add: bool,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u6};

        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            // Q6K stores 256 elements per block
            // Layout: ql[128] has low 4 bits, qh[64] has high 2 bits
            // Grouped in 128-element chunks
            for n in (0..QK_K).step_by(128) {
                let idx = n / 128;
                let ql = &mut block.ql[64 * idx..64 * idx + 64];
                let qh = &mut block.qh[32 * idx..32 * idx + 32];

                for l in 0..32 {
                    // Extract 4 6-bit values from this position
                    // q1: ql[l] low nibble + qh[l] bits 0-1
                    // q2: ql[l+32] low nibble + qh[l] bits 2-3
                    // q3: ql[l] high nibble + qh[l] bits 4-5
                    // q4: ql[l+32] high nibble + qh[l] bits 6-7

                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;

                    // Get perturbation indices
                    let idx1 = perturb_offset + n + l;
                    let idx2 = perturb_offset + n + l + 32;
                    let idx3 = perturb_offset + n + l + 64;
                    let idx4 = perturb_offset + n + l + 96;

                    // Apply perturbations
                    let new_q1 = if idx1 < perturbation.len() {
                        let p = perturbation[idx1] * epsilon;
                        let delta = stochastic_round_u6(p.abs(), &mut rng_state) as i8;
                        let delta = if p >= 0.0 { delta } else { -delta };
                        if add {
                            (q1 + delta).clamp(-32, 31)
                        } else {
                            (q1 - delta).clamp(-32, 31)
                        }
                    } else {
                        q1
                    };

                    let new_q2 = if idx2 < perturbation.len() {
                        let p = perturbation[idx2] * epsilon;
                        let delta = stochastic_round_u6(p.abs(), &mut rng_state) as i8;
                        let delta = if p >= 0.0 { delta } else { -delta };
                        if add {
                            (q2 + delta).clamp(-32, 31)
                        } else {
                            (q2 - delta).clamp(-32, 31)
                        }
                    } else {
                        q2
                    };

                    let new_q3 = if idx3 < perturbation.len() {
                        let p = perturbation[idx3] * epsilon;
                        let delta = stochastic_round_u6(p.abs(), &mut rng_state) as i8;
                        let delta = if p >= 0.0 { delta } else { -delta };
                        if add {
                            (q3 + delta).clamp(-32, 31)
                        } else {
                            (q3 - delta).clamp(-32, 31)
                        }
                    } else {
                        q3
                    };

                    let new_q4 = if idx4 < perturbation.len() {
                        let p = perturbation[idx4] * epsilon;
                        let delta = stochastic_round_u6(p.abs(), &mut rng_state) as i8;
                        let delta = if p >= 0.0 { delta } else { -delta };
                        if add {
                            (q4 + delta).clamp(-32, 31)
                        } else {
                            (q4 - delta).clamp(-32, 31)
                        }
                    } else {
                        q4
                    };

                    // Re-encode back to ql/qh (add 32 to convert from -32..31 to 0..63)
                    let u1 = (new_q1 + 32) as u8;
                    let u2 = (new_q2 + 32) as u8;
                    let u3 = (new_q3 + 32) as u8;
                    let u4 = (new_q4 + 32) as u8;

                    ql[l] = (u1 & 0xF) | ((u3 & 0xF) << 4);
                    ql[l + 32] = (u2 & 0xF) | ((u4 & 0xF) << 4);
                    qh[l] = ((u1 >> 4) & 3)
                        | (((u2 >> 4) & 3) << 2)
                        | (((u3 >> 4) & 3) << 4)
                        | (((u4 >> 4) & 3) << 6);
                }
            }
        }
    }

    fn apply_quantized_update(xs: &mut [Self], perturbation: &[f32], scale: f32, seed: u64) {
        Self::apply_perturbation_stochastic(xs, perturbation, -scale, seed, true);
    }

    fn apply_quantized_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u6};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            for n in (0..QK_K).step_by(128) {
                let chunk_idx = n / 128;
                let ql = &mut block.ql[64 * chunk_idx..64 * chunk_idx + 64];
                let qh = &mut block.qh[32 * chunk_idx..32 * chunk_idx + 32];

                for l in 0..32 {
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;

                    let idx1 = perturb_offset + n + l;
                    let idx2 = perturb_offset + n + l + 32;
                    let idx3 = perturb_offset + n + l + 64;
                    let idx4 = perturb_offset + n + l + 96;

                    let new_q1 = if idx1 < perturbation.len() {
                        let raw = perturbation[idx1] * (-scale);
                        let desired = raw + gain * residuals[idx1].to_f32();
                        let delta = stochastic_round_u6(desired.abs(), &mut rng_state) as i8;
                        let delta = if desired >= 0.0 { delta } else { -delta };
                        let new_val = (q1 + delta).clamp(-32, 31);
                        let actual = (new_val - q1) as f32;
                        residuals[idx1] = f16::from_f32(raw + residuals[idx1].to_f32() - actual);
                        new_val
                    } else {
                        q1
                    };

                    let new_q2 = if idx2 < perturbation.len() {
                        let raw = perturbation[idx2] * (-scale);
                        let desired = raw + gain * residuals[idx2].to_f32();
                        let delta = stochastic_round_u6(desired.abs(), &mut rng_state) as i8;
                        let delta = if desired >= 0.0 { delta } else { -delta };
                        let new_val = (q2 + delta).clamp(-32, 31);
                        let actual = (new_val - q2) as f32;
                        residuals[idx2] = f16::from_f32(raw + residuals[idx2].to_f32() - actual);
                        new_val
                    } else {
                        q2
                    };

                    let new_q3 = if idx3 < perturbation.len() {
                        let raw = perturbation[idx3] * (-scale);
                        let desired = raw + gain * residuals[idx3].to_f32();
                        let delta = stochastic_round_u6(desired.abs(), &mut rng_state) as i8;
                        let delta = if desired >= 0.0 { delta } else { -delta };
                        let new_val = (q3 + delta).clamp(-32, 31);
                        let actual = (new_val - q3) as f32;
                        residuals[idx3] = f16::from_f32(raw + residuals[idx3].to_f32() - actual);
                        new_val
                    } else {
                        q3
                    };

                    let new_q4 = if idx4 < perturbation.len() {
                        let raw = perturbation[idx4] * (-scale);
                        let desired = raw + gain * residuals[idx4].to_f32();
                        let delta = stochastic_round_u6(desired.abs(), &mut rng_state) as i8;
                        let delta = if desired >= 0.0 { delta } else { -delta };
                        let new_val = (q4 + delta).clamp(-32, 31);
                        let actual = (new_val - q4) as f32;
                        residuals[idx4] = f16::from_f32(raw + residuals[idx4].to_f32() - actual);
                        new_val
                    } else {
                        q4
                    };

                    let u1 = (new_q1 + 32) as u8;
                    let u2 = (new_q2 + 32) as u8;
                    let u3 = (new_q3 + 32) as u8;
                    let u4 = (new_q4 + 32) as u8;

                    ql[l] = (u1 & 0xF) | ((u3 & 0xF) << 4);
                    ql[l + 32] = (u2 & 0xF) | ((u4 & 0xF) << 4);
                    qh[l] = ((u1 >> 4) & 3)
                        | (((u2 >> 4) & 3) << 2)
                        | (((u3 >> 4) & 3) << 4)
                        | (((u4 >> 4) & 3) << 6);
                }
            }
        }
    }

    fn simulate_update_with_residual(
        xs: &[Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_u6};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter().enumerate() {
            let perturb_offset = block_idx * QK_K;

            for n in (0..QK_K).step_by(128) {
                let chunk_idx = n / 128;
                let ql = &block.ql[64 * chunk_idx..64 * chunk_idx + 64];
                let qh = &block.qh[32 * chunk_idx..32 * chunk_idx + 32];

                for l in 0..32 {
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;

                    let idx1 = perturb_offset + n + l;
                    let idx2 = perturb_offset + n + l + 32;
                    let idx3 = perturb_offset + n + l + 64;
                    let idx4 = perturb_offset + n + l + 96;

                    if idx1 < perturbation.len() {
                        let raw = perturbation[idx1] * (-scale);
                        let desired = raw + gain * residuals[idx1].to_f32();
                        let delta = stochastic_round_u6(desired.abs(), &mut rng_state) as i8;
                        let delta = if desired >= 0.0 { delta } else { -delta };
                        let new_val = (q1 + delta).clamp(-32, 31);
                        let actual = (new_val - q1) as f32;
                        residuals[idx1] = f16::from_f32(raw + residuals[idx1].to_f32() - actual);
                    }

                    if idx2 < perturbation.len() {
                        let raw = perturbation[idx2] * (-scale);
                        let desired = raw + gain * residuals[idx2].to_f32();
                        let delta = stochastic_round_u6(desired.abs(), &mut rng_state) as i8;
                        let delta = if desired >= 0.0 { delta } else { -delta };
                        let new_val = (q2 + delta).clamp(-32, 31);
                        let actual = (new_val - q2) as f32;
                        residuals[idx2] = f16::from_f32(raw + residuals[idx2].to_f32() - actual);
                    }

                    if idx3 < perturbation.len() {
                        let raw = perturbation[idx3] * (-scale);
                        let desired = raw + gain * residuals[idx3].to_f32();
                        let delta = stochastic_round_u6(desired.abs(), &mut rng_state) as i8;
                        let delta = if desired >= 0.0 { delta } else { -delta };
                        let new_val = (q3 + delta).clamp(-32, 31);
                        let actual = (new_val - q3) as f32;
                        residuals[idx3] = f16::from_f32(raw + residuals[idx3].to_f32() - actual);
                    }

                    if idx4 < perturbation.len() {
                        let raw = perturbation[idx4] * (-scale);
                        let desired = raw + gain * residuals[idx4].to_f32();
                        let delta = stochastic_round_u6(desired.abs(), &mut rng_state) as i8;
                        let delta = if desired >= 0.0 { delta } else { -delta };
                        let new_val = (q4 + delta).clamp(-32, 31);
                        let actual = (new_val - q4) as f32;
                        residuals[idx4] = f16::from_f32(raw + residuals[idx4].to_f32() - actual);
                    }
                }
            }
        }
    }

    fn supports_quzo() -> bool {
        true
    }
}

impl GgmlType for BlockQ8K {
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = BlockQ8K;

    #[allow(unreachable_code)]
    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        #[cfg(target_feature = "avx2")]
        return super::avx::vec_dot_q8k_q8k(n, xs, ys);

        #[cfg(target_feature = "neon")]
        return super::neon::vec_dot_q8k_q8k(n, xs, ys);

        #[cfg(target_feature = "simd128")]
        return super::simd128::vec_dot_q8k_q8k(n, xs, ys);

        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(
            n.is_multiple_of(QK_K),
            "vec_dot_q8k_q8k: {n} is not divisible by {QK_K}"
        );
        // Generic implementation.
        let mut sumf = 0f32;
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let sum_i = xs
                .qs
                .iter()
                .zip(ys.qs.iter())
                .map(|(&x, &y)| x as i32 * y as i32)
                .sum::<i32>();
            sumf += sum_i as f32 * xs.d * ys.d
        }
        sumf
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "quantize_row_q8k: {k} is not divisible by {QK_K}"
        );
        for (i, y) in ys.iter_mut().enumerate() {
            let mut max = 0f32;
            let mut amax = 0f32;
            let xs = &xs[i * QK_K..(i + 1) * QK_K];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            if amax == 0f32 {
                y.d = 0f32;
                y.qs.fill(0)
            } else {
                let iscale = -128f32 / max;
                for (j, q) in y.qs.iter_mut().enumerate() {
                    // ggml uses nearest_int with bit magic here, maybe we want the same
                    // but we would have to test and benchmark it.
                    let v = (iscale * xs[j]).round();
                    *q = v.min(127.) as i8
                }
                for j in 0..QK_K / 16 {
                    let mut sum = 0i32;
                    for ii in 0..16 {
                        sum += y.qs[j * 16 + ii] as i32
                    }
                    y.bsums[j] = sum as i16
                }
                y.d = 1.0 / iscale
            }
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "dequantize_row_q8k: {k} is not divisible by {QK_K}"
        );
        for (i, x) in xs.iter().enumerate() {
            for (j, &q) in x.qs.iter().enumerate() {
                ys[i * QK_K + j] = x.d * q as f32
            }
        }
    }

    fn extract_scales(xs: &[Self]) -> Vec<f32> {
        xs.iter().map(|x| x.d).collect()
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], scales: &[f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "dequantize_row_q8k: {k} is not divisible by {QK_K}"
        );
        let nb = xs.len();
        debug_assert_eq!(
            scales.len(),
            nb,
            "scales length must match number of blocks"
        );
        for (i, x) in xs.iter().enumerate() {
            let scale_ratio = scales[i] / x.d;
            for (j, &q) in x.qs.iter().enumerate() {
                ys[i * QK_K + j] = q as f32 * scale_ratio;
            }
        }
    }

    fn apply_perturbation_stochastic(
        xs: &mut [Self],
        perturbation: &[f32],
        epsilon: f32,
        seed: u64,
        add: bool,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i8};

        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;

            for j in 0..QK_K {
                let idx = perturb_offset + j;
                if idx < perturbation.len() {
                    let p = perturbation[idx] * epsilon;
                    let delta = stochastic_round_i8(p, &mut rng_state);

                    let current = block.qs[j];
                    let new_val = if add {
                        (current as i16 + delta as i16).clamp(-128, 127) as i8
                    } else {
                        (current as i16 - delta as i16).clamp(-128, 127) as i8
                    };
                    block.qs[j] = new_val;
                }
            }
        }
    }

    fn apply_quantized_update(xs: &mut [Self], perturbation: &[f32], scale: f32, seed: u64) {
        Self::apply_perturbation_stochastic(xs, perturbation, -scale, seed, true);
    }

    fn apply_quantized_update_with_residual(
        xs: &mut [Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i8};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter_mut().enumerate() {
            let perturb_offset = block_idx * QK_K;
            for j in 0..QK_K {
                let idx = perturb_offset + j;
                if idx < perturbation.len() {
                    let current = block.qs[j];
                    let raw = perturbation[idx] * (-scale);
                    let desired = raw + gain * residuals[idx].to_f32();
                    let delta = stochastic_round_i8(desired, &mut rng_state);
                    let new_val = (current as i16 + delta as i16).clamp(-128, 127) as i8;
                    let actual = (new_val - current) as f32;
                    residuals[idx] = f16::from_f32(raw + residuals[idx].to_f32() - actual);
                    block.qs[j] = new_val;
                }
            }
        }
    }

    fn simulate_update_with_residual(
        xs: &[Self],
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) {
        use super::stochastic::{lcg_next, stochastic_round_i8};
        let mut rng_state = seed;
        for _ in 0..10 {
            lcg_next(&mut rng_state);
        }

        for (block_idx, block) in xs.iter().enumerate() {
            let perturb_offset = block_idx * QK_K;
            for j in 0..QK_K {
                let idx = perturb_offset + j;
                if idx < perturbation.len() {
                    let current = block.qs[j];
                    let raw = perturbation[idx] * (-scale);
                    let desired = raw + gain * residuals[idx].to_f32();
                    let delta = stochastic_round_i8(desired, &mut rng_state);
                    let new_val = (current as i16 + delta as i16).clamp(-128, 127) as i8;
                    let actual = (new_val - current) as f32;
                    residuals[idx] = f16::from_f32(raw + residuals[idx].to_f32() - actual);
                }
            }
        }
    }

    fn supports_quzo() -> bool {
        true
    }
}

// https://github.com/ggml-org/llama.cpp/blob/aa3ee0eb0b80efca126cedf9bcb4fb5864b46ce3/ggml/src/ggml-cpu/ggml-cpu.c#L1205
pub fn matmul<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(
        T::BLCK_SIZE,
        T::VecDotType::BLCK_SIZE,
        "Mismatched block sizes"
    );
    debug_assert_eq!(
        m * k,
        lhs.len(),
        "unexpected lhs length {} ({m},{k},{n})",
        lhs.len()
    );
    let k_in_blocks = k.div_ceil(T::BLCK_SIZE);

    // TODO: Pre-allocate this.
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_blocks];
    // f32, f16, and bf16 support direct copy
    if T::DIRECT_COPY {
        T::VecDotType::direct_copy(lhs, &mut lhs_b);
    } else {
        for row_idx in 0..m {
            let lhs_b_mut = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
            let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
            T::VecDotType::from_float(lhs, lhs_b_mut)
        }
    }

    for row_idx in 0..m {
        let lhs_row = &lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];

        dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .for_each(|(col_idx, dst)| {
                let rhs_col = &rhs_t[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                *dst = T::vec_dot(k, rhs_col, lhs_row);
            });
    }
    Ok(())
}

/// Fused perturbed matmul for QuZO training.
///
/// Computes: dst = (weights + ε*z) @ lhs
/// where z is generated on-the-fly using Philox RNG.
///
/// This preserves the SIMD benefits of quantized vec_dot by computing:
/// result = vec_dot(weights, lhs) + ε * dot(z, lhs_f32)
///
/// The perturbation correction term uses the original f32 inputs, which are
/// already available in the `lhs` parameter.
pub fn matmul_perturbed<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
    seed: u64,
    epsilon: f32,
) -> Result<()> {
    debug_assert_eq!(
        T::BLCK_SIZE,
        T::VecDotType::BLCK_SIZE,
        "Mismatched block sizes"
    );
    debug_assert_eq!(
        m * k,
        lhs.len(),
        "unexpected lhs length {} ({m},{k},{n})",
        lhs.len()
    );
    let k_in_blocks = k.div_ceil(T::BLCK_SIZE);

    // Quantize lhs for the regular vec_dot computation
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_blocks];
    if T::DIRECT_COPY {
        T::VecDotType::direct_copy(lhs, &mut lhs_b);
    } else {
        for row_idx in 0..m {
            let lhs_b_mut = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
            let lhs_row = &lhs[row_idx * k..(row_idx + 1) * k];
            T::VecDotType::from_float(lhs_row, lhs_b_mut)
        }
    }

    for row_idx in 0..m {
        let lhs_row_q = &lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
        let lhs_row_f32 = &lhs[row_idx * k..(row_idx + 1) * k];
        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];

        dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .for_each(|(col_idx, dst)| {
                let rhs_col = &rhs_t[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                let col_offset = col_idx * k;
                *dst = T::vec_dot_perturbed(
                    k,
                    rhs_col,
                    lhs_row_q,
                    lhs_row_f32,
                    seed,
                    col_offset,
                    epsilon,
                );
            });
    }
    Ok(())
}

pub fn matmul_f16<T: GgmlType>(
    mkn: (usize, usize, usize),
    lhs: &[f16],
    rhs_t: &[T],
    dst: &mut [f16],
) -> Result<()> {
    let (m, k, n) = mkn;
    if m * k != lhs.len() {
        crate::bail!("unexpected lhs length {} {mkn:?}", lhs.len());
    }

    let k_in_lhs_blocks = k.div_ceil(T::BLCK_SIZE);
    let k_in_rhs_blocks = k.div_ceil(T::VecDotType::BLCK_SIZE);
    let mut lhs_b = vec![T::VecDotType::zeros(); m * k_in_lhs_blocks];
    for row_idx in 0..m {
        let lhs_b = &mut lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
        let lhs_f32: Vec<_> = lhs.iter().map(|&x| x.to_f32()).collect();
        T::VecDotType::from_float(&lhs_f32, lhs_b);
    }
    let lhs_b = lhs_b.as_slice();

    for row_idx in 0..m {
        let lhs_row = &lhs_b[row_idx * k_in_lhs_blocks..(row_idx + 1) * k_in_lhs_blocks];
        let dst_row = &mut dst[row_idx * n..(row_idx + 1) * n];

        for (col_idx, dst) in dst_row.iter_mut().enumerate() {
            let rhs_col = &rhs_t[col_idx * k_in_rhs_blocks..(col_idx + 1) * k_in_rhs_blocks];
            let value = T::vec_dot(k, rhs_col, lhs_row);
            *dst = f16::from_f32(value);
        }
    }
    Ok(())
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = f32;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f32(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.copy_from_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.copy_from_slice(xs);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }

    fn extract_scales(_xs: &[Self]) -> Vec<f32> {
        // f32 has no blocking, return vec of 1.0
        vec![1.0]
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], _scales: &[f32]) {
        // For f32, scales don't make sense as there's no blocking
        // Just do normal dequantization (which is just a copy)
        Self::to_float(xs, ys);
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = f16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_f16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.convert_from_f32_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        xs.convert_to_f32_slice(ys);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }

    fn extract_scales(_xs: &[Self]) -> Vec<f32> {
        // f16 has no blocking, return vec of 1.0
        vec![1.0]
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], _scales: &[f32]) {
        // For f16, scales don't make sense as there's no blocking
        // Just do normal dequantization
        Self::to_float(xs, ys);
    }
}

impl GgmlType for bf16 {
    const DTYPE: GgmlDType = GgmlDType::BF16;
    const BLCK_SIZE: usize = 1;
    const DIRECT_COPY: bool = true;
    type VecDotType = bf16;

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(xs.len() >= n, "size mismatch xs {} < {n}", xs.len());
        debug_assert!(ys.len() >= n, "size mismatch ys {} < {n}", ys.len());
        let mut res = 0f32;
        unsafe { crate::cpu::vec_dot_bf16(xs.as_ptr(), ys.as_ptr(), &mut res, n) };
        res
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.convert_from_f32_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        xs.convert_to_f32_slice(ys);
    }

    fn direct_copy(xs: &[f32], ys: &mut [Self]) {
        Self::from_float(xs, ys)
    }

    fn extract_scales(_xs: &[Self]) -> Vec<f32> {
        // bf16 has no blocking, return vec of 1.0
        vec![1.0]
    }

    fn to_float_with_scales(xs: &[Self], ys: &mut [f32], _scales: &[f32]) {
        // For bf16, scales don't make sense as there's no blocking
        // Just do normal dequantization
        Self::to_float(xs, ys);
    }
}

macro_rules! verify_block_size {
    ( $block_type:ident ) => {
        const _: () =
            assert!($block_type::BLCK_SIZE == <$block_type as GgmlType>::VecDotType::BLCK_SIZE);
    };
}

macro_rules! verify_block_sizes {
    ( $( $block_type:ident ),* ) => {
        $(
            verify_block_size!($block_type);
        )*
    };
}

verify_block_sizes!(
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, BlockQ2K, BlockQ3K, BlockQ4K,
    BlockQ5K, BlockQ6K, BlockQ8K, f32, f16, bf16
);
