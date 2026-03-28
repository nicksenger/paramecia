use paramecia_core::{DType, Device, Result, Tensor, D};
use paramecia_tensor::glowstick::Shape;
use paramecia_tensor::Tensor as TypedTensor;

use super::shape::RopeCache2;

type TRopeCache = TypedTensor<RopeCache2>;

/// YARN (Yet Another RoPE extensioN) configuration for context extension.
///
/// YARN extends the context window by applying frequency-dependent interpolation:
/// - High frequencies (short wavelengths): linear interpolation
/// - Low frequencies (long wavelengths): no interpolation
/// - Mid frequencies: smooth transition between the two
///
/// Reference: https://arxiv.org/abs/2309.00071
#[derive(Debug, Clone)]
pub struct YarnConfig {
    /// Original context length the model was trained on
    pub original_context: usize,
    /// Target extended context length
    pub target_context: usize,
    /// High frequency cutoff factor (beta_fast in the paper). Default: 32.0
    /// Frequencies with wavelength < beta_fast * original_context get full interpolation.
    pub beta_fast: f32,
    /// Low frequency cutoff factor (beta_slow in the paper). Default: 1.0
    /// Frequencies with wavelength > beta_slow * original_context get no interpolation.
    pub beta_slow: f32,
    /// Extra attention scaling multiplier. Default: 1.0
    /// The full attention scale becomes: (0.1 * ln(scale_factor) + 1) * attn_factor
    pub attn_factor: f32,
}

impl Default for YarnConfig {
    fn default() -> Self {
        Self {
            original_context: 4096,
            target_context: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
        }
    }
}

impl YarnConfig {
    /// Create a YARN config for extending context from original to target length.
    pub fn new(original_context: usize, target_context: usize) -> Self {
        Self {
            original_context,
            target_context,
            ..Default::default()
        }
    }

    /// Create a YARN config for 1M token context extension.
    ///
    /// This is a convenience method for the common case of extending to 1M tokens.
    /// The original_context should be the model's native training context length.
    ///
    /// # Example
    /// ```
    /// use paramecia_model::YarnConfig;
    ///
    /// // Extend Qwen3-Next from 256K to 1M tokens (4x scale)
    /// let yarn = YarnConfig::for_1m_context(262_144);
    /// assert_eq!(yarn.target_context, 1_048_576);
    /// assert!((yarn.scale_factor() - 4.0).abs() < 0.01);
    /// ```
    pub fn for_1m_context(original_context: usize) -> Self {
        Self::new(original_context, 1_048_576)
    }

    /// Create a YARN config with custom beta parameters.
    ///
    /// - `beta_fast`: High frequency cutoff (default 32.0). Higher values = more interpolation.
    /// - `beta_slow`: Low frequency cutoff (default 1.0). Lower values = more preservation.
    pub fn with_betas(mut self, beta_fast: f32, beta_slow: f32) -> Self {
        self.beta_fast = beta_fast;
        self.beta_slow = beta_slow;
        self
    }

    /// Set a custom attention factor multiplier.
    pub fn with_attn_factor(mut self, attn_factor: f32) -> Self {
        self.attn_factor = attn_factor;
        self
    }

    /// The scale factor (s) = target_context / original_context
    pub fn scale_factor(&self) -> f32 {
        self.target_context as f32 / self.original_context as f32
    }

    /// Compute the YARN attention temperature multiplier.
    /// This scales the attention logits to compensate for extended context.
    /// Formula: 0.1 * ln(s) + 1.0, where s is the scale factor.
    pub fn attention_scale(&self) -> f32 {
        let s = self.scale_factor();
        if s <= 1.0 {
            1.0
        } else {
            (0.1 * s.ln() + 1.0) * self.attn_factor
        }
    }

    /// Check if YARN scaling is actually needed (target > original)
    pub fn is_enabled(&self) -> bool {
        self.target_context > self.original_context
    }
}

#[derive(Clone)]
pub(super) struct RotaryEmbedding {
    sin: TRopeCache,
    cos: TRopeCache,
    /// Number of dimensions to rotate (partial rotary embedding).
    /// For Qwen3-Next this is typically 64 (25% of head_dim 256).
    n_rot: usize,
    /// Full head dimension
    head_dim: usize,
    /// Whether metadata indicates interleaved mRoPE frequency layout.
    /// For Qwen3.5 this does NOT imply adjacent-pair rotation (`rope_i`); rotation
    /// remains the standard half-split rotary transform.
    interleaved: bool,
    /// mRoPE frequency sections from GGUF metadata.
    sections: Option<[usize; 4]>,
    /// YARN attention scale factor (1.0 if YARN disabled)
    yarn_attn_scale: f32,
}

impl std::fmt::Debug for RotaryEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RotaryEmbedding")
            .field("n_rot", &self.n_rot)
            .field("head_dim", &self.head_dim)
            .field("interleaved", &self.interleaved)
            .field("sections", &self.sections)
            .field("yarn_attn_scale", &self.yarn_attn_scale)
            .finish()
    }
}

impl RotaryEmbedding {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        dtype: DType,
        head_dim: usize,
        n_rot: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        interleaved: bool,
        sections: Option<[usize; 4]>,
        yarn_config: Option<&YarnConfig>,
        dev: &Device,
    ) -> Result<Self> {
        let mut inv_freq: Vec<f32> = (0..n_rot)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / n_rot as f64) as f32)
            .collect();

        let yarn_attn_scale = if let Some(yarn) = yarn_config {
            if yarn.is_enabled() {
                Self::apply_yarn_scaling(&mut inv_freq, n_rot, yarn);
                yarn.attention_scale()
            } else {
                1.0
            }
        } else {
            1.0
        };

        let freqs = Tensor::from_vec(
            Self::build_frequency_cache(max_position_embeddings, &inv_freq, sections, interleaved),
            (max_position_embeddings, inv_freq.len()),
            dev,
        )?
        .to_dtype(dtype)?;
        Ok(Self {
            sin: freqs.sin()?.try_into()?,
            cos: freqs.cos()?.try_into()?,
            n_rot,
            head_dim,
            interleaved,
            sections,
            yarn_attn_scale,
        })
    }

    fn build_frequency_cache(
        max_seq_len: usize,
        inv_freq: &[f32],
        sections: Option<[usize; 4]>,
        interleaved: bool,
    ) -> Vec<f32> {
        let mut freqs = Vec::with_capacity(max_seq_len * inv_freq.len());
        for pos in 0..max_seq_len {
            freqs.extend(Self::angles_for_position_streams(
                [pos as f32; 4],
                inv_freq,
                sections,
                interleaved,
            ));
        }
        freqs
    }

    fn angles_for_position_streams(
        position_streams: [f32; 4],
        inv_freq: &[f32],
        sections: Option<[usize; 4]>,
        interleaved: bool,
    ) -> Vec<f32> {
        inv_freq
            .iter()
            .enumerate()
            .map(|(pair_idx, inv_freq)| {
                let stream = Self::position_stream_for_pair(pair_idx, sections, interleaved);
                position_streams[stream] * *inv_freq
            })
            .collect()
    }

    fn position_stream_for_pair(
        pair_idx: usize,
        sections: Option<[usize; 4]>,
        interleaved: bool,
    ) -> usize {
        let Some(sections) = sections else {
            return 0;
        };

        let sect_dims = sections.iter().sum::<usize>();
        if sect_dims == 0 {
            return 0;
        }

        let sector = pair_idx % sect_dims;
        if interleaved {
            if sector % 3 == 1 && sector < 3 * sections[1] {
                1
            } else if sector % 3 == 2 && sector < 3 * sections[2] {
                2
            } else if sector % 3 == 0 && sector < 3 * sections[0] {
                0
            } else {
                3
            }
        } else {
            let sec_w = sections[0] + sections[1];
            let sec_e = sec_w + sections[2];
            if sector >= sections[0] && sector < sec_w {
                1
            } else if sector >= sec_w && sector < sec_e {
                2
            } else if sector >= sec_e {
                3
            } else {
                0
            }
        }
    }

    /// Apply YARN frequency scaling to inverse frequencies.
    ///
    /// YARN divides frequencies into three regions based on wavelength:
    /// 1. High freq (wavelength < beta_fast * orig_ctx): full linear interpolation
    /// 2. Low freq (wavelength > beta_slow * orig_ctx): no interpolation
    /// 3. Mid freq: smooth ramp between the two
    fn apply_yarn_scaling(inv_freq: &mut [f32], _n_rot: usize, yarn: &YarnConfig) {
        let scale = yarn.scale_factor();
        let orig_ctx = yarn.original_context as f32;

        // Compute wavelengths and apply per-dimension scaling
        for freq in inv_freq.iter_mut() {
            // Wavelength for this frequency: λ = 2π / freq
            // Since freq = 1 / theta^(dim/n_rot), wavelength = 2π * theta^(dim/n_rot)
            let wavelength = 2.0 * std::f32::consts::PI / *freq;

            // Compute the interpolation ramp based on wavelength
            // low = wavelength / (beta_fast * orig_ctx)
            // high = wavelength / (beta_slow * orig_ctx)
            let low = wavelength / (yarn.beta_fast * orig_ctx);
            let high = wavelength / (yarn.beta_slow * orig_ctx);

            // Smooth ramp from 0 (full interpolation) to 1 (no interpolation)
            // ramp = clamp((freq_factor - low) / (high - low), 0, 1)
            // where freq_factor relates to how the dimension maps to frequency bands
            //
            // The YARN paper uses a ramp based on dimension index:
            // ramp(d) = (α - d/n_rot) / (α - β) clamped to [0, 1]
            // But the wavelength-based approach is equivalent and more intuitive.
            let ramp = if high <= low {
                if wavelength < yarn.beta_fast * orig_ctx {
                    0.0 // Full interpolation
                } else {
                    1.0 // No interpolation
                }
            } else {
                ((wavelength / orig_ctx - yarn.beta_fast) / (yarn.beta_slow - yarn.beta_fast))
                    .clamp(0.0, 1.0)
            };

            // Apply interpolation: freq_new = (1 - ramp) * freq/s + ramp * freq
            // This reduces high frequencies (small wavelengths) more than low frequencies
            *freq = (1.0 - ramp) * (*freq / scale) + ramp * *freq;
        }
    }

    /// Get the YARN attention scale factor (1.0 if YARN is disabled)
    pub(super) fn attention_scale(&self) -> f32 {
        self.yarn_attn_scale
    }

    pub(super) fn apply<QS, KS>(
        &self,
        q: &TypedTensor<QS>,
        k: &TypedTensor<KS>,
        offset: usize,
    ) -> Result<(TypedTensor<QS>, TypedTensor<KS>)>
    where
        QS: Shape,
        KS: Shape,
    {
        let q = q.inner();
        let k = k.inner();
        let (_, _, seq_len, d) = q.dims4()?;

        let cos = self
            .cos
            .inner()
            .narrow(0, offset, seq_len)?
            .to_dtype(q.dtype())?
            .to_device(q.device())?;
        let sin = self
            .sin
            .inner()
            .narrow(0, offset, seq_len)?
            .to_dtype(q.dtype())?
            .to_device(q.device())?;

        // If n_rot == head_dim, rotate everything.
        //
        // Important: Qwen3.5 mRoPE still uses half-split rotation semantics
        // (`rotate_half` style), not adjacent-pair rotation (`rope_i`).
        // The interleaving metadata affects mRoPE frequency section layout, not
        // the core rotation kernel used here.
        if self.n_rot == self.head_dim {
            let q_contig = q.contiguous()?;
            let k_contig = k.contiguous()?;
            let q_embed = paramecia_nn::rotary_emb::rope(&q_contig, &cos, &sin)?;
            let k_embed = paramecia_nn::rotary_emb::rope(&k_contig, &cos, &sin)?;
            return Ok((q_embed.try_into()?, k_embed.try_into()?));
        }

        // Partial rotary embedding: only rotate first n_rot dimensions
        let q_rot = q.narrow(D::Minus1, 0, self.n_rot)?.contiguous()?;
        let q_pass = q.narrow(D::Minus1, self.n_rot, d - self.n_rot)?;
        let k_rot = k.narrow(D::Minus1, 0, self.n_rot)?.contiguous()?;
        let k_pass = k.narrow(D::Minus1, self.n_rot, d - self.n_rot)?;

        let q_rot_embed = paramecia_nn::rotary_emb::rope(&q_rot, &cos, &sin)?;
        let k_rot_embed = paramecia_nn::rotary_emb::rope(&k_rot, &cos, &sin)?;

        let q_embed = Tensor::cat(&[&q_rot_embed, &q_pass], D::Minus1)?;
        let k_embed = Tensor::cat(&[&k_rot_embed, &k_pass], D::Minus1)?;

        Ok((q_embed.try_into()?, k_embed.try_into()?))
    }
}

#[cfg(test)]
mod tests {
    use super::RotaryEmbedding;

    #[test]
    fn mrope_stream_selection_matches_contiguous_sections() {
        let inv_freq = [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0];
        let positions = [1.0, 2.0, 3.0, 4.0];
        let angles = RotaryEmbedding::angles_for_position_streams(
            positions,
            &inv_freq,
            Some([2, 1, 1, 2]),
            false,
        );
        assert_eq!(angles, vec![1.0, 10.0, 200.0, 3000.0, 40_000.0, 400_000.0,]);
    }

    #[test]
    fn mrope_stream_selection_matches_imrope_sections() {
        let inv_freq = [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0];
        let positions = [1.0, 2.0, 3.0, 4.0];
        let angles = RotaryEmbedding::angles_for_position_streams(
            positions,
            &inv_freq,
            Some([2, 1, 1, 2]),
            true,
        );
        assert_eq!(angles, vec![1.0, 20.0, 300.0, 1000.0, 40_000.0, 400_000.0,]);
    }
}
