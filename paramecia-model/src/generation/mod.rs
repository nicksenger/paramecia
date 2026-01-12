//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
use candle::{DType, Error, Result, Tensor};
use rand::{distr::Distribution, SeedableRng};

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
    // Note that the rng is not used for the Gumbel-Softmax sampling.
    GumbelSoftmax { temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        logits.argmax(candle::D::Minus1)?.to_scalar::<u32>()
    }

    fn sample_gumbel_softmax(&mut self, logits: &Tensor, temperature: f64) -> Result<u32> {
        let sampled = candle_nn::sampling::gumbel_softmax(logits, temperature, candle::D::Minus1)?;
        sampled.to_scalar::<u32>()
    }

    fn normalize_weights(&self, prs: &[f32]) -> Option<Vec<f32>> {
        let mut sanitized = Vec::with_capacity(prs.len());
        let mut sum = 0f64;
        for &w in prs {
            if w.is_finite() && w > 0.0 {
                sanitized.push(w);
                sum += w as f64;
            } else {
                sanitized.push(0.0);
            }
        }
        if !sum.is_finite() || sum <= f64::EPSILON {
            return None;
        }
        let inv = 1.0f32 / sum as f32;
        for w in &mut sanitized {
            *w *= inv;
        }
        Some(sanitized)
    }

    fn sample_multinomial(&mut self, prs: &[f32], fallback: u32) -> Result<u32> {
        let prs = match self.normalize_weights(prs) {
            Some(p) => p,
            None => return Ok(fallback),
        };

        let distr = rand::distr::weighted::WeightedIndex::new(&prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32, fallback: u32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                if let Some(pr) = prs.get_mut(*index) {
                    *pr = 0.0;
                }
            } else if let Some(pr) = prs.get(*index) {
                cumsum += *pr;
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs, fallback)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, prs: &mut Vec<f32>, top_k: usize, fallback: u32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs, fallback)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs, fallback)?;
            let selected = indices.get(index as usize).copied().unwrap_or(0);
            Ok(selected as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(
        &mut self,
        prs: &mut Vec<f32>,
        top_k: usize,
        top_p: f32,
        fallback: u32,
    ) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p, fallback)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs, fallback)?
            } else {
                self.sample_topp(&mut prs, top_p, fallback)?
            };
            let selected = indices.get(index as usize).copied().unwrap_or(0);
            Ok(selected as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            let mut prs = prs.to_vec1()?;
            f(&mut prs);
            Ok(prs)
        };

        let fallback = logits.argmax(candle::D::Minus1)?.to_scalar::<u32>()?;
        let next_token = match &self.sampling {
            Sampling::ArgMax => fallback,
            Sampling::GumbelSoftmax { temperature } => {
                self.sample_gumbel_softmax(&logits, *temperature)?
            }
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs, fallback)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs, fallback)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32, fallback)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k, fallback)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32, fallback)?
            }
        };
        Ok(next_token)
    }
}
