use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: candle_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn from_weights(weights: Tensor) -> Result<Self> {
        let (_in_size, out_size) = weights.dims2()?;
        let inner = candle_nn::Embedding::new(weights, out_size);
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
        let inner = candle_nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { inner, span }
    }
}

pub fn linear_b(d1: usize, d2: usize, b: bool, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_b(d1, d2, b, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear_no_bias(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Wrap the conv2d op to provide some tracing.
#[derive(Debug, Clone)]
pub struct Conv2d {
    inner: candle_nn::Conv2d,
    span: tracing::Span,
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<Conv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner, span })
}

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: crate::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = candle::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    pub fn from_weights(ws: std::sync::Arc<candle::quantized::QTensor>) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    /// Get the inner QTensor if this is a quantized matmul
    pub fn qtensor(&self) -> Option<&candle::quantized::QTensor> {
        match &self.inner {
            candle::quantized::QMatMul::QTensor(arc) => Some(arc.as_ref()),
            _ => None,
        }
    }

    /// Merge LoRA weights into the underlying quantized tensor
    /// Returns the merged QTensor with LoRA contribution baked in
    /// LoRA contribution: σ/√r × A × B^T (where r is the rank)
    pub fn merge_lora(
        &self,
        lora_a: &Tensor,
        lora_b: &Tensor,
        sigma: f64,
    ) -> Result<candle::quantized::QTensor> {
        match &self.inner {
            candle::quantized::QMatMul::QTensor(qtensor) => {
                let device = qtensor.device();
                let dtype = qtensor.dtype();

                // Dequantize to f32
                let weight = qtensor.dequantize(&device)?;

                // Compute LoRA delta: σ/√r × A × B^T
                // A: [out_features, rank], B: [rank, in_features]
                // Result: [out_features, in_features]
                let rank = lora_a.dim(1)? as f64;
                let scale = sigma / rank.sqrt();

                // Move LoRA tensors to same device if needed
                let lora_a = if !lora_a.device().same_device(&device) {
                    lora_a.to_device(&device)?
                } else {
                    lora_a.clone()
                };
                let lora_b = if !lora_b.device().same_device(&device) {
                    lora_b.to_device(&device)?
                } else {
                    lora_b.clone()
                };

                // LoRA delta = A @ B^T (since we store B as [rank, in_features])
                let lora_delta = lora_a.matmul(&lora_b)?;
                let lora_delta = (lora_delta * scale)?;

                // Add to weight
                let merged = (weight + lora_delta)?;

                // Re-quantize with same dtype
                candle::quantized::QTensor::quantize(&merged, dtype)
            }
            candle::quantized::QMatMul::TensorF16(t) => {
                let device = t.device();

                // Compute LoRA delta
                let rank = lora_a.dim(1)? as f64;
                let scale = sigma / rank.sqrt();

                let lora_a = lora_a.to_device(device)?.to_dtype(candle::DType::F16)?;
                let lora_b = lora_b.to_device(device)?.to_dtype(candle::DType::F16)?;
                let lora_delta = lora_a.matmul(&lora_b)?;
                let lora_delta = (lora_delta * scale)?;

                let merged = (t + lora_delta)?;

                // Return as a "fake" quantized tensor - use F16 dtype
                candle::quantized::QTensor::quantize(&merged, candle::quantized::GgmlDType::F16)
            }
            candle::quantized::QMatMul::Tensor(t) => {
                let device = t.device();

                // Compute LoRA delta
                let rank = lora_a.dim(1)? as f64;
                let scale = sigma / rank.sqrt();

                let lora_a = lora_a.to_device(device)?.to_dtype(t.dtype())?;
                let lora_b = lora_b.to_device(device)?.to_dtype(t.dtype())?;
                let lora_delta = lora_a.matmul(&lora_b)?;
                let lora_delta = (lora_delta * scale)?;

                let merged = (t + lora_delta)?;

                // Return as F32 quantized tensor
                candle::quantized::QTensor::quantize(&merged, candle::quantized::GgmlDType::F32)
            }
        }
    }

    /// Forward with custom block scale multipliers (for fine-tuning)
    pub fn forward_with_scales(&self, xs: &Tensor, scale_mults: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_dtype = xs.dtype();
        let result = match &self.inner {
            candle::quantized::QMatMul::QTensor(qtensor) => {
                // Ensure scale_mults is on the same device as the qtensor
                let qtensor_device = qtensor.device();
                let scale_mults = if !scale_mults.device().same_device(&qtensor_device) {
                    scale_mults.to_device(&qtensor_device)?
                } else {
                    scale_mults.clone()
                };
                let scale_mults = scale_mults.to_dtype(candle::DType::F32)?;
                let modified = qtensor.modify_block_scales(&scale_mults)?;
                let qmatmul = candle::quantized::QMatMul::from_qtensor(modified)?;
                // Use F16 path for F16 input
                if input_dtype == candle::DType::F16 {
                    qmatmul.forward_via_f16(xs)?
                } else {
                    qmatmul.forward(xs)?
                }
            }
            _ => {
                if input_dtype == candle::DType::F16 {
                    self.inner.forward_via_f16(xs)?
                } else {
                    self.inner.forward(xs)?
                }
            }
        };
        Ok(result)
    }

    /// Forward with optional scales and LoRA adapters (for EGGROLL fine-tuning)
    ///
    /// LoRA contribution: σ/√r × (x @ B) @ A^T where r is the rank
    pub fn forward_with_scales_and_lora(
        &self,
        xs: &Tensor,
        scale_mults: Option<&Tensor>,
        lora_a: Option<&Tensor>,
        lora_b: Option<&Tensor>,
        sigma: f64,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_dtype = xs.dtype();

        // Base forward (with optional scales)
        let base_output = if let Some(scales) = scale_mults {
            match &self.inner {
                candle::quantized::QMatMul::QTensor(qtensor) => {
                    let qtensor_device = qtensor.device();
                    let scales = if !scales.device().same_device(&qtensor_device) {
                        scales.to_device(&qtensor_device)?
                    } else {
                        scales.clone()
                    };
                    let scales = scales.to_dtype(candle::DType::F32)?;
                    let modified = qtensor.modify_block_scales(&scales)?;
                    let qmatmul = candle::quantized::QMatMul::from_qtensor(modified)?;
                    // Use F16 path for F16 input
                    if input_dtype == candle::DType::F16 {
                        qmatmul.forward_via_f16(xs)?
                    } else {
                        qmatmul.forward(xs)?
                    }
                }
                _ => self.inner.forward(xs)?,
            }
        } else {
            // Use F16 path for F16 input
            if input_dtype == candle::DType::F16 {
                self.inner.forward_via_f16(xs)?
            } else {
                self.inner.forward(xs)?
            }
        };

        // Add LoRA contribution if present (LoRA in input dtype)
        let result = if let (Some(a), Some(b)) = (lora_a, lora_b) {
            let rank = a.dim(1)? as f64;
            let scale = sigma / rank.sqrt();

            // Handle batched input: flatten to 2D, apply LoRA, reshape back
            let xs_dims = xs.dims();
            let lora_output = if xs_dims.len() == 3 {
                let (batch, seq, features) = (xs_dims[0], xs_dims[1], xs_dims[2]);
                let xs_2d = xs.reshape((batch * seq, features))?;
                let xb = xs_2d.matmul(b)?;
                let out_2d = xb.matmul(&a.t()?)?;
                let out_features = a.dim(0)?;
                out_2d.reshape((batch, seq, out_features))?
            } else {
                let xb = xs.matmul(b)?;
                xb.matmul(&a.t()?)?
            };

            let lora_contrib = (lora_output * scale)?;
            (base_output + lora_contrib)?
        } else {
            base_output
        };

        Ok(result)
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_dtype = xs.dtype();
        // Use F16 path for F16 input for better performance
        if input_dtype == candle::DType::F16 {
            self.inner.forward_via_f16(xs)
        } else {
            self.inner.forward(xs)
        }
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let inner = candle_nn::LayerNorm::new(weight, bias, eps);
        let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
        Self { inner, span }
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

pub fn layer_norm<C: Into<candle_nn::LayerNormConfig>>(
    size: usize,
    c: C,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    let inner = candle_nn::layer_norm(size, c, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
    Ok(LayerNorm { inner, span })
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    pub fn forward_diff(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward_diff(x)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}
