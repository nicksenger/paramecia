use paramecia_core::{Module, Result, Tensor};
use paramecia_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: paramecia_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let inner = paramecia_nn::embedding(d1, d2, vb)?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");
        Ok(Self { inner, span })
    }

    pub fn from_weights(weights: Tensor) -> Result<Self> {
        let (_in_size, out_size) = weights.dims2()?;
        let inner = paramecia_nn::Embedding::new(weights, out_size);
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
    inner: paramecia_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
        let inner = paramecia_nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");
        Self { inner, span }
    }
}

pub fn linear_b(d1: usize, d2: usize, b: bool, vb: VarBuilder) -> Result<Linear> {
    let inner = paramecia_nn::linear_b(d1, d2, b, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = paramecia_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = paramecia_nn::linear_no_bias(d1, d2, vb)?;
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
    inner: paramecia_nn::Conv2d,
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
    cfg: paramecia_nn::Conv2dConfig,
    vs: paramecia_nn::VarBuilder,
) -> Result<Conv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = paramecia_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(Conv2d { inner, span })
}

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul {
    inner: paramecia_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: crate::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = paramecia_core::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    pub fn from_weights(ws: std::sync::Arc<paramecia_core::quantized::QTensor>) -> Result<Self> {
        let inner = paramecia_core::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    /// Create a QMatMul from a dequantized Tensor (e.g. after concatenating split weights)
    pub fn from_tensor(tensor: paramecia_core::Tensor) -> Self {
        let inner = paramecia_core::quantized::QMatMul::Tensor(tensor);
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Self { inner, span }
    }

    /// Create a QMatMul with shared (mutable) storage for training.
    /// Takes read locks during forward pass; optimizer takes write locks for updates.
    pub fn from_shared(shared: paramecia_core::quantized::SharedQTensor) -> Result<Self> {
        let inner = paramecia_core::quantized::QMatMul::from_shared(shared)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    /// Get the SharedQTensor if this is a Shared variant (for training).
    /// Returns None for other variants.
    pub fn shared_qtensor(&self) -> Option<paramecia_core::quantized::SharedQTensor> {
        self.inner.shared_qtensor()
    }

    /// Get the inner QTensor if this is a quantized matmul
    pub fn qtensor(&self) -> Option<&paramecia_core::quantized::QTensor> {
        match &self.inner {
            paramecia_core::quantized::QMatMul::QTensor(arc) => Some(arc.as_ref()),
            _ => None,
        }
    }

    /// Get the output dimension of this QMatMul (first dim of the weight tensor).
    /// Works for both Shared and non-Shared variants.
    pub fn out_dim(&self) -> Option<usize> {
        self.shared_qtensor()
            .map(|qt| qt.shape().dims()[0])
            .or_else(|| self.qtensor().map(|qt| qt.shape().dims()[0]))
    }

    /// Convert to shared/training mode in-place.
    /// Returns the SharedQTensor if conversion succeeded, None otherwise.
    pub fn make_shared(&mut self) -> Option<paramecia_core::quantized::SharedQTensor> {
        self.inner.make_shared()
    }

    /// Fused gate+up SwiGLU: computes silu(gate @ x) * (up @ x) in one kernel.
    /// Uses fused CUDA kernel for batch_size=1 decode, falls back to separate ops otherwise.
    pub fn gate_up_swiglu(gate: &Self, up: &Self, x: &Tensor) -> Result<Tensor> {
        paramecia_core::quantized::QMatMul::gate_up_swiglu(&gate.inner, &up.inner, x)
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_dtype = xs.dtype();
        // Use F16 path for F16 input for better performance
        if input_dtype == paramecia_core::DType::F16 {
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
    inner: paramecia_nn::LayerNorm,
    span: tracing::Span,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        let inner = paramecia_nn::LayerNorm::new(weight, bias, eps);
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

pub fn layer_norm<C: Into<paramecia_nn::LayerNormConfig>>(
    size: usize,
    c: C,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    let inner = paramecia_nn::layer_norm(size, c, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
    Ok(LayerNorm { inner, span })
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: paramecia_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = paramecia_nn::rms_norm(size, eps, vb)?;
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
