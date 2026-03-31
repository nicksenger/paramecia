//! Utilities for quanitized network layers
//!
//! This module contains various implementations of standard neural network layers, modules and
//! utilities including embedding, linear layers, and various normalization techniques.
//! Most implementations provide quantized weights support.

use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use paramecia_core::quantized::{QTensor, SharedQTensor};
use paramecia_core::{Module, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Embedding {
    inner: paramecia_nn::Embedding,
    span: tracing::Span,
}

impl Embedding {
    pub fn new(d1: usize, d2: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = vb.get((d1, d2), "weight")?.dequantize(vb.device())?;
        let inner = paramecia_nn::Embedding::new(embeddings, d2);
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
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn from_arc(weight: std::sync::Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let weight = QMatMul::from_weights(weight)?;
        Ok(Self { weight, bias })
    }

    pub fn from_weights(weight: QMatMul, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> paramecia_core::Result<Tensor> {
        let x = x.apply(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?.dequantize(vb.device())?)
    } else {
        None
    };
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias })
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let bias = vb.get(out_dim, "bias")?.dequantize(vb.device())?;
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

pub fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<paramecia_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    let bias = vb.get(size, "bias")?.dequantize(vb.device())?;
    Ok(paramecia_nn::LayerNorm::new(weight, bias, eps))
}

pub fn layer_norm_no_bias(
    size: usize,
    eps: f64,
    vb: VarBuilder,
) -> Result<paramecia_nn::LayerNorm> {
    let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
    Ok(paramecia_nn::LayerNorm::new_no_bias(weight, eps))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = QMatMul::new(in_dim, out_dim, vb)?;
    Ok(Linear { weight, bias: None })
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    /// Shared weight for training (QuZO perturbations). When Some, forward()
    /// re-dequantizes from this to pick up perturbations.
    shared_weight: Option<SharedQTensor>,
    eps: f64,
    /// If true, use zero-centered (Gemma-style): x * (1 + weight) / rms(x)
    zero_centered: bool,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
        Ok(Self {
            weight,
            shared_weight: None,
            eps,
            zero_centered: false,
            span,
        })
    }

    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self {
            weight,
            shared_weight: None,
            eps,
            zero_centered: false,
            span,
        })
    }

    /// Create zero-centered (Gemma-style) RmsNorm from QTensor
    pub fn from_qtensor_zero_centered(weight: QTensor, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm-zc");
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self {
            weight,
            shared_weight: None,
            eps,
            zero_centered: true,
            span,
        })
    }

    pub fn from_weight(weight: Tensor, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        Ok(Self {
            weight,
            shared_weight: None,
            eps,
            zero_centered: false,
            span,
        })
    }

    /// Create RmsNorm with a shared (mutable) weight for training.
    /// Forward will re-dequantize from the SharedQTensor to pick up QuZO perturbations.
    pub fn from_shared(weight: Tensor, eps: f64, shared: SharedQTensor) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        Ok(Self {
            weight,
            shared_weight: Some(shared),
            eps,
            zero_centered: false,
            span,
        })
    }

    pub fn to_dtype(self, dtype: paramecia_core::DType) -> Result<Self> {
        Ok(Self {
            weight: self.weight.to_dtype(dtype)?,
            shared_weight: self.shared_weight,
            eps: self.eps,
            zero_centered: self.zero_centered,
            span: self.span,
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn shared_weight(&self) -> Option<&SharedQTensor> {
        self.shared_weight.as_ref()
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn zero_centered(&self) -> bool {
        self.zero_centered
    }
}

impl RmsNorm {
    /// Enable zero-centered (Gemma-style) normalization: x * (1 + weight) / rms(x)
    pub fn set_zero_centered(&mut self, enabled: bool) {
        self.zero_centered = enabled;
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        // When shared_weight is present, re-dequantize to pick up QuZO perturbations.
        // Norm tensors are small (e.g. [2048]) so this is negligible cost.
        let weight = if let Some(ref shared) = self.shared_weight {
            let qt = shared.read().unwrap();
            let w = qt.dequantize(x.device())?;
            w.to_dtype(x.dtype())?
        } else if self.weight.dtype() != x.dtype() {
            self.weight.to_dtype(x.dtype())?
        } else {
            self.weight.clone()
        };

        // Zero-centered (Gemma-style): use (1 + weight) instead of weight
        if self.zero_centered {
            let one = Tensor::ones_like(&weight)?;
            let adjusted_weight = (&one + &weight)?;
            paramecia_nn::ops::rms_norm(x, &adjusted_weight, self.eps as f32)
        } else {
            paramecia_nn::ops::rms_norm(x, &weight, self.eps as f32)
        }
    }
}
