use std::marker::PhantomData;

use glowstick::Shape;

use crate::{Error, Tensor};

/// Typed wrapper for a quantized matrix multiply weight.
///
/// `S` is the weight shape, e.g. `Shape2<OutDim, InDim>`.
/// The forward method is generic over input/output shapes; runtime validation
/// via `try_into()` catches mismatches (zero-cost when shapes are correct).
pub struct QMatMul<S: Shape>(
    pub(crate) paramecia_core::quantized::QMatMul,
    pub(crate) PhantomData<S>,
);

impl<S: Shape> std::fmt::Debug for QMatMul<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul<{}>({:?})", std::any::type_name::<S>(), self.0)
    }
}

impl<S: Shape> Clone for QMatMul<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

/// Runtime validation: extract dims from the inner QMatMul and verify they match S.
impl<S: Shape> TryFrom<paramecia_core::quantized::QMatMul> for QMatMul<S> {
    type Error = Error;
    #[track_caller]
    fn try_from(qmm: paramecia_core::quantized::QMatMul) -> Result<Self, Self::Error> {
        let caller = std::panic::Location::caller().to_string();
        let runtime_shape = weight_dims(&qmm)?;
        let type_level_shape = S::iter().collect::<Vec<_>>();

        if S::RANK != runtime_shape.len() {
            return Err(Error::RankMismatch {
                context: "Typed qmatmul conversion (core QMatMul -> QMatMul<S>)",
                typed_shape: std::any::type_name::<S>(),
                caller: caller.into_boxed_str(),
                runtime_rank: runtime_shape.len(),
                type_level_rank: S::RANK,
                runtime_shape: runtime_shape.into_boxed_slice(),
                type_level_shape: type_level_shape.into_boxed_slice(),
            });
        }

        for (dim, (a, b)) in runtime_shape
            .iter()
            .copied()
            .zip(type_level_shape.iter().copied())
            .enumerate()
        {
            if b != 0 && a != b {
                return Err(Error::DimensionMismatch {
                    context: "Typed qmatmul conversion (core QMatMul -> QMatMul<S>)",
                    typed_shape: std::any::type_name::<S>(),
                    caller: caller.clone().into_boxed_str(),
                    dim,
                    runtime: a,
                    type_level: b,
                    runtime_shape: runtime_shape.clone().into_boxed_slice(),
                    type_level_shape: type_level_shape.clone().into_boxed_slice(),
                });
            }
        }
        Ok(Self(qmm, PhantomData))
    }
}

/// Extract weight dimensions from any QMatMul variant.
fn weight_dims(qmm: &paramecia_core::quantized::QMatMul) -> Result<Vec<usize>, Error> {
    use paramecia_core::quantized::QMatMul as CoreQMM;
    match qmm {
        CoreQMM::QTensor(t) => Ok(t.shape().dims().to_vec()),
        CoreQMM::Shared(t) => Ok(t.dims()),
        CoreQMM::Tensor(t) => Ok(t.dims().to_vec()),
        CoreQMM::TensorF16(t) => Ok(t.dims().to_vec()),
    }
}

impl<S: Shape> QMatMul<S> {
    /// Get a reference to the inner core QMatMul.
    pub fn inner(&self) -> &paramecia_core::quantized::QMatMul {
        &self.0
    }

    /// Consume and extract the inner core QMatMul.
    pub fn into_inner(self) -> paramecia_core::quantized::QMatMul {
        self.0
    }

    /// Get the output dimension (first dim of the weight tensor).
    pub fn out_dim(&self) -> Option<usize> {
        weight_dims(&self.0).ok().and_then(|d| d.first().copied())
    }

    /// Get the inner QTensor if this is a quantized variant.
    pub fn qtensor(&self) -> Option<&paramecia_core::quantized::QTensor> {
        match &self.0 {
            paramecia_core::quantized::QMatMul::QTensor(arc) => Some(arc.as_ref()),
            _ => None,
        }
    }

    /// Get the SharedQTensor if this is a Shared variant.
    pub fn shared_qtensor(&self) -> Option<paramecia_core::quantized::SharedQTensor> {
        self.0.shared_qtensor()
    }

    /// Convert to shared/training mode in-place.
    pub fn make_shared(&mut self) -> Option<paramecia_core::quantized::SharedQTensor> {
        self.0.make_shared()
    }

    /// Forward pass: project input through this weight.
    ///
    /// The output shape `OutS` must be specified by the caller (or inferred from context).
    /// Runtime validation via `try_into()` ensures the output has the expected shape.
    pub fn forward<InS, OutS>(&self, input: &Tensor<InS>) -> paramecia_core::Result<Tensor<OutS>>
    where
        InS: Shape,
        OutS: Shape,
    {
        let result = self.forward_core(input.inner())?;
        Ok(result.try_into()?)
    }

    /// Forward pass returning an untyped core tensor.
    ///
    /// Use this when the caller does dynamic reshapes (attention, MoE routing)
    /// that make static output shape tracking impractical.
    pub fn forward_untyped(
        &self,
        x: &paramecia_core::Tensor,
    ) -> paramecia_core::Result<paramecia_core::Tensor> {
        self.forward_core(x)
    }

    /// Core forward: F16-dispatch forwarding to the inner QMatMul.
    fn forward_core(
        &self,
        x: &paramecia_core::Tensor,
    ) -> Result<paramecia_core::Tensor, paramecia_core::Error> {
        if x.dtype() == paramecia_core::DType::F16 {
            self.0.forward_via_f16(x)
        } else {
            paramecia_core::Module::forward(&self.0, x)
        }
    }

    /// Fused gate+up SwiGLU: computes silu(gate @ x) * (up @ x).
    pub fn gate_up_swiglu<GateS, InS, OutS>(
        gate: &QMatMul<GateS>,
        up: &QMatMul<S>,
        x: &Tensor<InS>,
    ) -> paramecia_core::Result<Tensor<OutS>>
    where
        InS: Shape,
        GateS: Shape,
        OutS: Shape,
    {
        let result = paramecia_core::quantized::QMatMul::gate_up_swiglu(
            gate.inner(),
            up.inner(),
            x.inner(),
        )?;
        Ok(result.try_into()?)
    }

    /// Fused gate+up SwiGLU returning an untyped core tensor.
    ///
    /// Use this when the caller does dynamic reshapes that make static
    /// output shape tracking impractical.
    pub fn gate_up_swiglu_untyped(
        gate: &QMatMul<S>,
        up: &QMatMul<S>,
        x: &paramecia_core::Tensor,
    ) -> paramecia_core::Result<paramecia_core::Tensor> {
        paramecia_core::quantized::QMatMul::gate_up_swiglu(gate.inner(), up.inner(), x)
    }

    /// Construct a typed QMatMul from a SharedQTensor, validating dimensions against `S`.
    pub fn from_shared(shared: paramecia_core::quantized::SharedQTensor) -> Result<Self, Error> {
        let inner = paramecia_core::quantized::QMatMul::from_shared(shared)?;
        inner.try_into()
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U2, U3, U4},
        Shape2,
    };

    use super::*;

    fn make_qmatmul(out_dim: usize, in_dim: usize) -> paramecia_core::quantized::QMatMul {
        let t = paramecia_core::Tensor::ones(
            (out_dim, in_dim),
            paramecia_core::DType::F32,
            &paramecia_core::Device::Cpu,
        )
        .unwrap();
        paramecia_core::quantized::QMatMul::Tensor(t)
    }

    #[test]
    fn qmatmul_forward_2d() {
        // Weight: Shape2<U4, U3> (4 outputs, 3 inputs)
        let core_qmm = make_qmatmul(4, 3);
        let qmm: QMatMul<Shape2<U4, U3>> = core_qmm.try_into().unwrap();

        // Input: Shape2<U2, U3> (2 tokens, 3 features)
        let input = Tensor::<Shape2<U2, U3>>::ones(
            paramecia_core::DType::F32,
            &paramecia_core::Device::Cpu,
        )
        .unwrap();

        // Output: Shape2<U2, U4> (2 tokens, 4 features)
        let output: Tensor<Shape2<U2, U4>> = qmm.forward(&input).unwrap();
        assert_shape_eq!(output, Shape2<U2, U4>);
        assert_eq!(output.dims(), vec![2, 4]);
    }

    #[test]
    fn qmatmul_try_from_rejects_wrong_dims() {
        let core_qmm = make_qmatmul(5, 3);
        let result: Result<QMatMul<Shape2<U4, U3>>, _> = core_qmm.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn qmatmul_out_dim() {
        let core_qmm = make_qmatmul(4, 3);
        let qmm: QMatMul<Shape2<U4, U3>> = core_qmm.try_into().unwrap();
        assert_eq!(qmm.out_dim(), Some(4));
    }

    #[test]
    fn qmatmul_forward_untyped() {
        let core_qmm = make_qmatmul(4, 3);
        let qmm: QMatMul<Shape2<U4, U3>> = core_qmm.try_into().unwrap();

        let input = paramecia_core::Tensor::ones(
            (2, 3),
            paramecia_core::DType::F32,
            &paramecia_core::Device::Cpu,
        )
        .unwrap();

        let output = qmm.forward_untyped(&input).unwrap();
        assert_eq!(output.dims(), &[2, 4]);
    }

    #[test]
    fn qmatmul_from_shared() {
        let t = paramecia_core::Tensor::ones(
            (4, 3),
            paramecia_core::DType::F32,
            &paramecia_core::Device::Cpu,
        )
        .unwrap();
        let qt = paramecia_core::quantized::QTensor::quantize(
            &t,
            paramecia_core::quantized::GgmlDType::F32,
        )
        .unwrap();
        let shared = paramecia_core::quantized::SharedQTensor::new(qt);
        let qmm: QMatMul<Shape2<U4, U3>> = QMatMul::from_shared(shared).unwrap();
        assert_eq!(qmm.out_dim(), Some(4));
    }
}
