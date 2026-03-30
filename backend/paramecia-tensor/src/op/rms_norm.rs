use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;
use paramecia_core::quantized::SharedQTensor;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Typed RMS normalization op that holds its own weight tensor.
///
/// Shape-preserving: input and output have the same shape `S`.
/// The weight is a 1D tensor of size equal to the last dimension of `S`.
pub struct RmsNormOp<S> {
    weight: paramecia_core::Tensor,
    shared_weight: Option<SharedQTensor>,
    eps: f32,
    zero_centered: bool,
    _phantom: PhantomData<S>,
}

impl<S> RmsNormOp<S> {
    pub fn new(weight: paramecia_core::Tensor, eps: f64) -> Self {
        Self {
            weight,
            shared_weight: None,
            eps: eps as f32,
            zero_centered: false,
            _phantom: PhantomData,
        }
    }

    pub fn new_with_shared(
        weight: paramecia_core::Tensor,
        eps: f64,
        shared_weight: Option<SharedQTensor>,
        zero_centered: bool,
    ) -> Self {
        Self {
            weight,
            shared_weight,
            eps: eps as f32,
            zero_centered,
            _phantom: PhantomData,
        }
    }
}

impl<S: Shape> std::fmt::Debug for RmsNormOp<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RmsNormOp<{}>(eps={})",
            std::any::type_name::<S>(),
            self.eps
        )
    }
}

impl<S: Shape> Clone for RmsNormOp<S> {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            shared_weight: self.shared_weight.clone(),
            eps: self.eps,
            zero_centered: self.zero_centered,
            _phantom: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<S> Combinator for RmsNormOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("rms_norm", S);
        let input = input.into_inner();
        let weight = if let Some(shared) = self.shared_weight.as_ref() {
            let qt = shared.read().map_err(|e| {
                Error::Msg(format!("RmsNormOp shared_weight read lock failed: {e}"))
            })?;
            let weight = qt.dequantize(input.device())?;
            if weight.dtype() != input.dtype() {
                weight.to_dtype(input.dtype())?
            } else {
                weight
            }
        } else if self.weight.dtype() != input.dtype() {
            self.weight.to_dtype(input.dtype())?
        } else {
            self.weight.clone()
        };
        let result = if self.zero_centered {
            let one = paramecia_core::Tensor::ones_like(&weight)?;
            let adjusted_weight = (&one + &weight)?;
            paramecia_nn::ops::rms_norm(&input, &adjusted_weight, self.eps)?
        } else {
            paramecia_nn::ops::rms_norm(&input, &weight, self.eps)?
        };
        result.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for RmsNormOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "RmsNorm",
            Some(&pretty_shape(std::any::type_name::<
                <S as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U2, U3},
        Shape2,
    };

    use super::*;

    #[test]
    fn rms_norm_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        // Weight is 1D of size 3 (last dim of Shape2<U2, U3>)
        let weight = paramecia_core::Tensor::ones(3, paramecia_core::DType::F32, &device).unwrap();
        let mut op = RmsNormOp::<Shape2<U2, U3>>::new(weight, 1e-5);

        let res = op.forward(&mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);
    }
}
