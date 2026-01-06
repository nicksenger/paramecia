use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Wraps an untyped `paramecia_core::Tensor` into a typed `Tensor<S>`.
///
/// This is the combinator form of `Tensor::<S>::try_from(core_tensor)`.
pub struct TryTypedOp<S>(PhantomData<S>);
impl<S> Default for TryTypedOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for TryTypedOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = paramecia_core::Tensor;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: paramecia_core::Tensor) -> Self::Out {
        let _span = crate::op::trace::forward_output!("try_typed", S);
        input.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for TryTypedOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "TryTyped",
            Some(&pretty_shape(std::any::type_name::<
                <S as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        num::{U2, U3},
        Shape2,
    };

    use super::*;

    #[test]
    fn try_typed_op_ok() {
        let device = paramecia_core::Device::Cpu;
        let core =
            paramecia_core::Tensor::ones((2, 3), paramecia_core::DType::F32, &device).unwrap();

        let mut op = TryTypedOp::<Shape2<U2, U3>>::default();
        let typed = op.forward(&mut (), core).unwrap();
        assert_eq!(typed.dims(), vec![2, 3]);
    }

    #[test]
    fn try_typed_op_err() {
        let device = paramecia_core::Device::Cpu;
        let core =
            paramecia_core::Tensor::ones((3, 2), paramecia_core::DType::F32, &device).unwrap();

        let mut op = TryTypedOp::<Shape2<U2, U3>>::default();
        let result = op.forward(&mut (), core);
        assert!(result.is_err());
    }
}
