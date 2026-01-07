use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::Tensor;
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Drops the typed wrapper, returning the underlying `paramecia_core::Tensor`.
///
/// This is the combinator form of `Tensor::into_inner()`.
pub struct IntoInnerOp<S>(PhantomData<S>);
impl<S> Default for IntoInnerOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// Drops the typed wrapper inside `Result`, preserving any prior error.
///
/// This enables composing `Tensor::into_inner()` after fallible typed ops.
pub struct IntoInnerResultOp<S, E>(PhantomData<(S, E)>);
impl<S, E> Default for IntoInnerResultOp<S, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, E> Combinator for IntoInnerResultOp<S, E>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Result<Tensor<S>, E>;
    type Out = Result<paramecia_core::Tensor, E>;
    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("into_inner", S);
        input.map(Tensor::into_inner)
    }
}
#[primitive(property = Visualize)]
impl<S, E> Vis for IntoInnerResultOp<S, E>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "IntoInnerResult",
            Some(&pretty_shape(std::any::type_name::<
                <S as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for IntoInnerOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = paramecia_core::Tensor;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("into_inner", S);
        input.into_inner()
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for IntoInnerOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "IntoInner",
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
    fn into_inner_op() {
        let device = paramecia_core::Device::Cpu;
        let a = Tensor::<Shape2<U2, U3>>::ones(paramecia_core::DType::F32, &device).unwrap();

        let mut op = IntoInnerOp::<Shape2<U2, U3>>::default();
        let core = op.forward(&mut (), a);
        assert_eq!(core.dims(), &[2, 3]);
    }
}
