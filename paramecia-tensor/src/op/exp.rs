use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Applies element-wise exponential.
/// Shape-preserving: input and output have the same shape.
#[macro_export]
macro_rules! exp {
    ($t:expr) => {{
        use $crate::op::exp::Exp;
        $t.exp_typed()
    }};
}

pub trait Exp {
    type Out;
    fn exp_typed(self) -> Self::Out;
}
impl<S> Exp for Tensor<S>
where
    S: Shape,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn exp_typed(self) -> Self::Out {
        let result = self.0.exp()?;
        result.try_into()
    }
}

pub struct ExpOp<S>(PhantomData<S>);
impl<S> Default for ExpOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for ExpOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("exp", S);
        exp!(input)
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for ExpOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Exp",
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
    fn exp_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = ExpOp<Shape2<U2, U3>>;
        let res = MyOp::forward(&mut ExpOp(PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);
    }
}
