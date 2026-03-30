use std::marker::PhantomData;

use glowstick::cmp::Greater;
use glowstick::num::Unsigned;
use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Computes cumulative sum along a given dimension.
/// Shape-preserving: input and output have the same shape.
#[macro_export]
macro_rules! cumsum {
    ($t:expr, $dim:ty) => {{
        use $crate::op::cumsum::CumSum;
        ($t, std::marker::PhantomData::<$dim>).cumsum_typed()
    }};
}

pub trait CumSum {
    type Out;
    fn cumsum_typed(self) -> Self::Out;
}
impl<S, Dim> CumSum for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn cumsum_typed(self) -> Self::Out {
        let result = self.0 .0.cumsum(<Dim as Unsigned>::USIZE)?;
        result.try_into()
    }
}

pub struct CumSumOp<S, Dim>(PhantomData<S>, PhantomData<Dim>);
impl<S, Dim> Default for CumSumOp<S, Dim> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, Dim> Combinator for CumSumOp<S, Dim>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("cumsum", S);
        cumsum!(input, Dim)
    }
}
#[primitive(property = Visualize)]
impl<S, Dim> Vis for CumSumOp<S, Dim>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("CumSum(dim={})", <Dim as Unsigned>::USIZE),
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
        num::{U1, U2, U3},
        Shape2,
    };

    use super::*;

    #[test]
    fn cumsum_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = CumSumOp<Shape2<U2, U3>, U1>;
        let res = MyOp::forward(&mut CumSumOp(PhantomData, PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);
    }
}
