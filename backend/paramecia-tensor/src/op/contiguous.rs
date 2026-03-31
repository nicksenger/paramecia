use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Ensures a tensor has contiguous memory layout.
/// Shape-preserving: input and output have the same shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{contiguous, Tensor};
/// use glowstick::{Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let result = contiguous!(a)?;
///
/// assert_eq!(result.dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! contiguous {
    ($t:expr) => {{
        use $crate::op::contiguous::Contiguous;
        $t.contiguous_typed()
    }};
}

pub trait Contiguous {
    type Out;
    fn contiguous_typed(self) -> Self::Out;
}
impl<S> Contiguous for Tensor<S>
where
    S: Shape,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn contiguous_typed(self) -> Self::Out {
        let result = self.0.contiguous()?;
        result.try_into()
    }
}

pub struct ContiguousOp<S>(PhantomData<S>);
impl<S> Default for ContiguousOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for ContiguousOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("contiguous", S);
        contiguous!(input)
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for ContiguousOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Contiguous",
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
    fn contiguous_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = ContiguousOp<Shape2<U2, U3>>;
        let res = MyOp::forward(&mut ContiguousOp(PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);
    }
}
