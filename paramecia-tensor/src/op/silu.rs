use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Applies the SiLU (Swish) activation function element-wise.
/// Shape-preserving: input and output have the same shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{silu, Tensor};
/// use glowstick::{Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let result = silu!(a)?;
///
/// assert_eq!(result.dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! silu {
    ($t:expr) => {{
        use $crate::op::silu::Silu;
        $t.silu()
    }};
}

pub trait Silu {
    type Out;
    fn silu(self) -> Self::Out;
}
impl<S> Silu for Tensor<S>
where
    S: Shape,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn silu(self) -> Self::Out {
        let result = paramecia_nn::ops::silu(&self.0)?;
        result.try_into()
    }
}

pub struct SiluOp<S>(PhantomData<S>);
impl<S> Default for SiluOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for SiluOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("silu", S);
        silu!(input)
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for SiluOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Silu",
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
    fn silu_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = SiluOp<Shape2<U2, U3>>;
        let res = MyOp::forward(&mut SiluOp(PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);
    }
}
