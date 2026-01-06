use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Applies the sigmoid activation function element-wise.
/// Shape-preserving: input and output have the same shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{sigmoid, Tensor};
/// use glowstick::{Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let result = sigmoid!(a)?;
///
/// assert_eq!(result.dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! sigmoid {
    ($t:expr) => {{
        use $crate::op::sigmoid::Sigmoid;
        $t.sigmoid()
    }};
}

pub trait Sigmoid {
    type Out;
    fn sigmoid(self) -> Self::Out;
}
impl<S> Sigmoid for Tensor<S>
where
    S: Shape,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn sigmoid(self) -> Self::Out {
        let result = paramecia_nn::ops::sigmoid(&self.0)?;
        result.try_into()
    }
}

pub struct SigmoidOp<S>(PhantomData<S>);
impl<S> Default for SigmoidOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for SigmoidOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("sigmoid", S);
        sigmoid!(input)
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for SigmoidOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Sigmoid",
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
    fn sigmoid_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = SigmoidOp<Shape2<U2, U3>>;
        let res = MyOp::forward(&mut SigmoidOp(PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);
    }
}
