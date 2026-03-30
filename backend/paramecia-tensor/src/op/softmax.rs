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

/// Applies the softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{softmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let softmaxed = softmax!(a, U1)?;
///
/// assert_eq!(softmaxed.dims(), &[2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! softmax {
    ($t:expr,$i:ty) => {{
        use $crate::op::softmax::Softmax;
        ($t, std::marker::PhantomData::<$i>).softmax()
    }};
    ($t:expr,$i:ty,$($is:ty),+) => {{
        $crate::softmax!($crate::softmax!($t,$i),$($is),+)
    }};
}

pub trait Softmax {
    type Out;
    fn softmax(self) -> Self::Out;
}
impl<S, Dim> Softmax for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn softmax(self) -> Self::Out {
        let result = paramecia_nn::ops::softmax(&self.0 .0, <Dim as Unsigned>::USIZE)?;
        result.try_into()
    }
}

pub struct SoftmaxOp<S, Dim>(PhantomData<S>, PhantomData<Dim>);
impl<S, Dim> Default for SoftmaxOp<S, Dim> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, Dim> Combinator for SoftmaxOp<S, Dim>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("softmax", S);
        softmax!(input, Dim)
    }
}
#[primitive(property = Visualize)]
impl<S, Dim> Vis for SoftmaxOp<S, Dim>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("Softmax(dim={})", <Dim as Unsigned>::USIZE),
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
        num::{U1, U2, U3, U4},
        Shape3,
    };

    use super::*;

    #[test]
    fn softmax_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape3<U2, U3, U4>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = SoftmaxOp<Shape3<U2, U3, U4>, U1>;
        let res = MyOp::forward(&mut SoftmaxOp(PhantomData, PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape3<U2, U3, U4>);
    }
}
