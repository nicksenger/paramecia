use std::marker::PhantomData;

use glowstick::{op::broadcast, Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Performs element-wise multiplication with broadcasting.
/// The righthand tensor is broadcast to match the lefthand tensor's shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{broadcast_mul, Tensor};
/// use glowstick::{Shape1, Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape2<U1, U3>>::ones(DType::F32, &device)?;
/// let c = broadcast_mul!(a, b)?;
///
/// assert_eq!(c.dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! broadcast_mul {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::broadcast_mul::BroadcastMul;
        ($t1, $t2).broadcast_mul()
    }};
}

pub trait BroadcastMul {
    type Out;
    fn broadcast_mul(&self) -> Self::Out;
}

fn broadcast_mul_impl(
    a: &paramecia_core::Tensor,
    b: &paramecia_core::Tensor,
) -> Result<paramecia_core::Tensor, Error> {
    Ok(a.broadcast_mul(b)?)
}

impl<S1, S2> BroadcastMul for (Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_mul(&self) -> Self::Out {
        broadcast_mul_impl(self.0.inner(), self.1.inner())?.try_into()
    }
}
impl<S1, S2> BroadcastMul for (Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_mul(&self) -> Self::Out {
        broadcast_mul_impl(self.0.inner(), self.1.inner())?.try_into()
    }
}
impl<S1, S2> BroadcastMul for (&Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_mul(&self) -> Self::Out {
        broadcast_mul_impl(self.0.inner(), self.1.inner())?.try_into()
    }
}
impl<S1, S2> BroadcastMul for (&Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_mul(&self) -> Self::Out {
        broadcast_mul_impl(self.0.inner(), self.1.inner())?.try_into()
    }
}

pub struct BroadcastMulOp<S1, S2>(PhantomData<S1>, PhantomData<S2>);
impl<S1, S2> Default for BroadcastMulOp<S1, S2> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S1, S2> Combinator for BroadcastMulOp<S1, S2>
where
    S1: Shape + glowstick::ShapeDiagnostic,
    S2: Shape + glowstick::ShapeDiagnostic,
    (S1, S2): broadcast::Compatible,
{
    type In = (Tensor<S1>, Tensor<S2>);
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: (Tensor<S1>, Tensor<S2>)) -> Self::Out {
        let _span = crate::op::trace::forward_2!("broadcast_mul", S1, S2);
        broadcast_mul!(input.0, input.1)
    }
}
#[primitive(property = Visualize)]
impl<S1, S2> Vis for BroadcastMulOp<S1, S2>
where
    S1: Shape + ShapeDiagnostic,
    S2: Shape + ShapeDiagnostic,
    (S1, S2): broadcast::Compatible,
    <(S1, S2) as broadcast::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "BroadcastMul",
            Some(&pretty_shape(std::any::type_name::<
                <<(S1, S2) as broadcast::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U1, U2},
        Shape2,
    };

    use super::*;

    #[test]
    fn broadcast_mul_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U1, U2>>;
        type B = Tensor<Shape2<U2, U2>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();
        let b = B::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = BroadcastMulOp<Shape2<U1, U2>, Shape2<U2, U2>>;
        let res = MyOp::forward(
            &mut BroadcastMulOp(PhantomData, PhantomData),
            &mut (),
            (a, b),
        )
        .unwrap();
        assert_shape_eq!(res, Shape2<U2, U2>);
    }
}
