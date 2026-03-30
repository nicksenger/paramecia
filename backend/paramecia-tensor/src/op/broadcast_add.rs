use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{op::broadcast, Shape, ShapeDiagnostic};

use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Performs addition of the lefthand tensor and righthand tensor(s). The righthand
/// tensor(s) must be compatible for broadcast to the shape of the lefthand tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{broadcast_add, Tensor};
/// use glowstick::{Shape1, Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape2<U2, U2>>::ones(DType::F32, &device)?;
/// let c = broadcast_add!(a, b)?;
///
/// assert_eq!(
///     c.inner().to_vec2::<f32>()?,
///     vec![
///         vec![2., 2.],
///         vec![2., 2.]
///     ]
/// );
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! broadcast_add {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::broadcast_add::BroadcastAdd;
        ($t1, $t2).broadcast_add()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        use $crate::op::broadcast_add::BroadcastAdd;
        ($t1, $t2)
            .broadcast_add()
            .and_then(|t| $crate::broadcast_add!(&t, $t2s))
    }};
}

pub trait BroadcastAdd {
    type Out;
    fn broadcast_add(&self) -> Self::Out;
}
impl<S1, S2> BroadcastAdd for (Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0
            .inner()
            .broadcast_add(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, S2> BroadcastAdd for (Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.inner())?.try_into()
    }
}
impl<S1, S2> BroadcastAdd for (&Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0
            .inner()
            .broadcast_add(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, S2> BroadcastAdd for (&Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn broadcast_add(&self) -> Self::Out {
        self.0.inner().broadcast_add(self.1.inner())?.try_into()
    }
}

pub struct BroadcastAddOp<S1, S2>(PhantomData<S1>, PhantomData<S2>);
impl<S1, S2> Default for BroadcastAddOp<S1, S2> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S1, S2> Combinator for BroadcastAddOp<S1, S2>
where
    S1: Shape + glowstick::ShapeDiagnostic,
    S2: Shape + glowstick::ShapeDiagnostic,
    (S1, S2): broadcast::Compatible,
{
    type In = (Tensor<S1>, Tensor<S2>);
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: (Tensor<S1>, Tensor<S2>)) -> Self::Out {
        let _span = crate::op::trace::forward_2!("broadcast_add", S1, S2);
        broadcast_add!(input.0, input.1)
    }
}
#[primitive(property = Visualize)]
impl<S1, S2> Vis for BroadcastAddOp<S1, S2>
where
    S1: Shape + ShapeDiagnostic,
    S2: Shape + ShapeDiagnostic,
    (S1, S2): broadcast::Compatible,
    <(S1, S2) as broadcast::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "BroadcastAdd",
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
    fn broadcast_add_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U1, U2>>;
        type B = Tensor<Shape2<U2, U2>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();
        let b = B::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = BroadcastAddOp<Shape2<U1, U2>, Shape2<U2, U2>>;
        let res = MyOp::forward(
            &mut BroadcastAddOp(PhantomData, PhantomData),
            &mut (),
            (a, b),
        )
        .unwrap();
        assert_shape_eq!(res, Shape2<U2, U2>);
    }
}
