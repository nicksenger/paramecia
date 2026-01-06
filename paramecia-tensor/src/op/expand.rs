use std::marker::PhantomData;

use glowstick::{op::broadcast, Shape, ShapeDiagnostic};

use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Broadcasts the lefthand tensor to the shape of the provided righthand tensor
/// or shape.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let c = expand!(&a, &b)?;
/// let d = expand!(&a, [U1, U4, U3, U2])?;
///
/// assert_eq!(c.dims(), &[1, 4, 3, 2]);
/// assert_eq!(d.dims(), &[1, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
///
/// When broadcasting to a shape, a combination of type-level integers and
/// expressions bound to dynamic dimensions may be provided.
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{expand, Tensor};
/// use glowstick::{Shape2, Shape4, num::{U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     B: BatchSize,
///     N: SequenceLength
/// }
///
/// let device = Device::Cpu;
/// let [batch_size, seq_len] = [4, 12];
/// let a = Tensor::<Shape2<U1, U2>>::ones(DType::F32, &device)?;
/// let b = expand!(&a, [{ batch_size } => B, { seq_len } => N, U3, U2])?;
///
/// assert_eq!(b.dims(), &[4, 12, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! expand {
    ($t:expr,[$($ds:tt)+]) => {{
        use $crate::op::expand::BroadcastAs;
        (
            $t,
            std::marker::PhantomData::<glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>>,
        )
            .expand(&paramecia_core::Shape::from_dims(&$crate::reshape_val!($($ds)+).into_array()))
    }};
    ($t1:expr,$t2:expr) => {{
        use $crate::op::expand::BroadcastAs;
        (
            $t1,
            $t2,
        )
            .expand($t2.inner().shape())
    }}
}

pub trait BroadcastAs {
    type Out;
    fn expand(&self, shape: &paramecia_core::Shape) -> Self::Out;
}

impl<S1, S2> BroadcastAs for (Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &paramecia_core::Shape) -> Self::Out {
        self.0.inner().expand(shape.clone())?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &paramecia_core::Shape) -> Self::Out {
        self.0.inner().expand(shape.clone())?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S1, S2): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &paramecia_core::Shape) -> Self::Out {
        self.0.inner().expand(shape.clone())?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, &Tensor<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &paramecia_core::Shape) -> Self::Out {
        self.0.inner().expand(shape.clone())?.try_into()
    }
}

impl<S1, S2> BroadcastAs for (&Tensor<S1>, PhantomData<S2>)
where
    S1: Shape,
    S2: Shape,
    (S2, S1): broadcast::Compatible,
{
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn expand(&self, shape: &paramecia_core::Shape) -> Self::Out {
        self.0.inner().expand(shape.clone())?.try_into()
    }
}

pub struct ExpandOp<S1, S2>(PhantomData<S1>, PhantomData<S2>);
impl<S1, S2> Default for ExpandOp<S1, S2> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S1, S2> Combinator for ExpandOp<S1, S2>
where
    S1: Shape + glowstick::ShapeDiagnostic,
    S2: Shape + glowstick::ShapeDiagnostic,
    (S2, S1): broadcast::Compatible,
{
    type In = (Tensor<S1>, Tensor<S2>);
    type Out = Result<Tensor<<(S2, S1) as broadcast::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: (Tensor<S1>, Tensor<S2>)) -> Self::Out {
        let _span = crate::op::trace::forward_2!("expand", S1, S2);
        let shape = input.1.inner().shape().clone();
        input.0.inner().expand(shape)?.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S1, S2> Vis for ExpandOp<S1, S2>
where
    S1: Shape + ShapeDiagnostic,
    S2: Shape + ShapeDiagnostic,
    (S2, S1): broadcast::Compatible,
    <(S2, S1) as broadcast::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Expand",
            Some(&pretty_shape(std::any::type_name::<
                <<(S2, S1) as broadcast::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use glowstick::{
        num::{U1, U2, U3, U4},
        Shape2, Shape4,
    };

    use crate::Tensor;

    #[test]
    fn expand_op() {
        let device = paramecia_core::Device::Cpu;

        type S1 = Shape2<U1, U2>;
        type S2 = Shape4<U1, U4, U3, U2>;
        let a = Tensor::<S1>::ones(paramecia_core::DType::F32, &device).unwrap();
        let b = Tensor::<S2>::ones(paramecia_core::DType::F32, &device).unwrap();

        let res =
            ExpandOp::<S1, S2>::forward(&mut ExpandOp(PhantomData, PhantomData), &mut (), (a, b))
                .unwrap();
        assert_eq!(res.dims(), &[1, 4, 3, 2]);
    }
}
