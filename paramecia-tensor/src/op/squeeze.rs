use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::squeeze, Shape, ShapeDiagnostic};

use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Squeezes the specified dimensions from a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{squeeze, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U2, U3, U1>>::ones(DType::F32, &device)?;
/// let squeezed = squeeze![a, U0, U3]?; // Squeezes dimensions 0 and 3
///
/// assert_eq!(squeezed.dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! squeeze {
    [$t:expr,$i:ty] => {{
        glowstick::op::squeeze::check::<_, _, $i>(&$t);
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
            .and_then(|t| $crate::squeeze_next![t, $($is),+])
    }};
}
#[macro_export]
macro_rules! squeeze_next {
    [$t:expr,$i:ty] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<<$i as std::ops::Sub<glowstick::num::U1>>::Output>).squeeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::squeeze::Squeeze;
        ($t, std::marker::PhantomData::<$i>).squeeze()
            .and_then(|t| $crate::squeeze_next![t, $($is),+])
    }};
}

pub trait Squeeze {
    type Out;
    fn squeeze(&self) -> Self::Out;
}
impl<S, Dim> Squeeze for (&Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn squeeze(&self) -> Self::Out {
        self.0.inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}
impl<S, Dim> Squeeze for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn squeeze(&self) -> Self::Out {
        self.0.inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}

pub struct SqueezeOp<S, Dim>(PhantomData<S>, PhantomData<Dim>);
impl<S, Dim> Default for SqueezeOp<S, Dim> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, Dim> Combinator for SqueezeOp<S, Dim>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim) as squeeze::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("squeeze", S);
        input.inner().squeeze(<Dim as Unsigned>::USIZE)?.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S, Dim> Vis for SqueezeOp<S, Dim>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim): squeeze::Compatible,
    <(S, Dim) as squeeze::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("Squeeze(dim={})", <Dim as Unsigned>::USIZE),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, Dim) as squeeze::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use glowstick::{
        assert_shape_eq,
        num::{U0, U1, U2, U3},
        Shape3, Shape4,
    };

    use crate::Tensor;

    #[test]
    fn squeeze_op() {
        let device = paramecia_core::Device::Cpu;

        type S = Shape4<U1, U2, U3, U1>;
        type A = Tensor<S>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        let res = SqueezeOp::<S, U0>::forward(&mut SqueezeOp(PhantomData, PhantomData), &mut (), a)
            .unwrap();
        assert_shape_eq!(res, Shape3<U2, U3, U1>);
    }
}
