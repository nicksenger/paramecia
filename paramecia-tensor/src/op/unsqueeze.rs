use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::unsqueeze, Shape, ShapeDiagnostic};

use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Unsqueezes a tensor at the specified dimension(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{unsqueeze, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U1, U4>>::ones(DType::F32, &device)?;
/// let unsqueezed = unsqueeze![a, U0, U4]?;
///
/// assert_eq!(unsqueezed.dims(), &[1, 2, 1, 4, 1]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! unsqueeze {
    [$t:expr,$i:ty] => {{
        use $crate::op::unsqueeze::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        use $crate::op::unsqueeze::Unsqueeze;
        ($t, std::marker::PhantomData::<$i>).unsqueeze()
            .and_then(|t| $crate::unsqueeze![t, $($is),+])
    }};
}

#[allow(unused)]
pub trait Unsqueeze {
    type Out;
    fn unsqueeze(&self) -> Self::Out;
}
impl<S, Dim> Unsqueeze for (&Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn unsqueeze(&self) -> Self::Out {
        self.0
            .inner()
            .unsqueeze(<Dim as Unsigned>::USIZE)?
            .try_into()
    }
}
impl<S, Dim> Unsqueeze for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn unsqueeze(&self) -> Self::Out {
        self.0
            .inner()
            .unsqueeze(<Dim as Unsigned>::USIZE)?
            .try_into()
    }
}

pub struct UnsqueezeOp<S, Dim>(pub PhantomData<S>, pub PhantomData<Dim>);
#[primitive(property = Arrow)]
impl<S, Dim> Combinator for UnsqueezeOp<S, Dim>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim) as unsqueeze::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("unsqueeze", S);
        unsqueeze![input, Dim]
    }
}
#[primitive(property = Visualize)]
impl<S, Dim> Vis for UnsqueezeOp<S, Dim>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim): unsqueeze::Compatible,
    <(S, Dim) as unsqueeze::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("Unsqueeze(dim={})", <Dim as Unsigned>::USIZE),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, Dim) as unsqueeze::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}
impl<S, Dim> Default for UnsqueezeOp<S, Dim> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U0, U1, U2, U3, U4},
        Shape3, Shape4,
    };

    use super::*;

    #[test]
    fn unsqueeze_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape3<U2, U3, U4>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = UnsqueezeOp<Shape3<U2, U3, U4>, U0>;
        let res = MyOp::forward(&mut UnsqueezeOp(PhantomData, PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape4<U1, U2, U3, U4>);
    }
}
