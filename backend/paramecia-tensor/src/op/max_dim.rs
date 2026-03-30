use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{
    num::{Unsigned, U0, U1},
    op::narrow,
    Shape, ShapeDiagnostic,
};

use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Computes the maximum of a tensor along a specified dimension, resulting in a tensor with size `U1` at that dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{max_dim, Tensor};
/// use glowstick::{Shape4, num::{U1, U2, U3, U4, U5}, dyndims};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U2, U3, U4, U5>>::ones(DType::F32, &device)?;
/// let maxed = max_dim!(a, U1)?;
///
/// assert_eq!(maxed.dims(), &[2, 1, 4, 5]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! max_dim {
    [$t:expr,$i:ty] => {{
        use $crate::op::max_dim::MaxDim;
        ($t, std::marker::PhantomData, std::marker::PhantomData::<$i>).max_dim()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::max_dim![$crate::max_dim![$t,$i],$($is),+]
    }};
}

pub trait MaxDim {
    type Out;
    fn max_dim(self) -> Self::Out;
}
impl<T, S, Dim> MaxDim for (T, PhantomData<S>, PhantomData<Dim>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, U0, U1) as narrow::Compatible>::Out>, Error>;
    fn max_dim(self) -> Self::Out {
        Ok(Tensor(
            self.0
                .borrow()
                .inner()
                .max_keepdim(<Dim as Unsigned>::USIZE)?,
            PhantomData,
        ))
    }
}

pub struct MaxDimOp<S, Dim>(PhantomData<S>, PhantomData<Dim>);
impl<S, Dim> Default for MaxDimOp<S, Dim> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, Dim> Combinator for MaxDimOp<S, Dim>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim, U0, U1) as narrow::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("max_dim", S);
        max_dim!(input, Dim)
    }
}
#[primitive(property = Visualize)]
impl<S, Dim> Vis for MaxDimOp<S, Dim>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
    <(S, Dim, U0, U1) as narrow::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("MaxDim(dim={})", <Dim as Unsigned>::USIZE),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, Dim, U0, U1) as narrow::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U1, U2, U3, U4, U5},
        Shape4,
    };

    use super::*;

    #[test]
    fn max_dim_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape4<U2, U3, U4, U5>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = MaxDimOp<Shape4<U2, U3, U4, U5>, U1>;
        let res = MyOp::forward(&mut MaxDimOp(PhantomData, PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape4<U2, U1, U4, U5>);
    }
}
