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

/// Computes the mean of a tensor along a specified dimension, resulting in a tensor with size `U1` at that dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{mean_dim, Tensor};
/// use glowstick::{Shape4, num::{U1, U2, U3, U4, U5}, dyndims};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U2, U3, U4, U5>>::ones(DType::F32, &device)?;
/// let meaned = mean_dim!(a, U1)?;
///
/// assert_eq!(meaned.dims(), &[2, 1, 4, 5]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! mean_dim {
    [$t:expr,$i:ty] => {{
        use $crate::op::mean_dim::MeanDim;
        ($t, std::marker::PhantomData, std::marker::PhantomData::<$i>).mean_dim()
    }};
    [$t:expr,$i:ty,$($is:ty),+] => {{
        $crate::mean_dim![$crate::mean_dim![$t,$i],$($is),+]
    }};
}

pub trait MeanDim {
    type Out;
    fn mean_dim(self) -> Self::Out;
}
impl<T, S, Dim> MeanDim for (T, PhantomData<S>, PhantomData<Dim>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, U0, U1) as narrow::Compatible>::Out>, Error>;
    fn mean_dim(self) -> Self::Out {
        Ok(Tensor(
            self.0
                .borrow()
                .inner()
                .mean_keepdim(<Dim as Unsigned>::USIZE)?,
            PhantomData,
        ))
    }
}

pub struct MeanDimOp<S, Dim>(PhantomData<S>, PhantomData<Dim>);
impl<S, Dim> Default for MeanDimOp<S, Dim> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, Dim> Combinator for MeanDimOp<S, Dim>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim, U0, U1) as narrow::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("mean_dim", S);
        mean_dim!(input, Dim)
    }
}
#[primitive(property = Visualize)]
impl<S, Dim> Vis for MeanDimOp<S, Dim>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim, U0, U1): narrow::Compatible,
    <(S, Dim, U0, U1) as narrow::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("MeanDim(dim={})", <Dim as Unsigned>::USIZE),
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
    fn mean_dim_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape4<U2, U3, U4, U5>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = MeanDimOp<Shape4<U2, U3, U4, U5>, U1>;
        let res = MyOp::forward(&mut MeanDimOp(PhantomData, PhantomData), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape4<U2, U1, U4, U5>);
    }
}
