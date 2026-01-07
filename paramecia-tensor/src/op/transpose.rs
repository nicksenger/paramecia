use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::transpose, Shape, ShapeDiagnostic};

use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Swaps the dimensions of a tensor.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{transpose, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let transposed = transpose!(a, U1, U2)?;
///
/// assert_eq!(transposed.dims(), &[2, 4, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! transpose {
    ($t:expr,$d1:ty,$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose()
    }};
    ($t:expr,$d1:ty:$d2:ty,$($d1s:ty:$d2s:ty),+) => {{
        use $crate::op::transpose::Transpose;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .transpose().and_then(|t| $crate::transpose!(&t, $($d1s:$d2s),+))
    }};
}

pub trait Transpose {
    type Out;
    fn transpose(&self) -> Self::Out;
}
impl<T, S, Dim1, Dim2> Transpose for (T, PhantomData<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as transpose::Compatible>::Out>, Error>;
    fn transpose(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .transpose(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}

pub struct TransposeOp<S, Dim1, Dim2>(PhantomData<S>, PhantomData<Dim1>, PhantomData<Dim2>);
impl<S, Dim1, Dim2> Default for TransposeOp<S, Dim1, Dim2> {
    fn default() -> Self {
        Self(PhantomData, PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, Dim1, Dim2> Combinator for TransposeOp<S, Dim1, Dim2>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim1, Dim2) as transpose::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("transpose", S);
        transpose!(input, Dim1, Dim2)
    }
}
#[primitive(property = Visualize)]
impl<S, Dim1, Dim2> Vis for TransposeOp<S, Dim1, Dim2>
where
    S: Shape + ShapeDiagnostic,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): transpose::Compatible,
    <(S, Dim1, Dim2) as transpose::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!(
                "Transpose(dim1={}, dim2={})",
                <Dim1 as Unsigned>::USIZE,
                <Dim2 as Unsigned>::USIZE
            ),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, Dim1, Dim2) as transpose::Compatible>::Out as ShapeDiagnostic>::Out,
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
    fn transpose_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape3<U2, U3, U4>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = TransposeOp<Shape3<U2, U3, U4>, U1, U2>;
        let res = MyOp::forward(
            &mut TransposeOp(PhantomData, PhantomData, PhantomData),
            &mut (),
            a,
        )
        .unwrap();
        assert_shape_eq!(res, Shape3<U2, U4, U3>);
    }
}
