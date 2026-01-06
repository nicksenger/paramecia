use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::flatten, Shape, ShapeDiagnostic};

use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Flattens the given tensor from the specified start dimension to the end
/// dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{flatten, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let flattened = flatten!(a, [U0, U2])?;
///
/// assert_eq!(flattened.dims(), &[12, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! flatten {
    ($t:expr,[$d1:ty,$d2:ty]) => {{
        use $crate::op::flatten::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten()
    }};
    ($t:expr,[$d1:ty,$d2:ty],$([$d1s:ty,$d2s:ty]),+) => {{
        use $crate::op::flatten::Flatten;
        (
            $t,
            std::marker::PhantomData::<$d1>,
            std::marker::PhantomData::<$d2>,
        )
            .flatten().and_then(|t| $crate::flatten!(&t, $([$d1s,$d2s]),+))
    }};
}

pub trait Flatten {
    type Out;
    fn flatten(&self) -> Self::Out;
}
impl<S, Dim1, Dim2> Flatten for (Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as flatten::Compatible>::Out>, Error>;
    fn flatten(&self) -> Self::Out {
        self.0
            .inner()
            .flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}
impl<S, Dim1, Dim2> Flatten for (&Tensor<S>, PhantomData<Dim1>, PhantomData<Dim2>)
where
    S: Shape,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type Out = Result<Tensor<<(S, Dim1, Dim2) as flatten::Compatible>::Out>, Error>;
    fn flatten(&self) -> Self::Out {
        self.0
            .inner()
            .flatten(
                <Dim1 as glowstick::num::Unsigned>::USIZE,
                <Dim2 as glowstick::num::Unsigned>::USIZE,
            )?
            .try_into()
    }
}

pub struct FlattenOp<S, Dim1, Dim2>(PhantomData<S>, PhantomData<Dim1>, PhantomData<Dim2>);
impl<S, Dim1, Dim2> Default for FlattenOp<S, Dim1, Dim2> {
    fn default() -> Self {
        Self(PhantomData, PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, Dim1, Dim2> Combinator for FlattenOp<S, Dim1, Dim2>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim1, Dim2) as flatten::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("flatten", S);
        flatten!(input, [Dim1, Dim2])
    }
}
#[primitive(property = Visualize)]
impl<S, Dim1, Dim2> Vis for FlattenOp<S, Dim1, Dim2>
where
    S: Shape + ShapeDiagnostic,
    Dim1: Unsigned,
    Dim2: Unsigned,
    (S, Dim1, Dim2): flatten::Compatible,
    <(S, Dim1, Dim2) as flatten::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!(
                "Flatten(dim1={}, dim2={})",
                <Dim1 as Unsigned>::USIZE,
                <Dim2 as Unsigned>::USIZE
            ),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, Dim1, Dim2) as flatten::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        dynamic::{DynProduct, Term},
        dyndims,
        num::{U0, U1, U2, U3, U4},
        Dyn, Shape2, Shape3, Shape4,
    };

    use super::*;

    #[test]
    fn flatten_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape4<U1, U4, U3, U2>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = FlattenOp<Shape4<U1, U4, U3, U2>, U0, U2>;
        let res = MyOp::forward(
            &mut FlattenOp(PhantomData, PhantomData, PhantomData),
            &mut (),
            a,
        )
        .unwrap();
        assert_eq!(res.dims(), &[12, 2]);
    }

    #[test]
    #[ignore]
    fn flatten_dyn() {
        dyndims! {
            B: BatchSize,
            N: SequenceLength
        }

        let device = paramecia_core::Device::Cpu;

        type S = Shape3<B, N, U4>;
        type A = Tensor<S>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = FlattenOp<S, U0, U1>;
        let res = MyOp::forward(
            &mut FlattenOp(PhantomData, PhantomData, PhantomData),
            &mut (),
            a,
        )
        .unwrap();
        assert_shape_eq!(
            &res,
            Shape2<Dyn<Term<U1, DynProduct<BatchSize, SequenceLength>>>, U4>
        );
    }
}
