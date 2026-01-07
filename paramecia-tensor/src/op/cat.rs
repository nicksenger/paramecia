use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::cat_dyn, Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Concatenates the given tensors along a specified dimension.
/// A dynamic dimension must be provided for the return type.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{cat, Tensor};
/// use glowstick::{Shape4, num::*, dyndims};
///
/// dyndims! {
///     B: BatchSize
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape4<U1, U4, U3, U2>>::ones(DType::F32, &device)?;
/// let concatenated = cat!(vec![a, b].as_slice(), U0 => B)?;
///
/// assert_eq!(concatenated.dims(), &[2, 4, 3, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! cat {
    ($ts:expr,$i:ty => $d:ty) => {{
        use $crate::op::cat::Cat;
        (
            $ts,
            std::marker::PhantomData::<$i>,
            std::marker::PhantomData::<$d>,
        )
            .cat()
    }};
}

pub trait Cat {
    type Out;
    fn cat(self) -> Self::Out;
}
impl<S, I, D> Cat for (&[Tensor<S>], PhantomData<I>, PhantomData<glowstick::Dyn<D>>)
where
    S: Shape,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type Out = Result<Tensor<<(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>, Error>;
    fn cat(self) -> Self::Out {
        let tensors: Vec<&paramecia_core::Tensor> = self.0.iter().map(|t| t.inner()).collect();
        let result = paramecia_core::Tensor::cat(&tensors, <I as Unsigned>::USIZE)?;
        result.try_into()
    }
}
impl<S, I, D> Cat
    for (
        &[&Tensor<S>],
        PhantomData<I>,
        PhantomData<glowstick::Dyn<D>>,
    )
where
    S: Shape,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type Out = Result<Tensor<<(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>, Error>;
    fn cat(self) -> Self::Out {
        let tensors: Vec<&paramecia_core::Tensor> = self.0.iter().map(|t| t.inner()).collect();
        let result = paramecia_core::Tensor::cat(&tensors, <I as Unsigned>::USIZE)?;
        result.try_into()
    }
}

pub struct CatOp<S, Dim, N>(PhantomData<S>, PhantomData<Dim>, PhantomData<N>);
impl<S, Dim, N> Default for CatOp<S, Dim, N> {
    fn default() -> Self {
        Self(PhantomData, PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, I, D> Combinator for CatOp<S, I, D>
where
    S: Shape + glowstick::ShapeDiagnostic,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
{
    type In = Vec<Tensor<S>>;
    type Out = Result<Tensor<<(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Vec<Tensor<S>>) -> Self::Out {
        let _span = crate::op::trace::forward_vec!("cat", S);
        let tensors: Vec<&paramecia_core::Tensor> = input.iter().map(|t| t.inner()).collect();
        let result = paramecia_core::Tensor::cat(&tensors, <I as Unsigned>::USIZE)?;
        result.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S, I, D> Vis for CatOp<S, I, D>
where
    S: Shape + ShapeDiagnostic,
    (S, I, glowstick::Dyn<D>): cat_dyn::Compatible,
    I: Unsigned,
    <(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("Cat(dim={})", <I as Unsigned>::USIZE),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, I, glowstick::Dyn<D>) as cat_dyn::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use glowstick::{
        num::{U0, U1, U2, U3, U4},
        Shape4,
    };

    use crate::Tensor;

    #[test]
    fn cat_op() {
        glowstick::dyndims! {
            B: BatchSize
        }

        let device = paramecia_core::Device::Cpu;

        type S = Shape4<U1, U4, U3, U2>;
        type A = Tensor<S>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();
        let b = A::ones(paramecia_core::DType::F32, &device).unwrap();

        let res = CatOp::<S, U0, B>::forward(
            &mut CatOp(PhantomData, PhantomData, PhantomData),
            &mut (),
            vec![a, b],
        )
        .unwrap();
        assert_eq!(res.dims(), &[2, 4, 3, 2]);
    }
}
