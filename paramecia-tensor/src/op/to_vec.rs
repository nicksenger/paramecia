use std::marker::PhantomData;

use glowstick::Shape;
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Extracts host data as `Vec<T>` from a rank-1 tensor.
pub struct ToVec1Op<S, T>(PhantomData<(S, T)>);
impl<S, T> Default for ToVec1Op<S, T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, T> Combinator for ToVec1Op<S, T>
where
    S: Shape + glowstick::ShapeDiagnostic,
    T: paramecia_core::WithDType,
{
    type In = Tensor<S>;
    type Out = Result<Vec<T>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("to_vec", S);
        input.inner().to_vec1::<T>().map_err(Into::into)
    }
}
#[primitive(property = Visualize)]
impl<S, T> Vis for ToVec1Op<S, T>
where
    S: Shape,
    T: paramecia_core::WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("ToVec1")
    }
}

/// Flattens any tensor then extracts host data as `Vec<T>`.
pub struct FlattenAllToVec1Op<S, T>(PhantomData<(S, T)>);
impl<S, T> Default for FlattenAllToVec1Op<S, T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, T> Combinator for FlattenAllToVec1Op<S, T>
where
    S: Shape + glowstick::ShapeDiagnostic,
    T: paramecia_core::WithDType,
{
    type In = Tensor<S>;
    type Out = Result<Vec<T>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("to_vec", S);
        input
            .inner()
            .flatten_all()?
            .to_vec1::<T>()
            .map_err(Into::into)
    }
}
#[primitive(property = Visualize)]
impl<S, T> Vis for FlattenAllToVec1Op<S, T>
where
    S: Shape,
    T: paramecia_core::WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("FlattenAllToVec1")
    }
}

/// Extracts host data as `Vec<Vec<T>>` from a rank-2 tensor.
pub struct ToVec2Op<S, T>(PhantomData<(S, T)>);
impl<S, T> Default for ToVec2Op<S, T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, T> Combinator for ToVec2Op<S, T>
where
    S: Shape + glowstick::ShapeDiagnostic,
    T: paramecia_core::WithDType,
{
    type In = Tensor<S>;
    type Out = Result<Vec<Vec<T>>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("to_vec", S);
        input.inner().to_vec2::<T>().map_err(Into::into)
    }
}
#[primitive(property = Visualize)]
impl<S, T> Vis for ToVec2Op<S, T>
where
    S: Shape,
    T: paramecia_core::WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("ToVec2")
    }
}

/// Casts to `T`, flattens the tensor, then extracts host data as `Vec<T>`.
pub struct CastFlattenToVec1Op<S, T>(PhantomData<(S, T)>);
impl<S, T> Default for CastFlattenToVec1Op<S, T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, T> Combinator for CastFlattenToVec1Op<S, T>
where
    S: Shape + glowstick::ShapeDiagnostic,
    T: paramecia_core::WithDType,
{
    type In = Tensor<S>;
    type Out = Result<Vec<T>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("to_vec", S);
        input
            .inner()
            .to_dtype(<T as paramecia_core::WithDType>::DTYPE)?
            .flatten_all()?
            .to_vec1::<T>()
            .map_err(Into::into)
    }
}
#[primitive(property = Visualize)]
impl<S, T> Vis for CastFlattenToVec1Op<S, T>
where
    S: Shape,
    T: paramecia_core::WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("CastFlattenToVec1")
    }
}

/// Casts to `T` then extracts host data as `Vec<Vec<T>>`.
pub struct CastToVec2Op<S, T>(PhantomData<(S, T)>);
impl<S, T> Default for CastToVec2Op<S, T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, T> Combinator for CastToVec2Op<S, T>
where
    S: Shape + glowstick::ShapeDiagnostic,
    T: paramecia_core::WithDType,
{
    type In = Tensor<S>;
    type Out = Result<Vec<Vec<T>>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("to_vec", S);
        input
            .inner()
            .to_dtype(<T as paramecia_core::WithDType>::DTYPE)?
            .to_vec2::<T>()
            .map_err(Into::into)
    }
}
#[primitive(property = Visualize)]
impl<S, T> Vis for CastToVec2Op<S, T>
where
    S: Shape,
    T: paramecia_core::WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("CastToVec2")
    }
}

/// Casts each input tensor to requested dtypes, flattens, and extracts host vectors.
pub struct CastFlattenToVec1PairOp<SA, SB, A, B>(PhantomData<(SA, SB, A, B)>);
impl<SA, SB, A, B> Default for CastFlattenToVec1PairOp<SA, SB, A, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<SA, SB, A, B> Combinator for CastFlattenToVec1PairOp<SA, SB, A, B>
where
    SA: Shape + glowstick::ShapeDiagnostic,
    SB: Shape + glowstick::ShapeDiagnostic,
    A: paramecia_core::WithDType,
    B: paramecia_core::WithDType,
{
    type In = (Tensor<SA>, Tensor<SB>);
    type Out = Result<(Vec<A>, Vec<B>), Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_2!("to_vec", SA, SB);
        let (a, b) = input;
        let a_vec = a
            .inner()
            .to_dtype(<A as paramecia_core::WithDType>::DTYPE)?
            .flatten_all()?
            .to_vec1::<A>()?;
        let b_vec = b
            .inner()
            .to_dtype(<B as paramecia_core::WithDType>::DTYPE)?
            .flatten_all()?
            .to_vec1::<B>()?;
        Ok((a_vec, b_vec))
    }
}
#[primitive(property = Visualize)]
impl<SA, SB, A, B> Vis for CastFlattenToVec1PairOp<SA, SB, A, B>
where
    SA: Shape,
    SB: Shape,
    A: paramecia_core::WithDType,
    B: paramecia_core::WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("CastFlattenToVec1Pair")
    }
}

/// Casts each input tensor to requested dtypes, then extracts host matrices.
pub struct CastToVec2PairOp<SA, SB, A, B>(PhantomData<(SA, SB, A, B)>);
impl<SA, SB, A, B> Default for CastToVec2PairOp<SA, SB, A, B> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<SA, SB, A, B> Combinator for CastToVec2PairOp<SA, SB, A, B>
where
    SA: Shape + glowstick::ShapeDiagnostic,
    SB: Shape + glowstick::ShapeDiagnostic,
    A: paramecia_core::WithDType,
    B: paramecia_core::WithDType,
{
    type In = (Tensor<SA>, Tensor<SB>);
    type Out = Result<(Vec<Vec<A>>, Vec<Vec<B>>), Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_2!("to_vec", SA, SB);
        let (a, b) = input;
        let a_vec = a
            .inner()
            .to_dtype(<A as paramecia_core::WithDType>::DTYPE)?
            .to_vec2::<A>()?;
        let b_vec = b
            .inner()
            .to_dtype(<B as paramecia_core::WithDType>::DTYPE)?
            .to_vec2::<B>()?;
        Ok((a_vec, b_vec))
    }
}
#[primitive(property = Visualize)]
impl<SA, SB, A, B> Vis for CastToVec2PairOp<SA, SB, A, B>
where
    SA: Shape,
    SB: Shape,
    A: paramecia_core::WithDType,
    B: paramecia_core::WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("CastToVec2Pair")
    }
}
