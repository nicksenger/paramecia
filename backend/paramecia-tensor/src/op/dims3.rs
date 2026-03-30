use std::marker::PhantomData;

use glowstick::Shape;
use inception::primitive;
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};

use crate::{Error, Tensor};

/// Extracts runtime dimensions `(d0, d1, d2)` from a rank-3 typed tensor.
pub struct Dims3Op<S>(PhantomData<S>);
impl<S> Default for Dims3Op<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for Dims3Op<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<(usize, usize, usize), Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("dims3", S);
        Ok(input.inner().dims3()?)
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for Dims3Op<S>
where
    S: Shape,
{
    fn visualize() -> Graph {
        Graph::leaf("Dims3")
    }
}
