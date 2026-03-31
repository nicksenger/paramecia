use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Splits the last dimension into two explicit dimensions.
///
/// Input shape: `[..., D]`
/// Output shape: `[..., A, B]`
pub struct UnflattenLastOp<SIn, SOut> {
    dim_a: usize,
    dim_b: usize,
    _phantom: PhantomData<(SIn, SOut)>,
}

impl<SIn, SOut> UnflattenLastOp<SIn, SOut> {
    pub fn new(dim_a: usize, dim_b: usize) -> Self {
        Self {
            dim_a,
            dim_b,
            _phantom: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<SIn, SOut> Combinator for UnflattenLastOp<SIn, SOut>
where
    SIn: Shape + glowstick::ShapeDiagnostic,
    SOut: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<SIn>;
    type Out = Result<Tensor<SOut>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("unflatten_last", SIn);
        let core = input.into_inner();
        let mut dims = core.dims().to_vec();
        if dims.is_empty() {
            return Err(Error::Msg(
                "UnflattenLastOp requires rank >= 1 tensor".to_string(),
            ));
        }
        dims.pop();
        dims.push(self.dim_a);
        dims.push(self.dim_b);
        core.reshape(dims)?.try_into()
    }
}

#[primitive(property = Visualize)]
impl<SIn, SOut> Vis for UnflattenLastOp<SIn, SOut>
where
    SIn: Shape + ShapeDiagnostic,
    SOut: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "UnflattenLast",
            Some(&pretty_shape(std::any::type_name::<
                <SOut as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}
