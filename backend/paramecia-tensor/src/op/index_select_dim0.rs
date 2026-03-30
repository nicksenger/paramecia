use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

use crate::{Error, Tensor};

/// Selects rows along dimension 0 using the provided index tensor.
pub struct IndexSelectDim0Op<SIn, SIdx, SOut>(PhantomData<(SIn, SIdx, SOut)>);
impl<SIn, SIdx, SOut> Default for IndexSelectDim0Op<SIn, SIdx, SOut> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<SIn, SIdx, SOut> Combinator for IndexSelectDim0Op<SIn, SIdx, SOut>
where
    SIn: Shape + glowstick::ShapeDiagnostic,
    SIdx: Shape + glowstick::ShapeDiagnostic,
    SOut: Shape + glowstick::ShapeDiagnostic,
{
    type In = (Tensor<SIn>, Tensor<SIdx>);
    type Out = Result<Tensor<SOut>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_2!("index_select_dim0", SIn, SIdx);
        let (input, indices) = input;
        let selected = input.inner().index_select(indices.inner(), 0)?;
        selected.try_into()
    }
}
#[primitive(property = Visualize)]
impl<SIn, SIdx, SOut> Vis for IndexSelectDim0Op<SIn, SIdx, SOut>
where
    SIn: Shape,
    SIdx: Shape,
    SOut: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "IndexSelect(dim=0)",
            Some(&pretty_shape(std::any::type_name::<
                <SOut as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}
