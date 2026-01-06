use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

use crate::{Error, Tensor};

/// Scatter-adds `src` rows into `base` along dimension 0 at the provided indices.
pub struct IndexAddDim0Op<SBase, SIdx, SSrc>(PhantomData<(SBase, SIdx, SSrc)>);
impl<SBase, SIdx, SSrc> Default for IndexAddDim0Op<SBase, SIdx, SSrc> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<SBase, SIdx, SSrc> Combinator for IndexAddDim0Op<SBase, SIdx, SSrc>
where
    SBase: Shape + glowstick::ShapeDiagnostic,
    SIdx: Shape + glowstick::ShapeDiagnostic,
    SSrc: Shape + glowstick::ShapeDiagnostic,
{
    type In = (Tensor<SBase>, Tensor<SIdx>, Tensor<SSrc>);
    type Out = Result<Tensor<SBase>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_3!("index_add_dim0", SBase, SIdx, SSrc);
        let (base, indices, src) = input;
        let out = base.inner().index_add(indices.inner(), src.inner(), 0)?;
        out.try_into()
    }
}
#[primitive(property = Visualize)]
impl<SBase, SIdx, SSrc> Vis for IndexAddDim0Op<SBase, SIdx, SSrc>
where
    SBase: Shape + ShapeDiagnostic,
    SIdx: Shape,
    SSrc: Shape,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "IndexAdd(dim=0)",
            Some(&pretty_shape(std::any::type_name::<
                <SBase as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}
