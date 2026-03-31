use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Reshapes a rank-3 tensor `[D0, D1, D2]` into `[D0*D1, D2]`.
///
/// The output shape is validated by typed conversion to `Tensor<SOut>`,
/// so dynamic/static contracts remain enforced.
pub struct FlattenPrefix2Op<SIn, SOut>(PhantomData<(SIn, SOut)>);
impl<SIn, SOut> Default for FlattenPrefix2Op<SIn, SOut> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[primitive(property = Arrow)]
impl<SIn, SOut> Combinator for FlattenPrefix2Op<SIn, SOut>
where
    SIn: Shape + glowstick::ShapeDiagnostic,
    SOut: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<SIn>;
    type Out = Result<Tensor<SOut>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("flatten_prefix2", SIn);
        let (d0, d1, d2) = input.inner().dims3()?;
        let flattened = d0
            .checked_mul(d1)
            .ok_or_else(|| Error::Msg("FlattenPrefix2Op overflow for D0 * D1".to_string()))?;
        input.into_inner().reshape((flattened, d2))?.try_into()
    }
}

#[primitive(property = Visualize)]
impl<SIn, SOut> Vis for FlattenPrefix2Op<SIn, SOut>
where
    SIn: Shape + ShapeDiagnostic,
    SOut: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "FlattenPrefix2",
            Some(&pretty_shape(std::any::type_name::<
                <SOut as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}
