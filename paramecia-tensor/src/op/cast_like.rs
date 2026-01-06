use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Casts the right tensor to the dtype and device of the left tensor.
///
/// Returns `(left, right_casted)` preserving shapes.
pub struct CastLikeOp<SLeft, SRight = SLeft>(PhantomData<(SLeft, SRight)>);
impl<SLeft, SRight> Default for CastLikeOp<SLeft, SRight> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[primitive(property = Arrow)]
impl<SLeft, SRight> Combinator for CastLikeOp<SLeft, SRight>
where
    SLeft: Shape + glowstick::ShapeDiagnostic,
    SRight: Shape + glowstick::ShapeDiagnostic,
{
    type In = (Tensor<SLeft>, Tensor<SRight>);
    type Out = Result<(Tensor<SLeft>, Tensor<SRight>), Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_2!("cast_like", SLeft, SRight);
        let (left, right) = input;
        let right = right
            .to_dtype(left.inner().dtype())?
            .to_device(left.inner().device())?;
        Ok((left, right))
    }
}

#[primitive(property = Visualize)]
impl<SLeft, SRight> Vis for CastLikeOp<SLeft, SRight>
where
    SLeft: Shape + ShapeDiagnostic,
    SRight: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf("CastLike")
    }
}
