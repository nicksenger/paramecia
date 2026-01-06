use std::marker::PhantomData;

use glowstick::Shape;
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Computes top-k routing directly from router logits.
///
/// Input: `[batch, seq, experts]` logits.
/// Output: `(top_weights, top_indices)`, each shaped `[batch, seq, topk]`.
pub struct TopkFromLogitsOp<SIn, SWeights, SIndices> {
    topk: usize,
    _phantom: PhantomData<(SIn, SWeights, SIndices)>,
}

impl<SIn, SWeights, SIndices> TopkFromLogitsOp<SIn, SWeights, SIndices> {
    pub fn new(topk: usize) -> Self {
        Self {
            topk,
            _phantom: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<SIn, SWeights, SIndices> Combinator for TopkFromLogitsOp<SIn, SWeights, SIndices>
where
    SIn: Shape + glowstick::ShapeDiagnostic,
    SWeights: Shape + glowstick::ShapeDiagnostic,
    SIndices: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<SIn>;
    type Out = Result<(Tensor<SWeights>, Tensor<SIndices>), Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("topk_from_logits", SIn);
        let logits = input.inner().contiguous()?;
        let (batch, seq, experts) = logits.dims3()?;
        let logits_flat = logits.reshape((batch * seq, experts))?;
        let (top_weights, top_indices) =
            paramecia_core::deltanet_ops::topk_softmax(&logits_flat, self.topk)?;
        let top_weights = top_weights.reshape((batch, seq, self.topk))?;
        let top_indices = top_indices.reshape((batch, seq, self.topk))?;
        Ok((top_weights.try_into()?, top_indices.try_into()?))
    }
}

#[primitive(property = Visualize)]
impl<SIn, SWeights, SIndices> Vis for TopkFromLogitsOp<SIn, SWeights, SIndices>
where
    SIn: Shape,
    SWeights: Shape,
    SIndices: Shape,
{
    fn visualize() -> Graph {
        Graph::leaf("TopkFromLogits")
    }
}
