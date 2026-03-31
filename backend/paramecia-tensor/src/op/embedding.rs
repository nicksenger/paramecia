use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Typed embedding lookup op.
///
/// Takes token indices with shape `InS` and produces embeddings with shape `OutS`.
/// The output shape's last dimension is the embedding hidden size.
pub struct EmbeddingOp<InS, OutS> {
    inner: paramecia_nn::Embedding,
    _phantom: PhantomData<(InS, OutS)>,
}

impl<InS, OutS> EmbeddingOp<InS, OutS> {
    pub fn new(embedding: paramecia_nn::Embedding) -> Self {
        Self {
            inner: embedding,
            _phantom: PhantomData,
        }
    }

    pub fn embeddings(&self) -> &paramecia_core::Tensor {
        self.inner.embeddings()
    }

    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
}

impl<InS: Shape, OutS: Shape> std::fmt::Debug for EmbeddingOp<InS, OutS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EmbeddingOp<{}, {}>(hidden_size={})",
            std::any::type_name::<InS>(),
            std::any::type_name::<OutS>(),
            self.inner.hidden_size()
        )
    }
}

impl<InS: Shape, OutS: Shape> Clone for EmbeddingOp<InS, OutS> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<InS, OutS> Combinator for EmbeddingOp<InS, OutS>
where
    InS: Shape + glowstick::ShapeDiagnostic,
    OutS: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<InS>;
    type Out = Result<Tensor<OutS>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<InS>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("embedding", InS);
        let result = paramecia_core::Module::forward(&self.inner, &input.into_inner())?;
        result.try_into()
    }
}
#[primitive(property = Visualize)]
impl<InS, OutS> Vis for EmbeddingOp<InS, OutS>
where
    InS: Shape + ShapeDiagnostic,
    OutS: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Embedding",
            Some(&pretty_shape(std::any::type_name::<
                <OutS as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U3, U4},
        Shape1, Shape2,
    };

    use super::*;

    #[test]
    fn embedding_op() {
        let device = paramecia_core::Device::Cpu;

        // Embedding matrix: 10 tokens, 4-dim embeddings
        let emb_weight =
            paramecia_core::Tensor::ones((10, 4), paramecia_core::DType::F32, &device).unwrap();
        let embedding = paramecia_nn::Embedding::new(emb_weight, 4);

        // Input: 3 token indices
        let input = Tensor::<Shape1<U3>>::from_vec(vec![0u32, 1, 2], &device).unwrap();

        // Output: Shape2<U3, U4> (3 tokens × 4 features)
        let mut op = EmbeddingOp::<Shape1<U3>, Shape2<U3, U4>>::new(embedding);
        let res = op.forward(&mut (), input).unwrap();
        assert_shape_eq!(res, Shape2<U3, U4>);
        assert_eq!(res.dims(), vec![3, 4]);
    }
}
