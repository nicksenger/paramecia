use std::marker::PhantomData;

use glowstick::Shape;
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Remaps index tensor values through an optional lookup tensor.
///
/// If `expert_remap` is `None`, this is an identity op.
/// If present, each index value in `input` is remapped with `index_select`.
pub struct RemapIndicesOp<SIndices, SMap: Shape> {
    expert_remap: Option<Tensor<SMap>>,
    _phantom: PhantomData<SIndices>,
}

impl<SIndices, SMap> RemapIndicesOp<SIndices, SMap>
where
    SMap: Shape,
{
    pub fn new(expert_remap: Option<Tensor<SMap>>) -> Self {
        Self {
            expert_remap,
            _phantom: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<SIndices, SMap> Combinator for RemapIndicesOp<SIndices, SMap>
where
    SIndices: Shape + glowstick::ShapeDiagnostic,
    SMap: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<SIndices>;
    type Out = Result<Tensor<SIndices>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("remap_indices", SIndices);
        let Some(remap) = &self.expert_remap else {
            return Ok(input);
        };
        let original_shape = input.inner().shape().clone();
        let flat_indices = input.inner().flatten_all()?;
        let remapped_flat = remap.inner().index_select(&flat_indices, 0)?;
        remapped_flat
            .reshape(original_shape)?
            .to_dtype(paramecia_core::DType::U32)?
            .try_into()
    }
}

#[primitive(property = Visualize)]
impl<SIndices, SMap> Vis for RemapIndicesOp<SIndices, SMap>
where
    SIndices: Shape,
    SMap: Shape,
{
    fn visualize() -> Graph {
        Graph::leaf("RemapIndices")
    }
}
