use std::marker::PhantomData;

use glowstick::Shape;
use inception::primitive;
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};
use paramecia_core::DType;

use crate::{Error, Tensor};

/// Groups flattened top-k routing outputs by expert index.
///
/// Input tensors must be rank-2 `[tokens, top_k]`:
/// - `indices`: expert ids per token slot (cast to `u32`)
/// - `weights`: routing weights per token slot (cast to `f32`)
///
/// Output:
/// - `top_x[expert]`: token ids assigned to `expert`
/// - `selected_rws[expert]`: routing weights aligned with `top_x[expert]`
pub struct GroupTopKAssignmentsOp<S> {
    num_experts: usize,
    marker: PhantomData<S>,
}

impl<S> GroupTopKAssignmentsOp<S> {
    pub fn new(num_experts: usize) -> Self {
        Self {
            num_experts,
            marker: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<S> Combinator for GroupTopKAssignmentsOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = (Tensor<S>, Tensor<S>);
    type Out = Result<(Vec<Vec<u32>>, Vec<Vec<f32>>), Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_2!("group_topk_assignments", S, S);
        let (indices, weights) = input;
        let indices_vec: Vec<Vec<u32>> = indices.inner().to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let weights_vec: Vec<Vec<f32>> = weights.inner().to_dtype(DType::F32)?.to_vec2::<f32>()?;

        if indices_vec.len() != weights_vec.len() {
            return Err(Error::Msg(format!(
                "top-k grouping token length mismatch: indices rows {}, weights rows {}",
                indices_vec.len(),
                weights_vec.len()
            )));
        }

        let mut top_x = vec![Vec::new(); self.num_experts];
        let mut selected_rws = vec![Vec::new(); self.num_experts];

        for (token_idx, (token_experts, token_weights)) in
            indices_vec.iter().zip(weights_vec.iter()).enumerate()
        {
            if token_experts.len() != token_weights.len() {
                return Err(Error::Msg(format!(
                    "top-k grouping width mismatch at token {}: indices {}, weights {}",
                    token_idx,
                    token_experts.len(),
                    token_weights.len()
                )));
            }

            for (&expert_id, &weight) in token_experts.iter().zip(token_weights.iter()) {
                let expert_idx = expert_id as usize;
                if expert_idx < self.num_experts {
                    top_x[expert_idx].push(token_idx as u32);
                    selected_rws[expert_idx].push(weight);
                }
            }
        }

        Ok((top_x, selected_rws))
    }
}

#[primitive(property = Visualize)]
impl<S> Vis for GroupTopKAssignmentsOp<S>
where
    S: Shape,
{
    fn visualize() -> Graph {
        Graph::leaf("GroupTopKAssignments")
    }
}
