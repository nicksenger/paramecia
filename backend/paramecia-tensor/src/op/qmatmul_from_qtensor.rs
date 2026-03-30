use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};
use paramecia_core::quantized::QTensor;

use crate::{Error, QMatMul};

/// Builds a typed `QMatMul<S>` from an untyped quantized tensor.
pub struct QMatMulFromQTensorOp<S>(PhantomData<S>);
impl<S> Default for QMatMulFromQTensorOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for QMatMulFromQTensorOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = QTensor;
    type Out = Result<QMatMul<S>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_output!("qmatmul_from_qtensor", S);
        let qmm = paramecia_core::quantized::QMatMul::from_qtensor(input)?;
        qmm.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for QMatMulFromQTensorOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "QMatMulFromQTensor",
            Some(&pretty_shape(std::any::type_name::<
                <S as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}
