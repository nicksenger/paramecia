use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, QMatMul, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Arrow combinator adapter for `QMatMul<WS>`.
///
/// Takes an owned `Tensor<InS>`, forwards through the weight matrix,
/// and returns `Result<Tensor<OutS>, Error>`.
pub struct QMatMulOp<WS: Shape, InS, OutS> {
    weight: QMatMul<WS>,
    _phantom: PhantomData<(InS, OutS)>,
}

impl<WS, InS, OutS> QMatMulOp<WS, InS, OutS>
where
    WS: Shape,
{
    pub fn new(weight: QMatMul<WS>) -> Self {
        Self {
            weight,
            _phantom: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<WS, InS, OutS> Combinator for QMatMulOp<WS, InS, OutS>
where
    WS: Shape + glowstick::ShapeDiagnostic,
    InS: Shape + glowstick::ShapeDiagnostic,
    OutS: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<InS>;
    type Out = Result<Tensor<OutS>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<InS>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("qmatmul_op", InS);
        self.weight.forward(&input).map_err(Error::from)
    }
}
#[primitive(property = Visualize)]
impl<WS, InS, OutS> Vis for QMatMulOp<WS, InS, OutS>
where
    WS: Shape + ShapeDiagnostic,
    InS: Shape + ShapeDiagnostic,
    OutS: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "QMatMul",
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
        num::{U2, U3, U4},
        Shape2,
    };

    use super::*;

    fn make_qmatmul(out_dim: usize, in_dim: usize) -> paramecia_core::quantized::QMatMul {
        let t = paramecia_core::Tensor::ones(
            (out_dim, in_dim),
            paramecia_core::DType::F32,
            &paramecia_core::Device::Cpu,
        )
        .unwrap();
        paramecia_core::quantized::QMatMul::Tensor(t)
    }

    #[test]
    fn qmatmul_op_forward() {
        let core_qmm = make_qmatmul(4, 3);
        let qmm: QMatMul<Shape2<U4, U3>> = core_qmm.try_into().unwrap();

        let mut op = QMatMulOp::<Shape2<U4, U3>, Shape2<U2, U3>, Shape2<U2, U4>>::new(qmm);

        let input = Tensor::<Shape2<U2, U3>>::ones(
            paramecia_core::DType::F32,
            &paramecia_core::Device::Cpu,
        )
        .unwrap();

        let output = op.forward(&mut (), input).unwrap();
        assert_shape_eq!(output, Shape2<U2, U4>);
    }
}
