use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::narrow_dyn_start, Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

pub trait NarrowDynStart {
    type Out;
    fn narrow_dyn_start(&self) -> Self::Out;
}
impl<T, S, Dim, Len> NarrowDynStart
    for (T, PhantomData<S>, PhantomData<Dim>, usize, PhantomData<Len>)
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow_dyn_start::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Len) as narrow_dyn_start::Compatible>::Out>, Error>;
    fn narrow_dyn_start(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, <Len as Unsigned>::USIZE)?
            .try_into()
    }
}

pub struct NarrowDynStartOp<S, Dim, Len> {
    _phantom: PhantomData<(S, Dim, Len)>,
    start: usize,
}
impl<S, Dim, Len> NarrowDynStartOp<S, Dim, Len> {
    pub fn new(start: usize) -> Self {
        Self {
            _phantom: PhantomData,
            start,
        }
    }
}
#[primitive(property = Arrow)]
impl<S, Dim, Len> Combinator for NarrowDynStartOp<S, Dim, Len>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow_dyn_start::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim, Len) as narrow_dyn_start::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("narrow_dyn_start", S);
        input
            .inner()
            .narrow(
                <Dim as Unsigned>::USIZE,
                self.start,
                <Len as Unsigned>::USIZE,
            )?
            .try_into()
    }
}
#[primitive(property = Visualize)]
impl<S, Dim, Len> Vis for NarrowDynStartOp<S, Dim, Len>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    Len: Unsigned,
    (S, Dim, Len): narrow_dyn_start::Compatible,
    <(S, Dim, Len) as narrow_dyn_start::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!(
                "NarrowDynStart(dim={}, len={})",
                <Dim as Unsigned>::USIZE,
                <Len as Unsigned>::USIZE
            ),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, Dim, Len) as narrow_dyn_start::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use glowstick::{
        assert_shape_eq,
        num::{U0, U1, U2, U3, U4},
        Shape3,
    };

    use crate::Tensor;

    #[test]
    fn narrow_dyn_start_op() {
        let device = paramecia_core::Device::Cpu;

        type S = Shape3<U2, U3, U4>;
        type A = Tensor<S>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = NarrowDynStartOp<S, U0, U1>;
        let res = MyOp::forward(
            &mut NarrowDynStartOp {
                _phantom: PhantomData,
                start: 1,
            },
            &mut (),
            a,
        )
        .unwrap();
        assert_shape_eq!(res, Shape3<U1, U3, U4>);
    }
}
