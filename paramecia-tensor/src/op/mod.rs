pub mod argmax_dim;
pub mod argmin_dim;
pub mod broadcast_add;
pub mod broadcast_mul;
pub mod cast_like;
pub mod cat;
pub mod clamp;
pub mod contiguous;
pub mod conv;
pub mod cumsum;
pub mod dims2;
pub mod dims3;
pub mod embedding;
pub mod exp;
pub mod expand;
pub mod flatten;
pub mod flatten_prefix2;
pub mod from_vec_on_device;
pub mod gather;
pub mod group_topk_assignments;
pub mod index_add_dim0;
pub mod index_select_dim0;
pub mod into_inner;
pub mod log_softmax;
pub mod matmul;
pub mod max_dim;
pub mod mean_dim;
pub mod min_dim;
pub mod narrow;
pub mod narrow_dyn;
pub mod narrow_dyn_start;
pub mod qmatmul_from_qtensor;
pub mod qmatmul_op;
pub mod remap_indices;
pub mod reshape;
pub mod residual_add;
pub mod rms_norm;
pub mod sigmoid;
pub mod silu;
pub mod softmax;
pub mod squeeze;
pub mod sum_dim;
pub mod tensor_device_info;
pub mod to_device;
pub mod to_dtype;
pub mod to_vec;
pub mod topk_from_logits;
pub(crate) mod trace;
pub mod transpose;
pub mod try_typed;
pub mod unflatten_last;
pub mod unsqueeze;
pub mod var_dim;

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U0, U1, U2},
        Shape2,
    };
    use inception::Inception;
    use paramecia_arrow::{Arrow, Combinator, LiftResult};

    use super::*;
    use crate::{matmul::MatmulOp, narrow::NarrowOp, Error, Tensor};

    #[test]
    fn pipelined_ops() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U1>>;
        type B = Tensor<Shape2<U1, U2>>;
        let a = A::from_vec(vec![4f32, 5.], &device).unwrap();
        let b = B::from_vec(vec![5f32, 4.], &device).unwrap();

        type Op1 = matmul::MatmulOp<Shape2<U2, U1>, Shape2<U1, U2>>;
        type Op2Inner = narrow::NarrowOp<Shape2<U2, U2>, U0, U0, U1>;
        // Op1 returns Result<...>, so Op2 is lifted.
        type Op2 =
            LiftResult<Op2Inner, Result<Tensor<Shape2<U2, U2>>, Error>, Tensor<Shape2<U1, U2>>>;
        type Op3Inner = narrow::NarrowOp<Shape2<U1, U2>, U1, U1, U1>;
        // Op2 returns Result<...>, so Op3 is lifted.
        type Op3 =
            LiftResult<Op3Inner, Result<Tensor<Shape2<U1, U2>>, Error>, Tensor<Shape2<U1, U1>>>;

        #[derive(Inception)]
        #[inception(properties = [Arrow])]
        struct SubPipeline(Op2, Op3);

        #[derive(Inception)]
        #[inception(properties = [Arrow])]
        struct MyPipeline(Op1, SubPipeline);

        let mut pipeline = MyPipeline(
            MatmulOp::default(),
            SubPipeline(
                LiftResult::new(NarrowOp::default()),
                LiftResult::new(NarrowOp::default()),
            ),
        );
        let res = pipeline.forward(&mut (), (a, b)).unwrap();

        assert_shape_eq!(res, Shape2<U1, U1>);
    }
}
