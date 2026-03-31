use glowstick::Shape;
use inception::primitive;
use paramecia_arrow::{ArrowGraph, ArrowNode, Ident, Identified};
use typosaurus::collections::sp::Node;
use typosaurus::num::consts::*;

macro_rules! impl_tensor_graph_primitive {
    ($id:ty, [$($gen:ident),*], $ty:ty $(where $($where:tt)*)?) => {
        #[primitive(property = Ident)]
        impl<$($gen),*> Identified for $ty $(where $($where)*)? {
            type Id = $id;
        }

        #[primitive(property = ArrowGraph)]
        impl<$($gen),*> ArrowNode for $ty $(where $($where)*)? {
            type Graph = Node<<Self as Identified>::Id, $ty>;
        }
    };
}

impl_tensor_graph_primitive!(U1, [S, Dim], super::argmax_dim::ArgMaxDimOp<S, Dim>);
impl_tensor_graph_primitive!(U2, [S, Dim], super::argmin_dim::ArgMinDimOp<S, Dim>);
impl_tensor_graph_primitive!(U3, [S1, S2], super::broadcast_add::BroadcastAddOp<S1, S2>);
impl_tensor_graph_primitive!(U4, [S1, S2], super::broadcast_mul::BroadcastMulOp<S1, S2>);
impl_tensor_graph_primitive!(U5, [SLeft, SRight], super::cast_like::CastLikeOp<SLeft, SRight>);
impl_tensor_graph_primitive!(U6, [S, Dim, N], super::cat::CatOp<S, Dim, N>);
impl_tensor_graph_primitive!(U7, [S], super::clamp::ClampOp<S>);
impl_tensor_graph_primitive!(U8, [S], super::contiguous::ContiguousOp<S>);
impl_tensor_graph_primitive!(U9, [T, K, P1, P2, S, D], super::conv::Conv2dOp<T, K, P1, P2, S, D>);
impl_tensor_graph_primitive!(U10, [S, Dim], super::cumsum::CumSumOp<S, Dim>);
impl_tensor_graph_primitive!(U11, [S], super::dims2::Dims2Op<S>);
impl_tensor_graph_primitive!(U12, [S], super::dims3::Dims3Op<S>);
impl_tensor_graph_primitive!(U13, [InS, OutS], super::embedding::EmbeddingOp<InS, OutS>);
impl_tensor_graph_primitive!(U14, [S], super::exp::ExpOp<S>);
impl_tensor_graph_primitive!(U15, [S1, S2], super::expand::ExpandOp<S1, S2>);
impl_tensor_graph_primitive!(U16, [S, Dim1, Dim2], super::flatten::FlattenOp<S, Dim1, Dim2>);
impl_tensor_graph_primitive!(U17, [SIn, SOut], super::flatten_prefix2::FlattenPrefix2Op<SIn, SOut>);
impl_tensor_graph_primitive!(U18, [S, T], super::from_vec_on_device::FromVec1OnDeviceOp<S, T>);
impl_tensor_graph_primitive!(U19, [S, T], super::from_vec_on_device::FromVecColOnDeviceOp<S, T>);
impl_tensor_graph_primitive!(U20, [S1, S2, Dim], super::gather::GatherOp<S1, S2, Dim>);
impl_tensor_graph_primitive!(
    U21,
    [S],
    super::group_topk_assignments::GroupTopKAssignmentsOp<S>
);
impl_tensor_graph_primitive!(U22, [SBase, SIdx, SSrc], super::index_add_dim0::IndexAddDim0Op<SBase, SIdx, SSrc>);
impl_tensor_graph_primitive!(U23, [SIn, SIdx, SOut], super::index_select_dim0::IndexSelectDim0Op<SIn, SIdx, SOut>);
impl_tensor_graph_primitive!(U24, [S, E], super::into_inner::IntoInnerResultOp<S, E>);
impl_tensor_graph_primitive!(U25, [S], super::into_inner::IntoInnerOp<S>);
impl_tensor_graph_primitive!(U26, [S, Dim], super::log_softmax::LogSoftmaxOp<S, Dim>);
impl_tensor_graph_primitive!(U27, [S1, S2], super::matmul::MatmulOp<S1, S2>);
impl_tensor_graph_primitive!(U28, [S, Dim], super::max_dim::MaxDimOp<S, Dim>);
impl_tensor_graph_primitive!(U29, [S, Dim], super::mean_dim::MeanDimOp<S, Dim>);
impl_tensor_graph_primitive!(U30, [S, Dim], super::min_dim::MinDimOp<S, Dim>);
impl_tensor_graph_primitive!(U31, [S, Dim, Start, Len], super::narrow::NarrowOp<S, Dim, Start, Len>);
impl_tensor_graph_primitive!(U32, [S, Dim, DynDim], super::narrow_dyn::NarrowDynOp<S, Dim, DynDim>);
impl_tensor_graph_primitive!(U33, [S, Dim, Len], super::narrow_dyn_start::NarrowDynStartOp<S, Dim, Len>);
impl_tensor_graph_primitive!(
    U34,
    [S],
    super::qmatmul_from_qtensor::QMatMulFromQTensorOp<S>
);
impl_tensor_graph_primitive!(U35, [WS, InS, OutS], super::qmatmul_op::QMatMulOp<WS, InS, OutS> where WS: Shape);
impl_tensor_graph_primitive!(U36, [SIndices, SMap], super::remap_indices::RemapIndicesOp<SIndices, SMap> where SMap: Shape);
impl_tensor_graph_primitive!(U37, [S, S1, S2], super::reshape::ReshapeOp<S, S1, S2>);
impl_tensor_graph_primitive!(U38, [S], super::residual_add::ResidualAddOp<S>);
impl_tensor_graph_primitive!(U39, [S], super::rms_norm::RmsNormOp<S>);
impl_tensor_graph_primitive!(U40, [S], super::sigmoid::SigmoidOp<S>);
impl_tensor_graph_primitive!(U41, [S], super::silu::SiluOp<S>);
impl_tensor_graph_primitive!(U42, [S, Dim], super::softmax::SoftmaxOp<S, Dim>);
impl_tensor_graph_primitive!(U43, [S, Dim], super::squeeze::SqueezeOp<S, Dim>);
impl_tensor_graph_primitive!(U44, [S, Dim], super::sum_dim::SumDimOp<S, Dim>);
impl_tensor_graph_primitive!(U45, [S], super::tensor_device_info::TensorDeviceInfoOp<S>);
impl_tensor_graph_primitive!(U46, [S], super::to_device::ToDeviceOp<S>);
impl_tensor_graph_primitive!(U47, [S], super::to_dtype::ToDtypeOp<S>);
impl_tensor_graph_primitive!(U48, [S, T], super::to_vec::ToVec1Op<S, T>);
impl_tensor_graph_primitive!(U49, [S, T], super::to_vec::FlattenAllToVec1Op<S, T>);
impl_tensor_graph_primitive!(U50, [S, T], super::to_vec::ToVec2Op<S, T>);
impl_tensor_graph_primitive!(U51, [S, T], super::to_vec::CastFlattenToVec1Op<S, T>);
impl_tensor_graph_primitive!(U52, [S, T], super::to_vec::CastToVec2Op<S, T>);
impl_tensor_graph_primitive!(U53, [SA, SB, A, B], super::to_vec::CastFlattenToVec1PairOp<SA, SB, A, B>);
impl_tensor_graph_primitive!(U54, [SA, SB, A, B], super::to_vec::CastToVec2PairOp<SA, SB, A, B>);
impl_tensor_graph_primitive!(U55, [SIn, SWeights, SIndices], super::topk_from_logits::TopkFromLogitsOp<SIn, SWeights, SIndices>);
impl_tensor_graph_primitive!(U56, [S, Dim1, Dim2], super::transpose::TransposeOp<S, Dim1, Dim2>);
impl_tensor_graph_primitive!(U57, [S], super::try_typed::TryTypedOp<S>);
impl_tensor_graph_primitive!(U58, [SIn, SOut], super::unflatten_last::UnflattenLastOp<SIn, SOut>);
impl_tensor_graph_primitive!(U59, [S, Dim], super::unsqueeze::UnsqueezeOp<S, Dim>);
impl_tensor_graph_primitive!(U60, [S, Dim], super::var_dim::VarDimOp<S, Dim>);
