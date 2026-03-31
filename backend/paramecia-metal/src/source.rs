pub const AFFINE: &str = include_str!("metal_src/affine.metal");
pub const BINARY: &str = include_str!("metal_src/binary.metal");
pub const CAST: &str = include_str!("metal_src/cast.metal");
pub const CONV: &str = include_str!("metal_src/conv.metal");
pub const DELTANET: &str = include_str!("metal_src/deltanet.metal");
pub const FILL: &str = include_str!("metal_src/fill.metal");
pub const INDEXING: &str = include_str!("metal_src/indexing.metal");
pub const MLX_GEMM: &str = include_str!("metal_src/mlx_gemm.metal");
pub const MLX_SORT: &str = include_str!("metal_src/mlx_sort.metal");
pub const QUANTIZED: &str = include_str!("metal_src/quantized.metal");
pub const RANDOM: &str = include_str!("metal_src/random.metal");
pub const REDUCE: &str = include_str!("metal_src/reduce.metal");
pub const SORT: &str = include_str!("metal_src/sort.metal");
pub const TERNARY: &str = include_str!("metal_src/ternary.metal");
pub const UNARY: &str = include_str!("metal_src/unary.metal");
pub const SDPA: &str = include_str!("metal_src/scaled_dot_product_attention.metal");
pub const QUZO_PERTURB: &str = include_str!("metal_src/quzo_perturb.metal");
pub const FLASH_ATTN_Q8: &str = include_str!("metal_src/flash_attn_q8.metal");
pub const QUZO_FUSED: &str = include_str!("metal_src/quzo_fused.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Binary,
    Cast,
    Conv,
    Deltanet,
    Fill,
    Gemm,
    Indexing,
    MlxSort,
    Quantized,
    Random,
    Reduce,
    Sort,
    Ternary,
    Unary,
    Sdpa,
    QuzoPerturb,
    FlashAttnQ8,
    QuzoFused,
}
