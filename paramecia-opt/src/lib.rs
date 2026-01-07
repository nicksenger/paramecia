pub mod batched_eval;
pub mod eggroll;
pub mod qzo;

pub use batched_eval::{
    compute_batched_fitness,
    compute_single_perturbation_fitness,
    compute_single_perturbation_fitness_with_moe_loss,
    evaluate_examples_chunked,
    // Strategy 2: Multi-perturbation batching (experimental)
    BatchedEvalConfig,
    BatchedPerturbationModel,
    BatchedPerturbations,
    // Strategy 1: Single-perturbation with chunked examples (recommended, matches paper)
    ChunkedEvalConfig,
    ChunkedEvalResult,
    ChunkedForward,
    PreparedPerturbation,
    SinglePerturbationConfig,
    StackedLoRA,
    StackedScales,
};
pub use eggroll::{
    Eggroll, EggrollParams, LayerConfig, LayerPerturbations, LowRankPerturbation, PopulationMember,
};
pub use qzo::moe::{ExpertMetrics, LoadBalanceLoss, RouterStats, ZLoss};
pub use qzo::{LossOutput, ParamsQZO, QZO};
