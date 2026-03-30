use core::ffi::c_void;
#[allow(dead_code)]
extern "C" {
    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    // MMA tensor core matmul kernels (sm_80+ / Ampere+)
    pub fn mul_mat_q8_0_mma_launch(
        vx: *const c_void, // quantized weights [nrows_x, ncols_x]
        vy: *const c_void, // Q8_1 activations [ncols_y, nrows_y]
        dst: *mut f32,     // output [nrows_x, ncols_y]
        ncols_x: i32,
        nrows_x: i32,
        ncols_y: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: i64,
    );

    pub fn mul_mat_q4_K_mma_launch(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        ncols_y: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: i64,
    );

    pub fn mul_mat_q6_K_mma_launch(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        ncols_y: i32,
        nrows_y: i32,
        nrows_dst: i32,
        stream: i64,
    );

    // MMA flash attention for Q8_0 KV cache (sm_80+, prefill only)
    pub fn flash_attn_q8_mma_launch(
        q: *const c_void, // Q [b, seq_q, h, d] f16
        k: *const c_void, // K quantized Q8_0
        v: *const c_void, // V quantized Q8_0
        out: *mut c_void, // output [b, seq_q, h, d] f16
        scale: f32,
        b: i32,
        h: i32,
        h_k: i32,
        d: i32,
        seq_q: i32,
        seq_k: i32,
        q_offset: i32,
        causal: i32,
        q_stride_b: u64,
        q_stride_seq: u64,
        q_stride_h: u64,
        q_stride_d: u64,
        k_stride_b: u64,
        k_stride_seq: u64,
        k_stride_h: u64,
        k_stride_d: u64,
        v_stride_b: u64,
        v_stride_seq: u64,
        v_stride_h: u64,
        v_stride_d: u64,
        o_stride_b: u64,
        o_stride_seq: u64,
        o_stride_h: u64,
        o_stride_d: u64,
        stream: i64,
    );
}
