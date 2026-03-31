use std::fs;
use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let glsl_dir = Path::new(&manifest_dir).join("src/glsl");

    println!("cargo:rerun-if-changed=src/glsl");

    let compiler = shaderc::Compiler::new().expect("Failed to create shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("Failed to create compile options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    options.set_target_spirv(shaderc::SpirvVersion::V1_6);
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    // Set up include callback for common.comp
    let glsl_dir_clone = glsl_dir.clone();
    options.set_include_callback(move |requested, _include_type, _source_path, _depth| {
        let include_path = glsl_dir_clone.join(requested);
        let content = fs::read_to_string(&include_path)
            .map_err(|e| format!("Failed to read {}: {}", include_path.display(), e))?;
        Ok(shaderc::ResolvedInclude {
            resolved_name: include_path.to_string_lossy().into_owned(),
            content,
        })
    });

    // Shaders that need HEAD_DIM variants (d64, d128, d256)
    let head_dim_shaders = [
        "flash_attn_q8",
        "flash_attn_q8_split",
        "gla_step",
        "gla_step_coopmat",
        "delta_net_parallel",
        "delta_net_parallel_coopmat",
    ];

    // Shaders that are compiled once (no variants)
    let single_shaders = [
        "matmul_q8_0",
        "matmul_q8_0_dp4a",
        "matmul_q8_0_coopmat",
        "matmul_q4_k",
        "matmul_q4_k_coopmat",
        "matmul_q5_k",
        "matmul_q5_k_coopmat",
        "matmul_q6_k",
        "matmul_q6_k_coopmat",
        "matmul_q3_k",
        "matmul_q3_k_coopmat",
        "matmul_q2_k",
        "matmul_q2_k_coopmat",
        // Hierarchical tiled matmul kernels (Phase 1-2: Q8_0 and K-quants)
        "matmul_tiled_q8_0",
        "matmul_tiled_q8_0_prefill", // Phase 3: Prefill-optimized (BM=64)
        "matmul_tiled_q8_0_prefill_opt", // Phase 4: K-loop unroll
        "matmul_tiled_q8_0_prefill_vec", // Phase 4: Vectorized loading
        "matmul_tiled_q8_0_prefill_db", // Phase 4: Double buffering
        "matmul_tiled_q8_0_prefill_coopmat", // Phase 6: Hierarchical tiled coopmat
        "matmul_tiled_q2_k",
        "matmul_tiled_q2_k_prefill",         // Phase 3: Prefill-optimized
        "matmul_tiled_q2_k_prefill_coopmat", // Phase 6: Hierarchical coopmat
        "matmul_tiled_q3_k",
        "matmul_tiled_q3_k_prefill",         // Phase 3: Prefill-optimized
        "matmul_tiled_q3_k_prefill_coopmat", // Phase 6: Hierarchical coopmat
        "matmul_tiled_q4_k",
        "matmul_tiled_q4_k_prefill",         // Phase 3: Prefill-optimized
        "matmul_tiled_q4_k_prefill_coopmat", // Phase 6: Hierarchical coopmat
        "matmul_tiled_q5_k",
        "matmul_tiled_q5_k_prefill",         // Phase 3: Prefill-optimized
        "matmul_tiled_q5_k_prefill_coopmat", // Phase 6: Hierarchical coopmat
        "matmul_tiled_q6_k",
        "matmul_tiled_q6_k_prefill",         // Phase 3: Prefill-optimized
        "matmul_tiled_q6_k_prefill_coopmat", // Phase 6: Hierarchical coopmat
        // Matrix-Vector variants (optimized for decode phase m=1)
        "matmul_vec_q8_0",
        "matmul_vec_q4_k",
        "matmul_vec_q5_k",
        "matmul_vec_q6_k",
        "matmul_vec_q3_k",
        "matmul_vec_q2_k",
        "depthwise_conv1d",
        "l2_normalize_scale",
        "fused_swiglu",
        "gated_rms_norm",
        "quantize_q8_0",
        "dequantize_q8_0",
        "indexed_moe_forward_q8_0",
        "indexed_moe_forward_q8_0_coopmat",
        "indexed_moe_forward_q4_k",
        "indexed_moe_forward_q4_k_coopmat",
        "indexed_moe_forward_q5_k",
        "indexed_moe_forward_q5_k_coopmat",
        "indexed_moe_forward_q6_k",
        "indexed_moe_forward_q6_k_coopmat",
        "indexed_moe_forward_q3_k",
        "indexed_moe_forward_q3_k_coopmat",
        "indexed_moe_forward_q2_k",
        "indexed_moe_forward_q2_k_coopmat",
        "indexed_moe_gate_up_swiglu_q8_0",
        "indexed_moe_gate_up_swiglu_q8_0_coopmat",
        "indexed_moe_gate_up_swiglu_q4_k",
        "indexed_moe_gate_up_swiglu_q4_k_coopmat",
        "indexed_moe_gate_up_swiglu_q5_k",
        "indexed_moe_gate_up_swiglu_q5_k_coopmat",
        "indexed_moe_gate_up_swiglu_q6_k",
        "indexed_moe_gate_up_swiglu_q6_k_coopmat",
        "indexed_moe_gate_up_swiglu_q3_k",
        "indexed_moe_gate_up_swiglu_q3_k_coopmat",
        "indexed_moe_gate_up_swiglu_q2_k",
        "indexed_moe_gate_up_swiglu_q2_k_coopmat",
        "fused_matmul_q8_0",
        "fused_matmul_q4_k",
        "fused_matmul_q2_k",
        "fused_matmul_q3_k",
        "fused_matmul_q5_k",
        "fused_matmul_q6_k",
        "fused_matmul_bf16",
        "flash_attn_q8_split_k_reduce",
        "solve_triangular",
    ];

    // Compile HEAD_DIM variant shaders
    for shader_name in &head_dim_shaders {
        let source_path = glsl_dir.join(format!("{}.comp", shader_name));
        if !source_path.exists() {
            println!("cargo:warning=Shader not found: {}", source_path.display());
            continue;
        }
        println!("cargo:rerun-if-changed={}", source_path.display());
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|_| panic!("Failed to read {}.comp", shader_name));

        for head_dim in [64, 128, 256] {
            let mut opts = options.clone();
            opts.add_macro_definition("HEAD_DIM", Some(&head_dim.to_string()));

            let spv_name = format!("{}_d{}", shader_name, head_dim);
            match compiler.compile_into_spirv(
                &source,
                shaderc::ShaderKind::Compute,
                &format!("{}_d{}.comp", shader_name, head_dim),
                "main",
                Some(&opts),
            ) {
                Ok(artifact) => {
                    let spv_path = Path::new(&out_dir).join(format!("{}.spv", spv_name));
                    fs::write(&spv_path, artifact.as_binary_u8())
                        .expect("Failed to write SPIR-V file");
                }
                Err(e) => {
                    panic!("Failed to compile {}: {}", spv_name, e);
                }
            }
        }
    }

    // QuZO perturbation shaders: each compiled twice (perturb + restore_update)
    let quzo_formats: &[(&str, &str)] = &[
        ("quzo_q8_0", "quzo_q8_0"),
        ("quzo_q4_k", "quzo_q4_k"),
        ("quzo_q2_k", "quzo_q2_k"),
        ("quzo_q3_k", "quzo_q3_k"),
        ("quzo_q5_k", "quzo_q5_k"),
        ("quzo_q6_k", "quzo_q6_k"),
        ("quzo_bf16", "quzo_bf16"),
    ];

    for &(base_name, source_name) in quzo_formats {
        let source_path = glsl_dir.join(format!("{}.comp", source_name));
        if !source_path.exists() {
            println!("cargo:warning=Shader not found: {}", source_path.display());
            continue;
        }
        println!("cargo:rerun-if-changed={}", source_path.display());
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|_| panic!("Failed to read {}.comp", source_name));

        // Compile perturb variant
        for (mode_define, spv_prefix) in [
            ("MODE_PERTURB", "perturb"),
            ("MODE_RESTORE_UPDATE", "restore_update"),
        ] {
            let mut opts = options.clone();
            opts.add_macro_definition(mode_define, Some("1"));

            let format_suffix = base_name.strip_prefix("quzo_").unwrap();
            let spv_name = format!("{}_{}", spv_prefix, format_suffix);
            match compiler.compile_into_spirv(
                &source,
                shaderc::ShaderKind::Compute,
                &format!("{}_{}.comp", source_name, spv_prefix),
                "main",
                Some(&opts),
            ) {
                Ok(artifact) => {
                    let spv_path = Path::new(&out_dir).join(format!("{}.spv", spv_name));
                    fs::write(&spv_path, artifact.as_binary_u8())
                        .expect("Failed to write SPIR-V file");
                }
                Err(e) => {
                    panic!("Failed to compile {}: {}", spv_name, e);
                }
            }
        }
    }

    // Compile single-variant shaders
    for shader_name in &single_shaders {
        let source_path = glsl_dir.join(format!("{}.comp", shader_name));
        if !source_path.exists() {
            println!("cargo:warning=Shader not found: {}", source_path.display());
            continue;
        }
        println!("cargo:rerun-if-changed={}", source_path.display());
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|_| panic!("Failed to read {}.comp", shader_name));

        match compiler.compile_into_spirv(
            &source,
            shaderc::ShaderKind::Compute,
            &format!("{}.comp", shader_name),
            "main",
            Some(&options),
        ) {
            Ok(artifact) => {
                let spv_path = Path::new(&out_dir).join(format!("{}.spv", shader_name));
                fs::write(&spv_path, artifact.as_binary_u8()).expect("Failed to write SPIR-V file");
            }
            Err(e) => {
                panic!("Failed to compile {}: {}", shader_name, e);
            }
        }
    }
}
