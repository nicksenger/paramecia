use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-changed=src/deltanet.cu");
    println!("cargo::rerun-if-changed=src/quantized.cu");
    println!("cargo::rerun-if-changed=src/flash_attn_q8.cu");
    println!("cargo::rerun-if-changed=src/quzo_perturb.cu");
    println!("cargo::rerun-if-changed=src/quzo_fused.cu");
    println!("cargo::rerun-if-changed=src/mma/mma_utils.cuh");
    println!("cargo::rerun-if-changed=src/mma/mmq_mma.cu");
    println!("cargo::rerun-if-changed=src/mma/flash_attn_q8_mma.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");

    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .kernel_paths_glob("src/*.cu");

    println!("cargo::warning={{builder:?}}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
        }
    }

    let moe_files = vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ];

    let mut obj_files = Vec::new();

    for file in moe_files {
        let path = PathBuf::from(file);
        let file_stem = path.file_stem().unwrap().to_str().unwrap();
        let obj_file = out_dir.join(format!("{}.o", file_stem));
        obj_files.push(obj_file.clone());

        let mut cmd = Command::new("nvcc");
        cmd.arg("-c")
            .arg(file)
            .arg("-o")
            .arg(&obj_file)
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-Isrc")
            .arg("-Isrc/moe");

        if let Ok(cap) = env::var("CUDA_COMPUTE_CAP") {
            cmd.arg(format!("--gpu-architecture=sm_{}", cap));
        } else {
            cmd.arg("--gpu-architecture=sm_86");
        }

        if is_target_msvc {
            cmd.arg("-D_USE_MATH_DEFINES");
        } else {
            cmd.arg("-Xcompiler").arg("-fPIC");
        }

        let status = cmd.status().expect("Failed to run nvcc");
        if !status.success() {
            panic!("nvcc failed for {}", file);
        }
    }

    if is_target_msvc {
        let lib_path = out_dir.join("moe.lib");
        let mut lib = Command::new("lib");
        lib.arg(format!("/OUT:{}", lib_path.display()));
        for obj in obj_files {
            lib.arg(obj);
        }
        let status = lib.status().expect("Failed to run lib");
        if !status.success() {
            panic!("lib failed");
        }
    } else {
        let lib_path = out_dir.join("libmoe.a");
        let mut ar = Command::new("ar");
        ar.arg("crus").arg(&lib_path);
        for obj in obj_files {
            ar.arg(obj);
        }
        let status = ar.status().expect("Failed to run ar");
        if !status.success() {
            panic!("ar failed");
        }
    }

    // Build MMA tensor core kernels (sm_75+ required for MMA PTX instructions)
    let mma_files = vec!["src/mma/mmq_mma.cu", "src/mma/flash_attn_q8_mma.cu"];

    let mut mma_obj_files = Vec::new();

    // MMA kernels require at least sm_80 (Ampere) for m16n8k16 mma.sync PTX instructions
    let mma_arch = if let Ok(cap) = env::var("CUDA_COMPUTE_CAP") {
        let cap_num: u32 = cap.parse().unwrap_or(80);
        if cap_num < 80 {
            80
        } else {
            cap_num
        }
    } else {
        80
    };

    for file in mma_files {
        let path = PathBuf::from(file);
        let file_stem = path.file_stem().unwrap().to_str().unwrap();
        let obj_file = out_dir.join(format!("{}.o", file_stem));
        mma_obj_files.push(obj_file.clone());

        let mut cmd = Command::new("nvcc");
        cmd.arg("-c")
            .arg(file)
            .arg("-o")
            .arg(&obj_file)
            .arg("--expt-relaxed-constexpr")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-Isrc")
            .arg("-Isrc/mma")
            .arg(format!("--gpu-architecture=sm_{}", mma_arch));

        if is_target_msvc {
            cmd.arg("-D_USE_MATH_DEFINES");
        } else {
            cmd.arg("-Xcompiler").arg("-fPIC");
        }

        let status = cmd.status().expect("Failed to run nvcc for MMA kernel");
        if !status.success() {
            panic!("nvcc failed for MMA kernel {}", file);
        }
    }

    if is_target_msvc {
        let lib_path = out_dir.join("mma.lib");
        let mut lib = Command::new("lib");
        lib.arg(format!("/OUT:{}", lib_path.display()));
        for obj in mma_obj_files {
            lib.arg(obj);
        }
        let status = lib.status().expect("Failed to run lib for MMA");
        if !status.success() {
            panic!("lib failed for MMA");
        }
    } else {
        let lib_path = out_dir.join("libmma.a");
        let mut ar = Command::new("ar");
        ar.arg("crus").arg(&lib_path);
        for obj in mma_obj_files {
            ar.arg(obj);
        }
        let status = ar.status().expect("Failed to run ar for MMA");
        if !status.success() {
            panic!("ar failed for MMA");
        }
    }

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=mma");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}
