use std::collections::HashMap;
use std::sync::{PoisonError, RwLock, TryLockError};

pub mod spirv_shaders;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    I64,
    U32,
    U8,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanKernelError {
    #[error("{0}")]
    Message(String),
    #[error("{0:?}")]
    LockError(#[from] LockError),
    #[error("{0:?}")]
    IoError(#[from] std::io::Error),
    #[error("{0:?}")]
    ShadercError(#[from] shaderc::Error),
    #[error("Failed to create shader compiler")]
    FailedToCreateCompiler,
    #[error("Failed to create compute pipeline: {0}")]
    FailedToCreatePipeline(String),
    #[error("Vulkan API error: {0}")]
    VkError(#[from] ash::vk::Result),
}

impl From<String> for VulkanKernelError {
    fn from(e: String) -> Self {
        VulkanKernelError::Message(e)
    }
}

impl From<&str> for VulkanKernelError {
    fn from(e: &str) -> Self {
        VulkanKernelError::Message(e.to_string())
    }
}

impl<T> From<TryLockError<T>> for VulkanKernelError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => {
                VulkanKernelError::LockError(LockError::Poisoned(p.to_string()))
            }
            TryLockError::WouldBlock => VulkanKernelError::LockError(LockError::WouldBlock),
        }
    }
}

impl<T> From<PoisonError<T>> for VulkanKernelError {
    fn from(p: PoisonError<T>) -> Self {
        VulkanKernelError::LockError(LockError::Poisoned(p.to_string()))
    }
}

/// Configuration for a shader kernel: source path + macro definitions.
#[derive(Debug)]
pub struct KernelConfig {
    pub path: String,
    pub defines: Vec<(&'static str, String)>,
}

impl KernelConfig {
    /// Compile GLSL to SPIR-V bytes using shaderc at runtime.
    pub fn compile(
        &self,
        additional: Option<&[(&str, &str)]>,
    ) -> Result<Vec<u32>, VulkanKernelError> {
        let full_path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), self.path);
        let shader_source = std::fs::read_to_string(&full_path)?;
        let compiler = shaderc::Compiler::new()?;
        let mut options = shaderc::CompileOptions::new()?;

        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as u32,
        );
        options.set_target_spirv(shaderc::SpirvVersion::V1_6);
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        // Include callback for common.comp
        options.set_include_callback(|requested, _include_type, source_path, _depth| {
            let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
            let full_source_path = if std::path::Path::new(source_path).is_absolute() {
                std::path::Path::new(source_path).to_path_buf()
            } else {
                manifest_dir.join(source_path)
            };
            let base_dir = full_source_path.parent().unwrap_or(manifest_dir);
            let include_path = base_dir.join(requested);
            let canonical = std::fs::canonicalize(&include_path)
                .map_err(|e| format!("Failed to canonicalize {}: {}", include_path.display(), e))?;
            let content = std::fs::read_to_string(&canonical)
                .map_err(|e| format!("Failed to read {}: {}", canonical.display(), e))?;
            Ok(shaderc::ResolvedInclude {
                resolved_name: canonical.to_string_lossy().into_owned(),
                content,
            })
        });

        for (key, val) in &self.defines {
            options.add_macro_definition(key, Some(val));
        }
        if let Some(add_defs) = additional {
            for (key, val) in add_defs {
                options.add_macro_definition(key, Some(val));
            }
        }

        let compiled = compiler.compile_into_spirv(
            &shader_source,
            shaderc::ShaderKind::Compute,
            &self.path,
            "main",
            Some(&options),
        )?;

        Ok(compiled.as_binary().to_vec())
    }

    /// Compile and create a vk::ShaderModule on the given device.
    pub fn compile_to_module(
        &self,
        device: &ash::Device,
        additional: Option<&[(&str, &str)]>,
    ) -> Result<ash::vk::ShaderModule, VulkanKernelError> {
        let spirv = self.compile(additional)?;
        let create_info = ash::vk::ShaderModuleCreateInfo::default().code(&spirv);
        let module = unsafe { device.create_shader_module(&create_info, None) }?;
        Ok(module)
    }
}

macro_rules! register_kernel {
    ($name:expr, $path:expr, $(($key:expr, $val:expr)),* $(,)?) => {
        ($name.to_string(), KernelConfig {
            path: $path.to_owned(),
            defines: vec![$(($key, $val.to_string())),*],
        })
    };
}

/// Manages kernel configs, compiled shader modules, and compute pipelines.
/// Pipelines are created lazily and cached.
pub struct Kernels {
    configs: HashMap<String, KernelConfig>,
    /// Cache: kernel_name -> SPIR-V words
    compiled: RwLock<HashMap<String, Vec<u32>>>,
    /// Cache: kernel_name -> (vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout)
    pipelines: RwLock<HashMap<String, CachedPipeline>>,
    /// If true, create descriptor set layouts with PUSH_DESCRIPTOR_BIT_KHR
    /// and use vkCmdPushDescriptorSetKHR instead of allocating descriptor sets.
    use_push_descriptors: std::sync::atomic::AtomicBool,
}

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernels")
            .field("num_configs", &self.configs.len())
            .finish()
    }
}

#[derive(Clone)]
pub struct CachedPipeline {
    pub pipeline: ash::vk::Pipeline,
    pub layout: ash::vk::PipelineLayout,
    pub desc_set_layout: ash::vk::DescriptorSetLayout,
    /// Whether this pipeline uses push descriptors (no descriptor set allocation needed).
    pub push_descriptors: bool,
}

impl Kernels {
    /// Enable push descriptor support for all subsequently loaded pipelines.
    /// Must be called before any pipeline is loaded.
    pub fn enable_push_descriptors(&self) {
        self.use_push_descriptors
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn new() -> Result<Self, VulkanKernelError> {
        let mut configs = HashMap::new();

        // --- CAST KERNELS (all dtype pairs) ---
        {
            // (name_suffix, glsl_type, is_bf16, need_uint_cast_when_src)
            let types: &[(&str, &str, &str, &str)] = &[
                ("f32", "float", "0", "0"),
                ("f16", "float16_t", "0", "0"),
                ("bf16", "uint16_t", "1", "0"),
                ("u32", "uint", "0", "0"),
                ("u8", "uint8_t", "0", "1"),
                ("i64", "int64_t", "0", "0"),
            ];
            for &(src_name, src_glsl, src_bf16, src_uint_cast) in types {
                for &(dst_name, dst_glsl, dst_bf16, _) in types {
                    if src_name == dst_name {
                        continue;
                    }
                    let name = format!("cast_{}_{}", src_name, dst_name);
                    let config = KernelConfig {
                        path: "src/glsl/cast.comp".to_owned(),
                        defines: vec![
                            ("SRC_TYPE", src_glsl.to_string()),
                            ("DST_TYPE", dst_glsl.to_string()),
                            ("NEED_UINT_CAST", src_uint_cast.to_string()),
                            ("SRC_BF16", src_bf16.to_string()),
                            ("DST_BF16", dst_bf16.to_string()),
                        ],
                    };
                    configs.insert(name, config);
                }
            }
        }

        // --- UNARY KERNELS ---
        for op in [
            "neg", "abs", "sign", "gelu", "gelu_erf", "erf", "relu", "silu", "ceil", "floor",
            "round", "sqr", "sqrt", "sin", "cos", "tan", "sigmoid", "exp", "log", "recip", "tanh",
        ] {
            for (dtype, glsl_type, is_bf16) in [
                ("f32", "float", "0"),
                ("f16", "float16_t", "0"),
                ("bf16", "uint16_t", "1"),
            ] {
                let (name, config) = register_kernel!(
                    format!("{}_{}", op, dtype),
                    "src/glsl/unary.comp",
                    ("OP", format!("{}_op", op)),
                    ("INNER_TYPE", "float"),
                    ("OUTER_TYPE", glsl_type),
                    ("BF16", is_bf16)
                );
                configs.insert(name, config);
            }
        }

        // --- BINARY KERNELS ---
        for op in ["add", "sub", "div", "mul", "minimum", "maximum"] {
            for (dtype, inner_type, outer_type, is_bf16) in [
                ("f32", "float", "float", "0"),
                ("f16", "float16_t", "float16_t", "0"),
                ("bf16", "float", "uint16_t", "1"),
                ("i64", "int64_t", "int64_t", "0"),
            ] {
                let (name, config) = register_kernel!(
                    format!("{}_{}", op, dtype),
                    "src/glsl/binary.comp",
                    ("OP", format!("{}_op", op)),
                    ("INNER_TYPE", inner_type),
                    ("OUTER_TYPE", outer_type),
                    ("BF16", is_bf16)
                );
                configs.insert(name, config);
            }
        }

        // --- REDUCE KERNELS ---
        for (op_name, op, to_index) in [
            ("sum", "0", "0"),
            ("argmax", "1", "1"),
            ("max", "1", "0"),
            ("argmin", "2", "1"),
            ("min", "2", "0"),
        ] {
            for (dtype, _inner_type, outer_type, _is_bf16) in [
                ("f32", "float", "float", "0"),
                ("u32", "uint", "uint", "0"),
                ("f16", "float16_t", "float16_t", "0"),
                ("i64", "int64_t", "int64_t", "0"),
            ] {
                let (name, config) = register_kernel!(
                    format!("{}_partial_{}", op_name, dtype),
                    "src/glsl/reduce_partial.comp",
                    ("OP", op),
                    ("TYPE", outer_type),
                    ("TO_INDEX", to_index)
                );
                configs.insert(name, config);
                let (name, config) = register_kernel!(
                    format!("{}_combine_{}", op_name, dtype),
                    "src/glsl/reduce_combine.comp",
                    ("OP", op),
                    ("TYPE", outer_type),
                    ("TO_INDEX", to_index)
                );
                configs.insert(name, config);
            }
        }

        // --- AFFINE / ELU / POWF KERNELS ---
        for op in ["affine", "elu", "powf"] {
            for (dtype, inner_type, outer_type, is_bf16) in [
                ("f32", "float", "float", "0"),
                ("f16", "float16_t", "float16_t", "0"),
                ("bf16", "float", "uint16_t", "1"),
                ("i64", "int64_t", "int64_t", "0"),
            ] {
                let (name, config) = register_kernel!(
                    format!("{}_{}", op, dtype),
                    "src/glsl/affine_elu.comp",
                    ("OP", format!("{}_op", op)),
                    ("INNER_TYPE", inner_type),
                    ("OUTER_TYPE", outer_type),
                    ("BF16", is_bf16)
                );
                configs.insert(name, config);
            }
        }

        // --- GATHER KERNELS ---
        for (idx_dtype, idx_type, max_idx) in [
            ("u8", "uint8_t", "0xFF"),
            ("u32", "uint", "0xFFFFFFFFU"),
            ("i64", "int64_t", "0x7FFFFFFFFFFFFFFF"),
        ] {
            for (dtype, glsl_type, is_bf16) in [
                ("u8", "uint8_t", "0"),
                ("u32", "uint", "0"),
                ("i64", "int64_t", "0"),
                ("bf16", "uint", "1"),
                ("f32", "float", "0"),
            ] {
                let (name, config) = register_kernel!(
                    format!("gather_{}_{}", idx_dtype, dtype),
                    "src/glsl/gather.comp",
                    ("IDX_TYPE", idx_type),
                    ("MAX_IDX", max_idx),
                    ("TYPE", glsl_type),
                    ("BF16", is_bf16)
                );
                configs.insert(name, config);
            }
        }

        // --- SCATTER_SET KERNELS ---
        for (idx_dtype, idx_type) in [("u8", "uint8_t"), ("u32", "uint"), ("i64", "int64_t")] {
            for (dtype, glsl_type, is_bf16) in [
                ("u32", "uint", "0"),
                ("bf16", "float", "1"),
                ("f32", "float", "0"),
            ] {
                let (name, config) = register_kernel!(
                    format!("scatter_set_{}_{}", idx_dtype, dtype),
                    "src/glsl/scatter.comp",
                    ("IDX_TYPE", idx_type),
                    ("TYPE", glsl_type),
                    ("BF16", is_bf16)
                );
                configs.insert(name, config);
            }
        }

        // --- SCATTER_ADD_SET KERNELS ---
        for (idx_dtype, idx_type) in [("u8", "uint8_t"), ("u32", "uint"), ("i64", "int64_t")] {
            for (dtype, glsl_type, is_bf16) in [
                ("u32", "uint", "0"),
                ("bf16", "float", "1"),
                ("f32", "float", "0"),
            ] {
                let (name, config) = register_kernel!(
                    format!("scatter_add_set_{}_{}", idx_dtype, dtype),
                    "src/glsl/scatter_add.comp",
                    ("IDX_TYPE", idx_type),
                    ("TYPE", glsl_type),
                    ("BF16", is_bf16)
                );
                configs.insert(name, config);
            }
        }

        // --- INDEX_{ADD,SELECT} KERNELS ---
        for shader in ["index_add", "index_select"] {
            let path = format!("src/glsl/{}.comp", shader);
            for (idx_dtype, idx_glsl_type) in
                [("u8", "uint8_t"), ("u32", "uint"), ("i64", "int64_t")]
            {
                for (dtype, glsl_type) in [
                    ("u8", "uint8_t"),
                    ("u32", "uint"),
                    ("i64", "int64_t"),
                    ("f32", "float"),
                    ("f16", "float16_t"),
                ] {
                    let (name, config) = register_kernel!(
                        format!("{}_{}_{}", shader, idx_dtype, dtype),
                        &path,
                        ("IDX_TYPE", idx_glsl_type),
                        ("TYPE", glsl_type),
                        ("BF16", "0")
                    );
                    configs.insert(name, config);
                }
                let (name, config) = register_kernel!(
                    format!("{}_{}_bf16", shader, idx_dtype),
                    &path,
                    ("IDX_TYPE", idx_glsl_type),
                    ("TYPE", "uint16_t"),
                    ("BF16", "1")
                );
                configs.insert(name, config);
            }
        }

        // --- CONST_SET KERNELS ---
        for (width, glsl_type) in [
            (8u64, "uint8_t"),
            (16, "uint16_t"),
            (32, "uint"),
            (0, "int64_t"),
        ] {
            let (name, config) = register_kernel!(
                format!("const_set_{}", width),
                "src/glsl/const_set.comp",
                ("MASK", ((1u64 << width) - 1).to_string()),
                ("TYPE", glsl_type)
            );
            configs.insert(name, config);
        }

        // --- COPY2D KERNELS ---
        for (dtype, glsl_type) in [
            ("u8", "uint8_t"),
            ("u32", "uint"),
            ("i64", "int64_t"),
            ("f32", "float"),
            ("f16", "float16_t"),
            ("bf16", "uint16_t"),
        ] {
            let (name, config) = register_kernel!(
                format!("copy2d_{}", dtype),
                "src/glsl/copy2d.comp",
                ("TYPE", glsl_type)
            );
            configs.insert(name, config);
        }

        // --- COPY_STRIDED_SRC KERNELS ---
        for (dtype, glsl_type) in [
            ("f16", "float16_t"),
            ("f32", "float"),
            ("u8", "uint8_t"),
            ("u32", "uint"),
            ("i64", "int64_t"),
            ("bf16", "uint16_t"),
        ] {
            let (name, config) = register_kernel!(
                format!("copy_strided_src_{}", dtype),
                "src/glsl/copy_strided_src.comp",
                ("TYPE", glsl_type)
            );
            configs.insert(name, config);
        }

        // --- CMP KERNELS ---
        for (dtype, glsl_type) in [
            ("u8", "uint8_t"),
            ("u32", "uint"),
            ("i64", "int64_t"),
            ("f32", "float"),
            ("f16", "float16_t"),
            ("bf16", "uint16_t"),
        ] {
            for (op_name, op) in [
                ("eq", "=="),
                ("ne", "!="),
                ("lt", "<"),
                ("gt", ">"),
                ("le", "<="),
                ("ge", ">="),
            ] {
                let (name, config) = register_kernel!(
                    format!("{}_{}", op_name, dtype),
                    "src/glsl/cmp.comp",
                    ("OP", op),
                    ("TYPE", glsl_type)
                );
                configs.insert(name, config);
            }
        }

        // --- RANDOM KERNELS ---
        {
            let (name, config) =
                register_kernel!("rand_uniform_f32", "src/glsl/rand.comp", ("UNIFORM", "1"));
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("rand_normal_f32", "src/glsl/rand.comp", ("UNIFORM", "0"));
            configs.insert(name, config);
        }

        // --- GEMM KERNELS ---
        for (dtype, glsl_type, is_bf16) in [
            ("f32", "float", "0"),
            ("f16", "float16_t", "0"),
            ("bf16", "uint16_t", "1"),
        ] {
            let (name, config) = register_kernel!(
                format!("gemm_{}", dtype),
                "src/glsl/gemm.comp",
                ("TYPE", glsl_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- GEMM SPLIT-K VARIANTS (f32 partials output for non-f32 types) ---
        for (dtype, glsl_type, is_bf16) in [("f16", "float16_t", "0"), ("bf16", "uint16_t", "1")] {
            let (name, config) = register_kernel!(
                format!("gemm_{}_splitk", dtype),
                "src/glsl/gemm.comp",
                ("TYPE", glsl_type),
                ("BF16", is_bf16),
                ("SPLIT_K_F32", "1")
            );
            configs.insert(name, config);
        }

        // --- GEMM F16ACC KERNELS (FP16 accumulation for bandwidth-bound cases) ---
        {
            let (name, config) = register_kernel!(
                "gemm_f16_f16acc",
                "src/glsl/gemm.comp",
                ("TYPE", "float16_t"),
                ("BF16", "0"),
                ("F16ACC", "1")
            );
            configs.insert(name, config);
        }

        // --- GEMM SPLIT-K REDUCE ---
        // f32 output (default)
        {
            let (name, config) = register_kernel!(
                "matmul_split_k_reduce",
                "src/glsl/matmul_split_k_reduce.comp",
            );
            configs.insert(name, config);
        }
        // Typed output reduce variants for f16/bf16
        for (dtype, glsl_type, is_bf16) in [("f16", "float16_t", "0"), ("bf16", "uint16_t", "1")] {
            let (name, config) = register_kernel!(
                format!("matmul_split_k_reduce_{}", dtype),
                "src/glsl/matmul_split_k_reduce.comp",
                ("TYPE", glsl_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- GEMM COOPMAT KERNELS ---
        for (dtype, glsl_type, is_bf16) in [
            ("f32", "float", "0"),
            ("f16", "float16_t", "0"),
            ("bf16", "uint16_t", "1"),
        ] {
            let (name, config) = register_kernel!(
                format!("gemm_{}_coopmat", dtype),
                "src/glsl/gemm_coopmat.comp",
                ("TYPE", glsl_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- SORT KERNELS ---
        for (dtype, glsl_type, is_bf16) in [
            ("f32", "float", "0"),
            ("u32", "uint", "0"),
            ("i64", "int64_t", "0"),
            ("bf16", "uint16_t", "1"),
            ("f16", "float16_t", "0"),
            ("u8", "uint8_t", "0"),
        ] {
            let (name, config) = register_kernel!(
                format!("arg_sort_{}", dtype),
                "src/glsl/sort.comp",
                ("TYPE", glsl_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- LAYERNORM KERNELS ---
        for (dtype, outer_type, is_bf16) in [
            ("f32", "float", "0"),
            ("f16", "float16_t", "0"),
            ("bf16", "uint16_t", "1"),
        ] {
            let (name, config) = register_kernel!(
                format!("layernorm_{}", dtype),
                "src/glsl/layernorm.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- RMSNORM KERNELS ---
        for (dtype, outer_type, is_bf16) in [
            ("f32", "float", "0"),
            ("f16", "float16_t", "0"),
            ("bf16", "uint16_t", "1"),
        ] {
            let (name, config) = register_kernel!(
                format!("rmsnorm_{}", dtype),
                "src/glsl/rmsnorm.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- SOFTMAX KERNELS ---
        for (dtype, outer_type, is_bf16) in [
            ("f32", "float", "0"),
            ("f16", "float16_t", "0"),
            ("bf16", "uint16_t", "1"),
        ] {
            let (name, config) = register_kernel!(
                format!("softmax_{}", dtype),
                "src/glsl/softmax.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- TOP-K SOFTMAX KERNELS ---
        for (dtype, act_dtype) in [("f32", "0"), ("f16", "1"), ("bf16", "2")] {
            let (name, config) = register_kernel!(
                format!("topk_softmax_{}", dtype),
                "src/glsl/topk_softmax.comp",
                ("ACT_DTYPE", act_dtype)
            );
            configs.insert(name, config);
        }

        // --- ROPE KERNELS ---
        for (dtype, outer_type, is_bf16) in [
            ("f32", "float", "0"),
            ("f16", "float16_t", "0"),
            ("bf16", "uint16_t", "1"),
        ] {
            let (name, config) = register_kernel!(
                format!("rope_{}", dtype),
                "src/glsl/rope.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16),
                ("THD", "0")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                format!("rope_thd_{}", dtype),
                "src/glsl/rope.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16),
                ("THD", "1")
            );
            configs.insert(name, config);
        }

        // --- ROPE_I KERNELS ---
        for (dtype, outer_type, is_bf16) in [
            ("f32", "float", "0"),
            ("f16", "float16_t", "0"),
            ("bf16", "uint16_t", "1"),
        ] {
            let (name, config) = register_kernel!(
                format!("rope_i_{}", dtype),
                "src/glsl/ropei.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- WHERE KERNELS ---
        for (cond_type_name, cond_glsl) in [("u8", "uint8_t"), ("u32", "uint")] {
            for (arg_type_name, arg_glsl) in [
                ("f32", "float"),
                ("f16", "float16_t"),
                ("bf16", "uint16_t"),
                ("i64", "int64_t"),
                ("u32", "uint"),
                ("u8", "uint8_t"),
            ] {
                let (name, config) = register_kernel!(
                    format!("where_{}_{}", cond_type_name, arg_type_name),
                    "src/glsl/where.comp",
                    ("COND_TYPE", cond_glsl),
                    ("ARG_TYPE", arg_glsl)
                );
                configs.insert(name, config);
            }
        }

        // --- CONV KERNELS ---
        {
            let (name, config) = register_kernel!(
                "conv1d_f32",
                "src/glsl/conv1d.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
        }
        for (dtype, inner_type, outer_type, is_bf16) in [
            ("f32", "float", "float", "0"),
            ("f16", "float", "float16_t", "0"),
            ("bf16", "float", "uint16_t", "1"),
            ("u32", "float", "uint", "0"),
        ] {
            let (name, config) = register_kernel!(
                format!("conv_transpose1d_{}", dtype),
                "src/glsl/conv_transpose1d.comp",
                ("INNER_TYPE", inner_type),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }
        for (dtype, inner_type, outer_type, is_bf16) in [
            ("f16", "float16_t", "float16_t", "0"),
            ("f32", "float", "float", "0"),
            ("bf16", "float", "uint16_t", "1"),
        ] {
            let (name, config) = register_kernel!(
                format!("conv2d_{}", dtype),
                "src/glsl/conv2d.comp",
                ("INNER_TYPE", inner_type),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }
        {
            let (name, config) = register_kernel!(
                "conv_transpose2d_f32",
                "src/glsl/conv_transpose2d.comp",
                ("INNER_TYPE", "float"),
                ("OUTER_TYPE", "float"),
                ("BF16", "0")
            );
            configs.insert(name, config);
        }

        // --- POOL2D KERNELS ---
        for (op, is_avg) in [("avg", "1"), ("max", "0")] {
            for (dtype, inner_type, outer_type, is_bf16) in [
                ("f32", "float", "float", "0"),
                ("f16", "float16_t", "float", "0"),
                ("bf16", "uint16_t", "float", "1"),
            ] {
                let (name, config) = register_kernel!(
                    format!("pool2d_{}_{}", op, dtype),
                    "src/glsl/pool2d.comp",
                    ("AVG", is_avg),
                    ("INNER_TYPE", inner_type),
                    ("OUTER_TYPE", outer_type),
                    ("BF16", is_bf16)
                );
                configs.insert(name, config);
            }
        }

        // --- UPSAMPLE KERNELS ---
        for (dtype, inner_type, outer_type, is_bf16) in [
            ("f32", "float", "float", "0"),
            ("bf16", "float", "uint16_t", "1"),
            ("f16", "float", "float16_t", "0"),
            ("u8", "uint8_t", "uint8_t", "0"),
            ("u32", "uint", "uint", "0"),
        ] {
            let (name, config) = register_kernel!(
                format!("upsample_nearest1d_{}", dtype),
                "src/glsl/upsample_nearest1d.comp",
                ("INNER_TYPE", inner_type),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                format!("upsample_nearest2d_{}", dtype),
                "src/glsl/upsample_nearest2d.comp",
                ("INNER_TYPE", inner_type),
                ("OUTER_TYPE", outer_type),
                ("BF16", is_bf16)
            );
            configs.insert(name, config);
        }

        // --- QUANTIZED MATMUL KERNELS ---
        {
            let (name, config) = register_kernel!("matmul_q8_0", "src/glsl/matmul_q8_0.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_q8_0_dp4a", "src/glsl/matmul_q8_0_dp4a.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_q8_0_dp4a_act_f16",
                "src/glsl/matmul_q8_0_dp4a.comp",
                ("ACT_DTYPE", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_q8_0_dp4a_act_bf16",
                "src/glsl/matmul_q8_0_dp4a.comp",
                ("ACT_DTYPE", "2")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!("matmul_q4_k", "src/glsl/matmul_q4_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!("matmul_q5_k", "src/glsl/matmul_q5_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!("matmul_q6_k", "src/glsl/matmul_q6_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!("matmul_q3_k", "src/glsl/matmul_q3_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!("matmul_q2_k", "src/glsl/matmul_q2_k.comp",);
            configs.insert(name, config);

            // Matrix-Vector multiplication (optimized for m=1 decode phase)
            let (name, config) =
                register_kernel!("matmul_vec_q8_0", "src/glsl/matmul_vec_q8_0.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q4_k", "src/glsl/matmul_vec_q4_k.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q5_k", "src/glsl/matmul_vec_q5_k.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q6_k", "src/glsl/matmul_vec_q6_k.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q3_k", "src/glsl/matmul_vec_q3_k.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q2_k", "src/glsl/matmul_vec_q2_k.comp",);
            configs.insert(name, config);

            // Cooperative K-reduction variant (BLOCK_SIZE threads per output row)
            let (name, config) =
                register_kernel!("matmul_vec_q8_0_coop", "src/glsl/matmul_vec_q8_0_coop.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_vec_q8_0_coop_act_f16",
                "src/glsl/matmul_vec_q8_0_coop.comp",
                ("ACT_DTYPE", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_vec_q8_0_coop_act_bf16",
                "src/glsl/matmul_vec_q8_0_coop.comp",
                ("ACT_DTYPE", "2")
            );
            configs.insert(name, config);

            // Integer dot product variant (requires VK_KHR_shader_integer_dot_product)
            let (name, config) =
                register_kernel!("matmul_vec_q8_0_idp", "src/glsl/matmul_vec_q8_0_idp.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_vec_q8_0_idp_act_f16",
                "src/glsl/matmul_vec_q8_0_idp.comp",
                ("ACT_DTYPE", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_vec_q8_0_idp_act_bf16",
                "src/glsl/matmul_vec_q8_0_idp.comp",
                ("ACT_DTYPE", "2")
            );
            configs.insert(name, config);

            // Cooperative K-quant variants (shared activation loading + NUM_ROWS)
            let (name, config) =
                register_kernel!("matmul_vec_q4_k_coop", "src/glsl/matmul_vec_q4_k_coop.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q5_k_coop", "src/glsl/matmul_vec_q5_k_coop.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q6_k_coop", "src/glsl/matmul_vec_q6_k_coop.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q3_k_coop", "src/glsl/matmul_vec_q3_k_coop.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_vec_q2_k_coop", "src/glsl/matmul_vec_q2_k_coop.comp",);
            configs.insert(name, config);
        }

        // --- HIERARCHICAL TILED MATMUL KERNELS (Phase 1-2: Q8_0 and K-quants) ---
        {
            let (name, config) =
                register_kernel!("matmul_tiled_q8_0", "src/glsl/matmul_tiled_q8_0.comp",);
            configs.insert(name, config);
            // Phase 3: Prefill-optimized variants (BM=64, 4 warps)
            let (name, config) = register_kernel!(
                "matmul_tiled_q8_0_prefill",
                "src/glsl/matmul_tiled_q8_0_prefill.comp",
            );
            configs.insert(name, config);
            // Phase 4: Optimized prefill variants
            let (name, config) = register_kernel!(
                "matmul_tiled_q8_0_prefill_opt",
                "src/glsl/matmul_tiled_q8_0_prefill_opt.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q8_0_prefill_vec",
                "src/glsl/matmul_tiled_q8_0_prefill_vec.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q8_0_prefill_db",
                "src/glsl/matmul_tiled_q8_0_prefill_db.comp",
            );
            configs.insert(name, config);
            // Phase 6: Hierarchical tiled coopmat
            let (name, config) = register_kernel!(
                "matmul_tiled_q8_0_prefill_coopmat",
                "src/glsl/matmul_tiled_q8_0_prefill_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_tiled_q2_k", "src/glsl/matmul_tiled_q2_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q2_k_prefill",
                "src/glsl/matmul_tiled_q2_k_prefill.comp",
            );
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_tiled_q3_k", "src/glsl/matmul_tiled_q3_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q3_k_prefill",
                "src/glsl/matmul_tiled_q3_k_prefill.comp",
            );
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_tiled_q4_k", "src/glsl/matmul_tiled_q4_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q4_k_prefill",
                "src/glsl/matmul_tiled_q4_k_prefill.comp",
            );
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_tiled_q5_k", "src/glsl/matmul_tiled_q5_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q5_k_prefill",
                "src/glsl/matmul_tiled_q5_k_prefill.comp",
            );
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_tiled_q6_k", "src/glsl/matmul_tiled_q6_k.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q6_k_prefill",
                "src/glsl/matmul_tiled_q6_k_prefill.comp",
            );
            configs.insert(name, config);
            // Phase 6: Hierarchical tiled coopmat variants for K-quants
            let (name, config) = register_kernel!(
                "matmul_tiled_q2_k_prefill_coopmat",
                "src/glsl/matmul_tiled_q2_k_prefill_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q3_k_prefill_coopmat",
                "src/glsl/matmul_tiled_q3_k_prefill_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q4_k_prefill_coopmat",
                "src/glsl/matmul_tiled_q4_k_prefill_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q5_k_prefill_coopmat",
                "src/glsl/matmul_tiled_q5_k_prefill_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "matmul_tiled_q6_k_prefill_coopmat",
                "src/glsl/matmul_tiled_q6_k_prefill_coopmat.comp",
            );
            configs.insert(name, config);
        }

        // --- COOPERATIVE MATRIX MATMUL KERNELS ---
        {
            let (name, config) =
                register_kernel!("matmul_q8_0_coopmat", "src/glsl/matmul_q8_0_coopmat.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_q4_k_coopmat", "src/glsl/matmul_q4_k_coopmat.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_q5_k_coopmat", "src/glsl/matmul_q5_k_coopmat.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_q6_k_coopmat", "src/glsl/matmul_q6_k_coopmat.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_q3_k_coopmat", "src/glsl/matmul_q3_k_coopmat.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("matmul_q2_k_coopmat", "src/glsl/matmul_q2_k_coopmat.comp",);
            configs.insert(name, config);
        }

        // --- FLASH ATTENTION Q8 KERNELS ---
        for head_dim in [64, 128, 256] {
            let (name, config) = register_kernel!(
                format!("flash_attn_q8_d{}", head_dim),
                "src/glsl/flash_attn_q8.comp",
                ("HEAD_DIM", head_dim.to_string())
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                format!("flash_attn_q8_idp_d{}", head_dim),
                "src/glsl/flash_attn_q8.comp",
                ("HEAD_DIM", head_dim.to_string()),
                ("USE_INTEGER_DOT", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                format!("flash_attn_q8_split_d{}", head_dim),
                "src/glsl/flash_attn_q8_split.comp",
                ("HEAD_DIM", head_dim.to_string())
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                format!("flash_attn_q8_split_idp_d{}", head_dim),
                "src/glsl/flash_attn_q8_split.comp",
                ("HEAD_DIM", head_dim.to_string()),
                ("USE_INTEGER_DOT", "1")
            );
            configs.insert(name, config);
        }
        {
            let (name, config) = register_kernel!(
                "flash_attn_q8_split_k_reduce",
                "src/glsl/flash_attn_q8_split_k_reduce.comp",
            );
            configs.insert(name, config);
        }

        // --- GLA STEP (DeltaNet autoregressive) KERNELS ---
        for head_dim in [64, 128, 256] {
            let (name, config) = register_kernel!(
                format!("gla_step_d{}", head_dim),
                "src/glsl/gla_step.comp",
                ("HEAD_DIM", head_dim.to_string())
            );
            configs.insert(name, config);
        }

        // --- DELTA NET PARALLEL (prefill) KERNELS ---
        for head_dim in [64, 128, 256] {
            let (name, config) = register_kernel!(
                format!("delta_net_parallel_d{}", head_dim),
                "src/glsl/delta_net_parallel.comp",
                ("HEAD_DIM", head_dim.to_string())
            );
            configs.insert(name, config);
        }

        // --- DELTANET OPS KERNELS ---
        {
            let (name, config) =
                register_kernel!("depthwise_conv1d", "src/glsl/depthwise_conv1d.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("l2_normalize_scale", "src/glsl/l2_normalize_scale.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!("fused_swiglu", "src/glsl/fused_swiglu.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("gated_rms_norm", "src/glsl/gated_rms_norm.comp",);
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("solve_triangular", "src/glsl/solve_triangular.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!("quantize_q8_0", "src/glsl/quantize_q8_0.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "quantize_q8_0_act_f16",
                "src/glsl/quantize_q8_0.comp",
                ("ACT_DTYPE", "1")
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "quantize_q8_0_act_bf16",
                "src/glsl/quantize_q8_0.comp",
                ("ACT_DTYPE", "2")
            );
            configs.insert(name, config);
            let (name, config) =
                register_kernel!("dequantize_q8_0", "src/glsl/dequantize_q8_0.comp",);
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q8_0",
                "src/glsl/indexed_moe_forward_q8_0.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q4_k",
                "src/glsl/indexed_moe_forward_q4_k.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q5_k",
                "src/glsl/indexed_moe_forward_q5_k.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q6_k",
                "src/glsl/indexed_moe_forward_q6_k.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q3_k",
                "src/glsl/indexed_moe_forward_q3_k.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q2_k",
                "src/glsl/indexed_moe_forward_q2_k.comp",
            );
            configs.insert(name, config);

            // --- Cooperative Matrix MoE Forward kernels ---
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q8_0_coopmat",
                "src/glsl/indexed_moe_forward_q8_0_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q4_k_coopmat",
                "src/glsl/indexed_moe_forward_q4_k_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q5_k_coopmat",
                "src/glsl/indexed_moe_forward_q5_k_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q6_k_coopmat",
                "src/glsl/indexed_moe_forward_q6_k_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q3_k_coopmat",
                "src/glsl/indexed_moe_forward_q3_k_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_forward_q2_k_coopmat",
                "src/glsl/indexed_moe_forward_q2_k_coopmat.comp",
            );
            configs.insert(name, config);

            // --- Fused MoE Gate+Up+SwiGLU kernels ---
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q8_0",
                "src/glsl/indexed_moe_gate_up_swiglu_q8_0.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q4_k",
                "src/glsl/indexed_moe_gate_up_swiglu_q4_k.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q5_k",
                "src/glsl/indexed_moe_gate_up_swiglu_q5_k.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q6_k",
                "src/glsl/indexed_moe_gate_up_swiglu_q6_k.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q3_k",
                "src/glsl/indexed_moe_gate_up_swiglu_q3_k.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q2_k",
                "src/glsl/indexed_moe_gate_up_swiglu_q2_k.comp",
            );
            configs.insert(name, config);

            // --- Cooperative Matrix MoE Gate+Up+SwiGLU kernels ---
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q8_0_coopmat",
                "src/glsl/indexed_moe_gate_up_swiglu_q8_0_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q4_k_coopmat",
                "src/glsl/indexed_moe_gate_up_swiglu_q4_k_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q5_k_coopmat",
                "src/glsl/indexed_moe_gate_up_swiglu_q5_k_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q6_k_coopmat",
                "src/glsl/indexed_moe_gate_up_swiglu_q6_k_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q3_k_coopmat",
                "src/glsl/indexed_moe_gate_up_swiglu_q3_k_coopmat.comp",
            );
            configs.insert(name, config);
            let (name, config) = register_kernel!(
                "indexed_moe_gate_up_swiglu_q2_k_coopmat",
                "src/glsl/indexed_moe_gate_up_swiglu_q2_k_coopmat.comp",
            );
            configs.insert(name, config);
        }

        // --- FUSED MATMUL+PERTURB KERNELS ---
        for fmt in &["q8_0", "q4_k", "q2_k", "q3_k", "q5_k", "q6_k", "bf16"] {
            let (name, config) = register_kernel!(
                format!("fused_matmul_{}", fmt),
                &format!("src/glsl/fused_matmul_{}.comp", fmt),
            );
            configs.insert(name, config);
        }

        // --- QuZO PERTURBATION KERNELS ---
        {
            let quzo_formats = ["q8_0", "q4_k", "q2_k", "q3_k", "q5_k", "q6_k", "bf16"];
            for fmt in &quzo_formats {
                let source_name = format!("quzo_{}", fmt);
                let source_path = format!("src/glsl/{}.comp", source_name);
                // Perturb variant
                let (name, config) = register_kernel!(
                    format!("perturb_{}", fmt),
                    &source_path,
                    ("MODE_PERTURB", "1")
                );
                configs.insert(name, config);
                // Restore+update variant
                let (name, config) = register_kernel!(
                    format!("restore_update_{}", fmt),
                    &source_path,
                    ("MODE_RESTORE_UPDATE", "1")
                );
                configs.insert(name, config);
            }
        }

        Ok(Self {
            configs,
            compiled: RwLock::new(HashMap::new()),
            pipelines: RwLock::new(HashMap::new()),
            use_push_descriptors: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Load (or compile on demand) SPIR-V for a kernel.
    pub fn load_spirv(
        &self,
        name: &str,
        additional: Option<&[(&str, &str)]>,
    ) -> Result<Vec<u32>, VulkanKernelError> {
        if let Some(spirv) = self.compiled.read()?.get(name).cloned() {
            return Ok(spirv);
        }
        let config = self.configs.get(name).ok_or_else(|| {
            VulkanKernelError::Message(format!("Kernel config for '{}' not found", name))
        })?;
        let spirv = config.compile(additional)?;
        self.compiled
            .write()?
            .insert(name.to_string(), spirv.clone());
        Ok(spirv)
    }

    /// Load (or create) a compute pipeline for the given kernel name.
    ///
    /// If `require_full_subgroups` is true, sets the
    /// `REQUIRE_FULL_SUBGROUPS` pipeline shader stage create flag (Vulkan 1.3).
    /// This guarantees the driver launches full subgroups, which is needed for
    /// correctness when shaders use subgroup operations with workgroup sizes
    /// that aren't multiples of the subgroup size.
    pub fn load_pipeline(
        &self,
        device: &ash::Device,
        name: &str,
        additional_defines: Option<&[(&str, &str)]>,
        push_constant_size: u32,
        num_buffers: u32,
        specialization_constants: Option<&[u32]>,
        require_full_subgroups: bool,
    ) -> Result<CachedPipeline, VulkanKernelError> {
        // Create cache key that includes specialization constants and flags
        let cache_key = if let Some(spec) = specialization_constants {
            if require_full_subgroups {
                format!("{}_{:?}_rfs", name, spec)
            } else {
                format!("{}_{:?}", name, spec)
            }
        } else if require_full_subgroups {
            format!("{}_rfs", name)
        } else {
            name.to_string()
        };

        // Check cache first
        if let Some(cached) = self.pipelines.read()?.get(&cache_key).cloned() {
            return Ok(cached);
        }

        let spirv = self.load_spirv(name, additional_defines)?;

        // Create shader module
        let shader_create_info = ash::vk::ShaderModuleCreateInfo::default().code(&spirv);
        let shader_module = unsafe { device.create_shader_module(&shader_create_info, None) }?;

        // Create descriptor set layout with `num_buffers` storage buffer bindings
        let push_desc = self
            .use_push_descriptors
            .load(std::sync::atomic::Ordering::Relaxed);
        let bindings: Vec<ash::vk::DescriptorSetLayoutBinding> = (0..num_buffers)
            .map(|i| {
                ash::vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let mut desc_layout_info =
            ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        if push_desc {
            desc_layout_info = desc_layout_info
                .flags(ash::vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
        }
        let desc_set_layout =
            unsafe { device.create_descriptor_set_layout(&desc_layout_info, None) }?;

        // Create pipeline layout with push constants
        let push_constant_range = ash::vk::PushConstantRange {
            stage_flags: ash::vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: push_constant_size,
        };
        let set_layouts = [desc_set_layout];
        let ranges = [push_constant_range];
        let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(if push_constant_size > 0 { &ranges } else { &[] });
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;

        // Create specialization info if provided
        let (spec_entries, spec_data, spec_info);
        let entry_name = c"main";
        let mut stage = ash::vk::PipelineShaderStageCreateInfo::default()
            .stage(ash::vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_name);

        if require_full_subgroups {
            stage = stage.flags(ash::vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
        }

        if let Some(constants) = specialization_constants {
            spec_entries = constants
                .iter()
                .enumerate()
                .map(|(i, _)| ash::vk::SpecializationMapEntry {
                    constant_id: i as u32,
                    offset: (i * 4) as u32,
                    size: 4,
                })
                .collect::<Vec<_>>();

            spec_data = constants
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect::<Vec<_>>();

            spec_info = ash::vk::SpecializationInfo::default()
                .map_entries(&spec_entries)
                .data(&spec_data);

            stage = stage.specialization_info(&spec_info);
        }

        let pipeline_info = ash::vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device.create_compute_pipelines(ash::vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| e)?[0];

        // Destroy shader module (no longer needed after pipeline creation)
        unsafe { device.destroy_shader_module(shader_module, None) };

        let cached = CachedPipeline {
            pipeline,
            layout: pipeline_layout,
            desc_set_layout,
            push_descriptors: push_desc,
        };
        self.pipelines.write()?.insert(cache_key, cached.clone());
        Ok(cached)
    }

    /// Clean up all cached Vulkan objects. Must be called before device destruction.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        if let Ok(pipelines) = self.pipelines.read() {
            for cached in pipelines.values() {
                device.destroy_pipeline(cached.pipeline, None);
                device.destroy_pipeline_layout(cached.layout, None);
                device.destroy_descriptor_set_layout(cached.desc_set_layout, None);
            }
        }
    }
}
