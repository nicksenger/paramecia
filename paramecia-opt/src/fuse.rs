//! Task Arithmetic Fusion for GGUF models.
//!
//! Implements the formula: θ_merged = θ_base + Σ wᵢ(θᵢ - θ_base)
//! Which simplifies to: θ_merged = θ_base(1 - Σwᵢ) + Σ wᵢθᵢ

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use paramecia_core::quantized::gguf_file::{self, Value, ValueType};
use paramecia_core::quantized::{GgmlDType, QTensor};
use paramecia_core::{Device, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn};

/// How to resolve differing quantization dtypes across fusion members.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantConflictStrategy {
    /// Reject the fusion if any tensor has conflicting dtypes.
    Reject,
    /// Use the highest-precision dtype for any conflicting tensors.
    Highest,
    /// Use the lowest-precision dtype for any conflicting tensors.
    Lowest,
}

/// Options for task arithmetic model fusion.
pub struct FuseOptions {
    /// Path to base GGUF model (reference point for computing deltas)
    pub base: PathBuf,
    /// Models to fuse: list of (path, weight) pairs.
    pub models: Vec<(PathBuf, f32)>,
    /// Output path for merged GGUF
    pub output: PathBuf,
    /// How to resolve quantization dtype conflicts across members.
    pub quant_conflict_strategy: QuantConflictStrategy,
}

/// Parsed model specification
struct ModelSpec {
    path: PathBuf,
    weight: f32,
}

/// Convert GgmlDType to u32 for GGUF format
fn ggml_dtype_to_u32(dtype: GgmlDType) -> u32 {
    match dtype {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q5_0 => 6,
        GgmlDType::Q5_1 => 7,
        GgmlDType::Q8_0 => 8,
        GgmlDType::Q8_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 11,
        GgmlDType::Q4K => 12,
        GgmlDType::Q5K => 13,
        GgmlDType::Q6K => 14,
        GgmlDType::Q8K => 15,
        GgmlDType::BF16 => 30,
    }
}

/// Convert ValueType to u32 for GGUF format
fn value_type_to_u32(vt: ValueType) -> u32 {
    match vt {
        ValueType::U8 => 0,
        ValueType::I8 => 1,
        ValueType::U16 => 2,
        ValueType::I16 => 3,
        ValueType::U32 => 4,
        ValueType::I32 => 5,
        ValueType::F32 => 6,
        ValueType::Bool => 7,
        ValueType::String => 8,
        ValueType::Array => 9,
        ValueType::U64 => 10,
        ValueType::I64 => 11,
        ValueType::F64 => 12,
    }
}

/// Write a GGUF string (length-prefixed)
fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    w.write_u64::<LittleEndian>(bytes.len() as u64)?;
    w.write_all(bytes)?;
    Ok(())
}

/// Write a Value to the output
fn write_value<W: Write>(w: &mut W, value: &Value) -> Result<()> {
    match value {
        Value::U8(v) => w.write_u8(*v)?,
        Value::I8(v) => w.write_i8(*v)?,
        Value::U16(v) => w.write_u16::<LittleEndian>(*v)?,
        Value::I16(v) => w.write_i16::<LittleEndian>(*v)?,
        Value::U32(v) => w.write_u32::<LittleEndian>(*v)?,
        Value::I32(v) => w.write_i32::<LittleEndian>(*v)?,
        Value::U64(v) => w.write_u64::<LittleEndian>(*v)?,
        Value::I64(v) => w.write_i64::<LittleEndian>(*v)?,
        Value::F32(v) => w.write_f32::<LittleEndian>(*v)?,
        Value::F64(v) => w.write_f64::<LittleEndian>(*v)?,
        Value::Bool(v) => w.write_u8(u8::from(*v))?,
        Value::String(v) => write_string(w, v.as_str())?,
        Value::Array(arr) => {
            let elem_type = if arr.is_empty() {
                ValueType::U32
            } else {
                arr[0].value_type()
            };
            w.write_u32::<LittleEndian>(value_type_to_u32(elem_type))?;
            w.write_u64::<LittleEndian>(arr.len() as u64)?;
            for elem in arr {
                write_value(w, elem)?;
            }
        }
    }
    Ok(())
}

/// Precision rank for GgmlDType (higher = more precise).
fn dtype_precision_rank(dtype: GgmlDType) -> u32 {
    match dtype {
        GgmlDType::Q2K => 1,
        GgmlDType::Q3K => 2,
        GgmlDType::Q4_0 => 3,
        GgmlDType::Q4_1 => 4,
        GgmlDType::Q4K => 5,
        GgmlDType::Q5_0 => 6,
        GgmlDType::Q5_1 => 7,
        GgmlDType::Q5K => 8,
        GgmlDType::Q6K => 9,
        GgmlDType::Q8_0 => 10,
        GgmlDType::Q8_1 => 11,
        GgmlDType::Q8K => 12,
        GgmlDType::BF16 => 13,
        GgmlDType::F16 => 14,
        GgmlDType::F32 => 15,
    }
}

/// Resolve the output dtype for a tensor given dtypes from all members + base.
fn resolve_dtype(
    name: &str,
    dtypes: &[GgmlDType],
    strategy: QuantConflictStrategy,
) -> Result<GgmlDType> {
    debug_assert!(!dtypes.is_empty());
    let first = dtypes[0];
    if dtypes.iter().all(|&d| d == first) {
        return Ok(first);
    }
    match strategy {
        QuantConflictStrategy::Reject => {
            bail!(
                "Quantization dtype conflict for tensor '{}': {:?}",
                name,
                dtypes
            );
        }
        QuantConflictStrategy::Highest => Ok(*dtypes
            .iter()
            .max_by_key(|d| dtype_precision_rank(**d))
            .unwrap()),
        QuantConflictStrategy::Lowest => Ok(*dtypes
            .iter()
            .min_by_key(|d| dtype_precision_rank(**d))
            .unwrap()),
    }
}

/// Check if a tensor should be copied verbatim (not fused)
fn is_metadata_tensor(name: &str) -> bool {
    name.contains("expert_mask") || name.contains("expert_remap")
}

/// Load a QTensor from a GGUF file given tensor info
fn load_qtensor(
    file: &mut BufReader<File>,
    tensor_data_offset: u64,
    info: &gguf_file::TensorInfo,
    device: &Device,
) -> Result<QTensor> {
    let dims: Vec<usize> = info.shape.dims().to_vec();
    let elem_count: usize = dims.iter().product();
    let dtype = info.ggml_dtype;
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    let size_in_bytes = elem_count / block_size * type_size;

    // Seek to tensor data
    file.seek(SeekFrom::Start(tensor_data_offset + info.offset))?;

    // Read raw bytes
    let mut data = vec![0u8; size_in_bytes];
    file.read_exact(&mut data)?;

    // Create QTensor
    QTensor::new(
        paramecia_core::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(data),
            device,
            dtype,
        )?,
        paramecia_core::Shape::from_dims(&dims),
    )
    .map_err(|e| anyhow::anyhow!("Failed to create QTensor: {}", e))
}

/// Apply task arithmetic fusion to tensors
/// Formula: θ_merged = θ_base(1 - Σwᵢ) + Σ wᵢθᵢ
fn fuse_tensors(base: &Tensor, models: &[Tensor], weights: &[f32]) -> Result<Tensor> {
    let sum_weights: f32 = weights.iter().sum();
    let base_weight = 1.0 - sum_weights;

    // Start with base * (1 - sum_weights)
    let mut result =
        (base * base_weight as f64).map_err(|e| anyhow::anyhow!("Failed to scale base: {}", e))?;

    // Add weighted model contributions
    for (model, &weight) in models.iter().zip(weights.iter()) {
        let weighted =
            (model * weight as f64).map_err(|e| anyhow::anyhow!("Failed to scale model: {}", e))?;
        result =
            (&result + &weighted).map_err(|e| anyhow::anyhow!("Failed to add tensors: {}", e))?;
    }

    Ok(result)
}

/// Parse a model specification string in the format "path:weight".
pub fn parse_model_spec(spec: &str) -> Result<(PathBuf, f32)> {
    let parts: Vec<&str> = spec.rsplitn(2, ':').collect();
    if parts.len() != 2 {
        bail!("Model spec must be in format 'path:weight', got: {}", spec);
    }
    let weight: f32 = parts[0]
        .parse()
        .with_context(|| format!("Invalid weight in model spec: {}", spec))?;
    let path = PathBuf::from(parts[1]);
    if !path.exists() {
        bail!("Model file not found: {}", path.display());
    }
    Ok((path, weight))
}

/// Perform task arithmetic fusion of multiple GGUF models.
pub fn fuse_models(options: &FuseOptions) -> Result<()> {
    // Convert to internal ModelSpec
    let model_specs: Vec<ModelSpec> = options
        .models
        .iter()
        .map(|(path, weight)| ModelSpec {
            path: path.clone(),
            weight: *weight,
        })
        .collect();

    if model_specs.is_empty() {
        bail!("At least one model must be specified");
    }

    let sum_weights: f32 = model_specs.iter().map(|s| s.weight).sum();

    // Print header
    info!("Task Arithmetic Fusion");
    info!("  Base:   {} (reference)", options.base.display());
    for (i, spec) in model_specs.iter().enumerate() {
        let suffix = if i == 0 { " (dtype source)" } else { "" };
        info!(
            "  Fusing: {} weight={:.2}{}",
            spec.path.display(),
            spec.weight,
            suffix
        );
    }
    if sum_weights > 1.0 {
        warn!(
            "Total weight {:.2} > 1.0 (extrapolating beyond base)",
            sum_weights
        );
    }

    // Open base file and read GGUF content
    let mut base_file = BufReader::new(File::open(&options.base)?);
    let base_gguf = gguf_file::Content::read(&mut base_file)
        .map_err(|e| anyhow::anyhow!("Failed to read base GGUF: {}", e))?;

    // Open model files
    let mut model_files: Vec<BufReader<File>> = model_specs
        .iter()
        .map(|spec| Ok(BufReader::new(File::open(&spec.path)?)))
        .collect::<Result<Vec<_>>>()?;

    // Read model GGUF contents
    let model_ggufs: Vec<gguf_file::Content> = model_files
        .iter_mut()
        .enumerate()
        .map(|(i, f)| {
            gguf_file::Content::read(f)
                .map_err(|e| anyhow::anyhow!("Failed to read model {} GGUF: {}", i, e))
        })
        .collect::<Result<Vec<_>>>()?;

    // Build tensor index for models
    let model_tensor_maps: Vec<HashMap<&str, &gguf_file::TensorInfo>> = model_ggufs
        .iter()
        .map(|g| {
            g.tensor_infos
                .iter()
                .map(|(k, v)| (k.as_str(), v))
                .collect()
        })
        .collect();

    // Verify all models have compatible tensors (check both directions)
    let base_tensor_names: Vec<&str> = base_gguf.tensor_infos.keys().map(|s| s.as_str()).collect();
    let first_model_tensor_names: Vec<&str> = model_ggufs[0]
        .tensor_infos
        .keys()
        .map(|s| s.as_str())
        .collect();

    // Check models have all base tensors
    for (i, map) in model_tensor_maps.iter().enumerate() {
        for name in &base_tensor_names {
            if !is_metadata_tensor(name) && !map.contains_key(name) {
                bail!(
                    "Model {} missing base tensor: {} (incompatible tensor layout?)",
                    i,
                    name
                );
            }
        }
    }

    // Check base has all first-model tensors (since first model defines the output layout)
    for name in &first_model_tensor_names {
        if !is_metadata_tensor(name) && !base_gguf.tensor_infos.contains_key(*name) {
            bail!(
                "Base model missing tensor: {} (incompatible tensor layout?)",
                name
            );
        }
    }

    // Prepare output
    let output_file = File::create(&options.output)?;
    let mut output = BufWriter::new(output_file);

    // Copy metadata from base, add fusion info
    let mut metadata: Vec<(&str, Value)> = base_gguf
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();

    // Add fusion metadata
    let model_paths: Vec<Value> = model_specs
        .iter()
        .map(|s| Value::String(s.path.to_string_lossy().into_owned()))
        .collect();
    let model_weights: Vec<Value> = model_specs.iter().map(|s| Value::F32(s.weight)).collect();

    metadata.push((
        "fusion.method",
        Value::String("task_arithmetic".to_string()),
    ));
    metadata.push((
        "fusion.base",
        Value::String(options.base.to_string_lossy().into_owned()),
    ));
    metadata.push(("fusion.models", Value::Array(model_paths)));
    metadata.push(("fusion.weights", Value::Array(model_weights)));

    metadata.sort_by_key(|(k, _)| *k);

    // Prepare tensor list - use order from first model, resolve dtypes via strategy
    let first_model_gguf = &model_ggufs[0];
    let tensor_names: Vec<&str> = first_model_gguf
        .tensor_infos
        .keys()
        .map(|s| s.as_str())
        .collect();
    let n_tensors = tensor_names.len();

    // Write GGUF header
    output.write_u32::<LittleEndian>(0x46554747)?; // "GGUF" magic
    output.write_u32::<LittleEndian>(3)?; // Version 3
    output.write_u64::<LittleEndian>(n_tensors as u64)?;
    output.write_u64::<LittleEndian>(metadata.len() as u64)?;

    // Write metadata
    for (key, value) in &metadata {
        write_string(&mut output, key)?;
        output.write_u32::<LittleEndian>(value_type_to_u32(value.value_type()))?;
        write_value(&mut output, value)?;
    }

    // Calculate tensor sizes and write tensor info headers
    info!("Writing {} tensor headers...", n_tensors);
    let mut tensor_infos: Vec<(&str, GgmlDType, Vec<usize>, usize)> = Vec::new();
    let mut data_offset: usize = 0;

    for name in &tensor_names {
        // Collect dtypes from all members for this tensor
        let mut member_dtypes: Vec<GgmlDType> = model_ggufs
            .iter()
            .filter_map(|g| g.tensor_infos.get(*name).map(|i| i.ggml_dtype))
            .collect();
        // Include base dtype if present
        if let Some(base_info) = base_gguf.tensor_infos.get(*name) {
            member_dtypes.push(base_info.ggml_dtype);
        }
        let dtype = resolve_dtype(name, &member_dtypes, options.quant_conflict_strategy)?;

        // Dims from first model (shapes must match across members)
        let info = first_model_gguf.tensor_infos.get(*name).unwrap();
        let dims: Vec<usize> = info.shape.dims().to_vec();
        let elem_count: usize = dims.iter().product();
        let block_size = dtype.block_size();
        let type_size = dtype.type_size();
        let size_in_bytes = elem_count / block_size * type_size;

        // Write tensor header
        write_string(&mut output, name)?;
        output.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims.iter().rev() {
            output.write_u64::<LittleEndian>(dim as u64)?;
        }
        output.write_u32::<LittleEndian>(ggml_dtype_to_u32(dtype))?;
        output.write_u64::<LittleEndian>(data_offset as u64)?;

        let padding = 31 - (31 + size_in_bytes) % 32;
        data_offset += size_in_bytes + padding;

        tensor_infos.push((name, dtype, dims, size_in_bytes));
    }

    // Pad header to 32-byte alignment
    let pos = output.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    output.write_all(&vec![0u8; padding])?;

    // Process and write tensor data
    info!("Processing {} tensors...", n_tensors);
    let device = Device::Cpu;
    let weights: Vec<f32> = model_specs.iter().map(|s| s.weight).collect();

    for (idx, (name, target_dtype, dims, size_in_bytes)) in tensor_infos.iter().enumerate() {
        let start = Instant::now();

        let result_bytes: Vec<u8> = if is_metadata_tensor(name) {
            // Copy metadata tensor from base
            let base_info = base_gguf.tensor_infos.get(*name).unwrap();
            base_file.seek(SeekFrom::Start(
                base_gguf.tensor_data_offset + base_info.offset,
            ))?;
            let mut data = vec![0u8; *size_in_bytes];
            base_file.read_exact(&mut data)?;
            data
        } else {
            // Load tensors from all sources
            let base_info = base_gguf.tensor_infos.get(*name).unwrap();
            let base_qtensor = load_qtensor(
                &mut base_file,
                base_gguf.tensor_data_offset,
                base_info,
                &device,
            )?;

            let model_qtensors: Vec<QTensor> = model_ggufs
                .iter()
                .zip(model_files.iter_mut())
                .map(|(gguf, file)| {
                    let info = gguf.tensor_infos.get(*name).unwrap();
                    load_qtensor(file, gguf.tensor_data_offset, info, &device)
                })
                .collect::<Result<Vec<_>>>()?;

            // Dequantize all to F32
            let base_f32 = base_qtensor
                .dequantize(&device)
                .map_err(|e| anyhow::anyhow!("Failed to dequantize base: {}", e))?;

            let models_f32: Vec<Tensor> = model_qtensors
                .iter()
                .map(|qt| {
                    qt.dequantize(&device)
                        .map_err(|e| anyhow::anyhow!("Failed to dequantize model: {}", e))
                })
                .collect::<Result<Vec<_>>>()?;

            // Apply task arithmetic fusion
            let fused = fuse_tensors(&base_f32, &models_f32, &weights)?;

            // Requantize to target dtype
            let fused_qtensor = QTensor::quantize(&fused, *target_dtype)
                .map_err(|e| anyhow::anyhow!("Failed to quantize fused tensor: {}", e))?;

            // Get raw bytes
            fused_qtensor
                .data()
                .map_err(|e| anyhow::anyhow!("Failed to get tensor data: {}", e))?
                .to_vec()
        };

        // Write tensor data
        output.write_all(&result_bytes)?;

        // Write padding
        let padding = 31 - (31 + size_in_bytes) % 32;
        if padding > 0 {
            output.write_all(&vec![0u8; padding])?;
        }

        let elapsed = start.elapsed();
        let shape_str: String = dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        debug!(
            "  [{}/{}] {} {:?} {} ... {}ms",
            idx + 1,
            n_tensors,
            name,
            target_dtype,
            shape_str,
            elapsed.as_millis()
        );
    }

    output.flush()?;

    let output_size = std::fs::metadata(&options.output)?.len();
    info!(
        "Written: {} ({:.2} GB)",
        options.output.display(),
        output_size as f64 / 1e9
    );

    Ok(())
}
