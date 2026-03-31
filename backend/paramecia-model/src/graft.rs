//! Combine MTP head from one GGUF with remaining tensors from another.
//!
//! This module takes two Qwen3-Next GGUF files and creates a new GGUF that
//! combines the MTP (Multi-Token Prediction) head from the first file with
//! all other tensors from the second file.
//!
//! ## Key properties:
//!
//! - **No dequantization/requantization**: Tensor data is copied as raw bytes directly
//!   from source files to output. Quantization format is preserved exactly.
//! - **Streaming I/O**: Only ~8MB buffer in memory at a time, regardless of model size.
//!   Tensors are never fully loaded into memory.
//! - **Bit-perfect copy**: The output tensors are byte-for-byte identical to the source.

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use clap::Args;
use paramecia_core::quantized::gguf_file::{self, TensorInfo, Value, ValueType};
use paramecia_core::quantized::GgmlDType;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Options for combining MTP head from one GGUF with base tensors from another.
#[derive(Args, Debug, Clone)]
pub struct GraftOptions {
    /// Path to GGUF file containing the MTP head to use
    #[arg(long)]
    pub mtp_source: PathBuf,

    /// Path to GGUF file containing the base model tensors
    #[arg(long)]
    pub base_model: PathBuf,

    /// Path for the output combined GGUF file
    #[arg(long)]
    pub output: PathBuf,

    /// Show verbose output (list all tensors being combined)
    #[arg(long, short)]
    pub verbose: bool,
}

/// Check if a tensor name belongs to the MTP head
fn is_mtp_tensor(name: &str) -> bool {
    name.starts_with("mtp.") || name.contains(".mtp.")
}

/// Compute storage size in bytes for a tensor from its info
fn tensor_size_in_bytes(info: &TensorInfo) -> usize {
    let elem_count = info.shape.elem_count();
    let block_size = info.ggml_dtype.block_size();
    let type_size = info.ggml_dtype.type_size();
    elem_count / block_size * type_size
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
fn write_string<W: std::io::Write>(w: &mut W, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    w.write_u64::<LittleEndian>(bytes.len() as u64)?;
    w.write_all(bytes)?;
    Ok(())
}

/// Write a Value to the output
fn write_value<W: std::io::Write>(w: &mut W, value: &Value) -> Result<()> {
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
            // Determine array element type
            let elem_type = if arr.is_empty() {
                ValueType::U32 // Doesn't matter for empty arrays
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

/// Information about where to read a tensor from
struct TensorReadInfo {
    name: String,
    size_in_bytes: usize,
    source_offset: u64,     // Offset within the source file's tensor data section
    source_data_start: u64, // Start of tensor data section in source file
    is_mtp: bool,           // true = read from mtp_file, false = read from base_file
    ggml_dtype: GgmlDType,
    dims: Vec<usize>,
}

/// Combine MTP head from one GGUF file with base tensors from another,
/// writing a new combined GGUF file.
pub fn graft(options: &GraftOptions) -> Result<()> {
    println!("Combine MTP GGUF Tool (Streaming)");
    println!("==================================\n");

    // Open and read metadata from both files (metadata is small)
    println!("Reading MTP source: {:?}", options.mtp_source);
    let mut mtp_file = BufReader::new(
        File::open(&options.mtp_source)
            .with_context(|| format!("Failed to open MTP source: {:?}", options.mtp_source))?,
    );
    let mtp_content = gguf_file::Content::read(&mut mtp_file)
        .with_context(|| "Failed to read MTP source GGUF content")?;

    println!("Reading base model: {:?}", options.base_model);
    let mut base_file = BufReader::new(
        File::open(&options.base_model)
            .with_context(|| format!("Failed to open base model: {:?}", options.base_model))?,
    );
    let base_content = gguf_file::Content::read(&mut base_file)
        .with_context(|| "Failed to read base model GGUF content")?;

    // Collect MTP tensor names from source
    let mtp_tensor_names: HashSet<String> = mtp_content
        .tensor_infos
        .keys()
        .filter(|name| is_mtp_tensor(name))
        .cloned()
        .collect();

    println!("\nFound {} MTP tensors in source", mtp_tensor_names.len());

    if mtp_tensor_names.is_empty() {
        anyhow::bail!("No MTP tensors found in source file. MTP tensors should start with 'mtp.' or contain '.mtp.'");
    }

    // Collect non-MTP tensor names from base model
    let base_tensor_names: Vec<String> = base_content
        .tensor_infos
        .keys()
        .filter(|name| !is_mtp_tensor(name))
        .cloned()
        .collect();

    println!(
        "Found {} non-MTP tensors in base model",
        base_tensor_names.len()
    );

    // Check for existing MTP tensors in base model (will be replaced)
    let base_mtp_count = base_content
        .tensor_infos
        .keys()
        .filter(|name| is_mtp_tensor(name))
        .count();

    if base_mtp_count > 0 {
        println!(
            "Note: {} MTP tensors in base model will be replaced",
            base_mtp_count
        );
    }

    // Build ordered list of tensors (sorted for deterministic output)
    let mut mtp_names_sorted: Vec<_> = mtp_tensor_names.iter().cloned().collect();
    mtp_names_sorted.sort();
    let mut base_names_sorted = base_tensor_names.clone();
    base_names_sorted.sort();

    let total_tensors = mtp_names_sorted.len() + base_names_sorted.len();

    if options.verbose {
        println!("\nMTP tensors from source:");
        for name in &mtp_names_sorted {
            let info = &mtp_content.tensor_infos[name];
            println!(
                "  {} : {:?} ({:?})",
                name,
                info.shape.dims(),
                info.ggml_dtype
            );
        }
        println!("\nBase tensors:");
        for name in &base_names_sorted {
            let info = &base_content.tensor_infos[name];
            println!(
                "  {} : {:?} ({:?})",
                name,
                info.shape.dims(),
                info.ggml_dtype
            );
        }
    }

    // Build tensor read info (small metadata, not actual data)
    let mut tensor_infos: Vec<TensorReadInfo> = Vec::with_capacity(total_tensors);

    for name in &mtp_names_sorted {
        let info = &mtp_content.tensor_infos[name];
        tensor_infos.push(TensorReadInfo {
            name: name.clone(),
            size_in_bytes: tensor_size_in_bytes(info),
            source_offset: info.offset,
            source_data_start: mtp_content.tensor_data_offset,
            is_mtp: true,
            ggml_dtype: info.ggml_dtype,
            dims: info.shape.dims().to_vec(),
        });
    }

    for name in &base_names_sorted {
        let info = &base_content.tensor_infos[name];
        tensor_infos.push(TensorReadInfo {
            name: name.clone(),
            size_in_bytes: tensor_size_in_bytes(info),
            source_offset: info.offset,
            source_data_start: base_content.tensor_data_offset,
            is_mtp: false,
            ggml_dtype: info.ggml_dtype,
            dims: info.shape.dims().to_vec(),
        });
    }

    // Create output file
    if let Some(parent) = options.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
        }
    }

    println!("\nWriting output to: {:?}", options.output);

    let output_file = File::create(&options.output)
        .with_context(|| format!("Failed to create output file: {:?}", options.output))?;
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, output_file);

    // Merge metadata: base model metadata + MTP-related metadata from MTP source
    let mut merged_metadata: std::collections::HashMap<&str, &Value> =
        std::collections::HashMap::new();

    // Start with all base model metadata
    for (k, v) in base_content.metadata.iter() {
        merged_metadata.insert(k.as_str(), v);
    }

    // Add/override with MTP-related metadata from MTP source
    let mut mtp_metadata_keys: Vec<&str> = Vec::new();
    for (k, v) in mtp_content.metadata.iter() {
        // Include metadata that mentions "mtp" (case-insensitive)
        if k.to_lowercase().contains("mtp") {
            merged_metadata.insert(k.as_str(), v);
            mtp_metadata_keys.push(k.as_str());
        }
    }
    mtp_metadata_keys.sort();

    if options.verbose && !mtp_metadata_keys.is_empty() {
        println!("\nMTP metadata from source:");
        for key in &mtp_metadata_keys {
            println!("  {}: {:?}", key, mtp_content.metadata.get(*key).unwrap());
        }
    }

    // Convert to sorted vec for deterministic output
    let mut metadata: Vec<(&str, &Value)> = merged_metadata.into_iter().collect();
    metadata.sort_by_key(|(k, _)| *k);

    println!(
        "Merged metadata: {} base entries + {} MTP entries = {} total",
        base_content.metadata.len(),
        mtp_metadata_keys.len(),
        metadata.len()
    );

    // === Write GGUF Header ===
    output.write_u32::<LittleEndian>(0x46554747)?; // "GGUF" magic
    output.write_u32::<LittleEndian>(2)?; // Version 2
    output.write_u64::<LittleEndian>(total_tensors as u64)?;
    output.write_u64::<LittleEndian>(metadata.len() as u64)?;

    // Write metadata
    for (name, value) in &metadata {
        write_string(&mut output, name)?;
        output.write_u32::<LittleEndian>(value_type_to_u32(value.value_type()))?;
        write_value(&mut output, value)?;
    }

    // === Write Tensor Headers ===
    println!("Writing tensor headers...");
    let mut data_offset: usize = 0;

    for info in &tensor_infos {
        write_string(&mut output, &info.name)?;

        output.write_u32::<LittleEndian>(info.dims.len() as u32)?;
        for &dim in info.dims.iter().rev() {
            output.write_u64::<LittleEndian>(dim as u64)?;
        }
        output.write_u32::<LittleEndian>(ggml_dtype_to_u32(info.ggml_dtype))?;
        output.write_u64::<LittleEndian>(data_offset as u64)?;

        let padding = 31 - (31 + info.size_in_bytes) % 32;
        data_offset += info.size_in_bytes + padding;
    }

    // Pad header to 32-byte alignment
    let pos = output.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    output.write_all(&vec![0u8; padding])?;

    // === Stream Tensor Data ===
    println!(
        "Streaming {} tensors (raw byte copy, no requantization)...",
        total_tensors
    );

    // Use a buffer for copying (8MB)
    const BUFFER_SIZE: usize = 8 * 1024 * 1024;
    let mut buffer = vec![0u8; BUFFER_SIZE];

    let mut tensors_written = 0;
    for info in &tensor_infos {
        // Select source file
        let source_file: &mut BufReader<File> = if info.is_mtp {
            &mut mtp_file
        } else {
            &mut base_file
        };

        // Seek to tensor data in source file
        let source_pos = info.source_data_start + info.source_offset;
        source_file.seek(SeekFrom::Start(source_pos))?;

        // Raw byte copy - no dequantization, no QTensor creation, no requantization
        let mut remaining = info.size_in_bytes;
        while remaining > 0 {
            let to_read = remaining.min(BUFFER_SIZE);
            source_file.read_exact(&mut buffer[..to_read])?;
            output.write_all(&buffer[..to_read])?;
            remaining -= to_read;
        }

        // Write padding to 32-byte alignment
        let padding = 31 - (31 + info.size_in_bytes) % 32;
        if padding > 0 {
            output.write_all(&vec![0u8; padding])?;
        }

        tensors_written += 1;
        if tensors_written % 100 == 0 || tensors_written == total_tensors {
            print!("\r  Streamed {}/{} tensors", tensors_written, total_tensors);
            std::io::stdout().flush()?;
        }
    }
    println!();

    output.flush()?;

    // Get final file size
    let output_size = std::fs::metadata(&options.output)?.len();

    println!(
        "\nSuccess! Written {} tensors ({:.2} GB)",
        total_tensors,
        output_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    println!("\nSummary:");
    println!(
        "  MTP tensors from: {:?} ({} tensors)",
        options.mtp_source,
        mtp_names_sorted.len()
    );
    println!(
        "  Base tensors from: {:?} ({} tensors)",
        options.base_model,
        base_names_sorted.len()
    );
    println!("  Output: {:?}", options.output);
    println!(
        "\nPeak memory usage: ~{} MB (streaming buffer only)",
        BUFFER_SIZE / (1024 * 1024)
    );

    Ok(())
}

// ============================================================================
// Composite graft — assemble GGUF from multiple source models
// ============================================================================

/// Opened GGUF source file with parsed content.
struct GgufSource {
    file: BufReader<File>,
    content: gguf_file::Content,
}

fn open_gguf_source(path: &Path) -> Result<GgufSource> {
    let mut file = BufReader::new(
        File::open(path).with_context(|| format!("Failed to open GGUF: {:?}", path))?,
    );
    let content = gguf_file::Content::read(&mut file)
        .with_context(|| format!("Failed to read GGUF content: {:?}", path))?;
    Ok(GgufSource { file, content })
}

/// Source for a layer in a composite — path + source layer index.
#[derive(Debug, Clone)]
pub struct GraftLayerSource {
    pub path: PathBuf,
    pub layer_idx: u32,
}

/// Resolved composite specification — all weights references are file paths.
#[derive(Debug, Clone)]
pub struct GraftComposite {
    pub embedding: PathBuf,
    pub layers: Vec<GraftLayerSource>,
    pub lm_head: PathBuf,
    pub mtp_head: PathBuf,
}

/// Options for composite grafting.
pub struct GraftCompositeOptions {
    /// Composite specification with all paths resolved.
    pub composite: GraftComposite,
    /// Output path.
    pub output: PathBuf,
    pub verbose: bool,
}

/// Tensor read info for composite assembly.
struct CompositeReadInfo {
    output_name: String,
    source_path: PathBuf,
    source_tensor_name: String,
    size_in_bytes: usize,
    ggml_dtype: GgmlDType,
    dims: Vec<usize>,
}

/// Combine tensors from multiple GGUF sources into a new GGUF based on
/// a composite specification. All weights references must be pre-resolved
/// to file paths by the caller.
pub fn graft_composite(options: &GraftCompositeOptions) -> Result<()> {
    let comp = &options.composite;

    // Open and cache all unique source GGUF files.
    let mut all_paths: Vec<PathBuf> = vec![
        comp.embedding.clone(),
        comp.lm_head.clone(),
        comp.mtp_head.clone(),
    ];
    for layer in &comp.layers {
        all_paths.push(layer.path.clone());
    }
    all_paths.sort();
    all_paths.dedup();

    let mut source_cache: HashMap<PathBuf, GgufSource> = HashMap::new();
    for path in &all_paths {
        source_cache.insert(path.clone(), open_gguf_source(path)?);
    }

    let mut read_infos: Vec<CompositeReadInfo> = Vec::new();

    // Embedding
    {
        let source = &source_cache[&comp.embedding];
        if let Some(info) = source.content.tensor_infos.get("token_embd.weight") {
            read_infos.push(CompositeReadInfo {
                output_name: "token_embd.weight".to_string(),
                source_path: comp.embedding.clone(),
                source_tensor_name: "token_embd.weight".to_string(),
                size_in_bytes: tensor_size_in_bytes(info),
                ggml_dtype: info.ggml_dtype,
                dims: info.shape.dims().to_vec(),
            });
        }
    }

    // Layers
    for (out_idx, layer_source) in comp.layers.iter().enumerate() {
        let out_prefix = format!("blk.{}.", out_idx);
        let src_prefix = format!("blk.{}.", layer_source.layer_idx);
        let source = &source_cache[&layer_source.path];

        let mut names: Vec<_> = source
            .content
            .tensor_infos
            .keys()
            .filter(|n| n.starts_with(&src_prefix))
            .cloned()
            .collect();
        names.sort();
        for name in names {
            let info = &source.content.tensor_infos[&name];
            let output_name = name.replacen(&src_prefix, &out_prefix, 1);
            read_infos.push(CompositeReadInfo {
                output_name,
                source_path: layer_source.path.clone(),
                source_tensor_name: name,
                size_in_bytes: tensor_size_in_bytes(info),
                ggml_dtype: info.ggml_dtype,
                dims: info.shape.dims().to_vec(),
            });
        }
    }

    // LM head
    {
        let source = &source_cache[&comp.lm_head];
        for name in ["output.weight", "output_norm.weight"] {
            if let Some(info) = source.content.tensor_infos.get(name) {
                read_infos.push(CompositeReadInfo {
                    output_name: name.to_string(),
                    source_path: comp.lm_head.clone(),
                    source_tensor_name: name.to_string(),
                    size_in_bytes: tensor_size_in_bytes(info),
                    ggml_dtype: info.ggml_dtype,
                    dims: info.shape.dims().to_vec(),
                });
            }
        }
    }

    // MTP head
    {
        let source = &source_cache[&comp.mtp_head];
        let mut mtp_names: Vec<_> = source
            .content
            .tensor_infos
            .keys()
            .filter(|n| is_mtp_tensor(n))
            .cloned()
            .collect();
        mtp_names.sort();
        for name in mtp_names {
            let info = &source.content.tensor_infos[&name];
            read_infos.push(CompositeReadInfo {
                output_name: name.clone(),
                source_path: comp.mtp_head.clone(),
                source_tensor_name: name,
                size_in_bytes: tensor_size_in_bytes(info),
                ggml_dtype: info.ggml_dtype,
                dims: info.shape.dims().to_vec(),
            });
        }
    }

    // Sort by output name for deterministic ordering
    read_infos.sort_by(|a, b| a.output_name.cmp(&b.output_name));

    if options.verbose {
        println!("Composite tensor list ({} tensors):", read_infos.len());
        for ri in &read_infos {
            println!(
                "  {} <- {:?}:{}",
                ri.output_name,
                ri.source_path.file_name().unwrap_or_default(),
                ri.source_tensor_name
            );
        }
    }

    // Merge metadata from the first source (embedding), update block_count.
    // Use embedding source as the "default" for metadata.
    let default_source = &source_cache[&comp.embedding];
    let n_output_layers = comp.layers.len();
    let mut metadata: Vec<(String, Value)> = Vec::new();
    for (key, value) in default_source.content.metadata.iter() {
        if key.ends_with(".block_count") {
            metadata.push((key.clone(), Value::U32(n_output_layers as u32)));
        } else {
            metadata.push((key.clone(), value.clone()));
        }
    }
    metadata.sort_by(|a, b| a.0.cmp(&b.0));

    // Write output GGUF
    if let Some(parent) = options.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let output_file = File::create(&options.output)?;
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, output_file);

    // Header
    output.write_u32::<LittleEndian>(0x46554747)?; // "GGUF"
    output.write_u32::<LittleEndian>(3)?; // Version 3
    output.write_u64::<LittleEndian>(read_infos.len() as u64)?;
    output.write_u64::<LittleEndian>(metadata.len() as u64)?;

    // Metadata
    for (key, value) in &metadata {
        write_string(&mut output, key)?;
        output.write_u32::<LittleEndian>(value_type_to_u32(value.value_type()))?;
        write_value(&mut output, value)?;
    }

    // Tensor headers
    let mut data_offset: usize = 0;
    for ri in &read_infos {
        write_string(&mut output, &ri.output_name)?;
        output.write_u32::<LittleEndian>(ri.dims.len() as u32)?;
        for &dim in ri.dims.iter().rev() {
            output.write_u64::<LittleEndian>(dim as u64)?;
        }
        output.write_u32::<LittleEndian>(ggml_dtype_to_u32(ri.ggml_dtype))?;
        output.write_u64::<LittleEndian>(data_offset as u64)?;

        let padding = 31 - (31 + ri.size_in_bytes) % 32;
        data_offset += ri.size_in_bytes + padding;
    }

    // Pad header to 32-byte alignment
    let pos = output.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    output.write_all(&vec![0u8; padding])?;

    // Stream tensor data
    const BUFFER_SIZE: usize = 8 * 1024 * 1024;
    let mut buffer = vec![0u8; BUFFER_SIZE];

    for ri in &read_infos {
        let source = source_cache
            .get_mut(&ri.source_path)
            .ok_or_else(|| anyhow::anyhow!("Source {:?} not in cache", ri.source_path))?;
        let info = source
            .content
            .tensor_infos
            .get(&ri.source_tensor_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Tensor {} not found in {:?}",
                    ri.source_tensor_name,
                    ri.source_path
                )
            })?;
        let source_pos = source.content.tensor_data_offset + info.offset;
        source.file.seek(SeekFrom::Start(source_pos))?;

        let mut remaining = ri.size_in_bytes;
        while remaining > 0 {
            let to_read = remaining.min(BUFFER_SIZE);
            source.file.read_exact(&mut buffer[..to_read])?;
            output.write_all(&buffer[..to_read])?;
            remaining -= to_read;
        }

        let padding = 31 - (31 + ri.size_in_bytes) % 32;
        if padding > 0 {
            output.write_all(&vec![0u8; padding])?;
        }
    }

    output.flush()?;
    Ok(())
}
