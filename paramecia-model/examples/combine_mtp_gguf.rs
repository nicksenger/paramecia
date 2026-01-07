//! Combine MTP head from one GGUF with remaining tensors from another
//!
//! This example takes two Qwen3-Next GGUF files and creates a new GGUF that
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
//!
//! ## Use cases:
//!
//! - Transplanting a fine-tuned MTP head onto a different base model
//! - Combining MTP weights from a specialized model with a quantized base
//! - Swapping MTP heads between models without quality loss from requantization
//!
//! ## MTP tensor naming patterns:
//!
//! - Top-level: `mtp.fc.weight`, `mtp.pre_fc_norm_embedding.weight`, `mtp.norm.weight`
//! - Per-layer: `blk.{idx}.mtp.attn_q.weight`, `blk.{idx}.mtp.ffn_gate_exps.weight`, etc.

use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use candle::quantized::gguf_file::{self, TensorInfo, Value, ValueType};
use candle::quantized::GgmlDType;
use clap::Parser;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Combine MTP head from one GGUF with base tensors from another"
)]
struct Args {
    /// Path to GGUF file containing the MTP head to use
    #[arg(long)]
    mtp_source: PathBuf,

    /// Path to GGUF file containing the base model tensors
    #[arg(long)]
    base_model: PathBuf,

    /// Path for the output combined GGUF file
    #[arg(long)]
    output: PathBuf,

    /// Show verbose output (list all tensors being combined)
    #[arg(long, short)]
    verbose: bool,
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

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Combine MTP GGUF Tool (Streaming)");
    println!("==================================\n");

    // Open and read metadata from both files (metadata is small)
    println!("Reading MTP source: {:?}", args.mtp_source);
    let mut mtp_file = BufReader::new(
        File::open(&args.mtp_source)
            .with_context(|| format!("Failed to open MTP source: {:?}", args.mtp_source))?,
    );
    let mtp_content = gguf_file::Content::read(&mut mtp_file)
        .with_context(|| "Failed to read MTP source GGUF content")?;

    println!("Reading base model: {:?}", args.base_model);
    let mut base_file = BufReader::new(
        File::open(&args.base_model)
            .with_context(|| format!("Failed to open base model: {:?}", args.base_model))?,
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

    if args.verbose {
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
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
        }
    }

    println!("\nWriting output to: {:?}", args.output);

    let output_file = File::create(&args.output)
        .with_context(|| format!("Failed to create output file: {:?}", args.output))?;
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, output_file);

    // Merge metadata: base model metadata + MTP-related metadata from MTP source
    // MTP metadata keys typically contain "mtp" (e.g., "qwen3.mtp_depth", "mtp.*")
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

    if args.verbose && !mtp_metadata_keys.is_empty() {
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
    // IMPORTANT: This performs raw byte copying - no dequantization or requantization!
    // Tensor data is copied bit-for-bit from source files, preserving exact quantization.
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
        // The quantized blocks are copied exactly as they exist in the source file
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
    let output_size = std::fs::metadata(&args.output)?.len();

    println!(
        "\nSuccess! Written {} tensors ({:.2} GB)",
        total_tensors,
        output_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    println!("\nSummary:");
    println!(
        "  MTP tensors from: {:?} ({} tensors)",
        args.mtp_source,
        mtp_names_sorted.len()
    );
    println!(
        "  Base tensors from: {:?} ({} tensors)",
        args.base_model,
        base_names_sorted.len()
    );
    println!("  Output: {:?}", args.output);
    println!(
        "\nPeak memory usage: ~{} MB (streaming buffer only)",
        BUFFER_SIZE / (1024 * 1024)
    );

    Ok(())
}
