//! Pruning tools for GGUF models.
//!
//! Provides expert pruning (removing unused MoE experts based on tuning statistics)
//! and layer pruning (removing transformer layers from the end of the model).

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use byteorder::{LittleEndian, WriteBytesExt};
use paramecia_core::quantized::gguf_file::{self, TensorInfo, Value, ValueType};
use paramecia_core::quantized::GgmlDType;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

// ============================================================================
// Shared GGUF helpers
// ============================================================================

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

/// Helper to get u32 metadata value
fn get_u32_metadata(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<u32> {
    for key in keys {
        if let Some(value) = metadata.get(*key) {
            match value {
                Value::U32(v) => return Some(*v),
                Value::I32(v) => return Some(*v as u32),
                Value::U64(v) => return Some(*v as u32),
                Value::I64(v) => return Some(*v as u32),
                _ => continue,
            }
        }
    }
    None
}

/// Compute storage size in bytes for a tensor
fn tensor_size_in_bytes(info: &TensorInfo) -> usize {
    let elem_count = info.shape.elem_count();
    let block_size = info.ggml_dtype.block_size();
    let type_size = info.ggml_dtype.type_size();
    elem_count / block_size * type_size
}

/// Extract layer index from tensor name like "blk.5.ffn_gate_exps.weight"
fn extract_layer_index(name: &str) -> Option<usize> {
    if let Some(start) = name.find("blk.") {
        let rest = &name[start + 4..];
        if let Some(end) = rest.find('.') {
            return rest[..end].parse().ok();
        }
    }
    None
}

// ============================================================================
// Expert pruning
// ============================================================================

/// Options for pruning MoE experts with explicit per-layer retained indices.
pub struct PruneExpertsOptions {
    /// Path to input GGUF model file
    pub input_model: PathBuf,
    /// Path for output GGUF with pruned experts
    pub output_model: PathBuf,
    /// Per-layer list of original expert indices to retain (sorted).
    /// Length must equal the number of layers in the model.
    pub retained_indices: Vec<Vec<u16>>,
    /// Show verbose output
    pub verbose: bool,
}

/// Tuning file header (must match paramecia-text/src/tuning.rs)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TuningHeader {
    magic: [u8; 4],
    version: u16,
    n_top_k: u16,
    n_tail: u16,
    n_tokens: u32,
    n_assistant_tokens: u32,
    vocab_size: u32,
    n_layers: u16,
    n_experts_per_tok: u8,
    reserved: [u8; 7],
}

impl TuningHeader {
    fn is_valid(&self) -> bool {
        self.magic == *b"PGEN" && (self.version == 1 || self.version == 2)
    }

    fn has_expert_data(&self) -> bool {
        self.version >= 2 && self.n_layers > 0 && self.n_experts_per_tok > 0
    }

    fn entries_per_token(&self) -> usize {
        self.n_top_k as usize + self.n_tail as usize
    }
}

/// Logit entry (must match paramecia-text/src/tuning.rs)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LogitEntry {
    token_id: u32,
    log_prob_bits: u16,
}

/// Collect all .bin files from a path (file or directory, recursive)
fn collect_tuning_files(path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if path.is_file() {
        if path.extension().map(|e| e == "bin").unwrap_or(false) {
            files.push(path.to_path_buf());
        }
    } else if path.is_dir() {
        for entry in walkdir::WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let p = entry.path();
            if p.is_file() && p.extension().map(|e| e == "bin").unwrap_or(false) {
                files.push(p.to_path_buf());
            }
        }
    }

    Ok(files)
}

/// Expert usage statistics per layer
#[derive(Debug, Default)]
struct ExpertUsage {
    /// Counts per expert: counts[layer][expert_id] = count
    counts: HashMap<usize, HashMap<u8, u64>>,
    /// Number of layers detected from tuning files
    n_layers: usize,
    /// Number of experts per token from tuning files
    n_experts_per_tok: usize,
    /// Total tokens processed
    total_tokens: u64,
}

impl ExpertUsage {
    fn new() -> Self {
        Self::default()
    }

    fn record(&mut self, layer: usize, expert_id: u8) {
        *self
            .counts
            .entry(layer)
            .or_default()
            .entry(expert_id)
            .or_insert(0) += 1;
    }

    /// Get the top N experts by usage for a given layer, sorted by usage descending
    fn top_n_experts(&self, layer: usize, n: usize) -> Vec<u16> {
        let Some(layer_counts) = self.counts.get(&layer) else {
            return Vec::new();
        };

        let mut sorted: Vec<_> = layer_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count descending

        sorted
            .into_iter()
            .take(n)
            .map(|(&id, _)| id as u16)
            .collect()
    }
}

/// Read expert usage from a single tuning file
fn read_expert_usage(path: &Path, usage: &mut ExpertUsage, verbose: bool) -> Result<()> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut header_bytes = [0u8; 32];
    reader.read_exact(&mut header_bytes)?;
    let header: TuningHeader = *bytemuck::from_bytes(&header_bytes);

    if !header.is_valid() {
        anyhow::bail!("Invalid tuning file: {:?}", path);
    }

    if !header.has_expert_data() {
        if verbose {
            info!("  Skipping {:?} (no expert data)", path);
        }
        return Ok(());
    }

    let n_layers = header.n_layers as usize;
    let n_experts_per_tok = header.n_experts_per_tok as usize;
    let n_assistant_tokens = header.n_assistant_tokens as usize;
    let n_tokens = header.n_tokens as usize;

    // Update global stats
    if usage.n_layers == 0 {
        usage.n_layers = n_layers;
        usage.n_experts_per_tok = n_experts_per_tok;
    } else if usage.n_layers != n_layers || usage.n_experts_per_tok != n_experts_per_tok {
        anyhow::bail!(
            "Inconsistent model config: expected {} layers, {} experts/tok, got {} layers, {} experts/tok",
            usage.n_layers, usage.n_experts_per_tok,
            n_layers, n_experts_per_tok
        );
    }

    // Skip token IDs (u32 * n_tokens)
    reader.seek(SeekFrom::Current(4 * n_tokens as i64))?;

    // Skip assistant mask (u8 * n_tokens)
    reader.seek(SeekFrom::Current(n_tokens as i64))?;

    // Skip logit data (LogitEntry * entries_per_token * n_assistant_tokens)
    let entries_per_token = header.entries_per_token();
    let logit_entry_size = std::mem::size_of::<LogitEntry>();
    reader.seek(SeekFrom::Current(
        (logit_entry_size * entries_per_token * n_assistant_tokens) as i64,
    ))?;

    // Skip tail mass values (f16 * n_assistant_tokens)
    reader.seek(SeekFrom::Current(2 * n_assistant_tokens as i64))?;

    // Read expert indices
    let indices_per_token = n_layers * n_experts_per_tok;
    let mut expert_buffer = vec![0u8; indices_per_token];

    for _ in 0..n_assistant_tokens {
        reader.read_exact(&mut expert_buffer)?;

        for layer in 0..n_layers {
            let layer_start = layer * n_experts_per_tok;
            for i in 0..n_experts_per_tok {
                let expert_id = expert_buffer[layer_start + i];
                usage.record(layer, expert_id);
            }
        }
    }

    usage.total_tokens += n_assistant_tokens as u64;

    if verbose {
        info!(
            "  {:?}: {} assistant tokens, {} layers, {} experts/tok",
            path.file_name().unwrap_or_default(),
            n_assistant_tokens,
            n_layers,
            n_experts_per_tok
        );
    }

    Ok(())
}

/// Compute per-layer retained expert indices from tuning data files.
///
/// Reads expert usage statistics from binary tuning files and returns the
/// top `experts_to_keep` experts per layer, sorted by original index.
pub fn compute_expert_indices_from_tuning(
    input_model: &Path,
    tuning_data: &Path,
    experts_to_keep: usize,
    verbose: bool,
) -> Result<Vec<Vec<u16>>> {
    // Read input GGUF to get model config
    let mut input_file = BufReader::new(
        File::open(input_model)
            .with_context(|| format!("Failed to open input model: {:?}", input_model))?,
    );
    let input_content = gguf_file::Content::read(&mut input_file)
        .with_context(|| "Failed to read input GGUF content")?;

    let n_experts = get_u32_metadata(
        &input_content.metadata,
        &[
            "qwen3next.expert_count",
            "qwen3moe.expert_count",
            "llama.expert_count",
        ],
    )
    .ok_or_else(|| anyhow::anyhow!("Could not find expert_count in model metadata"))?
        as usize;

    let n_experts_per_tok = get_u32_metadata(
        &input_content.metadata,
        &[
            "qwen3next.expert_used_count",
            "qwen3moe.expert_used_count",
            "llama.expert_used_count",
        ],
    )
    .unwrap_or(8) as usize;

    let n_layers = get_u32_metadata(
        &input_content.metadata,
        &[
            "qwen3next.block_count",
            "qwen3moe.block_count",
            "llama.block_count",
        ],
    )
    .ok_or_else(|| anyhow::anyhow!("Could not find block_count in model metadata"))?
        as usize;

    if experts_to_keep >= n_experts {
        anyhow::bail!(
            "Requested {} experts to keep but model only has {} experts per layer.",
            experts_to_keep,
            n_experts
        );
    }

    if experts_to_keep < n_experts_per_tok {
        anyhow::bail!(
            "Must keep at least {} experts (experts_used_count), requested {}",
            n_experts_per_tok,
            experts_to_keep
        );
    }

    // Collect and read tuning files
    let tuning_files = collect_tuning_files(tuning_data)?;
    if tuning_files.is_empty() {
        anyhow::bail!("No .bin tuning files found at {:?}", tuning_data);
    }

    let mut usage = ExpertUsage::new();
    for path in &tuning_files {
        read_expert_usage(path, &mut usage, verbose)
            .with_context(|| format!("Failed to read {:?}", path))?;
    }

    if usage.total_tokens == 0 {
        anyhow::bail!("No expert data found in tuning files (version 2+ required)");
    }

    // Compute top-N experts per layer
    let mut result = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let mut kept = usage.top_n_experts(layer, experts_to_keep);
        kept.sort();
        result.push(kept);
    }

    Ok(result)
}

/// Check if a tensor is an expert weight tensor that should be pruned
fn is_expert_tensor(name: &str) -> bool {
    name.contains(".ffn_gate_exps.")
        || name.contains(".ffn_up_exps.")
        || name.contains(".ffn_down_exps.")
}

/// Per-layer pruning info
struct LayerPruneInfo {
    /// Original expert indices to keep, sorted
    kept_experts: Vec<u16>,
    /// Mask: 0.0 for kept, -inf for pruned (shape: [n_experts_original])
    mask: Vec<f32>,
    /// Remap: new_index for each original index (shape: [n_experts_original])
    /// For pruned experts, value is 0 (doesn't matter since mask prevents selection)
    remap: Vec<u16>,
}

/// Tensor to be written (for expert pruning)
enum ExpertOutputTensor {
    /// Copy from source file (with offset and size)
    Copy {
        name: String,
        source_offset: u64,
        size_in_bytes: usize,
        ggml_dtype: GgmlDType,
        dims: Vec<usize>,
    },
    /// Pruned expert tensor - copy only kept experts
    PrunedExperts {
        name: String,
        source_offset: u64,
        bytes_per_expert: usize,
        kept_indices: Vec<u16>,
        ggml_dtype: GgmlDType,
        #[allow(dead_code)]
        original_dims: Vec<usize>,
        new_dims: Vec<usize>,
    },
    /// New F32 tensor (mask or remap)
    NewF32 { name: String, data: Vec<f32> },
}

impl ExpertOutputTensor {
    fn name(&self) -> &str {
        match self {
            ExpertOutputTensor::Copy { name, .. } => name,
            ExpertOutputTensor::PrunedExperts { name, .. } => name,
            ExpertOutputTensor::NewF32 { name, .. } => name,
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            ExpertOutputTensor::Copy { size_in_bytes, .. } => *size_in_bytes,
            ExpertOutputTensor::PrunedExperts {
                bytes_per_expert,
                kept_indices,
                ..
            } => bytes_per_expert * kept_indices.len(),
            ExpertOutputTensor::NewF32 { data, .. } => data.len() * 4,
        }
    }

    fn ggml_dtype(&self) -> GgmlDType {
        match self {
            ExpertOutputTensor::Copy { ggml_dtype, .. } => *ggml_dtype,
            ExpertOutputTensor::PrunedExperts { ggml_dtype, .. } => *ggml_dtype,
            ExpertOutputTensor::NewF32 { .. } => GgmlDType::F32,
        }
    }

    fn dims(&self) -> &[usize] {
        match self {
            ExpertOutputTensor::Copy { dims, .. } => dims,
            ExpertOutputTensor::PrunedExperts { new_dims, .. } => new_dims,
            ExpertOutputTensor::NewF32 { data, .. } => {
                // 1D tensor, but we need to return a slice
                // This is a bit of a hack - we store dims inline
                unsafe { std::slice::from_raw_parts(&data.len() as *const usize, 1) }
            }
        }
    }
}

/// Prune MoE experts from a GGUF model using explicit per-layer retained indices.
pub fn prune_experts(options: &PruneExpertsOptions) -> Result<()> {
    info!("Prune MoE Experts Tool");
    info!("======================");

    // Step 1: Read input GGUF to get model config
    info!("Reading input model: {:?}", options.input_model);
    let mut input_file = BufReader::new(
        File::open(&options.input_model)
            .with_context(|| format!("Failed to open input model: {:?}", options.input_model))?,
    );
    let input_content = gguf_file::Content::read(&mut input_file)
        .with_context(|| "Failed to read input GGUF content")?;

    info!(
        "  {} tensors, {} metadata entries",
        input_content.tensor_infos.len(),
        input_content.metadata.len()
    );

    // Get expert count from model metadata
    let n_experts = get_u32_metadata(
        &input_content.metadata,
        &[
            "qwen3next.expert_count",
            "qwen3moe.expert_count",
            "llama.expert_count",
        ],
    )
    .ok_or_else(|| anyhow::anyhow!("Could not find expert_count in model metadata"))?
        as usize;

    let n_layers = get_u32_metadata(
        &input_content.metadata,
        &[
            "qwen3next.block_count",
            "qwen3moe.block_count",
            "llama.block_count",
        ],
    )
    .ok_or_else(|| anyhow::anyhow!("Could not find block_count in model metadata"))?
        as usize;

    if options.retained_indices.len() != n_layers {
        anyhow::bail!(
            "retained_indices has {} entries but model has {} layers",
            options.retained_indices.len(),
            n_layers
        );
    }

    let experts_to_keep = options
        .retained_indices
        .first()
        .map(|v| v.len())
        .unwrap_or(0);

    info!(
        "  Model: {} layers, {} experts/layer, keeping ~{} per layer",
        n_layers, n_experts, experts_to_keep
    );

    // Step 2: Build pruning info per layer from retained_indices
    info!("Building expert pruning info...");

    let mut layer_prune_info: Vec<LayerPruneInfo> = Vec::with_capacity(n_layers);

    for (layer, kept_experts) in options.retained_indices.iter().enumerate() {
        // Build mask and remap
        let mut mask = vec![f32::NEG_INFINITY; n_experts];
        let mut remap = vec![0u16; n_experts];

        for (new_idx, &orig_idx) in kept_experts.iter().enumerate() {
            mask[orig_idx as usize] = 0.0;
            remap[orig_idx as usize] = new_idx as u16;
        }

        if options.verbose {
            debug!("  Layer {:2}: retaining experts {:?}", layer, kept_experts);
        }

        layer_prune_info.push(LayerPruneInfo {
            kept_experts: kept_experts.clone(),
            mask,
            remap,
        });
    }

    // Step 3: Build output tensor list
    info!("Building output tensor list...");

    let mut output_tensors: Vec<ExpertOutputTensor> = Vec::new();
    let mut tensor_names: Vec<_> = input_content.tensor_infos.keys().cloned().collect();
    tensor_names.sort();

    let mut expert_tensors_pruned = 0;
    let mut original_expert_bytes = 0usize;
    let mut pruned_expert_bytes = 0usize;

    for name in &tensor_names {
        let info = &input_content.tensor_infos[name];

        if is_expert_tensor(name) {
            if let Some(layer_idx) = extract_layer_index(name) {
                if layer_idx < n_layers {
                    let prune_info = &layer_prune_info[layer_idx];

                    // Expert tensor shape: [n_experts, dim1, dim2, ...]
                    let dims = info.shape.dims();
                    if dims.is_empty() || dims[0] != n_experts {
                        anyhow::bail!(
                            "Unexpected expert tensor shape for {}: {:?}, expected first dim = {}",
                            name,
                            dims,
                            n_experts
                        );
                    }

                    // Calculate bytes per expert
                    let total_bytes = tensor_size_in_bytes(info);
                    let bytes_per_expert = total_bytes / n_experts;

                    // New dims with reduced expert count
                    let layer_keep = prune_info.kept_experts.len();
                    let mut new_dims = dims.to_vec();
                    new_dims[0] = layer_keep;

                    original_expert_bytes += total_bytes;
                    pruned_expert_bytes += bytes_per_expert * layer_keep;
                    expert_tensors_pruned += 1;

                    output_tensors.push(ExpertOutputTensor::PrunedExperts {
                        name: name.clone(),
                        source_offset: input_content.tensor_data_offset + info.offset,
                        bytes_per_expert,
                        kept_indices: prune_info.kept_experts.clone(),
                        ggml_dtype: info.ggml_dtype,
                        original_dims: dims.to_vec(),
                        new_dims,
                    });
                    continue;
                }
            }
        }

        // Non-expert tensor: copy as-is
        output_tensors.push(ExpertOutputTensor::Copy {
            name: name.clone(),
            source_offset: input_content.tensor_data_offset + info.offset,
            size_in_bytes: tensor_size_in_bytes(info),
            ggml_dtype: info.ggml_dtype,
            dims: info.shape.dims().to_vec(),
        });
    }

    // Add mask and remap tensors for each layer
    for (layer, prune_info) in layer_prune_info.iter().enumerate() {
        output_tensors.push(ExpertOutputTensor::NewF32 {
            name: format!("blk.{}.expert_mask", layer),
            data: prune_info.mask.clone(),
        });
        // Store remap as F32 so dequantization works correctly
        output_tensors.push(ExpertOutputTensor::NewF32 {
            name: format!("blk.{}.expert_remap", layer),
            data: prune_info.remap.iter().map(|&x| x as f32).collect(),
        });
    }

    // Sort by name for deterministic output
    output_tensors.sort_by(|a, b| a.name().cmp(b.name()));

    info!("  {} expert tensors will be pruned", expert_tensors_pruned);
    info!(
        "  Expert data: {:.2} GB -> {:.2} GB ({:.1}% reduction)",
        original_expert_bytes as f64 / 1e9,
        pruned_expert_bytes as f64 / 1e9,
        100.0 * (1.0 - pruned_expert_bytes as f64 / original_expert_bytes as f64)
    );

    // Step 5: Write output GGUF
    if let Some(parent) = options.output_model.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    info!("Writing output to: {:?}", options.output_model);

    let output_file = File::create(&options.output_model)?;
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, output_file);

    // Prepare metadata - keep expert_count unchanged (router still outputs n_experts logits)
    // The mask and remap tensors handle conversion to pruned expert indices
    let mut metadata: Vec<(&str, Value)> = Vec::new();
    for (key, value) in input_content.metadata.iter() {
        metadata.push((key.as_str(), value.clone()));
    }

    // Add pruning metadata so model knows the pruned tensor sizes
    metadata.push((
        "expert_pruning.original_experts",
        Value::U32(n_experts as u32),
    ));
    metadata.push((
        "expert_pruning.kept_experts",
        Value::U32(experts_to_keep as u32),
    ));

    metadata.sort_by_key(|(k, _)| *k);

    // Write header
    output.write_u32::<LittleEndian>(0x46554747)?; // "GGUF" magic
    output.write_u32::<LittleEndian>(3)?; // Version 3
    output.write_u64::<LittleEndian>(output_tensors.len() as u64)?;
    output.write_u64::<LittleEndian>(metadata.len() as u64)?;

    // Write metadata
    for (key, value) in &metadata {
        write_string(&mut output, key)?;
        output.write_u32::<LittleEndian>(value_type_to_u32(value.value_type()))?;
        write_value(&mut output, value)?;
    }

    // Write tensor headers
    info!("Writing {} tensor headers...", output_tensors.len());
    let mut data_offset: usize = 0;

    // We need to store dims separately for NewF32 tensors
    let mut tensor_dims: Vec<Vec<usize>> = Vec::new();

    for tensor in &output_tensors {
        write_string(&mut output, tensor.name())?;

        let dims: Vec<usize> = match tensor {
            ExpertOutputTensor::NewF32 { data, .. } => vec![data.len()],
            _ => tensor.dims().to_vec(),
        };
        tensor_dims.push(dims.clone());

        output.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims.iter().rev() {
            output.write_u64::<LittleEndian>(dim as u64)?;
        }
        output.write_u32::<LittleEndian>(ggml_dtype_to_u32(tensor.ggml_dtype()))?;
        output.write_u64::<LittleEndian>(data_offset as u64)?;

        let size = tensor.size_in_bytes();
        let padding = 31 - (31 + size) % 32;
        data_offset += size + padding;
    }

    // Pad header to 32-byte alignment
    let pos = output.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    output.write_all(&vec![0u8; padding])?;

    // Write tensor data
    info!("Streaming tensor data...");
    const BUFFER_SIZE: usize = 8 * 1024 * 1024;
    let mut buffer = vec![0u8; BUFFER_SIZE];

    let total_tensors = output_tensors.len();
    for (idx, tensor) in output_tensors.iter().enumerate() {
        match tensor {
            ExpertOutputTensor::Copy {
                source_offset,
                size_in_bytes,
                ..
            } => {
                input_file.seek(SeekFrom::Start(*source_offset))?;
                let mut remaining = *size_in_bytes;
                while remaining > 0 {
                    let to_read = remaining.min(BUFFER_SIZE);
                    input_file.read_exact(&mut buffer[..to_read])?;
                    output.write_all(&buffer[..to_read])?;
                    remaining -= to_read;
                }
            }
            ExpertOutputTensor::PrunedExperts {
                source_offset,
                bytes_per_expert,
                kept_indices,
                ..
            } => {
                // For each kept expert, seek to its data and copy
                for &expert_idx in kept_indices {
                    let expert_offset =
                        source_offset + (expert_idx as u64 * *bytes_per_expert as u64);
                    input_file.seek(SeekFrom::Start(expert_offset))?;

                    let mut remaining = *bytes_per_expert;
                    while remaining > 0 {
                        let to_read = remaining.min(BUFFER_SIZE);
                        input_file.read_exact(&mut buffer[..to_read])?;
                        output.write_all(&buffer[..to_read])?;
                        remaining -= to_read;
                    }
                }
            }
            ExpertOutputTensor::NewF32 { data, .. } => {
                for &val in data {
                    output.write_f32::<LittleEndian>(val)?;
                }
            }
        }

        // Write padding
        let size = tensor.size_in_bytes();
        let padding = 31 - (31 + size) % 32;
        if padding > 0 {
            output.write_all(&vec![0u8; padding])?;
        }

        if (idx + 1) % 100 == 0 || idx + 1 == total_tensors {
            debug!("  Streamed {}/{} tensors", idx + 1, total_tensors);
        }
    }

    output.flush()?;

    let output_size = std::fs::metadata(&options.output_model)?.len();
    let input_size = std::fs::metadata(&options.input_model)?.len();

    info!("Success!");
    info!("  Input:  {:.2} GB", input_size as f64 / 1e9);
    info!(
        "  Output: {:.2} GB ({:.1}% of original)",
        output_size as f64 / 1e9,
        100.0 * output_size as f64 / input_size as f64
    );
    info!("  Experts per layer: {} -> {}", n_experts, experts_to_keep);
    info!("The pruned model includes expert_mask and expert_remap tensors.");
    info!("Model code must apply these during inference.");

    Ok(())
}

// ============================================================================
// Layer pruning
// ============================================================================

/// Options for pruning transformer layers from a GGUF model.
pub struct PruneLayersOptions {
    /// Path to input GGUF model file
    pub input_model: PathBuf,
    /// Path for output GGUF with pruned layers
    pub output_model: PathBuf,
    /// Layer indices to retain (sorted). Layers are renumbered 0..N in output.
    pub retained_layers: Vec<u32>,
    /// Show verbose output
    pub verbose: bool,
}

/// Check if a tensor belongs to a specific layer (blk.N.*)
fn is_layer_tensor(name: &str) -> bool {
    name.contains("blk.")
}

/// Tensor to be written (for layer pruning)
struct LayerOutputTensor {
    name: String,
    source_offset: u64,
    size_in_bytes: usize,
    ggml_dtype: GgmlDType,
    dims: Vec<usize>,
}

/// Renumber a tensor name from `blk.{old_idx}.*` to `blk.{new_idx}.*`.
fn renumber_layer_tensor(name: &str, old_idx: usize, new_idx: usize) -> String {
    let old_prefix = format!("blk.{}.", old_idx);
    let new_prefix = format!("blk.{}.", new_idx);
    name.replacen(&old_prefix, &new_prefix, 1)
}

/// Prune transformer layers from a GGUF model, keeping the specified layers
/// and renumbering them contiguously starting from 0.
pub fn prune_layers(options: &PruneLayersOptions) -> Result<()> {
    info!("Prune Layers Tool");
    info!("=================");

    // Step 1: Read input GGUF to get model config
    info!("Reading input model: {:?}", options.input_model);
    let mut input_file = BufReader::new(
        File::open(&options.input_model)
            .with_context(|| format!("Failed to open input model: {:?}", options.input_model))?,
    );
    let input_content = gguf_file::Content::read(&mut input_file)
        .with_context(|| "Failed to read input GGUF content")?;

    info!(
        "  {} tensors, {} metadata entries",
        input_content.tensor_infos.len(),
        input_content.metadata.len()
    );

    // Get layer count from model metadata
    let n_layers = get_u32_metadata(
        &input_content.metadata,
        &[
            "qwen3next.block_count",
            "qwen3moe.block_count",
            "llama.block_count",
        ],
    )
    .ok_or_else(|| anyhow::anyhow!("Could not find block_count in model metadata"))?
        as usize;

    info!("  Original model: {} layers", n_layers);

    // Validate retained_layers
    if options.retained_layers.is_empty() {
        anyhow::bail!("Must retain at least 1 layer (got empty list)");
    }

    for &idx in &options.retained_layers {
        if idx as usize >= n_layers {
            anyhow::bail!(
                "Layer index {} is out of range (model has {} layers)",
                idx,
                n_layers
            );
        }
    }

    let layers_to_keep = options.retained_layers.len();

    // Build old→new layer index mapping
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_idx) in options.retained_layers.iter().enumerate() {
        old_to_new.insert(old_idx as usize, new_idx);
    }

    info!(
        "  Keeping {} layers: {:?}",
        layers_to_keep, options.retained_layers
    );

    // Step 2: Filter and renumber tensors
    info!("Filtering and renumbering tensors...");

    let mut output_tensors: Vec<LayerOutputTensor> = Vec::new();
    let mut tensor_names: Vec<_> = input_content.tensor_infos.keys().cloned().collect();
    tensor_names.sort();

    let mut kept_count = 0;
    let mut pruned_count = 0;
    let mut kept_bytes = 0usize;
    let mut pruned_bytes = 0usize;

    for name in &tensor_names {
        let info = &input_content.tensor_infos[name];
        let size = tensor_size_in_bytes(info);

        if is_layer_tensor(name) {
            if let Some(layer_idx) = extract_layer_index(name) {
                if let Some(&new_idx) = old_to_new.get(&layer_idx) {
                    // Retained layer — renumber
                    let new_name = renumber_layer_tensor(name, layer_idx, new_idx);
                    output_tensors.push(LayerOutputTensor {
                        name: new_name,
                        source_offset: input_content.tensor_data_offset + info.offset,
                        size_in_bytes: size,
                        ggml_dtype: info.ggml_dtype,
                        dims: info.shape.dims().to_vec(),
                    });
                    kept_count += 1;
                    kept_bytes += size;
                } else {
                    if options.verbose {
                        debug!("  Pruning: {} (layer {})", name, layer_idx);
                    }
                    pruned_count += 1;
                    pruned_bytes += size;
                }
            } else {
                // Couldn't parse layer index, keep it to be safe
                warn!("Could not parse layer index from: {}", name);
                output_tensors.push(LayerOutputTensor {
                    name: name.clone(),
                    source_offset: input_content.tensor_data_offset + info.offset,
                    size_in_bytes: size,
                    ggml_dtype: info.ggml_dtype,
                    dims: info.shape.dims().to_vec(),
                });
                kept_count += 1;
                kept_bytes += size;
            }
        } else {
            // Non-layer tensor (embedding, norm, MTP head, etc.) - always keep
            output_tensors.push(LayerOutputTensor {
                name: name.clone(),
                source_offset: input_content.tensor_data_offset + info.offset,
                size_in_bytes: size,
                ggml_dtype: info.ggml_dtype,
                dims: info.shape.dims().to_vec(),
            });
            kept_count += 1;
            kept_bytes += size;
        }
    }

    // Re-sort by the (renumbered) name for deterministic output
    output_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    info!(
        "  Kept: {} tensors ({:.2} GB)",
        kept_count,
        kept_bytes as f64 / 1e9
    );
    info!(
        "  Pruned: {} tensors ({:.2} GB)",
        pruned_count,
        pruned_bytes as f64 / 1e9
    );

    // Step 3: Prepare metadata with updated block_count
    info!("Preparing metadata...");

    let mut metadata: Vec<(String, Value)> = Vec::new();
    let mut updated_block_count = false;

    for (key, value) in input_content.metadata.iter() {
        // Update block_count to reflect the pruned layer count
        if key == "qwen3next.block_count"
            || key == "qwen3moe.block_count"
            || key == "llama.block_count"
        {
            metadata.push((key.clone(), Value::U32(layers_to_keep as u32)));
            updated_block_count = true;
            info!("  Updated {}: {} -> {}", key, n_layers, layers_to_keep);
        } else {
            metadata.push((key.clone(), value.clone()));
        }
    }

    if !updated_block_count {
        warn!("Could not find block_count key to update");
    }

    // Add pruning metadata
    metadata.push((
        "layer_pruning.original_layers".to_string(),
        Value::U32(n_layers as u32),
    ));
    metadata.push((
        "layer_pruning.kept_layers".to_string(),
        Value::U32(layers_to_keep as u32),
    ));

    metadata.sort_by_key(|(k, _)| k.clone());

    // Step 4: Write output GGUF
    if let Some(parent) = options.output_model.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    info!("Writing output to: {:?}", options.output_model);

    let output_file = File::create(&options.output_model)?;
    let mut output = BufWriter::with_capacity(8 * 1024 * 1024, output_file);

    // Write header
    output.write_u32::<LittleEndian>(0x46554747)?; // "GGUF" magic
    output.write_u32::<LittleEndian>(3)?; // Version 3
    output.write_u64::<LittleEndian>(output_tensors.len() as u64)?;
    output.write_u64::<LittleEndian>(metadata.len() as u64)?;

    // Write metadata
    for (key, value) in &metadata {
        write_string(&mut output, key)?;
        output.write_u32::<LittleEndian>(value_type_to_u32(value.value_type()))?;
        write_value(&mut output, value)?;
    }

    // Write tensor headers
    info!("Writing {} tensor headers...", output_tensors.len());
    let mut data_offset: usize = 0;

    for tensor in &output_tensors {
        write_string(&mut output, &tensor.name)?;

        output.write_u32::<LittleEndian>(tensor.dims.len() as u32)?;
        for &dim in tensor.dims.iter().rev() {
            output.write_u64::<LittleEndian>(dim as u64)?;
        }
        output.write_u32::<LittleEndian>(ggml_dtype_to_u32(tensor.ggml_dtype))?;
        output.write_u64::<LittleEndian>(data_offset as u64)?;

        let size = tensor.size_in_bytes;
        let padding = 31 - (31 + size) % 32;
        data_offset += size + padding;
    }

    // Pad header to 32-byte alignment
    let pos = output.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    output.write_all(&vec![0u8; padding])?;

    // Write tensor data
    info!("Streaming tensor data...");
    const BUFFER_SIZE: usize = 8 * 1024 * 1024;
    let mut buffer = vec![0u8; BUFFER_SIZE];

    let total_tensors = output_tensors.len();
    for (idx, tensor) in output_tensors.iter().enumerate() {
        input_file.seek(SeekFrom::Start(tensor.source_offset))?;
        let mut remaining = tensor.size_in_bytes;
        while remaining > 0 {
            let to_read = remaining.min(BUFFER_SIZE);
            input_file.read_exact(&mut buffer[..to_read])?;
            output.write_all(&buffer[..to_read])?;
            remaining -= to_read;
        }

        // Write padding
        let padding = 31 - (31 + tensor.size_in_bytes) % 32;
        if padding > 0 {
            output.write_all(&vec![0u8; padding])?;
        }

        if (idx + 1) % 100 == 0 || idx + 1 == total_tensors {
            debug!("  Streamed {}/{} tensors", idx + 1, total_tensors);
        }
    }

    output.flush()?;

    let output_size = std::fs::metadata(&options.output_model)?.len();
    let input_size = std::fs::metadata(&options.input_model)?.len();

    info!("Success!");
    info!("  Input:  {:.2} GB", input_size as f64 / 1e9);
    info!(
        "  Output: {:.2} GB ({:.1}% of original)",
        output_size as f64 / 1e9,
        100.0 * output_size as f64 / input_size as f64
    );
    info!("  Layers: {} -> {}", n_layers, layers_to_keep);
    info!(
        "The pruned model has {} layers removed.",
        n_layers - layers_to_keep
    );

    Ok(())
}
