//! Inspect GGUF model files: list tensors, show metadata and statistics.

use anyhow::Result;
use clap::Args;
use paramecia_core::quantized::gguf_file;
use paramecia_core::Device;
use std::path::PathBuf;

/// Options for inspecting a GGUF model file.
#[derive(Args, Debug, Clone)]
pub struct InspectOptions {
    /// Path to GGUF file
    #[arg(long)]
    pub model_path: PathBuf,

    /// Filter tensors by name (case-insensitive substring match)
    #[arg(long)]
    pub filter: Option<String>,

    /// Show model metadata (skip tokenizer)
    #[arg(long, default_value = "true")]
    pub show_metadata: bool,

    /// Show tensor value statistics (min, max, mean) for filtered tensors
    #[arg(long, default_value = "false")]
    pub show_stats: bool,
}

/// Create a natural sort key that handles numbers properly
/// e.g., "blk.1" < "blk.2" < "blk.10"
fn natural_sort_key(s: &str) -> Vec<(bool, String)> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut is_digit = false;

    for ch in s.chars() {
        let ch_is_digit = ch.is_ascii_digit();
        if current.is_empty() {
            is_digit = ch_is_digit;
            current.push(ch);
        } else if ch_is_digit == is_digit {
            current.push(ch);
        } else {
            // Convert numbers to zero-padded strings for proper comparison
            if is_digit {
                if let Ok(num) = current.parse::<u32>() {
                    result.push((true, format!("{:010}", num)));
                } else {
                    result.push((false, current.clone()));
                }
            } else {
                result.push((false, current.clone()));
            }
            current.clear();
            current.push(ch);
            is_digit = ch_is_digit;
        }
    }

    // Handle the last segment
    if !current.is_empty() {
        if is_digit {
            if let Ok(num) = current.parse::<u32>() {
                result.push((true, format!("{:010}", num)));
            } else {
                result.push((false, current));
            }
        } else {
            result.push((false, current));
        }
    }

    result
}

/// Inspect a GGUF model file, printing tensor names, metadata, and optional statistics.
pub fn inspect_model(options: &InspectOptions) -> Result<()> {
    let mut file = std::fs::File::open(&options.model_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    println!("GGUF file: {:?}", options.model_path);
    println!("Tensor count: {}", content.tensor_infos.len());

    // Print model metadata only (skip tokenizer which is huge)
    if options.show_metadata {
        println!("\n=== Model Metadata ===");
        let mut keys: Vec<_> = content.metadata.keys().collect();
        keys.sort();
        for key in keys {
            // Skip tokenizer metadata (huge)
            if key.starts_with("tokenizer.ggml.") {
                continue;
            }
            println!("  {}: {:?}", key, content.metadata.get(key).unwrap());
        }
    }

    // Print tensor names
    println!("\n=== Tensors ===");
    let filter = options.filter.as_ref().map(|f| f.to_lowercase());
    let device = Device::Cpu;

    // Sort tensors with natural (human-friendly) ordering
    let mut sorted_tensors: Vec<_> = content.tensor_infos.iter().collect();
    sorted_tensors.sort_by(|(name_a, _), (name_b, _)| {
        natural_sort_key(name_a).cmp(&natural_sort_key(name_b))
    });

    let mut count = 0;
    for (name, info) in sorted_tensors {
        let show = match &filter {
            Some(f) => name.to_lowercase().contains(f.as_str()),
            None => true,
        };

        if show {
            print!(
                "  {} : {:?} ({:?})",
                name,
                info.shape.dims(),
                info.ggml_dtype
            );

            // Show stats if requested
            if options.show_stats {
                if let Ok(qtensor) = content.tensor(&mut file, name, &device) {
                    if let Ok(tensor) = qtensor.dequantize(&device) {
                        let flat = tensor.flatten_all().unwrap();
                        let min = flat.min(0).unwrap().to_vec0::<f32>().unwrap_or(f32::NAN);
                        let max = flat.max(0).unwrap().to_vec0::<f32>().unwrap_or(f32::NAN);
                        let mean = flat
                            .mean_all()
                            .unwrap()
                            .to_vec0::<f32>()
                            .unwrap_or(f32::NAN);
                        print!(" [min={:.4}, max={:.4}, mean={:.4}]", min, max, mean);
                    }
                }
            }
            println!();
            count += 1;
        }
    }

    println!(
        "\nShown {} of {} tensors",
        count,
        content.tensor_infos.len()
    );

    Ok(())
}
