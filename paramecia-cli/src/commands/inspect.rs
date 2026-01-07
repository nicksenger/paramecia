//! Inspect subcommand — inspect tensors in a GGUF model file.

use anyhow::Result;
use paramecia_engine::{GgufTensor, MetadataValue, ModelDescription};

use crate::args::InspectArgs;

pub fn run(args: &InspectArgs) -> Result<()> {
    let desc = paramecia_engine::describe_model(&args.model_path)?;
    print_description(&desc, args);
    Ok(())
}

fn print_description(desc: &ModelDescription, args: &InspectArgs) {
    println!("GGUF file: {:?}", args.model_path);
    println!("Layers: {}", desc.n_layers);
    if desc.n_experts > 0 {
        println!("Experts: {}", desc.n_experts);
    }
    println!(
        "Parameters: {} total, {} active",
        format_param_count(desc.n_total_parameters),
        format_param_count(desc.n_active_parameters),
    );
    println!("Tensor count: {}", desc.tensors.len());

    // Print metadata (skip tokenizer which is huge)
    if args.show_metadata {
        println!("\n=== Model Metadata ===");
        let mut metadata: Vec<_> = desc.metadata.iter().collect();
        metadata.sort_by_key(|(k, _)| k.as_str());
        for (key, value) in metadata {
            if key.starts_with("tokenizer.ggml.") {
                continue;
            }
            println!("  {}: {}", key, format_metadata_value(value));
        }
    }

    // Print tensors
    println!("\n=== Tensors ===");
    let filter = args.filter.as_ref().map(|f| f.to_lowercase());

    let mut sorted_tensors: Vec<&GgufTensor> = desc.tensors.iter().collect();
    sorted_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let mut count = 0;
    for tensor in &sorted_tensors {
        let show = match &filter {
            Some(f) => tensor.name.to_lowercase().contains(f.as_str()),
            None => true,
        };

        if show {
            println!(
                "  {} : {:?} ({:?})",
                tensor.name, tensor.shape, tensor.dtype,
            );
            count += 1;
        }
    }

    println!("\nShown {} of {} tensors", count, desc.tensors.len());
}

fn format_param_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

fn format_metadata_value(v: &MetadataValue) -> String {
    match v {
        MetadataValue::U8(x) => format!("{x}"),
        MetadataValue::I8(x) => format!("{x}"),
        MetadataValue::U16(x) => format!("{x}"),
        MetadataValue::I16(x) => format!("{x}"),
        MetadataValue::U32(x) => format!("{x}"),
        MetadataValue::I32(x) => format!("{x}"),
        MetadataValue::U64(x) => format!("{x}"),
        MetadataValue::I64(x) => format!("{x}"),
        MetadataValue::F32(x) => format!("{x}"),
        MetadataValue::F64(x) => format!("{x}"),
        MetadataValue::Bool(x) => format!("{x}"),
        MetadataValue::String(x) => format!("{x:?}"),
        MetadataValue::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_metadata_value).collect();
            if items.len() <= 5 {
                format!("[{}]", items.join(", "))
            } else {
                format!(
                    "[{}, ... ({} more)]",
                    items[..5].join(", "),
                    items.len() - 5,
                )
            }
        }
    }
}
