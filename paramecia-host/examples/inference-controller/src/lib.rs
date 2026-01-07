//! Inference WASM controller component for paramecia.
//!
//! This demonstrates the inference interface:
//! 1. Loads a model via structure::load-model
//! 2. Fills context with a hardcoded prompt (100 facts about Rust)
//! 3. Uses predict-completion for concurrent streaming generation
//! 4. Cancels the completion after receiving 100 tokens
//! 5. Unloads the model and prints stats
//!
//! Shows both concurrent streaming and cancellation capabilities.

mod bindings {
    include!(concat!(env!("OUT_DIR"), "/paramecia_bindgen.rs"));
}

use bindings::exports::wasi::cli::run::Guest;
use bindings::paramecia::controller::inference;
use bindings::paramecia::controller::structure;
use bindings::paramecia::controller::types::{ModelInput, Weights};
use bindings::wasi::clocks::monotonic_clock;

const FACTS: &[&str] = &[
    "Rust is fast",
    "Rust is safe",
    "Rust has cool types",
    "Rust is difficult",
];

struct InferenceController;

impl Guest for InferenceController {
    async fn run() -> Result<(), ()> {
        // Load a model first
        let model = structure::load_model(Weights::HostDefault)
            .await
            .map_err(|_| ())?;

        let mut prompt = String::from("<|im_start|>system\nHere are 100 key points about Rust:\n");
        for i in 0..100 {
            prompt.push_str("- ");
            prompt.push_str(FACTS[i % FACTS.len()]);
            prompt.push('\n');
        }
        prompt.push_str("<|im_end|>\n<|im_start|>user\nWhat is the Rust programming language?<|im_end|>\n<|im_start|>assistant\n");

        // Fill context with the prompt and read progress from the stream
        let prefill_start = monotonic_clock::now();
        let mut prefill_tokens = 0u32;
        let mut stream = inference::fill_context(model.clone(), &[ModelInput::Text(prompt)]);
        while let Some(progress) = stream.next().await {
            match progress {
                Ok(n) => prefill_tokens = n,
                Err(_e) => return Err(()),
            }
        }
        let prefill_ns = monotonic_clock::now() - prefill_start;

        // Use predict-completion for concurrent streaming generation.
        // Cancel after receiving 100 tokens to demonstrate cancellation.
        let (tx, rx) = async_channel::unbounded::<String>();

        let gen_start = monotonic_clock::now();
        let mut completion_stream = inference::predict_completion(model.clone());
        let mut cancelled = false;

        // Writer task: streams output to stdout as it arrives
        let writer = async move {
            while let Ok(text) = rx.recv().await {
                print_stdout(text.as_bytes()).await;
            }
        };

        // Reader task: drains completion stream at max speed and sends to channel
        let model_for_cancel = model.clone();
        let reader = async move {
            let mut gen_tokens = 0u32;
            while let Some(result) = completion_stream.next().await {
                match result {
                    Ok(predicted) => {
                        gen_tokens += 1;

                        // Send decoded text to writer (non-blocking)
                        if let Some(text) = &predicted.text {
                            let _ = tx.send(text.clone()).await;
                        }

                        // Cancel after 100 tokens
                        if gen_tokens >= 100 && !cancelled {
                            let _ = tx
                                .send("\n[Cancelling after 100 tokens...]\n".to_string())
                                .await;
                            inference::cancel_prediction(model_for_cancel.clone())
                                .await
                                .map_err(|_| ())?;
                            cancelled = true;
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(format!("\n[Stream ended with error: {}]\n", e))
                            .await;
                        break;
                    }
                }
            }
            drop(tx);
            Ok::<_, ()>(gen_tokens)
        };

        // Run both concurrently
        let (reader_result, _) = futures::join!(reader, writer);
        let gen_tokens = reader_result?;

        let gen_ns = monotonic_clock::now() - gen_start;

        // Print stats
        let prefill_secs = prefill_ns as f64 / 1_000_000_000.0;
        let gen_secs = gen_ns as f64 / 1_000_000_000.0;
        let prefill_rate = if prefill_secs > 0.0 {
            prefill_tokens as f64 / prefill_secs
        } else {
            0.0
        };
        let gen_rate = if gen_secs > 0.0 {
            gen_tokens as f64 / gen_secs
        } else {
            0.0
        };

        let stats = format!(
            "\n\n--- Stats ---\nPrefill: {} tokens in {:.2}s ({:.1} tok/s)\nGeneration: {} tokens in {:.2}s ({:.1} tok/s)\n",
            prefill_tokens, prefill_secs, prefill_rate,
            gen_tokens, gen_secs, gen_rate,
        );
        print_stdout(stats.as_bytes()).await;

        // Unload the model
        let _ = structure::unload_model(model.clone()).await;

        Ok(())
    }
}

/// Write bytes to WASI stdout via stream.
async fn print_stdout(data: &[u8]) {
    let (mut tx, rx) = bindings::wit_stream::new::<u8>();
    futures::join!(
        async {
            let _ = bindings::wasi::cli::stdout::write_via_stream(rx).await;
        },
        async {
            let _ = tx.write_all(data.to_vec()).await;
            drop(tx);
        }
    );
}

bindings::export!(InferenceController with_types_in bindings);
