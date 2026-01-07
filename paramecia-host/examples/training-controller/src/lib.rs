//! Training controller example for paramecia.
//!
//! Demonstrates combined inference + training orchestration:
//! 1. Loads two models via structure::load-model (inference + training)
//! 2. Prefills the inference model with a system prompt + user question
//! 3. Takes a snapshot of the inference state
//! 4. Runs inference loop and training actor concurrently:
//!    - Inference: generate tokens, check acceptance, send to channel
//!    - Training actor: forwards samples to host, paced by step completions
//!    - Forwarder tracks in-flight samples and blocks when buffer is full
//! 5. Stops after TARGET_TRAIN_STEPS training steps
//!
//! Prints loss from each training step and accept/reject decisions.

mod bindings {
    include!(concat!(env!("OUT_DIR"), "/paramecia_bindgen.rs"));
}

use bindings::exports::wasi::cli::run::Guest;
use bindings::paramecia::controller::inference;
use bindings::paramecia::controller::structure;
use bindings::paramecia::controller::structure_ext;
use bindings::paramecia::controller::training;
use bindings::paramecia::controller::types::{
    Model, ModelInput, Predicted, TrainingData, TrainingSample, Weights,
};
use bindings::wasi::clocks::monotonic_clock;

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

struct TrainingController;

const SYSTEM_PROMPT: &str = "Welcome to The Rust Programming Language, an introductory \
book about Rust. The Rust programming language helps you write faster, more reliable \
software. High-level ergonomics and low-level control are often at odds in programming \
language design; Rust challenges that conflict. Through balancing powerful technical \
capacity and a great developer experience, Rust gives you the option to control low-level \
details (such as memory usage) without all the hassle traditionally associated with such \
control.\n\n\
Rust is for people who crave speed and stability in a language. By speed, we mean both \
how quickly Rust code can run and the speed at which Rust lets you write programs. The \
Rust compiler's checks ensure stability through feature additions and refactoring. This \
is in contrast to the brittle legacy code in languages without these checks, which \
developers are often afraid to modify. By striving for zero-cost abstractions\u{2014}higher-level \
features that compile to lower-level code as fast as code written manually\u{2014}Rust endeavors \
to make safe code be fast code as well.";

const USER_PROMPT: &str = "What is Rust?";

const MAX_TOKENS: u32 = 20;
const TARGET_TRAIN_STEPS: u32 = 2;

/// How many samples the host consumes per training step.
/// Must match host config: minibatch_size (2) * n_grad_steps (2) = 4.
const EFFECTIVE_BATCH: usize = 4;

/// Max samples the forwarder will send ahead of what training has consumed.
const SAMPLE_BUFFER_SIZE: usize = 8;

impl Guest for TrainingController {
    async fn run() -> Result<(), ()> {
        // --- Phase 1: Load separate inference and training models ---
        print_line("[1] Loading inference model...").await;
        let inference_model = structure::load_model(Weights::HostDefault)
            .await
            .map_err(|e| {
                // Can't use print_line in map_err easily, just drop
                let _ = e;
            })?;
        print_line("[1] Inference model loaded.").await;

        print_line("[1] Loading training model...").await;
        let training_model = structure::load_model(Weights::HostDefault)
            .await
            .map_err(|e| {
                let _ = e;
            })?;
        print_line("[1] Training model ready.").await;

        // --- Phase 2: Prefill inference model ---
        print_line("[2] Prefilling model...").await;

        let prompt = format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
             <|im_start|>user\n{USER_PROMPT}<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        let prefill_start = monotonic_clock::now();
        let mut prefill_stream =
            inference::fill_context(inference_model.clone(), &[ModelInput::Text(prompt)]);
        let mut total_tokens = 0u32;
        while let Some(progress) = prefill_stream.next().await {
            match progress {
                Ok(n) => {
                    total_tokens = n;
                    print_line(&format!("[2] Prefill progress: {n} tokens")).await;
                }
                Err(e) => {
                    print_line(&format!("[ERROR] fill-context: {e:?}")).await;
                    return Err(());
                }
            }
        }

        let prefill_ns = monotonic_clock::now() - prefill_start;
        let prefill_secs = prefill_ns as f64 / 1_000_000_000.0;
        let prefill_rate = if prefill_secs > 0.0 {
            total_tokens as f64 / prefill_secs
        } else {
            0.0
        };
        print_line(&format!(
            "[2] Prefill complete: {total_tokens} tokens in {prefill_secs:.2}s ({prefill_rate:.1} tok/s)."
        ))
        .await;

        // --- Phase 3: Save snapshot ---
        print_line("[3] Taking snapshot...").await;
        let snapshot = match structure_ext::take_snapshot(inference_model.clone()).await {
            Ok(s) => s,
            Err(e) => {
                print_line(&format!("[ERROR] take-snapshot: {e:?}")).await;
                return Err(());
            }
        };
        print_line(&format!(
            "[3] Snapshot taken (pos={}, tokens={}).",
            snapshot.state_position, snapshot.num_tokens
        ))
        .await;

        // --- Phase 4: Inference loop + training actor in parallel ---
        let context_prefix =
            format!("<|im_start|>user\n{USER_PROMPT}<|im_end|>\n<|im_start|>assistant\n");

        let (sample_sender, sample_receiver) = async_channel::unbounded::<TrainingSample>();

        static NEXT_SAMPLE_ID: AtomicU64 = AtomicU64::new(1);
        NEXT_SAMPLE_ID.store(1, Ordering::Relaxed);

        static STEPS_COMPLETED: AtomicUsize = AtomicUsize::new(0);
        STEPS_COMPLETED.store(0, Ordering::Relaxed);

        let (step_notify_tx, step_notify_rx) = async_channel::unbounded::<()>();

        // Training actor
        let model_for_training = training_model.clone();
        let training_actor = async {
            let (mut sample_tx, sample_rx) = bindings::wit_stream::new::<TrainingSample>();

            let train_result = training::train_model(model_for_training.clone(), sample_rx);

            match train_result {
                Ok(mut loss_stream) => {
                    let receiver_for_close = sample_receiver.clone();
                    let model_for_cancel = model_for_training.clone();
                    futures::join!(
                        // Forwarder
                        async {
                            let mut samples_sent: usize = 0;

                            loop {
                                let consumed =
                                    STEPS_COMPLETED.load(Ordering::Relaxed) * EFFECTIVE_BATCH;
                                let in_flight = samples_sent.saturating_sub(consumed);

                                if in_flight >= SAMPLE_BUFFER_SIZE {
                                    if step_notify_rx.recv().await.is_err() {
                                        break;
                                    }
                                    continue;
                                }

                                match sample_receiver.recv().await {
                                    Ok(sample) => {
                                        if sample_tx.write_one(sample).await.is_some() {
                                            break;
                                        }
                                        samples_sent += 1;
                                    }
                                    Err(_) => break,
                                }
                            }
                            drop(sample_tx);
                        },
                        // Loss reader
                        async {
                            let mut step_count = 0u32;
                            while let Some(step_result) = loss_stream.next().await {
                                match step_result {
                                    Ok(result) => {
                                        step_count += 1;
                                        STEPS_COMPLETED
                                            .store(step_count as usize, Ordering::Relaxed);
                                        let _ = step_notify_tx.send(()).await;
                                        let ids: Vec<u64> =
                                            result.sample_ids.iter().map(|&(_, lo)| lo).collect();
                                        print_line(&format!(
                                            "  [Train] Step {step_count}/{TARGET_TRAIN_STEPS}: loss = {:.6}, samples = {ids:?}",
                                            result.loss,
                                        ))
                                        .await;
                                        if step_count >= TARGET_TRAIN_STEPS {
                                            print_line("  [Train] Finished, canceling...").await;
                                            let _ =
                                                training::cancel_training(model_for_cancel.clone())
                                                    .await;
                                            receiver_for_close.close();
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        print_line(&format!("  [Train] Error: {e:?}")).await;
                                        let _ = training::cancel_training(model_for_cancel.clone())
                                            .await;
                                        receiver_for_close.close();
                                        break;
                                    }
                                }
                            }
                            drop(step_notify_tx);
                            print_line(&format!("[Train] Completed {step_count} training steps."))
                                .await;
                        }
                    );
                }
                Err(e) => {
                    drop(sample_tx);
                    print_line(&format!("[Train] train-model failed: {e:?}")).await;
                }
            }
        };

        // Inference loop
        let model_for_inference = inference_model.clone();
        let inference_loop = async {
            let mut iteration = 0u32;
            let mut accepted_count = 0u32;
            let mut rejected_count = 0u32;

            loop {
                if sample_sender.is_closed() {
                    break;
                }

                iteration += 1;
                print_line(&format!("\n=== Iteration {iteration} ===")).await;

                // Generate completion (up to MAX_TOKENS tokens)
                let mut completion_stream =
                    inference::predict_completion(model_for_inference.clone());
                let mut predictions: Vec<Predicted> = Vec::new();
                let mut text_so_far = String::new();

                while let Some(result) = completion_stream.next().await {
                    match result {
                        Ok(predicted) => {
                            if let Some(ref text) = predicted.text {
                                text_so_far.push_str(text);
                            }
                            predictions.push(predicted);

                            if predictions.len() as u32 >= MAX_TOKENS {
                                let _ =
                                    inference::cancel_prediction(model_for_inference.clone()).await;
                                break;
                            }
                        }
                        Err(e) => {
                            print_line(&format!("  [Infer] Stream error: {e:?}")).await;
                            break;
                        }
                    }
                }

                print_line(&format!(
                    "  Generated {n} tokens: \"{text}\"",
                    n = predictions.len(),
                    text = truncate_str(&text_so_far, 120),
                ))
                .await;

                let lower = text_so_far.to_lowercase();
                let accepted = lower.contains("fast") || lower.contains("safe");

                if accepted {
                    accepted_count += 1;
                    print_line(&format!(
                        "  >> ACCEPTED [{accepted_count} accepted, {rejected_count} rejected]"
                    ))
                    .await;

                    let id = NEXT_SAMPLE_ID.fetch_add(1, Ordering::Relaxed);
                    let sample = TrainingSample {
                        id: (0, id),
                        data: vec![
                            TrainingData::Context(ModelInput::Text(context_prefix.clone())),
                            TrainingData::Target(predictions),
                        ],
                    };
                    if sample_sender.send(sample).await.is_err() {
                        break;
                    }
                } else {
                    rejected_count += 1;
                    print_line(&format!(
                        "  >> REJECTED [{accepted_count} accepted, {rejected_count} rejected]"
                    ))
                    .await;
                }

                // Restore snapshot for next iteration
                if let Err(e) = structure_ext::restore_snapshot(
                    model_for_inference.clone(),
                    snapshot.clone(),
                )
                .await
                {
                    print_line(&format!("[ERROR] restore-snapshot: {e:?}")).await;
                    break;
                }

                // Reseed so we get a different generation from the same state
                let seed = iteration as u64 * 12345 + 42;
                let _ = structure_ext::reseed(seed).await;
            }

            drop(sample_sender);

            print_line(&format!("\n=== Finished after {iteration} iterations ===")).await;
            print_line(&format!(
                "  Total accepted: {accepted_count}, rejected: {rejected_count}"
            ))
            .await;

            Ok::<_, ()>(())
        };

        let (inference_result, _) = futures::join!(inference_loop, training_actor);
        inference_result?;

        let _ = structure_ext::drop_snapshot(snapshot);
        let _ = structure::unload_model(training_model).await;
        let _ = structure::unload_model(inference_model).await;
        print_line("\nDone.").await;

        Ok(())
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

/// Write a line to WASI stdout via stream.
async fn print_line(msg: &str) {
    let line = format!("{msg}\n");
    let (mut tx, rx) = bindings::wit_stream::new::<u8>();
    futures::join!(
        async {
            let _ = bindings::wasi::cli::stdout::write_via_stream(rx).await;
        },
        async {
            let _ = tx.write_all(line.into_bytes()).await;
            drop(tx);
        }
    );
}

bindings::export!(TrainingController with_types_in bindings);
