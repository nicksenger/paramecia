use crate::Controller;
use crate::convert::to_wit;
use crate::host::HostState;
use crate::paramecia::host::types as wit;
use anyhow::Result;
use paramecia_engine::ModelEngineBuilder;
use paramecia_engine::types as exec;
use std::path::Path;
use wasmtime::component::{Component, HasSelf, Linker, StreamReader};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::p3;

/// Create a wasmtime Engine configured for async component model.
pub fn create_engine() -> Result<Engine> {
    let mut config = Config::new();
    config.async_support(true);
    config.wasm_component_model(true);
    config.wasm_component_model_async(true);
    config.wasm_component_model_async_builtins(true);
    config.wasm_component_model_async_stackful(true);
    Engine::new(&config)
}

/// Create a Linker with WASI p3 and all 6 paramecia interface imports.
pub fn create_linker(engine: &Engine) -> Result<Linker<HostState>> {
    let mut linker = Linker::new(engine);

    // Add WASIp3 imports
    p3::add_to_linker(&mut linker)?;

    // Add all 7 paramecia interfaces
    add_structure_to_linker(&mut linker)?;
    add_structure_ext_to_linker(&mut linker)?;
    add_inference_to_linker(&mut linker)?;
    add_inference_ext_to_linker(&mut linker)?;
    add_training_to_linker(&mut linker)?;
    add_training_ext_to_linker(&mut linker)?;
    add_interactive_to_linker(&mut linker)?;

    Ok(linker)
}

// ---------------------------------------------------------------------------
// structure interface (4 async functions)
// ---------------------------------------------------------------------------

fn add_structure_to_linker(linker: &mut Linker<HostState>) -> Result<()> {
    let mut instance = linker.instance("paramecia:host/structure@0.1.0")?;

    use wasmtime::component::Accessor;

    // load-model: async func(source: weights) -> result<model, error>
    instance.func_wrap_concurrent(
        "load-model",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (source,): (wit::Weights,)| {
            Box::pin(async move {
                let (config, default_path) = accessor.with(|mut data| {
                    let host = data.get();
                    (host.builder_config.clone(), host.default_model_path.clone())
                });

                // Resolve weights to model path
                let model_path = match source {
                    wit::Weights::HostDefault => default_path,
                    wit::Weights::Checkpoint(ckpt) => {
                        let path = accessor.with(|mut data| {
                            let host = data.get();
                            host.checkpoints.get(&ckpt.id).cloned()
                        });
                        match path {
                            Some(p) => p,
                            None => {
                                return Ok((Err::<wit::Model, wit::Error>(
                                    wit::Error::CheckpointError(format!(
                                        "Checkpoint {:?} not found",
                                        ckpt.id
                                    )),
                                ),));
                            }
                        }
                    }
                    wit::Weights::Path(p) => std::path::PathBuf::from(p),
                };
                // Build executor (blocking model load)
                let config_clone = config.clone();
                let model_path_clone = model_path.clone();
                let build_result = tokio::task::spawn_blocking(move || {
                    let mut builder = ModelEngineBuilder::new(&model_path_clone)
                        .cpu(config_clone.cpu)
                        .offload_mode(config_clone.offload_mode)
                        .kv_cache_quant(config_clone.kv_cache_quant)
                        .temperature(config_clone.temperature)
                        .top_k(config_clone.top_k)
                        .tail_samples(config_clone.tail_samples)
                        .seed(config_clone.seed)
                        .snapshot_dir(&config_clone.snapshot_dir)
                        .tokenizer_repo(&config_clone.tokenizer_repo)
                        .repeat_penalty(config_clone.repeat_penalty)
                        .presence_penalty(config_clone.presence_penalty)
                        .penalty_last_n(config_clone.penalty_last_n)
                        .prefetch(config_clone.prefetch)
                        .training_config(config_clone.training_config);

                    if let Some(ref path) = config_clone.tokenizer_path {
                        builder = builder.tokenizer_path(path);
                    }
                    if let Some(ref split) = config_clone.layer_split {
                        builder = builder.layer_split(split);
                    }
                    if let Some(ref p) = config_clone.top_p {
                        builder = builder.top_p(*p);
                    }
                    if let Some(depth) = config_clone.inference_mtp_speculation_depth {
                        builder = builder.enable_mtp_for_inference(depth);
                    }
                    if let Some(depth) = config_clone.training_mtp_speculation_depth {
                        builder = builder.enable_mtp_for_training(depth);
                    }
                    builder = builder
                        .allow_mtp_inference_override(config_clone.mtp_allow_inference_override);

                    builder.build()
                })
                .await;

                let executor = match build_result {
                    Ok(Ok(exec)) => exec,
                    Ok(Err(e)) => {
                        return Ok((Err(wit::Error::ModelError(format!(
                            "Failed to load model: {e}"
                        ))),));
                    }
                    Err(e) => {
                        return Ok((Err(wit::Error::ModelError(format!(
                            "Model load task panicked: {e}"
                        ))),));
                    }
                };

                let num_layers = executor.num_layers();
                let num_experts_per_token = executor.num_experts_per_token();

                // Register in models
                let uuid = accessor.with(|mut data| {
                    let host = data.get();
                    let uuid = host.next_uuid();
                    host.models.insert(
                        uuid,
                        crate::host::ModelEntry {
                            executor,
                            source_path: model_path,
                        },
                    );
                    uuid
                });

                let model = wit::Model {
                    id: uuid,
                    n_experts: num_experts_per_token as u32,
                    n_layers: num_layers as u32,
                };

                Ok((Ok(model),))
            })
        },
    )?;

    // unload-model: async func(model: model) -> result<_, error>
    instance.func_wrap_concurrent(
        "unload-model",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (model,): (wit::Model,)| {
            Box::pin(async move {
                let result = accessor.with(|mut data| {
                    let host = data.get();
                    if host.models.remove(&model.id).is_some() {
                        Ok(())
                    } else {
                        Err(wit::Error::ModelError(format!(
                            "Model {:?} not found",
                            model.id
                        )))
                    }
                });
                Ok((result,))
            })
        },
    )?;

    // save-checkpoint: async func(model: model) -> result<checkpoint, error>
    instance.func_wrap_concurrent(
        "save-checkpoint",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (model,): (wit::Model,)| {
            Box::pin(async move {
                let executor = accessor.with(|mut data| {
                    let host = data.get();
                    host.lookup_model(&model.id)
                        .map(|entry| entry.executor.clone())
                });

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => return Ok((Err(e),)),
                };

                let result = executor.save_checkpoint().await;
                match result {
                    Ok(path) => {
                        let uuid = accessor.with(|mut data| {
                            let host = data.get();
                            let uuid = host.next_uuid();
                            host.checkpoints.insert(uuid, path);
                            uuid
                        });
                        Ok((Ok(wit::Checkpoint { id: uuid }),))
                    }
                    Err(e) => Ok((Err(e.into()),)),
                }
            })
        },
    )?;

    // delete-checkpoint: async func(checkpoint: checkpoint) -> result<_, error>
    instance.func_wrap_concurrent(
        "delete-checkpoint",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (checkpoint,): (wit::Checkpoint,)| {
            Box::pin(async move {
                let path = accessor.with(|mut data| {
                    let host = data.get();
                    host.checkpoints.remove(&checkpoint.id)
                });
                match path {
                    Some(path) => {
                        if let Err(e) = std::fs::remove_file(&path) {
                            Ok((Err(wit::Error::CheckpointError(format!(
                                "Failed to delete checkpoint: {e}"
                            ))),))
                        } else {
                            Ok((Ok(()),))
                        }
                    }
                    None => Ok((Err(wit::Error::CheckpointError(format!(
                        "Checkpoint {:?} not found",
                        checkpoint.id
                    ))),)),
                }
            })
        },
    )?;

    // describe-model: async func(source: weights) -> result<model-description, error>
    instance.func_wrap_concurrent(
        "describe-model",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (source,): (wit::Weights,)| {
            Box::pin(async move {
                let (default_path, checkpoints) = accessor.with(|mut data| {
                    let host = data.get();
                    (host.default_model_path.clone(), host.checkpoints.clone())
                });

                let model_path = match source {
                    wit::Weights::HostDefault => default_path,
                    wit::Weights::Checkpoint(ckpt) => match checkpoints.get(&ckpt.id) {
                        Some(p) => p.clone(),
                        None => {
                            return Ok((Err::<wit::ModelDescription, wit::Error>(
                                wit::Error::CheckpointError(format!(
                                    "Checkpoint {:?} not found",
                                    ckpt.id
                                )),
                            ),));
                        }
                    },
                    wit::Weights::Path(p) => std::path::PathBuf::from(p),
                };

                let result = tokio::task::spawn_blocking(move || {
                    paramecia_engine::describe_model(&model_path)
                })
                .await;

                match result {
                    Ok(Ok(desc)) => {
                        let wit_desc: wit::ModelDescription = desc.into();
                        Ok((Ok(wit_desc),))
                    }
                    Ok(Err(e)) => Ok((Err(e.into()),)),
                    Err(e) => Ok((Err(wit::Error::ModelError(format!(
                        "describe_model task panicked: {e}"
                    ))),)),
                }
            })
        },
    )?;

    // update-metadata: async func(checkpoint: checkpoint, metadata: list<tuple<string, metadata-value>>) -> result<_, error>
    instance.func_wrap_concurrent(
        "update-metadata",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (checkpoint, metadata): (wit::Checkpoint, Vec<(String, wit::MetadataValue)>)| {
            Box::pin(async move {
                let checkpoint_path = accessor.with(|mut data| {
                    let host = data.get();
                    host.checkpoints.get(&checkpoint.id).cloned()
                });

                let checkpoint_path = match checkpoint_path {
                    Some(p) => p,
                    None => {
                        return Ok((Err::<(), wit::Error>(wit::Error::CheckpointError(format!(
                            "Checkpoint {:?} not found",
                            checkpoint.id
                        ))),));
                    }
                };

                let metadata_updates: std::collections::HashMap<String, exec::MetadataValue> =
                    metadata.into_iter().map(|(k, v)| (k, v.into())).collect();

                let path = checkpoint_path.clone();
                let result = tokio::task::spawn_blocking(move || {
                    paramecia_engine::update_model_metadata(&path, &metadata_updates)
                })
                .await
                .map_err(|e| wit::Error::CheckpointError(format!("Task join error: {e}")))?;

                match result {
                    Ok(()) => Ok((Ok(()),)),
                    Err(e) => Ok((Err(wit::Error::CheckpointError(format!(
                        "Failed to update metadata: {e}"
                    ))),)),
                }
            })
        },
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// structure-ext interface (1 sync + 7 async functions)
// ---------------------------------------------------------------------------

fn add_structure_ext_to_linker(linker: &mut Linker<HostState>) -> Result<()> {
    let mut instance = linker.instance("paramecia:host/structure-ext@0.1.0")?;

    use wasmtime::component::Accessor;

    // take-snapshot: async func(model: model) -> result<snapshot, error>
    instance.func_wrap_concurrent(
        "take-snapshot",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (model,): (wit::Model,)| {
            Box::pin(async move {
                let executor = accessor.with(|mut data| {
                    let host = data.get();
                    host.lookup_model(&model.id)
                        .map(|entry| entry.executor.clone())
                });

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => return Ok((Err(e),)),
                };

                let result = executor.take_snapshot().await;
                match result {
                    Ok(snap) => {
                        let host_uuid = accessor.with(|mut data| {
                            let host = data.get();
                            let uuid = host.next_uuid();
                            host.snapshots.insert(
                                uuid,
                                crate::host::SnapshotEntry {
                                    model_id: model.id,
                                    executor_snapshot_id: snap.id,
                                },
                            );
                            uuid
                        });
                        Ok((Ok(wit::Snapshot {
                            id: host_uuid,
                            state_position: snap.state_position,
                            num_tokens: snap.num_tokens,
                        }),))
                    }
                    Err(e) => Ok((Err(e.into()),)),
                }
            })
        },
    )?;

    // persist-snapshot: async func(handle: snapshot) -> result<persisted-snapshot, error>
    instance.func_wrap_concurrent(
        "persist-snapshot",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (handle,): (wit::Snapshot,)| {
            Box::pin(async move {
                let lookup = accessor.with(|mut data| {
                    let host = data.get();
                    match host.snapshots.get(&handle.id) {
                        Some(entry) => {
                            let model_entry = host.models.get(&entry.model_id);
                            match model_entry {
                                Some(me) => Ok((
                                    crate::host::SnapshotEntry {
                                        model_id: entry.model_id,
                                        executor_snapshot_id: entry.executor_snapshot_id,
                                    },
                                    me.executor.clone(),
                                )),
                                None => Err(wit::Error::ModelError(format!(
                                    "Model {:?} for snapshot {:?} not found",
                                    entry.model_id, handle.id
                                ))),
                            }
                        }
                        None => Err(wit::Error::SnapshotError(format!(
                            "Snapshot {:?} not found",
                            handle.id
                        ))),
                    }
                });

                let (entry, executor) = match lookup {
                    Ok((e, ex)) => (e, ex),
                    Err(e) => return Ok((Err(e),)),
                };

                let result = executor.persist_snapshot(entry.executor_snapshot_id).await;
                match result {
                    Ok((_persisted, path)) => {
                        // Remove from in-memory snapshots, register persisted path
                        accessor.with(|mut data| {
                            let host = data.get();
                            host.snapshots.remove(&handle.id);
                            host.persisted_snapshots.insert(handle.id, path);
                        });

                        Ok((Ok(wit::PersistedSnapshot { id: handle.id }),))
                    }
                    Err(e) => Ok((Err(e.into()),)),
                }
            })
        },
    )?;

    // drop-snapshot: func(snapshot: snapshot) — sync, no result
    instance.func_wrap_async(
        "drop-snapshot",
        |mut caller: wasmtime::StoreContextMut<'_, HostState>, (snapshot,): (wit::Snapshot,)| {
            Box::new(async move {
                let entry = caller
                    .data()
                    .snapshots
                    .get(&snapshot.id)
                    .map(|e| (e.model_id, e.executor_snapshot_id));
                if let Some((model_id, exec_snap_id)) = entry
                    && let Some(model_entry) = caller.data().models.get(&model_id)
                {
                    let _ = model_entry.executor.drop_snapshot(exec_snap_id).await;
                }
                // Always remove from our registry (even if executor call failed)
                caller.data_mut().snapshots.remove(&snapshot.id);
                Ok(())
            })
        },
    )?;

    // delete-snapshot: async func(snapshot: persisted-snapshot) -> result<_, error>
    instance.func_wrap_concurrent(
        "delete-snapshot",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (snapshot,): (wit::PersistedSnapshot,)| {
            Box::pin(async move {
                let path = accessor.with(|mut data| {
                    let host = data.get();
                    host.persisted_snapshots.remove(&snapshot.id)
                });
                match path {
                    Some(path) => {
                        if let Err(e) = std::fs::remove_file(&path) {
                            Ok((Err(wit::Error::SnapshotError(format!(
                                "Failed to delete snapshot: {e}"
                            ))),))
                        } else {
                            Ok((Ok(()),))
                        }
                    }
                    None => Ok((Err(wit::Error::SnapshotError(format!(
                        "Persisted snapshot {:?} not found",
                        snapshot.id
                    ))),)),
                }
            })
        },
    )?;

    // restore-snapshot: async func(model: model, snapshot: snapshot) -> result<_, error>
    instance.func_wrap_concurrent(
        "restore-snapshot",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (model, snapshot): (wit::Model, wit::Snapshot)| {
            Box::pin(async move {
                let lookup = accessor.with(|mut data| {
                    let host = data.get();
                    match host.snapshots.get(&snapshot.id) {
                        Some(entry) => {
                            if entry.model_id != model.id {
                                return Err(wit::Error::SnapshotError(format!(
                                    "Snapshot {:?} belongs to a different model",
                                    snapshot.id
                                )));
                            }
                            match host.models.get(&model.id) {
                                Some(me) => Ok((entry.executor_snapshot_id, me.executor.clone())),
                                None => Err(wit::Error::ModelError(format!(
                                    "Model {:?} not found",
                                    model.id
                                ))),
                            }
                        }
                        None => Err(wit::Error::SnapshotError(format!(
                            "Snapshot {:?} not found",
                            snapshot.id
                        ))),
                    }
                });

                let (exec_snap_id, executor) = match lookup {
                    Ok(v) => v,
                    Err(e) => return Ok((Err(e),)),
                };

                match executor.restore_snapshot(exec_snap_id).await {
                    Ok(()) => Ok((Ok(()),)),
                    Err(e) => Ok((Err(e.into()),)),
                }
            })
        },
    )?;

    // load-snapshot: async func(model: model, snapshot: persisted-snapshot) -> result<snapshot, error>
    instance.func_wrap_concurrent(
        "load-snapshot",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (model, persisted): (wit::Model, wit::PersistedSnapshot)| {
            Box::pin(async move {
                let lookup = accessor.with(|mut data| {
                    let host = data.get();
                    let path = host
                        .persisted_snapshots
                        .get(&persisted.id)
                        .cloned()
                        .ok_or_else(|| {
                            wit::Error::SnapshotError(format!(
                                "Persisted snapshot {:?} not found",
                                persisted.id
                            ))
                        })?;
                    let executor = host
                        .lookup_model(&model.id)
                        .map(|entry| entry.executor.clone())?;
                    Ok((path, executor))
                });

                let (path, executor) = match lookup {
                    Ok(v) => v,
                    Err(e) => return Ok((Err::<wit::Snapshot, wit::Error>(e),)),
                };

                match executor.load_persisted_snapshot(path).await {
                    Ok(snap) => {
                        let host_uuid = accessor.with(|mut data| {
                            let host = data.get();
                            let uuid = host.next_uuid();
                            host.snapshots.insert(
                                uuid,
                                crate::host::SnapshotEntry {
                                    model_id: model.id,
                                    executor_snapshot_id: snap.id,
                                },
                            );
                            uuid
                        });
                        Ok((Ok(wit::Snapshot {
                            id: host_uuid,
                            state_position: snap.state_position,
                            num_tokens: snap.num_tokens,
                        }),))
                    }
                    Err(e) => Ok((Err(e.into()),)),
                }
            })
        },
    )?;

    // reseed: async func(seed: u64) -> result<_, error>
    instance.func_wrap_concurrent(
        "reseed",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (seed,): (u64,)| {
            Box::pin(async move {
                // Reseed all loaded models
                let executors: Vec<_> = accessor.with(|mut data| {
                    let host = data.get();
                    host.models.values().map(|e| e.executor.clone()).collect()
                });

                for executor in executors {
                    if let Err(e) = executor.reseed(seed).await {
                        return Ok((Err::<(), wit::Error>(e.into()),));
                    }
                }

                Ok((Ok(()),))
            })
        },
    )?;

    // prune-experts: async func(source: weights, request: expert-pruning-request) -> result<checkpoint, error>
    instance.func_wrap_concurrent(
        "prune-experts",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (source, request): (wit::Weights, wit::ExpertPruningRequest)| {
            Box::pin(async move {
                let (checkpoint_dir, default_model_path, checkpoints) =
                    accessor.with(|mut data| {
                        let host = data.get();
                        (
                            host.checkpoint_dir.clone(),
                            host.default_model_path.clone(),
                            host.checkpoints.clone(),
                        )
                    });

                let source_path = match source {
                    wit::Weights::HostDefault => default_model_path,
                    wit::Weights::Checkpoint(ckpt) => match checkpoints.get(&ckpt.id) {
                        Some(p) => p.clone(),
                        None => {
                            return Ok((Err::<wit::Checkpoint, wit::Error>(
                                wit::Error::CheckpointError(format!(
                                    "Checkpoint {:?} not found",
                                    ckpt.id
                                )),
                            ),));
                        }
                    },
                    wit::Weights::Path(p) => std::path::PathBuf::from(p),
                };

                // Extract retained indices based on pruning request
                let retained = match request {
                    wit::ExpertPruningRequest::Naive(experts) => experts,
                    wit::ExpertPruningRequest::ShapleyMerge(_spec) => {
                        return Ok((Err(wit::Error::NotAvailable(
                            "Shapley-merge expert pruning is not yet implemented".to_string(),
                        )),));
                    }
                };

                let retained_indices = retained.indices;

                let output_path =
                    checkpoint_dir.join(format!("pruned-experts-{}.gguf", uuid::Uuid::new_v4()));
                let output_path_clone = output_path.clone();

                let prune_result = tokio::task::spawn_blocking(move || {
                    paramecia_engine::prune_experts(
                        &source_path,
                        &output_path_clone,
                        &retained_indices,
                    )
                })
                .await;

                match prune_result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        return Ok((Err(wit::Error::CheckpointError(format!(
                            "Expert pruning failed: {e}"
                        ))),));
                    }
                    Err(e) => {
                        return Ok((Err(wit::Error::CheckpointError(format!(
                            "Expert pruning task panicked: {e}"
                        ))),));
                    }
                };

                let uuid = accessor.with(|mut data| {
                    let host = data.get();
                    let uuid = host.next_uuid();
                    host.checkpoints.insert(uuid, output_path);
                    uuid
                });

                Ok((Ok(wit::Checkpoint { id: uuid }),))
            })
        },
    )?;

    // prune-layers: async func(source: weights, retained-layers: layers) -> result<checkpoint, error>
    instance.func_wrap_concurrent(
        "prune-layers",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (source, retained): (wit::Weights, wit::Layers)| {
            Box::pin(async move {
                let (checkpoint_dir, default_model_path, checkpoints) =
                    accessor.with(|mut data| {
                        let host = data.get();
                        (
                            host.checkpoint_dir.clone(),
                            host.default_model_path.clone(),
                            host.checkpoints.clone(),
                        )
                    });

                let source_path = match source {
                    wit::Weights::HostDefault => default_model_path,
                    wit::Weights::Checkpoint(ckpt) => match checkpoints.get(&ckpt.id) {
                        Some(p) => p.clone(),
                        None => {
                            return Ok((Err::<wit::Checkpoint, wit::Error>(
                                wit::Error::CheckpointError(format!(
                                    "Checkpoint {:?} not found",
                                    ckpt.id
                                )),
                            ),));
                        }
                    },
                    wit::Weights::Path(p) => std::path::PathBuf::from(p),
                };

                // Indices are already layer indices — pass directly
                let mut retained_layers = retained.indices;
                retained_layers.sort();

                let output_path =
                    checkpoint_dir.join(format!("pruned-layers-{}.gguf", uuid::Uuid::new_v4()));
                let output_path_clone = output_path.clone();

                let prune_result = tokio::task::spawn_blocking(move || {
                    paramecia_engine::prune_layers(
                        &source_path,
                        &output_path_clone,
                        &retained_layers,
                    )
                })
                .await;

                match prune_result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        return Ok((Err(wit::Error::CheckpointError(format!(
                            "Layer pruning failed: {e}"
                        ))),));
                    }
                    Err(e) => {
                        return Ok((Err(wit::Error::CheckpointError(format!(
                            "Layer pruning task panicked: {e}"
                        ))),));
                    }
                };

                let uuid = accessor.with(|mut data| {
                    let host = data.get();
                    let uuid = host.next_uuid();
                    host.checkpoints.insert(uuid, output_path);
                    uuid
                });

                Ok((Ok(wit::Checkpoint { id: uuid }),))
            })
        },
    )?;

    // graft: async func(composite: model-composite) -> result<checkpoint, error>
    instance.func_wrap_concurrent(
        "graft",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (composite,): (wit::ModelComposite,)| {
            Box::pin(async move {
                let (checkpoint_dir, default_model_path, checkpoints) =
                    accessor.with(|mut data| {
                        let host = data.get();
                        (
                            host.checkpoint_dir.clone(),
                            host.default_model_path.clone(),
                            host.checkpoints.clone(),
                        )
                    });

                // Helper to resolve wit::Weights → PathBuf
                let resolve = |w: &wit::Weights| -> Result<std::path::PathBuf, wit::Error> {
                    match w {
                        wit::Weights::HostDefault => Ok(default_model_path.clone()),
                        wit::Weights::Checkpoint(ckpt) => {
                            checkpoints.get(&ckpt.id).cloned().ok_or_else(|| {
                                wit::Error::CheckpointError(format!(
                                    "Checkpoint {:?} not found",
                                    ckpt.id
                                ))
                            })
                        }
                        wit::Weights::Path(p) => Ok(std::path::PathBuf::from(p)),
                    }
                };

                // Helper to resolve wit::LayerRef -> (path, layer_idx)
                let resolve_layer_ref =
                    |lr: &wit::LayerRef| -> Result<(std::path::PathBuf, u32), wit::Error> {
                        Ok((resolve(&lr.source)?, lr.layer_idx))
                    };

                // Build resolved composite
                let embedding = match resolve(&composite.embedding) {
                    Ok(p) => p,
                    Err(e) => return Ok((Err::<wit::Checkpoint, wit::Error>(e),)),
                };

                let mut layers = Vec::with_capacity(composite.layers.len());
                for lr in &composite.layers {
                    match resolve_layer_ref(lr) {
                        Ok(resolved) => layers.push(resolved),
                        Err(e) => return Ok((Err(e),)),
                    }
                }

                let lm_head = match composite.lm_head.as_ref().map(&resolve).transpose() {
                    Ok(p) => p.unwrap_or_else(|| default_model_path.clone()),
                    Err(e) => return Ok((Err(e),)),
                };

                let mtp_head = match composite.mtp_head.as_ref().map(resolve).transpose() {
                    Ok(p) => p.unwrap_or_else(|| default_model_path.clone()),
                    Err(e) => return Ok((Err(e),)),
                };

                let output_path =
                    checkpoint_dir.join(format!("graft-{}.gguf", uuid::Uuid::new_v4()));
                let output_path_clone = output_path.clone();

                let graft_result = tokio::task::spawn_blocking(move || {
                    paramecia_engine::graft_composite_from_paths(
                        &embedding,
                        &layers,
                        &lm_head,
                        &mtp_head,
                        &output_path_clone,
                    )
                })
                .await;

                match graft_result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        return Ok((Err(wit::Error::CheckpointError(format!(
                            "Graft failed: {e}"
                        ))),));
                    }
                    Err(e) => {
                        return Ok((Err(wit::Error::CheckpointError(format!(
                            "Graft task panicked: {e}"
                        ))),));
                    }
                };

                let uuid = accessor.with(|mut data| {
                    let host = data.get();
                    let uuid = host.next_uuid();
                    host.checkpoints.insert(uuid, output_path);
                    uuid
                });

                Ok((Ok(wit::Checkpoint { id: uuid }),))
            })
        },
    )?;

    // fuse: async func(base: weights, members: list<fusion-member>, strategy: quant-conflict-strategy) -> result<checkpoint, error>
    instance.func_wrap_concurrent(
        "fuse",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (base, members, strategy): (
            wit::Weights,
            Vec<wit::FusionMember>,
            wit::QuantConflictStrategy,
        )| {
            Box::pin(async move {
                // Extract paths and config from host state
                let (checkpoint_dir, default_model_path, checkpoints) =
                    accessor.with(|mut data| {
                        let host = data.get();
                        (
                            host.checkpoint_dir.clone(),
                            host.default_model_path.clone(),
                            host.checkpoints.clone(),
                        )
                    });

                // Resolve base weights to path
                let base_path = match base {
                    wit::Weights::HostDefault => default_model_path.clone(),
                    wit::Weights::Checkpoint(ckpt) => match checkpoints.get(&ckpt.id) {
                        Some(p) => p.clone(),
                        None => {
                            return Ok((Err::<wit::Checkpoint, wit::Error>(
                                wit::Error::CheckpointError(format!(
                                    "Checkpoint {:?} not found",
                                    ckpt.id
                                )),
                            ),));
                        }
                    },
                    wit::Weights::Path(p) => std::path::PathBuf::from(p),
                };

                // Resolve each member's weights to (PathBuf, f32)
                let mut member_pairs = Vec::with_capacity(members.len());
                for m in &members {
                    let path = match &m.weights {
                        wit::Weights::HostDefault => default_model_path.clone(),
                        wit::Weights::Checkpoint(ckpt) => match checkpoints.get(&ckpt.id) {
                            Some(p) => p.clone(),
                            None => {
                                return Ok((Err::<wit::Checkpoint, wit::Error>(
                                    wit::Error::CheckpointError(format!(
                                        "Checkpoint {:?} not found",
                                        ckpt.id
                                    )),
                                ),));
                            }
                        },
                        wit::Weights::Path(p) => std::path::PathBuf::from(p),
                    };
                    member_pairs.push((path, m.contribution));
                }

                // Run fusion on blocking thread (CPU-intensive + I/O-heavy)
                let base_path_clone = base_path.clone();
                let output_path =
                    checkpoint_dir.join(format!("fused-{}.gguf", uuid::Uuid::new_v4()));
                let output_path_clone = output_path.clone();
                let exec_strategy: exec::QuantConflictStrategy = strategy.into();
                let fuse_result = tokio::task::spawn_blocking(move || {
                    paramecia_engine::fuse_models(
                        &base_path_clone,
                        &member_pairs,
                        &output_path_clone,
                        exec_strategy,
                    )
                })
                .await;

                match fuse_result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Ok((Err(e.into()),)),
                    Err(e) => {
                        return Ok((Err(wit::Error::CheckpointError(format!(
                            "Fusion task panicked: {e}"
                        ))),));
                    }
                };

                // Register as checkpoint
                let uuid = accessor.with(|mut data| {
                    let host = data.get();
                    let uuid = host.next_uuid();
                    host.checkpoints.insert(uuid, output_path);
                    uuid
                });

                Ok((Ok(wit::Checkpoint { id: uuid }),))
            })
        },
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// inference interface (2 sync + 1 async functions)
// ---------------------------------------------------------------------------

fn add_inference_to_linker(linker: &mut Linker<HostState>) -> Result<()> {
    let mut instance = linker.instance("paramecia:host/inference@0.1.0")?;

    use wasmtime::component::Accessor;

    // fill-context: func(model: inference-model, input: list<model-input>) -> stream<result<u32, error>>
    instance.func_wrap_async(
        "fill-context",
        |mut caller: wasmtime::StoreContextMut<'_, HostState>,
         (model, inputs): (wit::Model, Vec<wit::ModelInput>)| {
            Box::new(async move {
                let executor = caller
                    .data()
                    .lookup_model(&model.id)
                    .map(|entry| entry.executor.clone());

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => {
                        // Return a stream that immediately yields the error
                        let (wit_tx, wit_rx) = tokio::sync::mpsc::channel(1);
                        let _ = wit_tx.send(Err(e)).await;
                        let producer = crate::stream_producer::MpscStreamProducer::new(wit_rx);
                        let reader = StreamReader::new(&mut caller, producer);
                        return Ok((reader,));
                    }
                };

                let exec_inputs: Vec<exec::ModelInput> =
                    inputs.into_iter().map(Into::into).collect();
                let mut exec_rx = executor
                    .fill_context_inputs(&exec_inputs)
                    .await
                    .map_err(|e| wasmtime::Error::msg(format!("{e}")))?;

                let (wit_tx, wit_rx) = tokio::sync::mpsc::channel(32);
                tokio::spawn(async move {
                    while let Some(result) = exec_rx.recv().await {
                        let wit_result: Result<u32, wit::Error> = to_wit::<u32, u32>(result);
                        if wit_tx.send(wit_result).await.is_err() {
                            break;
                        }
                    }
                });

                let producer = crate::stream_producer::MpscStreamProducer::new(wit_rx);
                let reader = StreamReader::new(&mut caller, producer);
                Ok((reader,))
            })
        },
    )?;

    // predict-completion: func(model: inference-model) -> stream<result<predicted, error>>
    instance.func_wrap_async(
        "predict-completion",
        |mut caller: wasmtime::StoreContextMut<'_, HostState>, (model,): (wit::Model,)| {
            Box::new(async move {
                let executor = caller
                    .data()
                    .lookup_model(&model.id)
                    .map(|entry| entry.executor.clone());

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => {
                        let (wit_tx, wit_rx) = tokio::sync::mpsc::channel(1);
                        let _ = wit_tx.send(Err(e)).await;
                        let producer = crate::stream_producer::MpscStreamProducer::new(wit_rx);
                        let reader = StreamReader::new(&mut caller, producer);
                        return Ok((reader,));
                    }
                };

                let (mut exec_rx, cancel_tx) = executor
                    .predict_completion(4096)
                    .await
                    .map_err(|e| wasmtime::Error::msg(format!("{e}")))?;

                // Store cancel sender
                caller
                    .data()
                    .prediction_cancels
                    .lock()
                    .unwrap()
                    .insert(model.id, cancel_tx);

                let (wit_tx, wit_rx) = tokio::sync::mpsc::channel(32);
                tokio::spawn(async move {
                    while let Some(result) = exec_rx.recv().await {
                        let wit_result: Result<wit::Predicted, wit::Error> =
                            to_wit::<exec::Predicted, wit::Predicted>(result);
                        if wit_tx.send(wit_result).await.is_err() {
                            break;
                        }
                    }
                });

                let producer = crate::stream_producer::MpscStreamProducer::new(wit_rx);
                let reader = StreamReader::new(&mut caller, producer);
                Ok((reader,))
            })
        },
    )?;

    // cancel-prediction: async func(model: inference-model) -> result<_, error>
    instance.func_wrap_concurrent(
        "cancel-prediction",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (model,): (wit::Model,)| {
            Box::pin(async move {
                accessor.with(|mut data| {
                    let host = data.get();
                    if let Some(cancel_tx) =
                        host.prediction_cancels.lock().unwrap().remove(&model.id)
                    {
                        let _ = cancel_tx.send(());
                    }
                });
                Ok((Ok::<(), wit::Error>(()),))
            })
        },
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// inference-ext interface (1 async + 1 sync functions)
// ---------------------------------------------------------------------------

fn add_inference_ext_to_linker(linker: &mut Linker<HostState>) -> Result<()> {
    let mut instance = linker.instance("paramecia:host/inference-ext@0.1.0")?;

    use wasmtime::component::Accessor;

    // predict-token: async func(model: inference-model) -> result<predicted, error>
    instance.func_wrap_concurrent(
        "predict-token",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (model,): (wit::Model,)| {
            Box::pin(async move {
                let executor = accessor.with(|mut data| {
                    let host = data.get();
                    host.lookup_model(&model.id)
                        .map(|entry| entry.executor.clone())
                });

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => return Ok((Err(e),)),
                };

                let result = executor.predict_token().await;
                Ok((to_wit::<exec::Predicted, wit::Predicted>(result),))
            })
        },
    )?;

    // predict-completions-batched: func(model: model, input: list<list<model-input>>) -> stream<result<list<predicted>, error>>
    instance.func_wrap_async(
        "predict-completions-batched",
        |mut caller: wasmtime::StoreContextMut<'_, HostState>,
         (model, inputs): (wit::Model, Vec<Vec<wit::ModelInput>>)| {
            Box::new(async move {
                let executor = caller
                    .data()
                    .lookup_model(&model.id)
                    .map(|entry| entry.executor.clone());

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => {
                        let (wit_tx, wit_rx) = tokio::sync::mpsc::channel(1);
                        let _ = wit_tx.send(Err(e)).await;
                        let producer = crate::stream_producer::MpscStreamProducer::new(wit_rx);
                        let reader = StreamReader::new(&mut caller, producer);
                        return Ok((reader,));
                    }
                };

                // Convert Vec<Vec<wit::ModelInput>> -> Vec<Vec<exec::ModelInput>>
                let exec_inputs: Vec<Vec<exec::ModelInput>> = inputs
                    .into_iter()
                    .map(|seq| seq.into_iter().map(Into::into).collect())
                    .collect();

                let (mut exec_rx, cancel_tx) =
                    match executor.predict_completions_batched(&exec_inputs).await {
                        Ok(pair) => pair,
                        Err(e) => {
                            let (wit_tx, wit_rx) = tokio::sync::mpsc::channel(1);
                            let _ = wit_tx
                                .send(Err::<Vec<wit::Predicted>, wit::Error>(e.into()))
                                .await;
                            let producer = crate::stream_producer::MpscStreamProducer::new(wit_rx);
                            let reader = StreamReader::new(&mut caller, producer);
                            return Ok((reader,));
                        }
                    };

                // Store cancel sender
                caller
                    .data()
                    .prediction_cancels
                    .lock()
                    .unwrap()
                    .insert(model.id, cancel_tx);

                let (wit_tx, wit_rx) = tokio::sync::mpsc::channel(32);
                tokio::spawn(async move {
                    while let Some(result) = exec_rx.recv().await {
                        let wit_result: Result<Vec<wit::Predicted>, wit::Error> = match result {
                            Ok(preds) => Ok(preds.into_iter().map(Into::into).collect()),
                            Err(e) => Err(e.into()),
                        };
                        if wit_tx.send(wit_result).await.is_err() {
                            break;
                        }
                    }
                });

                let producer = crate::stream_producer::MpscStreamProducer::new(wit_rx);
                let reader = StreamReader::new(&mut caller, producer);
                Ok((reader,))
            })
        },
    )?;

    // commit-token: func(model: inference-model, token: u32) -> result<_, error>
    instance.func_wrap_async(
        "commit-token",
        |caller: wasmtime::StoreContextMut<'_, HostState>, (model, token): (wit::Model, u32)| {
            Box::new(async move {
                let executor = caller
                    .data()
                    .lookup_model(&model.id)
                    .map(|entry| entry.executor.clone());

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => return Ok((Err(e),)),
                };

                let result = executor.commit_token(token).await;
                Ok((to_wit::<(), ()>(result),))
            })
        },
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// training interface (2 sync + 2 async functions)
// ---------------------------------------------------------------------------

fn add_training_to_linker(linker: &mut Linker<HostState>) -> Result<()> {
    let mut instance = linker.instance("paramecia:host/training@0.1.0")?;

    use wasmtime::component::Accessor;

    // Macro for shared train-model/validate-model logic.
    // Must be a macro (not a helper fn) so the async block's lifetime
    // is tied to the StoreContextMut<'_> from func_wrap_async.
    macro_rules! train_or_validate_handler {
        ($is_validation:expr) => {
            |mut caller: wasmtime::StoreContextMut<'_, HostState>,
             (model, data): (wit::Model, StreamReader<wit::TrainingSample>)| {
                Box::new(async move {
                    let executor = caller
                        .data()
                        .lookup_model(&model.id)
                        .map(|entry| entry.executor.clone());

                    let executor = match executor {
                        Ok(e) => e,
                        Err(e) => {
                            let mut data = data;
                            data.close(&mut caller);
                            let err: Result<
                                StreamReader<Result<wit::StepResult, wit::Error>>,
                                wit::Error,
                            > = Err(e);
                            return Ok((err,));
                        }
                    };

                    // Channels for executor ↔ actor communication
                    let (exec_sample_tx, exec_sample_rx) = tokio::sync::mpsc::channel(32);
                    let (exec_loss_tx, mut exec_loss_rx) = tokio::sync::mpsc::channel(32);
                    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

                    // Store cancel sender
                    caller
                        .data()
                        .training_cancels
                        .lock()
                        .unwrap()
                        .insert(model.id, cancel_tx);

                    // Send training command to unified actor
                    if let Err(e) = executor
                        .train_model(exec_sample_rx, $is_validation, exec_loss_tx, cancel_rx)
                        .await
                    {
                        let mut data = data;
                        data.close(&mut caller);
                        let err: Result<
                            StreamReader<Result<wit::StepResult, wit::Error>>,
                            wit::Error,
                        > = Err(e.into());
                        return Ok((err,));
                    }

                    // Pipe guest stream (WIT) → exec types → actor
                    let (wit_sample_tx, mut wit_sample_rx) = tokio::sync::mpsc::channel(32);
                    let consumer = crate::stream_producer::MpscStreamConsumer::new(wit_sample_tx);
                    data.pipe(&mut caller, consumer);

                    tokio::spawn(async move {
                        while let Some(wit_sample) = wit_sample_rx.recv().await {
                            let exec_sample: exec::TrainingSample = wit_sample.into();
                            if exec_sample_tx.send(exec_sample).await.is_err() {
                                break;
                            }
                        }
                    });

                    // Converter: exec results → WIT results
                    let (wit_loss_tx, wit_loss_rx) = tokio::sync::mpsc::channel(32);
                    tokio::spawn(async move {
                        while let Some(result) = exec_loss_rx.recv().await {
                            let wit_result: Result<wit::StepResult, wit::Error> =
                                to_wit::<exec::StepResult, wit::StepResult>(result);
                            if wit_loss_tx.send(wit_result).await.is_err() {
                                break;
                            }
                        }
                    });

                    let producer = crate::stream_producer::MpscStreamProducer::new(wit_loss_rx);
                    let reader = StreamReader::new(&mut caller, producer);
                    let ok: Result<StreamReader<Result<wit::StepResult, wit::Error>>, wit::Error> =
                        Ok(reader);
                    Ok((ok,))
                })
            }
        };
    }

    // train-model
    instance.func_wrap_async("train-model", train_or_validate_handler!(false))?;

    // validate-model
    instance.func_wrap_async("validate-model", train_or_validate_handler!(true))?;

    // cancel-training: async func(model: training-model) -> result<_, error>
    instance.func_wrap_concurrent(
        "cancel-training",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (model,): (wit::Model,)| {
            Box::pin(async move {
                accessor.with(|mut data| {
                    let host = data.get();
                    if let Some(cancel_tx) = host.training_cancels.lock().unwrap().remove(&model.id)
                    {
                        let _ = cancel_tx.send(());
                    }
                });
                Ok((Ok::<(), wit::Error>(()),))
            })
        },
    )?;

    // set-hyper-parameters: async func(model: training-model, update: hyper-parameter-update) -> result<_, error>
    instance.func_wrap_concurrent(
        "set-hyper-parameters",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (model, update): (wit::Model, wit::HyperParameterUpdate)| {
            Box::pin(async move {
                let executor = accessor.with(|mut data| {
                    let host = data.get();
                    host.lookup_model(&model.id)
                        .map(|entry| entry.executor.clone())
                });

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => return Ok((Err(e),)),
                };

                let exec_update: exec::HyperParameterUpdate = update.into();
                let result = executor.set_hyper_parameters(exec_update).await;
                Ok((to_wit::<(), ()>(result),))
            })
        },
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// training-ext interface (3 async functions)
// ---------------------------------------------------------------------------

fn add_training_ext_to_linker(linker: &mut Linker<HostState>) -> Result<()> {
    let mut instance = linker.instance("paramecia:host/training-ext@0.1.0")?;

    use wasmtime::component::Accessor;

    // perturb-up: async func(model: model, seed: option<u64>) -> result<positive-model, error>
    instance.func_wrap_concurrent(
        "perturb-up",
        |accessor: &Accessor<HostState, HasSelf<HostState>>, (model, seed): (wit::Model, Option<u64>)| {
            Box::pin(async move {
                let executor = accessor.with(|mut data| {
                    let host = data.get();
                    host.lookup_model(&model.id)
                        .map(|entry| entry.executor.clone())
                });

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => return Ok((Err(e),)),
                };

                let result = executor.perturb_up(seed).await;
                match result {
                    Ok(()) => Ok((Ok(wit::PositiveModel { model }),)),
                    Err(e) => Ok((Err(e.into()),)),
                }
            })
        },
    )?;

    // perturb-down: async func(perturbed: positive-model) -> result<negative-model, error>
    instance.func_wrap_concurrent(
        "perturb-down",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (perturbed,): (wit::PositiveModel,)| {
            Box::pin(async move {
                let model_id = perturbed.model.id;
                let executor = accessor.with(|mut data| {
                    let host = data.get();
                    host.lookup_model(&model_id)
                        .map(|entry| entry.executor.clone())
                });

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => return Ok((Err(e),)),
                };

                let result = executor.perturb_down().await;
                match result {
                    Ok(()) => Ok((Ok(wit::NegativeModel {
                        model: perturbed.model,
                    }),)),
                    Err(e) => Ok((Err(e.into()),)),
                }
            })
        },
    )?;

    // update: async func(perturbed: negative-model, loss-up: f32, loss-down: f32) -> result<training-model, error>
    instance.func_wrap_concurrent(
        "update",
        |accessor: &Accessor<HostState, HasSelf<HostState>>,
         (perturbed, loss_up, loss_down): (wit::NegativeModel, f32, f32)| {
            Box::pin(async move {
                let model_id = perturbed.model.id;
                let executor = accessor.with(|mut data| {
                    let host = data.get();
                    host.lookup_model(&model_id)
                        .map(|entry| entry.executor.clone())
                });

                let executor = match executor {
                    Ok(e) => e,
                    Err(e) => return Ok((Err(e),)),
                };

                let result = executor.update(loss_up, loss_down).await;
                match result {
                    Ok(()) => Ok((Ok(perturbed.model),)),
                    Err(e) => Ok((Err(e.into()),)),
                }
            })
        },
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// interactive interface (1 sync function returning bidirectional streams)
// ---------------------------------------------------------------------------

fn add_interactive_to_linker(linker: &mut Linker<HostState>) -> Result<()> {
    let mut instance = linker.instance("paramecia:host/interactive@0.1.0")?;

    // connect: func(outputs: stream<agent-event>) -> result<stream<result<user-event, error>>, error>
    instance.func_wrap_async(
        "connect",
        |mut caller: wasmtime::StoreContextMut<'_, HostState>,
         (outputs,): (StreamReader<wit::AgentEvent>,)| {
            Box::new(async move {
                // Take the interactive endpoint from host state (one-shot)
                let endpoint = caller.data_mut().interactive_endpoint.take();
                let endpoint = match endpoint {
                    Some(ep) => ep,
                    None => {
                        let mut outputs = outputs;
                        outputs.close(&mut caller);
                        let err: Result<
                            StreamReader<Result<wit::UserEvent, wit::Error>>,
                            wit::Error,
                        > = Err(wit::Error::NotAvailable(
                            "No interactive UI connected".to_string(),
                        ));
                        return Ok((err,));
                    }
                };

                let paramecia_bridge::ControllerHostEndpoint {
                    mut user_rx,
                    agent_tx,
                } = endpoint;

                // Guest -> Backend direction: pipe guest's agent-event stream
                // to the bridge's agent_tx channel
                let (wit_agent_tx, mut wit_agent_rx) =
                    tokio::sync::mpsc::channel::<wit::AgentEvent>(32);
                let consumer = crate::stream_producer::MpscStreamConsumer::new(wit_agent_tx);
                outputs.pipe(&mut caller, consumer);

                // Converter: WIT AgentEvent -> bridge ControllerAgentEvent -> agent_tx
                //
                // Passes through delta token counts from the guest. The TUI-side
                // ControllerBackend accumulates these into a running tally.
                tokio::spawn(async move {
                    while let Some(wit_event) = wit_agent_rx.recv().await {
                        let bridge_event = match wit_event {
                            wit::AgentEvent::ContextTokens(ct) => {
                                paramecia_bridge::ControllerAgentEvent::ContextTokens {
                                    context_tokens: ct,
                                }
                            }
                            wit::AgentEvent::Token(t) => {
                                paramecia_bridge::ControllerAgentEvent::Token { content: t }
                            }
                            wit::AgentEvent::ToolCall(tc) => {
                                paramecia_bridge::ControllerAgentEvent::ToolCall {
                                    id: tc.id,
                                    name: tc.name,
                                    arguments: tc.arguments,
                                    n_tokens: tc.n_tokens,
                                }
                            }
                            wit::AgentEvent::Done => paramecia_bridge::ControllerAgentEvent::Done,
                            wit::AgentEvent::Initialized => {
                                paramecia_bridge::ControllerAgentEvent::Initialized
                            }
                            wit::AgentEvent::Error(msg) => {
                                paramecia_bridge::ControllerAgentEvent::Error(msg)
                            }
                        };
                        if agent_tx.send(bridge_event).await.is_err() {
                            break;
                        }
                    }
                });

                // Backend -> Guest direction: read user events from bridge,
                // convert to WIT UserEvent, and produce a stream for the guest
                let (wit_user_tx, wit_user_rx) =
                    tokio::sync::mpsc::channel::<Result<wit::UserEvent, wit::Error>>(32);
                tokio::spawn(async move {
                    while let Some(bridge_event) = user_rx.recv().await {
                        let wit_event = match bridge_event {
                            paramecia_bridge::ControllerUserEvent::Prefill(s) => {
                                wit::UserEvent::Prefill(s)
                            }
                            paramecia_bridge::ControllerUserEvent::Prompt(s) => {
                                wit::UserEvent::Prompt(s)
                            }
                            paramecia_bridge::ControllerUserEvent::Cancel => wit::UserEvent::Cancel,
                            paramecia_bridge::ControllerUserEvent::ToolResult {
                                id,
                                name,
                                content,
                            } => wit::UserEvent::ToolResult(wit::ToolExecResult {
                                id,
                                name,
                                content,
                            }),
                        };
                        if wit_user_tx.send(Ok(wit_event)).await.is_err() {
                            break;
                        }
                    }
                });

                let producer = crate::stream_producer::MpscStreamProducer::new(wit_user_rx);
                let reader = StreamReader::new(&mut caller, producer);
                let ok: Result<StreamReader<Result<wit::UserEvent, wit::Error>>, wit::Error> =
                    Ok(reader);
                Ok((ok,))
            })
        },
    )?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compile a WASM component from file.
pub fn compile_component(engine: &Engine, path: &Path) -> Result<Component> {
    Component::from_file(engine, path)
}

/// Run a controller component to completion.
pub async fn run_host(
    engine: &Engine,
    linker: &Linker<HostState>,
    component: &Component,
    state: HostState,
) -> Result<()> {
    let mut store = Store::new(engine, state);

    let instance = Controller::instantiate_async(&mut store, component, linker).await?;

    let run_iface = instance.wasi_cli_run();
    let _ = store
        .run_concurrent(async |accessor| {
            let (result, _task_exit) = run_iface.call_run(accessor).await?;
            match result {
                Ok(()) => Ok::<(), anyhow::Error>(()),
                Err(()) => anyhow::bail!("Controller exited with error"),
            }
        })
        .await?;

    Ok(())
}
