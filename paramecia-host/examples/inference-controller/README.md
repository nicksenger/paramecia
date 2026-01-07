# Inference Controller

A WASM controller component for paramecia that demonstrates concurrent streaming and
cancellation. Prefills a hardcoded prompt, generates tokens using `predict-completion`,
and cancels after 20 tokens.

## Prerequisites

```bash
# WASM target for compilation
rustup target add wasm32-wasip1
```

## Build

```bash
cd paramecia-controller/examples/inference-controller

cargo build --release --target

# Download WASI adapter matching wasmtime version (v41, one-time)
curl -L https://github.com/bytecodealliance/wasmtime/releases/download/v41.0.0/wasi_snapshot_preview1.reactor.wasm \
  -o /tmp/wasi_snapshot_preview1.reactor.wasm

# Convert to component
wasm-tools component new \
  target/wasm32-unknown-unknown/release/inference_controller.wasm \
  -o inference_controller.wasm \
  --adapt wasi_snapshot_preview1=/tmp/wasi_snapshot_preview1.reactor.wasm
```

**Why wasip1?** WASIp3 (0.3.0-rc) features like `async func` and `stream<T>` aren't
fully supported by the tooling yet. We build for wasip1 and use adapters to bridge to
wasip3 at runtime. This will be simpler once WASIp3 stabilizes.

## Run

From the workspace root:

```bash
cargo run --example controller_runner -p paramecia-controller -- \
    --controller paramecia-controller/examples/inference-controller/inference_controller.wasm \
    --model-path /path/to/model.gguf
```

## Writing Your Own Controller

Copy this crate as a starting point. The `../../wit/` directory contains the interface
definitions your controller can use:

- **`fill-context(text)`** — tokenize and fill model context, returns progress stream
- **`predict-token()`** — run forward pass and sample, returns predicted token with logits
- **`commit-token(token)`** — commit a token into context (call between predict-token calls)
- **`predict-completion()`** — generate full completion, returns stream of predictions
- **`cancel-prediction()`** — cancel ongoing predict-completion operation
- **`save-snapshot()`** — save model state, returns snapshot handle
- **`restore-snapshot(handle)`** — restore model state from snapshot
- **`delete-snapshot(handle)`** — delete a snapshot file
- **`reset-state()`** — reset to initial state (or initial snapshot if provided)
- **`host-channel(input: stream<u8>)`** — bidirectional byte channel (not currently supported)

The controller exports `wasi:cli/run` as its async entry point. The host calls this and the
controller controls the entire inference loop.

### Message-Passing Architecture

The host uses an actor pattern with message passing (tokio::mpsc) instead of shared
mutexes. Each inference command is sent to an actor task that owns the model state,
ensuring no lock contention and safe async execution.

### Async Streaming

WASIp3 async support allows the guest to use native Rust `async/await`:
- Functions returning `stream<T>` can be consumed with `.next().await`
- `async func` declarations in WIT are awaitable from the guest
- Use `async-channel` for concurrent streaming (unbounded channels work in no_std WASM)
