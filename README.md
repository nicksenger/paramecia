### Note from the author

Paramecia are single-celled organisms common in freshwater. In biology, they're often used as models for various cellular functions like genetics and reproduction. 

The goal of this project is to create small, self-contained agents with analogous characteristics. Specifically, it's intended to:

- [x] Run as a single process, deliverable as a single binary
- [x] Perform decently on my machine (24gb VRAM + 64gb DRAM)
- [x] Be capable of reading and editing its own source code
- [ ] Incrementally modify its own weights and source-code without human intervention

Currently, you can run this project and talk to it about One Piece, etc, but it is not comparable to frontier-level cloud next-token predictors. The focus here is reflection and self-improvement, not utility.

Please note that the ultimate goal of this project is to implement what could be considered a recursive self-improvement process, and that the legal landscape around artificial intelligence experiments of this nature is rapidly evolving. As of today, where I am located (USA), any projects of this kind employing >=10^26 floating point operations must notify authorities.

To put things in perspective, with my own hardware it would take ~90,000 years of continuous operation to reach this number, but users and contributors should make sure that they are acting in compliance with any applicable regulations in their own jurisdiction.

# Paramecia

Paramecia is an offline, single-process agentic CLI. It currently uses GGUF-quantized Qwen3-Next-80B-A3B-MoE for inference.

## Project Roadmap

The project will be conducted in 3 phases: 

1. [ ] **Scaffolding** (current): all necessary utilities regarding basic execution, optimization, code modification, and inference will be prepared. Coding will be performed with the assistance of state-of-the-art next-token-prediction utilities.
2. [ ] **Priming**: the model will be fine-tuned on a collection of tokens related to its construction.
3. [ ] **Inception**: this one's a surprise :)

## Workspace overview

This repository is a Cargo workspace with the following crates:

- `paramecia-cli`: The `paramecia` binary (TUI + programmatic mode).
- `paramecia-harness`: Core agent logic, configuration, prompts, project-context collection, and session logging.
- `paramecia-tools`: Tool traits, permissioning, and built-in tools (`bash`, `read_file`, `write_file`, `search_replace`, `grep`, `todo`), plus MCP tool adapters.
- `paramecia-mcp`: Model Context Protocol (MCP) client implementation (HTTP/stdio transports).
- `paramecia-acp`: Agent Communication Protocol (ACP) server-side types and helpers for external integrations.
- `paramecia-llm`: Local LLM backend with streaming generation, chat templating, and tool-call parsing.
- `paramecia-model`: The high-performance model runtime used by the local backend (details below).
- `paramecia-opt`: Optimization and evaluation utilities (used for local training/tuning workflows).

## `paramecia-model` (model runtime)

`paramecia-model` is the low-level, performance-focused inference engine used by `paramecia-llm`’s `LocalBackend`. Its primary target is Qwen3-Next in GGUF format, with an emphasis on long-context, quantized inference and practical throughput on commodity hardware.

Key capabilities:

- **GGUF loading with quantized weights** via Candle’s GGUF support
- **Hybrid Qwen3-Next architecture support**:
  - Full-attention layers (with RoPE and Q projections)
  - Linear attention / “Gated Delta Net” recurrent layers
  - Mixture-of-Experts (MoE) FFN with shared experts
- **MoE offload modes** (`DeviceOffloadMode`) to trade VRAM for throughput:
  - Keep experts on GPU, offload experts fully to CPU, or split projections across devices.
- **Prefetch-based pipelining** (`layer_pipeline`) to overlap GPU compute with CPU MoE work and hide transfer latency.
- **KV-cache implementations optimized for autoregressive decoding**:
  - Preallocated (O(1) append) caches to avoid quadratic `cat` behavior.
  - Optional **quantized KV cache** (`KvCacheQuantization`, default Q4K) for long-context memory scaling.
- **Prefix caching** (`PrefixCache`) to reuse shared conversation prefixes across turns (restore K/V and recurrent state).
- **Sampling + logit processing** (`generation`) and utilities for repetition/presence penalties.

Reference entry points:

- Qwen3-Next model implementation: `paramecia-model/src/qwen3_next.rs`
- Accelerated ops wiring (CUDA/Metal/CPU dispatch via Candle): `paramecia-model/src/ops.rs`
- Prefetch pipeline: `paramecia-model/src/layer_pipeline.rs`
- Standalone inference example: `paramecia-model/examples/qwen3_next.rs`

## Getting started

### Prerequisites

- Rust toolchain pinned by `rust-toolchain.toml` (currently Rust 1.90).
- A local Qwen3-Next GGUF file (e.g. `Qwen3-Next-*-GGUF/*.gguf`).
- (Optional) GPU backend:
  - `--features cuda` for NVIDIA CUDA
  - `--features flash-attn` to enable Candle’s flash-attention integration (CUDA)
  - `--features metal` for Apple Metal (please note that I cannot evaluate the performance of Metal on my hardware currently- the model is too big even at q2)
  - `--features accelerate` for Apple Accelerate

Note: this workspace currently depends on a Candle fork with additional kernels for the Qwen3-Next DeltaNet operations and quantized flash attention.

### Build and run

Interactive TUI:

```bash
PARAMECIA_MODEL_PATH=/path/to/model.gguf cargo run -p paramecia-cli --release
```

Programmatic mode (single prompt, prints the final answer, then exits):

```bash
PARAMECIA_MODEL_PATH=/path/to/model.gguf cargo run -p paramecia-cli --release -- --prompt "List the files in this repository"
```

The CLI auto-selects a device by default (CUDA → Metal → CPU). Override with config or by setting the provider’s `local_device` to `cpu`, `cuda`, or `metal`.

## Configuration

Paramecia loads configuration in this order:

1. Global config: `~/.paramecia/config.toml` (override base directory with `PARAMECIA_HOME`)
2. Per-project config: `.paramecia/config.toml` in the current working directory
3. Optional agent overlay: `--agent NAME` loads `~/.paramecia/agents/NAME.toml`

For the local backend, the following environment variables are supported:

- `PARAMECIA_MODEL_PATH`: GGUF model path (required unless set in config)
- `PARAMECIA_TOKENIZER_PATH`: optional `tokenizer.json` path (otherwise downloaded)
- `PARAMECIA_CONTEXT_LENGTH`: maximum context tokens (default 131072)
- `PARAMECIA_OFFLOAD`: MoE offload mode (`none`, `up`, `updown`, `experts`)
- `PARAMECIA_KV_CACHE_QUANT`: KV cache mode (`f16`, `bf16`, `q8`, `q4`/`q4k`)
- `PARAMECIA_NO_PREFETCH=1`: disable the prefetch pipeline (debug/comparison)
- Sampling overrides: `PARAMECIA_TEMPERATURE`, `PARAMECIA_TOP_P`, `PARAMECIA_TOP_K`, `PARAMECIA_REPEAT_PENALTY`, `PARAMECIA_PRESENCE_PENALTY`

## License

Licensed under the MIT License. See `LICENSE`.
