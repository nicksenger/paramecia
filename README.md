> Wealth, fame, power.
> Gold Roger, the kind of the agents, attained this and everything else the world had to offer,
> and his dying words drove countless souls to the cloud.
>
> "You want my tensors? You can have them!"
> "I left everything I gathered together in weight space"
> "Now all you have to do is find it."
>
> These words lured men to shop online
> for RAM more expensive than they ever dared to imagine!
>
> This is the time known as the AI Slop Era!

# Paramecia

Paramecia provides offline, single-process, next-token predition.

It's intended to be run on consumer hardware, and currently uses the Qwen3-Next architecture in GGUF format.

### Contents

- Full weight optimization in low-memory environments (zeroth-order)
- Multi-device inference with batching, expert offloading, KV-cache quantization, flash attention, multi-token prediction, etc
- Vulkan, Metal, and CUDA backends based on [candle](https://github.com/huggingface/candle) with model-specific kernels and optimizations
- Wasm-pluggable controller interface for programmatic training and inference orchestration
- Integrated agentic TUI with builtin tools, MCP support, and permissioning
- Model visualization and other misc. utilities

### Usage

1. Set `PARAMECIA_MODEL_PATH` to a Qwen3-Next gguf (i-quants are not supported)
2. Run `cargo run --release --features={cuda/metal/vulkan} -- --help`

### Design Decisions

- TUI: This was originally branched off of [revibe](https://github.com/nicksenger/revibe) which was an experiment I did to see whether Devstral 2 could rewrite Mistral's Vibe CLI in Rust. Not much is different here really except the backends. I may have also replaced some of the mutexes with channels and fixed a few UI issues.
- Core: This started as a fork of [candle](https://github.com/huggingface/candle) and contains the core (untyped)tensors and ops. I added a Vulkan backend and some model-specific shaders. Eventually I decided it was most likely never going upstream anyways, so just pulled it in directly.
- Tensors: the goal is to have the model and other parts of the project use the typed tensors exclusively instead of the tensors from core, providing shape information at compile time to support various features/enhancements. Still WIP
- Model architecture: I chose Qwen3-Next because the coder model is the best one I can run locally on my hardware currently. The forward pass is mostly written as arrow-combinators which, though it looks odd and verbose, allows inspecting the computation graph. This enables the visualizations and in theory shader-specialization (given the typed shapes) & various other optimizations.
- Training: Since RAM is usually the limiting factor for tuning on consumer hardware, I chose to support a zeroth-order optimization method that can be applied directly to quantized weights. In theory, this means that if your hardware can run inference for a model, then it can also tune that model. The caveat is that it may take a long time :) 
- WASM/WASI: The host interface gives full control over the model/inference/etc to a WASM guest-module. This allows some flexibility around language, sharing training modules, etc 
- Visualization: Based on [iced](https://github.com/iced-rs/iced), renders the arrow graph using sugiyama. Originally I did graphviz dot but this is more fun.

