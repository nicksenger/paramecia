
cargo run --features=flash-attn --release -p paramecia-model --example=qwen3_next -- --model-path=/home/dev/qwen3next-ud-mtp-q2.gguf --prefetch-pipeline --mtp

cargo run --features=flash-attn --release -p paramecia-llm --example=local_agent -- --model-path=/home/dev/qwen3next-ud-mtp-q2.gguf

PARAMECIA_MODEL_PATH=/home/chip/qwen3/Qwen3-Next-80B-A3B-Instruct-UD-Q4_K_XL.gguf cargo run --features=flash-attn --release
