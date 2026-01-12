
cargo run --features=cuda --release -p paramecia-model --example=qwen3_next -- --model-path=/home/chip/qwen3/Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL.gguf --prefetch-pipeline

cargo run --features=cuda --release -p paramecia-llm --example=local_agent -- --model-path=/home/chip/qwen3/Qwen3-Next-80B-A3B-Instruct-UD-Q2_K_XL.gguf

PARAMECIA_MODEL_PATH=/home/chip/qwen3/Qwen3-Next-80B-A3B-Instruct-UD-Q4_K_XL.gguf cargo run --features=cuda --release
