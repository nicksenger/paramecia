#[derive(Debug, thiserror::Error)]
pub enum HostError {
    #[error("WASM compilation failed: {0}")]
    Compilation(#[from] wasmtime::Error),
    #[error("Model error: {0}")]
    Model(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Invalid state: {0}")]
    InvalidState(String),
}
