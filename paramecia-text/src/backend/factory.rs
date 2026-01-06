//! Backend factory for creating LLM backends.

use super::{Backend, ProviderConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Type of backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    /// Local quantized backend using the Qwen3 architecture.
    #[default]
    Local,
    /// Controller backend using a WASM guest module (requires a bridge, not created via factory).
    Controller,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::Controller => write!(f, "controller"),
        }
    }
}

impl std::str::FromStr for BackendType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "local" => Ok(Self::Local),
            "controller" => Ok(Self::Controller),
            _ => Err(format!(
                "Unknown backend type: {s}. Supported: 'local', 'controller'."
            )),
        }
    }
}

/// Factory for creating LLM backends.
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend for the given provider configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot be created.
    pub fn create(provider: &ProviderConfig) -> Result<Arc<dyn Backend>, String> {
        match provider.backend {
            BackendType::Local => {
                let backend = crate::backend::local::LocalBackend::new(provider.clone())
                    .map_err(|e| e.to_string())?;
                Ok(Arc::new(backend))
            }
            BackendType::Controller => Err(
                "Controller backend requires a bridge — use ControllerBackend::new() directly"
                    .to_string(),
            ),
        }
    }
}
