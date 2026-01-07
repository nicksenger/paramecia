//! Backend factory for creating LLM backends.

use super::{Backend, ProviderConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Type of backend to use.
///
/// Currently only Local is supported for local-only operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    /// Local quantized backend using the Qwen3 architecture.
    #[default]
    Local,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
        }
    }
}

impl std::str::FromStr for BackendType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "local" => Ok(Self::Local),
            _ => Err(format!(
                "Unknown backend type: {s}. Only 'local' is supported."
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
    pub fn create(
        provider: &ProviderConfig,
        timeout: Duration,
    ) -> Result<Arc<dyn Backend>, String> {
        match provider.backend {
            BackendType::Local => {
                let backend = crate::backend::local::LocalBackend::new(provider.clone(), timeout)
                    .map_err(|e| e.to_string())?;
                Ok(Arc::new(backend))
            }
        }
    }
}
