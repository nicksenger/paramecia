#[allow(dead_code)]
mod expert_pipeline;
mod generation;
#[allow(dead_code)]
mod graft;
#[allow(dead_code)]
mod inspect;
#[allow(dead_code)]
mod layer_pipeline;
pub mod models;
#[allow(dead_code)]
mod ops;
#[allow(dead_code)]
mod quantized_nn;
#[allow(dead_code)]
mod quantized_var_builder;
#[allow(dead_code)]
mod snapshot;
#[allow(dead_code)]
mod utils;

pub use generation::LogitsProcessor;
pub use graft::{graft_composite, GraftComposite, GraftCompositeOptions, GraftLayerSource};
pub use models::qwen3_next::{
    select_best_device, DeviceOffloadMode, KvCacheQuantization, LayerDeviceMap, ModelWeights,
    PrefixCache, YarnConfig,
};
pub use paramecia_arrow::vis;
pub use paramecia_core::quantized::gguf_file;
pub use paramecia_core::quantized::GgmlDType;
pub use paramecia_core::{DType, Device, Result, Tensor};
pub use token_output_stream::TokenOutputStream;
pub use utils::{apply_penalties, apply_penalties_slice};

/// Build the model computation graph used by the visualizer.
pub fn visualization_graph() -> vis::Graph {
    use paramecia_arrow::vis::Vis;
    models::qwen3_next::ModelWeights::visualize()
}

/// Token output stream for streaming text generation
mod token_output_stream {
    use tokenizers::Tokenizer;

    /// Wrapper around tokenizer for streaming output
    pub struct TokenOutputStream {
        tokenizer: Tokenizer,
        tokens: Vec<u32>,
        prev_index: usize,
        current_index: usize,
    }

    impl TokenOutputStream {
        pub fn new(tokenizer: Tokenizer) -> Self {
            Self {
                tokenizer,
                tokens: Vec::new(),
                prev_index: 0,
                current_index: 0,
            }
        }

        pub fn tokenizer(&self) -> &Tokenizer {
            &self.tokenizer
        }

        fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
            match self.tokenizer.decode(tokens, true) {
                Ok(s) => Ok(s),
                Err(e) => anyhow::bail!("tokenizer decode error: {}", e),
            }
        }

        pub fn next_token(&mut self, token: u32) -> anyhow::Result<Option<String>> {
            let prev_text = if self.tokens.is_empty() {
                String::new()
            } else {
                let tokens = &self.tokens[self.prev_index..self.current_index];
                self.decode(tokens)?
            };
            self.tokens.push(token);
            let text = self.decode(&self.tokens[self.prev_index..])?;
            // Emit whenever we have new complete text - allow all character types
            // including newlines/tabs to avoid buffering on whitespace boundaries
            if text.len() > prev_text.len() && text.chars().last().is_some() {
                // Use strip_prefix for safe UTF-8 handling - if the prefix doesn't match
                // exactly (which shouldn't happen but guards against edge cases), fall back
                // to character-based extraction
                let new_text = if let Some(suffix) = text.strip_prefix(&prev_text) {
                    suffix.to_string()
                } else {
                    // Fallback: skip the same number of characters as prev_text
                    text.chars().skip(prev_text.chars().count()).collect()
                };
                self.current_index = self.tokens.len();
                Ok(Some(new_text))
            } else {
                Ok(None)
            }
        }

        pub fn decode_rest(&self) -> anyhow::Result<Option<String>> {
            let prev_text = if self.tokens.is_empty() {
                String::new()
            } else {
                let tokens = &self.tokens[self.prev_index..self.current_index];
                self.decode(tokens)?
            };
            let text = self.decode(&self.tokens[self.prev_index..])?;
            if text.len() > prev_text.len() {
                // Use strip_prefix for safe UTF-8 handling
                let new_text = if let Some(suffix) = text.strip_prefix(&prev_text) {
                    suffix.to_string()
                } else {
                    // Fallback: skip the same number of characters as prev_text
                    text.chars().skip(prev_text.chars().count()).collect()
                };
                Ok(Some(new_text))
            } else {
                Ok(None)
            }
        }

        pub fn decode_all(&self) -> anyhow::Result<String> {
            self.decode(&self.tokens)
        }

        pub fn get_token(&self, idx: usize) -> Option<u32> {
            self.tokens.get(idx).copied()
        }

        pub fn tokenizer_mut(&mut self) -> &mut Tokenizer {
            &mut self.tokenizer
        }

        pub fn clear(&mut self) {
            self.tokens.clear();
            self.prev_index = 0;
            self.current_index = 0;
        }
    }
}
