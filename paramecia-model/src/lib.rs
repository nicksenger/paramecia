pub mod generation;
pub mod layer_pipeline;
pub mod models;
pub mod ops;
pub mod quantized_nn;
pub mod quantized_var_builder;
pub mod qwen3_next;
pub mod tokio_experts;
pub mod utils;

pub use generation::{LogitsProcessor, Sampling};
pub use qwen3_next::{
    DeviceOffloadMode, KvCacheQuantization, LoraAdapter, ModelWeights, PrefixCache,
};

/// Token output stream for streaming text generation
pub mod token_output_stream {
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
