//! Model-specific tool prompts and descriptions.
//!
//! This module provides model-aware tool descriptions that can be customized
//! based on the LLM being used. Different models may benefit from different
//! prompt styles, examples, and levels of detail.
//!
//! ## Integration with Tool Aliases
//!
//! This module works together with [`crate::tool_aliases`] to provide
//! complete model compatibility:
//!
//! - `model_prompts`: Provides model-specific prompts and examples
//! - `tool_aliases`: Maps tool names and parameters to model expectations
//!
//! ## Example
//!
//! ```rust
//! use paramecia_tools::{ModelPromptBuilder, ToolAliasManager};
//!
//! let model = "qwen3-coder";
//!
//! // Get model-specific prompts
//! let builder = ModelPromptBuilder::for_model(model);
//! let enhanced_prompt = builder.enhance_tool_prompt("bash", "Base prompt");
//!
//! // Get tool name aliases
//! let aliases = ToolAliasManager::for_model(model);
//! let model_name = aliases.to_model_name("bash"); // "run_shell_command"
//! ```

use std::collections::HashMap;

/// Model family for prompt customization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModelFamily {
    /// Qwen Coder models (qwen3-coder, qwen2.5-coder, qwen3-next)
    QwenCoder,
    /// Qwen VL (vision-language) models
    QwenVl,
    /// Qwen base models
    QwenBase,
    /// Claude models
    Claude,
    /// GPT models
    Gpt,
    /// Generic/unknown model
    #[default]
    Generic,
}

impl ModelFamily {
    /// Detect model family from model name.
    #[must_use]
    pub fn from_model_name(model: &str) -> Self {
        let model_lower = model.to_lowercase();

        // Qwen Coder variants
        if model_lower.contains("qwen") && model_lower.contains("coder") {
            return Self::QwenCoder;
        }
        if model_lower.contains("qwen3-next") || model_lower.contains("qwen3next") {
            return Self::QwenCoder;
        }
        if model_lower.contains("coder-model") {
            return Self::QwenCoder;
        }

        // Qwen VL variants
        if model_lower.contains("qwen") && model_lower.contains("-vl") {
            return Self::QwenVl;
        }
        if model_lower.contains("vision-model") {
            return Self::QwenVl;
        }

        // Qwen base
        if model_lower.contains("qwen") {
            return Self::QwenBase;
        }

        // Claude
        if model_lower.contains("claude") {
            return Self::Claude;
        }

        // GPT
        if model_lower.contains("gpt") {
            return Self::Gpt;
        }

        Self::Generic
    }

    /// Check if this model family prefers XML-style tool calls.
    #[must_use]
    pub fn prefers_xml_tool_calls(&self) -> bool {
        matches!(self, Self::QwenCoder | Self::QwenVl)
    }

    /// Get the tool call example format for this model family.
    #[must_use]
    pub fn tool_call_example_format(&self) -> ToolCallExampleFormat {
        match self {
            Self::QwenCoder | Self::QwenVl => ToolCallExampleFormat::HybridXmlJson,
            _ => ToolCallExampleFormat::FunctionCall,
        }
    }
}

/// Format for tool call examples in prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallExampleFormat {
    /// Hybrid XML-JSON format: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
    /// Used by Qwen3-Next and modern Qwen models.
    HybridXmlJson,
    /// Standard function call JSON
    FunctionCall,
}

/// Platform-specific prompt variations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Unix,
    Windows,
}

impl Platform {
    /// Detect current platform.
    #[must_use]
    pub fn current() -> Self {
        if cfg!(target_os = "windows") {
            Self::Windows
        } else {
            Self::Unix
        }
    }
}

/// Model-specific prompt builder.
#[derive(Debug, Clone)]
pub struct ModelPromptBuilder {
    model_family: ModelFamily,
    platform: Platform,
}

impl Default for ModelPromptBuilder {
    fn default() -> Self {
        Self {
            model_family: ModelFamily::Generic,
            platform: Platform::current(),
        }
    }
}

impl ModelPromptBuilder {
    /// Create a new prompt builder for a specific model.
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        Self {
            model_family: ModelFamily::from_model_name(model),
            platform: Platform::current(),
        }
    }

    /// Get the model family.
    #[must_use]
    pub fn model_family(&self) -> ModelFamily {
        self.model_family
    }

    /// Get the platform.
    #[must_use]
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Generate a tool call example for the given tool.
    #[must_use]
    pub fn tool_call_example(&self, tool_name: &str, params: &[(&str, &str)]) -> String {
        match self.model_family.tool_call_example_format() {
            ToolCallExampleFormat::HybridXmlJson => {
                let args: HashMap<&str, &str> = params.iter().copied().collect();
                let args_json = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());
                format!(
                    "<tool_call>\n{{\"name\": \"{tool_name}\", \"arguments\": {args_json}}}\n</tool_call>"
                )
            }
            ToolCallExampleFormat::FunctionCall => {
                let args: HashMap<&str, &str> = params.iter().copied().collect();
                let args_json =
                    serde_json::to_string_pretty(&args).unwrap_or_else(|_| "{}".to_string());
                format!("{tool_name}({args_json})")
            }
        }
    }

    /// Get the shell command prefix for the current platform.
    #[must_use]
    pub fn shell_prefix(&self) -> &'static str {
        match self.platform {
            Platform::Unix => "bash -c",
            Platform::Windows => "cmd.exe /c",
        }
    }

    /// Get platform-specific bash/shell tool description.
    #[must_use]
    pub fn bash_description(&self) -> String {
        let shell_info = match self.platform {
            Platform::Unix => {
                "This tool executes commands as `bash -c <command>`. \
                 Commands can start background processes using `&`. \
                 The command runs in its own process group."
            }
            Platform::Windows => {
                "This tool executes commands as `cmd.exe /c <command>`. \
                 Commands can start background processes using `start /b`."
            }
        };

        format!(
            "{shell_info}\n\n\
             **Background vs Foreground Execution:**\n\
             - `is_background: true`: For long-running servers, watchers, daemons\n\
             - `is_background: false`: For one-time commands, builds, tests, git\n\n\
             **Returns:** stdout, stderr, exit code, background PIDs if any"
        )
    }

    /// Get platform-specific file path examples.
    #[must_use]
    pub fn path_example(&self) -> &'static str {
        match self.platform {
            Platform::Unix => "/path/to/your/project/src/main.rs",
            Platform::Windows => "C:\\path\\to\\your\\project\\src\\main.rs",
        }
    }

    /// Get platform-specific directory listing command.
    #[must_use]
    pub fn list_dir_command(&self) -> &'static str {
        match self.platform {
            Platform::Unix => "ls -la",
            Platform::Windows => "dir",
        }
    }

    /// Build the enhanced prompt for a tool.
    #[must_use]
    pub fn enhance_tool_prompt(&self, tool_name: &str, base_prompt: &str) -> String {
        let mut enhanced = base_prompt.to_string();

        // Add model-specific example if applicable
        if self.model_family.prefers_xml_tool_calls() {
            let example = match tool_name {
                "bash" => Some(self.tool_call_example(
                    "bash",
                    &[("command", "git status"), ("is_background", "false")],
                )),
                "read_file" => {
                    Some(self.tool_call_example("read_file", &[("path", self.path_example())]))
                }
                "grep" => {
                    Some(self.tool_call_example("grep", &[("pattern", "TODO"), ("path", "src/")]))
                }
                "write_file" => Some(self.tool_call_example(
                    "write_file",
                    &[("path", "output.txt"), ("content", "Hello, world!")],
                )),
                _ => None,
            };

            if let Some(ex) = example {
                enhanced.push_str("\n\n**Example tool call:**\n```\n");
                enhanced.push_str(&ex);
                enhanced.push_str("\n```");
            }
        }

        enhanced
    }
}

/// Get model-specific tool prompt for a given tool.
#[must_use]
pub fn get_model_tool_prompt(model: &str, tool_name: &str, base_prompt: &str) -> String {
    let builder = ModelPromptBuilder::for_model(model);
    builder.enhance_tool_prompt(tool_name, base_prompt)
}

/// Load the prompt file for a tool.
/// All prompts use Qwen3-Next native XML format since it's the only supported architecture.
#[must_use]
pub fn select_prompt_for_model(_model: &str, tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "bash" => Some(include_str!("builtins/prompts/bash.md")),
        "read_file" => Some(include_str!("builtins/prompts/read_file.md")),
        "grep" => Some(include_str!("builtins/prompts/grep.md")),
        "write_file" => Some(include_str!("builtins/prompts/write_file.md")),
        "search_replace" => Some(include_str!("builtins/prompts/search_replace.md")),
        "todo" => Some(include_str!("builtins/prompts/todo.md")),
        _ => None,
    }
}

/// Get complete model configuration including prompts and aliases.
///
/// This is a convenience struct that combines all model-specific settings.
#[derive(Debug, Clone)]
pub struct ModelToolConfig {
    /// The model family.
    pub family: ModelFamily,
    /// The prompt builder for this model.
    pub prompt_builder: ModelPromptBuilder,
    /// Tool name mappings (internal -> model).
    pub tool_name_map: HashMap<String, String>,
    /// Parameter name mappings per tool.
    pub param_name_map: HashMap<String, HashMap<String, String>>,
}

impl ModelToolConfig {
    /// Create a model tool configuration.
    #[must_use]
    pub fn for_model(model: &str) -> Self {
        use crate::tool_aliases::ToolAliasConfig;

        let family = ModelFamily::from_model_name(model);
        let prompt_builder = ModelPromptBuilder::for_model(model);
        let alias_config = ToolAliasConfig::for_model_family(family);

        // Build tool name map
        let tool_name_map: HashMap<String, String> = [
            ("bash", "run_shell_command"),
            ("grep", "grep_search"),
            ("search_replace", "edit"),
            ("todo", "todo_write"),
        ]
        .into_iter()
        .filter(|(k, _)| alias_config.has_alias(k))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

        // Build parameter name maps
        let mut param_name_map = HashMap::new();

        // read_file: path -> absolute_path
        let mut read_file_params = HashMap::new();
        read_file_params.insert("path".to_string(), "absolute_path".to_string());
        param_name_map.insert("read_file".to_string(), read_file_params);

        // write_file: path -> file_path
        let mut write_file_params = HashMap::new();
        write_file_params.insert("path".to_string(), "file_path".to_string());
        param_name_map.insert("write_file".to_string(), write_file_params);

        Self {
            family,
            prompt_builder,
            tool_name_map,
            param_name_map,
        }
    }

    /// Check if tool name aliasing is active.
    #[must_use]
    pub fn uses_aliases(&self) -> bool {
        !self.tool_name_map.is_empty()
    }

    /// Get the model-expected tool name.
    #[must_use]
    pub fn get_tool_name<'a>(&'a self, internal_name: &'a str) -> &'a str {
        self.tool_name_map
            .get(internal_name)
            .map(String::as_str)
            .unwrap_or(internal_name)
    }

    /// Get the model-expected parameter name.
    #[must_use]
    pub fn get_param_name<'a>(&'a self, tool_name: &str, internal_param: &'a str) -> &'a str {
        self.param_name_map
            .get(tool_name)
            .and_then(|m| m.get(internal_param))
            .map(String::as_str)
            .unwrap_or(internal_param)
    }

    /// Get prompt for a tool.
    #[must_use]
    pub fn get_prompt(&self, tool_name: &str) -> Option<String> {
        let family_str = match self.family {
            ModelFamily::QwenCoder => "qwen-coder",
            ModelFamily::QwenVl => "qwen-vl",
            _ => "generic",
        };

        select_prompt_for_model(family_str, tool_name)
            .map(|p| self.prompt_builder.enhance_tool_prompt(tool_name, p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_family_detection() {
        assert_eq!(
            ModelFamily::from_model_name("qwen3-coder"),
            ModelFamily::QwenCoder
        );
        assert_eq!(
            ModelFamily::from_model_name("qwen2.5-coder-32b"),
            ModelFamily::QwenCoder
        );
        assert_eq!(
            ModelFamily::from_model_name("qwen3-next"),
            ModelFamily::QwenCoder
        );
        assert_eq!(
            ModelFamily::from_model_name("qwen3next"),
            ModelFamily::QwenCoder
        );
        assert_eq!(
            ModelFamily::from_model_name("qwen2-vl"),
            ModelFamily::QwenVl
        );
        assert_eq!(
            ModelFamily::from_model_name("qwen-7b"),
            ModelFamily::QwenBase
        );
        assert_eq!(
            ModelFamily::from_model_name("claude-3-opus"),
            ModelFamily::Claude
        );
        assert_eq!(ModelFamily::from_model_name("gpt-4"), ModelFamily::Gpt);
        assert_eq!(
            ModelFamily::from_model_name("unknown-model"),
            ModelFamily::Generic
        );
    }

    #[test]
    fn test_xml_preference() {
        assert!(ModelFamily::QwenCoder.prefers_xml_tool_calls());
        assert!(ModelFamily::QwenVl.prefers_xml_tool_calls());
        assert!(!ModelFamily::Claude.prefers_xml_tool_calls());
        assert!(!ModelFamily::Generic.prefers_xml_tool_calls());
    }

    #[test]
    fn test_tool_call_example_qwen_coder() {
        let builder = ModelPromptBuilder::for_model("qwen3-coder");
        let example = builder.tool_call_example("bash", &[("command", "ls")]);

        // Now uses hybrid XML-JSON format
        assert!(example.contains("<tool_call>"));
        assert!(example.contains("\"name\": \"bash\""));
        assert!(example.contains("\"arguments\":"));
        assert!(example.contains("</tool_call>"));
    }

    #[test]
    fn test_tool_call_example_qwen3_next() {
        let builder = ModelPromptBuilder::for_model("qwen3-next");
        let example = builder.tool_call_example("bash", &[("command", "ls")]);

        // Uses hybrid XML-JSON format
        assert!(example.contains("<tool_call>"));
        assert!(example.contains("\"name\": \"bash\""));
        assert!(example.contains("\"arguments\":"));
        assert!(example.contains("</tool_call>"));
    }

    #[test]
    fn test_tool_call_example_qwen_vl() {
        let builder = ModelPromptBuilder::for_model("qwen2-vl");
        let example = builder.tool_call_example("bash", &[("command", "ls")]);

        assert!(example.contains("<tool_call>"));
        assert!(example.contains("\"name\": \"bash\""));
        assert!(example.contains("\"arguments\":"));
        assert!(example.contains("</tool_call>"));
    }

    #[test]
    fn test_tool_call_example_generic() {
        let builder = ModelPromptBuilder::for_model("gpt-4");
        let example = builder.tool_call_example("bash", &[("command", "ls")]);

        assert!(example.contains("bash("));
        assert!(!example.contains("<tool_call>"));
    }

    #[test]
    fn test_enhance_prompt() {
        let builder = ModelPromptBuilder::for_model("qwen3-next");
        let enhanced = builder.enhance_tool_prompt("bash", "Base bash prompt.");

        assert!(enhanced.contains("Base bash prompt."));
        assert!(enhanced.contains("Example tool call"));
        assert!(enhanced.contains("<tool_call>"));
        // Should use hybrid XML-JSON format
        assert!(enhanced.contains("\"name\":"));
    }

    #[test]
    fn test_select_prompt_for_model() {
        let qwen_prompt = select_prompt_for_model("qwen3-coder", "bash");
        assert!(qwen_prompt.is_some());
        assert!(qwen_prompt.unwrap().contains("is_background"));

        let generic_prompt = select_prompt_for_model("gpt-4", "bash");
        assert!(generic_prompt.is_some());
    }
}
