//! Configuration management for Vibe.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{VibeError, VibeResult};
use crate::paths::{AGENTS_DIR, CONFIG_FILE, ENV_FILE, local_config_file};
use crate::prompts::load_prompt;
use crate::session::SessionLoggingConfig;
use paramecia_llm::backend::BackendType;
use paramecia_tools::types::ToolConfig;

/// MCP server transport type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    /// HTTP transport.
    Http,
    /// Streamable HTTP transport.
    #[serde(rename = "streamable-http")]
    StreamableHttp,
    /// Standard I/O transport.
    Stdio,
}

/// MCP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server name/alias.
    pub name: String,
    /// Transport type.
    pub transport: McpTransport,
    /// URL for HTTP transports.
    #[serde(default)]
    pub url: Option<String>,
    /// Command for stdio transport.
    #[serde(default)]
    pub command: Option<String>,
    /// Arguments for stdio transport.
    #[serde(default)]
    pub args: Vec<String>,
    /// Additional headers for HTTP transports.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Environment variable containing API key.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Optional usage hint.
    #[serde(default)]
    pub prompt: Option<String>,
}

/// Provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider name.
    pub name: String,
    /// Base URL for the API.
    pub api_base: String,
    /// Environment variable for the API key.
    #[serde(default)]
    pub api_key_env_var: String,
    /// Backend type.
    #[serde(default)]
    pub backend: BackendType,
    /// Path to a local quantized model (used when backend == local).
    #[serde(default)]
    pub local_model_path: Option<String>,
    /// Path to a tokenizer.json (used when backend == local).
    #[serde(default)]
    pub local_tokenizer_path: Option<String>,
    /// Maximum tokens to generate for local decoding.
    #[serde(default)]
    pub local_max_tokens: Option<usize>,
    /// Preferred device for local decoding.
    #[serde(default)]
    pub local_device: Option<String>,
    /// Device offload mode: "none", "up", "updown", or "experts" (default: "experts").
    #[serde(default)]
    pub local_offload: Option<String>,
    /// Maximum context length for local inference (default: 16384).
    #[serde(default)]
    pub local_context_length: Option<usize>,
    /// KV cache quantization mode: "f16", "bf16", "q8", "q4" (default: "q4").
    #[serde(default)]
    pub local_kv_cache_quant: Option<String>,
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier.
    pub name: String,
    /// Provider for this model.
    pub provider: String,
    /// Alias for the model.
    #[serde(default)]
    pub alias: Option<String>,
    /// Temperature for generation (default: 0.7).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold (default: 0.8).
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-k sampling limit (default: 20).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Min-p sampling threshold (default: 0.0, disabled).
    #[serde(default)]
    pub min_p: f32,
    /// Repetition penalty (multiplicative) (default: 1.1).
    /// Divides logits of previously seen tokens by this value.
    /// A value of 1.0 means disabled.
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f32,
    /// Presence penalty (additive/flat) (default: 1.0).
    /// Subtracts this value from logits of tokens that have appeared.
    /// A value of 0.0 means disabled.
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,
    /// Thinking budget in tokens (default: 500).
    /// After this many tokens, </think> is injected to end thinking.
    #[serde(default = "default_thinking_budget")]
    pub thinking_budget: usize,
    /// Price per million input tokens.
    #[serde(default)]
    pub input_price: f64,
    /// Price per million output tokens.
    #[serde(default)]
    pub output_price: f64,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.8
}

fn default_top_k() -> usize {
    20
}

fn default_repeat_penalty() -> f32 {
    1.1
}

fn default_presence_penalty() -> f32 {
    1.0
}

fn default_thinking_budget() -> usize {
    500
}

impl ModelConfig {
    /// Get the effective alias (falls back to name).
    #[must_use]
    pub fn alias(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.name)
    }
}

/// Training variant for self-improvement.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingVariant {
    /// Remove functions from source code and evaluate ability to reimplement them.
    /// The agent must restore functionality based on compiler errors and context.
    #[default]
    DropoutRepair,
    /// Replicate mode: present partial chunks from the codebase and evaluate
    /// the model's ability to generate matching continuations.
    /// Scored using cosine similarity of embeddings + Hirschberg alignment.
    Replicate,
}

/// Configuration for Replicate training variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicateConfig {
    /// Chunk size in tokens.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Overlap between chunks in tokens.
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    /// Weight for cosine similarity in reward calculation.
    #[serde(default = "default_cosine_weight")]
    pub cosine_weight: f32,
    /// Weight for Hirschberg alignment score in reward calculation.
    #[serde(default = "default_hirschberg_weight")]
    pub hirschberg_weight: f32,
    /// File extensions to include for chunking.
    #[serde(default = "default_chunk_extensions")]
    pub extensions: Vec<String>,
}

fn default_chunk_size() -> usize {
    256
}

fn default_chunk_overlap() -> usize {
    32
}

fn default_cosine_weight() -> f32 {
    0.6
}

fn default_hirschberg_weight() -> f32 {
    0.4
}

fn default_chunk_extensions() -> Vec<String> {
    vec!["rs".to_string(), "txt".to_string(), "md".to_string()]
}

impl Default for ReplicateConfig {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
            cosine_weight: default_cosine_weight(),
            hirschberg_weight: default_hirschberg_weight(),
            extensions: default_chunk_extensions(),
        }
    }
}

/// Tuning/training configuration for local model optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningConfig {
    /// Training variant to use.
    #[serde(default)]
    pub variant: TrainingVariant,
    /// Path to the quantized model file (GGUF).
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    /// Path to the tokenizer.json file.
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,
    /// Path to the source root for self-fix training.
    #[serde(default)]
    pub source_root: Option<PathBuf>,
    /// Number of self-fix problems per generation.
    #[serde(default = "default_problems_per_generation")]
    pub problems_per_generation: usize,
    /// Total number of generations to run.
    #[serde(default = "default_generations")]
    pub generations: usize,
    /// Maximum tokens to generate per agent turn.
    #[serde(default = "default_tuning_max_tokens")]
    pub max_tokens: usize,
    /// Population size for evolution strategies.
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    /// Weight for successful compilation (0.0-1.0).
    #[serde(default = "default_compile_weight")]
    pub compile_weight: f32,
    /// Weight for passing tests (0.0-1.0).
    #[serde(default = "default_test_weight")]
    pub test_weight: f32,
    /// Weight for absence of warnings (0.0-1.0).
    #[serde(default = "default_warnings_weight")]
    pub warnings_weight: f32,
    /// Weight for absence of clippy lints (0.0-1.0).
    #[serde(default = "default_clippy_weight")]
    pub clippy_weight: f32,
    /// Timeout for compilation checks in seconds.
    #[serde(default = "default_compile_timeout")]
    pub compile_timeout_secs: u64,
    /// Timeout for test execution in seconds.
    #[serde(default = "default_test_timeout")]
    pub test_timeout_secs: u64,
    /// Number of functions to remove per problem (DropoutRepair mode).
    #[serde(default = "default_functions_to_remove")]
    pub functions_to_remove: usize,
    /// Minimum function size in lines to consider for removal.
    #[serde(default = "default_min_function_lines")]
    pub min_function_lines: usize,
    /// Maximum agent turns per problem.
    #[serde(default = "default_max_agent_turns")]
    pub max_agent_turns: usize,
    /// Configuration for Replicate training variant.
    #[serde(default)]
    pub replicate: ReplicateConfig,
    /// Path to import existing optimizer state (SafeTensors).
    #[serde(default)]
    pub import_path: Option<PathBuf>,
    /// Path to export optimizer state (SafeTensors).
    #[serde(default)]
    pub safetensors_export: Option<PathBuf>,
    /// Path to export merged GGUF model.
    #[serde(default)]
    pub gguf_export: Option<PathBuf>,
}

fn default_problems_per_generation() -> usize {
    1
}
fn default_generations() -> usize {
    10
}
fn default_tuning_max_tokens() -> usize {
    2048
}
fn default_population_size() -> usize {
    4
}
fn default_compile_weight() -> f32 {
    0.5
}
fn default_test_weight() -> f32 {
    0.3
}
fn default_warnings_weight() -> f32 {
    0.1
}
fn default_clippy_weight() -> f32 {
    0.1
}
fn default_compile_timeout() -> u64 {
    300
}
fn default_test_timeout() -> u64 {
    600
}
fn default_functions_to_remove() -> usize {
    1
}
fn default_min_function_lines() -> usize {
    3
}
fn default_max_agent_turns() -> usize {
    10
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            variant: TrainingVariant::default(),
            model_path: None,
            tokenizer_path: None,
            source_root: None,
            problems_per_generation: default_problems_per_generation(),
            generations: default_generations(),
            max_tokens: default_tuning_max_tokens(),
            population_size: default_population_size(),
            compile_weight: default_compile_weight(),
            test_weight: default_test_weight(),
            warnings_weight: default_warnings_weight(),
            clippy_weight: default_clippy_weight(),
            compile_timeout_secs: default_compile_timeout(),
            test_timeout_secs: default_test_timeout(),
            functions_to_remove: default_functions_to_remove(),
            min_function_lines: default_min_function_lines(),
            max_agent_turns: default_max_agent_turns(),
            replicate: ReplicateConfig::default(),
            import_path: None,
            safetensors_export: None,
            gguf_export: None,
        }
    }
}

/// Project context configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContextConfig {
    /// Maximum characters to include.
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    /// Default number of commits to include.
    #[serde(default = "default_commit_count")]
    pub default_commit_count: usize,
    /// Maximum doc bytes.
    #[serde(default = "default_max_doc_bytes")]
    pub max_doc_bytes: usize,
    /// Truncation buffer size.
    #[serde(default = "default_truncation_buffer")]
    pub truncation_buffer: usize,
    /// Maximum depth for directory traversal.
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    /// Maximum files to include.
    #[serde(default = "default_max_files")]
    pub max_files: usize,
    /// Maximum directories to show per level.
    #[serde(default = "default_max_dirs_per_level")]
    pub max_dirs_per_level: usize,
    /// Timeout for context gathering in seconds.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: f64,
}

fn default_max_chars() -> usize {
    16_000
}
fn default_commit_count() -> usize {
    5
}
fn default_max_doc_bytes() -> usize {
    8 * 1024
}
fn default_truncation_buffer() -> usize {
    1_000
}
fn default_max_depth() -> usize {
    2
}
fn default_max_files() -> usize {
    1000
}
fn default_max_dirs_per_level() -> usize {
    20
}
fn default_timeout_seconds() -> f64 {
    2.0
}

impl Default for ProjectContextConfig {
    fn default() -> Self {
        Self {
            max_chars: default_max_chars(),
            default_commit_count: default_commit_count(),
            max_doc_bytes: default_max_doc_bytes(),
            truncation_buffer: default_truncation_buffer(),
            max_depth: default_max_depth(),
            max_files: default_max_files(),
            max_dirs_per_level: default_max_dirs_per_level(),
            timeout_seconds: default_timeout_seconds(),
        }
    }
}

/// Main Vibe configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VibeConfig {
    /// Active model alias.
    #[serde(default = "default_active_model")]
    pub active_model: String,

    /// Whether to use vim keybindings.
    #[serde(default)]
    pub vim_keybindings: bool,

    /// Disable welcome banner animation.
    #[serde(default)]
    pub disable_welcome_banner_animation: bool,

    /// Displayed working directory (for UI display purposes).
    #[serde(default)]
    pub displayed_workdir: String,

    /// Auto-compact threshold in tokens.
    #[serde(default = "default_auto_compact_threshold")]
    pub auto_compact_threshold: u32,

    /// Enable context warnings.
    #[serde(default)]
    pub context_warnings: bool,

    /// UI theme.
    #[serde(default = "default_theme")]
    pub textual_theme: String,

    /// Custom instructions.
    #[serde(default)]
    pub instructions: String,

    /// Working directory override.
    #[serde(default, skip_serializing)]
    pub workdir: Option<PathBuf>,

    /// System prompt ID.
    #[serde(default = "default_system_prompt_id")]
    pub system_prompt_id: String,

    /// Include commit signature in context.
    #[serde(default)]
    pub include_commit_signature: bool,

    /// Include model info in prompt.
    #[serde(default = "default_true")]
    pub include_model_info: bool,

    /// Include project context.
    #[serde(default = "default_true")]
    pub include_project_context: bool,

    /// Include prompt detail.
    #[serde(default = "default_true")]
    pub include_prompt_detail: bool,

    /// Use minimal system prompt (for local models).
    /// When true, disables commit signature, model info, project context, and prompt detail.
    #[serde(default)]
    pub use_minimal_system_prompt: bool,

    /// Enable update checks.
    #[serde(default = "default_true")]
    pub enable_update_checks: bool,

    /// API timeout in seconds.
    #[serde(default = "default_api_timeout")]
    pub api_timeout: f64,

    /// Provider configurations.
    #[serde(default = "default_providers")]
    pub providers: Vec<ProviderConfig>,

    /// Model configurations.
    #[serde(default = "default_models")]
    pub models: Vec<ModelConfig>,

    /// Project context configuration.
    #[serde(default)]
    pub project_context: ProjectContextConfig,

    /// Session logging configuration.
    #[serde(default)]
    pub session_logging: SessionLoggingConfig,

    /// Tool configurations.
    #[serde(default)]
    pub tools: HashMap<String, ToolConfig>,

    /// Additional tool search paths.
    #[serde(default)]
    pub tool_paths: Vec<String>,

    /// MCP server configurations.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,

    /// Enabled tool patterns.
    #[serde(default)]
    pub enabled_tools: Vec<String>,

    /// Disabled tool patterns.
    #[serde(default)]
    pub disabled_tools: Vec<String>,

    /// Tuning/training configuration.
    #[serde(default)]
    pub tuning: TuningConfig,
}

fn default_active_model() -> String {
    "local".to_string()
}

fn default_auto_compact_threshold() -> u32 {
    200_000
}

fn default_theme() -> String {
    "textual-dark".to_string()
}

fn default_system_prompt_id() -> String {
    "cli".to_string()
}

fn default_true() -> bool {
    true
}

fn default_api_timeout() -> f64 {
    720.0
}

fn default_providers() -> Vec<ProviderConfig> {
    vec![ProviderConfig {
        name: "local".to_string(),
        api_base: String::new(),
        api_key_env_var: String::new(),
        backend: BackendType::Local,
        local_model_path: None,
        local_tokenizer_path: None,
        local_max_tokens: Some(4096),
        local_device: None,
        local_offload: None,        // Defaults to "experts" in backend
        local_context_length: None, // Defaults to 131072 in backend
        local_kv_cache_quant: None, // Defaults to "f16" in backend
    }]
}

fn default_models() -> Vec<ModelConfig> {
    vec![ModelConfig {
        name: "devstral".to_string(),
        provider: "local".to_string(),
        alias: Some("local".to_string()),
        temperature: default_temperature(),
        top_p: default_top_p(),
        top_k: default_top_k(),
        min_p: 0.0,
        repeat_penalty: default_repeat_penalty(),
        presence_penalty: default_presence_penalty(),
        thinking_budget: default_thinking_budget(),
        input_price: 0.0,
        output_price: 0.0,
    }]
}

impl Default for VibeConfig {
    fn default() -> Self {
        Self {
            active_model: default_active_model(),
            vim_keybindings: false,
            disable_welcome_banner_animation: false,
            displayed_workdir: String::new(),
            auto_compact_threshold: default_auto_compact_threshold(),
            context_warnings: false,
            textual_theme: default_theme(),
            instructions: String::new(),
            workdir: None,
            system_prompt_id: default_system_prompt_id(),
            include_commit_signature: false,
            include_model_info: true,
            include_project_context: true,
            include_prompt_detail: true,
            use_minimal_system_prompt: false,
            enable_update_checks: true,
            api_timeout: default_api_timeout(),
            providers: default_providers(),
            models: default_models(),
            project_context: ProjectContextConfig::default(),
            session_logging: SessionLoggingConfig::default(),
            tools: HashMap::new(),
            tool_paths: Vec::new(),
            mcp_servers: Vec::new(),
            enabled_tools: Vec::new(),
            disabled_tools: Vec::new(),
            tuning: TuningConfig::default(),
        }
    }
}

impl VibeConfig {
    /// Load configuration from files.
    ///
    /// Loads in order: global config, local config, agent config.
    ///
    /// # Errors
    ///
    /// Returns an error if config files cannot be read or parsed.
    pub fn load(agent: Option<&str>) -> VibeResult<Self> {
        // Load API keys from .env
        load_env_file();

        // Start with defaults
        let mut config = Self::default();

        // Load global config
        if CONFIG_FILE.exists() {
            let content = std::fs::read_to_string(&*CONFIG_FILE)?;
            let file_config: VibeConfig = toml::from_str(&content)?;
            config = merge_configs(config, file_config);
        }

        // Load local config
        let workdir = std::env::current_dir().unwrap_or_default();
        let local_config_path = local_config_file(&workdir);
        if local_config_path.exists() {
            let content = std::fs::read_to_string(&local_config_path)?;
            let file_config: VibeConfig = toml::from_str(&content)?;
            config = merge_configs(config, file_config);
        }

        // Load agent config
        if let Some(agent_name) = agent {
            let agent_config = Self::load_agent_config(agent_name)?;
            config = merge_configs(config, agent_config);
        }

        config.validate()?;
        Ok(config)
    }

    /// Load agent-specific configuration.
    fn load_agent_config(name: &str) -> VibeResult<VibeConfig> {
        let path = AGENTS_DIR.join(format!("{name}.toml"));
        if !path.exists() {
            return Err(VibeError::Config(format!(
                "Agent config '{}' not found at {}",
                name,
                path.display()
            )));
        }

        let content = std::fs::read_to_string(&path)?;
        Ok(toml::from_str(&content)?)
    }

    /// Validate the configuration.
    fn validate(&self) -> VibeResult<()> {
        // Check that active model exists
        self.get_active_model()?;

        // Note: API key validation is deferred to request time to allow
        // the application to start without API keys set. Errors will
        // occur when actual API calls are made.

        Ok(())
    }

    /// Get the effective working directory.
    #[must_use]
    pub fn effective_workdir(&self) -> PathBuf {
        self.workdir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
    }

    /// Get the system prompt content.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt cannot be loaded.
    pub fn system_prompt(&self) -> VibeResult<String> {
        load_prompt(&self.system_prompt_id).map_err(VibeError::Config)
    }

    /// Get the active model configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found.
    pub fn get_active_model(&self) -> VibeResult<&ModelConfig> {
        self.models
            .iter()
            .find(|m| m.alias() == self.active_model)
            .ok_or_else(|| {
                VibeError::Config(format!(
                    "Active model '{}' not found in configuration",
                    self.active_model
                ))
            })
    }

    /// Get the provider for a model.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider is not found.
    pub fn get_provider_for_model(&self, model: &ModelConfig) -> VibeResult<&ProviderConfig> {
        self.providers
            .iter()
            .find(|p| p.name == model.provider)
            .ok_or_else(|| {
                VibeError::Config(format!(
                    "Provider '{}' for model '{}' not found",
                    model.provider, model.name
                ))
            })
    }

    /// Check if the active model uses the local backend.
    #[must_use]
    pub fn is_using_local_backend(&self) -> bool {
        if let Ok(model) = self.get_active_model() {
            if let Ok(provider) = self.get_provider_for_model(model) {
                return provider.backend == BackendType::Local;
            }
        }
        false
    }

    /// Check if minimal system prompt should be used.
    /// Returns true only if explicitly set via configuration.
    #[must_use]
    pub fn should_use_minimal_prompt(&self) -> bool {
        self.use_minimal_system_prompt
    }

    /// Save configuration updates.
    ///
    /// # Errors
    ///
    /// Returns an error if the config cannot be saved.
    pub fn save(&self) -> VibeResult<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| VibeError::Config(format!("Failed to serialize config: {e}")))?;

        if let Some(parent) = CONFIG_FILE.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&*CONFIG_FILE, content)?;
        Ok(())
    }
}

/// Load environment variables from .env file.
fn load_env_file() {
    if ENV_FILE.exists() {
        let _ = dotenvy::from_path(&*ENV_FILE);
    }
}

/// Merge two configurations, with the second taking precedence.
fn merge_configs(base: VibeConfig, overlay: VibeConfig) -> VibeConfig {
    // For now, just use the overlay values directly
    // A more sophisticated merge could be implemented if needed
    VibeConfig {
        active_model: if overlay.active_model != default_active_model() {
            overlay.active_model
        } else {
            base.active_model
        },
        vim_keybindings: overlay.vim_keybindings || base.vim_keybindings,
        disable_welcome_banner_animation: overlay.disable_welcome_banner_animation
            || base.disable_welcome_banner_animation,
        displayed_workdir: if overlay.displayed_workdir.is_empty() {
            base.displayed_workdir
        } else {
            overlay.displayed_workdir
        },
        auto_compact_threshold: overlay.auto_compact_threshold,
        context_warnings: overlay.context_warnings || base.context_warnings,
        textual_theme: if overlay.textual_theme != default_theme() {
            overlay.textual_theme
        } else {
            base.textual_theme
        },
        instructions: if overlay.instructions.is_empty() {
            base.instructions
        } else {
            overlay.instructions
        },
        workdir: overlay.workdir.or(base.workdir),
        system_prompt_id: if overlay.system_prompt_id != default_system_prompt_id() {
            overlay.system_prompt_id
        } else {
            base.system_prompt_id
        },
        include_commit_signature: overlay.include_commit_signature,
        include_model_info: overlay.include_model_info,
        include_project_context: overlay.include_project_context,
        include_prompt_detail: overlay.include_prompt_detail,
        use_minimal_system_prompt: overlay.use_minimal_system_prompt
            || base.use_minimal_system_prompt,
        enable_update_checks: overlay.enable_update_checks,
        api_timeout: overlay.api_timeout,
        providers: if overlay.providers.is_empty() {
            base.providers
        } else {
            overlay.providers
        },
        models: if overlay.models.is_empty() {
            base.models
        } else {
            overlay.models
        },
        project_context: overlay.project_context,
        session_logging: overlay.session_logging,
        tools: {
            let mut merged = base.tools;
            merged.extend(overlay.tools);
            merged
        },
        tool_paths: {
            let mut merged = base.tool_paths;
            merged.extend(overlay.tool_paths);
            merged
        },
        mcp_servers: if overlay.mcp_servers.is_empty() {
            base.mcp_servers
        } else {
            overlay.mcp_servers
        },
        enabled_tools: if overlay.enabled_tools.is_empty() {
            base.enabled_tools
        } else {
            overlay.enabled_tools
        },
        disabled_tools: {
            let mut merged = base.disabled_tools;
            merged.extend(overlay.disabled_tools);
            merged
        },
        tuning: merge_tuning_configs(base.tuning, overlay.tuning),
    }
}

/// Merge tuning configurations.
fn merge_tuning_configs(base: TuningConfig, overlay: TuningConfig) -> TuningConfig {
    TuningConfig {
        variant: overlay.variant,
        model_path: overlay.model_path.or(base.model_path),
        tokenizer_path: overlay.tokenizer_path.or(base.tokenizer_path),
        source_root: overlay.source_root.or(base.source_root),
        problems_per_generation: overlay.problems_per_generation,
        generations: overlay.generations,
        max_tokens: overlay.max_tokens,
        population_size: overlay.population_size,
        compile_weight: overlay.compile_weight,
        test_weight: overlay.test_weight,
        warnings_weight: overlay.warnings_weight,
        clippy_weight: overlay.clippy_weight,
        compile_timeout_secs: overlay.compile_timeout_secs,
        test_timeout_secs: overlay.test_timeout_secs,
        functions_to_remove: overlay.functions_to_remove,
        min_function_lines: overlay.min_function_lines,
        max_agent_turns: overlay.max_agent_turns,
        replicate: overlay.replicate,
        import_path: overlay.import_path.or(base.import_path),
        safetensors_export: overlay.safetensors_export.or(base.safetensors_export),
        gguf_export: overlay.gguf_export.or(base.gguf_export),
    }
}
