//! Example demonstrating the LocalBackend in an agentic loop.
//!
//! This example exercises the LocalBackend similarly to how it would be used
//! in the actual agentic harness, with tool definitions and multi-turn
//! conversation support. Always uses streaming mode.
//!
//! Usage:
//!   cargo run --features=cuda --release -p paramecia-llm --example=local_agent \
//!     -- --model-path=/path/to/model.gguf --prompt="List files in the current directory"
//!
//! To match TUI harness behavior EXACTLY (same system prompt, tools, and context):
//!   cargo run --features=cuda --release -p paramecia-llm --example=local_agent \
//!     -- --model-path=/path/to/model.gguf --include-context \
//!     --prompt="What files are in this project?"
//!
//! This uses the same `get_universal_system_prompt_with_tools` function as the TUI,
//! ensuring identical system prompt, tool prompts, and project context.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use futures::StreamExt;
use paramecia_harness::{
    VibeConfig, generate_multishot_examples, get_universal_system_prompt_with_tools,
};
use paramecia_llm::backend::{
    Backend, BackendType, CompletionOptions, LocalBackend, ModelConfig, ProviderConfig,
};
use paramecia_llm::{AvailableTool, LlmMessage, Role, StrToolChoice, ToolChoice};
use paramecia_tools::ToolManager;

#[derive(Parser, Debug)]
#[command(author, version, about = "Local agent example", long_about = None)]
struct Args {
    /// Path to the GGUF model file.
    #[arg(long)]
    model_path: String,

    /// User prompt to send to the agent.
    #[arg(
        long,
        default_value = "Please list files in the current directory using the bash tool."
    )]
    prompt: String,

    /// Maximum tokens to generate per turn.
    #[arg(long, default_value = "1024")]
    max_tokens: usize,

    /// Maximum context length.
    #[arg(long, default_value = "131072")]
    context_length: usize,

    /// Device offload mode: none, up, updown, or experts.
    #[arg(long, default_value = "experts")]
    offload: String,

    /// Run on CPU only.
    #[arg(long)]
    cpu: bool,

    /// Maximum number of agentic turns.
    #[arg(long, default_value = "5")]
    max_turns: usize,

    /// Disable tool definitions.
    #[arg(long)]
    no_tools: bool,

    /// Include project context (directory structure, git status) matching the TUI.
    /// Uses current working directory.
    #[arg(long)]
    include_context: bool,

    /// Working directory for project context (defaults to current directory).
    #[arg(long)]
    workdir: Option<PathBuf>,

    /// Add extra padding tokens to the system prompt for OOM testing.
    /// Specify the approximate number of extra tokens to add.
    #[arg(long, default_value = "0")]
    extra_tokens: usize,

    /// Print the full system prompt as provided to the agent.
    #[arg(long)]
    print_system_prompt: bool,

    /// Print the rendered prompt that is actually sent to the model (includes tools via chat template).
    #[arg(long)]
    print_rendered_prompt: bool,
}

/// Get tools from the ToolManager (matching TUI behavior).
fn get_tools_from_manager(tool_manager: &ToolManager) -> Vec<AvailableTool> {
    tool_manager
        .tool_infos()
        .into_iter()
        .map(|info| AvailableTool::function(info.name, info.description, info.parameters))
        .collect()
}

/// Execute a tool using the ToolManager.
async fn execute_tool_via_manager(
    tool_manager: &ToolManager,
    name: &str,
    args: &serde_json::Value,
) -> String {
    match tool_manager.get(name) {
        Ok(tool_instance) => {
            match tool_instance.execute(args.clone()).await {
                Ok(result) => {
                    // Convert the result Value to a string for the LLM
                    if let Some(s) = result.as_str() {
                        s.to_string()
                    } else {
                        serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
                    }
                }
                Err(e) => format!("Tool execution error: {}", e),
            }
        }
        Err(e) => format!("Tool not found: {} - {}", name, e),
    }
}

/// Format the prompt as it would be sent to the model (for display purposes).
/// This replicates the LocalBackend::build_prompt logic to show the full formatted prompt.
fn format_prompt_for_display(messages: &[LlmMessage], tools: Option<&[AvailableTool]>) -> String {
    let mut prompt = String::new();

    let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);
    let first_is_system = messages
        .first()
        .map(|m| m.role == Role::System)
        .unwrap_or(false);

    if has_tools {
        prompt.push_str("<|im_start|>system\n");

        if first_is_system {
            if let Some(content) = &messages[0].content {
                prompt.push_str(content);
                prompt.push_str("\n\n");
            }
        }

        // Add tool definitions in Qwen3-Next native XML format
        prompt.push_str("# Tools\n\nYou have access to the following tools:\n\n");

        if let Some(tools) = tools {
            for tool in tools {
                prompt.push_str("## ");
                prompt.push_str(&tool.function.name);
                prompt.push_str("\n\n");
                prompt.push_str(&tool.function.description);
                prompt.push_str("\n\n");

                // Format parameters from JSON Schema
                let params = &tool.function.parameters;
                if let Some(properties) = params.get("properties").and_then(|p| p.as_object()) {
                    let required: Vec<&str> = params
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                        .unwrap_or_default();

                    prompt.push_str("**Parameters:**\n\n");

                    for (name, schema) in properties {
                        let param_type = schema
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("string");
                        let description = schema
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");
                        let is_required = required.contains(&name.as_str());

                        prompt.push_str("- `");
                        prompt.push_str(name);
                        prompt.push_str("` (");
                        prompt.push_str(param_type);
                        if is_required {
                            prompt.push_str(", required");
                        }
                        prompt.push_str("): ");
                        prompt.push_str(description);
                        prompt.push('\n');
                    }
                    prompt.push('\n');
                }

                // Add example usage with hybrid XML-JSON format
                prompt.push_str("**Usage:**\n\n");
                prompt.push_str("<tool_call>\n{\"name\": \"");
                prompt.push_str(&tool.function.name);
                prompt.push_str("\", \"arguments\": {");

                if let Some(properties) = params.get("properties").and_then(|p| p.as_object()) {
                    let mut first = true;
                    for (name, _) in properties.iter().take(2) {
                        if !first {
                            prompt.push_str(", ");
                        }
                        prompt.push_str("\"");
                        prompt.push_str(name);
                        prompt.push_str("\": \"value\"");
                        first = false;
                    }
                }

                prompt.push_str("}}\n</tool_call>\n\n");
            }
        }

        prompt
            .push_str("\n## Tool Call Format\n\nTo call a tool, use a JSON object within <tool_call></tool_call> tags:\n\n");
        prompt.push_str("<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param_name\": \"value\"}}\n</tool_call>\n\n");
        prompt.push_str("You may call multiple tools in sequence. Always wait for tool results before proceeding.\n");
        prompt.push_str("<|im_end|>\n");
    } else if first_is_system {
        prompt.push_str("<|im_start|>system\n");
        if let Some(content) = &messages[0].content {
            prompt.push_str(content);
        }
        prompt.push_str("<|im_end|>\n");
    }

    // Add conversation messages (skip first system message if already handled)
    let start_idx = if first_is_system { 1 } else { 0 };
    for message in &messages[start_idx..] {
        match message.role {
            Role::System => {
                prompt.push_str("<|im_start|>system\n");
                if let Some(content) = &message.content {
                    prompt.push_str(content);
                }
                prompt.push_str("<|im_end|>\n");
            }
            Role::User => {
                prompt.push_str("<|im_start|>user\n");
                if let Some(content) = &message.content {
                    prompt.push_str(content);
                }
                prompt.push_str("<|im_end|>\n");
            }
            Role::Assistant => {
                prompt.push_str("<|im_start|>assistant\n");
                if let Some(content) = &message.content {
                    prompt.push_str(content);
                }
                prompt.push_str("<|im_end|>\n");
            }
            Role::Tool => {
                prompt.push_str("<|im_start|>user\n<tool_response>\n");
                if let Some(content) = &message.content {
                    prompt.push_str(content);
                }
                prompt.push_str("\n</tool_response><|im_end|>\n");
            }
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Get the system prompt using the exact same function as the TUI harness.
/// This ensures the example uses the identical system prompt as the TUI.
fn get_system_prompt_from_harness(
    tool_manager: &ToolManager,
    include_context: bool,
    workdir: Option<&PathBuf>,
) -> String {
    // Load config from files exactly like the TUI does
    // This picks up user instructions, custom prompts, etc.
    let mut config = match VibeConfig::load(None) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Warning: Failed to load config, using defaults: {}", e);
            VibeConfig::default()
        }
    };

    // Enable/disable project context based on flag
    config.include_project_context = include_context;

    // Set workdir if provided
    if let Some(wd) = workdir {
        config.workdir = Some(wd.clone());
    }

    // Use the exact same system prompt builder as the TUI
    get_universal_system_prompt_with_tools(tool_manager, &config)
}

/// Generate padding text to add extra tokens to the prompt for OOM testing.
/// Uses ~4 characters per token as a rough estimate.
fn generate_padding_text(target_tokens: usize) -> String {
    if target_tokens == 0 {
        return String::new();
    }

    // Each line is roughly 80 chars = ~20 tokens
    // We'll use varied text to avoid compression
    let lines = [
        "The quick brown fox jumps over the lazy dog near the riverbank at sunset.",
        "A wise programmer always tests their code before deploying to production.",
        "Machine learning models require careful tuning of hyperparameters for best results.",
        "Rust provides memory safety without garbage collection through its ownership system.",
        "Neural networks can learn complex patterns from large amounts of training data.",
        "The architecture of modern CPUs includes multiple levels of cache memory.",
        "Distributed systems must handle network partitions and eventual consistency.",
        "Functional programming emphasizes immutability and pure functions without side effects.",
        "Database indexes improve query performance but increase storage requirements.",
        "Version control systems track changes to source code over time for collaboration.",
    ];

    let chars_needed = target_tokens * 4; // ~4 chars per token
    let mut result = String::with_capacity(chars_needed + 100);
    result.push_str("\n\n--- Additional Context for Testing ---\n\n");

    let mut i = 0;
    while result.len() < chars_needed {
        result.push_str(lines[i % lines.len()]);
        result.push('\n');
        i += 1;
    }

    result
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("=== Local Agent Example ===\n");
    println!("Model: {}", args.model_path);
    println!("Context length: {} tokens", args.context_length);
    println!("Offload mode: {}", args.offload);
    println!("Max tokens per turn: {}", args.max_tokens);
    println!("Max turns: {}", args.max_turns);
    println!();

    // Create provider config
    let provider = ProviderConfig {
        name: "local".to_string(),
        api_base: String::new(),
        api_key_env_var: String::new(),
        backend: BackendType::Local,
        local_model_path: Some(args.model_path.clone()),
        local_tokenizer_path: None,
        local_max_tokens: Some(args.max_tokens),
        local_device: Some(if args.cpu {
            "cpu".to_string()
        } else {
            "cuda".to_string()
        }),
        local_offload: Some(args.offload.clone()),
        local_context_length: Some(args.context_length),
        local_kv_cache_quant: None, // Use default (Q4K)
    };

    println!("Loading model...");
    let start = std::time::Instant::now();
    let local_backend = LocalBackend::new(provider, Duration::from_secs(300))?;
    println!("Model loaded in {:.1}s\n", start.elapsed().as_secs_f64());

    // Keep a reference to the chat template for display
    let chat_template = local_backend.chat_template().clone();

    let backend: Arc<dyn Backend> = Arc::new(local_backend);

    // Setup ToolManager if requested (matches TUI behavior exactly)
    let tool_manager = ToolManager::with_configs(HashMap::new());

    // Setup tools
    let tools = if args.no_tools {
        vec![]
    } else {
        get_tools_from_manager(&tool_manager)
    };

    // Model config - use recommended sampling settings for Qwen3-Next
    // temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, repeat_penalty=1.05, presence_penalty=1.0, thinking_budget=500
    let model_config = ModelConfig::new("qwen3-next");
    println!(
        "Sampling: temperature={}, top_p={}, top_k={}, min_p={}, repeat_penalty={}, presence_penalty={}, thinking_budget={}",
        model_config.temperature,
        model_config.top_p,
        model_config.top_k,
        model_config.min_p,
        model_config.repeat_penalty,
        model_config.presence_penalty,
        model_config.thinking_budget
    );

    // Build messages with system prompt
    // Use the exact same system prompt as the TUI harness
    let mut system_prompt =
        get_system_prompt_from_harness(&tool_manager, args.include_context, args.workdir.as_ref());

    // Add padding tokens if requested (for OOM testing)
    if args.extra_tokens > 0 {
        let padding = generate_padding_text(args.extra_tokens);
        system_prompt.push_str(&padding);
        println!("Added ~{} extra padding tokens", args.extra_tokens);
    }

    println!(
        "System prompt: {} chars (full{})",
        system_prompt.len(),
        if args.include_context {
            " + context"
        } else {
            ""
        }
    );
    println!("Tools: {} tools", tools.len());

    // Print full system prompt if requested
    if args.print_system_prompt {
        println!("\n=== CHAT TEMPLATE (from GGUF) ===\n");
        println!("{}", chat_template.template_string());
        println!("\n=== END CHAT TEMPLATE ===\n");

        println!("=== SYSTEM PROMPT TEXT ===\n");
        println!("{}", system_prompt);
        println!("\n=== END SYSTEM PROMPT TEXT ===\n");

        // Also print the full formatted prompt with tool definitions as sent to the model
        println!("=== FULL FORMATTED PROMPT (with tools) ===\n");
        let temp_messages = vec![
            LlmMessage::system(&system_prompt),
            LlmMessage::user("(example user message)"),
        ];
        let tools_for_print = if tools.is_empty() {
            None
        } else {
            Some(tools.as_slice())
        };
        // Use the chat template to format the prompt
        match chat_template.apply(&temp_messages, tools_for_print) {
            Ok(formatted) => println!("{}", formatted),
            Err(e) => {
                println!("(Chat template failed: {}, using fallback)", e);
                let formatted = format_prompt_for_display(&temp_messages, tools_for_print);
                println!("{}", formatted);
            }
        }
        println!("\n=== END FULL FORMATTED PROMPT ===\n");
    }
    if args.include_context {
        let workdir = args
            .workdir
            .clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        println!("Project context: {}", workdir.display());
    }

    // Build messages with system prompt and multishot examples (matching TUI Agent behavior)
    let mut messages = vec![LlmMessage::system(&system_prompt)];

    // Add multishot examples to teach the model tool usage format
    // This matches the behavior of the Agent in paramecia-harness
    let workdir = args
        .workdir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let multishot_examples = generate_multishot_examples(&tool_manager, &workdir);
    println!("Multishot examples: {} messages", multishot_examples.len());
    messages.extend(multishot_examples);

    // Add user message
    messages.push(LlmMessage::user(&args.prompt));

    // Count tokens in the initial prompt
    let tools_ref = if tools.is_empty() {
        None
    } else {
        Some(tools.as_slice())
    };
    let initial_tokens = backend
        .count_tokens(&model_config, &messages, tools_ref)
        .await?;
    println!("Initial prompt: {} tokens\n", initial_tokens);

    // Print the actual rendered prompt that will be sent to the model
    if args.print_system_prompt || args.print_rendered_prompt {
        println!("=== RENDERED PROMPT (sent to model via chat template) ===\n");
        match chat_template.apply(&messages, tools_ref) {
            Ok(rendered) => {
                println!("{}", rendered);
                println!("\n--- Prompt stats ---");
                println!("Total chars: {}", rendered.len());
                println!("Messages: {}", messages.len());
                println!(
                    "Tools included: {}",
                    tools_ref.map(|t| t.len()).unwrap_or(0)
                );
                if let Some(tools) = tools_ref {
                    println!(
                        "Tool names: {:?}",
                        tools.iter().map(|t| &t.function.name).collect::<Vec<_>>()
                    );
                }
            }
            Err(e) => {
                println!("(Failed to render chat template: {})", e);
                println!("This means tools may not be properly included in the prompt!");
            }
        }
        println!("\n=== END RENDERED PROMPT ===\n");
    }

    println!("=== Conversation ===\n");
    println!("User: {}\n", args.prompt);

    // Agentic loop
    for turn in 0..args.max_turns {
        println!("--- Turn {} ---", turn + 1);

        let options = CompletionOptions {
            max_tokens: Some(args.max_tokens as u32),
            tool_choice: if tools.is_empty() {
                None
            } else {
                Some(ToolChoice::String(StrToolChoice::Auto))
            },
            ..Default::default()
        };

        let tools_ref = if tools.is_empty() {
            None
        } else {
            Some(tools.as_slice())
        };

        let start = std::time::Instant::now();

        let mut stream = backend
            .complete_streaming(&model_config, &messages, tools_ref, &options)
            .await?;

        let mut full_content = String::new();
        let mut tool_calls = Vec::new();
        let mut finish_reason = None;
        let mut usage = None;

        print!("\nAssistant: ");
        std::io::stdout().flush()?;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;

            // Capture finish_reason and usage from final chunk
            if chunk.finish_reason.is_some() {
                finish_reason = chunk.finish_reason.clone();
            }
            if chunk.usage.is_some() {
                usage = chunk.usage.clone();
            }

            // Handle tool calls
            if let Some(tcs) = &chunk.message.tool_calls {
                for tc in tcs {
                    // Merge or add tool call
                    let idx = tc.index.unwrap_or(0) as usize;
                    while tool_calls.len() <= idx {
                        tool_calls.push(paramecia_llm::ToolCall {
                            id: None,
                            index: Some(tool_calls.len()),
                            function: paramecia_llm::FunctionCall {
                                name: None,
                                arguments: None,
                            },
                            r#type: "function".to_string(),
                        });
                    }
                    if tc.id.is_some() {
                        tool_calls[idx].id = tc.id.clone();
                    }
                    if tc.function.name.is_some() {
                        tool_calls[idx].function.name = tc.function.name.clone();
                    }
                    if let Some(args) = &tc.function.arguments {
                        let current = tool_calls[idx]
                            .function
                            .arguments
                            .get_or_insert_with(String::new);
                        current.push_str(args);
                    }
                }
            }

            // Print content as it arrives (streaming!)
            if let Some(content) = &chunk.message.content {
                if !content.is_empty() {
                    print!("{}", content);
                    std::io::stdout().flush()?;
                    full_content.push_str(content);
                }
            }
        }
        println!(); // Newline after streaming content

        // Build the final message
        let final_tool_calls =
            if tool_calls.is_empty() || tool_calls.iter().all(|tc| tc.function.name.is_none()) {
                None
            } else {
                Some(tool_calls)
            };

        let result = paramecia_llm::LlmChunk {
            message: LlmMessage {
                role: Role::Assistant,
                content: if full_content.is_empty() {
                    None
                } else {
                    Some(full_content)
                },
                tool_calls: final_tool_calls,
                name: None,
                tool_call_id: None,
            },
            finish_reason,
            usage,
        };

        let duration = start.elapsed().as_secs_f64();

        // Check for tool calls
        if let Some(tool_calls) = &result.message.tool_calls {
            if !tool_calls.is_empty() {
                println!("\n[Tool calls detected]");

                // Add the assistant message with tool calls to history
                messages.push(result.message.clone());

                // Execute each tool
                for tc in tool_calls {
                    let tool_name = tc.function.name.as_deref().unwrap_or("unknown");
                    let tool_args: serde_json::Value = tc
                        .function
                        .arguments
                        .as_ref()
                        .and_then(|a| serde_json::from_str(a).ok())
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    println!("\n> Calling tool: {} with args: {}", tool_name, tool_args);

                    // Execute the tool via ToolManager
                    let tool_result =
                        execute_tool_via_manager(&tool_manager, tool_name, &tool_args).await;
                    println!(
                        "< Result: {}",
                        if tool_result.len() > 200 {
                            format!("{}...(truncated)", &tool_result[..200])
                        } else {
                            tool_result.clone()
                        }
                    );

                    // Add tool response to messages
                    let tool_call_id = tc.id.clone().unwrap_or_else(|| format!("call_{}", turn));
                    messages.push(LlmMessage::tool(&tool_call_id, tool_name, tool_result));
                }

                if let Some(usage) = &result.usage {
                    println!(
                        "\n[Tokens: {} prompt, {} completion | {:.1}s | {:.1} tok/s]",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        duration,
                        usage.completion_tokens as f64 / duration
                    );
                }

                // Continue the loop for tool calls
                continue;
            }
        }

        // No tool calls - add message and potentially break
        messages.push(result.message.clone());

        if let Some(usage) = &result.usage {
            println!(
                "\n[Tokens: {} prompt, {} completion | {:.1}s | {:.1} tok/s]",
                usage.prompt_tokens,
                usage.completion_tokens,
                duration,
                usage.completion_tokens as f64 / duration
            );
        }

        // Check finish reason
        if result.finish_reason.as_deref() == Some("stop") {
            println!("\n[Conversation ended naturally]");
            break;
        }

        // If no tool calls and we have content, we're done
        if result.message.content.is_some() && result.message.tool_calls.is_none() {
            println!("\n[Response complete]");
            break;
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
