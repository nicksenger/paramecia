//! System prompt builder.

use crate::config::ParameciaConfig;
use crate::project_context::{ProjectContextProvider, is_dangerous_directory, load_project_doc};
use crate::prompts::UtilityPrompt;
use paramecia_tools::ToolManager;
use std::path::Path;
use tracing::debug;

/// Check if a directory is a git repository.
fn is_git_repository(path: &Path) -> bool {
    let git_dir = path.join(".git");
    git_dir.exists() && git_dir.is_dir()
}

/// Get the platform name.
fn get_platform_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "freebsd") {
        "FreeBSD"
    } else {
        "Unix-like"
    }
}

/// Get the default shell used by the system.
fn get_default_shell() -> &'static str {
    if cfg!(target_os = "windows") {
        "cmd.exe"
    } else {
        "sh"
    }
}

/// Get the OS-specific system prompt section.
fn get_os_system_prompt() -> String {
    let shell = get_default_shell();
    let platform = get_platform_name();
    let mut prompt = format!(
        "The operating system is {} with shell `{}`",
        platform, shell
    );

    if cfg!(target_os = "windows") {
        prompt.push_str(&get_windows_system_prompt());
    }

    prompt
}

/// Get Windows-specific system prompt section.
fn get_windows_system_prompt() -> String {
    r#"
### COMMAND COMPATIBILITY RULES (MUST FOLLOW):
- DO NOT use Unix commands like `ls`, `grep`, `cat` - they won't work on Windows
- Use: `dir` (Windows) for directory listings
- Use: backslashes (\\) for paths
- Check command availability with: `where command` (Windows)
- Script shebang: Not applicable on Windows
### ALWAYS verify commands work on the detected platform before suggesting them"#
        .to_string()
}

/// Load user instructions from the config or instructions file.
fn load_user_instructions(config: &ParameciaConfig) -> Option<String> {
    if !config.instructions.is_empty() {
        return Some(config.instructions.clone());
    }

    // Try to load from instructions file in config directory
    let instructions_file = crate::paths::CONFIG_DIR.join("instructions.md");
    if instructions_file.exists() {
        std::fs::read_to_string(&instructions_file).ok()
    } else {
        None
    }
}

/// Convert JSON-in-XML tool call examples to pure XML format.
///
/// Finds blocks like:
/// ```text
/// <tool_call>
/// {"name": "bash", "arguments": {"command": "git status"}}
/// </tool_call>
/// ```
/// And converts them to:
/// ```text
/// <tool_call>
/// <function=bash>
/// <parameter=command>
/// git status
/// </parameter>
/// </function>
/// </tool_call>
/// ```
fn convert_tool_examples_to_xml(prompt: &str) -> String {
    let mut result = String::with_capacity(prompt.len());
    let mut remaining = prompt;

    while let Some(start) = remaining.find("<tool_call>") {
        // Copy everything before this block
        result.push_str(&remaining[..start]);

        if let Some(end) = remaining[start..].find("</tool_call>") {
            let end_abs = start + end + "</tool_call>".len();
            let inner = remaining[start + "<tool_call>".len()..start + end].trim();

            // Only convert if the inner content looks like JSON (starts with '{')
            if inner.starts_with('{') {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(inner) {
                    let name = parsed["name"].as_str().unwrap_or("unknown");
                    result.push_str("<tool_call>\n<function=");
                    result.push_str(name);
                    result.push('>');

                    if let Some(args) = parsed["arguments"].as_object() {
                        for (key, value) in args {
                            result.push_str("\n<parameter=");
                            result.push_str(key);
                            result.push_str(">\n");
                            match value {
                                serde_json::Value::String(s) => result.push_str(s),
                                other => result.push_str(&other.to_string()),
                            }
                            result.push_str("\n</parameter>");
                        }
                    }

                    result.push_str("\n</function>\n</tool_call>");
                } else {
                    // JSON parse failed, keep original
                    result.push_str(&remaining[start..end_abs]);
                }
            } else {
                // Not JSON content (already XML or other format), keep original
                result.push_str(&remaining[start..end_abs]);
            }

            remaining = &remaining[end_abs..];
        } else {
            // No closing tag found, keep the rest as-is
            result.push_str(&remaining[start..]);
            remaining = "";
        }
    }

    // Append anything after the last block
    result.push_str(remaining);
    result
}

/// Load tool prompts from the tool manager.
///
/// When `tool_call_format` is `"xml"`, any JSON-in-XML examples in the
/// tool prompts are converted to pure XML format to match the chat template.
fn load_tool_prompts(tool_manager: &ToolManager, tool_call_format: &str) -> Vec<String> {
    let mut prompts = Vec::new();
    let use_xml = tool_call_format != "json_in_xml";
    let mut tool_names = tool_manager.available_tools();
    tool_names.sort();

    for tool_name in tool_names {
        if let Ok(tool_arc) = tool_manager.get(&tool_name)
            && let Ok(Some(prompt)) =
                tool_arc.blocking_inspect(|tool| tool.prompt().map(str::to_string))
        {
            let prompt = if use_xml {
                convert_tool_examples_to_xml(&prompt)
            } else {
                prompt
            };
            prompts.push(format_tool_prompt(&tool_name, &prompt));
        }
    }

    prompts
}

fn format_tool_prompt(tool_name: &str, prompt: &str) -> String {
    format!("Tool: `{tool_name}`\n\n{}", prompt.trim())
}

/// Build the universal system prompt.
///
/// This combines the base system prompt with:
/// - Model information
/// - OS/shell information
/// - Tool prompts
/// - User instructions
/// - Project context
/// - Project documentation
pub fn get_universal_system_prompt(config: &ParameciaConfig) -> String {
    let tool_manager = ToolManager::with_configs_and_builtin_filter(
        config.tools.clone(),
        &config.builtin_tools,
        config.no_builtin_tools,
    );
    get_universal_system_prompt_with_tools(&tool_manager, config)
}

/// Build the universal system prompt with an existing tool manager.
pub fn get_universal_system_prompt_with_tools(
    tool_manager: &ToolManager,
    config: &ParameciaConfig,
) -> String {
    let mut sections = Vec::new();

    // Base system prompt
    if let Ok(base_prompt) = config.system_prompt() {
        sections.push(base_prompt);
    }

    // When using minimal system prompt (explicitly configured), skip verbose sections
    if config.should_use_minimal_prompt() {
        let prompt = sections.join("\n\n");
        debug!("Using minimal system prompt ({} chars)", prompt.len());
        return prompt;
    }

    // OS and tool prompts
    if config.include_prompt_detail {
        sections.push(get_os_system_prompt());

        // Current working directory
        let workdir = config.effective_workdir();
        sections.push(format!(
            "The current working directory is: {}",
            workdir.to_string_lossy()
        ));

        // Tool prompts (converted to match configured tool_call_format)
        let tool_prompts = load_tool_prompts(tool_manager, config.tool_call_format());
        if !tool_prompts.is_empty() {
            sections.push(format!(
                "Here is a list of tools you have available:\n\n{}",
                tool_prompts.join("\n\n---\n\n")
            ));
        }

        // User instructions
        if let Some(instructions) = load_user_instructions(config) {
            let trimmed = instructions.trim();
            if !trimmed.is_empty() {
                sections.push(trimmed.to_string());
            }
        }
    }

    // Project context
    if config.include_project_context {
        let workdir = config.effective_workdir();
        let (is_dangerous, reason) = is_dangerous_directory(&workdir);

        let context = if is_dangerous {
            let template = UtilityPrompt::DangerousDirectory.read();
            template
                .replace("{reason}", &reason.to_lowercase())
                .replace("{abs_path}", &workdir.to_string_lossy())
        } else {
            let mut provider =
                ProjectContextProvider::new(config.project_context.clone(), &workdir);
            provider.get_full_context()
        };

        sections.push(context);

        // Git repository context
        if is_git_repository(&workdir) {
            sections.push(UtilityPrompt::Git.content().to_string());
        }

        // Project documentation
        if let Some(doc) = load_project_doc(&workdir, config.project_context.max_doc_bytes) {
            let trimmed = doc.trim();
            if !trimmed.is_empty() {
                sections.push(trimmed.to_string());
            }
        }
    }

    sections.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_platform_name() {
        let name = get_platform_name();
        assert!(!name.is_empty());
    }

    #[test]
    fn test_get_default_shell() {
        let shell = get_default_shell();
        assert!(!shell.is_empty());
    }

    #[test]
    fn test_get_os_system_prompt() {
        let prompt = get_os_system_prompt();
        assert!(prompt.contains("operating system"));
    }

    #[test]
    fn test_convert_tool_examples_to_xml_simple() {
        let input = r#"Use the `bash` tool.

**Example:**

<tool_call>
{"name": "bash", "arguments": {"command": "git status"}}
</tool_call>
"#;
        let result = convert_tool_examples_to_xml(input);
        assert!(
            result.contains("<function=bash>"),
            "Should contain XML function tag, got: {}",
            result
        );
        assert!(
            result.contains("<parameter=command>"),
            "Should contain XML parameter tag, got: {}",
            result
        );
        assert!(
            result.contains("git status"),
            "Should contain the command value, got: {}",
            result
        );
        assert!(
            !result.contains("\"name\":"),
            "Should not contain JSON name field, got: {}",
            result
        );
    }

    #[test]
    fn test_convert_tool_examples_to_xml_multiple() {
        let input = r#"**Example 1:**

<tool_call>
{"name": "grep", "arguments": {"pattern": "fn main", "path": "src/"}}
</tool_call>

**Example 2:**

<tool_call>
{"name": "grep", "arguments": {"pattern": "TODO", "path": "."}}
</tool_call>
"#;
        let result = convert_tool_examples_to_xml(input);
        // Both blocks should be converted
        assert_eq!(
            result.matches("<function=grep>").count(),
            2,
            "Should have 2 function tags, got: {}",
            result
        );
        assert!(
            !result.contains("\"name\":"),
            "Should not contain any JSON, got: {}",
            result
        );
    }

    #[test]
    fn test_convert_tool_examples_preserves_non_toolcall_text() {
        let input = "Regular text without any tool calls should be unchanged.";
        let result = convert_tool_examples_to_xml(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_convert_tool_examples_search_replace() {
        // The search_replace prompt has complex JSON with escaped newlines and quotes
        let input = r#"**Example:**

<tool_call>
{"name": "search_replace", "arguments": {"file_path": "/path/to/src/main.rs", "content": "<<<<<<< SEARCH\nfn old() {\n    println!(\"old\");\n}\n=======\nfn new() {\n    println!(\"new\");\n}\n>>>>>>> REPLACE"}}
</tool_call>

**Multiple replacements:** Set `replace_all` to true."#;
        let result = convert_tool_examples_to_xml(input);
        assert!(
            result.contains("<function=search_replace>"),
            "Should contain XML function tag, got: {}",
            result
        );
        assert!(
            result.contains("<parameter=file_path>"),
            "Should contain file_path parameter, got: {}",
            result
        );
        assert!(
            result.contains("<parameter=content>"),
            "Should contain content parameter, got: {}",
            result
        );
        // The JSON-escaped \n should be unescaped to real newlines
        assert!(
            result.contains("<<<<<<< SEARCH"),
            "Should contain SEARCH marker, got: {}",
            result
        );
        assert!(
            result.contains(">>>>>>> REPLACE"),
            "Should contain REPLACE marker, got: {}",
            result
        );
        assert!(
            !result.contains("\"name\":"),
            "Should not contain JSON, got: {}",
            result
        );
    }

    #[test]
    fn test_format_tool_prompt_includes_name_header() {
        let result = format_tool_prompt("todo", "Manage a structured todo list.");
        assert!(result.starts_with("Tool: `todo`"));
        assert!(result.contains("Manage a structured todo list."));
    }

    #[test]
    fn test_universal_system_prompt_has_delineated_tool_section() {
        let mut config = ParameciaConfig::default();
        config.include_project_context = false;
        config.instructions.clear();

        let tool_manager = ToolManager::with_configs_and_builtin_filter(
            config.tools.clone(),
            &config.builtin_tools,
            config.no_builtin_tools,
        );
        let prompt = get_universal_system_prompt_with_tools(&tool_manager, &config);

        assert!(prompt.contains("The operating system is"));
        assert!(prompt.contains("The current working directory is:"));
        assert!(prompt.contains("Here is a list of tools you have available:"));
        assert!(prompt.contains("Tool: `bash`"));
        assert!(prompt.contains("Tool: `grep`"));
        assert!(prompt.contains("Tool: `read_file`"));
        assert!(prompt.contains("Tool: `search_replace`"));
        assert!(prompt.contains("Tool: `todo`"));
        assert!(prompt.contains("Tool: `write_file`"));
    }

    #[test]
    fn test_universal_system_prompt_respects_builtin_tool_filter() {
        let mut config = ParameciaConfig::default();
        config.include_project_context = false;
        config.instructions.clear();
        config.builtin_tools = vec!["grep".to_string(), "read_file".to_string()];

        let tool_manager = ToolManager::with_configs_and_builtin_filter(
            config.tools.clone(),
            &config.builtin_tools,
            config.no_builtin_tools,
        );
        let prompt = get_universal_system_prompt_with_tools(&tool_manager, &config);

        assert!(prompt.contains("Tool: `grep`"));
        assert!(prompt.contains("Tool: `read_file`"));
        assert!(!prompt.contains("Tool: `bash`"));
        assert!(!prompt.contains("Tool: `write_file`"));
        assert!(!prompt.contains("Tool: `search_replace`"));
        assert!(!prompt.contains("Tool: `todo`"));
    }

    #[test]
    fn test_universal_system_prompt_can_disable_all_builtin_tools() {
        let mut config = ParameciaConfig::default();
        config.include_project_context = false;
        config.instructions.clear();
        config.no_builtin_tools = true;

        let tool_manager = ToolManager::with_configs_and_builtin_filter(
            config.tools.clone(),
            &config.builtin_tools,
            config.no_builtin_tools,
        );
        let prompt = get_universal_system_prompt_with_tools(&tool_manager, &config);

        assert!(!prompt.contains("Here is a list of tools you have available:"));
        assert!(!prompt.contains("Tool: `bash`"));
        assert!(!prompt.contains("Tool: `grep`"));
        assert!(!prompt.contains("Tool: `read_file`"));
        assert!(!prompt.contains("Tool: `search_replace`"));
        assert!(!prompt.contains("Tool: `todo`"));
        assert!(!prompt.contains("Tool: `write_file`"));
    }
}
