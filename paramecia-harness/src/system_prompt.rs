//! System prompt builder.

use crate::config::VibeConfig;
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
fn load_user_instructions(config: &VibeConfig) -> Option<String> {
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

/// Load tool prompts from the tool manager.
fn load_tool_prompts(tool_manager: &ToolManager) -> Vec<String> {
    let mut prompts = Vec::new();

    for tool_name in tool_manager.available_tools() {
        if let Ok(tool_arc) = tool_manager.get(&tool_name)
            && let Ok(Some(prompt)) =
                tool_arc.blocking_inspect(|tool| tool.prompt().map(str::to_string))
        {
            prompts.push(prompt);
        }
    }

    prompts
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
///
/// Note: Multishot examples demonstrating tool usage are also added as
/// conversation turns in agent.rs. See multishot_examples.rs for details.
pub fn get_universal_system_prompt(config: &VibeConfig) -> String {
    let tool_manager = ToolManager::with_configs(config.tools.clone());
    get_universal_system_prompt_with_tools(&tool_manager, config)
}

/// Build the universal system prompt with an existing tool manager.
pub fn get_universal_system_prompt_with_tools(
    tool_manager: &ToolManager,
    config: &VibeConfig,
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

        // Tool prompts
        let tool_prompts = load_tool_prompts(tool_manager);
        if !tool_prompts.is_empty() {
            sections.push(tool_prompts.join("\n---\n"));
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
}
