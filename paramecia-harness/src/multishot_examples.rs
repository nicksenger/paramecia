//! Multishot examples for teaching the model tool usage through examples.
//!
//! Instead of providing verbose tool-specific prompts in the system message,
//! we provide a series of example conversation turns that demonstrate how to
//! use each tool correctly. This approach is more effective for teaching the
//! model the correct tool call format.

use paramecia_llm::LlmMessage;
use paramecia_tools::ToolManager;
use std::path::Path;

/// Generate multishot example messages that demonstrate tool usage.
///
/// These messages are injected into the conversation history after the system
/// prompt to teach the model the correct format for tool calls.
pub fn generate_multishot_examples(
    tool_manager: &ToolManager,
    workdir: &Path,
) -> Vec<LlmMessage> {
    let mut messages = Vec::new();

    // Only add examples for tools that are actually available
    let available = tool_manager.available_tools();

    // Example 1: read_file
    if available.iter().any(|t| t == "read_file") {
        let example_path = workdir.join("README.md");
        let example_path_str = example_path.to_string_lossy();
        messages.push(LlmMessage::user(format!(
            "Please use the read_file tool to read {}",
            example_path_str
        )));
        messages.push(LlmMessage::assistant(format!(
            "Ok, I'll read the file at the path specified.\n\n\
                <tool_call>\n\
                {{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{}\"}}}}\n\
                </tool_call>",
            example_path_str
        )));
    }

    // Example 2: bash
    if available.iter().any(|t| t == "bash") {
        messages.push(LlmMessage::user(
            "Good job, now please use the bash tool to list files in the current directory",
        ));
        messages.push(LlmMessage::assistant(
            "Ok, I'll list the files in the current directory using bash.\n\n\
            <tool_call>\n\
            {\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}\n\
            </tool_call>",
        ));
    }

    // Example 3: grep
    if available.iter().any(|t| t == "grep") {
        messages.push(LlmMessage::user(
            "Excellent. Now use the grep tool to search for 'TODO' in the src directory",
        ));
        messages.push(LlmMessage::assistant(
            "I'll search for 'TODO' comments in the source code.\n\n\
            <tool_call>\n\
            {\"name\": \"grep\", \"arguments\": {\"pattern\": \"TODO\", \"path\": \"src/\"}}\n\
            </tool_call>",
        ));
    }

    // Example 4: write_file
    if available.iter().any(|t| t == "write_file") {
        messages.push(LlmMessage::user(
            "Great. Now write a simple hello world to /tmp/hello.txt",
        ));
        messages.push(LlmMessage::assistant(
            "I'll create the file with hello world content.\n\n\
            <tool_call>\n\
            {\"name\": \"write_file\", \"arguments\": {\"path\": \"/tmp/hello.txt\", \"content\": \"Hello, World!\"}}\n\
            </tool_call>",
        ));
    }

    // Example 5: search_replace
    if available.iter().any(|t| t == "search_replace") {
        messages.push(LlmMessage::user(
            "Perfect. Now use search_replace to change 'foo' to 'bar' in /tmp/test.txt",
        ));
        messages.push(LlmMessage::assistant(
            "I'll replace 'foo' with 'bar' in the file.\n\n\
            <tool_call>\n\
            {\"name\": \"search_replace\", \"arguments\": {\"file_path\": \"/tmp/test.txt\", \"content\": \"<<<<<<< SEARCH\\nfoo\\n=======\\nbar\\n>>>>>>> REPLACE\"}}\n\
            </tool_call>",
        ));
    }

    // Final acknowledgment to transition to real conversation
    if !messages.is_empty() {
        messages.push(LlmMessage::user(
            "Excellent work! You've demonstrated correct tool usage. Now let's begin the real task.",
        ));
        messages.push(LlmMessage::assistant(
            "Understood! I'm ready to help you with your task. I'll use the tools as demonstrated \
            when needed. What would you like me to do?",
        ));
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use paramecia_llm::Role;

    #[test]
    fn test_generate_multishot_examples() {
        let tool_manager = ToolManager::new();
        let workdir = std::env::current_dir().unwrap_or_default();
        let messages = generate_multishot_examples(&tool_manager, &workdir);

        // Should have examples for each built-in tool
        assert!(!messages.is_empty());

        // Messages should alternate user/assistant
        for (i, msg) in messages.iter().enumerate() {
            if i % 2 == 0 {
                assert_eq!(msg.role, Role::User);
            } else {
                assert_eq!(msg.role, Role::Assistant);
            }
        }

        // Assistant messages should contain tool_call format
        let assistant_msgs: Vec<_> = messages
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .collect();

        for msg in &assistant_msgs[..assistant_msgs.len() - 1] {
            // Skip the final acknowledgment
            let content = msg.content.as_ref().unwrap();
            assert!(
                content.contains("<tool_call>"),
                "Expected tool_call in: {}",
                content
            );
        }
    }

    #[test]
    fn test_multishot_examples_contain_key_tools() {
        let tool_manager = ToolManager::new();
        let workdir = std::env::current_dir().unwrap_or_default();
        let messages = generate_multishot_examples(&tool_manager, &workdir);

        let all_content: String = messages
            .iter()
            .filter_map(|m| m.content.as_ref())
            .cloned()
            .collect();

        // Check that key tools are demonstrated
        assert!(all_content.contains("read_file"));
        assert!(all_content.contains("bash"));
        assert!(all_content.contains("grep"));
        assert!(all_content.contains("write_file"));
        assert!(all_content.contains("search_replace"));
    }
}
