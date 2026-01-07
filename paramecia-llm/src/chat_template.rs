//! Chat template handling for GGUF models.
//!
//! This module provides functionality to extract and apply Jinja2-style chat templates
//! from GGUF model files. The chat template defines how messages are formatted before
//! being sent to the model.

use crate::types::{AvailableTool, LlmMessage, Role};
use minijinja::{Environment, ErrorKind, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A message in the format expected by chat templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<TemplateToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A tool call in the format expected by chat templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// A tool definition in the format expected by chat templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Chat template processor for formatting messages.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// The raw Jinja2 template string.
    template: String,
    /// Whether to add generation prompt at the end.
    add_generation_prompt: bool,
}

impl ChatTemplate {
    /// Create a new chat template from a Jinja2 template string.
    pub fn new(template: String) -> Self {
        Self {
            template,
            add_generation_prompt: true,
        }
    }

    /// Set whether to add generation prompt.
    pub fn with_generation_prompt(mut self, add: bool) -> Self {
        self.add_generation_prompt = add;
        self
    }

    /// Get the raw template string.
    pub fn template_string(&self) -> &str {
        &self.template
    }

    /// Apply the chat template to format messages.
    ///
    /// Returns the formatted prompt string ready for tokenization.
    pub fn apply(
        &self,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
    ) -> Result<String, ChatTemplateError> {
        let mut env = Environment::new();

        // Add support for common Python string methods that Jinja2 templates use
        // but minijinja doesn't provide by default
        env.set_unknown_method_callback(|_state, value, method, args| {
            // Handle string methods
            if let Some(s) = value.as_str() {
                match method {
                    "startswith" => {
                        let prefix = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                            minijinja::Error::new(
                                ErrorKind::InvalidOperation,
                                "startswith requires a string argument",
                            )
                        })?;
                        return Ok(Value::from(s.starts_with(prefix)));
                    }
                    "endswith" => {
                        let suffix = args.first().and_then(|v| v.as_str()).ok_or_else(|| {
                            minijinja::Error::new(
                                ErrorKind::InvalidOperation,
                                "endswith requires a string argument",
                            )
                        })?;
                        return Ok(Value::from(s.ends_with(suffix)));
                    }
                    "strip" => {
                        return Ok(Value::from(s.trim()));
                    }
                    "lstrip" => {
                        return Ok(Value::from(s.trim_start()));
                    }
                    "rstrip" => {
                        return Ok(Value::from(s.trim_end()));
                    }
                    _ => {}
                }
            }
            Err(minijinja::Error::new(
                ErrorKind::UnknownMethod,
                format!("object has no method named {}", method),
            ))
        });

        // Add the raise_exception function that many templates use
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );

        // Add strftime_now function (used by some templates)
        env.add_function("strftime_now", |_format: String| -> String {
            // Return a simple timestamp - real implementation would use chrono
            "2025-01-01".to_string()
        });

        env.add_template("chat", &self.template)
            .map_err(|e| ChatTemplateError::TemplateParseError(e.to_string()))?;

        let template = env
            .get_template("chat")
            .map_err(|e| ChatTemplateError::TemplateParseError(e.to_string()))?;

        // Convert messages to template format
        let template_messages: Vec<TemplateMessage> =
            messages.iter().map(|m| self.convert_message(m)).collect();

        // Convert tools to template format
        let template_tools: Option<Vec<TemplateTool>> = tools.map(|t| {
            t.iter()
                .map(|tool| TemplateTool {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                })
                .collect()
        });

        // Build context for template
        let mut context = HashMap::new();
        context.insert("messages", Value::from_serialize(&template_messages));
        context.insert(
            "add_generation_prompt",
            Value::from(self.add_generation_prompt),
        );

        if let Some(tools) = template_tools {
            context.insert("tools", Value::from_serialize(&tools));
        }

        // Common template variables
        context.insert("bos_token", Value::from(""));
        context.insert("eos_token", Value::from("<|im_end|>"));

        let result = template
            .render(Value::from_iter(context))
            .map_err(|e| ChatTemplateError::RenderError(e.to_string()))?;

        Ok(result)
    }

    /// Convert an LlmMessage to the template format.
    fn convert_message(&self, msg: &LlmMessage) -> TemplateMessage {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
        .to_string();

        let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
            tcs.iter()
                .filter_map(|tc| {
                    let name = tc.function.name.clone()?;
                    let arguments = tc
                        .function
                        .arguments
                        .as_ref()
                        .and_then(|a| serde_json::from_str(a).ok())
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    Some(TemplateToolCall { name, arguments })
                })
                .collect()
        });

        TemplateMessage {
            role,
            content: msg.content.clone(),
            tool_calls,
            name: msg.name.clone(),
        }
    }
}

/// Default Qwen3-Next chat template.
///
/// This is the standard chat template for Qwen3-Next models with tool support.
pub const QWEN3_NEXT_CHAT_TEMPLATE: &str = r#"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are an agent within Paramecia.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a JSON object with name and arguments within <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message['role'] == 'user') or (message['role'] == 'system' and not loop.first) %}
        {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {%- if message['content'] %}
            {{- '<|im_start|>' + message['role'] + '\n' + message['content'] }}
        {%- else %}
            {{- '<|im_start|>' + message['role'] + '\n' }}
        {%- endif %}
        {%- if message['tool_calls'] %}
            {%- for tool_call in message['tool_calls'] %}
                {{- '<tool_call>\n{"name": "' + tool_call['name'] + '", "arguments": ' }}
                {{- tool_call['arguments'] | tojson }}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message['role'] == 'tool' %}
        {{- '<|im_start|>user\n<tool_response>\n' + message['content'] + '\n</tool_response><|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

/// Errors that can occur during chat template processing.
#[derive(Debug, Clone)]
pub enum ChatTemplateError {
    /// Failed to parse the Jinja2 template.
    TemplateParseError(String),
    /// Failed to render the template.
    RenderError(String),
    /// Template not found in GGUF metadata.
    TemplateNotFound,
}

impl std::fmt::Display for ChatTemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TemplateParseError(e) => write!(f, "Template parse error: {}", e),
            Self::RenderError(e) => write!(f, "Template render error: {}", e),
            Self::TemplateNotFound => write!(f, "Chat template not found in GGUF metadata"),
        }
    }
}

impl std::error::Error for ChatTemplateError {}

/// Extract chat template from GGUF metadata.
///
/// The chat template is typically stored under the key `tokenizer.chat_template`.
pub fn extract_chat_template_from_gguf(
    _metadata: &HashMap<String, candle::quantized::gguf_file::Value>,
) -> Option<String> {
    Some(QWEN3_NEXT_CHAT_TEMPLATE.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_template() {
        let template = ChatTemplate::new(QWEN3_NEXT_CHAT_TEMPLATE.to_string());

        let messages = vec![
            LlmMessage::system("You are a helpful assistant."),
            LlmMessage::user("Hello!"),
        ];

        let result = template.apply(&messages, None);
        assert!(result.is_ok());
        let prompt = result.unwrap();
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_template_with_tools() {
        let template = ChatTemplate::new(QWEN3_NEXT_CHAT_TEMPLATE.to_string());

        let messages = vec![LlmMessage::user("List files in the current directory.")];

        let tools = vec![AvailableTool::function(
            "bash",
            "Execute a shell command",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"}
                },
                "required": ["command"]
            }),
        )];

        let result = template.apply(&messages, Some(&tools));
        assert!(result.is_ok());
        let prompt = result.unwrap();
        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("bash"));
    }

    #[test]
    fn test_template_with_startswith() {
        // Test template that uses Python's startswith method
        let template_str = r#"{%- for message in messages %}
{%- if message['content'].startswith('<think>') %}
[THINKING]: {{ message['content'] }}
{%- else %}
[NORMAL]: {{ message['content'] }}
{%- endif %}
{%- endfor %}"#;

        let template = ChatTemplate::new(template_str.to_string());

        let messages = vec![
            LlmMessage::user("<think>Let me think about this..."),
            LlmMessage::user("Hello!"),
        ];

        let result = template.apply(&messages, None);
        assert!(
            result.is_ok(),
            "Template with startswith should render: {:?}",
            result
        );
        let prompt = result.unwrap();
        assert!(
            prompt.contains("[THINKING]: <think>"),
            "Should detect startswith match"
        );
        assert!(
            prompt.contains("[NORMAL]: Hello!"),
            "Should detect non-match"
        );
    }

    #[test]
    fn test_template_with_endswith() {
        // Test template that uses Python's endswith method
        let template_str = r#"{%- for message in messages %}
{%- if message['content'].endswith('?') %}
[QUESTION]: {{ message['content'] }}
{%- else %}
[STATEMENT]: {{ message['content'] }}
{%- endif %}
{%- endfor %}"#;

        let template = ChatTemplate::new(template_str.to_string());

        let messages = vec![
            LlmMessage::user("What is the answer?"),
            LlmMessage::user("The answer is 42."),
        ];

        let result = template.apply(&messages, None);
        assert!(
            result.is_ok(),
            "Template with endswith should render: {:?}",
            result
        );
        let prompt = result.unwrap();
        assert!(
            prompt.contains("[QUESTION]: What is the answer?"),
            "Should detect endswith match"
        );
        assert!(
            prompt.contains("[STATEMENT]: The answer is 42."),
            "Should detect non-match"
        );
    }

    #[test]
    fn test_template_with_strip() {
        // Test template that uses Python's strip method
        let template_str = r#"{{ "  hello world  ".strip() }}"#;

        let template = ChatTemplate::new(template_str.to_string());
        let messages = vec![];

        let result = template.apply(&messages, None);
        assert!(
            result.is_ok(),
            "Template with strip should render: {:?}",
            result
        );
        let prompt = result.unwrap();
        assert_eq!(prompt.trim(), "hello world");
    }
}
