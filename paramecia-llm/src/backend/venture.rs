//! Venture OpenAI-compatible API backend implementation.

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

use super::{Backend, ChunkStream, CompletionOptions, ModelConfig, ProviderConfig};
use crate::error::{LlmError, LlmResult};
use crate::types::{AvailableTool, FunctionCall, LlmChunk, LlmMessage, LlmUsage, Role, ToolCall};

/// OpenAI-compatible API backend
///
/// Since this backend is intended for use of external models, it dumps all conversations into a "venture"
/// directory in the paramecia config directory (generally ~/.paramecia) in ChatML format. 
///
/// This is strictly for debugging purposes, since the external ecosystem is volatile and unpredictable.
pub struct VentureBackend {
    client: reqwest::Client,
    provider: ProviderConfig,
    api_key: Option<String>,
    venture_id: String,
}

impl VentureBackend {
    /// Create a new venture backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn new(provider: ProviderConfig, timeout: Duration) -> LlmResult<Self> {
        let api_key = if provider.api_key_env_var.is_empty() {
            None
        } else {
            std::env::var(&provider.api_key_env_var).ok()
        };

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(LlmError::RequestFailed)?;

        let venture_id = uuid::Uuid::new_v4().to_string();

        Ok(Self {
            client,
            provider,
            api_key,
            venture_id,
        })
    }

    /// Get the venture ID for this backend instance.
    #[must_use]
    pub fn venture_id(&self) -> &str {
        &self.venture_id
    }

    /// Get the ventures directory path.
    fn ventures_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".paramecia")
            .join("ventures")
    }

    /// Get the venture file path for this session.
    fn venture_file_path(&self, model_name: &str) -> PathBuf {
        Self::ventures_dir().join(model_name).join(format!("{}.txt", self.venture_id))
    }

    /// Format a message in ChatML format.
    fn format_message_chatml(msg: &LlmMessage) -> String {
        let role = msg.role.to_string();
        let mut content = msg.content.clone().unwrap_or_default();

        // For assistant messages with tool calls, append the tool calls in a structured format
        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                let name = tc.function.name.as_deref().unwrap_or("unknown");
                let args = tc.function.arguments.as_deref().unwrap_or("{}");
                content.push_str(&format!("\n<tool_call>\n{name}\n{args}\n</tool_call>"));
            }
        }

        // For tool responses, include the tool name and call ID
        if msg.role == Role::Tool {
            let name = msg.name.as_deref().unwrap_or("unknown");
            let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
            format!(
                "<|im_start|>{role}[{name}:{tool_call_id}]\n{content}<|im_end|>"
            )
        } else {
            format!("<|im_start|>{role}\n{content}<|im_end|>")
        }
    }

    /// Format all messages as a ChatML conversation.
    fn format_conversation_chatml(messages: &[LlmMessage]) -> String {
        messages
            .iter()
            .map(Self::format_message_chatml)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Save the conversation to the ventures directory.
    async fn save_conversation(&self, messages: &[LlmMessage], response: &LlmMessage, model_name: &str) {
        // Create all messages including the response
        let mut all_messages = messages.to_vec();
        all_messages.push(response.clone());

        let content = Self::format_conversation_chatml(&all_messages);

        let ventures_dir = Self::ventures_dir();
        let file_path = self.venture_file_path(model_name);
        let model_dir = ventures_dir.join(model_name);

        tracing::info!(
            "VentureBackend: saving conversation ({} bytes) to {}",
            content.len(),
            file_path.display()
        );

        // Create directory if needed and write file
        if let Err(e) = tokio::fs::create_dir_all(&model_dir).await {
            tracing::error!("Failed to create model directory {}: {e}", model_dir.display());
            return;
        }

        if let Err(e) = tokio::fs::write(&file_path, &content).await {
            tracing::error!("Failed to write venture file {}: {e}", file_path.display());
        } else {
            tracing::info!("Saved venture conversation to {}", file_path.display());
        }
    }

    fn build_headers(
        &self,
        extra: Option<&std::collections::HashMap<String, String>>,
    ) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(ref key) = self.api_key
            && let Ok(val) = HeaderValue::from_str(&format!("Bearer {key}"))
        {
            headers.insert(AUTHORIZATION, val);
        }

        if let Some(extra) = extra {
            for (key, value) in extra {
                if let (Ok(name), Ok(val)) = (
                    reqwest::header::HeaderName::try_from(key),
                    HeaderValue::from_str(value),
                ) {
                    headers.insert(name, val);
                }
            }
        }

        headers
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.provider.api_base)
    }
}

// API request/response types (OpenAI-compatible format)
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    /// Tool call ID - required when sending to API, may be empty in streaming deltas
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default = "default_function_type")]
    type_: String,
    /// Function details - may have partial data in streaming deltas
    #[serde(default)]
    function: OpenAIFunctionCall,
    /// Index is only used during streaming responses, skip when serializing requests
    #[serde(skip_serializing, default)]
    index: Option<usize>,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    /// Function name - may be empty in streaming deltas
    #[serde(default)]
    name: String,
    /// Function arguments JSON - accumulated across streaming deltas
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    type_: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Option<ResponseMessage>,
    delta: Option<ResponseMessage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default, deserialize_with = "deserialize_content")]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

/// Deserialize content that can be either a string or an array of content chunks.
fn deserialize_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};

    struct ContentVisitor;

    impl<'de> Visitor<'de> for ContentVisitor {
        type Value = Option<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or array of content chunks")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v.to_string()))
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v))
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut parts = Vec::new();
            while let Some(chunk) = seq.next_element::<serde_json::Value>()? {
                if let Some(obj) = chunk.as_object() {
                    // Handle {"type": "text", "text": "..."} format
                    if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                        parts.push(text.to_string());
                    }
                }
            }
            if parts.is_empty() {
                Ok(None)
            } else {
                Ok(Some(parts.join("\n")))
            }
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

impl VentureBackend {
    fn convert_message(msg: &LlmMessage) -> OpenAIMessage {
        // For assistant messages, ensure we don't send empty content when there are tool_calls
        // OpenAI-compatible APIs may require assistant messages to have either content or tool_calls
        let content = match (&msg.content, &msg.tool_calls) {
            // If assistant has tool_calls, empty content should be None
            (Some(c), Some(_)) if c.is_empty() => None,
            (content, _) => content.clone(),
        };

        // For tool response messages, don't include name field as some APIs don't expect it
        let name = if msg.role == Role::Tool {
            None
        } else {
            msg.name.clone()
        };

        OpenAIMessage {
            role: msg.role.to_string(),
            content,
            tool_calls: msg.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| OpenAIToolCall {
                        // id is required for tool calls
                        id: tc.id.clone().unwrap_or_default(),
                        type_: tc.r#type.clone(),
                        function: OpenAIFunctionCall {
                            name: tc.function.name.clone().unwrap_or_default(),
                            arguments: tc.function.arguments.clone().unwrap_or_default(),
                        },
                        index: tc.index,
                    })
                    .collect()
            }),
            name,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }

    fn convert_tool(tool: &AvailableTool) -> OpenAITool {
        OpenAITool {
            type_: "function".to_string(),
            function: OpenAIFunction {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
            },
        }
    }

    fn convert_response_message(msg: &ResponseMessage) -> LlmMessage {
        LlmMessage {
            role: Role::Assistant,
            content: msg.content.clone(),
            tool_calls: msg.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        // Convert String to Option<String>, treating empty as None
                        id: if tc.id.is_empty() {
                            None
                        } else {
                            Some(tc.id.clone())
                        },
                        index: tc.index,
                        function: FunctionCall {
                            name: if tc.function.name.is_empty() {
                                None
                            } else {
                                Some(tc.function.name.clone())
                            },
                            arguments: if tc.function.arguments.is_empty() {
                                None
                            } else {
                                Some(tc.function.arguments.clone())
                            },
                        },
                        r#type: tc.type_.clone(),
                    })
                    .collect()
            }),
            name: None,
            tool_call_id: None,
        }
    }

    /// Sanitize messages for Anthropic compatibility.
    /// Anthropic requires that every assistant message with tool_use must be followed
    /// by a tool_result message. If the conversation ends with an incomplete tool call,
    /// we need to handle it.
    fn sanitize_messages_for_api(messages: &[LlmMessage]) -> Vec<LlmMessage> {
        if messages.is_empty() {
            return messages.to_vec();
        }

        let mut result = messages.to_vec();

        // Check if the last message is an assistant with tool_calls
        if let Some(last) = result.last() {
            if last.role == Role::Assistant && last.tool_calls.is_some() {
                // Check if there's a tool_calls array with actual calls
                if let Some(tool_calls) = &last.tool_calls {
                    if !tool_calls.is_empty() {
                        // This is an incomplete conversation - assistant made tool calls
                        // but there's no tool result yet. Remove the tool_calls from this
                        // message to make it valid for Anthropic.
                        tracing::warn!(
                            "Sanitizing incomplete tool call - removing {} tool calls from last message",
                            tool_calls.len()
                        );
                        if let Some(last_mut) = result.last_mut() {
                            last_mut.tool_calls = None;
                            // If content is also empty, remove the message entirely
                            if last_mut.content.as_ref().is_none_or(|c| c.is_empty()) {
                                result.pop();
                            }
                        }
                    }
                }
            }
        }

        result
    }

    /// Merge streaming tool call deltas into accumulated tool calls.
    /// Tool calls are merged by their index field.
    fn merge_tool_calls(accumulated: &mut Vec<ToolCall>, deltas: &[ToolCall]) {
        for delta in deltas {
            let index = delta.index.unwrap_or(0);

            // Find existing tool call with this index
            if let Some(existing) = accumulated.iter_mut().find(|tc| tc.index == Some(index)) {
                // Merge: use first non-empty id
                if existing.id.is_none() || existing.id.as_ref().is_some_and(|s| s.is_empty()) {
                    if let Some(id) = &delta.id {
                        if !id.is_empty() {
                            existing.id = Some(id.clone());
                        }
                    }
                }
                // Merge: use first non-empty name
                if existing.function.name.is_none()
                    || existing.function.name.as_ref().is_some_and(|s| s.is_empty())
                {
                    if let Some(name) = &delta.function.name {
                        if !name.is_empty() {
                            existing.function.name = Some(name.clone());
                        }
                    }
                }
                // Merge: concatenate arguments
                if let Some(args) = &delta.function.arguments {
                    if !args.is_empty() {
                        if let Some(existing_args) = &mut existing.function.arguments {
                            existing_args.push_str(args);
                        } else {
                            existing.function.arguments = Some(args.clone());
                        }
                    }
                }
                // Merge: use first non-empty type
                if existing.r#type.is_empty() && !delta.r#type.is_empty() {
                    existing.r#type = delta.r#type.clone();
                }
            } else {
                // New tool call - add it with the index set
                let mut new_tc = delta.clone();
                new_tc.index = Some(index);
                accumulated.push(new_tc);
            }
        }
    }
}

/// A stream wrapper that saves the conversation when the stream completes or is dropped.
struct VentureSavingStream {
    inner: std::pin::Pin<Box<dyn futures::Stream<Item = LlmResult<LlmChunk>> + Send>>,
    input_messages: Vec<LlmMessage>,
    accumulated_content: std::sync::Arc<std::sync::Mutex<String>>,
    accumulated_tool_calls: std::sync::Arc<std::sync::Mutex<Vec<ToolCall>>>,
    venture_file_path: PathBuf,
    ventures_dir: PathBuf,
    model_name: String,
    saved: bool,
}

impl VentureSavingStream {
    fn save_conversation_sync(&self) {
        let content = self
            .accumulated_content
            .lock()
            .map(|c| c.clone())
            .unwrap_or_default();
        let tool_calls = self
            .accumulated_tool_calls
            .lock()
            .map(|t| t.clone())
            .ok()
            .filter(|t| !t.is_empty());

        tracing::info!(
            "VentureSavingStream: saving conversation with {} chars of content",
            content.len()
        );

        let response = LlmMessage {
            role: Role::Assistant,
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls,
            name: None,
            tool_call_id: None,
        };

        let mut all_messages = self.input_messages.clone();
        all_messages.push(response);

        let formatted = VentureBackend::format_conversation_chatml(&all_messages);

        // Use blocking IO since we're in Drop
        let model_dir = self.ventures_dir.join(&self.model_name);
        if let Err(e) = std::fs::create_dir_all(&model_dir) {
            tracing::error!("Failed to create model directory {}: {e}", model_dir.display());
            return;
        }

        if let Err(e) = std::fs::write(&self.venture_file_path, &formatted) {
            tracing::error!("Failed to write venture file {}: {e}", self.venture_file_path.display());
        } else {
            tracing::info!(
                "Saved venture conversation ({} bytes) to {}",
                formatted.len(),
                self.venture_file_path.display()
            );
        }
    }
}

impl Drop for VentureSavingStream {
    fn drop(&mut self) {
        tracing::info!("VentureSavingStream::drop called, saved={}", self.saved);
        if !self.saved {
            self.save_conversation_sync();
            self.saved = true;
        }
    }
}

impl futures::Stream for VentureSavingStream {
    type Item = LlmResult<LlmChunk>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

#[async_trait]
impl Backend for VentureBackend {
    async fn complete(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
        options: &CompletionOptions,
    ) -> LlmResult<LlmChunk> {
        // Sanitize messages to handle incomplete tool call conversations
        let sanitized_messages = Self::sanitize_messages_for_api(messages);

        let request = ChatRequest {
            model: &model.name,
            messages: sanitized_messages
                .iter()
                .map(Self::convert_message)
                .collect(),
            temperature: model.temperature,
            max_tokens: options.max_tokens,
            tools: tools.map(|t| t.iter().map(Self::convert_tool).collect()),
            tool_choice: options.tool_choice.as_ref().map(|tc| {
                serde_json::to_value(tc).unwrap_or(serde_json::Value::String("auto".to_string()))
            }),
            stream: false,
            stream_options: None,
        };

        let response = self
            .client
            .post(self.endpoint())
            .headers(self.build_headers(options.extra_headers.as_ref()))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if !status.is_success() {
            // Log the error and request to a file for debugging
            let error_log = format!(
                "=== API Error ===\nStatus: {}\nProvider: {}\nEndpoint: {}\n\n=== Request ===\n{}\n\n=== Response ===\n{}\n",
                status,
                self.provider.name,
                self.endpoint(),
                serde_json::to_string_pretty(&request).unwrap_or_default(),
                body
            );
            let error_path = Self::ventures_dir().join("error.log");
            let _ = std::fs::create_dir_all(Self::ventures_dir());
            let _ = std::fs::write(&error_path, &error_log);
            tracing::error!("API error logged to {}", error_path.display());

            return Err(LlmError::ApiError {
                provider: self.provider.name.clone(),
                status: status.as_u16(),
                message: body,
            });
        }

        let chat_response: ChatResponse = serde_json::from_str(&body).map_err(|e| {
            LlmError::ParseError(format!("Failed to parse response: {e}\nBody: {body}"))
        })?;

        let choice = chat_response
            .choices
            .first()
            .ok_or_else(|| LlmError::ParseError("No choices in response".to_string()))?;

        let message = choice
            .message
            .as_ref()
            .ok_or_else(|| LlmError::ParseError("No message in choice".to_string()))?;

        let response_message = Self::convert_response_message(message);

        // Save the conversation to the ventures directory
        self.save_conversation(messages, &response_message, &model.name).await;

        Ok(LlmChunk {
            message: response_message,
            finish_reason: choice.finish_reason.clone(),
            usage: chat_response.usage.map(|u| LlmUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
            }),
        })
    }

    async fn complete_streaming(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
        options: &CompletionOptions,
    ) -> LlmResult<ChunkStream> {
        // Sanitize messages to handle incomplete tool call conversations
        let sanitized_messages = Self::sanitize_messages_for_api(messages);

        let request = ChatRequest {
            model: &model.name,
            messages: sanitized_messages
                .iter()
                .map(Self::convert_message)
                .collect(),
            temperature: model.temperature,
            max_tokens: options.max_tokens,
            tools: tools.map(|t| t.iter().map(Self::convert_tool).collect()),
            tool_choice: options.tool_choice.as_ref().map(|tc| {
                serde_json::to_value(tc).unwrap_or(serde_json::Value::String("auto".to_string()))
            }),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
        };

        let response = self
            .client
            .post(self.endpoint())
            .headers(self.build_headers(options.extra_headers.as_ref()))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();

            // Log the error and request to a file for debugging
            let error_log = format!(
                "=== API Error (Streaming) ===\nStatus: {}\nProvider: {}\nEndpoint: {}\n\n=== Request ===\n{}\n\n=== Response ===\n{}\n",
                status,
                self.provider.name,
                self.endpoint(),
                serde_json::to_string_pretty(&request).unwrap_or_default(),
                body
            );
            let error_path = Self::ventures_dir().join("error.log");
            let _ = std::fs::create_dir_all(Self::ventures_dir());
            let _ = std::fs::write(&error_path, &error_log);
            tracing::error!("API error logged to {}", error_path.display());

            return Err(LlmError::ApiError {
                provider: self.provider.name.clone(),
                status: status.as_u16(),
                message: body,
            });
        }

        // Capture data needed for saving the conversation
        let input_messages = messages.to_vec();
        let venture_file_path = self.venture_file_path(&model.name);
        let ventures_dir = Self::ventures_dir();
        let model_name = model.name.clone();

        // Track accumulated response content
        let accumulated_content = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let accumulated_tool_calls =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::<ToolCall>::new()));

        // Buffer for partial SSE lines that span network chunks
        let line_buffer = std::sync::Arc::new(std::sync::Mutex::new(String::new()));

        let content_clone = accumulated_content.clone();
        let tool_calls_clone = accumulated_tool_calls.clone();
        let buffer_clone = line_buffer.clone();

        let stream = response.bytes_stream().map(move |result| {
            result.map_err(LlmError::RequestFailed).map(|bytes| {
                let text = String::from_utf8_lossy(&bytes);

                // Prepend any buffered partial line from previous chunk
                let full_text = {
                    let mut buffer = buffer_clone.lock().unwrap();
                    let combined = format!("{}{}", buffer, text);
                    buffer.clear();
                    combined
                };

                // Accumulate content from ALL SSE messages in this chunk
                let mut combined_content = String::new();
                let mut combined_tool_calls: Vec<ToolCall> = Vec::new();
                let mut last_finish_reason: Option<String> = None;
                let mut last_usage: Option<LlmUsage> = None;

                // Split into lines, keeping track of whether the last line is complete
                let ends_with_newline = full_text.ends_with('\n');
                let lines: Vec<&str> = full_text.lines().collect();

                for (i, line) in lines.iter().enumerate() {
                    // If this is the last line and doesn't end with newline, buffer it
                    if i == lines.len() - 1 && !ends_with_newline {
                        let mut buffer = buffer_clone.lock().unwrap();
                        buffer.push_str(line);
                        continue;
                    }

                    // Parse SSE format
                    if let Some(data) = line.strip_prefix("data: ")
                        && data != "[DONE]"
                        && let Ok(chunk) = serde_json::from_str::<StreamChunk>(data)
                        && let Some(choice) = chunk.choices.first()
                    {
                        let message = if let Some(delta) = &choice.delta {
                            Self::convert_response_message(delta)
                        } else if let Some(msg) = &choice.message {
                            Self::convert_response_message(msg)
                        } else {
                            LlmMessage::assistant("")
                        };

                        // Accumulate content from this SSE message
                        if let Some(content) = &message.content {
                            combined_content.push_str(content);
                        }
                        if let Some(tool_calls) = &message.tool_calls {
                            // Merge tool calls by index instead of just appending
                            Self::merge_tool_calls(&mut combined_tool_calls, tool_calls);
                        }

                        // Keep track of the last finish_reason and usage
                        if choice.finish_reason.is_some() {
                            last_finish_reason = choice.finish_reason.clone();
                        }
                        if let Some(u) = chunk.usage {
                            last_usage = Some(LlmUsage {
                                prompt_tokens: u.prompt_tokens,
                                completion_tokens: u.completion_tokens,
                            });
                        }
                    }
                }

                // Update the accumulated content for saving
                if !combined_content.is_empty() {
                    if let Ok(mut acc) = content_clone.lock() {
                        acc.push_str(&combined_content);
                    }
                }
                if !combined_tool_calls.is_empty() {
                    if let Ok(mut acc) = tool_calls_clone.lock() {
                        // Merge tool calls by index for proper accumulation
                        Self::merge_tool_calls(&mut acc, &combined_tool_calls);
                    }
                }

                // Return combined chunk with all content from this network chunk
                LlmChunk {
                    message: LlmMessage {
                        role: Role::Assistant,
                        content: if combined_content.is_empty() {
                            None
                        } else {
                            Some(combined_content)
                        },
                        tool_calls: if combined_tool_calls.is_empty() {
                            None
                        } else {
                            Some(combined_tool_calls)
                        },
                        name: None,
                        tool_call_id: None,
                    },
                    finish_reason: last_finish_reason,
                    usage: last_usage,
                }
            })
        });

        // Wrap stream to save conversation when complete
        let saving_stream = VentureSavingStream {
            inner: Box::pin(stream),
            input_messages,
            accumulated_content,
            accumulated_tool_calls,
            venture_file_path,
            ventures_dir,
            model_name,
            saved: false,
        };

        Ok(Box::pin(saving_stream))
    }

    async fn count_tokens(
        &self,
        model: &ModelConfig,
        messages: &[LlmMessage],
        tools: Option<&[AvailableTool]>,
    ) -> LlmResult<u32> {
        // Use a completion with max_tokens=1 to get token count
        let result = self
            .complete(
                model,
                messages,
                tools,
                &CompletionOptions {
                    max_tokens: Some(1),
                    ..Default::default()
                },
            )
            .await?;

        result
            .usage
            .map(|u| u.prompt_tokens)
            .ok_or(LlmError::MissingUsage)
    }
}
