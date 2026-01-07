use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// ACP Protocol version
pub const PROTOCOL_VERSION: &str = "1.0.0";

/// Tool call kind
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolKind {
    /// Read tool kind
    Read,
    /// Search tool kind
    Search,
    /// Other tool kind
    Other,
}

/// Text content block
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextContentBlock {
    #[serde(rename = "type")]
    pub r#type: String,
    pub text: String,
}

/// Tool call content variant
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolCallContentVariant {
    /// Content variant
    Content { content: TextContentBlock },
}

/// Tool call start session update
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolCallStart {
    #[serde(rename = "sessionUpdate")]
    pub session_update: String,
    pub title: String,
    pub content: Option<Vec<ToolCallContentVariant>>,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    pub kind: ToolKind,
    #[serde(rename = "rawInput")]
    pub raw_input: String,
}

/// Tool call progress session update
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolCallProgress {
    #[serde(rename = "sessionUpdate")]
    pub session_update: String,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    pub status: String,
    #[serde(rename = "rawOutput")]
    pub raw_output: Option<String>,
    pub content: Option<Vec<ToolCallContentVariant>>,
}

/// Agent message chunk session update
#[derive(Debug, Serialize, Deserialize)]
pub struct AgentMessageChunk {
    #[serde(rename = "sessionUpdate")]
    pub session_update: String,
    pub content: TextContentBlock,
}

/// Session update enum
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
pub enum SessionUpdate {
    /// Tool call update
    ToolCall(ToolCallStart),
    /// Tool call progress update
    ToolCallUpdate(ToolCallProgress),
    /// Agent message chunk update
    AgentMessageChunk(AgentMessageChunk),
}

/// Agent capabilities
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentCapabilities {
    pub load_session: bool,
    pub prompt_capabilities: PromptCapabilities,
}

/// Prompt capabilities
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PromptCapabilities {
    pub audio: bool,
    pub embedded_context: bool,
    pub image: bool,
}

/// Agent implementation information
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Implementation {
    pub name: String,
    pub title: String,
    pub version: String,
}

/// Authentication method
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AuthMethod {
    pub id: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "fieldMeta")]
    pub field_meta: serde_json::Value,
}

/// Initialize request
#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeRequest {
    #[serde(rename = "clientCapabilities")]
    pub client_capabilities: Option<ClientCapabilities>,
}

/// Client capabilities
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClientCapabilities {
    pub terminal: bool,
    pub fs: Option<FsCapabilities>,
    #[serde(rename = "fieldMeta")]
    pub field_meta: serde_json::Value,
}

/// Filesystem capabilities
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FsCapabilities {
    #[serde(rename = "readTextFile")]
    pub read_text_file: bool,
    #[serde(rename = "writeTextFile")]
    pub write_text_file: bool,
}

/// Initialize response
#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeResponse {
    #[serde(rename = "agentCapabilities")]
    pub agent_capabilities: AgentCapabilities,
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    #[serde(rename = "agentInfo")]
    pub agent_info: Implementation,
    #[serde(rename = "authMethods")]
    pub auth_methods: Vec<AuthMethod>,
}

/// New session request
#[derive(Debug, Serialize, Deserialize)]
pub struct NewSessionRequest {
    pub cwd: PathBuf,
}

/// New session response
#[derive(Debug, Serialize, Deserialize)]
pub struct NewSessionResponse {
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub models: SessionModelState,
    pub modes: SessionModeState,
}

/// Session model state
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionModelState {
    #[serde(rename = "currentModelId")]
    pub current_model_id: String,
    #[serde(rename = "availableModels")]
    pub available_models: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    #[serde(rename = "modelId")]
    pub model_id: String,
    pub name: String,
}

/// Session mode state
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionModeState {
    #[serde(rename = "currentModeId")]
    pub current_mode_id: String,
    #[serde(rename = "availableModes")]
    pub available_modes: Vec<ModeInfo>,
}

/// Mode information
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModeInfo {
    #[serde(rename = "modeId")]
    pub mode_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}
