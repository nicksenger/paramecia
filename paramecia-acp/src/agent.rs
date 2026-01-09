use crate::error::AcpError;
use crate::types::*;
use async_trait::async_trait;
use paramecia_harness::agent::Agent as ParameciaAgent;
use paramecia_harness::config::VibeConfig;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// ACP Agent implementation
pub struct AcpAgent {
    sessions: Arc<Mutex<HashMap<String, AcpSession>>>,
    client_capabilities: Option<ClientCapabilities>,
}

/// ACP Session
pub struct AcpSession {
    pub id: String,
    pub agent: ParameciaAgent,
    pub task: Option<tokio::task::JoinHandle<()>>,
}

#[async_trait]
pub trait AgentTrait {
    async fn initialize(
        &mut self,
        params: InitializeRequest,
    ) -> Result<InitializeResponse, AcpError>;
    async fn new_session(
        &mut self,
        params: NewSessionRequest,
    ) -> Result<NewSessionResponse, AcpError>;
    // Other ACP methods would go here
}

#[async_trait]
impl AgentTrait for AcpAgent {
    async fn initialize(
        &mut self,
        params: InitializeRequest,
    ) -> Result<InitializeResponse, AcpError> {
        // Check if client supports terminal authentication before moving
        let _supports_terminal_auth = params
            .client_capabilities
            .as_ref()
            .and_then(|cc| cc.field_meta.as_object())
            .and_then(|meta| meta.get("terminal-auth"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        self.client_capabilities = params.client_capabilities;

        let auth_methods = vec![];
        let response = InitializeResponse {
            agent_capabilities: AgentCapabilities {
                load_session: false,
                prompt_capabilities: PromptCapabilities {
                    audio: false,
                    embedded_context: true,
                    image: false,
                },
            },
            protocol_version: PROTOCOL_VERSION.to_string(),
            agent_info: Implementation {
                name: "paramecia".to_string(),
                title: "Paramecia".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            auth_methods,
        };

        Ok(response)
    }

    async fn new_session(
        &mut self,
        params: NewSessionRequest,
    ) -> Result<NewSessionResponse, AcpError> {
        let _cwd = params.cwd;

        // Load configuration
        let config = VibeConfig::load(None).map_err(|e| AcpError::InternalError(e.to_string()))?;

        // Create agent with default mode
        let agent =
            ParameciaAgent::new(config.clone(), paramecia_harness::modes::AgentMode::Default)
                .map_err(|e| AcpError::InternalError(e.to_string()))?;

        let session_id = agent.session_id().to_string();
        let session = AcpSession {
            id: session_id.clone(),
            agent,
            task: None,
        };

        // Store session
        self.sessions
            .lock()
            .await
            .insert(session_id.clone(), session);

        let response = NewSessionResponse {
            session_id: session_id.clone(),
            models: SessionModelState {
                current_model_id: config.active_model.clone(),
                available_models: config
                    .models
                    .iter()
                    .map(|model| crate::types::ModelInfo {
                        model_id: model.alias().to_string(),
                        name: model.alias().to_string(),
                    })
                    .collect(),
            },
            modes: SessionModeState {
                current_mode_id: paramecia_harness::modes::AgentMode::Default.to_string(),
                available_modes: vec![
                    crate::types::ModeInfo {
                        mode_id: paramecia_harness::modes::AgentMode::Default.to_string(),
                        name: paramecia_harness::modes::AgentMode::Default
                            .display_name()
                            .to_string(),
                        description: Some(
                            paramecia_harness::modes::AgentMode::Default
                                .description()
                                .to_string(),
                        ),
                    },
                    crate::types::ModeInfo {
                        mode_id: paramecia_harness::modes::AgentMode::Plan.to_string(),
                        name: paramecia_harness::modes::AgentMode::Plan
                            .display_name()
                            .to_string(),
                        description: Some(
                            paramecia_harness::modes::AgentMode::Plan
                                .description()
                                .to_string(),
                        ),
                    },
                    crate::types::ModeInfo {
                        mode_id: paramecia_harness::modes::AgentMode::AcceptEdits.to_string(),
                        name: paramecia_harness::modes::AgentMode::AcceptEdits
                            .display_name()
                            .to_string(),
                        description: Some(
                            paramecia_harness::modes::AgentMode::AcceptEdits
                                .description()
                                .to_string(),
                        ),
                    },
                    crate::types::ModeInfo {
                        mode_id: paramecia_harness::modes::AgentMode::AutoApprove.to_string(),
                        name: paramecia_harness::modes::AgentMode::AutoApprove
                            .display_name()
                            .to_string(),
                        description: Some(
                            paramecia_harness::modes::AgentMode::AutoApprove
                                .description()
                                .to_string(),
                        ),
                    },
                ],
            },
        };

        Ok(response)
    }
}

impl Default for AcpAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl AcpAgent {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            client_capabilities: None,
        }
    }
}
