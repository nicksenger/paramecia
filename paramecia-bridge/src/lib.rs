//! Bridge types for interactive controller <-> TUI communication.
//!
//! These types are the payload for tokio mpsc channels that connect
//! a ControllerBackend (paramecia-text) with the host-side connect()
//! handler (paramecia-controller).

use tokio::sync::mpsc;

/// UUID type matching the WIT definition: tuple<u64, u64>
pub type Uuid = (u64, u64);

/// Events sent from the TUI toward the WASM guest controller.
#[derive(Debug, Clone)]
pub enum ControllerUserEvent {
    /// A request from the interface to prefill specific text.
    Prefill(String),
    /// A user-entered prompt.
    Prompt(String),
    /// A request to cancel the current operation.
    Cancel,
    /// A tool execution result.
    ToolResult {
        /// Tool call ID.
        id: Uuid,
        /// Tool name.
        name: String,
        /// Tool output content.
        content: String,
    },
}

/// Events received from the WASM guest controller.
#[derive(Debug, Clone)]
pub enum ControllerAgentEvent {
    /// The controller has finished initialization (model loaded, ready to accept requests).
    Initialized,
    /// Context token usage update.
    ContextTokens {
        /// Number of tokens to add to the context
        context_tokens: u32,
    },
    /// A text delta from the assistant.
    Token {
        /// Text content delta.
        content: String,
    },
    /// A complete, parsed tool call.
    ToolCall {
        /// Unique identifier for this tool call.
        id: Uuid,
        /// Name of the tool/function to call.
        name: String,
        /// Parsed arguments as JSON string.
        arguments: String,
        /// Number of tokens the tool-call was composed of
        n_tokens: u32,
    },
    /// Generation is finished.
    Done,
    /// An error occurred.
    Error(String),
}

/// Backend-side endpoint (for ControllerBackend in paramecia-text).
pub struct ControllerBridge {
    /// Sender for user events toward the WASM guest.
    pub user_tx: mpsc::Sender<ControllerUserEvent>,
    /// Receiver for agent events from the WASM guest.
    pub agent_rx: mpsc::Receiver<ControllerAgentEvent>,
}

/// Host-side endpoint (for connect() handler in paramecia-controller).
pub struct ControllerHostEndpoint {
    /// Receiver for user events from the TUI.
    pub user_rx: mpsc::Receiver<ControllerUserEvent>,
    /// Sender for agent events toward the TUI.
    pub agent_tx: mpsc::Sender<ControllerAgentEvent>,
}

/// Create a matched pair of bridge endpoints.
pub fn create_bridge(buffer: usize) -> (ControllerBridge, ControllerHostEndpoint) {
    let (user_tx, user_rx) = mpsc::channel(buffer);
    let (agent_tx, agent_rx) = mpsc::channel(buffer);
    (
        ControllerBridge { user_tx, agent_rx },
        ControllerHostEndpoint { user_rx, agent_tx },
    )
}
