//! Session update tools for ACP
//!
//! These tools handle session updates and state synchronization
//! between the ACP protocol and Paramecia's agent.

use crate::types::{
    AgentMessageChunk, SessionUpdate, TextContentBlock, ToolCallContentVariant, ToolCallProgress,
    ToolCallStart, ToolKind,
};
use paramecia_harness::events::{ToolCallEvent, ToolResultEvent};
use paramecia_harness::utils::is_user_cancellation_event;
use serde_json::json;

/// Tool kind mapping
const TOOL_KIND: &[(&str, ToolKind)] = &[("read_file", ToolKind::Read), ("grep", ToolKind::Search)];

/// Session update tool for ACP
pub struct AcpSessionUpdateTool;

impl AcpSessionUpdateTool {
    /// Update tool state for ACP session
    pub fn update_tool_state(_session_id: &str, _tool_call_id: &str) {
        // For now, this is a placeholder for future ACP-specific tool state management
        // The main session update logic is handled by the event conversion functions
    }
}

/// Convert tool call events to ACP session updates
pub fn tool_call_session_update(event: &ToolCallEvent) -> Option<SessionUpdate> {
    let tool_name = event.tool_name.as_str();
    let tool_kind = TOOL_KIND
        .iter()
        .find(|(name, _)| *name == tool_name)
        .map(|(_, kind)| kind.clone())
        .unwrap_or(ToolKind::Other);

    // Create a simple display for the tool call
    let display_text = format!("Calling {} with args: {:?}", tool_name, event.args);

    let content = vec![ToolCallContentVariant::Content {
        content: TextContentBlock {
            r#type: "text".to_string(),
            text: display_text,
        },
    }];

    Some(SessionUpdate::ToolCall(ToolCallStart {
        session_update: "tool_call".to_string(),
        title: format!("Calling {}", tool_name),
        content: Some(content),
        tool_call_id: event.tool_call_id.clone(),
        kind: tool_kind,
        raw_input: json!(event.args).to_string(),
    }))
}

/// Convert tool result events to ACP session updates
pub fn tool_result_session_update(event: &ToolResultEvent) -> Option<SessionUpdate> {
    let (tool_status, raw_output) = if is_user_cancellation_event(event.skip_reason.as_deref()) {
        ("failed", event.skip_reason.clone().unwrap_or_default())
    } else if event.result.is_some() {
        (
            "completed",
            event
                .result
                .as_ref()
                .map(|r| r.to_string())
                .unwrap_or_default(),
        )
    } else {
        (
            "failed",
            event
                .error
                .as_ref()
                .map(|e| e.to_string())
                .unwrap_or_default(),
        )
    };

    let content_text = if tool_status == "failed" {
        raw_output.clone()
    } else {
        format!("Tool {} completed successfully", event.tool_name)
    };

    let content = vec![ToolCallContentVariant::Content {
        content: TextContentBlock {
            r#type: "text".to_string(),
            text: content_text,
        },
    }];

    Some(SessionUpdate::ToolCallUpdate(ToolCallProgress {
        session_update: "tool_call_update".to_string(),
        tool_call_id: event.tool_call_id.clone(),
        status: tool_status.to_string(),
        raw_output: Some(raw_output),
        content: Some(content),
    }))
}

/// Convert assistant message events to ACP session updates
pub fn assistant_message_session_update(content: String) -> SessionUpdate {
    SessionUpdate::AgentMessageChunk(AgentMessageChunk {
        session_update: "agent_message_chunk".to_string(),
        content: TextContentBlock {
            r#type: "text".to_string(),
            text: content,
        },
    })
}
