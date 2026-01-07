//! Agent operation modes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Safety level for a mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeSafety {
    /// Safe - read-only operations.
    Safe,
    /// Neutral - normal operation with approvals.
    Neutral,
    /// Destructive - allows some destructive operations.
    Destructive,
    /// YOLO - auto-approves everything.
    Yolo,
}

/// Tools allowed in plan mode.
pub const PLAN_MODE_TOOLS: &[&str] = &["grep", "read_file", "todo"];

/// Tools to auto-approve in accept edits mode.
pub const ACCEPT_EDITS_TOOLS: &[&str] = &["write_file", "search_replace"];

/// Operating mode for the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    /// Default mode - ask for approval on each tool.
    #[default]
    Default,
    /// Plan mode - read-only tools only, auto-approved.
    Plan,
    /// Accept edits mode - auto-approves file edits only.
    AcceptEdits,
    /// Auto-approve mode - automatically approve all tools.
    AutoApprove,
}

impl AgentMode {
    /// Get the display name for this mode.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::Plan => "Plan",
            Self::AcceptEdits => "Accept Edits",
            Self::AutoApprove => "Auto Approve",
        }
    }

    /// Get the description for this mode.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Default => "Requires approval for tool executions",
            Self::Plan => "Read-only mode for exploration and planning",
            Self::AcceptEdits => "Auto-approves file edits only",
            Self::AutoApprove => "Auto-approves all tool executions",
        }
    }

    /// Get the safety level for this mode.
    #[must_use]
    pub fn safety(&self) -> ModeSafety {
        match self {
            Self::Default => ModeSafety::Neutral,
            Self::Plan => ModeSafety::Safe,
            Self::AcceptEdits => ModeSafety::Destructive,
            Self::AutoApprove => ModeSafety::Yolo,
        }
    }

    /// Whether this mode auto-approves tool execution.
    #[must_use]
    pub fn auto_approve(&self) -> bool {
        matches!(self, Self::AutoApprove | Self::Plan)
    }

    /// Whether this mode is read-only (plan mode).
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::Plan)
    }

    /// Get configuration overrides for this mode.
    #[must_use]
    pub fn config_overrides(&self) -> HashMap<String, serde_json::Value> {
        match self {
            Self::Default => HashMap::new(),
            Self::AutoApprove => HashMap::new(),
            Self::Plan => {
                let mut overrides = HashMap::new();
                // Enable only read-only tools in plan mode
                overrides.insert(
                    "enabled_tools".to_string(),
                    serde_json::json!(PLAN_MODE_TOOLS),
                );
                overrides
            }
            Self::AcceptEdits => {
                let mut overrides = HashMap::new();
                let mut tools = HashMap::new();
                for tool in ACCEPT_EDITS_TOOLS {
                    tools.insert(*tool, serde_json::json!({"permission": "always"}));
                }
                overrides.insert("tools".to_string(), serde_json::json!(tools));
                overrides
            }
        }
    }

    /// Get the ordered list of modes for cycling.
    #[must_use]
    pub fn get_mode_order() -> &'static [AgentMode] {
        &[
            AgentMode::Default,
            AgentMode::Plan,
            AgentMode::AcceptEdits,
            AgentMode::AutoApprove,
        ]
    }

    /// Get the next mode in the cycle.
    #[must_use]
    pub fn next_mode(&self) -> AgentMode {
        let order = Self::get_mode_order();
        let idx = order.iter().position(|m| m == self).unwrap_or(0);
        order[(idx + 1) % order.len()]
    }
}

impl std::fmt::Display for AgentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "default"),
            Self::Plan => write!(f, "plan"),
            Self::AcceptEdits => write!(f, "accept-edits"),
            Self::AutoApprove => write!(f, "auto-approve"),
        }
    }
}

impl std::str::FromStr for AgentMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "plan" => Ok(Self::Plan),
            "accept-edits" | "accept_edits" | "acceptedits" => Ok(Self::AcceptEdits),
            "auto-approve" | "auto_approve" | "autoapprove" => Ok(Self::AutoApprove),
            _ => Err(format!("Unknown mode: {s}")),
        }
    }
}
