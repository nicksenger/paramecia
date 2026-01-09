//! Utility functions and constants for Paramecia.

/// Tag for user cancellation events.
pub const CANCELLATION_TAG: &str = "user_cancellation";

/// Tag for tool errors.
pub const TOOL_ERROR_TAG: &str = "tool_error";

/// Tag for vibe stop events.
pub const VIBE_STOP_EVENT_TAG: &str = "vibe_stop_event";

/// Tag for vibe warning messages.
pub const VIBE_WARNING_TAG: &str = "vibe_warning";

/// All known tags for parsing.
pub const KNOWN_TAGS: &[&str] = &[
    CANCELLATION_TAG,
    TOOL_ERROR_TAG,
    VIBE_STOP_EVENT_TAG,
    VIBE_WARNING_TAG,
];

/// Reason for user cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationReason {
    /// Operation was cancelled by user.
    OperationCancelled,
    /// Tool execution was interrupted.
    ToolInterrupted,
    /// Tool execution produced no response.
    ToolNoResponse,
    /// Tool was skipped by user.
    ToolSkipped,
}

/// Tagged text for structured messages.
#[derive(Debug, Clone)]
pub struct TaggedText {
    /// The message content.
    pub message: String,
    /// The tag, if any.
    pub tag: String,
}

impl TaggedText {
    /// Create a new tagged text.
    #[must_use]
    pub fn new(message: impl Into<String>, tag: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            tag: tag.into(),
        }
    }

    /// Create a tagged text without a tag.
    #[must_use]
    pub fn untagged(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            tag: String::new(),
        }
    }
}

impl std::fmt::Display for TaggedText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.tag.is_empty() {
            write!(f, "{}", self.message)
        } else {
            write!(f, "<{}>{}</{}>", self.tag, self.message, self.tag)
        }
    }
}

/// Get a user cancellation message with appropriate tag.
#[must_use]
pub fn get_user_cancellation_message(
    reason: CancellationReason,
    tool_name: Option<&str>,
) -> TaggedText {
    let message = match reason {
        CancellationReason::OperationCancelled => "User cancelled the operation.".to_string(),
        CancellationReason::ToolInterrupted => "Tool execution interrupted by user.".to_string(),
        CancellationReason::ToolNoResponse => {
            "Tool execution interrupted - no response available".to_string()
        }
        CancellationReason::ToolSkipped => tool_name
            .map(|n| format!("{} execution skipped by user.", n))
            .unwrap_or_else(|| "Tool execution skipped by user.".to_string()),
    };

    TaggedText::new(message, CANCELLATION_TAG)
}

/// Check if an event represents a user cancellation.
pub fn is_user_cancellation_event(skip_reason: Option<&str>) -> bool {
    skip_reason.is_some_and(|reason| reason.contains(&format!("<{}>", CANCELLATION_TAG)))
}

/// Check if we're running on Windows.
#[must_use]
pub fn is_windows() -> bool {
    cfg!(target_os = "windows")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tagged_text_display() {
        let tagged = TaggedText::new("Test message", "tool_error");
        assert_eq!(tagged.to_string(), "<tool_error>Test message</tool_error>");

        let untagged = TaggedText::untagged("Plain message");
        assert_eq!(untagged.to_string(), "Plain message");
    }

    #[test]
    fn test_cancellation_messages() {
        let msg = get_user_cancellation_message(CancellationReason::OperationCancelled, None);
        assert!(msg.to_string().contains("User cancelled the operation"));
        assert!(msg.to_string().contains("<user_cancellation>"));

        let msg = get_user_cancellation_message(CancellationReason::ToolSkipped, Some("bash"));
        assert!(msg.to_string().contains("bash execution skipped"));
    }

    #[test]
    fn test_is_user_cancellation_event() {
        assert!(is_user_cancellation_event(Some(
            "<user_cancellation>Skipped</user_cancellation>"
        )));
        assert!(!is_user_cancellation_event(Some("Just skipped")));
        assert!(!is_user_cancellation_event(None));
    }
}
