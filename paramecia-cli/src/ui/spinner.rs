//! Spinner animation for loading states.

use rand::Rng;

/// Braille spinner frames for smooth animation (used for loading indicator).
const BRAILLE_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Gradient colors for the animated loading effect.
#[allow(dead_code)]
pub const GRADIENT_COLORS: &[&str] = &["#FFD800", "#FFAF00", "#FF8205", "#FA500F", "#E10500"];

/// Standard easter egg messages.
const EASTER_EGGS: &[&str] = &[
    "Simmering phở",
    "Drinking cà phê",
    "Eating bánh mì",
    "Petting con chó",
    "Brewing bia hơi",
    "Starting xe máy",
];

/// A spinner for loading animations.
#[derive(Debug, Clone)]
pub struct Spinner {
    frame_index: usize,
}

impl Default for Spinner {
    fn default() -> Self {
        Self::new()
    }
}

impl Spinner {
    /// Create a new braille spinner.
    #[must_use]
    pub fn new() -> Self {
        Self { frame_index: 0 }
    }

    /// Get the current frame character.
    #[must_use]
    pub fn current_frame(&self) -> char {
        BRAILLE_FRAMES[self.frame_index]
    }

    /// Advance to the next frame and return it.
    pub fn next_frame(&mut self) -> char {
        self.frame_index = (self.frame_index + 1) % BRAILLE_FRAMES.len();
        BRAILLE_FRAMES[self.frame_index]
    }

    /// Reset the spinner to the first frame.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.frame_index = 0;
    }
}

/// Get a random easter egg message (for use when easter egg is triggered).
/// Returns None if the random check fails (90% of the time).
fn get_easter_egg() -> Option<&'static str> {
    let mut rng = rand::rng();

    // 10% chance of easter egg (matching Mistral Vibe)
    const EASTER_EGG_PROBABILITY: f64 = 0.10;
    if rng.random::<f64>() >= EASTER_EGG_PROBABILITY {
        return None;
    }

    let idx = rng.random_range(0..EASTER_EGGS.len());
    EASTER_EGGS.get(idx).copied()
}

/// Easter egg messages for loading states (matching Mistral Vibe style).
/// Uses proper random selection and includes holiday-specific messages.
pub fn get_loading_message() -> &'static str {
    get_easter_egg().unwrap_or("Thinking")
}

/// Apply easter egg logic to a status message (matching Mistral Vibe's _apply_easter_egg).
/// Has a 10% chance to replace the status with an easter egg.
pub fn apply_easter_egg(status: &str) -> String {
    get_easter_egg()
        .map(String::from)
        .unwrap_or_else(|| status.to_string())
}

/// Get status text for a tool (matching Mistral Vibe's get_status_text pattern).
pub fn get_tool_status_text(tool_name: &str) -> &'static str {
    match tool_name {
        "read" | "Read" => "Reading",
        "write" | "Write" => "Writing",
        "edit" | "Edit" | "str_replace_editor" => "Editing",
        "bash" | "Bash" | "shell" | "Shell" => "Running command",
        "grep" | "Grep" => "Searching",
        "glob" | "Glob" => "Finding files",
        "ls" | "LS" => "Listing directory",
        "todo" | "Todo" | "TodoWrite" => "Updating todos",
        "semantic_search" | "SemanticSearch" => "Semantic search",
        "web_search" | "WebSearch" => "Searching the web",
        "agent" | "Agent" => "Delegating to agent",
        _ => "Running tool",
    }
}
