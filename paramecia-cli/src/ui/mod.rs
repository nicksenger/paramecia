//! Terminal UI components using ratatui.

pub mod app;
pub mod completion;
pub mod config_app;
pub mod context_progress;
mod game_of_life;
pub mod history;
pub mod input;
pub mod messages;
pub mod mode_indicator;
pub mod path_display;
mod spinner;
pub mod trust_dialog;
pub mod version_update;
pub mod welcome;
mod widgets;

pub use app::{App, AppAction};
