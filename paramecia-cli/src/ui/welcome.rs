//! Welcome banner widget.

use paramecia_harness::ParameciaConfig;
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Padding, Paragraph, Wrap},
};

use super::widgets::colors;

#[cfg(feature = "qwen3next_80b_a3b")]
const BUILD_FEATURE_NAME: &str = "qwen3next_80b_a3b";
#[cfg(feature = "qwen35moe_35b_a3b")]
const BUILD_FEATURE_NAME: &str = "qwen35moe_35b_a3b";
#[cfg(feature = "qwen35moe_122b_a10b")]
const BUILD_FEATURE_NAME: &str = "qwen35moe_122b_a10b";
#[cfg(feature = "qwen35moe_397b_a17b")]
const BUILD_FEATURE_NAME: &str = "qwen35moe_397b_a17b";
#[cfg(feature = "qwen35_0p8b")]
const BUILD_FEATURE_NAME: &str = "qwen35_0p8b";
#[cfg(feature = "qwen35_2b")]
const BUILD_FEATURE_NAME: &str = "qwen35_2b";
#[cfg(feature = "qwen35_4b")]
const BUILD_FEATURE_NAME: &str = "qwen35_4b";
#[cfg(feature = "qwen35_9b")]
const BUILD_FEATURE_NAME: &str = "qwen35_9b";
#[cfg(feature = "qwen35_27b")]
const BUILD_FEATURE_NAME: &str = "qwen35_27b";
#[cfg(not(any(
    feature = "qwen3next_80b_a3b",
    feature = "qwen35moe_35b_a3b",
    feature = "qwen35moe_122b_a10b",
    feature = "qwen35moe_397b_a17b",
    feature = "qwen35_0p8b",
    feature = "qwen35_2b",
    feature = "qwen35_4b",
    feature = "qwen35_9b",
    feature = "qwen35_27b"
)))]
const BUILD_FEATURE_NAME: &str = "qwen35_0p8b";

fn build_feature_display_name() -> String {
    format_build_feature_name(BUILD_FEATURE_NAME)
}

fn format_build_feature_name(feature_name: &str) -> String {
    let mut parts = feature_name.split('_');
    let family = match parts.next().unwrap_or_default() {
        "qwen35" => "Qwen 3.5".to_string(),
        "qwen35moe" => "Qwen 3.5 MoE".to_string(),
        "qwen3next" => "Qwen 3 Next".to_string(),
        other => title_case_token(other),
    };

    let mut display_parts = vec![family];
    for part in parts {
        display_parts.push(format_build_feature_part(part));
    }
    display_parts.join(" ")
}

fn format_build_feature_part(part: &str) -> String {
    if let Some(number) = part
        .strip_prefix('a')
        .and_then(|value| value.strip_suffix('b'))
    {
        return format!("A{}b", number.replace('p', "."));
    }

    if let Some(number) = part.strip_suffix('b') {
        return format!("{}b", number.replace('p', "."));
    }

    title_case_token(part)
}

fn title_case_token(token: &str) -> String {
    let mut chars = token.chars();
    match chars.next() {
        Some(first) => {
            let mut value = first.to_uppercase().collect::<String>();
            value.push_str(chars.as_str());
            value
        }
        None => String::new(),
    }
}

/// Welcome banner widget.
pub struct WelcomeBanner {
    /// Configuration.
    pub config: ParameciaConfig,
    /// Current animation frame for the P logo.
    pub color_index: usize,
}

impl WelcomeBanner {
    /// Create a new welcome banner.
    pub fn new(config: ParameciaConfig) -> Self {
        Self {
            config,
            color_index: 0,
        }
    }

    /// Set the animation frame.
    pub fn set_color_index(&mut self, color_index: usize) {
        self.color_index = color_index;
    }

    /// Render the welcome banner.
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        // Create animated border
        let border_color = colors::GRADIENT[self.color_index % colors::GRADIENT.len()];
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .border_type(ratatui::widgets::BorderType::Rounded)
            .padding(Padding::new(2, 0, 2, 0));

        let inner = block.inner(area);
        block.render(area, buf);

        // Create the P logo with animated gradient colors
        let logo_lines = self.create_animated_logo();

        // Create help content
        let content = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("Type ", Style::default().fg(colors::MUTED)),
                Span::styled("/help", Style::default().fg(colors::PRIMARY)),
                Span::styled(
                    " for more information • ",
                    Style::default().fg(colors::MUTED),
                ),
                Span::styled("/terminal-setup", Style::default().fg(colors::PRIMARY)),
                Span::styled(" for shift+enter", Style::default().fg(colors::MUTED)),
            ]),
        ];

        // Combine logo and content
        let mut all_lines = logo_lines;
        all_lines.extend(content);

        let paragraph = Paragraph::new(all_lines)
            .style(Style::default().fg(colors::TEXT))
            .block(Block::default().borders(Borders::NONE))
            .wrap(Wrap { trim: true })
            .alignment(Alignment::Center);

        paragraph.render(inner, buf);
    }

    /// Create the animated P logo with gradient colors and text information.
    fn create_animated_logo(&self) -> Vec<Line<'static>> {
        const BLOCK: &str = "▇▇";
        const SPACE: &str = "  ";
        const LOGO_TEXT_GAP: &str = "   ";
        let build_label = build_feature_display_name();

        // Get model and server counts
        let model_count = self.config.models.len();
        let mcp_server_count = self.config.mcp_servers.len();
        let models_text = if model_count == 1 { "model" } else { "models" };
        let servers_text = if mcp_server_count == 1 {
            "MCP server"
        } else {
            "MCP servers"
        };

        // Format workdir to use ~ for home directory
        let display_workdir = if let Ok(home) = std::env::var("HOME") {
            let workdir = self
                .config
                .effective_workdir()
                .to_string_lossy()
                .to_string();
            if workdir.starts_with(&home) {
                format!("~{}", &workdir[home.len()..])
            } else {
                workdir
            }
        } else {
            self.config
                .effective_workdir()
                .to_string_lossy()
                .to_string()
        };

        // Calculate maximum text width to ensure all lines have same total width
        let max_text_width = [
            format!("PARAMECIA v{}", env!("CARGO_PKG_VERSION")).len(),
            format!("Model: {build_label}").len(),
            format!("{model_count} {models_text} · {mcp_server_count} {servers_text}").len(),
            0, // Line 4 has no text
            display_workdir.len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        // Enhanced animation with smooth color transitions
        let get_line_color = |line_idx: usize| -> Color {
            // Use a more sophisticated color progression
            let base_offset = self.color_index % colors::GRADIENT.len();
            let line_offset = (line_idx * 2) % colors::GRADIENT.len();
            let final_offset = (base_offset + line_offset) % colors::GRADIENT.len();
            colors::GRADIENT[final_offset]
        };

        // Create the P logo with text information next to it (matching widgets.rs)
        vec![
            // Line 1: top bar - "  ▇▇▇▇▇▇▇▇▇▇  " (2 + 10 + 2 = 14)
            Line::from(vec![
                Span::raw(SPACE),
                Span::styled(
                    format!("{BLOCK}{BLOCK}{BLOCK}{BLOCK}{BLOCK}"),
                    Style::default().fg(get_line_color(0)),
                ),
                Span::raw(SPACE),
                Span::styled(
                    format!("{LOGO_TEXT_GAP}PARAMECIA v{}", env!("CARGO_PKG_VERSION")),
                    Style::default().bold(),
                ),
                Span::raw(" ".repeat(
                    max_text_width - format!("Paramecia v{}", env!("CARGO_PKG_VERSION")).len(),
                )),
            ]),
            // Line 2: left + right bump - "  ▇▇        ▇▇" (2 + 2 + 6 + 2 = 12)
            Line::from(vec![
                Span::raw(SPACE),
                Span::styled(BLOCK, Style::default().fg(get_line_color(1))),
                Span::raw(SPACE),
                Span::raw(SPACE),
                Span::raw(SPACE),
                Span::styled(BLOCK, Style::default().fg(get_line_color(1))),
                Span::styled(
                    format!("{LOGO_TEXT_GAP}Model: {build_label}"),
                    Style::default().fg(colors::MUTED),
                ),
                Span::raw(" ".repeat(max_text_width - format!("Model: {build_label}").len())),
            ]),
            // Line 3: middle bar - "  ▇▇▇▇▇▇▇▇    " (2 + 8 + 4 = 14)
            Line::from(vec![
                Span::raw(SPACE),
                Span::styled(
                    format!("{BLOCK}{BLOCK}{BLOCK}{BLOCK}"),
                    Style::default().fg(get_line_color(2)),
                ),
                Span::raw(SPACE),
                Span::raw(SPACE),
                Span::styled(
                    format!(
                        "{LOGO_TEXT_GAP}{model_count} {models_text} · {mcp_server_count} {servers_text}"
                    ),
                    Style::default().fg(colors::MUTED),
                ),
                Span::raw(
                    " ".repeat(
                        max_text_width
                            - format!(
                                "{model_count} {models_text} · {mcp_server_count} {servers_text}"
                            )
                            .len(),
                    ),
                ),
            ]),
            // Line 4: empty (gap in P shape)
            Line::from(vec![
                Span::raw(" ".repeat(15)), // Empty line with padding
            ]),
            // Line 5: left side only (P shape) - "  ▇▇" (2 + 2 = 4)
            Line::from(vec![
                Span::raw(SPACE),
                Span::styled(BLOCK, Style::default().fg(get_line_color(4))),
                Span::raw(" ".repeat(11)), // Align text with line 2
                Span::styled(
                    display_workdir.to_string(),
                    Style::default().fg(colors::MUTED),
                ),
                Span::raw(" ".repeat(max_text_width - display_workdir.len())),
            ]),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::format_build_feature_name;

    #[test]
    fn formats_dense_qwen_features() {
        assert_eq!(format_build_feature_name("qwen35_2b"), "Qwen 3.5 2b");
        assert_eq!(format_build_feature_name("qwen35_9b"), "Qwen 3.5 9b");
        assert_eq!(format_build_feature_name("qwen35_0p8b"), "Qwen 3.5 0.8b");
    }

    #[test]
    fn formats_moe_and_next_features() {
        assert_eq!(
            format_build_feature_name("qwen35moe_35b_a3b"),
            "Qwen 3.5 MoE 35b A3b"
        );
        assert_eq!(
            format_build_feature_name("qwen3next_80b_a3b"),
            "Qwen 3 Next 80b A3b"
        );
    }
}
