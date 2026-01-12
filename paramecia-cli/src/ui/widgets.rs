//! Widget rendering for the TUI.

use paramecia_harness::modes::AgentMode;
use pulldown_cmark::{Alignment, Event, Options, Parser, Tag, TagEnd};
use ratatui::{
    prelude::*,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Padding, Paragraph, Wrap},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use super::messages::*;
use super::spinner::Spinner;

/// Safely truncate a string to a maximum number of characters (not bytes).
/// This handles UTF-8 multi-byte characters correctly to avoid panics.
fn truncate_string(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
        format!("{}...", truncated)
    }
}

/// Colors for Paramecia's theme.
#[allow(dead_code)]
pub mod colors {
    use ratatui::style::{Color, Style};

    // Theme colors
    pub const PRIMARY: Color = Color::Rgb(0, 150, 255); // Blue primary
    pub const SUCCESS: Color = Color::Rgb(0, 200, 100); // Green for success
    pub const WARNING: Color = Color::Rgb(0, 200, 100); // Green for warnings
    pub const ERROR: Color = Color::Rgb(200, 50, 50); // Red for errors
    pub const MUTED: Color = Color::Rgb(128, 128, 128); // Gray for muted text (foreground-muted)
    pub const BORDER: Color = Color::Rgb(0, 100, 200); // #b05800 - Dark blue border
    pub const BACKGROUND: Color = Color::Rgb(20, 20, 20); // Dark background
    pub const SURFACE: Color = Color::Rgb(38, 38, 38); // $surface - slightly lighter than background
    pub const TEXT: Color = Color::Rgb(220, 220, 220); // Light text
    pub const ACCENT: Color = Color::Rgb(180, 220, 80); // Lime-green accent (biological theme)
    pub const YELLOW: Color = Color::Rgb(200, 230, 100); // Bright lime-green for highlights

    pub const MODE_SAFE: Color = Color::Rgb(0, 200, 100); // Green for safe mode
    pub const MODE_NEUTRAL: Color = Color::Rgb(128, 128, 128); // Gray for neutral mode
    pub const MODE_DESTRUCTIVE: Color = Color::Rgb(180, 220, 80); // Lime-green for destructive
    pub const MODE_YOLO: Color = Color::Rgb(0, 200, 100); // Green for yolo mode

    /// Gradient colors for animations (Blue primary gradient).
    pub const GRADIENT: &[Color] = &[
        Color::Rgb(0, 220, 255),   // Light cyan/blue
        Color::Rgb(0, 180, 255),   // Medium blue
        Color::Rgb(0, 150, 255),   // Primary blue
        Color::Rgb(0, 120, 200),   // Darker blue
        Color::Rgb(0, 200, 100),   // Bright green
        Color::Rgb(200, 230, 100), // Lime-green (biological theme)
    ];

    /// Foreground colors
    pub const FOREGROUND: Color = Color::Rgb(220, 220, 220); // $foreground
    pub const FOREGROUND_MUTED: Color = Color::Rgb(128, 128, 128); // $foreground-muted
    pub const TEXT_MUTED: Color = Color::Rgb(100, 100, 100); // $text-muted

    /// User message colors
    pub const USER_NAME: Color = Color::Rgb(0, 180, 255); // Blue for user name
    pub const USER_TEXT: Color = Color::Rgb(220, 220, 220); // White for user text

    /// Assistant message colors
    pub const ASSISTANT_NAME: Color = Color::Rgb(0, 150, 255); // Blue for assistant
    pub const ASSISTANT_TEXT: Color = Color::Rgb(220, 220, 220); // White for assistant text

    /// Tool colors
    pub const TOOL_NAME: Color = Color::Rgb(180, 120, 255); // Purple for tool names
    pub const TOOL_ARGS: Color = Color::Rgb(120, 200, 120); // Green for tool args
    pub const TOOL_SUCCESS: Color = Color::Rgb(120, 200, 120); // Green for success
    pub const TOOL_ERROR: Color = Color::Rgb(255, 120, 120); // Red for errors

    /// Trust dialog colors
    pub const DIALOG_BG: Color = Color::Rgb(30, 30, 30); // Dark background
    pub const DIALOG_TEXT: Color = Color::Rgb(220, 220, 220); // Light text
    pub const PATH_TEXT: Color = Color::Rgb(0, 150, 255); // Blue for path
    pub const OPTION_TEXT: Color = Color::Rgb(200, 200, 200); // Light gray for options
    pub const SELECTED_OPTION: Color = Color::Rgb(0, 150, 255); // Blue for selected
    pub const HELP_TEXT: Color = Color::Rgb(100, 100, 100); // Gray for help text (matching original)
    pub const SAVE_INFO_TEXT: Color = Color::Rgb(100, 180, 255); // Blue for save info

    /// Trust dialog color functions
    pub fn dialog_bg() -> Style {
        Style::default().bg(DIALOG_BG)
    }

    pub fn dialog_text() -> Style {
        Style::default().fg(DIALOG_TEXT)
    }

    pub fn path_text() -> Style {
        Style::default().fg(PATH_TEXT)
    }

    pub fn option_text() -> Style {
        Style::default().fg(OPTION_TEXT)
    }

    pub fn selected_option() -> Style {
        use ratatui::prelude::Stylize;
        Style::default().fg(SELECTED_OPTION).bold()
    }

    pub fn help_text() -> Style {
        Style::default().fg(HELP_TEXT)
    }

    pub fn save_info_text() -> Style {
        Style::default().fg(SAVE_INFO_TEXT)
    }

    /// Get the appropriate border color for the current mode
    pub fn mode_border_color(mode: &paramecia_harness::modes::AgentMode) -> Color {
        use paramecia_harness::modes::AgentMode;
        match mode {
            AgentMode::Default => MODE_NEUTRAL,
            AgentMode::Plan => MODE_SAFE,
            AgentMode::AcceptEdits => MODE_DESTRUCTIVE,
            AgentMode::AutoApprove => MODE_YOLO,
        }
    }
}

/// Render a user message.
/// In Python, user messages have:
///   - margin-top: 1
///   - padding: 1 0 (vertical padding)
///   - background: $surface
///     We simulate the surface background with box-drawing characters.
pub fn render_user_message(msg: &UserMessage, width: u16) -> Vec<Line<'static>> {
    let prompt_style = Style::default().fg(colors::YELLOW).bold();

    let content_style = if msg.pending {
        Style::default().fg(colors::FOREGROUND_MUTED).italic()
    } else {
        Style::default().fg(colors::FOREGROUND).bold()
    };

    let mut lines = Vec::new();

    // margin-top: 1
    lines.push(Line::from(""));

    // Create a surface-styled line (simulating padding and background)
    // Python uses padding: 1 0 which adds 1 line of padding top/bottom
    let surface_style = Style::default().bg(colors::SURFACE);

    // Top padding line with surface background
    let padding_line: String = " ".repeat(width as usize);
    lines.push(Line::styled(padding_line.clone(), surface_style));

    // Content line with surface background
    lines.push(Line::from(vec![
        Span::styled("> ", prompt_style.bg(colors::SURFACE)),
        Span::styled(msg.content.clone(), content_style.bg(colors::SURFACE)),
        // Fill remaining width with surface color
        Span::styled(
            " ".repeat((width as usize).saturating_sub(msg.content.len() + 2)),
            surface_style,
        ),
    ]));

    // Bottom padding line with surface background
    lines.push(Line::styled(padding_line, surface_style));

    lines
}

/// Render an assistant message with markdown formatting.
/// margin-top: 1 (empty line before)
pub fn render_assistant_message(msg: &AssistantMessage, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    // margin-top: 1
    lines.push(Line::from(""));

    if msg.content.is_empty() {
        // If no content yet, just show the bullet
        lines.push(Line::from(vec![Span::styled(
            "● ",
            Style::default().fg(colors::PRIMARY),
        )]));
        return lines;
    }

    let max_table_width = usize::from(width.max(1)).saturating_sub(2);

    // Parse markdown and render with styling
    let rendered_lines = render_markdown(&msg.content, max_table_width);

    let bullet_prefix = vec![Span::styled("● ", Style::default().fg(colors::YELLOW))];
    let indent_prefix = vec![Span::raw("  ")];
    let mut first_line = true;

    for line in rendered_lines {
        let wrapped_lines = if first_line {
            first_line = false;
            wrap_line_with_prefix(line, width, &bullet_prefix, &indent_prefix)
        } else {
            wrap_line_with_prefix(line, width, &indent_prefix, &indent_prefix)
        };

        lines.extend(wrapped_lines);
    }

    lines
}

/// Wrap a markdown-rendered line with prefixes for the first and subsequent wrapped lines.
fn wrap_line_with_prefix(
    mut line: Line<'static>,
    width: u16,
    first_prefix: &[Span<'static>],
    continuation_prefix: &[Span<'static>],
) -> Vec<Line<'static>> {
    let max_width = usize::from(width.max(1));

    let continuation_width = spans_width(continuation_prefix).min(max_width);
    let mut lines = Vec::new();
    let mut current_spans = first_prefix.to_vec();
    let mut current_width = spans_width(&current_spans).min(max_width);

    for span in line.spans.drain(..) {
        let mut remaining = span.content.into_owned();

        while !remaining.is_empty() {
            let available = max_width.saturating_sub(current_width.min(max_width));

            if available == 0 {
                lines.push(Line::from(std::mem::take(&mut current_spans)));
                current_spans = continuation_prefix.to_vec();
                current_width = continuation_width;

                if current_width >= max_width {
                    current_spans.clear();
                    current_width = 0;
                }
                continue;
            }

            let (fit, rest) = split_content_at_width(&remaining, available);

            if !fit.is_empty() {
                current_width = current_width
                    .saturating_add(display_width(&fit))
                    .min(max_width);
                current_spans.push(Span::styled(fit, span.style));
            }

            remaining = rest;

            if !remaining.is_empty() {
                lines.push(Line::from(std::mem::take(&mut current_spans)));
                current_spans = continuation_prefix.to_vec();
                current_width = continuation_width;

                if current_width >= max_width {
                    current_spans.clear();
                    current_width = 0;
                }
            }
        }
    }

    if current_spans.is_empty() && continuation_prefix.is_empty() {
        lines.push(Line::default());
    } else if current_spans.is_empty() {
        lines.push(Line::from(continuation_prefix.to_vec()));
    } else {
        lines.push(Line::from(current_spans));
    }

    lines
}

/// Compute the display width of a string using unicode width rules.
fn display_width(content: &str) -> usize {
    UnicodeWidthStr::width(content)
}

/// Compute the display width of a set of spans.
fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(Span::width).sum()
}

/// Split a string at the maximum width, keeping grapheme clusters intact.
fn split_content_at_width(content: &str, max_width: usize) -> (String, String) {
    if max_width == 0 {
        return (String::new(), content.to_string());
    }

    let mut current_width = 0usize;

    for (idx, grapheme) in content.grapheme_indices(true) {
        let next_width = current_width.saturating_add(UnicodeWidthStr::width(grapheme));
        if next_width > max_width {
            return (content[..idx].to_string(), content[idx..].to_string());
        }
        current_width = next_width;
    }

    (content.to_string(), String::new())
}

fn preprocess_markdown_tables(content: &str) -> String {
    let mut output: Vec<String> = Vec::new();
    let mut table_buffer: Vec<String> = Vec::new();
    let mut in_code_block = false;

    for line in content.lines() {
        let trimmed = line.trim_start();
        let is_fence = trimmed.starts_with("```") || trimmed.starts_with("~~~");

        if is_fence {
            flush_table_buffer(&mut table_buffer, &mut output);
            in_code_block = !in_code_block;
            output.push(line.to_string());
            continue;
        }

        if in_code_block {
            output.push(line.to_string());
            continue;
        }

        if is_tableish_line(line) {
            table_buffer.push(line.to_string());
        } else {
            flush_table_buffer(&mut table_buffer, &mut output);
            output.push(line.to_string());
        }
    }

    flush_table_buffer(&mut table_buffer, &mut output);

    output.join("\n")
}

fn flush_table_buffer(buffer: &mut Vec<String>, output: &mut Vec<String>) {
    if buffer.is_empty() {
        return;
    }

    let normalized = normalize_table_block(buffer);
    output.extend(normalized);
    buffer.clear();
}

fn is_tableish_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    let normalized = normalize_delimiters(trimmed);
    let delimiter_count = normalized.matches('|').count();
    if delimiter_count < 2 {
        return false;
    }

    let cell_count = normalized
        .trim_matches('|')
        .split('|')
        .filter(|cell| !cell.trim().is_empty())
        .count();

    cell_count >= 2
}

fn normalize_delimiters(line: &str) -> String {
    let mut normalized = line.replace('+', "|");
    // Treat adjacent pipes as a single delimiter to be forgiving.
    while normalized.contains("||") {
        normalized = normalized.replace("||", "|");
    }
    normalized
}

fn is_border_only_line(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|ch| ch == '+' || ch == '|' || ch == '-' || ch == '=')
}

fn is_separator_cells(cells: &[String]) -> bool {
    cells.iter().all(|cell| {
        cell.trim()
            .chars()
            .all(|ch| ch == '-' || ch == ':' || ch.is_whitespace())
    })
}

fn normalize_alignment_cell(cell: &str) -> String {
    let trimmed = cell.trim();
    let left = trimmed.starts_with(':');
    let right = trimmed.ends_with(':');
    let dash_count = trimmed.chars().filter(|ch| *ch == '-').count();
    let count = dash_count.max(3);

    format!(
        "{}{}{}",
        if left { ":" } else { "" },
        "-".repeat(count),
        if right { ":" } else { "" }
    )
}

fn normalize_table_block(lines: &[String]) -> Vec<String> {
    let mut rows: Vec<Vec<String>> = Vec::new();

    for line in lines {
        if is_border_only_line(line) {
            continue;
        }

        let replaced = normalize_delimiters(line);
        let trimmed = replaced.trim_matches('|');
        let cells: Vec<String> = trimmed
            .split('|')
            .map(|cell| cell.trim().to_string())
            .collect();

        if cells.iter().all(|cell| cell.is_empty()) {
            continue;
        }

        rows.push(cells);
    }

    if rows.is_empty() {
        return lines.to_vec();
    }

    let column_count = rows.iter().map(Vec::len).max().unwrap_or(0);
    if column_count < 2 {
        return lines.to_vec();
    }

    for row in rows.iter_mut() {
        if row.len() < column_count {
            row.resize(column_count, String::new());
        }
    }

    let mut normalized: Vec<String> = Vec::new();

    normalized.push(format!("| {} |", rows[0].join(" | ")));

    let (alignment_row, data_start_idx) = if rows.len() > 1 && is_separator_cells(&rows[1]) {
        (Some(rows[1].clone()), 2usize)
    } else {
        (None, 1usize)
    };

    let separator_cells = match alignment_row {
        Some(cells) => cells
            .iter()
            .map(|cell| normalize_alignment_cell(cell))
            .collect::<Vec<String>>(),
        None => (0..column_count).map(|_| String::from("---")).collect(),
    };

    normalized.push(format!("| {} |", separator_cells.join(" | ")));

    for row in rows.into_iter().skip(data_start_idx) {
        normalized.push(format!("| {} |", row.join(" | ")));
    }

    normalized
}

#[derive(Default)]
struct TableState {
    alignments: Vec<Alignment>,
    header_rows: Vec<Vec<Vec<Span<'static>>>>,
    body_rows: Vec<Vec<Vec<Span<'static>>>>,
    current_row: Vec<Vec<Span<'static>>>,
    current_cell: Vec<Span<'static>>,
    in_header: bool,
}

impl TableState {
    fn new(alignments: Vec<Alignment>) -> Self {
        Self {
            alignments,
            ..Self::default()
        }
    }

    fn start_head(&mut self) {
        self.in_header = true;
    }

    fn end_head(&mut self) {
        self.finish_row();
        self.in_header = false;
    }

    fn start_row(&mut self) {
        self.current_cell.clear();
        self.current_row.clear();
    }

    fn finish_row(&mut self) {
        if !self.current_cell.is_empty() {
            self.finish_cell();
        }

        if self.current_row.is_empty() {
            return;
        }

        if self.in_header {
            for cell in self.current_row.iter_mut() {
                for span in cell.iter_mut() {
                    span.style = span.style.add_modifier(Modifier::BOLD);
                }
            }
            self.header_rows.push(std::mem::take(&mut self.current_row));
        } else {
            self.body_rows.push(std::mem::take(&mut self.current_row));
        }
    }

    fn start_cell(&mut self) {
        self.current_cell.clear();
    }

    fn finish_cell(&mut self) {
        self.current_row
            .push(std::mem::take(&mut self.current_cell));
    }

    fn push_span(&mut self, span: Span<'static>) {
        self.current_cell.push(span);
    }

    fn column_count(&self) -> usize {
        let header_columns = self.header_rows.iter().map(Vec::len).max().unwrap_or(0);
        let body_columns = self.body_rows.iter().map(Vec::len).max().unwrap_or(0);
        header_columns.max(body_columns)
    }
}

fn compute_column_widths(table: &TableState, max_table_width: usize) -> Vec<usize> {
    let column_count = table.column_count();
    if column_count == 0 || max_table_width == 0 {
        return Vec::new();
    }

    let mut column_widths = vec![1usize; column_count];
    for row in table.header_rows.iter().chain(table.body_rows.iter()) {
        for (idx, cell) in row.iter().enumerate() {
            column_widths[idx] = column_widths[idx].max(spans_width(cell));
        }
    }

    let minimum_table_width = 4usize.saturating_mul(column_count).saturating_add(1);
    if max_table_width < minimum_table_width {
        return vec![1; column_count];
    }

    let max_content_width =
        max_table_width.saturating_sub(3usize.saturating_mul(column_count).saturating_add(1));
    if max_content_width == 0 {
        return vec![1; column_count];
    }
    let total_content_width: usize = column_widths.iter().sum();

    if total_content_width <= max_content_width {
        return column_widths;
    }

    distribute_column_widths(column_widths, max_content_width)
}

fn distribute_column_widths(column_widths: Vec<usize>, max_total_content: usize) -> Vec<usize> {
    if column_widths.is_empty() {
        return column_widths;
    }
    if max_total_content == 0 {
        return vec![1; column_widths.len()];
    }

    let base_widths: Vec<usize> = column_widths
        .into_iter()
        .map(|width| width.max(1))
        .collect();
    let total: usize = base_widths.iter().sum();

    if total == 0 || total <= max_total_content {
        return base_widths;
    }

    let total_f = total as f64;
    let limit_f = max_total_content as f64;

    let mut scaled_widths: Vec<usize> = base_widths
        .iter()
        .map(|width| {
            let scaled = (*width as f64) * limit_f / total_f;
            scaled.floor() as usize
        })
        .map(|width| width.max(1))
        .collect();

    let mut remainders: Vec<(usize, f64)> = base_widths
        .iter()
        .enumerate()
        .map(|(idx, width)| {
            let scaled = (*width as f64) * limit_f / total_f;
            let floored = scaled.floor().max(1.0);
            (idx, scaled - floored)
        })
        .collect();

    let mut remaining = max_total_content.saturating_sub(scaled_widths.iter().sum());

    remainders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut remainder_idx = 0usize;
    while remaining > 0 && !remainders.is_empty() {
        let (idx, _) = remainders[remainder_idx];
        scaled_widths[idx] = scaled_widths[idx].saturating_add(1);
        remaining = remaining.saturating_sub(1);
        remainder_idx = (remainder_idx + 1) % remainders.len();
    }

    scaled_widths
}

fn wrap_cell_spans(spans: &[Span<'static>], max_width: usize) -> Vec<Vec<Span<'static>>> {
    if max_width == 0 {
        return vec![Vec::new()];
    }

    let mut lines: Vec<Vec<Span<'static>>> = Vec::new();
    let mut current_line: Vec<Span<'static>> = Vec::new();
    let mut current_width = 0usize;

    for span in spans {
        let mut remaining = span.content.to_string();
        let style = span.style;

        while !remaining.is_empty() {
            let available = max_width.saturating_sub(current_width.min(max_width));
            if available == 0 {
                lines.push(std::mem::take(&mut current_line));
                current_width = 0;
                continue;
            }

            let (fit, rest) = split_content_at_width(&remaining, available);

            if !fit.is_empty() {
                current_width = current_width
                    .saturating_add(display_width(&fit))
                    .min(max_width);
                current_line.push(Span::styled(fit, style));
            }

            remaining = rest;

            if !remaining.is_empty() {
                lines.push(std::mem::take(&mut current_line));
                current_width = 0;
            }
        }
    }

    if current_line.is_empty() {
        if lines.is_empty() {
            lines.push(Vec::new());
        }
    } else {
        lines.push(current_line);
    }

    lines
}

fn wrap_row_cells(
    row: &[Vec<Span<'static>>],
    column_widths: &[usize],
) -> Vec<Vec<Vec<Span<'static>>>> {
    column_widths
        .iter()
        .enumerate()
        .map(|(idx, width)| {
            let cell = row.get(idx).cloned().unwrap_or_else(|| vec![Span::raw("")]);
            wrap_cell_spans(&cell, *width)
        })
        .collect()
}

fn render_table_rows(
    rows: &[Vec<Vec<Span<'static>>>],
    alignments: &[Alignment],
    column_widths: &[usize],
    border_style: Style,
    lines: &mut Vec<Line<'static>>,
) {
    for row in rows {
        let wrapped_cells = wrap_row_cells(row, column_widths);
        let row_height = wrapped_cells
            .iter()
            .map(|cell_lines| cell_lines.len())
            .max()
            .unwrap_or(1);

        for line_idx in 0..row_height {
            let mut spans = vec![Span::styled("│", border_style)];

            for (col_idx, col_width) in column_widths.iter().enumerate() {
                let line_spans = wrapped_cells
                    .get(col_idx)
                    .and_then(|cell| cell.get(line_idx))
                    .cloned()
                    .unwrap_or_else(Vec::new);
                let alignment = alignments.get(col_idx).copied().unwrap_or(Alignment::Left);
                let content_width = spans_width(&line_spans).min(*col_width);
                let remaining = col_width.saturating_sub(content_width);
                let (left_pad, right_pad) = match alignment {
                    Alignment::Right => (remaining, 0),
                    Alignment::Center => {
                        let half = remaining / 2;
                        (half, remaining.saturating_sub(half))
                    }
                    _ => (0, remaining),
                };

                spans.push(Span::raw(" ".repeat(left_pad.saturating_add(1))));
                spans.extend(line_spans);
                spans.push(Span::raw(" ".repeat(right_pad.saturating_add(1))));
                spans.push(Span::styled("│", border_style));
            }

            lines.push(Line::from(spans));
        }
    }
}

fn render_table(table: TableState, max_width: usize) -> Vec<Line<'static>> {
    let column_count = table.column_count();
    if column_count == 0 {
        return Vec::new();
    }

    let column_widths = compute_column_widths(&table, max_width);
    if column_widths.is_empty() {
        return Vec::new();
    }

    let border_style = Style::default().fg(colors::MUTED);
    let mut lines = Vec::new();

    lines.push(render_separator(
        &column_widths,
        border_style,
        '┌',
        '┬',
        '┐',
    ));

    render_table_rows(
        &table.header_rows,
        &table.alignments,
        &column_widths,
        border_style,
        &mut lines,
    );

    if !table.header_rows.is_empty() && !table.body_rows.is_empty() {
        lines.push(render_separator(
            &column_widths,
            border_style,
            '├',
            '┼',
            '┤',
        ));
    } else if !table.header_rows.is_empty() {
        lines.push(render_separator(
            &column_widths,
            border_style,
            '├',
            '┴',
            '┤',
        ));
    }

    for (row_idx, row) in table.body_rows.iter().enumerate() {
        render_table_rows(
            std::slice::from_ref(row),
            &table.alignments,
            &column_widths,
            border_style,
            &mut lines,
        );

        if row_idx + 1 < table.body_rows.len() {
            lines.push(render_separator(
                &column_widths,
                border_style,
                '├',
                '┼',
                '┤',
            ));
        }
    }

    lines.push(render_separator(
        &column_widths,
        border_style,
        '└',
        '┴',
        '┘',
    ));

    lines
}

fn render_separator(
    column_widths: &[usize],
    border_style: Style,
    left: char,
    middle: char,
    right: char,
) -> Line<'static> {
    let mut spans = Vec::new();
    spans.push(Span::styled(left.to_string(), border_style));

    for (idx, width) in column_widths.iter().enumerate() {
        let cell_width = width.saturating_add(2);
        spans.push(Span::styled("─".repeat(cell_width), border_style));

        if idx + 1 == column_widths.len() {
            spans.push(Span::styled(right.to_string(), border_style));
        } else {
            spans.push(Span::styled(middle.to_string(), border_style));
        }
    }

    Line::from(spans)
}

/// Render markdown content to styled lines.
fn render_markdown(content: &str, max_table_width: usize) -> Vec<Line<'static>> {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);

    let preprocessed = preprocess_markdown_tables(content);
    let parser = Parser::new_ext(&preprocessed, options);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut current_spans: Vec<Span<'static>> = Vec::new();

    // Style stack for nested formatting
    let mut bold = false;
    let mut italic = false;
    let mut in_code_block = false;
    let mut heading_level: Option<u8> = None;
    let mut list_depth: usize = 0;
    let mut table_state: Option<TableState> = None;

    for event in parser {
        match event {
            Event::Start(tag) => {
                match tag {
                    Tag::Table(alignments) => {
                        if !current_spans.is_empty() {
                            lines.push(Line::from(std::mem::take(&mut current_spans)));
                        }
                        table_state = Some(TableState::new(alignments));
                    }
                    Tag::TableHead => {
                        if let Some(table) = table_state.as_mut() {
                            table.start_head();
                        }
                    }
                    Tag::TableRow => {
                        if let Some(table) = table_state.as_mut() {
                            table.start_row();
                        }
                    }
                    Tag::TableCell => {
                        if let Some(table) = table_state.as_mut() {
                            table.start_cell();
                        }
                    }
                    Tag::Heading { level, .. } => {
                        heading_level = Some(level as u8);
                    }
                    Tag::Paragraph => {}
                    Tag::CodeBlock(_) => {
                        in_code_block = true;
                        // Flush current line before code block
                        if !current_spans.is_empty() {
                            lines.push(Line::from(std::mem::take(&mut current_spans)));
                        }
                    }
                    Tag::Strong => {
                        bold = true;
                    }
                    Tag::Emphasis => {
                        italic = true;
                    }
                    Tag::List(_) => {
                        list_depth += 1;
                    }
                    Tag::Item => {
                        // Flush current line
                        if !current_spans.is_empty() {
                            lines.push(Line::from(std::mem::take(&mut current_spans)));
                        }
                        // Add list bullet with indentation
                        let indent = "  ".repeat(list_depth.saturating_sub(1));
                        current_spans.push(Span::styled(
                            format!("{}• ", indent),
                            Style::default().fg(colors::PRIMARY),
                        ));
                    }
                    Tag::Link { dest_url, .. } => {
                        // We'll render the link text, then show the URL
                        current_spans.push(Span::styled("[", Style::default().fg(colors::ACCENT)));
                        // Store URL for later (simplified: just mark we're in a link)
                        let _ = dest_url; // Will be shown in End event
                    }
                    Tag::BlockQuote(_) => {
                        if !current_spans.is_empty() {
                            lines.push(Line::from(std::mem::take(&mut current_spans)));
                        }
                        current_spans.push(Span::styled("│ ", Style::default().fg(colors::MUTED)));
                    }
                    _ => {}
                }
            }
            Event::End(tag_end) => {
                match tag_end {
                    TagEnd::Heading(_) => {
                        heading_level = None;
                        if !current_spans.is_empty() {
                            lines.push(Line::from(std::mem::take(&mut current_spans)));
                        }
                        // Add blank line after heading
                        lines.push(Line::from(""));
                    }
                    TagEnd::Paragraph => {
                        if !current_spans.is_empty() {
                            lines.push(Line::from(std::mem::take(&mut current_spans)));
                        }
                    }
                    TagEnd::TableHead => {
                        if let Some(table) = table_state.as_mut() {
                            table.end_head();
                        }
                    }
                    TagEnd::TableRow => {
                        if let Some(table) = table_state.as_mut() {
                            table.finish_row();
                        }
                    }
                    TagEnd::TableCell => {
                        if let Some(table) = table_state.as_mut() {
                            table.finish_cell();
                        }
                    }
                    TagEnd::Table => {
                        if let Some(table) = table_state.take() {
                            lines.extend(render_table(table, max_table_width));
                            lines.push(Line::from(""));
                        }
                    }
                    TagEnd::CodeBlock => {
                        in_code_block = false;
                    }
                    TagEnd::Strong => {
                        bold = false;
                    }
                    TagEnd::Emphasis => {
                        italic = false;
                    }
                    TagEnd::List(_) => {
                        list_depth = list_depth.saturating_sub(1);
                    }
                    TagEnd::Item => {
                        if !current_spans.is_empty() {
                            lines.push(Line::from(std::mem::take(&mut current_spans)));
                        }
                    }
                    TagEnd::Link => {
                        current_spans.push(Span::styled("]", Style::default().fg(colors::ACCENT)));
                    }
                    TagEnd::BlockQuote(_) => {
                        if !current_spans.is_empty() {
                            lines.push(Line::from(std::mem::take(&mut current_spans)));
                        }
                    }
                    _ => {}
                }
            }
            Event::Text(text) => {
                let text_str = text.to_string();

                if let Some(table) = table_state.as_mut() {
                    if in_code_block {
                        for (line_idx, line) in text_str.lines().enumerate() {
                            if line_idx > 0 {
                                table.push_span(Span::raw(" "));
                            }
                            table.push_span(Span::styled(
                                line.to_string(),
                                Style::default().bg(colors::SURFACE).fg(colors::FOREGROUND),
                            ));
                        }
                    } else {
                        let style = build_text_style(bold, italic, heading_level);
                        table.push_span(Span::styled(text_str, style));
                    }
                    continue;
                }

                if in_code_block {
                    // Render code block lines with background
                    for line in text_str.lines() {
                        lines.push(Line::from(vec![Span::styled(
                            line.to_string(),
                            Style::default().bg(colors::SURFACE).fg(colors::FOREGROUND),
                        )]));
                    }
                } else {
                    let style = build_text_style(bold, italic, heading_level);
                    current_spans.push(Span::styled(text_str, style));
                }
            }
            Event::Code(code) => {
                if let Some(table) = table_state.as_mut() {
                    table.push_span(Span::styled(
                        format!("`{}`", code),
                        Style::default().bg(colors::SURFACE).fg(colors::FOREGROUND),
                    ));
                    continue;
                }

                // Inline code
                current_spans.push(Span::styled(
                    format!("`{}`", code),
                    Style::default().bg(colors::SURFACE).fg(colors::FOREGROUND),
                ));
            }
            Event::SoftBreak => {
                if let Some(table) = table_state.as_mut() {
                    table.push_span(Span::raw(" "));
                    continue;
                }

                current_spans.push(Span::raw(" "));
            }
            Event::HardBreak => {
                if let Some(table) = table_state.as_mut() {
                    table.push_span(Span::raw(" "));
                    continue;
                }

                if !current_spans.is_empty() {
                    lines.push(Line::from(std::mem::take(&mut current_spans)));
                }
            }
            Event::Rule => {
                if !current_spans.is_empty() {
                    lines.push(Line::from(std::mem::take(&mut current_spans)));
                }
                lines.push(Line::from(vec![Span::styled(
                    "────────────────────────────────",
                    Style::default().fg(colors::MUTED),
                )]));
            }
            _ => {}
        }
    }

    // Flush any remaining content
    if !current_spans.is_empty() {
        lines.push(Line::from(current_spans));
    }

    lines
}

/// Build a text style based on current formatting state.
fn build_text_style(bold: bool, italic: bool, heading_level: Option<u8>) -> Style {
    let mut style = Style::default();

    if let Some(level) = heading_level {
        match level {
            1 => {
                style = style.fg(colors::PRIMARY).add_modifier(Modifier::BOLD);
            }
            2 => {
                style = style.fg(colors::ACCENT).add_modifier(Modifier::BOLD);
            }
            3 => {
                style = style.fg(colors::SUCCESS).add_modifier(Modifier::BOLD);
            }
            4..=6 => {
                style = style.add_modifier(Modifier::BOLD);
            }
            _ => {}
        }
    } else {
        if bold {
            style = style.add_modifier(Modifier::BOLD);
        }
        if italic {
            style = style.add_modifier(Modifier::ITALIC);
        }
    }

    style
}

/// Game of Life-inspired cell patterns for tool call spinners.
/// These represent evolving cell states in a mini 1D cellular automaton.
const GOL_FRAMES: &[&str] = &["●○○", "○●○", "○○●", "○●○", "●●○", "○●●", "●○●", "●●●"];

/// Render a tool call message with Game of Life-style spinner.
/// margin-top: 1 (empty line before)
pub fn render_tool_call(
    msg: &ToolCallMessage,
    _spinner: &Spinner,
    color_index: usize,
) -> Vec<Line<'static>> {
    let icon = if msg.spinning {
        // Use Game of Life cell pattern for tool calls
        let frame = GOL_FRAMES[color_index % GOL_FRAMES.len()];
        let color = colors::GRADIENT[color_index % colors::GRADIENT.len()];
        Span::styled(format!("{} ", frame), Style::default().fg(color))
    } else {
        Span::styled("●●● ", Style::default().fg(colors::SUCCESS))
    };

    vec![
        // margin-top: 1
        Line::from(""),
        Line::from(vec![icon, Span::raw(msg.summary.clone())]),
    ]
}

/// Render a tool result message.
/// margin-top: 0 (no extra spacing before)
pub fn render_tool_result(msg: &ToolResultMessage) -> Vec<Line<'static>> {
    let border_style = Style::default().fg(colors::MUTED);

    // Determine text style based on result type
    let text_style = if msg.error.is_some() {
        Style::default().fg(colors::ERROR)
    } else if msg.skipped {
        Style::default().fg(colors::WARNING)
    } else {
        Style::default()
    };

    // Get display text and split into lines
    let display_text = msg.summary();
    let content_lines: Vec<&str> = display_text.lines().collect();
    let line_count = content_lines.len().max(1);

    let mut lines = Vec::new();

    // Render with expanding border (⎢ for middle lines, ⎣ for last line)
    for (i, line) in content_lines.iter().enumerate() {
        let border = if i == line_count - 1 { "⎣" } else { "⎢" };

        lines.push(Line::from(vec![
            Span::styled(format!("  {} ", border), border_style),
            Span::styled(line.to_string(), text_style),
        ]));
    }

    // If no lines, show single bordered line
    if lines.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("  ⎣ ", border_style),
            Span::styled(display_text, text_style),
        ]));
    }

    lines
}

/// Render a system message with markdown support.
/// margin-top: 1 for visibility (system messages are important notifications)
pub fn render_system_message(msg: &SystemMessage) -> Vec<Line<'static>> {
    let text_style = match msg.kind {
        SystemMessageKind::Info => Style::default(),
        SystemMessageKind::Warning => Style::default().fg(colors::WARNING),
        SystemMessageKind::Error => Style::default().fg(colors::ERROR),
    };

    let border_style = Style::default().fg(colors::MUTED);
    let content_lines: Vec<&str> = msg.content.lines().collect();
    let line_count = content_lines.len().max(1);

    let mut lines = Vec::new();

    // margin-top: 1
    lines.push(Line::from(""));

    // Render with expanding border
    for (i, line) in content_lines.iter().enumerate() {
        let border = if i == line_count - 1 { "⎣" } else { "⎢" };

        lines.push(Line::from(vec![
            Span::styled(format!("  {} ", border), border_style),
            Span::styled(line.to_string(), text_style),
        ]));
    }

    lines
}

/// Render a user command message.
/// margin-top: 0
pub fn render_user_command_message(msg: &UserCommandMessage) -> Vec<Line<'static>> {
    let border_style = Style::default().fg(colors::MUTED);
    let content_lines: Vec<&str> = msg.content.lines().collect();
    let line_count = content_lines.len().max(1);

    let mut lines = Vec::new();

    for (i, line) in content_lines.iter().enumerate() {
        let border = if i == line_count - 1 { "⎣" } else { "⎢" };
        lines.push(Line::from(vec![
            Span::styled(format!("  {} ", border), border_style),
            Span::raw(line.to_string()),
        ]));
    }

    lines
}

/// Render an error message.
/// margin-top: 0
pub fn render_error_message(msg: &ErrorMessage) -> Vec<Line<'static>> {
    let border_style = Style::default().fg(colors::MUTED);
    let error_style = Style::default().fg(colors::ERROR).bold();
    let content_lines: Vec<&str> = msg.content.lines().collect();
    let line_count = content_lines.len().max(1);

    let mut lines = Vec::new();

    for (i, line) in content_lines.iter().enumerate() {
        let border = if i == line_count - 1 { "⎣" } else { "⎢" };
        lines.push(Line::from(vec![
            Span::styled(format!("  {border} "), border_style),
            Span::styled(line.to_string(), error_style),
        ]));
    }

    if lines.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("  ⎣ ", border_style),
            Span::styled(msg.content.clone(), error_style),
        ]));
    }

    lines
}

/// Render a warning message.
/// margin-top: 0
pub fn render_warning_message(msg: &WarningMessage) -> Vec<Line<'static>> {
    let border_style = Style::default().fg(colors::MUTED);
    let warning_style = Style::default().fg(colors::WARNING);
    let content_lines: Vec<&str> = msg.content.lines().collect();
    let line_count = content_lines.len().max(1);

    let mut lines = Vec::new();

    for (i, line) in content_lines.iter().enumerate() {
        let border = if i == line_count - 1 { "⎣" } else { "⎢" };
        lines.push(Line::from(vec![
            Span::styled(format!("  {} ", border), border_style),
            Span::styled(line.to_string(), warning_style),
        ]));
    }

    lines
}

/// Render an interrupt message.
pub fn render_interrupt() -> Vec<Line<'static>> {
    vec![Line::from(vec![
        Span::styled("  ⎣ ", Style::default().fg(colors::MUTED)),
        Span::styled(
            "Interrupted · What would you like Paramecia to do instead?",
            Style::default().fg(colors::WARNING),
        ),
    ])]
}

/// Render a bash output message with surface background.
/// In Python: margin-top: 1, background: $surface, padding: 1 2
pub fn render_bash_output(
    msg: &super::messages::BashOutputMessage,
    width: u16,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let surface_style = Style::default().bg(colors::SURFACE);
    let padding = "  "; // padding: 1 2 means 2 chars horizontal padding

    // margin-top: 1
    lines.push(Line::from(""));

    // Helper to pad line to full width
    let pad_line = |spans: Vec<Span<'static>>, content_len: usize| -> Line<'static> {
        let mut all_spans = vec![Span::styled(padding, surface_style)];
        all_spans.extend(spans);
        let remaining = (width as usize).saturating_sub(content_len + padding.len() * 2);
        all_spans.push(Span::styled(
            " ".repeat(remaining + padding.len()),
            surface_style,
        ));
        Line::from(all_spans)
    };

    // Top padding line
    lines.push(Line::styled(" ".repeat(width as usize), surface_style));

    // CWD line with exit status
    let exit_part = if msg.exit_code == 0 {
        Span::styled(
            "✓",
            Style::default().fg(colors::SUCCESS).bg(colors::SURFACE),
        )
    } else {
        Span::styled(
            format!("✗ ({})", msg.exit_code),
            Style::default().fg(colors::ERROR).bg(colors::SURFACE),
        )
    };

    let cwd_len = msg.cwd.len() + 1 + if msg.exit_code == 0 { 1 } else { 6 };
    lines.push(pad_line(
        vec![
            Span::styled(
                msg.cwd.clone(),
                Style::default().fg(colors::MUTED).bg(colors::SURFACE),
            ),
            Span::styled(" ", surface_style),
            exit_part,
        ],
        cwd_len,
    ));

    // Command line
    let cmd_len = 2 + msg.command.len();
    lines.push(pad_line(
        vec![
            Span::styled(
                "> ",
                Style::default()
                    .fg(colors::PRIMARY)
                    .bold()
                    .bg(colors::SURFACE),
            ),
            Span::styled(
                msg.command.clone(),
                Style::default().fg(colors::FOREGROUND).bg(colors::SURFACE),
            ),
        ],
        cmd_len,
    ));

    // Blank line before output (matching Python's margin-bottom: 1 on command line)
    lines.push(Line::styled(" ".repeat(width as usize), surface_style));

    // Output lines with surface background
    for line in msg.output.lines() {
        let line_len = line.len();
        lines.push(pad_line(
            vec![Span::styled(
                line.to_string(),
                Style::default().fg(colors::FOREGROUND).bg(colors::SURFACE),
            )],
            line_len,
        ));
    }

    // Bottom padding line
    lines.push(Line::styled(" ".repeat(width as usize), surface_style));

    lines
}

/// Render a compact message.
/// margin-top: 1
pub fn render_compact(
    msg: &CompactMessage,
    _spinner: &Spinner,
    color_index: usize,
) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from("")]; // margin-top: 1

    if msg.in_progress {
        // Use Game of Life cell pattern for compact spinner
        let frame = GOL_FRAMES[color_index % GOL_FRAMES.len()];
        let color = colors::GRADIENT[color_index % colors::GRADIENT.len()];
        lines.push(Line::from(vec![
            Span::styled(format!("{} ", frame), Style::default().fg(color)),
            Span::raw("Compacting conversation history..."),
        ]));
    } else if let Some(error) = &msg.error {
        lines.push(Line::from(vec![
            Span::styled("○○○ ", Style::default().fg(colors::ERROR)),
            Span::styled(
                format!("Error: {error}"),
                Style::default().fg(colors::ERROR),
            ),
        ]));
    } else {
        let old = msg.old_tokens.unwrap_or(0);
        let new = msg.new_tokens.unwrap_or(0);
        lines.push(Line::from(vec![
            Span::styled("●●● ", Style::default().fg(colors::SUCCESS)),
            Span::styled(
                format!("Compacted: {} → {} tokens", old, new),
                Style::default().fg(colors::SUCCESS),
            ),
        ]));
    }

    lines
}

/// Render the loading indicator.
///
/// The animation works by having a "wave" that propagates through the text,
/// changing each character from the current color to the next color in sequence.
pub fn render_loading(
    elapsed_secs: u64,
    status: &str,
    spinner: &Spinner,
    transition_progress: usize,
) -> Line<'static> {
    let mut spans = Vec::new();

    // Total elements: spinner + each char of status + ellipsis + space
    let total_elements = 1 + status.chars().count() + 2;

    // Current and next color indices in the gradient
    let current_color_index = (transition_progress / total_elements) % colors::GRADIENT.len();
    let next_color_index = (current_color_index + 1) % colors::GRADIENT.len();

    // Position of the wave front (which character is currently transitioning)
    let wave_position = transition_progress % total_elements;

    // Get color for a position - characters behind the wave are the next color,
    // characters at or ahead of the wave are the current color
    let get_color = |position: usize| -> Color {
        if position < wave_position {
            colors::GRADIENT[next_color_index]
        } else {
            colors::GRADIENT[current_color_index]
        }
    };

    // Spinner character with wave color
    let spinner_color = get_color(0);
    spans.push(Span::styled(
        format!("{} ", spinner.current_frame()),
        Style::default().fg(spinner_color),
    ));

    // Each character of status with its own wave color
    for (i, c) in status.chars().enumerate() {
        let color = get_color(1 + i);
        spans.push(Span::styled(c.to_string(), Style::default().fg(color)));
    }

    // Ellipsis with wave color
    let ellipsis_pos = 1 + status.chars().count();
    let ellipsis_color = get_color(ellipsis_pos);
    spans.push(Span::styled("…", Style::default().fg(ellipsis_color)));

    // Space after ellipsis
    let space_color = get_color(ellipsis_pos + 1);
    spans.push(Span::styled(" ", Style::default().fg(space_color)));

    // Hint text (neutral color, not part of wave)
    spans.push(Span::styled(
        format!("({}s esc to interrupt)", elapsed_secs),
        Style::default().fg(colors::MUTED),
    ));

    Line::from(spans)
}

/// Render the input box.
pub fn render_input_box<'a>(
    content: &'a str,
    _cursor: usize,
    mode: AgentMode,
    multiline: bool,
) -> Paragraph<'a> {
    let border_color = colors::mode_border_color(&mode);

    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(border_color))
        .border_type(ratatui::widgets::BorderType::Plain) // Use plain borders like original
        .padding(Padding::new(1, 1, 0, 0));

    // Build input content with proper multiline handling
    let prompt = Span::styled("> ", Style::default().fg(colors::YELLOW).bold());

    // Show multiline indicator
    let multiline_indicator = if multiline {
        Span::styled("📝 ", Style::default().fg(colors::ACCENT))
    } else {
        Span::raw("")
    };

    // Show placeholder when empty
    let content_span = if content.is_empty() {
        Span::styled(
            "Ask anything...",
            Style::default().fg(colors::FOREGROUND_MUTED).italic(),
        )
    } else {
        Span::raw(content)
    };

    // For multiline mode, we need to handle each line separately
    if multiline && !content.is_empty() {
        let lines: Vec<Line> = content
            .lines()
            .enumerate()
            .map(|(i, line)| {
                if i == 0 {
                    // First line gets the full prompt
                    Line::from(vec![
                        multiline_indicator.clone(),
                        prompt.clone(),
                        Span::raw(line),
                    ])
                } else {
                    // Subsequent lines get indentation to match prompt width
                    Line::from(vec![Span::raw("  "), Span::raw(line)])
                }
            })
            .collect();

        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(colors::FOREGROUND))
    } else {
        // Single line mode
        let text = Line::from(vec![multiline_indicator, prompt, content_span]);
        Paragraph::new(text)
            .block(block)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(colors::FOREGROUND))
    }
}

/// Render the approval dialog.
pub fn render_approval_dialog(
    frame: &mut Frame,
    area: Rect,
    tool_name: &str,
    args: &serde_json::Value,
    selected: usize,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(colors::WARNING))
        .border_type(ratatui::widgets::BorderType::Rounded);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines = vec![
        // Title
        Line::from(vec![Span::styled(
            format!("⚠ {} command", tool_name),
            Style::default().fg(colors::WARNING).bold(),
        )]),
        Line::from(""),
    ];

    // Show args with special handling for different tool types
    if let Some(obj) = args.as_object() {
        for (key, value) in obj {
            let value_str = match value {
                serde_json::Value::String(s) => {
                    // Truncate long strings (safely handles UTF-8)
                    truncate_string(s, 60)
                }
                _ => value.to_string(),
            };

            // Special styling for command/path fields
            let value_style = if key == "command" || key == "path" || key == "file_path" {
                Style::default().fg(colors::PRIMARY)
            } else {
                Style::default()
            };

            lines.push(Line::from(vec![
                Span::styled(format!("{}: ", key), Style::default().fg(colors::MUTED)),
                Span::styled(value_str, value_style),
            ]));
        }
    }

    lines.push(Line::from(""));

    // Options with keyboard shortcuts
    let options = [
        ("Yes", "yes", "y"),
        (
            &format!("Yes and always allow {tool_name} for this session"),
            "yes",
            "",
        ),
        ("No and tell the agent what to do instead", "no", "n"),
    ];

    for (i, (text, color_type, shortcut)) in options.iter().enumerate() {
        let is_selected = i == selected;
        let cursor = if is_selected { "› " } else { "  " };

        // Style: yes options are green, no options are red
        let style = if is_selected {
            if *color_type == "no" {
                Style::default().fg(colors::ERROR).bold()
            } else {
                Style::default().fg(colors::SUCCESS).bold()
            }
        } else if *color_type == "no" {
            Style::default().fg(colors::ERROR)
        } else {
            Style::default().fg(colors::SUCCESS)
        };

        let mut spans = vec![
            Span::styled(cursor, style),
            Span::styled(format!("{}. {}", i + 1, text), style),
        ];

        // Show keyboard shortcut hint for first and last option
        if !shortcut.is_empty() {
            spans.push(Span::styled(
                format!(" ({})", shortcut),
                Style::default().fg(colors::MUTED),
            ));
        }

        lines.push(Line::from(spans));
    }

    lines.push(Line::from(""));

    // Enhanced help line with keyboard shortcuts
    lines.push(Line::from(vec![
        Span::styled("↑↓ navigate  ", Style::default().fg(colors::MUTED)),
        Span::styled("Enter", Style::default().fg(colors::TEXT)),
        Span::styled(" select  ", Style::default().fg(colors::MUTED)),
        Span::styled("y/n", Style::default().fg(colors::TEXT)),
        Span::styled(" quick  ", Style::default().fg(colors::MUTED)),
        Span::styled("ESC", Style::default().fg(colors::TEXT)),
        Span::styled(" reject", Style::default().fg(colors::MUTED)),
    ]));

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    frame.render_widget(paragraph, inner);
}

/// Render the completion popup.
pub fn render_completion_popup(
    frame: &mut Frame,
    area: Rect,
    suggestions: &[super::completion::CompletionSuggestion],
    selected_index: usize,
) {
    if suggestions.is_empty() {
        return;
    }

    // Calculate popup height (1 line per suggestion + 2 for padding)
    let height = (suggestions.len() as u16 + 2).min(12);

    // Position popup above the input area, ensuring it doesn't go off-screen
    let popup_y = if area.y >= height {
        area.y.saturating_sub(height) // Position above input
    } else {
        area.y + area.height // Position below input if not enough space above
    };

    // Calculate width based on longest suggestion
    let max_suggestion_width = suggestions
        .iter()
        .map(|s| {
            s.completion.len()
                + if s.description.is_empty() {
                    0
                } else {
                    s.description.len() + 4
                }
        })
        .max()
        .unwrap_or(0)
        .min(60);

    let popup_area = Rect {
        x: area.x + 2, // Align with input content (after "> ")
        y: popup_y,
        width: max_suggestion_width as u16,
        height,
    };

    // Clear the area with background
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(colors::MUTED))
        .border_type(ratatui::widgets::BorderType::Rounded)
        .style(Style::default().bg(colors::SURFACE))
        .padding(Padding::new(1, 1, 1, 1));

    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    // Render suggestions
    let mut lines = Vec::new();
    for (i, suggestion) in suggestions.iter().enumerate() {
        let is_selected = i == selected_index;

        let label_style = if is_selected {
            Style::default()
                .fg(colors::PRIMARY)
                .bold()
                .bg(colors::BACKGROUND)
        } else {
            Style::default().fg(colors::TEXT).bold()
        };

        let desc_style = if is_selected {
            Style::default()
                .fg(colors::MUTED)
                .italic()
                .bg(colors::BACKGROUND)
        } else {
            Style::default().fg(colors::MUTED)
        };

        let mut spans = vec![Span::styled(suggestion.completion.clone(), label_style)];
        if !suggestion.description.is_empty() {
            spans.push(Span::styled("  ", desc_style));
            spans.push(Span::styled(suggestion.description.clone(), desc_style));
        }

        lines.push(Line::from(spans));
    }

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}
