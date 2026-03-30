use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use paramecia_model::vis::pretty_shape;
use tracing::field::{Field, Visit};
use tracing::{Id, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

#[derive(Clone, Default)]
pub struct TensorOpAggregationLayer {
    shared: Arc<Mutex<Shared>>,
}

pub struct TensorOpAggregationGuard {
    shared: Arc<Mutex<Shared>>,
    top_k: usize,
    emit_summary_on_drop: bool,
}

#[derive(Clone)]
pub struct TensorOpAggregationHandle {
    shared: Arc<Mutex<Shared>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorOpSnapshotRow {
    pub key: String,
    pub node_stable_id: Option<u64>,
    pub count: u64,
    pub total_ns: u128,
    pub avg_ns: u128,
    pub max_ns: u128,
}

#[derive(Default)]
struct Shared {
    stats: HashMap<Arc<str>, OpStats>,
}

#[derive(Default)]
struct OpStats {
    node_stable_id: Option<u64>,
    count: u64,
    total_ns: u128,
    max_ns: u128,
}

struct SpanTiming {
    started_at: Instant,
    key: Arc<str>,
    node_stable_id: Option<u64>,
}

thread_local! {
    static RUN_SCOPE_DEPTH: Cell<u32> = const { Cell::new(0) };
    static RUN_TENSOR_OP_SEQ: Cell<u64> = const { Cell::new(0) };
}

#[derive(Default)]
struct TensorShapeVisitor {
    fields: Vec<(String, String)>,
    arrow_node_stable_id: Option<u64>,
    arrow_node_id: Option<u64>,
}

impl TensorShapeVisitor {
    fn has_tensor_shape(&self) -> bool {
        !self.fields.is_empty()
    }
}

impl Visit for TensorShapeVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if is_tracked_shape_field(field.name()) {
            self.fields.push((
                field.name().to_string(),
                normalize_shape_field_value(value.to_string()),
            ));
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if is_tracked_shape_field(field.name()) {
            self.fields.push((
                field.name().to_string(),
                normalize_shape_field_value(format!("{value:?}")),
            ));
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "arrow_node_stable_id" => self.arrow_node_stable_id = Some(value),
            "arrow_node_id" => self.arrow_node_id = Some(value),
            _ => {}
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if value < 0 {
            return;
        }
        let Ok(value_u64) = u64::try_from(value) else {
            return;
        };
        self.record_u64(field, value_u64);
    }
}

fn is_tracked_shape_field(name: &str) -> bool {
    matches!(name, "input_shape" | "output_shape")
        || name.starts_with("input_shape_")
        || name.starts_with("output_shape_")
}

fn build_key(
    op_name: &str,
    mut shape_fields: Vec<(String, String)>,
    node_seq: Option<u64>,
) -> Arc<str> {
    shape_fields.sort_by(|left, right| left.0.cmp(&right.0));
    let mut out = String::from(op_name);
    if !shape_fields.is_empty() {
        out.push(' ');
        for (idx, (_, value)) in shape_fields.iter().enumerate() {
            if idx > 0 {
                out.push_str(" x ");
            }
            out.push_str(value);
        }
    }
    if let Some(seq) = node_seq {
        let _ = write!(&mut out, " [node_seq={seq}]");
    }
    Arc::<str>::from(out)
}

fn build_arrow_node_key(node_stable_id: Option<u64>) -> Arc<str> {
    match node_stable_id {
        Some(stable_id) => Arc::<str>::from(format!("arrow_node [node_stable_id={stable_id}]")),
        None => Arc::<str>::from("arrow_node"),
    }
}

fn normalize_shape_field_value(mut raw: String) -> String {
    raw = raw.trim().to_string();
    if raw.starts_with('"') && raw.ends_with('"') && raw.len() >= 2 {
        raw = raw[1..raw.len() - 1].to_string();
    }

    if looks_like_shape_type_name(&raw) {
        let pretty = pretty_shape(&raw);
        if !pretty.is_empty() && pretty != "[]" {
            return pretty;
        }
    }
    raw
}

fn looks_like_shape_type_name(raw: &str) -> bool {
    raw.contains("DIM<")
        || raw.contains("RANK<")
        || raw.contains("glowstick::")
        || raw.contains("Dyn<")
}

fn is_run_root_span(name: &str) -> bool {
    matches!(name, "model" | "qwen3-next")
}

fn next_node_seq_if_in_run_scope() -> Option<u64> {
    RUN_SCOPE_DEPTH.with(|depth| {
        if depth.get() == 0 {
            return None;
        }
        RUN_TENSOR_OP_SEQ.with(|seq| {
            let next = seq.get().saturating_add(1);
            seq.set(next);
            Some(next)
        })
    })
}

fn enter_run_scope_if_root(name: &str) {
    if !is_run_root_span(name) {
        return;
    }

    RUN_SCOPE_DEPTH.with(|depth| {
        let current = depth.get();
        if current == 0 {
            RUN_TENSOR_OP_SEQ.with(|seq| seq.set(0));
        }
        depth.set(current.saturating_add(1));
    });
}

fn exit_run_scope_if_root(name: &str) {
    if !is_run_root_span(name) {
        return;
    }

    RUN_SCOPE_DEPTH.with(|depth| {
        let current = depth.get();
        if current > 0 {
            depth.set(current - 1);
        }
    });
}

impl<S> Layer<S> for TensorOpAggregationLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = TensorShapeVisitor::default();
        attrs.record(&mut visitor);

        if attrs.metadata().name() == "arrow_node" {
            let stable_id = visitor
                .arrow_node_stable_id
                .or(visitor.arrow_node_id)
                .filter(|value| *value > 0);
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(SpanTiming {
                    started_at: Instant::now(),
                    key: build_arrow_node_key(stable_id),
                    node_stable_id: stable_id,
                });
            }
            return;
        }

        if !visitor.has_tensor_shape() {
            return;
        }

        let node_seq = next_node_seq_if_in_run_scope();
        let key = build_key(attrs.metadata().name(), visitor.fields, node_seq);
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanTiming {
                started_at: Instant::now(),
                key,
                node_stable_id: None,
            });
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        enter_run_scope_if_root(span.metadata().name());
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        exit_run_scope_if_root(span.metadata().name());
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        let Some(timing) = extensions.remove::<SpanTiming>() else {
            return;
        };

        let elapsed_ns = timing.started_at.elapsed().as_nanos();
        let Ok(mut shared) = self.shared.lock() else {
            return;
        };
        let stats = shared.stats.entry(timing.key).or_default();
        if stats.node_stable_id.is_none() {
            stats.node_stable_id = timing.node_stable_id;
        }
        stats.count = stats.count.saturating_add(1);
        stats.total_ns = stats.total_ns.saturating_add(elapsed_ns);
        if elapsed_ns > stats.max_ns {
            stats.max_ns = elapsed_ns;
        }
    }
}

impl Drop for TensorOpAggregationGuard {
    fn drop(&mut self) {
        if !self.emit_summary_on_drop {
            return;
        }
        let top_rows = self
            .handle()
            .snapshot_top(self.top_k.max(1))
            .into_iter()
            .filter(|row| row.node_stable_id.is_none())
            .collect::<Vec<_>>();
        if top_rows.is_empty() {
            return;
        }

        let mut out = String::new();
        let _ = writeln!(&mut out, "\n=== Top Tensor Ops (by total time) ===");
        let _ = writeln!(
            &mut out,
            "{:<4} {:<12} {:<12} {:<12} Op",
            "Rank", "Total", "Avg", "Max"
        );
        for (idx, row) in top_rows.iter().enumerate() {
            let _ = writeln!(
                &mut out,
                "{:<4} {:<12} {:<12} {:<12} {}",
                idx + 1,
                fmt_duration(row.total_ns),
                fmt_duration(row.avg_ns),
                fmt_duration(row.max_ns),
                row.key
            );
        }

        eprintln!("{out}");
    }
}

fn fmt_duration(ns: u128) -> String {
    if ns >= 1_000_000 {
        format!("{:.3}ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.3}us", ns as f64 / 1_000.0)
    } else {
        format!("{ns}ns")
    }
}

pub fn tensor_op_aggregation(top_k: usize) -> (TensorOpAggregationLayer, TensorOpAggregationGuard) {
    tensor_op_aggregation_with_options(top_k, true)
}

pub fn tensor_op_aggregation_with_options(
    top_k: usize,
    emit_summary_on_drop: bool,
) -> (TensorOpAggregationLayer, TensorOpAggregationGuard) {
    let shared = Arc::new(Mutex::new(Shared::default()));
    (
        TensorOpAggregationLayer {
            shared: Arc::clone(&shared),
        },
        TensorOpAggregationGuard {
            shared,
            top_k,
            emit_summary_on_drop,
        },
    )
}

impl TensorOpAggregationGuard {
    pub fn handle(&self) -> TensorOpAggregationHandle {
        TensorOpAggregationHandle {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl TensorOpAggregationHandle {
    pub fn snapshot_top(&self, top_k: usize) -> Vec<TensorOpSnapshotRow> {
        let Ok(shared) = self.shared.lock() else {
            return Vec::new();
        };

        if shared.stats.is_empty() {
            return Vec::new();
        }

        let mut rows = shared
            .stats
            .iter()
            .map(|(key, stats)| {
                let avg_ns = if stats.count == 0 {
                    0
                } else {
                    stats.total_ns.saturating_div(u128::from(stats.count))
                };
                TensorOpSnapshotRow {
                    key: key.to_string(),
                    node_stable_id: stats.node_stable_id,
                    count: stats.count,
                    total_ns: stats.total_ns,
                    avg_ns,
                    max_ns: stats.max_ns,
                }
            })
            .collect::<Vec<_>>();

        rows.sort_by(|left, right| right.total_ns.cmp(&left.total_ns));
        rows.truncate(top_k.max(1));
        rows
    }
}
