use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use inception::*;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
static NEXT_SUBGRAPH_ID: AtomicUsize = AtomicUsize::new(0);
const MIN_DEDUP_SUBGRAPH_NODES: usize = 4;

fn next_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_subgraph_id() -> usize {
    NEXT_SUBGRAPH_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub stable_id: u64,
    pub label: String,
    pub is_custom: bool,
}

#[derive(Clone, Debug)]
pub struct SubGraph {
    pub id: usize,
    pub name: String,
    pub node_ids: Vec<usize>,
    pub children: Vec<SubGraph>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    /// Is a custom impl which needs to be eliminated,
    pub is_custom: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<(usize, usize, Option<String>)>,
    pub subgraphs: Vec<SubGraph>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    /// Labels for outgoing edges, parallel to `outputs`.
    pub output_labels: Vec<Option<String>>,
    /// Is a custom impl which needs to be eliminated,
    pub is_custom: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DotMode {
    InlineAll,
    DedupCompositeRepeated,
}

impl Graph {
    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
            && self.edges.is_empty()
            && self.subgraphs.is_empty()
            && self.inputs.is_empty()
            && self.outputs.is_empty()
            && self.output_labels.is_empty()
    }

    fn assign_stable_node_ids(&mut self) {
        let mut node_ids = self.nodes.iter().map(|node| node.id).collect::<Vec<_>>();
        node_ids.sort_unstable();
        let stable_id_by_node_id = node_ids
            .into_iter()
            .enumerate()
            .filter_map(|(idx, node_id)| {
                let stable_id = u64::try_from(idx.saturating_add(1)).ok()?;
                Some((node_id, stable_id))
            })
            .collect::<HashMap<_, _>>();

        for node in &mut self.nodes {
            node.stable_id = stable_id_by_node_id.get(&node.id).copied().unwrap_or(0);
        }
    }

    pub fn with_stable_node_ids(mut self) -> Self {
        self.assign_stable_node_ids();
        self
    }

    /// Create a graph with a single leaf node.
    pub fn leaf(label: &str) -> Self {
        let id = next_id();
        Graph {
            nodes: vec![Node {
                id,
                stable_id: 1,
                label: label.to_string(),
                is_custom: false,
            }],
            edges: vec![],
            subgraphs: vec![],
            inputs: vec![id],
            outputs: vec![id],
            output_labels: vec![None],
            is_custom: false,
        }
    }

    pub fn custom_leaf(label: &str) -> Self {
        let id = next_id();
        Graph {
            nodes: vec![Node {
                id,
                stable_id: 1,
                label: label.to_string(),
                is_custom: true,
            }],
            edges: vec![],
            subgraphs: vec![],
            inputs: vec![id],
            outputs: vec![id],
            output_labels: vec![None],
            is_custom: false,
        }
    }

    /// Create a custom leaf node with a label on its outgoing edge.
    pub fn custom_leaf_with_edges(label: &str, output_label: Option<&str>) -> Self {
        let id = next_id();
        Graph {
            nodes: vec![Node {
                id,
                stable_id: 1,
                label: label.to_string(),
                is_custom: true,
            }],
            edges: vec![],
            subgraphs: vec![],
            inputs: vec![id],
            outputs: vec![id],
            output_labels: vec![output_label.map(|s| s.to_string())],
            is_custom: false,
        }
    }

    /// Create a leaf node with a label on its outgoing edge.
    pub fn leaf_with_edges(label: &str, output_label: Option<&str>) -> Self {
        let id = next_id();
        Graph {
            nodes: vec![Node {
                id,
                stable_id: 1,
                label: label.to_string(),
                is_custom: false,
            }],
            edges: vec![],
            subgraphs: vec![],
            inputs: vec![id],
            outputs: vec![id],
            output_labels: vec![output_label.map(|s| s.to_string())],
            is_custom: false,
        }
    }

    /// Override all outgoing edge labels with the same value.
    pub fn with_output_label(mut self, output_label: Option<&str>) -> Self {
        self.output_labels = self
            .outputs
            .iter()
            .map(|_| output_label.map(|s| s.to_string()))
            .collect();
        self
    }

    /// Override outgoing edge labels using a pretty-printed output type.
    pub fn with_output_type<T>(self) -> Self {
        self.with_output_label(Some(&pretty_type(std::any::type_name::<T>())))
    }

    /// Connect two graphs in sequence: left's outputs -> right's inputs.
    pub fn sequence(mut left: Graph, right: Graph) -> Graph {
        // Identity: empty graph should behave like a no-op in composition.
        if left.is_empty() {
            return right.with_stable_node_ids();
        }
        if right.is_empty() {
            return left.with_stable_node_ids();
        }

        // Connect every output of left to every input of right
        for (out_idx, &out) in left.outputs.iter().enumerate() {
            let label = left.output_labels.get(out_idx).cloned().flatten();
            for &inp in &right.inputs {
                left.edges.push((out, inp, label.clone()));
            }
        }
        // Merge right into left
        left.nodes.extend(right.nodes);
        left.edges.extend(right.edges);
        left.subgraphs.extend(right.subgraphs);
        // New outputs are right's outputs; inputs stay as left's
        left.outputs = right.outputs;
        left.output_labels = right.output_labels;
        left.with_stable_node_ids()
    }

    /// Place multiple graphs side by side without any subgraph wrapping.
    pub fn parallel(branches: Vec<Graph>) -> Graph {
        let mut all_nodes = Vec::new();
        let mut all_edges = Vec::new();
        let mut all_subgraphs = Vec::new();
        let mut all_inputs = Vec::new();
        let mut all_outputs = Vec::new();
        let mut all_output_labels = Vec::new();

        for branch in branches {
            all_inputs.extend(&branch.inputs);
            all_outputs.extend(&branch.outputs);
            all_output_labels.extend(branch.output_labels);
            all_nodes.extend(branch.nodes);
            all_edges.extend(branch.edges);
            all_subgraphs.extend(branch.subgraphs);
        }

        Graph {
            nodes: all_nodes,
            edges: all_edges,
            subgraphs: all_subgraphs,
            inputs: all_inputs,
            outputs: all_outputs,
            output_labels: all_output_labels,
            is_custom: false,
        }
        .with_stable_node_ids()
    }

    pub fn parallel_custom(branches: Vec<Graph>) -> Graph {
        let mut g = Self::parallel(branches);
        g.is_custom = true;
        g
    }

    /// Wrap an inner graph in a named subgraph (cluster).
    pub fn wrap_subgraph(name: &str, inner: Graph) -> Graph {
        let all_node_ids: Vec<usize> = inner.nodes.iter().map(|n| n.id).collect();
        let sub = SubGraph {
            id: next_subgraph_id(),
            name: name.to_string(),
            node_ids: all_node_ids,
            children: inner.subgraphs,
            inputs: inner.inputs.clone(),
            outputs: inner.outputs.clone(),
            is_custom: false,
        };
        Graph {
            nodes: inner.nodes,
            edges: inner.edges,
            subgraphs: vec![sub],
            inputs: inner.inputs,
            outputs: inner.outputs,
            output_labels: inner.output_labels,
            is_custom: false,
        }
        .with_stable_node_ids()
    }

    /// Wrap an inner graph in a named subgraph (cluster).
    pub fn wrap_custom_subgraph(_name: &str, inner: Graph) -> Graph {
        inner.with_stable_node_ids()
        //let all_node_ids: Vec<usize> = inner.nodes.iter().map(|n| n.id).collect();
        //let sub = SubGraph {
        //    id: next_subgraph_id(),
        //    name: name.to_string(),
        //    node_ids: all_node_ids,
        //    children: inner.subgraphs,
        //    inputs: inner.inputs.clone(),
        //    outputs: inner.outputs.clone(),
        //    is_custom: true,
        //};
        //Graph {
        //    nodes: inner.nodes,
        //    edges: inner.edges,
        //    subgraphs: vec![sub],
        //    inputs: inner.inputs,
        //    outputs: inner.outputs,
        //    output_labels: inner.output_labels,
        //    is_custom: false,
        //}
    }

    /// Place multiple named branches side by side in a subgraph.
    pub fn zip(name: &str, branches: Vec<(&str, Graph)>) -> Graph {
        let mut all_nodes = Vec::new();
        let mut all_edges = Vec::new();
        let mut child_subgraphs = Vec::new();
        let mut all_node_ids = Vec::new();
        let mut all_inputs = Vec::new();
        let mut all_outputs = Vec::new();
        let mut all_output_labels = Vec::new();

        for (branch_name, branch) in branches {
            let branch_node_ids: Vec<usize> = branch.nodes.iter().map(|n| n.id).collect();
            all_node_ids.extend(&branch_node_ids);
            all_inputs.extend(&branch.inputs);
            all_outputs.extend(&branch.outputs);
            all_output_labels.extend(branch.output_labels);

            let child = SubGraph {
                id: next_subgraph_id(),
                name: branch_name.to_string(),
                node_ids: branch_node_ids,
                children: branch.subgraphs,
                inputs: branch.inputs,
                outputs: branch.outputs,
                is_custom: branch.is_custom,
            };
            child_subgraphs.push(child);

            all_nodes.extend(branch.nodes);
            all_edges.extend(branch.edges);
        }

        let sub = SubGraph {
            id: next_subgraph_id(),
            name: name.to_string(),
            node_ids: all_node_ids,
            children: child_subgraphs,
            inputs: all_inputs.clone(),
            outputs: all_outputs.clone(),
            is_custom: false,
        };

        Graph {
            nodes: all_nodes,
            edges: all_edges,
            subgraphs: vec![sub],
            inputs: all_inputs,
            outputs: all_outputs,
            output_labels: all_output_labels,
            is_custom: false,
        }
        .with_stable_node_ids()
    }

    pub fn zip_custom(name: &str, branches: Vec<(&str, Graph)>) -> Graph {
        let mut g = Self::zip(name, branches);
        g.is_custom = true;
        g
    }

    /// Render to graphviz DOT format.
    pub fn to_dot(&self) -> String {
        self.to_dot_mode(DotMode::DedupCompositeRepeated)
    }

    pub fn to_dot_inline(&self) -> String {
        self.to_dot_mode(DotMode::InlineAll)
    }

    pub fn to_dot_mode(&self, mode: DotMode) -> String {
        let plan = match mode {
            DotMode::InlineAll => RenderPlan::inline(self),
            DotMode::DedupCompositeRepeated => RenderPlan::dedup_composite_repeated(self),
        };
        render_dot(self, &plan)
    }
}

#[derive(Clone, Debug)]
struct CollapsedInstance {
    summary_node_id: usize,
    summary_label: String,
    is_custom: bool,
    inputs: HashSet<usize>,
    outputs: HashSet<usize>,
}

#[derive(Clone, Debug, Default)]
struct RenderPlan {
    nodes: Vec<Node>,
    edges: Vec<(usize, usize, Option<String>)>,
    emitted_node_ids: HashSet<usize>,
    collapsed: HashMap<usize, CollapsedInstance>,
}

impl RenderPlan {
    fn inline(graph: &Graph) -> Self {
        let emitted_node_ids = graph.nodes.iter().map(|n| n.id).collect();
        Self {
            nodes: graph.nodes.clone(),
            edges: graph.edges.clone(),
            emitted_node_ids,
            collapsed: HashMap::new(),
        }
    }

    fn dedup_composite_repeated(graph: &Graph) -> Self {
        let flattened = flatten_subgraphs(&graph.subgraphs);
        if flattened.is_empty() {
            return Self::inline(graph);
        }

        let node_labels: HashMap<usize, &str> = graph
            .nodes
            .iter()
            .map(|n| (n.id, n.label.as_str()))
            .collect();

        let mut signatures = HashMap::new();
        for sg in &flattened {
            let _ = signature_for_subgraph(sg, &node_labels, &mut signatures);
        }

        let mut sig_counts: HashMap<String, usize> = HashMap::new();
        for sg in &flattened {
            if let Some(sig) = signatures.get(&sg.id) {
                *sig_counts.entry(sig.clone()).or_insert(0) += 1;
            }
        }

        let mut parent_of: HashMap<usize, usize> = HashMap::new();
        for sg in &flattened {
            for child in &sg.children {
                parent_of.insert(child.id, sg.id);
            }
        }

        let mut first_seen: HashMap<String, usize> = HashMap::new();
        let mut collapse_candidates = HashSet::new();
        for sg in &flattened {
            let Some(sig) = signatures.get(&sg.id) else {
                continue;
            };
            let count = *sig_counts.get(sig).unwrap_or(&0);
            if count <= 1 || !dedup_eligible(sg) {
                continue;
            }
            if first_seen.contains_key(sig) {
                collapse_candidates.insert(sg.id);
            } else {
                first_seen.insert(sig.clone(), sg.id);
            }
        }

        let mut top_level_collapsed = HashSet::new();
        for &sg_id in &collapse_candidates {
            let mut cur = sg_id;
            let mut has_collapsed_ancestor = false;
            while let Some(parent) = parent_of.get(&cur).copied() {
                if collapse_candidates.contains(&parent) {
                    has_collapsed_ancestor = true;
                    break;
                }
                cur = parent;
            }
            if !has_collapsed_ancestor {
                top_level_collapsed.insert(sg_id);
            }
        }

        if top_level_collapsed.is_empty() {
            return Self::inline(graph);
        }

        let mut collapsed = HashMap::new();
        let mut node_to_collapsed = HashMap::new();
        let mut removed_node_ids = HashSet::new();
        let mut next_node_id = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;

        for sg in &flattened {
            if !top_level_collapsed.contains(&sg.id) {
                continue;
            }
            let sig = signatures
                .get(&sg.id)
                .cloned()
                .unwrap_or_else(|| sg.name.clone());
            let count = *sig_counts.get(&sig).unwrap_or(&1);
            let summary_node_id = next_node_id;
            next_node_id += 1;
            let summary_label = format!("{} (shared x{})", sg.name, count);
            let inputs: HashSet<usize> = sg.inputs.iter().copied().collect();
            let outputs: HashSet<usize> = sg.outputs.iter().copied().collect();
            for &nid in &sg.node_ids {
                node_to_collapsed.insert(nid, sg.id);
                removed_node_ids.insert(nid);
            }
            collapsed.insert(
                sg.id,
                CollapsedInstance {
                    summary_node_id,
                    summary_label,
                    is_custom: sg.is_custom,
                    inputs,
                    outputs,
                },
            );
        }

        let mut nodes = Vec::new();
        for n in &graph.nodes {
            if !removed_node_ids.contains(&n.id) {
                nodes.push(n.clone());
            }
        }
        for ci in collapsed.values() {
            nodes.push(Node {
                id: ci.summary_node_id,
                stable_id: 0,
                label: ci.summary_label.clone(),
                is_custom: ci.is_custom,
            });
        }

        let mut edge_set: HashSet<(usize, usize, Option<String>)> = HashSet::new();
        let mut edges = Vec::new();
        for (from, to, label) in &graph.edges {
            let from_owner = node_to_collapsed.get(from).copied();
            let to_owner = node_to_collapsed.get(to).copied();
            let rewritten = match (from_owner, to_owner) {
                (None, None) => Some((*from, *to)),
                (Some(c), Some(d)) if c == d => None,
                (Some(c), Some(d)) => {
                    let Some(c_info) = collapsed.get(&c) else {
                        continue;
                    };
                    let Some(d_info) = collapsed.get(&d) else {
                        continue;
                    };
                    if c_info.outputs.contains(from) && d_info.inputs.contains(to) {
                        Some((c_info.summary_node_id, d_info.summary_node_id))
                    } else {
                        None
                    }
                }
                (Some(c), None) => {
                    let Some(c_info) = collapsed.get(&c) else {
                        continue;
                    };
                    if c_info.outputs.contains(from) {
                        Some((c_info.summary_node_id, *to))
                    } else {
                        None
                    }
                }
                (None, Some(c)) => {
                    let Some(c_info) = collapsed.get(&c) else {
                        continue;
                    };
                    if c_info.inputs.contains(to) {
                        Some((*from, c_info.summary_node_id))
                    } else {
                        None
                    }
                }
            };
            if let Some((rf, rt)) = rewritten {
                let key = (rf, rt, label.clone());
                if edge_set.insert(key.clone()) {
                    edges.push(key);
                }
            }
        }

        let emitted_node_ids = nodes.iter().map(|n| n.id).collect();
        Self {
            nodes,
            edges,
            emitted_node_ids,
            collapsed,
        }
    }
}

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('<', "\\<")
        .replace('>', "\\>")
}

fn render_dot(graph: &Graph, plan: &RenderPlan) -> String {
    let mut out = String::new();
    out.push_str("digraph {\n");
    out.push_str("    rankdir=TB;\n");
    if graph.is_custom {
        out.push_str("    color=\"red\";\n");
    } else {
        out.push_str("    color=\"black\";\n");
    }
    out.push_str("    node [shape=box, style=rounded];\n");

    for node in &plan.nodes {
        out.push_str(&format!(
            "    n{} [label=\"{}\"{}];\n",
            node.id,
            dot_escape(&node.label),
            if node.is_custom { ",color=\"red\"" } else { "" }
        ));
    }

    for sg in &graph.subgraphs {
        if let Some(ci) = plan.collapsed.get(&sg.id) {
            out.push_str(&format!("    n{};\n", ci.summary_node_id));
        } else {
            write_subgraph(&mut out, sg, 1, plan);
        }
    }

    for (from, to, label) in &plan.edges {
        if let Some(l) = label {
            out.push_str(&format!(
                "    n{} -> n{} [label=\"{}\"];\n",
                from,
                to,
                dot_escape(l)
            ));
        } else {
            out.push_str(&format!("    n{} -> n{};\n", from, to));
        }
    }

    out.push_str("}\n");
    out
}

fn flatten_subgraphs(subgraphs: &[SubGraph]) -> Vec<&SubGraph> {
    fn walk<'a>(sg: &'a SubGraph, out: &mut Vec<&'a SubGraph>) {
        out.push(sg);
        for child in &sg.children {
            walk(child, out);
        }
    }

    let mut out = Vec::new();
    for sg in subgraphs {
        walk(sg, &mut out);
    }
    out
}

fn dedup_eligible(sg: &SubGraph) -> bool {
    if sg.node_ids.len() < MIN_DEDUP_SUBGRAPH_NODES {
        return false;
    }
    if !sg.children.is_empty() {
        return true;
    }
    sg.node_ids.len() > 1
}

fn signature_for_subgraph(
    sg: &SubGraph,
    node_labels: &HashMap<usize, &str>,
    memo: &mut HashMap<usize, String>,
) -> String {
    if let Some(sig) = memo.get(&sg.id) {
        return sig.clone();
    }

    let child_node_ids: HashSet<usize> = sg
        .children
        .iter()
        .flat_map(|c| c.node_ids.iter().copied())
        .collect();
    let mut direct_labels: Vec<String> = sg
        .node_ids
        .iter()
        .filter(|nid| !child_node_ids.contains(nid))
        .filter_map(|nid| node_labels.get(nid).copied())
        .map(|s| s.to_string())
        .collect();
    direct_labels.sort();

    let child_sigs: Vec<String> = sg
        .children
        .iter()
        .map(|child| signature_for_subgraph(child, node_labels, memo))
        .collect();

    let sig = format!(
        "name={}|custom={}|direct={:?}|children={:?}",
        sg.name, sg.is_custom, direct_labels, child_sigs
    );
    memo.insert(sg.id, sig.clone());
    sig
}

pub fn pretty_type(raw: &str) -> String {
    simplify_type_name(raw)
}

fn simplify_type_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let bytes = raw.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i] as char;
        if is_ident_start(c) {
            let start = i;
            i += 1;
            while i < bytes.len() {
                let ch = bytes[i] as char;
                if is_ident_continue(ch) || ch == ':' {
                    i += 1;
                } else {
                    break;
                }
            }

            let token = &raw[start..i];
            let short = token.rsplit("::").next().unwrap_or(token);

            if short == "TensorShape" && i < bytes.len() && bytes[i] == b'<' {
                let end = skip_balanced_angle(raw, i);
                let shape =
                    pretty_tensor_shape(&raw[start..end]).unwrap_or_else(|| "_".to_string());
                i = end;
                out.push_str(&shape);
                continue;
            }

            out.push_str(short);
            continue;
        }

        out.push(c);
        i += 1;
    }

    out
}

fn skip_balanced_angle(s: &str, start: usize) -> usize {
    let bytes = s.as_bytes();
    if start >= bytes.len() || bytes[start] != b'<' {
        return start;
    }

    let mut depth = 1usize;
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    bytes.len()
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic()
}

fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_ascii_alphanumeric()
}

fn pretty_tensor_shape(raw: &str) -> Option<String> {
    let shape_start = raw.find("TensorShape<")?;
    let angle_start = shape_start + "TensorShape".len();
    let shape_end = skip_balanced_angle(raw, angle_start);
    if shape_end <= angle_start + 1 || shape_end > raw.len() {
        return None;
    }

    let inner = raw.get(angle_start + 1..shape_end - 1)?.trim();
    let compact = pretty_shape(inner);
    if compact.starts_with('[') && compact.ends_with(']') {
        return Some(compact);
    }

    let dims = parse_tensor_shape_dims(inner)?;
    Some(format!("[{}]", dims.join(", ")))
}

fn parse_tensor_shape_dims(inner: &str) -> Option<Vec<String>> {
    let mut dims = Vec::new();
    if parse_list_expr(inner.trim(), &mut dims) && !dims.is_empty() {
        Some(dims)
    } else {
        None
    }
}

fn parse_list_expr(expr: &str, dims: &mut Vec<String>) -> bool {
    let list_pos = match expr.find("List<") {
        Some(pos) => pos,
        None => return false,
    };
    let angle_start = list_pos + "List".len();
    let angle_end = skip_balanced_angle(expr, angle_start);
    if angle_end <= angle_start + 1 || angle_end > expr.len() {
        return false;
    }

    let inner = match expr.get(angle_start + 1..angle_end - 1) {
        Some(v) => v.trim(),
        None => return false,
    };

    if inner == "()" {
        return true;
    }
    if !(inner.starts_with('(') && inner.ends_with(')')) {
        return false;
    }

    let pair = &inner[1..inner.len() - 1];
    let (head, tail) = match split_top_level_comma(pair) {
        Some(v) => v,
        None => return false,
    };
    let dim = match parse_dim_expr(head.trim()) {
        Some(v) => v,
        None => return false,
    };
    dims.push(dim);
    parse_list_expr(tail.trim(), dims)
}

fn parse_dim_expr(expr: &str) -> Option<String> {
    if expr.contains("Dyn<") {
        let label = last_ident(expr)?;
        let trimmed = label.trim_start_matches('_').to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    } else {
        parse_typenum_uint(expr).map(|v| v.to_string())
    }
}

fn parse_typenum_uint(expr: &str) -> Option<u64> {
    let t = expr.trim();
    if t.ends_with("UTerm") {
        return Some(0);
    }

    let pos = t.find("UInt<")?;
    let angle_start = pos + "UInt".len();
    let angle_end = skip_balanced_angle(t, angle_start);
    if angle_end <= angle_start + 1 || angle_end > t.len() {
        return None;
    }
    let args = t.get(angle_start + 1..angle_end - 1)?.trim();
    let (lhs, rhs) = split_top_level_comma(args)?;
    let rest = parse_typenum_uint(lhs.trim())?;
    let bit = if rhs.contains("B1") {
        1
    } else if rhs.contains("B0") {
        0
    } else {
        return None;
    };
    Some(rest * 2 + bit)
}

fn split_top_level_comma(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    let mut angle_depth = 0usize;
    let mut paren_depth = 0usize;
    for (idx, b) in bytes.iter().enumerate() {
        match *b {
            b'<' => angle_depth += 1,
            b'>' => angle_depth = angle_depth.saturating_sub(1),
            b'(' => paren_depth += 1,
            b')' => paren_depth = paren_depth.saturating_sub(1),
            b',' if angle_depth == 0 && paren_depth == 0 => {
                let left = s.get(..idx)?;
                let right = s.get(idx + 1..)?;
                return Some((left, right));
            }
            _ => {}
        }
    }
    None
}

fn last_ident(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let mut end = bytes.len();
    while end > 0 {
        let c = bytes[end - 1] as char;
        if is_ident_continue(c) {
            break;
        }
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut start = end;
    while start > 0 {
        let c = bytes[start - 1] as char;
        if is_ident_continue(c) {
            start -= 1;
        } else {
            break;
        }
    }
    s.get(start..end)
}

/// Parse a raw `std::any::type_name` of a glowstick `ShapeDiagnostic::Out` into a
/// human-readable shape string like `[BatchSize, SequenceLength, 2048]`.
///
/// Input format (from `type_name` with or without `glowstick::` prefixes):
/// ```text
/// (RANK<_3>, (DIM<Dyn<(_1, path::to::BatchSize)>>, DIM<(_2, _0, _4, _8)>, DIM<_8>))
/// ```
/// Output: `[BatchSize, 2048, 8]`
pub fn pretty_shape(raw: &str) -> String {
    let s = raw.replace("glowstick::", "");
    let mut dims = Vec::new();
    let bytes = s.as_bytes();
    let mut pos = 0;

    while pos < s.len() {
        if let Some(offset) = s[pos..].find("DIM<") {
            let content_start = pos + offset + 4;
            // Find the matching '>' for this DIM<
            let mut depth = 1;
            let mut end = content_start;
            while end < s.len() && depth > 0 {
                match bytes[end] {
                    b'<' => depth += 1,
                    b'>' => depth -= 1,
                    _ => {}
                }
                if depth > 0 {
                    end += 1;
                }
            }
            let content = &s[content_start..end];
            dims.push(format_dim(content).trim_start_matches('_').to_string());
            pos = end + 1;
        } else {
            break;
        }
    }

    if dims.is_empty() {
        return s.to_string();
    }

    format!("[{}]", dims.join(", "))
}

/// Format a single DIM content string.
fn format_dim(content: &str) -> String {
    let content = content.trim();
    if content.starts_with("Dyn<") {
        // Dynamic dim: Dyn<(_1, path::to::BatchSize)> or Dyn<(path::Label)>
        // Extract the label — last `::` segment before closing parens
        let inner = &content[4..content.len().saturating_sub(1)]; // strip Dyn< and >
        if let Some(last_colon) = inner.rfind("::") {
            let after = &inner[last_colon + 2..];
            let label: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if label.is_empty() {
                content.to_string()
            } else {
                label
            }
        } else {
            // No path separator — try to extract a bare label
            let trimmed = inner.trim_matches(|c: char| c == '(' || c == ')' || c.is_whitespace());
            trimmed.to_string()
        }
    } else {
        // Static dim: (_8, _1, _9, _2) or single _3
        parse_static_dim(content)
    }
}

/// Convert a static dim like `(_8, _1, _9, _2)` or `_3` into a decimal string like `8192` or `3`.
fn parse_static_dim(content: &str) -> String {
    let mut value: u64 = 0;
    let mut found = false;

    for part in content.split('_') {
        let trimmed = part.trim_matches(|c: char| !c.is_ascii_digit());
        if let Ok(d) = trimmed.parse::<u64>() {
            value = value * 10 + d;
            found = true;
        }
    }

    if found {
        value.to_string()
    } else {
        content.to_string()
    }
}

fn write_subgraph(out: &mut String, sg: &SubGraph, depth: usize, plan: &RenderPlan) {
    let indent = "    ".repeat(depth);
    out.push_str(&format!("{}subgraph cluster_{} {{\n", indent, sg.id));
    out.push_str(&format!(
        "{}    label=\"{}\";\n",
        indent,
        dot_escape(&sg.name)
    ));
    if sg.is_custom {
        out.push_str(&format!("{}    color=\"red\";\n", indent,));
    } else {
        out.push_str(&format!("{}    color=\"black\";\n", indent,));
    }
    out.push_str(&format!("{}    style=rounded;\n", indent));

    // Recurse into children
    for child in &sg.children {
        if let Some(ci) = plan.collapsed.get(&child.id) {
            out.push_str(&format!("{}    n{};\n", indent, ci.summary_node_id));
        } else {
            write_subgraph(out, child, depth + 1, plan);
        }
    }

    // Reference nodes that belong to this subgraph but not any child
    let child_node_ids: HashSet<usize> = sg
        .children
        .iter()
        .flat_map(|c| c.node_ids.iter().copied())
        .collect();
    for &nid in &sg.node_ids {
        if !child_node_ids.contains(&nid) && plan.emitted_node_ids.contains(&nid) {
            out.push_str(&format!("{}    n{};\n", indent, nid));
        }
    }

    out.push_str(&format!("{}}}\n", indent));
}

#[inception(property = Visualize)]
pub trait Vis {
    fn visualize() -> crate::vis::Graph;

    fn nothing() -> crate::vis::Graph {
        Default::default()
    }

    fn merge<H: Vis<Ret = crate::vis::Graph>, R: Vis<Ret = crate::vis::Graph>>(
        _t: L,
        _f: R,
    ) -> crate::vis::Graph {
        let l = <H as __inception_vis::Inductive>::visualize();
        let r = <R as __inception_vis::Inductive>::visualize();
        crate::vis::Graph::sequence(l, r)
    }

    fn merge_variant_field<H: Vis<Ret = crate::vis::Graph>, R: Vis<Ret = crate::vis::Graph>>(
        _t: L,
        _f: R,
    ) -> crate::vis::Graph {
        let l = <H as __inception_vis::Inductive>::visualize();
        let r = <R as __inception_vis::Inductive>::visualize();
        crate::vis::Graph::sequence(l, r)
    }

    fn join<F: Vis<Ret = crate::vis::Graph>>(_f: F) -> crate::vis::Graph {
        crate::vis::Graph::wrap_subgraph(Self::NAME, <F as __inception_vis::Inductive>::visualize())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use inception::Inception;

    pub struct Foo;
    #[primitive(property = Visualize)]
    impl Vis for Foo {
        fn visualize() -> Graph {
            Graph::leaf("FOO")
        }
    }

    pub struct Bar;
    #[primitive(property = Visualize)]
    impl Vis for Bar {
        fn visualize() -> Graph {
            Graph::leaf("BAR")
        }
    }

    #[derive(Inception)]
    #[inception(properties = [Visualize])]
    struct Waldo(Foo, Bar);

    #[derive(Inception)]
    #[inception(properties = [Visualize])]
    struct Inner(Foo, Bar);

    #[derive(Inception)]
    #[inception(properties = [Visualize])]
    struct Outer(Inner, Inner);

    #[derive(Inception)]
    #[inception(properties = [Visualize])]
    struct Tiny(Foo);

    #[derive(Inception)]
    #[inception(properties = [Visualize])]
    struct TinyOuter(Tiny, Tiny);

    #[derive(Inception)]
    #[inception(properties = [Visualize])]
    struct Big(Inner, Inner);

    #[derive(Inception)]
    #[inception(properties = [Visualize])]
    struct BigOuter(Big, Big);

    #[test]
    fn test_subgraph_ids_are_unique_even_with_same_label() {
        let g1 = Graph::wrap_subgraph("Repeat", Graph::leaf("A"));
        let g2 = Graph::wrap_subgraph("Repeat", Graph::leaf("B"));
        let g = Graph::sequence(g1, g2);
        let dot = g.to_dot();

        let cluster_lines: Vec<&str> = dot
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("subgraph cluster_"))
            .collect();
        assert_eq!(cluster_lines.len(), 2);

        let unique: std::collections::HashSet<&str> = cluster_lines.iter().copied().collect();
        assert_eq!(unique.len(), 2);

        assert!(!g.edges.is_empty());
    }

    #[test]
    fn test_nested_inception_fields_are_sequential() {
        let g = Outer::visualize();
        // Outer(Inner(Foo,Bar), Inner(Foo,Bar)) should include at least
        // Foo->Bar in each Inner and one edge from first Inner output to second Inner input.
        assert!(g.edges.len() >= 3, "edges: {:?}", g.edges);
    }

    #[test]
    fn test_dedup_skips_tiny_repeated_subgraphs() {
        let g = TinyOuter::visualize();
        let dot = g.to_dot();
        let tiny_labels = dot
            .lines()
            .filter(|line| line.contains("label=\"Tiny\";"))
            .count();
        assert_eq!(tiny_labels, 2);
        assert!(!dot.contains("Tiny (shared x"));
    }

    #[test]
    fn test_dedup_collapses_repeated_composite_subgraphs() {
        let g = BigOuter::visualize();
        let dot = g.to_dot();
        let big_cluster_labels = dot
            .lines()
            .filter(|line| line.contains("label=\"Big\";"))
            .count();
        assert_eq!(big_cluster_labels, 1, "dot:\n{dot}");
        assert!(dot.contains("Big (shared x2)"), "dot:\n{dot}");
    }

    #[test]
    fn test_sequence_with_empty_tail_preserves_outputs() {
        let g = Graph::sequence(Graph::leaf("Only"), Graph::default());
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.inputs.len(), 1);
    }

    #[test]
    fn test_pretty_shape_static() {
        // Static 2D shape: [8192, 2048]
        let raw = "(RANK<_2>, (DIM<(_8, _1, _9, _2)>, DIM<(_2, _0, _4, _8)>))";
        assert_eq!(pretty_shape(raw), "[8192, 2048]");
    }

    #[test]
    fn test_pretty_shape_single_digit() {
        let raw = "(RANK<_1>, (DIM<_8>,))";
        assert_eq!(pretty_shape(raw), "[8]");
    }

    #[test]
    fn test_pretty_shape_dynamic() {
        let raw = "(RANK<_3>, (DIM<Dyn<(_1, paramecia_model::models::qwen3_next::shape::BatchSize)>>, DIM<Dyn<(_1, paramecia_model::models::qwen3_next::shape::SequenceLength)>>, DIM<(_2, _0, _4, _8)>))";
        assert_eq!(pretty_shape(raw), "[BatchSize, SequenceLength, 2048]");
    }

    #[test]
    fn test_pretty_shape_with_glowstick_prefix() {
        let raw = "(glowstick::RANK<glowstick::_2>, (glowstick::DIM<(glowstick::_8, glowstick::_1, glowstick::_9, glowstick::_2)>, glowstick::DIM<(glowstick::_2, glowstick::_0, glowstick::_4, glowstick::_8)>))";
        assert_eq!(pretty_shape(raw), "[8192, 2048]");
    }

    #[test]
    fn test_pretty_type() {
        let raw = "core::result::Result<paramecia_tensor::tensor::Tensor<TensorShape<typosaurus::collections::list::List<(Dyn<dynamic::Term<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>, paramecia_model::models::qwen3_next::shape::_B>>, typosaurus::collections::list::List<(Dyn<dynamic::Term<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>, paramecia_model::models::qwen3_next::shape::_N>>, typosaurus::collections::list::List<(typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typenum::bit::B0>, typosaurus::collections::list::List<()>)>)>)>>>, paramecia_core::error::Error>";
        assert_eq!(pretty_type(raw), "Result<Tensor<[B, N, 2048]>, Error>");
    }

    #[test]
    fn test_pretty_type_strips_module_paths() {
        let raw = "alloc::vec::Vec<core::option::Option<paramecia_core::error::Error>>";
        assert_eq!(pretty_type(raw), "Vec<Option<Error>>");
    }
}
