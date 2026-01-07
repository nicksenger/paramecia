use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Duration;

use iced::widget::{
    Column, Container, Row, Scrollable, Space, Stack, button, text, text_input, tooltip,
};
use iced::{Background, Border, Color, Font, Length, Shadow, Subscription, keyboard};
use iced_sugiyama::{
    Cluster, EdgeEndpoint, EdgeEndpointKind, Graph as SugiyamaGraph, OutgoingEdgeStyle, Sugiyama,
};
use paramecia_arrow::vis::{Graph as ArrowGraph, SubGraph as ArrowSubGraph, Vis};

const WINDOW_WIDTH: f32 = 1400.0;
const WINDOW_HEIGHT: f32 = 900.0;
const NODE_WIDTH: f64 = 220.0;
const NODE_HEIGHT: f64 = 80.0;
const MAX_NODE_LABEL_CHARS: usize = 42;
const MIN_DEDUP_SUBGRAPH_NODES: usize = 4;
const DEFAULT_MAX_VISIBLE_CLUSTERS: usize = 180;
const DEFAULT_MIN_CLUSTER_VISIBLE_NODES: usize = 4;
const ALWAYS_INCLUDE_CLUSTER_DEPTH: usize = 2;
const LIVE_POLL_INTERVAL_MS: u64 = 200;
const MODEL_GRAPH_WIDGET_ID: &str = "model-graph";

#[derive(Clone, Debug)]
pub struct TensorOpSnapshot {
    pub sequence: u64,
    pub emitted_at_unix_ms: u128,
    pub rows: Vec<TensorOpSnapshotRow>,
}

#[derive(Clone, Debug)]
pub struct TensorOpSnapshotRow {
    pub key: Option<String>,
    pub node_stable_id: Option<u64>,
    pub count: u64,
    pub total_ns: u128,
}

#[derive(Clone, Debug)]
pub enum ChatOverlayEvent {
    Status(String),
    AssistantDelta(String),
    AssistantDone,
    Error(String),
}

#[derive(Clone, Debug)]
pub enum ChatOverlayCommand {
    SendPrompt(String),
    Interrupt,
}

pub struct VisualizerChannels {
    pub trace_rx: Receiver<TensorOpSnapshot>,
    pub chat_event_rx: Receiver<ChatOverlayEvent>,
    pub chat_command_tx: Sender<ChatOverlayCommand>,
}

pub fn go_with_channels(channels: VisualizerChannels) -> iced::Result {
    iced::application(
        |_state: &ModelVisualizer| "Paramecia Model Visualizer".to_string(),
        ModelVisualizer::update,
        ModelVisualizer::view,
    )
    .subscription(ModelVisualizer::subscription)
    .theme(|_| iced::Theme::Light)
    .default_font(Font::with_name("Times"))
    .window_size((WINDOW_WIDTH, WINDOW_HEIGHT))
    .antialiasing(true)
    .run_with(|| (ModelVisualizer::with_channels(channels), iced::Task::none()))
}

#[derive(Clone, Debug)]
struct NodeInfo {
    label: String,
}

#[derive(Clone, Debug)]
struct SubgraphInfo {
    name: String,
    node_ids: Vec<usize>,
    child_ids: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Membership {
    subgraph_id: usize,
    depth: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VisibleOwner {
    RealNode(usize),
    CollapsedSubgraph(usize),
}

#[derive(Clone, Debug)]
struct RenderedNode {
    label: String,
    tooltip: Option<String>,
    is_collapsed: bool,
    is_ghost: bool,
    heat_ratio: Option<f32>,
}

#[derive(Clone, Debug)]
struct RenderedCluster {
    subgraph_id: usize,
    name: String,
    heat_ratio: Option<f32>,
    has_timing: bool,
}

#[derive(Clone, Copy, Debug)]
struct ClusterCandidate {
    subgraph_id: usize,
    depth: usize,
    visible_member_count: usize,
    child_count: usize,
    is_redundant_wrapper: bool,
}

#[derive(Clone, Default)]
struct SharedPathDedup {
    forced_collapsed_subgraphs: HashSet<usize>,
}

#[derive(Clone)]
struct RenderedGraph {
    graph: SugiyamaGraph,
    node_views: HashMap<u32, RenderedNode>,
    collapsed_subgraph_by_display_id: HashMap<u32, usize>,
    clusters: Vec<Cluster>,
    cluster_by_index: HashMap<usize, RenderedCluster>,
    edge_labels: Vec<Option<String>>,
}

struct ModelVisualizer {
    source_graph: ArrowGraph,
    node_info_by_id: HashMap<usize, NodeInfo>,
    tensor_op_node_ids: HashSet<usize>,
    node_id_by_stable_id: HashMap<u64, usize>,
    subgraph_by_id: HashMap<usize, SubgraphInfo>,
    subgraph_contains_tensor_op: HashMap<usize, bool>,
    parent_by_subgraph: HashMap<usize, usize>,
    depth_by_subgraph: HashMap<usize, usize>,
    node_memberships: HashMap<usize, Vec<Membership>>,
    display_id_by_node: HashMap<usize, u32>,
    display_id_by_subgraph: HashMap<usize, u32>,
    node_total_ns_by_node_id: HashMap<usize, u128>,
    node_total_count_by_node_id: HashMap<usize, u64>,
    shared_path_dedup: SharedPathDedup,
    forced_expand_shared_subgraphs: HashSet<usize>,
    expanded_subgraphs: HashSet<usize>,
    rendered_graph: RenderedGraph,
    trace_rx: Receiver<TensorOpSnapshot>,
    chat_event_rx: Receiver<ChatOverlayEvent>,
    chat_command_tx: Sender<ChatOverlayCommand>,
    chat_backend_disconnected: bool,
    chat_backend_failure_reported: bool,
    chat_input: String,
    chat_lines: Vec<ChatOverlayLine>,
    is_chat_streaming: bool,
    last_live_sequence: u64,
    last_cumulative_ns_by_node_id: HashMap<usize, u128>,
    last_cumulative_count_by_node_id: HashMap<usize, u64>,
    received_first_trace_snapshot: bool,
}

#[derive(Clone, Debug)]
struct ChatOverlayLine {
    role: ChatOverlayRole,
    content: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChatOverlayRole {
    User,
    Assistant,
    System,
    Error,
}

impl ModelVisualizer {
    fn with_channels(channels: VisualizerChannels) -> Self {
        let source_graph = paramecia_model::models::qwen3_next::ModelWeights::visualize();

        let node_info_by_id = source_graph
            .nodes
            .iter()
            .map(|node| {
                (
                    node.id,
                    NodeInfo {
                        label: node.label.clone(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        let node_id_by_stable_id = source_graph
            .nodes
            .iter()
            .filter_map(|node| {
                if node.stable_id == 0 {
                    return None;
                }
                Some((node.stable_id, node.id))
            })
            .collect::<HashMap<_, _>>();
        let tensor_op_node_ids = source_graph
            .nodes
            .iter()
            .filter_map(|node| {
                if is_tensor_op_label(&node.label) {
                    Some(node.id)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();

        let (subgraph_by_id, parent_by_subgraph, depth_by_subgraph, mut node_memberships) =
            index_subgraphs(&source_graph.subgraphs);
        let subgraph_contains_tensor_op =
            build_subgraph_tensor_op_presence(&subgraph_by_id, &tensor_op_node_ids);
        let shared_path_dedup = build_shared_path_dedup(&source_graph);

        for memberships in node_memberships.values_mut() {
            memberships.sort_by(|left, right| {
                left.depth
                    .cmp(&right.depth)
                    .then(left.subgraph_id.cmp(&right.subgraph_id))
            });
        }

        let (display_id_by_node, display_id_by_subgraph) =
            assign_display_ids(&source_graph, &subgraph_by_id);

        let mut visualizer = Self {
            source_graph,
            node_info_by_id,
            tensor_op_node_ids,
            node_id_by_stable_id,
            subgraph_by_id,
            subgraph_contains_tensor_op,
            parent_by_subgraph,
            depth_by_subgraph,
            node_memberships,
            display_id_by_node,
            display_id_by_subgraph,
            node_total_ns_by_node_id: HashMap::new(),
            node_total_count_by_node_id: HashMap::new(),
            shared_path_dedup,
            forced_expand_shared_subgraphs: HashSet::new(),
            expanded_subgraphs: HashSet::new(),
            rendered_graph: RenderedGraph {
                graph: SugiyamaGraph::new(vec![], vec![]),
                node_views: Default::default(),
                collapsed_subgraph_by_display_id: Default::default(),
                clusters: Default::default(),
                cluster_by_index: Default::default(),
                edge_labels: Default::default(),
            },
            trace_rx: channels.trace_rx,
            chat_event_rx: channels.chat_event_rx,
            chat_command_tx: channels.chat_command_tx,
            chat_backend_disconnected: false,
            chat_backend_failure_reported: false,
            chat_input: String::new(),
            chat_lines: Vec::new(),
            is_chat_streaming: false,
            last_live_sequence: 0,
            last_cumulative_ns_by_node_id: HashMap::new(),
            last_cumulative_count_by_node_id: HashMap::new(),
            received_first_trace_snapshot: false,
        };
        visualizer.rendered_graph = visualizer.build_rendered_graph();

        visualizer
    }
}

#[derive(Debug, Clone)]
enum Message {
    NodePressed(u32),
    CollapseSubgraph(usize),
    ChatInputChanged(String),
    ChatSendPressed,
    ChatInterruptPressed,
    LiveTick,
}

impl ModelVisualizer {
    fn subscription(&self) -> Subscription<Message> {
        let tick = iced::time::every(Duration::from_millis(LIVE_POLL_INTERVAL_MS))
            .map(|_| Message::LiveTick);
        let hotkeys = keyboard::on_key_press(|key, _modifiers| {
            use keyboard::key;

            match key.as_ref() {
                keyboard::Key::Named(key::Named::Escape) => Some(Message::ChatInterruptPressed),
                _ => None,
            }
        });

        Subscription::batch(vec![tick, hotkeys])
    }

    fn update(&mut self, message: Message) -> iced::Task<Message> {
        match message {
            Message::NodePressed(display_id) => {
                if let Some(subgraph_id) = self
                    .rendered_graph
                    .collapsed_subgraph_by_display_id
                    .get(&display_id)
                    .copied()
                {
                    if self
                        .shared_path_dedup
                        .forced_collapsed_subgraphs
                        .contains(&subgraph_id)
                    {
                        self.forced_expand_shared_subgraphs.insert(subgraph_id);
                    }
                    self.expand_subgraph_with_ancestors(subgraph_id);
                    self.rendered_graph = self.build_rendered_graph();
                }
            }
            Message::CollapseSubgraph(subgraph_id) => {
                self.collapse_subgraph_and_descendants(subgraph_id);
                self.rendered_graph = self.build_rendered_graph();
            }
            Message::ChatInputChanged(value) => {
                self.chat_input = value;
            }
            Message::ChatSendPressed => {
                let prompt = self.chat_input.trim().to_string();
                if prompt.is_empty() {
                    return iced::Task::none();
                }
                if self.chat_backend_disconnected {
                    if !self.chat_backend_failure_reported {
                        self.chat_lines.push(ChatOverlayLine {
                            role: ChatOverlayRole::Error,
                            content: "Agent backend is not connected".to_string(),
                        });
                        self.chat_backend_failure_reported = true;
                    }
                    return iced::Task::none();
                }
                self.chat_lines.push(ChatOverlayLine {
                    role: ChatOverlayRole::User,
                    content: prompt.clone(),
                });
                self.chat_input.clear();
                self.is_chat_streaming = true;
                if self
                    .chat_command_tx
                    .send(ChatOverlayCommand::SendPrompt(prompt))
                    .is_err()
                {
                    self.chat_lines.push(ChatOverlayLine {
                        role: ChatOverlayRole::Error,
                        content: "Failed to send prompt to background agent".to_string(),
                    });
                    self.chat_backend_disconnected = true;
                    self.chat_backend_failure_reported = true;
                    self.is_chat_streaming = false;
                }
            }
            Message::ChatInterruptPressed => {
                if self
                    .chat_command_tx
                    .send(ChatOverlayCommand::Interrupt)
                    .is_err()
                {
                    self.chat_backend_disconnected = true;
                    if !self.chat_backend_failure_reported {
                        self.chat_lines.push(ChatOverlayLine {
                            role: ChatOverlayRole::Error,
                            content: "Agent backend disconnected".to_string(),
                        });
                        self.chat_backend_failure_reported = true;
                    }
                }
                self.is_chat_streaming = false;
            }
            Message::LiveTick => {
                let mut should_rerender_graph = false;
                let mut received_trace_snapshot = false;
                while let Some(snapshot) = self.try_recv_trace_snapshot() {
                    received_trace_snapshot = true;
                    if self.apply_live_snapshot(snapshot) {
                        should_rerender_graph = true;
                    }
                }
                while let Some(event) = self.try_recv_chat_event() {
                    self.apply_chat_event(event);
                }
                if should_rerender_graph || received_trace_snapshot {
                    self.rendered_graph = self.build_rendered_graph();
                }
                if received_trace_snapshot && !self.received_first_trace_snapshot {
                    self.received_first_trace_snapshot = true;
                    return iced_sugiyama::force_review::<Message>(
                        iced_sugiyama::Id::new(MODEL_GRAPH_WIDGET_ID),
                    );
                }
            }
        }

        iced::Task::none()
    }

    fn view(&self) -> Container<'_, Message> {
        let node_views = self.rendered_graph.node_views.clone();
        let cluster_by_index = self.rendered_graph.cluster_by_index.clone();
        let invisible_edge_sources = node_views
            .iter()
            .filter_map(|(id, node)| if node.is_ghost { Some(*id) } else { None })
            .collect::<HashSet<_>>();

        let graph_view = Sugiyama::<Message, iced::Theme, iced::Renderer>::new(
            &self.rendered_graph.graph,
            move |id| {
                let fallback = RenderedNode {
                    label: format!("node {id}"),
                    tooltip: None,
                    is_collapsed: false,
                    is_ghost: false,
                    heat_ratio: None,
                };

                let data = node_views.get(&id).cloned().unwrap_or(fallback);
                if data.is_ghost {
                    return Space::new(Length::Fixed(0.0), Length::Fixed(0.0)).into();
                }
                let label = truncate_label(&data.label, MAX_NODE_LABEL_CHARS);

                let title = if data.is_collapsed {
                    format!("[+] {label}")
                } else {
                    label
                };

                let mut node_button = button(text(title).size(14)).padding([8, 10]);
                if data.is_collapsed {
                    node_button = node_button.on_press(Message::NodePressed(id));
                }

                let heat_ratio = data.heat_ratio;
                let is_collapsed = data.is_collapsed;
                node_button = node_button.style(move |_theme, status| {
                    heat_node_button_style(status, heat_ratio, is_collapsed)
                });

                let node_button = node_button.width(Length::Shrink);
                if let Some(tooltip_label) = data.tooltip {
                    return tooltip(
                        node_button,
                        Container::new(text(tooltip_label).size(11))
                            .padding([4, 6])
                            .style(node_tooltip_style),
                        tooltip::Position::Top,
                    )
                    .gap(6)
                    .into();
                }

                node_button.into()
            },
        )
        .id(iced_sugiyama::Id::new(MODEL_GRAPH_WIDGET_ID))
        .edge_color(edge_color)
        .cluster_container(move |cluster_index, _| {
            let cluster = cluster_by_index.get(&cluster_index)?;
            let heat_ratio = cluster.heat_ratio;
            let has_timing = cluster.has_timing;
            let collapse = button(text("-").size(16))
                .on_press(Message::CollapseSubgraph(cluster.subgraph_id))
                .padding([0, 6])
                .style(move |theme, status| {
                    cluster_collapse_button_style(theme, status, heat_ratio)
                });
            let title = Container::new(text(cluster.name.clone()).size(14))
                .padding([2, 6])
                .style(cluster_title_style);
            let header = Row::new()
                .push(title)
                .push(Space::new(Length::Fill, Length::Shrink))
                .push(collapse)
                .width(Length::Fill)
                .align_y(iced::alignment::Vertical::Center);

            Some(
                Container::new(header)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding([8, 10])
                    .align_x(iced::alignment::Horizontal::Left)
                    .align_y(iced::alignment::Vertical::Top)
                    .style(move |theme| cluster_container_style(theme, heat_ratio, has_timing))
                    .into(),
            )
        })
        .edge_label(move |index, _| {
            self.rendered_graph
                .edge_labels
                .get(index)
                .and_then(|op| op.as_ref())
                .cloned()
        })
        .edge_label_element(move |index, _, _| {
            let label = self
                .rendered_graph
                .edge_labels
                .get(index)
                .and_then(|value| value.as_deref())
                .map(str::trim)
                .filter(|value| !value.is_empty())?;

            Some(
                Container::new(text(label.to_string()).size(10))
                    .padding([2, 6])
                    .style(edge_label_container_style)
                    .into(),
            )
        })
        .stroke_width(1.)
        .outgoing_edge_style(move |source_id| {
            if invisible_edge_sources.contains(&source_id) {
                OutgoingEdgeStyle::hidden()
            } else {
                OutgoingEdgeStyle::default()
            }
        })
        .edge_endpoint(|_, _, kind, endpoint| {
            let marker = match kind {
                EdgeEndpointKind::Source => "o",
                EdgeEndpointKind::Destination => directional_marker(endpoint),
            };
            Some(text(marker).size(12).into())
        })
        .node_size(|i| {
            self.rendered_graph
                .edge_labels
                .get(i as usize)
                .and_then(|op| op.as_ref())
                .map(|s| ((s.len() * 4) as f64, 40.))
                .unwrap_or_else(|| node_size(i))
        })
        .clusters(self.rendered_graph.clusters.clone())
        .cluster_color(hidden_cluster_outline)
        .edge_corner_radius(20.0)
        .edge_endpoint_extension(0.0);

        let legend = self.heat_legend();
        let controls = Row::new()
            .push(Space::new(Length::Fill, Length::Shrink))
            .push(legend)
            .width(Length::Fill)
            .align_y(iced::alignment::Vertical::Center);

        let graph_column = Column::new()
            .push(Container::new(controls).padding([8, 12]))
            .push(
                Container::new(graph_view)
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill);

        let chat_overlay = Container::new(
            Column::new()
                .push(Space::new(Length::Fill, Length::Fill))
                .push(
                    Row::new()
                        .push(Space::new(Length::Fill, Length::Shrink))
                        .push(self.chat_overlay_view())
                        .width(Length::Fill),
                )
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .padding(16)
        .width(Length::Fill)
        .height(Length::Fill);

        Container::new(
            Stack::new()
                .push(graph_column)
                .push(chat_overlay)
                .width(Length::Fill)
                .height(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
    }

    fn chat_overlay_view(&self) -> Container<'_, Message> {
        let mut lines = Column::new().spacing(6);
        for line in &self.chat_lines {
            let (prefix, color) = match line.role {
                ChatOverlayRole::User => ("You", Color::from_rgb8(0, 0, 0)),
                ChatOverlayRole::Assistant => ("Agent", Color::from_rgb8(0, 0, 0)),
                ChatOverlayRole::System => ("System", Color::from_rgb8(0, 0, 0)),
                ChatOverlayRole::Error => ("Error", Color::from_rgb8(164, 46, 46)),
            };
            lines = lines.push(
                text(format!("{prefix}: {}", line.content))
                    .size(12)
                    .color(color),
            );
        }
        if self.chat_lines.is_empty() {
            lines = lines.push(
                text("No chat activity yet")
                    .size(12)
                    .color(Color::from_rgb8(120, 120, 120)),
            );
        }

        let status = if self.is_chat_streaming {
            "Agent is responding..."
        } else if self.chat_backend_disconnected {
            "Agent backend disconnected"
        } else {
            "Ready"
        };

        let input = text_input("Type a message...", &self.chat_input)
            .on_input(Message::ChatInputChanged)
            .on_submit(Message::ChatSendPressed)
            .padding([6, 8])
            .size(13);

        Container::new(
            Column::new()
                .spacing(8)
                .push(
                    Row::new()
                        .spacing(8)
                        .push(text("Agent").size(14))
                        .push(Space::new(Length::Fill, Length::Shrink))
                        .push(text(status).size(11).color(Color::from_rgb8(110, 110, 110)))
                        .align_y(iced::alignment::Vertical::Center),
                )
                .push(
                    Scrollable::new(lines)
                        .height(Length::Fixed(190.0))
                        .width(Length::Fill),
                )
                .push(
                    Row::new()
                        .spacing(8)
                        .push(input)
                        .align_y(iced::alignment::Vertical::Center),
                ),
        )
        .padding([10, 10])
        .width(Length::Fixed(420.0))
        .style(chat_overlay_container_style)
    }

    fn try_recv_trace_snapshot(&mut self) -> Option<TensorOpSnapshot> {
        match self.trace_rx.try_recv() {
            Ok(snapshot) => Some(snapshot),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
    }

    fn try_recv_chat_event(&mut self) -> Option<ChatOverlayEvent> {
        match self.chat_event_rx.try_recv() {
            Ok(event) => Some(event),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.chat_backend_disconnected = true;
                if !self.chat_backend_failure_reported {
                    self.chat_lines.push(ChatOverlayLine {
                        role: ChatOverlayRole::Error,
                        content: "Agent backend disconnected".to_string(),
                    });
                    self.chat_backend_failure_reported = true;
                }
                None
            }
        }
    }

    fn apply_chat_event(&mut self, event: ChatOverlayEvent) {
        match event {
            ChatOverlayEvent::Status(message) => {
                self.chat_lines.push(ChatOverlayLine {
                    role: ChatOverlayRole::System,
                    content: message,
                });
            }
            ChatOverlayEvent::AssistantDelta(content) => {
                if content.is_empty() {
                    return;
                }
                if let Some(last) = self.chat_lines.last_mut()
                    && last.role == ChatOverlayRole::Assistant
                    && self.is_chat_streaming
                {
                    last.content.push_str(&content);
                    return;
                }
                self.chat_lines.push(ChatOverlayLine {
                    role: ChatOverlayRole::Assistant,
                    content,
                });
                self.is_chat_streaming = true;
            }
            ChatOverlayEvent::AssistantDone => {
                self.is_chat_streaming = false;
            }
            ChatOverlayEvent::Error(message) => {
                self.chat_lines.push(ChatOverlayLine {
                    role: ChatOverlayRole::Error,
                    content: message,
                });
                self.is_chat_streaming = false;
            }
        }
    }

    fn build_rendered_graph(&self) -> RenderedGraph {
        let mut visible_nodes = HashSet::<u32>::new();
        let mut node_views = HashMap::<u32, RenderedNode>::new();
        let mut collapsed_subgraph_by_display_id = HashMap::<u32, usize>::new();
        let heat_ratio_by_display_id = self.display_heat_ratio_by_display_id();

        for node in &self.source_graph.nodes {
            let owner = self.visible_owner_for_node(node.id);
            if let Some(display_id) = self.display_id_for_owner(owner) {
                visible_nodes.insert(display_id);
                let heat_ratio = heat_ratio_by_display_id.get(&display_id).copied();
                node_views
                    .entry(display_id)
                    .or_insert_with(|| self.rendered_node_for_owner(owner, heat_ratio));
                if let VisibleOwner::CollapsedSubgraph(subgraph_id) = owner {
                    collapsed_subgraph_by_display_id.insert(display_id, subgraph_id);
                }
            }
        }

        let mut nodes = visible_nodes.iter().copied().collect::<Vec<_>>();
        nodes.sort_unstable();

        let mut edge_set = HashSet::<(u32, u32, Option<String>)>::new();
        let mut edges = Vec::<(u32, u32)>::new();
        let mut edge_labels = Vec::<Option<String>>::new();

        for (from, to, label) in &self.source_graph.edges {
            let from_owner = self.visible_owner_for_node(*from);
            let to_owner = self.visible_owner_for_node(*to);
            let from_owner_is_tensor = self.visible_owner_is_tensor_op(from_owner);
            let to_owner_is_tensor = self.visible_owner_is_tensor_op(to_owner);
            let filtered_label = if from_owner_is_tensor && to_owner_is_tensor {
                label.clone()
            } else {
                None
            };

            let Some(from_display) = self.display_id_for_owner(from_owner) else {
                continue;
            };
            let Some(to_display) = self.display_id_for_owner(to_owner) else {
                continue;
            };

            if from_display == to_display {
                continue;
            }

            let key = (from_display, to_display, filtered_label.clone());
            if edge_set.insert(key) {
                edges.push((from_display, to_display));
                edge_labels.push(filtered_label);
            }
        }

        let (clusters, cluster_by_index) = self.build_visible_clusters(&visible_nodes);

        RenderedGraph {
            graph: SugiyamaGraph::new(nodes, edges),
            node_views,
            collapsed_subgraph_by_display_id,
            clusters,
            cluster_by_index,
            edge_labels,
        }
    }

    fn visible_owner_is_tensor_op(&self, owner: VisibleOwner) -> bool {
        match owner {
            VisibleOwner::RealNode(node_id) => self.tensor_op_node_ids.contains(&node_id),
            VisibleOwner::CollapsedSubgraph(_) => false,
        }
    }

    fn visible_owner_for_node(&self, node_id: usize) -> VisibleOwner {
        if let Some(memberships) = self.node_memberships.get(&node_id) {
            for membership in memberships {
                let is_forced_collapsed_shared = self
                    .shared_path_dedup
                    .forced_collapsed_subgraphs
                    .contains(&membership.subgraph_id)
                    && !self
                        .forced_expand_shared_subgraphs
                        .contains(&membership.subgraph_id);
                if is_forced_collapsed_shared {
                    return VisibleOwner::CollapsedSubgraph(membership.subgraph_id);
                }
                if !self.expanded_subgraphs.contains(&membership.subgraph_id) {
                    return VisibleOwner::CollapsedSubgraph(membership.subgraph_id);
                }
            }
        }

        VisibleOwner::RealNode(node_id)
    }

    fn display_id_for_owner(&self, owner: VisibleOwner) -> Option<u32> {
        match owner {
            VisibleOwner::RealNode(node_id) => self.display_id_by_node.get(&node_id).copied(),
            VisibleOwner::CollapsedSubgraph(subgraph_id) => {
                self.display_id_by_subgraph.get(&subgraph_id).copied()
            }
        }
    }

    fn rendered_node_for_owner(
        &self,
        owner: VisibleOwner,
        heat_ratio: Option<f32>,
    ) -> RenderedNode {
        let (owner_total_ns, owner_total_count) = self.owner_window_totals(owner);
        let tooltip = Some(self.node_timing_tooltip(owner_total_ns, owner_total_count));
        match owner {
            VisibleOwner::RealNode(node_id) => {
                if let Some(node) = self.node_info_by_id.get(&node_id) {
                    return RenderedNode {
                        label: node.label.clone(),
                        tooltip,
                        is_collapsed: false,
                        is_ghost: false,
                        heat_ratio,
                    };
                }

                RenderedNode {
                    label: format!("node {node_id}"),
                    tooltip,
                    is_collapsed: false,
                    is_ghost: false,
                    heat_ratio,
                }
            }
            VisibleOwner::CollapsedSubgraph(subgraph_id) => {
                if let Some(subgraph) = self.subgraph_by_id.get(&subgraph_id) {
                    let label = format!("{} ({})", subgraph.name, subgraph.node_ids.len());
                    return RenderedNode {
                        label,
                        tooltip,
                        is_collapsed: true,
                        is_ghost: false,
                        heat_ratio,
                    };
                }

                RenderedNode {
                    label: format!("subgraph {subgraph_id}"),
                    tooltip,
                    is_collapsed: true,
                    is_ghost: false,
                    heat_ratio,
                }
            }
        }
    }

    fn owner_window_totals(&self, owner: VisibleOwner) -> (u128, u64) {
        match owner {
            VisibleOwner::RealNode(node_id) => (
                self.node_total_ns_for_heat(node_id),
                self.node_total_count_for_heat(node_id),
            ),
            VisibleOwner::CollapsedSubgraph(subgraph_id) => {
                if let Some(subgraph) = self.subgraph_by_id.get(&subgraph_id) {
                    return subgraph.node_ids.iter().copied().fold(
                        (0u128, 0u64),
                        |(acc_ns, acc_count), node_id| {
                            (
                                acc_ns.saturating_add(self.node_total_ns_for_heat(node_id)),
                                acc_count.saturating_add(self.node_total_count_for_heat(node_id)),
                            )
                        },
                    );
                }
                (0, 0)
            }
        }
    }

    fn build_visible_clusters(
        &self,
        visible_nodes: &HashSet<u32>,
    ) -> (Vec<Cluster>, HashMap<usize, RenderedCluster>) {
        let max_visible_clusters =
            parse_positive_env_usize("PARAMECIA_VIS_MAX_CLUSTERS", DEFAULT_MAX_VISIBLE_CLUSTERS);
        let min_cluster_visible_nodes = parse_positive_env_usize(
            "PARAMECIA_VIS_MIN_CLUSTER_VISIBLE_NODES",
            DEFAULT_MIN_CLUSTER_VISIBLE_NODES,
        );
        let subgraph_total_ns_by_id = self
            .subgraph_by_id
            .iter()
            .map(|(subgraph_id, subgraph)| {
                let total = subgraph
                    .node_ids
                    .iter()
                    .copied()
                    .map(|node_id| self.node_total_ns_for_heat(node_id))
                    .fold(0u128, |acc, value| acc.saturating_add(value));
                (*subgraph_id, total)
            })
            .collect::<HashMap<_, _>>();
        let subgraph_total_count_by_id = self
            .subgraph_by_id
            .iter()
            .map(|(subgraph_id, subgraph)| {
                let total = subgraph
                    .node_ids
                    .iter()
                    .copied()
                    .map(|node_id| self.node_total_count_for_heat(node_id))
                    .fold(0u64, |acc, value| acc.saturating_add(value));
                (*subgraph_id, total)
            })
            .collect::<HashMap<_, _>>();
        let visible_display_total_ns_by_id = self.display_total_ns_by_display_id();
        let visible_heat_bounds = heat_bounds(&visible_display_total_ns_by_id);

        let mut candidates = self
            .expanded_subgraphs
            .iter()
            .copied()
            .filter(|subgraph_id| self.is_subgraph_visible(*subgraph_id))
            .filter(|subgraph_id| self.cluster_contains_tensor_op(*subgraph_id))
            .filter_map(|subgraph_id| {
                let subgraph = self.subgraph_by_id.get(&subgraph_id)?;
                let mut visible_member_count = 0usize;
                for node_id in &subgraph.node_ids {
                    let owner = self.visible_owner_for_node(*node_id);
                    let display_id = self.display_id_for_owner(owner)?;
                    if visible_nodes.contains(&display_id) {
                        visible_member_count = visible_member_count.saturating_add(1);
                    }
                }

                if visible_member_count == 0 {
                    return None;
                }

                let depth = self
                    .depth_by_subgraph
                    .get(&subgraph_id)
                    .copied()
                    .unwrap_or(0);
                Some(ClusterCandidate {
                    subgraph_id,
                    depth,
                    visible_member_count,
                    child_count: subgraph.child_ids.len(),
                    is_redundant_wrapper: self.is_redundant_wrapper_cluster(subgraph_id),
                })
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|left, right| {
            left.depth
                .cmp(&right.depth)
                .then(right.visible_member_count.cmp(&left.visible_member_count))
                .then(left.subgraph_id.cmp(&right.subgraph_id))
        });

        let mut clusters = Vec::new();
        let mut cluster_index_by_subgraph = HashMap::<usize, usize>::new();
        let mut cluster_by_index = HashMap::<usize, RenderedCluster>::new();

        for candidate in candidates {
            if clusters.len() >= max_visible_clusters {
                break;
            }

            if candidate.is_redundant_wrapper {
                continue;
            }

            let include_by_depth = candidate.depth <= ALWAYS_INCLUDE_CLUSTER_DEPTH;
            let include_by_size = candidate.visible_member_count >= min_cluster_visible_nodes
                || candidate.child_count >= 2;
            if !include_by_depth && !include_by_size {
                continue;
            }

            let subgraph_id = candidate.subgraph_id;
            let Some(subgraph) = self.subgraph_by_id.get(&subgraph_id) else {
                continue;
            };

            let mut members = HashSet::<u32>::new();
            for node_id in &subgraph.node_ids {
                let owner = self.visible_owner_for_node(*node_id);
                let Some(display_id) = self.display_id_for_owner(owner) else {
                    continue;
                };
                if visible_nodes.contains(&display_id) {
                    members.insert(display_id);
                }
            }

            if members.is_empty() {
                continue;
            }

            let mut member_nodes = members.into_iter().collect::<Vec<_>>();
            member_nodes.sort_unstable();
            let cluster_visible_total_ns = member_nodes.iter().fold(0u128, |acc, display_id| {
                acc.saturating_add(
                    visible_display_total_ns_by_id
                        .get(display_id)
                        .copied()
                        .unwrap_or(0),
                )
            });

            let mut cluster = Cluster::new(member_nodes).padding(18.0);

            if let Some(parent_subgraph_id) = self.visible_expanded_parent(subgraph_id)
                && let Some(parent_index) =
                    cluster_index_by_subgraph.get(&parent_subgraph_id).copied()
            {
                cluster = cluster.parent(parent_index);
            }

            let index = clusters.len();
            clusters.push(cluster);
            cluster_index_by_subgraph.insert(subgraph_id, index);
            let subgraph_total_ns = subgraph_total_ns_by_id
                .get(&subgraph_id)
                .copied()
                .unwrap_or(0);
            let subgraph_total_count = subgraph_total_count_by_id
                .get(&subgraph_id)
                .copied()
                .unwrap_or(0);
            let decorated_name = if subgraph_total_ns > 0 && subgraph_total_count > 0 {
                let avg_ns = subgraph_total_ns.saturating_div(u128::from(subgraph_total_count));
                format!(
                    "{} ({} x {})",
                    subgraph.name,
                    subgraph_total_count,
                    fmt_duration_short(avg_ns)
                )
            } else {
                subgraph.name.clone()
            };
            cluster_by_index.insert(
                index,
                RenderedCluster {
                    subgraph_id,
                    name: decorated_name,
                    heat_ratio: visible_heat_bounds.and_then(|(min_ns, max_ns)| {
                        if cluster_visible_total_ns == 0 {
                            None
                        } else {
                            Some(normalize_heat_ratio(
                                cluster_visible_total_ns,
                                min_ns,
                                max_ns,
                            ))
                        }
                    }),
                    has_timing: cluster_visible_total_ns > 0,
                },
            );
        }

        (clusters, cluster_by_index)
    }

    fn is_redundant_wrapper_cluster(&self, subgraph_id: usize) -> bool {
        let Some(subgraph) = self.subgraph_by_id.get(&subgraph_id) else {
            return false;
        };

        if subgraph.child_ids.len() != 1 {
            return false;
        }
        let child_id = subgraph.child_ids[0];
        let Some(child) = self.subgraph_by_id.get(&child_id) else {
            return false;
        };
        if subgraph.node_ids.len() != child.node_ids.len() {
            return false;
        }

        let child_nodes = child.node_ids.iter().copied().collect::<HashSet<_>>();
        for node_id in &subgraph.node_ids {
            if !child_nodes.contains(node_id) {
                return false;
            }
        }
        true
    }

    fn is_subgraph_visible(&self, subgraph_id: usize) -> bool {
        let mut current = self.parent_by_subgraph.get(&subgraph_id).copied();
        while let Some(parent_id) = current {
            if !self.expanded_subgraphs.contains(&parent_id) {
                return false;
            }
            current = self.parent_by_subgraph.get(&parent_id).copied();
        }

        true
    }

    fn cluster_contains_tensor_op(&self, subgraph_id: usize) -> bool {
        self.subgraph_contains_tensor_op
            .get(&subgraph_id)
            .copied()
            .unwrap_or(false)
    }

    fn node_total_ns_for_heat(&self, node_id: usize) -> u128 {
        self.node_total_ns_by_node_id
            .get(&node_id)
            .copied()
            .unwrap_or(0)
    }

    fn node_total_count_for_heat(&self, node_id: usize) -> u64 {
        self.node_total_count_by_node_id
            .get(&node_id)
            .copied()
            .unwrap_or(0)
    }

    fn node_timing_tooltip(&self, total_ns: u128, total_count: u64) -> String {
        if total_count == 0 || total_ns == 0 {
            return "No samples in current window".to_string();
        }
        let avg_ns = total_ns.saturating_div(u128::from(total_count));
        format!(
            "{} invocations, total {}, avg {}",
            total_count,
            fmt_duration_short(total_ns),
            fmt_duration_short(avg_ns)
        )
    }

    fn visible_expanded_parent(&self, subgraph_id: usize) -> Option<usize> {
        let mut current = self.parent_by_subgraph.get(&subgraph_id).copied();

        while let Some(parent_id) = current {
            if self.expanded_subgraphs.contains(&parent_id) && self.is_subgraph_visible(parent_id) {
                return Some(parent_id);
            }
            current = self.parent_by_subgraph.get(&parent_id).copied();
        }

        None
    }

    fn expand_subgraph_with_ancestors(&mut self, subgraph_id: usize) {
        let mut lineage = Vec::new();
        let mut current = Some(subgraph_id);

        while let Some(id) = current {
            lineage.push(id);
            current = self.parent_by_subgraph.get(&id).copied();
        }

        for id in lineage.into_iter().rev() {
            self.expanded_subgraphs.insert(id);
        }
    }

    fn collapse_subgraph_and_descendants(&mut self, subgraph_id: usize) {
        let mut stack = vec![subgraph_id];

        while let Some(current) = stack.pop() {
            self.expanded_subgraphs.remove(&current);
            self.forced_expand_shared_subgraphs.remove(&current);
            if let Some(subgraph) = self.subgraph_by_id.get(&current) {
                for child_id in &subgraph.child_ids {
                    stack.push(*child_id);
                }
            }
        }
    }

    fn heat_legend(&self) -> Container<'_, Message> {
        let totals_by_display_id = self.display_total_ns_by_display_id();
        let heat_range = heat_bounds(&totals_by_display_id);
        let (low_label, mid_label, high_label, has_heat) =
            if let Some((min_ns, max_ns)) = heat_range {
                let mid_ns = min_ns.saturating_add(max_ns).saturating_div(2);
                (
                    fmt_duration_short(min_ns),
                    fmt_duration_short(mid_ns),
                    fmt_duration_short(max_ns),
                    true,
                )
            } else {
                (
                    "0ns".to_string(),
                    "0ns".to_string(),
                    "0ns".to_string(),
                    false,
                )
            };
        let title = if has_heat {
            "Heat (total time)".to_string()
        } else {
            "Heat (waiting for traces)".to_string()
        };

        let legend_items = Row::new()
            .spacing(8)
            .align_y(iced::alignment::Vertical::Center)
            .push(Self::legend_chip(0.0, low_label))
            .push(Self::legend_chip(0.5, mid_label))
            .push(Self::legend_chip(1.0, high_label));

        Container::new(
            Row::new()
                .spacing(10)
                .align_y(iced::alignment::Vertical::Center)
                .push(text(title).size(12))
                .push(legend_items),
        )
        .padding([4, 8])
        .style(legend_container_style)
    }

    fn legend_chip(ratio: f32, label: String) -> Container<'static, Message> {
        let swatch = Container::new(Space::new(Length::Fixed(18.0), Length::Fixed(10.0)))
            .style(move |_theme| swatch_style(ratio));
        Container::new(
            Row::new()
                .spacing(6)
                .align_y(iced::alignment::Vertical::Center)
                .push(swatch)
                .push(text(label).size(11)),
        )
    }

    fn display_total_ns_by_display_id(&self) -> HashMap<u32, u128> {
        let mut totals_by_display_id = HashMap::<u32, u128>::new();
        for node in &self.source_graph.nodes {
            let node_total_ns = self.node_total_ns_for_heat(node.id);
            if node_total_ns == 0 {
                continue;
            }

            let owner = self.visible_owner_for_node(node.id);
            let Some(display_id) = self.display_id_for_owner(owner) else {
                continue;
            };

            let entry = totals_by_display_id.entry(display_id).or_insert(0);
            *entry = entry.saturating_add(node_total_ns);
        }
        totals_by_display_id
    }

    fn display_heat_ratio_by_display_id(&self) -> HashMap<u32, f32> {
        let totals_by_display_id = self.display_total_ns_by_display_id();
        if totals_by_display_id.is_empty() {
            return HashMap::new();
        }

        let Some((min_display_total_ns, max_display_total_ns)) = heat_bounds(&totals_by_display_id)
        else {
            return HashMap::new();
        };

        totals_by_display_id
            .into_iter()
            .map(|(display_id, total_ns)| {
                (
                    display_id,
                    normalize_heat_ratio(total_ns, min_display_total_ns, max_display_total_ns),
                )
            })
            .collect()
    }

    fn apply_live_snapshot(&mut self, snapshot: TensorOpSnapshot) -> bool {
        if snapshot.sequence < self.last_live_sequence {
            // Snapshot producer restarted; reset cumulative tracking and totals.
            self.last_cumulative_ns_by_node_id.clear();
            self.last_cumulative_count_by_node_id.clear();
            self.node_total_ns_by_node_id.clear();
            self.node_total_count_by_node_id.clear();
            self.last_live_sequence = 0;
        }
        if snapshot.sequence == self.last_live_sequence {
            return false;
        }

        self.last_live_sequence = snapshot.sequence;

        let mut current_cumulative_ns_by_node_id = HashMap::<usize, u128>::new();
        let mut current_cumulative_count_by_node_id = HashMap::<usize, u64>::new();

        for row in snapshot.rows {
            let mapped_to_node = row
                .node_stable_id
                .and_then(|stable_id| self.node_id_by_stable_id.get(&stable_id).copied())
                .or_else(|| {
                    row.key
                        .as_deref()
                        .and_then(parse_node_stable_id_from_key)
                        .and_then(|stable_id| self.node_id_by_stable_id.get(&stable_id).copied())
                });

            if let Some(node_id) = mapped_to_node {
                let ns_entry = current_cumulative_ns_by_node_id.entry(node_id).or_insert(0);
                *ns_entry = ns_entry.saturating_add(row.total_ns);
                let count_entry = current_cumulative_count_by_node_id
                    .entry(node_id)
                    .or_insert(0);
                *count_entry = count_entry.saturating_add(row.count);
            }
        }

        let mut node_deltas_ns = HashMap::<usize, u128>::new();
        let mut node_deltas_count = HashMap::<usize, u64>::new();
        for (node_id, cumulative_ns) in current_cumulative_ns_by_node_id {
            let previous_ns = self
                .last_cumulative_ns_by_node_id
                .get(&node_id)
                .copied()
                .unwrap_or(cumulative_ns);
            self.last_cumulative_ns_by_node_id
                .insert(node_id, cumulative_ns);

            let delta_ns = if cumulative_ns >= previous_ns {
                cumulative_ns.saturating_sub(previous_ns)
            } else {
                cumulative_ns
            };
            if delta_ns > 0 {
                node_deltas_ns.insert(node_id, delta_ns);
            }
        }

        for (node_id, cumulative_count) in current_cumulative_count_by_node_id {
            let previous_count = self
                .last_cumulative_count_by_node_id
                .get(&node_id)
                .copied()
                .unwrap_or(cumulative_count);
            self.last_cumulative_count_by_node_id
                .insert(node_id, cumulative_count);

            let delta_count = if cumulative_count >= previous_count {
                cumulative_count.saturating_sub(previous_count)
            } else {
                cumulative_count
            };
            if delta_count > 0 {
                node_deltas_count.insert(node_id, delta_count);
            }
        }

        for (node_id, delta_ns) in node_deltas_ns {
            let entry = self.node_total_ns_by_node_id.entry(node_id).or_insert(0);
            *entry = entry.saturating_add(delta_ns);
        }
        for (node_id, delta_count) in node_deltas_count {
            let entry = self.node_total_count_by_node_id.entry(node_id).or_insert(0);
            *entry = entry.saturating_add(delta_count);
        }

        true
    }
}

fn assign_display_ids(
    source_graph: &ArrowGraph,
    subgraph_by_id: &HashMap<usize, SubgraphInfo>,
) -> (HashMap<usize, u32>, HashMap<usize, u32>) {
    let mut display_id_by_node = HashMap::new();
    let mut display_id_by_subgraph = HashMap::new();

    let mut next_id: usize = 0;

    let mut node_ids = source_graph
        .nodes
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    node_ids.sort_unstable();
    node_ids.dedup();

    for node_id in node_ids {
        let Some(display_id) = next_display_id(&mut next_id) else {
            break;
        };
        display_id_by_node.insert(node_id, display_id);
    }

    let mut subgraph_ids = subgraph_by_id.keys().copied().collect::<Vec<_>>();
    subgraph_ids.sort_unstable();

    for subgraph_id in subgraph_ids {
        let Some(display_id) = next_display_id(&mut next_id) else {
            break;
        };
        display_id_by_subgraph.insert(subgraph_id, display_id);
    }

    (display_id_by_node, display_id_by_subgraph)
}

fn next_display_id(next_id: &mut usize) -> Option<u32> {
    let id = u32::try_from(*next_id).ok()?;
    *next_id = next_id.saturating_add(1);
    Some(id)
}

fn index_subgraphs(
    roots: &[ArrowSubGraph],
) -> (
    HashMap<usize, SubgraphInfo>,
    HashMap<usize, usize>,
    HashMap<usize, usize>,
    HashMap<usize, Vec<Membership>>,
) {
    let mut subgraph_by_id = HashMap::new();
    let mut parent_by_subgraph = HashMap::new();
    let mut depth_by_subgraph = HashMap::new();
    let mut node_memberships = HashMap::<usize, Vec<Membership>>::new();

    for root in roots {
        index_subgraph_recursive(
            root,
            None,
            0,
            &mut subgraph_by_id,
            &mut parent_by_subgraph,
            &mut depth_by_subgraph,
            &mut node_memberships,
        );
    }

    (
        subgraph_by_id,
        parent_by_subgraph,
        depth_by_subgraph,
        node_memberships,
    )
}

fn index_subgraph_recursive(
    subgraph: &ArrowSubGraph,
    parent: Option<usize>,
    depth: usize,
    subgraph_by_id: &mut HashMap<usize, SubgraphInfo>,
    parent_by_subgraph: &mut HashMap<usize, usize>,
    depth_by_subgraph: &mut HashMap<usize, usize>,
    node_memberships: &mut HashMap<usize, Vec<Membership>>,
) {
    if let Some(parent_id) = parent {
        parent_by_subgraph.insert(subgraph.id, parent_id);
    }

    depth_by_subgraph.insert(subgraph.id, depth);

    for node_id in &subgraph.node_ids {
        node_memberships
            .entry(*node_id)
            .or_default()
            .push(Membership {
                subgraph_id: subgraph.id,
                depth,
            });
    }

    let child_ids = subgraph
        .children
        .iter()
        .map(|child| child.id)
        .collect::<Vec<_>>();

    subgraph_by_id.insert(
        subgraph.id,
        SubgraphInfo {
            name: subgraph.name.clone(),
            node_ids: subgraph.node_ids.clone(),
            child_ids,
        },
    );

    for child in &subgraph.children {
        index_subgraph_recursive(
            child,
            Some(subgraph.id),
            depth.saturating_add(1),
            subgraph_by_id,
            parent_by_subgraph,
            depth_by_subgraph,
            node_memberships,
        );
    }
}

fn build_shared_path_dedup(source_graph: &ArrowGraph) -> SharedPathDedup {
    let flattened = flatten_subgraphs(&source_graph.subgraphs);
    if flattened.is_empty() {
        return SharedPathDedup::default();
    }

    let node_labels = source_graph
        .nodes
        .iter()
        .map(|node| (node.id, node.label.as_str()))
        .collect::<HashMap<_, _>>();

    let mut signatures = HashMap::<usize, String>::new();
    for subgraph in &flattened {
        let _ = signature_for_subgraph(subgraph, &node_labels, &mut signatures);
    }

    let mut signature_counts = HashMap::<String, usize>::new();
    for subgraph in &flattened {
        if let Some(signature) = signatures.get(&subgraph.id) {
            let count = signature_counts.entry(signature.clone()).or_insert(0);
            *count = count.saturating_add(1);
        }
    }

    let mut parent_of = HashMap::<usize, usize>::new();
    for subgraph in &flattened {
        for child in &subgraph.children {
            parent_of.insert(child.id, subgraph.id);
        }
    }

    let mut first_seen_by_signature = HashMap::<String, usize>::new();
    let mut collapse_candidates = HashSet::<usize>::new();

    for subgraph in &flattened {
        let Some(signature) = signatures.get(&subgraph.id) else {
            continue;
        };
        let count = signature_counts.get(signature).copied().unwrap_or(0);
        if count <= 1 || !dedup_eligible_subgraph(subgraph) {
            continue;
        }
        if first_seen_by_signature.contains_key(signature) {
            collapse_candidates.insert(subgraph.id);
        } else {
            first_seen_by_signature.insert(signature.clone(), subgraph.id);
        }
    }

    let mut forced_collapsed_subgraphs = HashSet::<usize>::new();
    for candidate in &collapse_candidates {
        let mut current = *candidate;
        let mut has_collapsed_ancestor = false;
        while let Some(parent_id) = parent_of.get(&current).copied() {
            if collapse_candidates.contains(&parent_id) {
                has_collapsed_ancestor = true;
                break;
            }
            current = parent_id;
        }
        if !has_collapsed_ancestor {
            forced_collapsed_subgraphs.insert(*candidate);
        }
    }

    SharedPathDedup {
        forced_collapsed_subgraphs,
    }
}

fn build_subgraph_tensor_op_presence(
    subgraph_by_id: &HashMap<usize, SubgraphInfo>,
    tensor_op_node_ids: &HashSet<usize>,
) -> HashMap<usize, bool> {
    subgraph_by_id
        .iter()
        .map(|(subgraph_id, subgraph)| {
            let contains = subgraph
                .node_ids
                .iter()
                .any(|node_id| tensor_op_node_ids.contains(node_id));
            (*subgraph_id, contains)
        })
        .collect()
}

fn flatten_subgraphs(roots: &[ArrowSubGraph]) -> Vec<&ArrowSubGraph> {
    fn walk<'a>(subgraph: &'a ArrowSubGraph, out: &mut Vec<&'a ArrowSubGraph>) {
        out.push(subgraph);
        for child in &subgraph.children {
            walk(child, out);
        }
    }

    let mut out = Vec::new();
    for root in roots {
        walk(root, &mut out);
    }
    out
}

fn dedup_eligible_subgraph(subgraph: &ArrowSubGraph) -> bool {
    if subgraph.node_ids.len() < MIN_DEDUP_SUBGRAPH_NODES {
        return false;
    }
    if !subgraph.children.is_empty() {
        return true;
    }
    subgraph.node_ids.len() > 1
}

fn signature_for_subgraph(
    subgraph: &ArrowSubGraph,
    node_labels: &HashMap<usize, &str>,
    memo: &mut HashMap<usize, String>,
) -> String {
    if let Some(signature) = memo.get(&subgraph.id) {
        return signature.clone();
    }

    let child_node_ids = subgraph
        .children
        .iter()
        .flat_map(|child| child.node_ids.iter().copied())
        .collect::<HashSet<_>>();
    let mut direct_labels = subgraph
        .node_ids
        .iter()
        .filter(|node_id| !child_node_ids.contains(node_id))
        .filter_map(|node_id| node_labels.get(node_id).copied())
        .map(str::to_string)
        .collect::<Vec<_>>();
    direct_labels.sort();

    let child_signatures = subgraph
        .children
        .iter()
        .map(|child| signature_for_subgraph(child, node_labels, memo))
        .collect::<Vec<_>>();

    let signature = format!(
        "name={}|custom={}|direct={:?}|children={:?}",
        subgraph.name, subgraph.is_custom, direct_labels, child_signatures
    );
    memo.insert(subgraph.id, signature.clone());
    signature
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let char_count = label.chars().count();
    if char_count <= max_chars {
        return label.to_string();
    }

    let keep = max_chars.saturating_sub(3);
    let mut out = String::new();
    for c in label.chars().take(keep) {
        out.push(c);
    }
    out.push_str("...");
    out
}

fn node_size(_node_id: u32) -> (f64, f64) {
    (NODE_WIDTH, NODE_HEIGHT)
}

fn edge_color(_index: usize) -> (Color, Color) {
    (Color::BLACK, Color::BLACK)
}

fn hidden_cluster_outline(_index: usize) -> Color {
    Color::TRANSPARENT
}

fn directional_marker(endpoint: EdgeEndpoint) -> &'static str {
    let angle = endpoint.angle_radians();
    if (-std::f32::consts::FRAC_PI_4..std::f32::consts::FRAC_PI_4).contains(&angle) {
        ">"
    } else if (std::f32::consts::FRAC_PI_4..3.0 * std::f32::consts::FRAC_PI_4).contains(&angle) {
        "v"
    } else if (-3.0 * std::f32::consts::FRAC_PI_4..-std::f32::consts::FRAC_PI_4).contains(&angle) {
        "^"
    } else {
        "<"
    }
}

fn heat_node_button_style(
    status: iced::widget::button::Status,
    heat_ratio: Option<f32>,
    is_collapsed: bool,
) -> iced::widget::button::Style {
    let mut base_bg = if let Some(ratio) = heat_ratio {
        heat_color(ratio)
    } else {
        match status {
            iced::widget::button::Status::Hovered if is_collapsed => {
                Color::from_rgb8(245, 245, 245)
            }
            iced::widget::button::Status::Pressed if is_collapsed => {
                Color::from_rgb8(236, 236, 236)
            }
            _ => Color::WHITE,
        }
    };
    if heat_ratio.is_some() {
        base_bg = match status {
            iced::widget::button::Status::Hovered => lighten(base_bg, 0.06),
            iced::widget::button::Status::Pressed => darken(base_bg, 0.08),
            _ => base_bg,
        };
    }

    iced::widget::button::Style {
        background: Some(Background::Color(base_bg)),
        text_color: Color::BLACK,
        border: iced::border::rounded(8).color(Color::BLACK).width(1.0),
        shadow: if is_collapsed {
            Shadow {
                offset: iced::Vector { x: 0., y: 1. },
                color: Color::from_rgba(0., 0., 0., 0.6),
                blur_radius: 3.,
            }
        } else {
            Shadow::default()
        },
    }
}

fn heat_color(heat_ratio: f32) -> Color {
    let t = heat_ratio.clamp(0.0, 1.0);
    let green = Color::from_rgb8(200, 230, 201);
    let yellow = Color::from_rgb8(255, 236, 179);
    let red = Color::from_rgb8(255, 138, 128);

    if t < 0.5 {
        lerp_color(green, yellow, t / 0.5)
    } else {
        lerp_color(yellow, red, (t - 0.5) / 0.5)
    }
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let tt = t.clamp(0.0, 1.0);
    Color::from_rgba(
        a.r + (b.r - a.r) * tt,
        a.g + (b.g - a.g) * tt,
        a.b + (b.b - a.b) * tt,
        a.a + (b.a - a.a) * tt,
    )
}

fn lighten(color: Color, amount: f32) -> Color {
    lerp_color(color, Color::WHITE, amount)
}

fn darken(color: Color, amount: f32) -> Color {
    lerp_color(color, Color::BLACK, amount)
}

fn legend_container_style(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgb8(247, 247, 247))),
        border: Border::default()
            .rounded(6)
            .width(1)
            .color(Color::from_rgb8(205, 205, 205)),
        text_color: Some(Color::from_rgb8(40, 40, 40)),
        ..Default::default()
    }
}

fn swatch_style(ratio: f32) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(heat_color(ratio))),
        border: Border::default()
            .rounded(3)
            .width(1)
            .color(Color::from_rgb8(80, 80, 80)),
        ..Default::default()
    }
}

fn parse_node_stable_id_from_key(key: &str) -> Option<u64> {
    let (_, tail) = key.rsplit_once("[node_stable_id=")?;
    let value = tail.strip_suffix(']')?;
    value.parse::<u64>().ok().filter(|stable_id| *stable_id > 0)
}

fn normalize_op_name(raw: &str) -> String {
    let mut out = String::new();
    let mut prev_was_sep = true;

    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            if ch.is_ascii_uppercase() {
                if !prev_was_sep && !out.ends_with('_') {
                    out.push('_');
                }
                out.push(ch.to_ascii_lowercase());
            } else {
                out.push(ch.to_ascii_lowercase());
            }
            prev_was_sep = false;
        } else if !prev_was_sep {
            out.push('_');
            prev_was_sep = true;
        }
    }

    while out.ends_with('_') {
        out.pop();
    }

    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn is_tensor_op_label(raw: &str) -> bool {
    let normalized = normalize_op_name(raw);
    if normalized.starts_with("flatten_")
        || normalized.starts_with("index_add_")
        || normalized.starts_with("index_select_")
        || normalized.starts_with("narrow_")
        || normalized.starts_with("narrow_dyn_")
        || normalized.starts_with("narrow_dyn_start_")
        || normalized.starts_with("squeeze_")
        || normalized.starts_with("sum_dim_")
        || normalized.starts_with("transpose_")
        || normalized.starts_with("unsqueeze_")
    {
        return true;
    }

    matches!(
        normalized.as_str(),
        "argmax_dim"
            | "argmin_dim"
            | "broadcast_add"
            | "broadcast_mul"
            | "cast_flatten_to_vec1"
            | "cast_flatten_to_vec1_pair"
            | "cast_like"
            | "cast_to_vec2"
            | "cast_to_vec2_pair"
            | "clamp"
            | "contiguous"
            | "conv"
            | "cumsum"
            | "dims2"
            | "dims3"
            | "embedding"
            | "exp"
            | "expand"
            | "flatten_all_to_vec1"
            | "flatten_prefix2"
            | "from_vec1_on_device"
            | "from_vec_col_on_device"
            | "gather"
            | "group_top_k_assignments"
            | "into_inner"
            | "into_inner_result"
            | "log_softmax"
            | "matmul"
            | "max_dim"
            | "mean_dim"
            | "min_dim"
            | "q_mat_mul"
            | "q_mat_mul_from_q_tensor"
            | "remap_indices"
            | "reshape"
            | "residual_add"
            | "rms_norm"
            | "sigmoid"
            | "silu"
            | "softmax"
            | "tensor_device_info"
            | "to_device"
            | "to_dtype"
            | "to_vec"
            | "to_vec1"
            | "to_vec2"
            | "topk_from_logits"
            | "try_typed"
            | "unflatten_last"
            | "var_dim"
    )
}

fn parse_positive_env_usize(name: &str, default_value: usize) -> usize {
    if let Ok(raw) = std::env::var(name)
        && let Ok(parsed) = raw.parse::<usize>()
        && parsed > 0
    {
        return parsed;
    }
    default_value
}

fn cluster_container_style(
    _theme: &iced::Theme,
    heat_ratio: Option<f32>,
    has_timing: bool,
) -> iced::widget::container::Style {
    let border_color = heat_ratio.map(heat_color).unwrap_or(Color::BLACK);
    let border_width = if has_timing { 2.0 } else { 1.0 };
    iced::widget::container::Style {
        border: iced::border::rounded(8)
            .color(border_color)
            .width(border_width),
        shadow: Default::default(),
        ..iced::widget::container::Style::default()
    }
}

fn cluster_title_style(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgba8(255, 255, 255, 0.8))),
        ..iced::widget::container::Style::default()
    }
}

fn cluster_collapse_button_style(
    _theme: &iced::Theme,
    status: iced::widget::button::Status,
    heat_ratio: Option<f32>,
) -> iced::widget::button::Style {
    let text_color = if let Some(ratio) = heat_ratio {
        match status {
            iced::widget::button::Status::Hovered => darken(heat_color(ratio), 0.1),
            iced::widget::button::Status::Pressed => darken(heat_color(ratio), 0.2),
            _ => heat_color(ratio),
        }
    } else {
        Color::BLACK
    };

    iced::widget::button::Style {
        background: None,
        text_color,
        border: Border::default(),
        shadow: Shadow::default(),
    }
}

fn edge_label_container_style(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgba8(255, 255, 255, 0.8))),
        border: Border {
            width: 0.0,
            color: Color::TRANSPARENT,
            ..Border::default()
        },
        ..iced::widget::container::Style::default()
    }
}

fn chat_overlay_container_style(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgba8(252, 252, 252, 0.95))),
        border: Border::default()
            .rounded(10)
            .width(1)
            .color(Color::from_rgb8(180, 180, 180)),
        text_color: Some(Color::from_rgb8(25, 25, 25)),
        shadow: Shadow {
            color: Color::from_rgba(0.0, 0.0, 0.0, 0.2),
            offset: iced::Vector::new(0.0, 1.0),
            blur_radius: 4.0,
        },
        ..iced::widget::container::Style::default()
    }
}

fn node_tooltip_style(_theme: &iced::Theme) -> iced::widget::container::Style {
    iced::widget::container::Style {
        background: Some(Background::Color(Color::from_rgba8(250, 250, 250, 0.94))),
        border: Border::default()
            .rounded(6)
            .width(1)
            .color(Color::from_rgb8(160, 160, 160)),
        text_color: Some(Color::from_rgb8(25, 25, 25)),
        ..iced::widget::container::Style::default()
    }
}

fn heat_bounds(totals_by_display_id: &HashMap<u32, u128>) -> Option<(u128, u128)> {
    let min_total = totals_by_display_id.values().copied().min()?;
    let max_total = totals_by_display_id.values().copied().max()?;
    Some((min_total, max_total))
}

fn normalize_heat_ratio(total_ns: u128, min_ns: u128, max_ns: u128) -> f32 {
    if max_ns <= min_ns {
        return 0.0;
    }
    let numer = total_ns.saturating_sub(min_ns);
    let denom = max_ns.saturating_sub(min_ns);
    (numer as f64 / denom as f64).clamp(0.0, 1.0) as f32
}

fn fmt_duration_short(ns: u128) -> String {
    if ns >= 1_000_000_000 {
        let s = ns as f64 / 1_000_000_000.0;
        if s >= 100.0 {
            format!("{s:.0}s")
        } else if s >= 10.0 {
            format!("{s:.1}s")
        } else {
            format!("{s:.2}s")
        }
    } else if ns >= 1_000_000 {
        let ms = ns as f64 / 1_000_000.0;
        if ms >= 100.0 {
            format!("{ms:.0}ms")
        } else if ms >= 10.0 {
            format!("{ms:.1}ms")
        } else {
            format!("{ms:.2}ms")
        }
    } else if ns >= 1_000 {
        let us = ns as f64 / 1_000.0;
        if us >= 100.0 {
            format!("{us:.0}us")
        } else if us >= 10.0 {
            format!("{us:.1}us")
        } else {
            format!("{us:.2}us")
        }
    } else {
        format!("{ns}ns")
    }
}
