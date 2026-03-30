use std::any::type_name;
use std::cell::Cell;

thread_local! {
    static RUN_CALL_DEPTH: Cell<u32> = const { Cell::new(0) };
    static RUN_NODE_SEQ: Cell<u64> = const { Cell::new(0) };
}

struct RunCallGuard;

impl RunCallGuard {
    fn enter() -> Self {
        RUN_CALL_DEPTH.with(|depth| {
            let current = depth.get();
            if current == 0 {
                RUN_NODE_SEQ.with(|seq| seq.set(0));
            }
            depth.set(current.saturating_add(1));
        });
        Self
    }
}

impl Drop for RunCallGuard {
    fn drop(&mut self) {
        RUN_CALL_DEPTH.with(|depth| {
            let current = depth.get();
            if current > 0 {
                depth.set(current - 1);
            }
        });
    }
}

fn next_node_seq() -> u64 {
    RUN_NODE_SEQ.with(|seq| {
        let next = seq.get().saturating_add(1);
        seq.set(next);
        next
    })
}

pub fn trace_forward<Node, Out>(f: impl FnOnce() -> Out) -> Out {
    let _guard = RunCallGuard::enter();
    let node_seq = next_node_seq();
    let _span = tracing::trace_span!(
        "arrow_node",
        arrow_node_id = node_seq,
        arrow_node_stable_id = node_seq,
        arrow_node_type = type_name::<Node>()
    )
    .entered();
    f()
}
