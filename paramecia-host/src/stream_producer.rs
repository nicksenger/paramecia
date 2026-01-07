//! Custom StreamProducer and StreamConsumer for tokio mpsc channels

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use wasmtime::StoreContextMut;
use wasmtime::component::{
    Destination, Source, StreamConsumer, StreamProducer, StreamResult, VecBuffer,
};

/// A StreamProducer that pulls from a tokio mpsc Receiver
pub struct MpscStreamProducer<T> {
    rx: tokio::sync::mpsc::Receiver<T>,
}

impl<T> MpscStreamProducer<T> {
    pub fn new(rx: tokio::sync::mpsc::Receiver<T>) -> Self {
        Self { rx }
    }
}

impl<T, D> StreamProducer<D> for MpscStreamProducer<T>
where
    T: Send + Sync + 'static,
{
    type Item = T;
    type Buffer = VecBuffer<T>;

    fn poll_produce<'a>(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        mut store: StoreContextMut<'a, D>,
        mut dst: Destination<'a, Self::Item, Self::Buffer>,
        _finish: bool,
    ) -> Poll<wasmtime::Result<StreamResult>> {
        // Get how many items the guest wants (or default to 32)
        let count = dst.remaining(&mut store).unwrap_or(32);

        // Handle 0-length reads
        if count == 0 {
            return Poll::Ready(Ok(StreamResult::Completed));
        }

        // Try to fill buffer with up to `count` items
        let mut buf = Vec::new();
        loop {
            // Try to receive from channel
            match self.rx.poll_recv(cx) {
                Poll::Ready(Some(item)) => {
                    buf.push(item);
                    // If we've filled the requested count, return what we have
                    if buf.len() >= count {
                        dst.set_buffer(buf.into());
                        return Poll::Ready(Ok(StreamResult::Completed));
                    }
                    // Otherwise continue trying to get more items (non-blocking)
                }
                Poll::Ready(None) => {
                    // Channel closed
                    if !buf.is_empty() {
                        // Return any buffered items first
                        dst.set_buffer(buf.into());
                    }
                    return Poll::Ready(Ok(StreamResult::Dropped));
                }
                Poll::Pending => {
                    // No more items immediately available
                    if !buf.is_empty() {
                        // Return what we got so far
                        dst.set_buffer(buf.into());
                        return Poll::Ready(Ok(StreamResult::Completed));
                    } else {
                        // Nothing ready yet, tell runtime to wait
                        return Poll::Pending;
                    }
                }
            }
        }
    }
}

/// A StreamConsumer that reads from a Source and forwards items to a tokio mpsc Sender.
/// Used for consuming guest-provided input streams in the host.
///
/// Items read from the WIT source are buffered internally when the mpsc channel is full,
/// ensuring no data loss under backpressure.
pub struct MpscStreamConsumer<T> {
    tx: tokio::sync::mpsc::Sender<T>,
    pending: VecDeque<T>,
}

impl<T> MpscStreamConsumer<T> {
    pub fn new(tx: tokio::sync::mpsc::Sender<T>) -> Self {
        Self {
            tx,
            pending: VecDeque::new(),
        }
    }
}

impl<T, D> StreamConsumer<D> for MpscStreamConsumer<T>
where
    T: wasmtime::component::Lift + Send + Sync + Unpin + 'static,
    D: 'static,
{
    type Item = T;

    fn poll_consume(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        mut store: StoreContextMut<D>,
        mut source: Source<'_, Self::Item>,
        finish: bool,
    ) -> Poll<wasmtime::Result<StreamResult>> {
        let this = self.get_mut();

        // First, drain any previously buffered items
        while let Some(item) = this.pending.pop_front() {
            match this.tx.try_send(item) {
                Ok(()) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Full(item)) => {
                    // Still full — put it back; self-wake so the runtime re-polls
                    // us once the adapter task has drained some items.
                    this.pending.push_front(item);
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    return Poll::Ready(Ok(StreamResult::Dropped));
                }
            }
        }

        // Read available items from the source
        let remaining = source.remaining(&mut store);
        if remaining > 0 {
            let mut buf: Vec<T> = Vec::with_capacity(remaining);
            source.read(&mut store, &mut buf)?;

            // Forward items to the mpsc channel
            let mut iter = buf.into_iter();
            for item in iter.by_ref() {
                match this.tx.try_send(item) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Full(item)) => {
                        // Channel full — buffer this item and all remaining;
                        // self-wake so the runtime re-polls us.
                        this.pending.push_back(item);
                        this.pending.extend(iter);
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        return Poll::Ready(Ok(StreamResult::Dropped));
                    }
                }
            }
            Poll::Ready(Ok(StreamResult::Completed))
        } else if finish {
            // Stream ended — signal cancelled so the writer gets notified
            Poll::Ready(Ok(StreamResult::Cancelled))
        } else {
            // No items yet, will be called again
            Poll::Ready(Ok(StreamResult::Completed))
        }
    }
}
