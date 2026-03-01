//! Lock-free MPSC (multi-producer, single-consumer) queue of slot indices.
//!
//! This is a faithful port of snmalloc's `FreeListMPSCQ`.
//!
//! ## Design
//!
//! Producers enqueue by atomically swapping `back` and linking the old back
//! to the new element. The consumer walks from `front` using `back` as a
//! bound — it never modifies `back`, avoiding races with producers.
//!
//! The element at `back` is always kept as a sentinel; it's consumed on the
//! *next* drain when a newer element takes its place as `back`. This means
//! one slot per queue is always "in transit", but ensures lock-freedom
//! without any risk of losing items.

use core::{
    ops::ControlFlow,
    sync::atomic::{AtomicU32, Ordering},
};
use crossbeam_utils::CachePadded;

/// Sentinel value indicating null/empty.
const NULL: u32 = u32::MAX;

/// A lock-free MPSC queue of slot indices.
///
/// Each field on its own cache line to avoid false sharing.
pub struct MpscQueue {
    /// Producers swap this atomically to enqueue.
    back: CachePadded<AtomicU32>,

    /// Consumer reads from here. Only touched by the owning CPU.
    front: CachePadded<AtomicU32>,

    /// Approximate length. Incremented on enqueue, decremented on drain.
    /// Not perfectly synchronized — just a heuristic for capacity decisions.
    len: CachePadded<AtomicU32>,
}

impl MpscQueue {
    pub const fn new() -> Self {
        Self {
            back: CachePadded::new(AtomicU32::new(NULL)),
            front: CachePadded::new(AtomicU32::new(NULL)),
            len: CachePadded::new(AtomicU32::new(0)),
        }
    }

    /// Approximate number of items in the queue.
    #[inline]
    pub fn approx_len(&self) -> u32 {
        self.len.load(Ordering::Relaxed)
    }

    /// Enqueue a slot index. Called by any thread (producer).
    ///
    /// Single atomic exchange — minimum synchronization.
    ///
    /// # Safety
    ///
    /// - `index` must be a valid index into `next`.
    /// - `index` must not already be on any queue or stack.
    #[inline]
    pub unsafe fn enqueue(&self, next: &[AtomicU32], index: u32) {
        debug_assert!((index as usize) < next.len());

        // Null-terminate our node.
        next[index as usize].store(NULL, Ordering::Relaxed);

        // Atomically place ourselves as the new back.
        // acq_rel: release so NULL in next is visible; acquire so we
        // don't race with the other thread's NULL init of next.
        let prev = self.back.swap(index, Ordering::AcqRel);

        self.len.fetch_add(1, Ordering::Relaxed);

        if prev != NULL {
            // Link the previous back to us.
            next[prev as usize].store(index, Ordering::Release);
        } else {
            // Queue was empty — we are the new front too.
            self.front.store(index, Ordering::Release);
        }
    }

    /// Drain items from the queue. Called by the owning CPU only (consumer).
    ///
    /// Walks from `front` toward `back`, using `back` as a bound.
    /// The element at `back` is NOT consumed (it's the sentinel).
    /// Calls `cb` for each consumed index. If `cb` returns `false`, stops
    /// early (the element was still consumed). Returns the number consumed.
    ///
    /// This matches snmalloc's `FreeListMPSCQ::dequeue` — we never modify
    /// `back`, avoiding races with concurrent producers.
    #[inline]
    pub fn drain(&self, next: &[AtomicU32], mut cb: impl FnMut(u32) -> ControlFlow<()>) -> usize {
        let mut curr = self.front.load(Ordering::Acquire);
        if curr == NULL {
            return 0;
        }

        // Read back as a bound. We won't process past this.
        let bound = self.back.load(Ordering::Relaxed);

        let mut count = 0;
        while curr != bound {
            let next_idx = next[curr as usize].load(Ordering::Acquire);

            // If next is NULL, the enqueuer hasn't finished linking yet.
            // Stop here; we'll pick up the rest on the next drain.
            if next_idx == NULL {
                break;
            }

            // Consume curr. If cb returns false, stop early.
            let cf = cb(curr);
            count += 1;
            curr = next_idx;
            if cf.is_break() {
                break;
            }
        }

        // Update front to point at the last unprocessed element (the sentinel).
        self.front.store(curr, Ordering::Release);
        if count > 0 {
            self.len.fetch_sub(count as u32, Ordering::Relaxed);
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_thread_enqueue_drain() {
        let queue = MpscQueue::new();
        let next: Vec<AtomicU32> = (0..8).map(|_| AtomicU32::new(NULL)).collect();

        // Empty drain
        let mut items = vec![];
        assert_eq!(
            queue.drain(&next, |i| {
                items.push(i);
                ControlFlow::Continue(())
            }),
            0,
        );

        // Enqueue 3 items
        unsafe {
            queue.enqueue(&next, 3);
            queue.enqueue(&next, 1);
            queue.enqueue(&next, 5);
        }

        // Drain: should get first 2 (3 and 1). Item 5 is the sentinel at back.
        items.clear();
        let n = queue.drain(&next, |i| {
            items.push(i);
            ControlFlow::Continue(())
        });
        assert_eq!(n, 2);
        assert_eq!(items, vec![3, 1]);

        // Enqueue one more so 5 is no longer the sentinel.
        unsafe {
            queue.enqueue(&next, 7);
        }

        // Now drain gets 5 (old sentinel), 7 is the new sentinel.
        items.clear();
        let n = queue.drain(&next, |i| {
            items.push(i);
            ControlFlow::Continue(())
        });
        assert_eq!(n, 1);
        assert_eq!(items, vec![5]);

        // Enqueue another to free 7.
        unsafe {
            queue.enqueue(&next, 2);
        }
        items.clear();
        queue.drain(&next, |i| {
            items.push(i);
            ControlFlow::Continue(())
        });
        assert_eq!(items, vec![7]);
    }

    #[test]
    fn concurrent_enqueue_drain() {
        use std::{sync::Arc, thread};

        // We need N+1 indices because one is always the sentinel.
        const N: u32 = 10_000;
        const TOTAL: u32 = N + 1;
        let next: Arc<Vec<AtomicU32>> =
            Arc::new((0..TOTAL).map(|_| AtomicU32::new(NULL)).collect());
        let queue = Arc::new(MpscQueue::new());

        // Multiple producer threads
        let mut handles = vec![];
        let threads = 4u32;
        let per_thread = N / threads;

        for t in 0..threads {
            let queue = Arc::clone(&queue);
            let next = Arc::clone(&next);
            handles.push(thread::spawn(move || {
                let start = t * per_thread;
                let end = if t == threads - 1 {
                    N
                } else {
                    start + per_thread
                };
                for i in start..end {
                    unsafe { queue.enqueue(&next, i) };
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Enqueue one final item to release the sentinel.
        unsafe { queue.enqueue(&next, N) };

        // Single consumer drains
        let mut seen = vec![false; TOTAL as usize];
        let mut count = 0;
        // May need multiple drains if linking is slow.
        for _ in 0..100 {
            queue.drain(&next, |idx| {
                assert!(!seen[idx as usize], "duplicate index {idx}");
                seen[idx as usize] = true;
                count += 1;
                ControlFlow::Continue(())
            });
            if count >= N {
                break;
            }
        }
        // We should have all N items (the final sentinel N is not consumed).
        assert_eq!(count, N);
    }
}
