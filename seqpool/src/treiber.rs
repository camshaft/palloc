//! ABA-safe lock-free Treiber stack using packed (index, generation) values.
//!
//! Each entry in the stack is identified by a 32-bit index into a pre-allocated
//! slot array, paired with a 32-bit generation counter to prevent ABA problems.
//! The two are packed into a single `AtomicU64` for lock-free CAS.
//!
//! The "next" pointers are stored externally in a parallel array of `AtomicU32`,
//! so slot data stays untouched.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Sentinel value indicating an empty/null link.
const NULL: u32 = u32::MAX;

/// A packed head value: low 32 bits = index, high 32 bits = generation.
#[inline(always)]
const fn pack(index: u32, generation: u32) -> u64 {
    (index as u64) | ((generation as u64) << 32)
}

#[inline(always)]
const fn unpack(val: u64) -> (u32, u32) {
    (val as u32, (val >> 32) as u32)
}

/// Empty head value (null index, generation 0).
const EMPTY: u64 = pack(NULL, 0);

/// A lock-free stack of slot indices.
///
/// The stack itself only stores the head (packed index + generation).
/// The "next" pointers for each index live in the external `next` array
/// that is passed to every push/pop call.
pub struct TreiberStack {
    head: AtomicU64,
}

impl TreiberStack {
    pub const fn new() -> Self {
        Self {
            head: AtomicU64::new(EMPTY),
        }
    }

    /// Push `index` onto the stack.
    ///
    /// # Safety
    ///
    /// - `index` must be a valid index into `next`.
    /// - `index` must not already be on the stack.
    #[inline]
    pub unsafe fn push(&self, next: &[AtomicU32], index: u32) {
        debug_assert!((index as usize) < next.len());

        let mut head = self.head.load(Ordering::Relaxed);
        loop {
            let (head_idx, head_gen) = unpack(head);

            // Point our node's next at the current head.
            next[index as usize].store(head_idx, Ordering::Relaxed);

            // Try to swing head to point at our node, bumping generation.
            match self.head.compare_exchange_weak(
                head,
                pack(index, head_gen.wrapping_add(1)),
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => head = actual,
            }
        }
    }

    /// Pop an index from the stack, or return `None` if empty.
    #[inline]
    pub fn pop(&self, next: &[AtomicU32]) -> Option<u32> {
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            let (head_idx, head_gen) = unpack(head);
            if head_idx == NULL {
                return None;
            }

            let next_idx = next[head_idx as usize].load(Ordering::Relaxed);

            match self.head.compare_exchange_weak(
                head,
                pack(next_idx, head_gen.wrapping_add(1)),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(head_idx),
                Err(actual) => head = actual,
            }
        }
    }

    /// Pop up to `batch_size` indices at once, returning the count popped.
    ///
    /// Popped indices are written into `out[0..returned_count]`.
    #[inline]
    pub fn pop_batch(&self, next: &[AtomicU32], out: &mut [u32]) -> usize {
        if out.is_empty() {
            return 0;
        }

        // Pop one at a time. We could do a multi-pop by chasing `next` pointers
        // within a single CAS, but that reads more cache lines under contention.
        // For now keep it simple — the MPSC remote queue is the fast path anyway.
        let mut count = 0;
        for slot in out.iter_mut() {
            match self.pop(next) {
                Some(idx) => {
                    *slot = idx;
                    count += 1;
                }
                None => break,
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_single() {
        let stack = TreiberStack::new();
        let next: Vec<AtomicU32> = (0..4).map(|_| AtomicU32::new(NULL)).collect();

        assert_eq!(stack.pop(&next), None);

        unsafe { stack.push(&next, 2) };
        unsafe { stack.push(&next, 0) };
        unsafe { stack.push(&next, 3) };

        // LIFO order
        assert_eq!(stack.pop(&next), Some(3));
        assert_eq!(stack.pop(&next), Some(0));
        assert_eq!(stack.pop(&next), Some(2));
        assert_eq!(stack.pop(&next), None);
    }

    #[test]
    fn push_pop_batch() {
        let stack = TreiberStack::new();
        let next: Vec<AtomicU32> = (0..8).map(|_| AtomicU32::new(NULL)).collect();

        for i in 0..8u32 {
            unsafe { stack.push(&next, i) };
        }

        let mut buf = [0u32; 4];
        let n = stack.pop_batch(&next, &mut buf);
        assert_eq!(n, 4);

        let n2 = stack.pop_batch(&next, &mut buf);
        assert_eq!(n2, 4);

        assert_eq!(stack.pop(&next), None);
    }

    #[test]
    fn concurrent_push_pop() {
        use std::{sync::Arc, thread};

        const N: u32 = 1000;
        let next: Arc<Vec<AtomicU32>> = Arc::new((0..N).map(|_| AtomicU32::new(NULL)).collect());
        let stack = Arc::new(TreiberStack::new());

        // Push all indices from multiple threads
        let mut handles = vec![];
        for chunk_start in (0..N).step_by(100) {
            let stack = Arc::clone(&stack);
            let next = Arc::clone(&next);
            handles.push(thread::spawn(move || {
                for i in chunk_start..(chunk_start + 100).min(N) {
                    unsafe { stack.push(&next, i) };
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Pop all — we should get exactly N unique indices
        let mut seen = vec![false; N as usize];
        let mut count = 0;
        while let Some(idx) = stack.pop(&next) {
            assert!(!seen[idx as usize], "duplicate index {idx}");
            seen[idx as usize] = true;
            count += 1;
        }
        assert_eq!(count, N);
    }
}
