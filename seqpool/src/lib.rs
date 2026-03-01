//! # seqpool
//!
//! A fixed-capacity pool allocator designed for high-performance networking.
//!
//! All memory is pre-allocated once at construction time. [`Pool::alloc`]
//! returns a [`Slot`] giving exclusive, mutable access to a memory region that
//! satisfies the [`Layout`] provided at construction. No bytes are ever zeroed;
//! the caller is responsible for initializing or interpreting the contents.
//!
//! [`Slot`] is [`Send`] + [`Sync`], so buffers can be allocated on one thread
//! and freed (dropped) on another. Dropping a [`Slot`] returns the region to
//! the pool immediately, making it available for the next [`Pool::alloc`] call.
//!
//! ## Example
//!
//! ```rust
//! use std::alloc::Layout;
//! use seqpool::Pool;
//!
//! // Pre-allocate 256 fixed-size packet buffers (1500 bytes each).
//! let pool = Pool::new(Layout::from_size_align(1500, 1).unwrap(), 256);
//! assert_eq!(pool.capacity(), 256);
//! assert_eq!(pool.available(), 256);
//!
//! let mut slot = pool.alloc().expect("pool has capacity");
//! slot[0] = 0xde;
//! slot[1] = 0xad;
//! assert_eq!(pool.available(), 255);
//!
//! drop(slot); // returns to pool
//! assert_eq!(pool.available(), 256);
//! ```

use std::{
    alloc::{Layout, alloc, dealloc},
    ops::{Deref, DerefMut},
    ptr::NonNull,
    sync::Arc,
};

use crossbeam_queue::ArrayQueue;

// ---------------------------------------------------------------------------
// Internal pool state
// ---------------------------------------------------------------------------

struct PoolInner {
    /// Pointer to the start of the pre-allocated memory block.
    base: NonNull<u8>,
    /// Layout of a single item, as provided by the caller.
    layout: Layout,
    /// Total number of slots.
    capacity: usize,
    /// Byte distance between the starts of consecutive slots.
    ///
    /// Equal to `layout.pad_to_align().size()`, rounded up to at least 1 so
    /// that every slot has a unique base address even for zero-sized layouts.
    stride: usize,
    /// Indices of currently-free slots.
    free: ArrayQueue<usize>,
    /// Layout used for the whole-block allocation (needed for `dealloc`).
    alloc_layout: Layout,
}

// SAFETY: `PoolInner` owns the allocation exclusively.  `ArrayQueue` is
// `Send + Sync`.  We never alias the individual slot pointers.
unsafe impl Send for PoolInner {}
unsafe impl Sync for PoolInner {}

impl Drop for PoolInner {
    fn drop(&mut self) {
        // SAFETY: `base` was allocated with `alloc_layout` in `Pool::new`.
        unsafe { dealloc(self.base.as_ptr(), self.alloc_layout) }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A fixed-capacity pool of identically-sized memory regions.
///
/// `Pool` is cheaply cloneable (it is a reference-counted handle to shared
/// state). All clones share the same backing memory and free list.
#[derive(Clone)]
pub struct Pool(Arc<PoolInner>);

impl Pool {
    /// Create a new pool.
    ///
    /// # Arguments
    ///
    /// * `layout` – the [`Layout`] every slot must satisfy.  All slots
    ///   returned by [`alloc`][Pool::alloc] are aligned to at least
    ///   `layout.align()` and are at least `layout.size()` bytes long.
    /// * `capacity` – the maximum number of simultaneously live [`Slot`]s.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero, if the total allocation would overflow
    /// `usize`, or if the global allocator cannot satisfy the request.
    pub fn new(layout: Layout, capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be greater than zero");

        // Stride = item size padded to alignment, so every slot is aligned.
        // Use at least 1 so zero-sized layouts still produce unique addresses.
        let stride = layout.pad_to_align().size().max(1);

        let total_size = stride.checked_mul(capacity).expect("pool size overflow");
        let alloc_layout =
            Layout::from_size_align(total_size, layout.align()).expect("invalid combined layout");

        // SAFETY: `total_size > 0` (capacity ≥ 1, stride ≥ 1).
        let base = unsafe {
            let ptr = alloc(alloc_layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(alloc_layout);
            }
            NonNull::new_unchecked(ptr)
        };

        let free = ArrayQueue::new(capacity);
        for i in 0..capacity {
            free.push(i).unwrap(); // can't fail: queue sized to capacity
        }

        Pool(Arc::new(PoolInner {
            base,
            layout,
            capacity,
            stride,
            free,
            alloc_layout,
        }))
    }

    /// Try to allocate a [`Slot`] from the pool.
    ///
    /// Returns `None` if every slot is currently in use.
    ///
    /// The returned [`Slot`] has exclusive mutable access to a region that
    /// satisfies the pool's [`Layout`].  **No bytes are zeroed.**  The caller
    /// is responsible for initializing or interpreting the contents.
    #[inline]
    pub fn alloc(&self) -> Option<Slot> {
        let idx = self.0.free.pop()?;
        // SAFETY: `idx < capacity`, so `base + idx * stride` is within the
        // allocation and properly aligned.
        let ptr = unsafe { NonNull::new_unchecked(self.0.base.as_ptr().add(idx * self.0.stride)) };
        Some(Slot {
            ptr,
            pool: Arc::clone(&self.0),
            idx,
        })
    }

    /// Returns the [`Layout`] shared by every slot in this pool.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.0.layout
    }

    /// Returns the maximum number of simultaneously live [`Slot`]s.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.0.capacity
    }

    /// Returns the number of slots currently available for allocation.
    ///
    /// This is a point-in-time snapshot; the value may change immediately.
    #[inline]
    pub fn available(&self) -> usize {
        self.0.free.len()
    }
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("layout", &self.0.layout)
            .field("capacity", &self.0.capacity)
            .field("available", &self.available())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Slot
// ---------------------------------------------------------------------------

/// An exclusive handle to a memory region inside a [`Pool`].
///
/// The region satisfies the pool's [`Layout`]: it is properly aligned and at
/// least `layout.size()` bytes long.  **The contents are uninitialized.**
///
/// Dropping a `Slot` returns the region to the pool, making it available for
/// the next [`Pool::alloc`] call.
///
/// `Slot` is [`Send`] + [`Sync`] so it can be transferred to another thread.
/// At most one `Slot` can refer to a given region at any time, giving
/// exclusive access without additional locking.
pub struct Slot {
    ptr: NonNull<u8>,
    pool: Arc<PoolInner>,
    idx: usize,
}

// SAFETY: The pointer is uniquely owned (the pool hands it out exactly once).
// `Arc<PoolInner>` is `Send + Sync` because `PoolInner` is.
unsafe impl Send for Slot {}
unsafe impl Sync for Slot {}

impl Slot {
    /// Returns the [`Layout`] of this slot's memory region.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.pool.layout
    }

    /// Returns a raw const pointer to the start of the memory region.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the start of the memory region.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Deref for Slot {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `layout.size()` bytes, properly aligned,
        // and we hold a shared reference (`&self`).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.pool.layout.size()) }
    }
}

impl DerefMut for Slot {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is valid for `layout.size()` bytes, properly aligned,
        // and we hold an exclusive reference (`&mut self`).
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.pool.layout.size()) }
    }
}

impl Drop for Slot {
    #[inline]
    fn drop(&mut self) {
        // Return the index to the free queue.
        // SAFETY: `idx` was originally taken from the queue, and this `Slot`
        // has exclusive ownership, so pushing it back is safe and can never
        // overflow the queue (capacity is fixed and each slot contributes
        // exactly one entry).
        self.pool
            .free
            .push(self.idx)
            .expect("BUG: pool free queue overflowed");
    }
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("ptr", &self.ptr)
            .field("idx", &self.idx)
            .field("layout", &self.pool.layout)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc as StdArc, Barrier},
        thread,
    };

    const LAYOUT: Layout = match Layout::from_size_align(1500, 1) {
        Ok(l) => l,
        Err(_) => panic!("invalid layout"),
    };

    #[test]
    fn basic_alloc_dealloc() {
        let pool = Pool::new(LAYOUT, 4);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.available(), 4);

        let s = pool.alloc().expect("pool has space");
        assert_eq!(pool.available(), 3);
        assert_eq!(s.len(), 1500);
        drop(s);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn exhaustion_returns_none() {
        let pool = Pool::new(LAYOUT, 2);
        let _s1 = pool.alloc().expect("first");
        let _s2 = pool.alloc().expect("second");
        assert!(pool.alloc().is_none(), "pool should be full");
    }

    #[test]
    fn memory_is_reused_after_free() {
        let pool = Pool::new(LAYOUT, 1);
        let ptr1 = pool.alloc().expect("first alloc").as_ptr();
        let ptr2 = pool.alloc().expect("second alloc after drop").as_ptr();
        // Only one slot exists, so the pointer must be identical.
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn slot_is_writable() {
        let pool = Pool::new(LAYOUT, 1);
        let mut slot = pool.alloc().unwrap();
        slot[0] = 0xde;
        slot[1] = 0xad;
        assert_eq!(slot[0], 0xde);
        assert_eq!(slot[1], 0xad);
    }

    #[test]
    fn slot_layout_matches_pool() {
        let pool = Pool::new(LAYOUT, 1);
        let slot = pool.alloc().unwrap();
        assert_eq!(slot.layout(), pool.layout());
    }

    #[test]
    fn pool_clone_shares_state() {
        let pool = Pool::new(LAYOUT, 4);
        let pool2 = pool.clone();
        let _s = pool.alloc().unwrap();
        // Both handles observe the same free count.
        assert_eq!(pool.available(), pool2.available());
    }

    #[test]
    fn zero_sized_layout() {
        let layout = Layout::from_size_align(0, 1).unwrap();
        let pool = Pool::new(layout, 8);
        assert_eq!(pool.capacity(), 8);
        // alloc/dealloc should not crash for zero-sized slots
        let s = pool.alloc().unwrap();
        assert_eq!(s.len(), 0);
        drop(s);
        assert_eq!(pool.available(), 8);
    }

    // ------------------------------------------------------------------
    // Stress: many threads allocating and freeing concurrently
    // ------------------------------------------------------------------

    #[test]
    fn stress_concurrent_alloc_free() {
        const THREADS: usize = 8;
        const OPS_PER_THREAD: usize = 10_000;
        const CAPACITY: usize = 64;

        let pool = Pool::new(LAYOUT, CAPACITY);
        let barrier = StdArc::new(Barrier::new(THREADS));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let pool = pool.clone();
                let barrier = StdArc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..OPS_PER_THREAD {
                        if let Some(slot) = pool.alloc() {
                            drop(slot);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        // All slots must have been returned.
        assert_eq!(pool.available(), CAPACITY);
    }

    // ------------------------------------------------------------------
    // Stress: allocate on one thread, send to another, drop there
    // ------------------------------------------------------------------

    #[test]
    fn stress_cross_thread_free() {
        use std::sync::mpsc;

        const PRODUCERS: usize = 4;
        const CONSUMERS: usize = 4;
        const MSGS: usize = 2_000;
        const CAPACITY: usize = 128;

        let pool = Pool::new(LAYOUT, CAPACITY);
        let (tx, rx) = mpsc::sync_channel::<Slot>(CAPACITY);
        let rx = StdArc::new(std::sync::Mutex::new(rx));

        // Consumer threads: receive slots and drop them.
        let consumer_handles: Vec<_> = (0..CONSUMERS)
            .map(|_| {
                let rx = StdArc::clone(&rx);
                thread::spawn(move || {
                    loop {
                        let slot = {
                            let guard = rx.lock().unwrap();
                            guard.recv()
                        };
                        match slot {
                            Ok(_slot) => { /* drop returns to pool */ }
                            Err(_) => break, // channel closed
                        }
                    }
                })
            })
            .collect();

        // Producer threads: allocate slots and send them.
        let barrier = StdArc::new(Barrier::new(PRODUCERS));
        let producer_handles: Vec<_> = (0..PRODUCERS)
            .map(|_| {
                let pool = pool.clone();
                let tx = tx.clone();
                let barrier = StdArc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut sent = 0;
                    while sent < MSGS / PRODUCERS {
                        if let Some(slot) = pool.alloc() {
                            tx.send(slot).expect("consumer alive");
                            sent += 1;
                        } else {
                            thread::yield_now();
                        }
                    }
                })
            })
            .collect();

        for h in producer_handles {
            h.join().expect("producer panicked");
        }
        drop(tx); // signal consumers to stop

        for h in consumer_handles {
            h.join().expect("consumer panicked");
        }

        assert_eq!(pool.available(), CAPACITY);
    }
}
