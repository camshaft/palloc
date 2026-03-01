//! # seqpool
//!
//! A fixed-capacity, fixed-size pool allocator for high-performance networking.
//!
//! ## Architecture (3 layers)
//!
//! 1. **Per-CPU caches** – one bounded LIFO list per logical CPU.  On Linux,
//!    the current CPU is identified via `sched_getcpu(2)`; on other platforms
//!    the current thread's ID is hashed to select a shard.  Overflow and
//!    underflow trigger batch transfers to/from a sharded arena.
//!
//! 2. **Sharded arenas** – `ARENA_MULTIPLIER × num_cpus` mutex-protected LIFO
//!    lists.  The slow path moves [`BATCH_SIZE`] slots at once to amortise lock
//!    acquisitions.
//!
//! 3. **Global free counter** – an [`AtomicUsize`] tracking the total number of
//!    unallocated slots across all layers.  Decremented atomically before every
//!    successful [`Pool::alloc`] (providing back-pressure), incremented after
//!    every [`Pool::free`].
//!
//! ## Memory layout
//!
//! Every slot occupies one stride-block in the pre-allocated slab:
//!
//! ```text
//! ┌──────────────────────┬─────────────────────────────────────────┐
//! │  SlotHeader (16 B)   │  user data  (layout.size() bytes)       │
//! └──────────────────────┴─────────────────────────────────────────┘
//! ^── base + N×stride              ^── base + N×stride + HEADER_SIZE
//! ```
//!
//! [`Slot`] is a single `NonNull<u8>` pointing at the *user-data* start.
//! The corresponding [`SlotHeader`] is always exactly [`HEADER_SIZE`] bytes
//! before it, giving O(1) access back to the pool with no per-`Slot`
//! reference count.
//!
//! This scheme requires `layout.align() ≤ HEADER_SIZE` (≤ 16 on 64-bit
//! systems) so that the fixed-offset arithmetic preserves correct alignment
//! for both regions.  Typical networking buffers (MTU-sized, alignment 1–8)
//! satisfy this constraint.
//!
//! ## Ownership model
//!
//! [`Slot`] has **no `Drop` implementation**.  The caller decides when and how
//! memory is returned to the pool, enabling patterns such as reference-counted
//! sub-slices where a single physical slot is freed only once all logical
//! references are dropped.  Return a slot to the pool with [`Pool::free`].
//!
//! ## Example
//!
//! ```rust
//! use std::alloc::Layout;
//! use seqpool::Pool;
//!
//! let pool = Pool::new(Layout::from_size_align(1500, 1).unwrap(), 256);
//! assert_eq!(pool.capacity(), 256);
//!
//! let mut slot = pool.alloc().expect("pool has capacity");
//! slot.as_mut_slice()[0] = 0xde;
//! assert_eq!(pool.available(), 255);
//!
//! pool.free(slot); // explicitly return the slot
//! assert_eq!(pool.available(), 256);
//! ```

use std::{
    alloc::{Layout, alloc, dealloc},
    hash::{Hash, Hasher},
    mem::{align_of, size_of},
    ptr::{NonNull, null_mut},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Maximum number of slots in a per-CPU cache before flushing a batch.
const LOCAL_CACHE_MAX: usize = 32;

/// Number of slots moved per batch between a CPU cache and an arena.
const BATCH_SIZE: usize = 16;

/// Number of sharded arenas = `ARENA_MULTIPLIER × num_cpus`.
const ARENA_MULTIPLIER: usize = 4;

// ── SlotHeader ────────────────────────────────────────────────────────────────

/// Metadata stored immediately before each slot's user-data region.
///
/// To reach the `SlotHeader` from a [`Slot`] data pointer, subtract
/// [`HEADER_SIZE`] bytes.
#[repr(C)]
struct SlotHeader {
    /// Raw back-pointer to the owning [`PoolInner`].
    pool: *const PoolInner,
    /// Intrusive linked-list link, valid only while the slot is in a free list.
    next: *mut SlotHeader,
}

/// Fixed byte distance from a [`Slot`]'s data pointer back to its
/// [`SlotHeader`].  Equal to `size_of::<SlotHeader>()`.
pub const HEADER_SIZE: usize = size_of::<SlotHeader>();

// ── Slot ──────────────────────────────────────────────────────────────────────

/// A handle to a fixed-size, uninitialized memory region inside a [`Pool`].
///
/// `Slot` is a **thin single-pointer** type (`NonNull<u8>`) with **no `Drop`
/// implementation**.  The caller decides when to return it to the pool via
/// [`Pool::free`].  A forgotten slot keeps its backing memory reserved for the
/// pool's lifetime, but is not otherwise unsafe.
///
/// `Slot` is `Send + Sync`: it can be allocated on one thread and freed on
/// another.
pub struct Slot(NonNull<u8>);

// SAFETY: The pointer is exclusively owned.  PoolInner (and thus the backing
// slab) is kept alive by the caller's Pool Arc for the Slot's lifetime.
unsafe impl Send for Slot {}
unsafe impl Sync for Slot {}

impl Slot {
    /// Returns a pointer to this slot's [`SlotHeader`].
    ///
    /// # Safety
    /// The slot must still be valid (not freed).
    #[inline]
    unsafe fn header(&self) -> *mut SlotHeader {
        // SAFETY: Pool::new guarantees the header is always HEADER_SIZE bytes
        // (= size_of::<SlotHeader>()) before the data pointer.
        unsafe { (self.0.as_ptr() as *mut SlotHeader).sub(1) }
    }

    #[inline]
    fn pool_inner(&self) -> &PoolInner {
        // SAFETY: pool pointer in the header is valid for the pool's lifetime,
        // and the caller ensures the Pool (Arc) outlives the Slot.
        unsafe { &*(*self.header()).pool }
    }

    /// Returns the [`Layout`] of this slot's data region.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.pool_inner().layout
    }

    /// Returns a raw const pointer to the start of the data region.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }

    /// Returns a raw mutable pointer to the start of the data region.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_ptr()
    }

    /// Returns the data region as a byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        let size = self.pool_inner().layout.size();
        // SAFETY: pointer is valid for `size` bytes, properly aligned.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), size) }
    }

    /// Returns the data region as a mutable byte slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let size = self.pool_inner().layout.size();
        // SAFETY: pointer is valid for `size` bytes; exclusive via `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr(), size) }
    }
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("data_ptr", &self.0.as_ptr())
            .field("layout", &self.pool_inner().layout)
            .finish()
    }
}

// ── Per-CPU cache ─────────────────────────────────────────────────────────────

struct CpuCache {
    head: *mut SlotHeader,
    len: usize,
}

// SAFETY: accessed only while holding the surrounding Mutex.
unsafe impl Send for CpuCache {}

impl CpuCache {
    const fn empty() -> Self {
        CpuCache {
            head: null_mut(),
            len: 0,
        }
    }

    /// Pop one slot from the front of the cache.
    #[inline]
    fn pop(&mut self) -> Option<*mut SlotHeader> {
        if self.head.is_null() {
            return None;
        }
        let h = self.head;
        // SAFETY: h is non-null and inside the pool's backing allocation.
        self.head = unsafe { (*h).next };
        unsafe { (*h).next = null_mut() };
        self.len -= 1;
        Some(h)
    }

    /// Push one slot onto the front of the cache.
    #[inline]
    fn push(&mut self, h: *mut SlotHeader) {
        // SAFETY: h is valid.
        unsafe { (*h).next = self.head };
        self.head = h;
        self.len += 1;
    }

    /// Drain the first `count` items into a null-terminated sub-list.
    ///
    /// Returns `(head, tail, actual_count)`.  When `actual_count == 0`,
    /// `head` and `tail` are null.
    fn drain(&mut self, count: usize) -> (*mut SlotHeader, *mut SlotHeader, usize) {
        if self.head.is_null() || count == 0 {
            return (null_mut(), null_mut(), 0);
        }
        let head = self.head;
        let mut tail = head;
        let mut n = 1usize;
        while n < count {
            // SAFETY: tail is non-null and valid.
            let next = unsafe { (*tail).next };
            if next.is_null() {
                break;
            }
            tail = next;
            n += 1;
        }
        // Sever the sub-list from the cache.
        // SAFETY: tail is valid.
        self.head = unsafe { (*tail).next };
        unsafe { (*tail).next = null_mut() };
        self.len -= n;
        (head, tail, n)
    }
}

// ── Sharded arena ─────────────────────────────────────────────────────────────

struct ArenaInner {
    head: *mut SlotHeader,
    len: usize,
}

// SAFETY: accessed only while holding the surrounding Mutex.
unsafe impl Send for ArenaInner {}

impl ArenaInner {
    const fn empty() -> Self {
        ArenaInner {
            head: null_mut(),
            len: 0,
        }
    }

    /// Prepend a null-terminated sub-list `head →…→ tail` of `count` items.
    #[inline]
    fn prepend(&mut self, head: *mut SlotHeader, tail: *mut SlotHeader, count: usize) {
        // SAFETY: tail is valid.
        unsafe { (*tail).next = self.head };
        self.head = head;
        self.len += count;
    }

    /// Drain the first `count` items into a null-terminated sub-list.
    ///
    /// Returns `(head, tail, actual_count)`.  When `actual_count == 0`,
    /// `head` and `tail` are null.
    fn drain(&mut self, count: usize) -> (*mut SlotHeader, *mut SlotHeader, usize) {
        if self.head.is_null() || count == 0 {
            return (null_mut(), null_mut(), 0);
        }
        let head = self.head;
        let mut tail = head;
        let mut n = 1usize;
        while n < count {
            // SAFETY: tail is non-null and valid.
            let next = unsafe { (*tail).next };
            if next.is_null() {
                break;
            }
            tail = next;
            n += 1;
        }
        // SAFETY: tail is valid.
        self.head = unsafe { (*tail).next };
        unsafe { (*tail).next = null_mut() };
        self.len -= n;
        (head, tail, n)
    }
}

// ── PoolInner ─────────────────────────────────────────────────────────────────

struct PoolInner {
    /// Base of the pre-allocated slab.
    base: NonNull<u8>,
    /// Layout of each slot's user-data region.
    layout: Layout,
    /// Total number of slots.
    capacity: usize,
    /// Layout used for the whole-slab allocation (needed for `dealloc`).
    alloc_layout: Layout,
    /// Layer 3: total number of unallocated slots across all layers.
    free_count: AtomicUsize,
    /// Layer 1: per-CPU caches (one per logical CPU).
    cpu_caches: Box<[Mutex<CpuCache>]>,
    /// Layer 2: sharded arenas.
    arenas: Box<[Mutex<ArenaInner>]>,
}

// SAFETY: PoolInner owns its slab exclusively; all shared fields are
// synchronised via atomics or mutexes.
unsafe impl Send for PoolInner {}
unsafe impl Sync for PoolInner {}

impl Drop for PoolInner {
    fn drop(&mut self) {
        // `free_count` tracks slots that are available for allocation (in any
        // layer: cpu caches, arenas, or the logical "ready" state).  It is
        // decremented by `alloc` and incremented by `free`.  When it equals
        // `capacity`, every slot has been returned — no `Slot` handle is live.
        debug_assert_eq!(
            self.free_count.load(Ordering::Relaxed),
            self.capacity,
            "Pool dropped while Slot handles are still outstanding \
             (free_count != capacity)"
        );
        // SAFETY: `base` was allocated with `alloc_layout` in `Pool::new`.
        unsafe { dealloc(self.base.as_ptr(), self.alloc_layout) }
    }
}

// ── CPU / shard selection ─────────────────────────────────────────────────────

/// Returns an index in `[0, num_shards)` representing the current CPU.
///
/// On Linux uses `sched_getcpu(2)` for a best-effort CPU index.
/// Falls back to hashing the current thread ID on other platforms.
#[inline]
fn current_shard(num_shards: usize) -> usize {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: pure read with no side effects.
        let cpu = unsafe { libc::sched_getcpu() };
        if cpu >= 0 {
            return (cpu as usize) % num_shards;
        }
    }
    // Fallback: hash thread ID.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    (h.finish() as usize) % num_shards
}

// ── Pool ──────────────────────────────────────────────────────────────────────

/// A fixed-capacity pool of identically-sized memory regions.
///
/// `Pool` is a cheap-to-clone, reference-counted handle.  All clones share
/// the same backing slab, CPU caches, and arenas.
#[derive(Clone)]
pub struct Pool(Arc<PoolInner>);

impl Pool {
    /// Create a new pool.
    ///
    /// # Arguments
    ///
    /// * `layout` – the [`Layout`] every slot must satisfy.
    /// * `capacity` – maximum number of simultaneously live [`Slot`]s.
    ///
    /// # Panics
    ///
    /// * If `capacity` is zero.
    /// * If `layout.align() > HEADER_SIZE` (the fixed-offset header requires
    ///   alignment ≤ header size; for typical networking buffers this is never
    ///   an issue).
    /// * On allocation failure.
    pub fn new(layout: Layout, capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be greater than zero");
        assert!(
            layout.align() <= HEADER_SIZE,
            "layout.align() ({}) exceeds HEADER_SIZE ({}); use alignment ≤ {}",
            layout.align(),
            HEADER_SIZE,
            HEADER_SIZE,
        );

        // effective_align = max(align_of::<SlotHeader>(), layout.align()).
        // Since layout.align() ≤ HEADER_SIZE = size_of::<SlotHeader>(),
        // and size_of is always a multiple of align_of, effective_align
        // divides HEADER_SIZE, so data_ptr = header_ptr + HEADER_SIZE is
        // correctly aligned to layout.align().
        let effective_align = align_of::<SlotHeader>().max(layout.align());

        // stride = round_up(HEADER_SIZE + user_data_size, effective_align).
        // Must be a multiple of effective_align so that `base + i×stride` is
        // correctly aligned for every i.
        //
        // For zero-sized layouts, stride = HEADER_SIZE (e.g. 16 on 64-bit).
        // Data pointer for slot i = base + i×stride + HEADER_SIZE, so the
        // pointers are base+16, base+32, base+48, … — all distinct.
        let raw_stride = HEADER_SIZE + layout.size();
        let stride = (raw_stride + effective_align - 1) & !(effective_align - 1);

        let total_size = stride.checked_mul(capacity).expect("pool size overflow");
        let alloc_layout =
            Layout::from_size_align(total_size, effective_align).expect("invalid layout");

        // SAFETY: total_size > 0 (capacity ≥ 1, stride ≥ HEADER_SIZE + 1).
        let base = unsafe {
            let ptr = alloc(alloc_layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(alloc_layout);
            }
            NonNull::new_unchecked(ptr)
        };

        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_arenas = (num_cpus * ARENA_MULTIPLIER).max(1);

        let cpu_caches: Box<[Mutex<CpuCache>]> = (0..num_cpus)
            .map(|_| Mutex::new(CpuCache::empty()))
            .collect();
        let arenas: Box<[Mutex<ArenaInner>]> = (0..num_arenas)
            .map(|_| Mutex::new(ArenaInner::empty()))
            .collect();

        let inner = PoolInner {
            base,
            layout,
            capacity,
            alloc_layout,
            free_count: AtomicUsize::new(capacity),
            cpu_caches,
            arenas,
        };

        // Wrap in Arc first so that pool_ptr is stable for the lifetime of
        // the inner allocation.
        let arc = Arc::new(inner);
        let pool_ptr: *const PoolInner = Arc::as_ptr(&arc);

        // Distribute all slots evenly across arenas (round-robin).
        // SAFETY: We have exclusive Arc access (refcount = 1) and the slab
        // is our own allocation.
        for i in 0..capacity {
            let header = unsafe { base.as_ptr().add(i * stride) as *mut SlotHeader };
            unsafe {
                (*header).pool = pool_ptr;
                (*header).next = null_mut();
            }
            let arena_idx = i % num_arenas;
            arc.arenas[arena_idx]
                .lock()
                .unwrap()
                .prepend(header, header, 1);
        }

        Pool(arc)
    }

    /// Attempt to allocate a [`Slot`] from the pool.
    ///
    /// Returns `None` if the pool is exhausted.  **No bytes are zeroed.**
    #[inline]
    pub fn alloc(&self) -> Option<Slot> {
        let pool = &*self.0;

        // ── Layer 3: reserve a slot by decrementing the global free count ──
        let mut free = pool.free_count.load(Ordering::Relaxed);
        loop {
            if free == 0 {
                return None;
            }
            match pool.free_count.compare_exchange_weak(
                free,
                free - 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(cur) => free = cur,
            }
        }

        let cpu = current_shard(pool.cpu_caches.len());

        // ── Layer 1: try per-CPU cache ────────────────────────────────────
        if let Some(h) = pool.cpu_caches[cpu].lock().unwrap().pop() {
            return Some(Self::header_to_slot(h));
        }

        // ── Layer 2: slow path – pull a batch from an arena ──────────────
        if let Some(h) = self.pull_from_arena(pool, cpu) {
            return Some(Self::header_to_slot(h));
        }

        // No slot found in any layer (very rare race) – undo reservation.
        pool.free_count.fetch_add(1, Ordering::Release);
        None
    }

    /// Free a [`Slot`], returning it to the pool.
    ///
    /// After this call, `slot`'s data pointer must not be used again.
    #[inline]
    pub fn free(&self, slot: Slot) {
        let pool = &*self.0;
        // SAFETY: header is HEADER_SIZE bytes before the data pointer and
        // is valid as long as the pool slab is alive.
        let h = unsafe { slot.header() };
        // slot has no Drop, so letting it move out here is a no-op.

        let cpu = current_shard(pool.cpu_caches.len());

        // ── Layer 1: push into per-CPU cache ─────────────────────────────
        let flush = {
            let mut cache = pool.cpu_caches[cpu].lock().unwrap();
            cache.push(h);
            if cache.len > LOCAL_CACHE_MAX {
                cache.drain(BATCH_SIZE)
            } else {
                (null_mut(), null_mut(), 0)
            }
        };

        // ── Layer 2: if cache was full, flush a batch to the arena ────────
        // Locks are always acquired sequentially (never nested) to avoid
        // deadlock: cpu_cache first, then arena.
        if flush.2 > 0 {
            let arena_idx = cpu % pool.arenas.len();
            pool.arenas[arena_idx]
                .lock()
                .unwrap()
                .prepend(flush.0, flush.1, flush.2);
        }

        // ── Layer 3: announce the newly-available slot ────────────────────
        pool.free_count.fetch_add(1, Ordering::Release);
    }

    /// Slow path: pull a batch from the nearest non-empty arena into the CPU
    /// cache, then return one slot header to the caller.
    ///
    /// The arena lock and the CPU-cache lock are acquired **sequentially**
    /// (never nested) to keep the lock ordering consistent with `free`.
    #[cold]
    fn pull_from_arena(&self, pool: &PoolInner, cpu: usize) -> Option<*mut SlotHeader> {
        let num_arenas = pool.arenas.len();
        let base = cpu % num_arenas;

        for offset in 0..num_arenas {
            let idx = (base + offset) % num_arenas;

            // Phase 1: drain a batch from the arena (arena lock only).
            let (batch_head, _batch_tail, batch_count) = {
                let mut arena = pool.arenas[idx].lock().unwrap();
                if arena.len == 0 {
                    continue;
                }
                arena.drain(BATCH_SIZE)
            }; // arena lock released here

            if batch_count == 0 {
                continue;
            }

            // Phase 2: push batch into CPU cache (cache lock only).
            let mut cache = pool.cpu_caches[cpu].lock().unwrap();
            let mut h = batch_head;
            while !h.is_null() {
                // SAFETY: h is valid (inside slab allocation).
                let next = unsafe { (*h).next };
                unsafe { (*h).next = null_mut() };
                cache.push(h);
                h = next;
            }
            return cache.pop();
        }
        None
    }

    /// Convert a raw `SlotHeader` pointer to a `Slot` pointing at user data.
    #[inline]
    fn header_to_slot(h: *mut SlotHeader) -> Slot {
        // SAFETY: data starts exactly HEADER_SIZE bytes after the header.
        let data_ptr = unsafe { h.add(1) as *mut u8 };
        Slot(unsafe { NonNull::new_unchecked(data_ptr) })
    }

    /// Returns the [`Layout`] of each slot.
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
        self.0.free_count.load(Ordering::Relaxed)
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
    fn basic_alloc_free() {
        let pool = Pool::new(LAYOUT, 4);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.available(), 4);

        let s = pool.alloc().expect("pool has space");
        assert_eq!(pool.available(), 3);
        assert_eq!(s.as_slice().len(), 1500);
        pool.free(s);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn exhaustion_returns_none() {
        let pool = Pool::new(LAYOUT, 2);
        let s1 = pool.alloc().expect("first");
        let s2 = pool.alloc().expect("second");
        assert!(pool.alloc().is_none(), "pool should be full");
        pool.free(s1);
        pool.free(s2);
    }

    #[test]
    fn memory_is_reused_after_free() {
        let pool = Pool::new(LAYOUT, 1);
        let s1 = pool.alloc().expect("first alloc");
        let ptr1 = s1.as_ptr();
        pool.free(s1);
        let s2 = pool.alloc().expect("second alloc after free");
        let ptr2 = s2.as_ptr();
        pool.free(s2);
        // Only one slot exists, so the data pointer must be identical.
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn slot_is_writable() {
        let pool = Pool::new(LAYOUT, 1);
        let mut slot = pool.alloc().unwrap();
        slot.as_mut_slice()[0] = 0xde;
        slot.as_mut_slice()[1] = 0xad;
        assert_eq!(slot.as_slice()[0], 0xde);
        assert_eq!(slot.as_slice()[1], 0xad);
        pool.free(slot);
    }

    #[test]
    fn slot_layout_matches_pool() {
        let pool = Pool::new(LAYOUT, 1);
        let slot = pool.alloc().unwrap();
        assert_eq!(slot.layout(), pool.layout());
        pool.free(slot);
    }

    #[test]
    fn pool_clone_shares_state() {
        let pool = Pool::new(LAYOUT, 4);
        let pool2 = pool.clone();
        let s = pool.alloc().unwrap();
        // Both handles observe the same free count.
        assert_eq!(pool.available(), pool2.available());
        pool.free(s);
    }

    #[test]
    fn zero_sized_layout() {
        let layout = Layout::from_size_align(0, 1).unwrap();
        let pool = Pool::new(layout, 8);
        assert_eq!(pool.capacity(), 8);
        let s = pool.alloc().unwrap();
        assert_eq!(s.as_slice().len(), 0);
        pool.free(s);
        assert_eq!(pool.available(), 8);
    }

    #[test]
    fn slot_header_back_pointer_is_valid() {
        let pool = Pool::new(LAYOUT, 4);
        let slot = pool.alloc().unwrap();
        // The pool pointer stored in the header must point to the same inner.
        let inner_via_header = slot.pool_inner() as *const PoolInner;
        let inner_via_pool = Arc::as_ptr(&pool.0);
        assert_eq!(inner_via_header, inner_via_pool);
        pool.free(slot);
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
                            pool.free(slot);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        assert_eq!(pool.available(), CAPACITY);
    }

    // ------------------------------------------------------------------
    // Stress: allocate on one thread, send to another, free there
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

        // Consumer threads: receive slots and free them.
        let consumer_handles: Vec<_> = (0..CONSUMERS)
            .map(|_| {
                let rx = StdArc::clone(&rx);
                let pool = pool.clone();
                thread::spawn(move || {
                    loop {
                        let slot = {
                            let guard = rx.lock().unwrap();
                            guard.recv()
                        };
                        match slot {
                            Ok(slot) => pool.free(slot),
                            Err(_) => break,
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
