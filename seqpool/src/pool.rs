//! The main pool implementation.
//!
//! `Pool<T>` pre-allocates a fixed number of `T` slots and provides lock-free
//! `alloc` / `free` operations that work via `&self`.
//!
//! ## Memory layout
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │  data: Box<[MaybeUninit<T>]>                         │  (slot_count entries)
//! │  next: Box<[AtomicU32]>                              │  (slot_count entries, 4B each)
//! │  home_cpu: Box<[AtomicU16]>                          │  (slot_count entries, 2B each)
//! │  local_heads: Box<[CachePadded<AtomicU32>]>          │  (num_cpus entries, cache-line each)
//! │  remote_queues: Box<[MpscQueue]>                     │  (num_cpus entries)
//! │  global_stack: TreiberStack                          │  (single atomic u64)
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! ## Allocation flow
//!
//! 1. **Per-CPU local list** (RSEQ fast path): Pop from `local_heads[cpu]` — zero atomics.
//! 2. **Per-CPU local list** (atomic fallback): CAS pop from local head.
//! 3. **Remote MPSC queue**: Drain up to `batch_size` from `remote_queues[cpu]`.
//!    Excess stays queued for next drain. This bounds local list growth.
//! 4. **Global Treiber stack**: Pop a batch from the global stack.
//!
//! ## Free flow
//!
//! 1. RSEQ push to current CPU's local list (zero atomics) — fast path.
//! 2. If not on home CPU: MPSC enqueue to home CPU's remote queue (one xchg).
//! 3. Remote queue is unbounded; the home CPU drains it in bounded batches on alloc.
//!
//! ## Bounded caches
//!
//! - On alloc, at most `batch_size` items are taken from remote or global.
//! - All slots start on the global stack. CPUs pull batches on demand.
//! - The global stack is the pressure relief valve — it's where excess slots
//!   live when no CPU needs them.

use crate::{
    mpsc::MpscQueue,
    rseq::{self, CpuHead, NULL},
    treiber::TreiberStack,
};
use core::{
    alloc::Layout,
    mem::MaybeUninit,
    ptr::NonNull,
    sync::atomic::{AtomicU16, AtomicU32, Ordering},
};
use crossbeam_utils::CachePadded;

/// Describes the storage configuration for a pool.
pub trait Storage: 'static + Send + Sync {
    /// The element type stored in each slot.
    type T: 'static;

    /// Total number of pre-allocated slots.
    fn slot_count(&self) -> u32;

    /// How many slots to transfer in a batch when refilling from
    /// remote queues or the global stack. Also caps how many items
    /// are drained from the remote queue per alloc slow path.
    fn batch_size(&self) -> u32;

    /// Max items in a per-CPU local free list before overflow to remote/global.
    fn local_cache_capacity(&self) -> u32 {
        self.batch_size() * 2
    }

    /// Max items in a per-CPU remote MPSC queue before overflow to global.
    fn remote_cache_capacity(&self) -> u32 {
        self.batch_size() * 4
    }
}

/// Compile-time-known fixed-type pool configuration.
///
/// Use this when the element type and size are known at compile time
/// (the common case for packet pools).
pub struct PoolConfig<T: 'static> {
    pub slot_count: u32,
    pub batch_size: u32,
    _marker: core::marker::PhantomData<T>,
}

impl<T: 'static + Send + Sync> PoolConfig<T> {
    pub fn new(slot_count: u32, batch_size: u32) -> Self {
        Self {
            slot_count,
            batch_size: batch_size.max(1),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: 'static + Send + Sync> Storage for PoolConfig<T> {
    type T = T;

    #[inline]
    fn slot_count(&self) -> u32 {
        self.slot_count
    }

    #[inline]
    fn batch_size(&self) -> u32 {
        self.batch_size
    }
}

/// A pre-allocated, lock-free object pool.
///
/// All methods take `&self`. The pool is `Send + Sync` and designed to be
/// created once (e.g., in a `static` or at program start) and shared.
pub struct Pool<S: Storage> {
    /// Base pointer to the contiguous slot data allocation.
    data_base: NonNull<MaybeUninit<S::T>>,

    /// Per-slot "next" pointers for free lists.
    next: Box<[AtomicU32]>,

    /// Per-slot home CPU — which CPU most recently allocated this slot.
    home_cpu: Box<[AtomicU16]>,

    /// Per-CPU local free list heads (cache-line padded).
    local_heads: Box<[CpuHead]>,

    /// Per-CPU local list approximate lengths (cache-line padded).
    local_counts: Box<[CachePadded<AtomicU32>]>,

    /// Per-CPU remote MPSC queues for cross-CPU frees.
    remote_queues: Box<[MpscQueue]>,

    /// Global fallback free stack — holds all slots initially.
    global_stack: TreiberStack,

    /// Number of CPUs.
    num_cpus: u16,

    /// The storage configuration.
    storage: S,
}

unsafe impl<S: Storage> Send for Pool<S> where S::T: Send {}
unsafe impl<S: Storage> Sync for Pool<S> where S::T: Send {}

impl<S: Storage> Pool<S> {
    /// Create a new pool. All slots start on the global free stack.
    pub fn new(storage: S) -> Self {
        let slot_count = storage.slot_count();
        assert!(slot_count > 0, "pool must have at least one slot");
        assert!(slot_count < NULL, "slot_count must be less than {NULL}");

        let num_cpus = rseq::possible_cpus().min(u16::MAX as usize) as u16;

        // Single contiguous allocation for slot data.
        let data_layout = Self::data_layout(&storage);
        let data_base = if data_layout.size() == 0 {
            NonNull::new(data_layout.align() as *mut MaybeUninit<S::T>).unwrap()
        } else {
            let ptr = unsafe { std::alloc::alloc(data_layout) };
            NonNull::new(ptr as *mut MaybeUninit<S::T>)
                .unwrap_or_else(|| std::alloc::handle_alloc_error(data_layout))
        };

        let next: Box<[AtomicU32]> = (0..slot_count).map(|_| AtomicU32::new(NULL)).collect();
        let home_cpu: Box<[AtomicU16]> = (0..slot_count).map(|_| AtomicU16::new(0)).collect();

        let local_heads: Box<[CpuHead]> = (0..num_cpus)
            .map(|_| CachePadded::new(AtomicU32::new(NULL)))
            .collect();

        let local_counts: Box<[CachePadded<AtomicU32>]> = (0..num_cpus)
            .map(|_| CachePadded::new(AtomicU32::new(0)))
            .collect();

        let remote_queues: Box<[MpscQueue]> = (0..num_cpus).map(|_| MpscQueue::new()).collect();

        let global_stack = TreiberStack::new();

        // All slots start on the global stack.
        for i in (0..slot_count).rev() {
            unsafe { global_stack.push(&next, i) };
        }

        Pool {
            data_base,
            next,
            home_cpu,
            local_heads,
            local_counts,
            remote_queues,
            global_stack,
            num_cpus,
            storage,
        }
    }

    /// Allocate a slot, returning a pointer to uninitialized memory.
    ///
    /// Returns `None` if the pool is exhausted (all slots are in use).
    ///
    /// The caller must:
    /// - Initialize the memory before reading from it.
    /// - Eventually call [`free`](Self::free) with the returned pointer.
    #[inline]
    pub fn alloc(&self) -> Option<NonNull<S::T>> {
        // Fast path: RSEQ per-CPU pop.
        match unsafe { rseq::per_cpu_pop(&self.local_heads, &self.next) } {
            Ok((index, cpu)) => {
                self.home_cpu[index as usize].store(cpu as u16, Ordering::Relaxed);
                return Some(self.index_to_ptr(index));
            }
            Err(_) => {}
        }

        // Atomic fallback: CAS pop from our CPU's local head.
        if let Some((index, cpu)) = self.try_atomic_pop() {
            self.home_cpu[index as usize].store(cpu as u16, Ordering::Relaxed);
            return Some(self.index_to_ptr(index));
        }

        self.alloc_slow()
    }

    /// Atomic fallback for per-CPU pop.
    #[inline]
    fn try_atomic_pop(&self) -> Option<(u32, usize)> {
        let cpu = rseq::cpu_id().min(self.num_cpus as usize - 1);
        let head = &self.local_heads[cpu];

        // CAS loop to pop one item.
        let mut current = head.load(Ordering::Acquire);
        loop {
            if current == NULL {
                return None;
            }
            let next_val = self.next[current as usize].load(Ordering::Relaxed);
            match head.compare_exchange_weak(current, next_val, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Some((current, cpu)),
                Err(actual) => current = actual,
            }
        }
    }

    /// Slow path: drain remote queue (bounded), then refill from global.
    #[cold]
    fn alloc_slow(&self) -> Option<NonNull<S::T>> {
        let cpu = rseq::cpu_id().min(self.num_cpus as usize - 1);
        let batch = self.storage.batch_size() as usize;

        // Drain remote queue — bounded by batch_size.
        // snmalloc-style: walk front→back, sentinel stays.
        let mut first = None;
        let mut drained = 0usize;
        self.remote_queues[cpu].drain(&self.next, |index| {
            if first.is_none() {
                first = Some(index);
            } else {
                self.atomic_push_local(cpu, index);
            }
            drained += 1;
            if drained >= batch {
                core::ops::ControlFlow::Break(())
            } else {
                core::ops::ControlFlow::Continue(())
            }
        });

        if let Some(index) = first {
            self.home_cpu[index as usize].store(cpu as u16, Ordering::Relaxed);
            return Some(self.index_to_ptr(index));
        }

        self.refill_from_global(cpu)
    }

    /// Take a batch of slots from the global stack and put them on our local list.
    /// Returns one slot to the caller, the rest go to `local_heads[cpu]`.
    #[cold]
    fn refill_from_global(&self, cpu: usize) -> Option<NonNull<S::T>> {
        let batch = self.storage.batch_size();
        let mut first = None;
        let mut count = 0u32;

        while count < batch {
            match self.global_stack.pop(&self.next) {
                Some(index) => {
                    if first.is_none() {
                        first = Some(index);
                    } else {
                        self.atomic_push_local(cpu, index);
                    }
                    count += 1;
                }
                None => break,
            }
        }

        first.map(|index| {
            self.home_cpu[index as usize].store(cpu as u16, Ordering::Relaxed);
            self.index_to_ptr(index)
        })
    }

    /// Free a previously allocated slot.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been returned by [`alloc`](Self::alloc) on this pool.
    /// - `ptr` must not have been freed already (no double-free).
    /// - The caller must not use `ptr` after calling `free`.
    #[inline]
    pub unsafe fn free(&self, ptr: NonNull<S::T>) {
        let index = self.ptr_to_index(ptr);
        debug_assert!((index as usize) < self.next.len());

        let home = self.home_cpu[index as usize].load(Ordering::Relaxed) as usize;
        let cpu = rseq::cpu_id().min(self.num_cpus as usize - 1);

        // 1. Try local list (if under capacity).
        let local_count = self.local_counts[cpu].load(Ordering::Relaxed);
        if local_count < self.storage.local_cache_capacity() {
            // Fast path: RSEQ push to current CPU's local list.
            match unsafe { rseq::per_cpu_push(&self.local_heads, &self.next, index) } {
                Ok(_) => {
                    self.local_counts[cpu].fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(_) => {
                    // RSEQ unavailable, try atomic CAS push.
                    self.atomic_push_local(cpu, index);
                    self.local_counts[cpu].fetch_add(1, Ordering::Relaxed);
                    return;
                }
            }
        }

        // 2. Local is full. Try remote queue for home CPU (if under capacity).
        let target = home.min(self.num_cpus as usize - 1);
        let remote_count = self.remote_queues[target].approx_len();
        if remote_count < self.storage.remote_cache_capacity() {
            unsafe {
                self.remote_queues[target].enqueue(&self.next, index);
            }
            return;
        }

        // 3. Both full. Push to global stack.
        unsafe {
            self.global_stack.push(&self.next, index);
        }
    }

    /// CAS push to a CPU's local list.
    fn atomic_push_local(&self, cpu: usize, index: u32) {
        let head = &self.local_heads[cpu];
        loop {
            let old = head.load(Ordering::Relaxed);
            self.next[index as usize].store(old, Ordering::Relaxed);
            match head.compare_exchange_weak(old, index, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return,
                Err(_) => {}
            }
        }
    }

    #[inline]
    fn index_to_ptr(&self, index: u32) -> NonNull<S::T> {
        debug_assert!((index as usize) < self.storage.slot_count() as usize);
        unsafe {
            let ptr = self.data_base.as_ptr().add(index as usize) as *mut S::T;
            NonNull::new_unchecked(ptr)
        }
    }

    /// Convert a pointer back to a slot index.
    #[inline]
    fn ptr_to_index(&self, ptr: NonNull<S::T>) -> u32 {
        let base = self.data_base.as_ptr() as usize;
        let addr = ptr.as_ptr() as usize;
        debug_assert!(addr >= base);
        let offset = addr - base;
        let slot_size = size_of::<MaybeUninit<S::T>>().max(1);
        let index = offset / slot_size;
        debug_assert!(index < self.storage.slot_count() as usize);
        debug_assert_eq!(offset % slot_size, 0);
        index as u32
    }

    /// Returns the total number of slots in the pool.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.storage.slot_count()
    }

    /// Returns the number of CPUs this pool is configured for.
    #[inline]
    pub fn num_cpus(&self) -> u16 {
        self.num_cpus
    }

    #[inline]
    fn data_layout(storage: &S) -> Layout {
        Layout::array::<MaybeUninit<S::T>>(storage.slot_count() as usize).expect("layout overflow")
    }
}

impl<S: Storage> Drop for Pool<S> {
    fn drop(&mut self) {
        let data_layout = Self::data_layout(&self.storage);
        if data_layout.size() > 0 {
            unsafe {
                std::alloc::dealloc(self.data_base.as_ptr() as *mut u8, data_layout);
            }
        }
    }
}

/// A wrapper around `NonNull<T>` that is `Send`.
///
/// # Safety
///
/// The caller must ensure the pointed-to memory is safe to access from
/// the destination thread (which it is for pool slots, since the pool is `Sync`).
#[repr(transparent)]
pub struct SendPtr<T>(pub NonNull<T>);

unsafe impl<T: Send> Send for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn new(ptr: NonNull<T>) -> Self {
        Self(ptr)
    }

    pub fn into_inner(self) -> NonNull<T> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_config<T: 'static + Send + Sync>(slot_count: u32, batch_size: u32) -> PoolConfig<T> {
        PoolConfig::new(slot_count, batch_size)
    }

    #[test]
    fn basic_alloc_free() {
        let pool = Pool::new(pool_config::<[u8; 1500]>(16, 4));

        let ptr = pool.alloc().expect("should allocate");
        unsafe { core::ptr::write_bytes(ptr.as_ptr(), 0xAB, 1) };
        unsafe { pool.free(ptr) };

        let ptr2 = pool.alloc().expect("should re-allocate");
        unsafe { pool.free(ptr2) };
    }

    #[test]
    fn exhaust_pool() {
        let pool = Pool::new(pool_config::<u64>(4, 2));

        let mut ptrs = vec![];
        for _ in 0..4 {
            ptrs.push(pool.alloc().expect("should allocate"));
        }
        assert!(pool.alloc().is_none(), "pool should be exhausted");

        unsafe { pool.free(ptrs.pop().unwrap()) };
        let p = pool.alloc().expect("should allocate after free");
        unsafe { pool.free(p) };
    }

    #[test]
    fn all_ptrs_unique() {
        let pool = Pool::new(pool_config::<u64>(100, 16));

        let mut ptrs: Vec<NonNull<u64>> = (0..100)
            .map(|_| pool.alloc().expect("should allocate"))
            .collect();

        let mut addrs: Vec<usize> = ptrs.iter().map(|p| p.as_ptr() as usize).collect();
        addrs.sort();
        addrs.dedup();
        assert_eq!(addrs.len(), 100);

        for ptr in ptrs.drain(..) {
            unsafe { pool.free(ptr) };
        }
    }

    #[test]
    fn global_stack_initial_distribution() {
        let pool = Pool::new(pool_config::<u64>(64, 8));

        let mut ptrs = vec![];
        for _ in 0..64 {
            ptrs.push(pool.alloc().expect("should allocate from global"));
        }
        assert!(pool.alloc().is_none());

        for ptr in ptrs {
            unsafe { pool.free(ptr) };
        }
    }

    #[test]
    fn cross_thread_alloc_free() {
        use std::{sync::Arc, thread};

        let slot_count = 1024;
        let iterations = 8;

        let pool = Arc::new(Pool::new(pool_config::<[u8; 1500]>(slot_count, 32)));
        // Bounded channel creates backpressure so the producer can't
        // outrun the consumer by more than the pool capacity.
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<SendPtr<[u8; 1500]>>(slot_count as usize / 2);

        let pool2 = Arc::clone(&pool);
        let producer = thread::spawn(move || {
            for i in 0..slot_count * iterations {
                let ptr = pool2.alloc().expect("should allocate");
                unsafe { (*ptr.as_ptr())[0] = i as u8 };
                tx.send(SendPtr::new(ptr)).unwrap();
            }
        });

        let pool2 = Arc::clone(&pool);
        let consumer = thread::spawn(move || {
            for _ in 0..slot_count * iterations {
                let sptr = rx.recv().unwrap();
                unsafe { pool2.free(sptr.into_inner()) };
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    fn concurrent_alloc_free_stress() {
        use std::{sync::Arc, thread};

        let pool = Arc::new(Pool::new(pool_config::<u64>(256, 16)));
        let num_threads = 4;
        let ops_per_thread = 10_000;

        let mut handles = vec![];
        for _ in 0..num_threads {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                let mut held: Vec<NonNull<u64>> = Vec::new();
                for i in 0..ops_per_thread {
                    if i % 3 != 0 && held.len() < 60 {
                        if let Some(ptr) = pool.alloc() {
                            unsafe { ptr.as_ptr().write(i as u64) };
                            held.push(ptr);
                        }
                    } else if let Some(ptr) = held.pop() {
                        unsafe { pool.free(ptr) };
                    }
                }
                for ptr in held {
                    unsafe { pool.free(ptr) };
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
