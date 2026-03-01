//! # seqpool
//!
//! A high-performance, lock-free, pre-allocated object pool designed for
//! cross-thread allocation patterns common in network protocol implementations.
//!
//! ## Design
//!
//! seqpool is inspired by [snmalloc](https://github.com/microsoft/snmalloc)'s
//! message-passing architecture but specialized for a fixed-layout, pre-allocated
//! pool pattern:
//!
//! - **Fixed layout per pool**: The allocation size is baked into the type via generics,
//!   so the compiler eliminates all size-class branching.
//! - **Pre-allocated**: All memory is allocated up front, giving predictable memory usage.
//! - **Lock-free**: All operations use atomic instructions only (`&self`).
//! - **Cross-thread optimized**: Designed for the pattern where one thread allocates
//!   and another thread frees (e.g., packet TX/RX pipelines).
//! - **CPU local data structures**: On Linux, RSEQ is used for fast CPU-local cache lookups.
//!
//! ## Architecture
//!
//! The pool uses snmalloc's message-passing pattern with three levels:
//!
//! 1. **Per-CPU local free list**: Each CPU has a local singly-linked free list.
//!    Allocation pops from this list with zero atomics (via RSEQ on Linux,
//!    or thread-local fallback). This is the fast path.
//!
//! 2. **Per-CPU remote MPSC queue**: Each CPU has a lock-free multi-producer
//!    single-consumer queue. When thread B frees a slot that was allocated on
//!    CPU A, it pushes to CPU A's remote queue. When CPU A's local list is empty,
//!    it drains its remote queue in a single atomic exchange, recovering a whole
//!    batch of freed slots at once.
//!
//! 3. **Global free stack**: A Treiber stack used for initial slot distribution
//!    and as a fallback when both the local list and remote queue are empty.
//!
//! This design means freed slots flow back to the CPU that allocated them,
//! minimizing contention and keeping cache lines hot.

mod mpsc;
mod pool;
mod rseq;
mod slot;
mod treiber;

pub use pool::{Pool, PoolConfig, SendPtr, Storage};

// #[cfg(test)]
// mod tests;
