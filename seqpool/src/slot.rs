//! Slot metadata for tracking which CPU a slot was allocated from.
//!
//! Each slot in the pool has a small metadata entry that records its "home" CPU.
//! When a slot is freed (possibly from a different thread/CPU), this metadata
//! tells us which CPU's MPSC queue to push to.

use core::sync::atomic::{AtomicU16, Ordering};

/// Per-slot metadata. Stored in a separate compact array from the slot data.
#[repr(C)]
pub struct SlotMeta {
    /// The CPU index this slot was most recently allocated from.
    /// Used to route frees back to the correct MPSC queue.
    home_cpu: AtomicU16,
}

impl SlotMeta {
    pub const fn new() -> Self {
        Self {
            home_cpu: AtomicU16::new(0),
        }
    }

    /// Set the home CPU for this slot (called on allocation).
    #[inline]
    pub fn set_home_cpu(&self, cpu: u16) {
        self.home_cpu.store(cpu, Ordering::Relaxed);
    }

    /// Get the home CPU for this slot (called on free).
    #[inline]
    pub fn home_cpu(&self) -> u16 {
        self.home_cpu.load(Ordering::Relaxed)
    }
}
