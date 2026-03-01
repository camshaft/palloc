//! Restartable sequences (RSEQ) support for CPU-local operations.
//!
//! On Linux, RSEQ allows executing a critical section that is atomically
//! committed relative to the current CPU. If the thread is preempted or
//! migrated, the kernel restarts the critical section from the beginning.
//! This gives us zero-atomic-instruction access to per-CPU data.
//!
//! ## Why not just read cpu_id?
//!
//! If we just read the CPU ID and then accessed per-CPU data with separate
//! instructions, we could get migrated between the read and the access —
//! then we'd be mutating the wrong CPU's data structure without any
//! synchronization. RSEQ solves this: the *entire* read-CPU + access-data
//! sequence is a critical section that restarts if we migrate.
//!
//! ## Operations
//!
//! This module provides two critical section operations for the pool:
//!
//! - [`per_cpu_pop`]: Pop a slot index from a per-CPU free list (alloc fast path)
//! - [`per_cpu_push`]: Push a slot index to a per-CPU free list (same-CPU free)
//!
//! Per-CPU heads are `CachePadded<AtomicU32>` to prevent false sharing between
//! CPUs — each head gets its own cache line (64B on x86, 128B on Apple Silicon).
//! On non-Linux or if RSEQ fails, operations return `Err(cpu_id_hint)` so the
//! caller can fall back to atomic operations.

use core::sync::atomic::AtomicU32;
use crossbeam_utils::CachePadded;

pub type CpuHead = CachePadded<AtomicU32>;

/// Null sentinel for free list links.
pub const NULL: u32 = u32::MAX;

/// Stride in bytes between per-CPU heads (= CachePadded alignment).
const HEAD_STRIDE: usize = size_of::<CpuHead>();

/// log2 of the stride, for shift-based addressing in assembly.
const HEAD_SHIFT: u32 = HEAD_STRIDE.trailing_zeros();

// Sanity: CachePadded must be a power of 2.
const _: () = assert!(HEAD_STRIDE.is_power_of_two());

/// Number of possible CPU indices (used to size per-CPU arrays).
pub fn possible_cpus() -> usize {
    #[cfg(target_os = "linux")]
    {
        linux::possible_cpus()
    }
    #[cfg(not(target_os = "linux"))]
    {
        fallback::possible_cpus()
    }
}

/// Get the current CPU index (best-effort, may be stale).
///
/// For operations that need atomicity with respect to CPU identity,
/// use [`per_cpu_pop`] or [`per_cpu_push`] instead.
#[inline]
pub fn cpu_id() -> usize {
    #[cfg(target_os = "linux")]
    {
        linux::cpu_id()
    }
    #[cfg(not(target_os = "linux"))]
    {
        fallback::cpu_id()
    }
}

/// Pop a slot index from the per-CPU free list head at `heads[cpu]`.
///
/// On success, returns `Ok((slot_index, cpu))`.
/// On failure (no RSEQ, empty list, or repeated preemption), returns
/// `Err(cpu_hint)` where `cpu_hint` is the last observed CPU.
///
/// # Safety
///
/// - `heads` must have length `>= possible_cpus()`.
/// - `next` must have length `>= pool_capacity`.
/// - The `heads` and `next` arrays must be valid for the duration of the call.
#[inline]
pub unsafe fn per_cpu_pop(heads: &[CpuHead], next: &[AtomicU32]) -> Result<(u32, usize), usize> {
    #[cfg(target_os = "linux")]
    {
        linux::per_cpu_pop(heads, next)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (heads, next);
        Err(fallback::cpu_id())
    }
}

/// Push a slot index onto the per-CPU free list head at `heads[cpu]`.
///
/// On success, returns `Ok(cpu)` with the CPU the push was committed on.
/// On failure, returns `Err(cpu_hint)`.
///
/// # Safety
///
/// - `heads` must have length `>= possible_cpus()`.
/// - `next` must have length `> index`.
/// - `index` must not already be on any free list.
#[inline]
pub unsafe fn per_cpu_push(
    heads: &[CpuHead],
    next: &[AtomicU32],
    index: u32,
) -> Result<usize, usize> {
    #[cfg(target_os = "linux")]
    {
        linux::per_cpu_push(heads, next, index)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (heads, next, index);
        Err(fallback::cpu_id())
    }
}

/// Whether RSEQ is available on this platform+thread.
#[inline]
pub fn is_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        linux::is_available()
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

// ============================================================================
// Linux RSEQ implementation
// ============================================================================
#[cfg(target_os = "linux")]
pub(crate) mod linux {
    use core::sync::atomic::AtomicU32;
    use crossbeam_utils::CachePadded;
    use std::{
        cell::Cell,
        ffi::CStr,
        ptr::NonNull,
        sync::atomic::{AtomicBool, Ordering},
    };

    use super::{HEAD_SHIFT, NULL};

    /// The kernel RSEQ ABI structure.
    #[repr(C, align(32))]
    pub struct Rseq {
        pub cpu_id_start: u32,
        pub cpu_id: u32,
        pub rseq_cs: u64,
        pub flags: u32,
    }

    #[cfg(target_arch = "x86_64")]
    pub const RSEQ_SIG: u32 = 0x53053053;

    #[cfg(target_arch = "aarch64")]
    pub const RSEQ_SIG: u32 = 0xd428bc00;

    thread_local! {
        static RSEQ_PTR: Cell<Option<NonNull<Rseq>>> = const { Cell::new(None) };
        static RSEQ_ALLOC: Cell<Option<RseqStorage>> = const { Cell::new(None) };
    }

    struct RseqStorage {
        slot: Box<Rseq>,
        registered: bool,
    }

    impl Drop for RseqStorage {
        fn drop(&mut self) {
            let Some(taken_address) = RSEQ_PTR.take() else {
                return;
            };
            if !self.registered {
                return;
            }
            if let Err(e) = sys_rseq(taken_address.as_ptr(), 1) {
                eprintln!("failed to deregister rseq on thread death: {e:?}");
            }
        }
    }

    static RSEQ_INIT_FAILED: AtomicBool = AtomicBool::new(false);

    #[inline]
    pub fn rseq() -> Option<NonNull<Rseq>> {
        if let Some(ptr) = RSEQ_PTR.get() {
            return Some(ptr);
        }
        if RSEQ_INIT_FAILED.load(Ordering::Relaxed) {
            return None;
        }
        rseq_init()
    }

    #[inline]
    pub fn cpu_id() -> usize {
        if let Some(ptr) = rseq() {
            let cpu = unsafe { (*ptr.as_ptr()).cpu_id_start };
            if cpu == u32::MAX {
                return super::fallback::cpu_id();
            }
            cpu as usize
        } else {
            super::fallback::cpu_id()
        }
    }

    pub fn is_available() -> bool {
        rseq().is_some()
    }

    pub fn possible_cpus() -> usize {
        use std::fs;

        let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/possible") else {
            return super::fallback::possible_cpus();
        };

        let max_cpu = content
            .trim()
            .split(',')
            .map(|range| {
                if let Some((_start, end)) = range.split_once('-') {
                    end.parse::<usize>().unwrap_or(0)
                } else {
                    range.parse::<usize>().unwrap_or(0)
                }
            })
            .max()
            .unwrap_or(0);

        (max_cpu + 1).max(1)
    }

    // ========================================================================
    // RSEQ critical sections
    // ========================================================================
    //
    // The per-CPU heads array uses CpuHead, so each entry is
    // HEAD_STRIDE bytes apart (64 on x86, 128 on aarch64 Apple Silicon).
    // We compute byte offsets as `cpu << HEAD_SHIFT` rather than `cpu * 4`.
    //
    // The `next` array is plain AtomicU32 (4 bytes each, packed).

    /// Pop from per-CPU local free list via RSEQ.
    ///
    /// Critical section:
    /// 1. Read cpu_id_start, bounds-check
    /// 2. Verify cpu_id matches (still on same CPU)
    /// 3. Compute &heads[cpu] via `base + (cpu << HEAD_SHIFT)`
    /// 4. Load head value — if NULL, bail
    /// 5. Load next[head] (4-byte stride)
    /// 6. Store next[head] into heads[cpu] — COMMIT
    #[inline]
    pub unsafe fn per_cpu_pop(
        heads: &[CpuHead],
        next: &[AtomicU32],
    ) -> Result<(u32, usize), usize> {
        let Some(rseq_ptr) = rseq() else {
            return Err(super::fallback::cpu_id());
        };
        per_cpu_pop_inner(rseq_ptr, heads, next)
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn per_cpu_pop_inner(
        rseq_ptr: NonNull<Rseq>,
        heads: &[CpuHead],
        next: &[AtomicU32],
    ) -> Result<(u32, usize), usize> {
        let result_index: u64;
        let result_cpu: u64;
        let success: u8;

        std::arch::asm!(
            // ---- rseq descriptor ----
            ".pushsection __rseq_cs, \"aw\"",
            ".balign 32",
            "77:",
            ".long 0",          // version
            ".long 0",          // flags
            ".quad 20f",        // start_ip
            ".quad (21f-20f)",  // post_commit_offset
            ".quad 29f",        // abort_ip
            ".popsection",

            // ---- abort handler prefix (RSEQ_SIG before label) ----
            "jmp 29f",
            ".long {RSEQ_SIG}",
            "29:",
            // Aborted — set failure flag and skip to end.
            "mov {success}, 0",
            "jmp 22f",

            // ---- pre-critical-section setup ----
            // Read cpu_id_start
            "mov {cpu:e}, [{rseq_ptr}+{cpu_id_offset_start}]",

            // Bounds check (cpu_id_start >= heads_len means out of range or unregistered)
            "cmp {cpu}, {heads_len}",
            "jge 29b",

            // Retry limit
            "dec {loop_count}",
            "jz 29b",

            // Compute byte offset for heads[cpu]: cpu << HEAD_SHIFT
            "mov {head_off}, {cpu}",
            "shl {head_off}, {HEAD_SHIFT}",

            // Install rseq_cs
            "lea {tmp}, [rip+77b]",
            "mov [{rseq_ptr}+{rseq_cs_offset}], {tmp}",

            // ---- critical section start ----
            "20:",

            // Verify still same CPU
            "cmp {cpu:e}, [{rseq_ptr}+{cpu_id_offset}]",
            "jnz 29b",

            // Load heads[cpu] (at heads_base + head_off)
            "mov {result:e}, [{heads_base}+{head_off}]",

            // If NULL, list is empty
            "cmp {result:e}, {NULL}",
            "je 29b",

            // Load next[result] (4 bytes each, packed)
            "mov {tmp:e}, [{next_base}+{result}*4]",

            // COMMIT: store next into heads[cpu]
            "mov [{heads_base}+{head_off}], {tmp:e}",

            // ---- post commit ----
            "21:",
            "mov {success}, 1",

            // ---- cleanup ----
            "22:",
            // Clear rseq_cs
            "mov QWORD PTR [{rseq_ptr}+{rseq_cs_offset}], 0",

            rseq_ptr = in(reg) rseq_ptr.as_ptr(),
            cpu = out(reg) result_cpu,
            head_off = out(reg) _,
            result = out(reg) result_index,
            tmp = out(reg) _,
            loop_count = inout(reg) 5u64 => _,
            heads_base = in(reg) heads.as_ptr(),
            heads_len = in(reg) heads.len() as u64,
            next_base = in(reg) next.as_ptr(),
            success = out(reg_byte) success,
            cpu_id_offset = const core::mem::offset_of!(Rseq, cpu_id),
            cpu_id_offset_start = const core::mem::offset_of!(Rseq, cpu_id_start),
            rseq_cs_offset = const core::mem::offset_of!(Rseq, rseq_cs),
            RSEQ_SIG = const RSEQ_SIG,
            NULL = const NULL,
            HEAD_SHIFT = const HEAD_SHIFT,
            options(nostack),
        );

        if success != 0 {
            Ok((result_index as u32, result_cpu as usize))
        } else {
            Err(result_cpu as usize)
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn per_cpu_pop_inner(
        rseq_ptr: NonNull<Rseq>,
        heads: &[CpuHead],
        next: &[AtomicU32],
    ) -> Result<(u32, usize), usize> {
        let result_index: u64;
        let result_cpu: u64;
        let success: u64;

        std::arch::asm!(
            ".pushsection __rseq_cs, \"aw\"",
            ".balign 32",
            "77:",
            ".long 0",
            ".long 0",
            ".quad 20f",
            ".quad (21f-20f)",
            ".quad 29f",
            ".popsection",

            "b 29f",
            ".long {RSEQ_SIG}",
            "29:",
            "mov {success}, 0",
            "b 22f",

            // Read cpu_id_start
            "ldr {cpu:w}, [{rseq_ptr}, #{cpu_id_offset_start}]",

            // Bounds check
            "cmp {cpu:w}, {heads_len:w}",
            "b.ge 29b",

            // Retry counter
            "subs {loop_count}, {loop_count}, #1",
            "b.eq 29b",

            // Compute byte offset: cpu << HEAD_SHIFT
            "lsl {head_off}, {cpu}, {HEAD_SHIFT}",

            "adrp {tmp}, 77b",
            "add {tmp}, {tmp}, #:lo12:77b",
            "str {tmp}, [{rseq_ptr}, #{rseq_cs_offset}]",

            // ---- critical section ----
            "20:",

            // Verify same CPU
            "ldr {tmp:w}, [{rseq_ptr}, #{cpu_id_offset}]",
            "cmp {cpu:w}, {tmp:w}",
            "b.ne 29b",

            // Load heads[cpu]
            "ldr {result:w}, [{heads_base}, {head_off}]",

            // Check NULL
            "cmp {result:w}, {NULL:w}",
            "b.eq 29b",

            // Load next[result] (4-byte stride)
            "add {tmp}, {next_base}, {result}, lsl #2",
            "ldr {tmp:w}, [{tmp}]",

            // COMMIT: store next into heads[cpu]
            "str {tmp:w}, [{heads_base}, {head_off}]",

            "21:",
            "mov {success}, 1",

            "22:",
            "str xzr, [{rseq_ptr}, #{rseq_cs_offset}]",

            rseq_ptr = in(reg) rseq_ptr.as_ptr(),
            cpu = out(reg) result_cpu,
            head_off = out(reg) _,
            result = out(reg) result_index,
            tmp = out(reg) _,
            loop_count = inout(reg) 5u64 => _,
            heads_base = in(reg) heads.as_ptr(),
            heads_len = in(reg) heads.len() as u64,
            next_base = in(reg) next.as_ptr(),
            success = out(reg) success,
            cpu_id_offset = const core::mem::offset_of!(Rseq, cpu_id),
            cpu_id_offset_start = const core::mem::offset_of!(Rseq, cpu_id_start),
            rseq_cs_offset = const core::mem::offset_of!(Rseq, rseq_cs),
            RSEQ_SIG = const RSEQ_SIG,
            NULL = in(reg) NULL as u64,
            HEAD_SHIFT = in(reg) HEAD_SHIFT as u64,
            options(nostack),
        );

        if success != 0 {
            Ok((result_index as u32, result_cpu as usize))
        } else {
            Err(result_cpu as usize)
        }
    }

    /// Push onto per-CPU local free list via RSEQ.
    ///
    /// Critical section:
    /// 1. Read cpu_id_start, bounds-check
    /// 2. Verify cpu_id matches
    /// 3. Load current heads[cpu]
    /// 4. Store old head into next[index]
    /// 5. Store index into heads[cpu] — COMMIT
    #[inline]
    pub unsafe fn per_cpu_push(
        heads: &[CpuHead],
        next: &[AtomicU32],
        index: u32,
    ) -> Result<usize, usize> {
        let Some(rseq_ptr) = rseq() else {
            return Err(super::fallback::cpu_id());
        };
        per_cpu_push_inner(rseq_ptr, heads, next, index)
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn per_cpu_push_inner(
        rseq_ptr: NonNull<Rseq>,
        heads: &[CpuHead],
        next: &[AtomicU32],
        index: u32,
    ) -> Result<usize, usize> {
        let result_cpu: u64;
        let success: u8;

        std::arch::asm!(
            ".pushsection __rseq_cs, \"aw\"",
            ".balign 32",
            "88:",
            ".long 0",
            ".long 0",
            ".quad 30f",
            ".quad (31f-30f)",
            ".quad 39f",
            ".popsection",

            "jmp 39f",
            ".long {RSEQ_SIG}",
            "39:",
            "mov {success}, 0",
            "jmp 32f",

            // Read cpu_id_start
            "mov {cpu:e}, [{rseq_ptr}+{cpu_id_offset_start}]",

            // Bounds check
            "cmp {cpu}, {heads_len}",
            "jge 39b",

            // Retry
            "dec {loop_count}",
            "jz 39b",

            // Compute byte offset: cpu << HEAD_SHIFT
            "mov {head_off}, {cpu}",
            "shl {head_off}, {HEAD_SHIFT}",

            "lea {tmp}, [rip+88b]",
            "mov [{rseq_ptr}+{rseq_cs_offset}], {tmp}",

            // ---- critical section ----
            "30:",

            // Verify same CPU
            "cmp {cpu:e}, [{rseq_ptr}+{cpu_id_offset}]",
            "jnz 39b",

            // Load current heads[cpu]
            "mov {tmp:e}, [{heads_base}+{head_off}]",

            // Store old head into next[index]
            "mov [{next_base}+{index}*4], {tmp:e}",

            // COMMIT: store index into heads[cpu]
            "mov [{heads_base}+{head_off}], {index:e}",

            "31:",
            "mov {success}, 1",

            "32:",
            "mov QWORD PTR [{rseq_ptr}+{rseq_cs_offset}], 0",

            rseq_ptr = in(reg) rseq_ptr.as_ptr(),
            cpu = out(reg) result_cpu,
            head_off = out(reg) _,
            tmp = out(reg) _,
            loop_count = inout(reg) 5u64 => _,
            heads_base = in(reg) heads.as_ptr(),
            heads_len = in(reg) heads.len() as u64,
            next_base = in(reg) next.as_ptr(),
            index = in(reg) index as u64,
            success = out(reg_byte) success,
            cpu_id_offset = const core::mem::offset_of!(Rseq, cpu_id),
            cpu_id_offset_start = const core::mem::offset_of!(Rseq, cpu_id_start),
            rseq_cs_offset = const core::mem::offset_of!(Rseq, rseq_cs),
            RSEQ_SIG = const RSEQ_SIG,
            HEAD_SHIFT = const HEAD_SHIFT,
            options(nostack),
        );

        if success != 0 {
            Ok(result_cpu as usize)
        } else {
            Err(result_cpu as usize)
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn per_cpu_push_inner(
        rseq_ptr: NonNull<Rseq>,
        heads: &[CpuHead],
        next: &[AtomicU32],
        index: u32,
    ) -> Result<usize, usize> {
        let result_cpu: u64;
        let success: u64;

        std::arch::asm!(
            ".pushsection __rseq_cs, \"aw\"",
            ".balign 32",
            "88:",
            ".long 0",
            ".long 0",
            ".quad 30f",
            ".quad (31f-30f)",
            ".quad 39f",
            ".popsection",

            "b 39f",
            ".long {RSEQ_SIG}",
            "39:",
            "mov {success}, 0",
            "b 32f",

            // Read cpu_id_start
            "ldr {cpu:w}, [{rseq_ptr}, #{cpu_id_offset_start}]",

            // Bounds check
            "cmp {cpu:w}, {heads_len:w}",
            "b.ge 39b",

            // Retry
            "subs {loop_count}, {loop_count}, #1",
            "b.eq 39b",

            // Compute byte offset
            "lsl {head_off}, {cpu}, {HEAD_SHIFT}",

            "adrp {tmp}, 88b",
            "add {tmp}, {tmp}, #:lo12:88b",
            "str {tmp}, [{rseq_ptr}, #{rseq_cs_offset}]",

            // ---- critical section ----
            "30:",

            // Verify same CPU
            "ldr {tmp:w}, [{rseq_ptr}, #{cpu_id_offset}]",
            "cmp {cpu:w}, {tmp:w}",
            "b.ne 39b",

            // Load current heads[cpu]
            "ldr {tmp:w}, [{heads_base}, {head_off}]",

            // Store old head into next[index]
            "add {next_addr}, {next_base}, {index}, lsl #2",
            "str {tmp:w}, [{next_addr}]",

            // COMMIT: store index into heads[cpu]
            "str {index:w}, [{heads_base}, {head_off}]",

            "31:",
            "mov {success}, 1",

            "32:",
            "str xzr, [{rseq_ptr}, #{rseq_cs_offset}]",

            rseq_ptr = in(reg) rseq_ptr.as_ptr(),
            cpu = out(reg) result_cpu,
            head_off = out(reg) _,
            tmp = out(reg) _,
            next_addr = out(reg) _,
            loop_count = inout(reg) 5u64 => _,
            heads_base = in(reg) heads.as_ptr(),
            heads_len = in(reg) heads.len() as u64,
            next_base = in(reg) next.as_ptr(),
            index = in(reg) index as u64,
            success = out(reg) success,
            cpu_id_offset = const core::mem::offset_of!(Rseq, cpu_id),
            cpu_id_offset_start = const core::mem::offset_of!(Rseq, cpu_id_start),
            rseq_cs_offset = const core::mem::offset_of!(Rseq, rseq_cs),
            RSEQ_SIG = const RSEQ_SIG,
            HEAD_SHIFT = in(reg) HEAD_SHIFT as u64,
            options(nostack),
        );

        if success != 0 {
            Ok(result_cpu as usize)
        } else {
            Err(result_cpu as usize)
        }
    }

    // ========================================================================
    // RSEQ registration
    // ========================================================================

    #[cold]
    fn rseq_init() -> Option<NonNull<Rseq>> {
        if let Ok(libc_rseq) = from_libc() {
            if let Some(ptr) = NonNull::new(libc_rseq) {
                RSEQ_PTR.set(Some(ptr));
                return Some(ptr);
            }
        }

        let rseq_storage = RseqStorage {
            slot: Box::new(Rseq {
                cpu_id_start: u32::MAX,
                cpu_id: u32::MAX,
                rseq_cs: 0,
                flags: 0,
            }),
            registered: false,
        };

        let ptr = NonNull::new(&raw const *rseq_storage.slot as *mut Rseq).unwrap();
        RSEQ_PTR.set(Some(ptr));
        RSEQ_ALLOC.set(Some(rseq_storage));

        match sys_rseq(ptr.as_ptr(), 0) {
            Ok(()) => {
                RSEQ_ALLOC.with(|c| {
                    let mut v = c.take().expect("just set above");
                    v.registered = true;
                    c.set(Some(v));
                });
                Some(ptr)
            }
            Err(_e) => {
                RSEQ_INIT_FAILED.store(true, Ordering::Relaxed);
                None
            }
        }
    }

    fn dlsym(symbol: &CStr) -> std::io::Result<*mut std::ffi::c_void> {
        unsafe {
            let _ = libc::dlerror();
            let address = libc::dlsym(libc::RTLD_DEFAULT, symbol.as_ptr());
            if let Some(ptr) = NonNull::new(libc::dlerror()) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "failed to dlsym {symbol:?}: {:?}",
                        std::ffi::CStr::from_ptr(ptr.as_ptr())
                    ),
                ));
            }
            Ok(address)
        }
    }

    fn thread_plus_offset(offset: libc::ptrdiff_t) -> *mut std::ffi::c_void {
        let output: *mut std::ffi::c_void;
        unsafe {
            #[cfg(target_arch = "aarch64")]
            std::arch::asm!("mrs {output}, tpidr_el0", output = out(reg) output);

            #[cfg(target_arch = "x86_64")]
            std::arch::asm!("mov {output}, fs:0", output = out(reg) output);
        }
        output.wrapping_offset(offset)
    }

    fn from_libc() -> std::io::Result<*mut Rseq> {
        let _size = dlsym(c"__rseq_size")?.cast::<u32>();
        let offset = dlsym(c"__rseq_offset")?.cast::<libc::ptrdiff_t>();
        let _flags = dlsym(c"__rseq_flags")?.cast::<u32>();
        Ok(thread_plus_offset(unsafe { offset.read() }).cast())
    }

    fn sys_rseq(rseq_abi: *mut Rseq, flags: i32) -> std::io::Result<()> {
        let ret = unsafe {
            libc::syscall(
                libc::SYS_rseq,
                rseq_abi,
                core::mem::size_of::<Rseq>() as u32,
                flags,
                RSEQ_SIG,
            )
        };
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

// ============================================================================
// Non-Linux fallback
// ============================================================================
pub(crate) mod fallback {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static CPU_ID: usize = {
            let cpus = possible_cpus();
            NEXT_ID.fetch_add(1, Ordering::Relaxed) % cpus
        };
    }

    #[inline]
    pub fn cpu_id() -> usize {
        CPU_ID.with(|&id| id)
    }

    pub fn possible_cpus() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_id_in_range() {
        let cpus = possible_cpus();
        assert!(cpus >= 1);
        let id = cpu_id();
        assert!(id < cpus, "cpu_id {id} >= possible_cpus {cpus}");
    }

    #[test]
    fn cpu_id_consistent_on_fallback() {
        if !is_available() {
            let id1 = cpu_id();
            let id2 = cpu_id();
            assert_eq!(id1, id2);
        }
    }

    #[test]
    fn cpu_id_across_threads() {
        use std::{sync::Arc, thread};

        let cpus = possible_cpus();
        let results: Arc<std::sync::Mutex<Vec<usize>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut handles = vec![];
        for _ in 0..cpus.min(8) {
            let results = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                let id = cpu_id();
                assert!(id < cpus);
                results.lock().unwrap().push(id);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let results = results.lock().unwrap();
        assert_eq!(results.len(), cpus.min(8));
    }

    #[test]
    fn possible_cpus_reasonable() {
        let cpus = possible_cpus();
        assert!(cpus >= 1);
        assert!(cpus <= 1024);
    }

    #[test]
    fn head_stride_is_cache_line() {
        // CachePadded should give us at least 64 bytes
        assert!(HEAD_STRIDE >= 64);
        assert!(HEAD_STRIDE.is_power_of_two());
    }
}
