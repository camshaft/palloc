//! Benchmark suite comparing `seqpool` against `jemalloc` and `mimalloc`.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench pool
//! ```

use std::{
    alloc::{GlobalAlloc, Layout},
    hint::black_box,
    sync::{Arc, Barrier},
    thread,
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use seqpool::Pool;

// Packet buffer: 1500 bytes, 1-byte aligned (matches typical MTU).
const PACKET_LAYOUT: Layout = match Layout::from_size_align(1500, 1) {
    Ok(l) => l,
    Err(_) => panic!("invalid layout"),
};
const CAPACITY: usize = 1024;

// ──────────────────────────────────────────────────────────────────────────────
// Single-threaded: alloc immediately followed by free
// ──────────────────────────────────────────────────────────────────────────────

fn bench_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread/alloc_free");
    group.throughput(Throughput::Elements(1));

    // ── seqpool ──────────────────────────────────────────────────────────────
    group.bench_function("seqpool", |b| {
        let pool = Pool::new(PACKET_LAYOUT, CAPACITY);
        b.iter(|| {
            let slot = black_box(pool.alloc().unwrap());
            pool.free(slot);
        });
    });

    // ── jemalloc ─────────────────────────────────────────────────────────────
    group.bench_function("jemalloc", |b| {
        let alloc = tikv_jemallocator::Jemalloc;
        b.iter(|| unsafe {
            let ptr = black_box(alloc.alloc(PACKET_LAYOUT));
            alloc.dealloc(ptr, PACKET_LAYOUT);
        });
    });

    // ── mimalloc ─────────────────────────────────────────────────────────────
    group.bench_function("mimalloc", |b| {
        let alloc = mimalloc::MiMalloc;
        b.iter(|| unsafe {
            let ptr = black_box(alloc.alloc(PACKET_LAYOUT));
            alloc.dealloc(ptr, PACKET_LAYOUT);
        });
    });

    group.finish();
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-threaded: N threads each doing alloc+free in a tight loop
// ──────────────────────────────────────────────────────────────────────────────

fn bench_multi_thread(c: &mut Criterion) {
    let thread_counts = [1usize, 2, 4, 8];

    let mut group = c.benchmark_group("multi_thread/alloc_free");

    for &threads in &thread_counts {
        group.throughput(Throughput::Elements(threads as u64));

        // ── seqpool ──────────────────────────────────────────────────────────
        group.bench_with_input(BenchmarkId::new("seqpool", threads), &threads, |b, &t| {
            // Use at least 2× the thread count to avoid spurious None returns.
            let capacity = (t * 16).max(CAPACITY);
            let pool = Pool::new(PACKET_LAYOUT, capacity);
            b.iter_custom(|iters| {
                let barrier = Arc::new(Barrier::new(t + 1));
                let handles: Vec<_> = (0..t)
                    .map(|_| {
                        let pool = pool.clone();
                        let barrier = Arc::clone(&barrier);
                        let per_thread = iters / t as u64;
                        thread::spawn(move || {
                            barrier.wait();
                            for _ in 0..per_thread {
                                let slot = black_box(pool.alloc().unwrap());
                                pool.free(slot);
                            }
                        })
                    })
                    .collect();
                let start = std::time::Instant::now();
                barrier.wait();
                for h in handles {
                    h.join().unwrap();
                }
                start.elapsed()
            });
        });

        // ── jemalloc ─────────────────────────────────────────────────────────
        group.bench_with_input(BenchmarkId::new("jemalloc", threads), &threads, |b, &t| {
            b.iter_custom(|iters| {
                let barrier = Arc::new(Barrier::new(t + 1));
                let handles: Vec<_> = (0..t)
                    .map(|_| {
                        let barrier = Arc::clone(&barrier);
                        let per_thread = iters / t as u64;
                        thread::spawn(move || {
                            let alloc = tikv_jemallocator::Jemalloc;
                            barrier.wait();
                            for _ in 0..per_thread {
                                unsafe {
                                    let ptr = black_box(alloc.alloc(PACKET_LAYOUT));
                                    alloc.dealloc(ptr, PACKET_LAYOUT);
                                }
                            }
                        })
                    })
                    .collect();
                let start = std::time::Instant::now();
                barrier.wait();
                for h in handles {
                    h.join().unwrap();
                }
                start.elapsed()
            });
        });

        // ── mimalloc ─────────────────────────────────────────────────────────
        group.bench_with_input(BenchmarkId::new("mimalloc", threads), &threads, |b, &t| {
            b.iter_custom(|iters| {
                let barrier = Arc::new(Barrier::new(t + 1));
                let handles: Vec<_> = (0..t)
                    .map(|_| {
                        let barrier = Arc::clone(&barrier);
                        let per_thread = iters / t as u64;
                        thread::spawn(move || {
                            let alloc = mimalloc::MiMalloc;
                            barrier.wait();
                            for _ in 0..per_thread {
                                unsafe {
                                    let ptr = black_box(alloc.alloc(PACKET_LAYOUT));
                                    alloc.dealloc(ptr, PACKET_LAYOUT);
                                }
                            }
                        })
                    })
                    .collect();
                let start = std::time::Instant::now();
                barrier.wait();
                for h in handles {
                    h.join().unwrap();
                }
                start.elapsed()
            });
        });
    }

    group.finish();
}

// ──────────────────────────────────────────────────────────────────────────────
// Cross-thread: producer allocates, consumer frees
// ──────────────────────────────────────────────────────────────────────────────

/// Thin wrapper that makes a raw pointer sendable across threads.
///
/// SAFETY: in the benchmarks below, each pointer is allocated on the producer
/// thread and freed on the consumer thread with no further aliasing.
struct SendPtr(*mut u8);
// SAFETY: see above.
unsafe impl Send for SendPtr {}

fn bench_cross_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_thread/producer_consumer");
    group.throughput(Throughput::Elements(1));

    // ── seqpool ──────────────────────────────────────────────────────────────
    group.bench_function("seqpool", |b| {
        use std::sync::mpsc;

        let pool = Pool::new(PACKET_LAYOUT, CAPACITY);
        let pool_consumer = pool.clone();
        let (tx, rx) = mpsc::sync_channel::<seqpool::Slot>(CAPACITY);

        // Consumer runs in a dedicated thread; needs a pool handle to free.
        let consumer = thread::spawn(move || {
            while let Ok(slot) = rx.recv() {
                pool_consumer.free(black_box(slot));
            }
        });

        b.iter(|| {
            let slot = pool.alloc().unwrap();
            tx.send(slot).unwrap();
        });

        drop(tx);
        consumer.join().unwrap();
    });

    // ── jemalloc ─────────────────────────────────────────────────────────────
    group.bench_function("jemalloc", |b| {
        use std::sync::mpsc;

        let (tx, rx) = mpsc::sync_channel::<SendPtr>(CAPACITY);

        let consumer = thread::spawn(move || {
            let alloc = tikv_jemallocator::Jemalloc;
            while let Ok(SendPtr(ptr)) = rx.recv() {
                unsafe { alloc.dealloc(black_box(ptr), PACKET_LAYOUT) };
            }
        });

        let alloc = tikv_jemallocator::Jemalloc;
        b.iter(|| {
            let ptr = unsafe { alloc.alloc(PACKET_LAYOUT) };
            tx.send(SendPtr(ptr)).unwrap();
        });

        drop(tx);
        consumer.join().unwrap();
    });

    // ── mimalloc ─────────────────────────────────────────────────────────────
    group.bench_function("mimalloc", |b| {
        use std::sync::mpsc;

        let (tx, rx) = mpsc::sync_channel::<SendPtr>(CAPACITY);

        let consumer = thread::spawn(move || {
            let alloc = mimalloc::MiMalloc;
            while let Ok(SendPtr(ptr)) = rx.recv() {
                unsafe { alloc.dealloc(black_box(ptr), PACKET_LAYOUT) };
            }
        });

        let alloc = mimalloc::MiMalloc;
        b.iter(|| {
            let ptr = unsafe { alloc.alloc(PACKET_LAYOUT) };
            tx.send(SendPtr(ptr)).unwrap();
        });

        drop(tx);
        consumer.join().unwrap();
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread,
    bench_multi_thread,
    bench_cross_thread
);
criterion_main!(benches);
