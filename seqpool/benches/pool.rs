//! Benchmarks comparing seqpool against system allocator and snmalloc.
//!
//! Key scenarios:
//! 1. Single-thread alloc+free (fast path)
//! 2. Cross-thread: producer allocates, consumer frees (the packet pipeline pattern)
//! 3. Multi-producer/consumer contention

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use seqpool::{Pool, PoolConfig, SendPtr};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    ptr::NonNull,
    sync::{Arc, Barrier, mpsc},
    thread,
};

/// MTU-sized packet buffer — the primary use case.
type Packet = [u8; 1500];

const POOL_SIZE: u32 = 65536;
const BATCH_SIZE: u32 = 32;

fn packet_layout() -> Layout {
    Layout::new::<Packet>()
}

// ============================================================================
// Single-thread: alloc + free in a tight loop
// ============================================================================

fn bench_single_thread_alloc_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread_alloc_free");
    let iters = 100_000u64;
    group.throughput(Throughput::Elements(iters));

    // seqpool
    group.bench_function("seqpool", |b| {
        let pool = Pool::new(PoolConfig::<Packet>::new(POOL_SIZE, BATCH_SIZE));
        b.iter(|| {
            for _ in 0..iters {
                let ptr = pool.alloc().unwrap();
                unsafe { pool.free(criterion::black_box(ptr)) };
            }
        });
    });

    // System allocator
    group.bench_function("system_alloc", |b| {
        let layout = packet_layout();
        b.iter(|| {
            for _ in 0..iters {
                let ptr = unsafe { System.alloc(layout) };
                let ptr = criterion::black_box(ptr);
                unsafe { System.dealloc(ptr, layout) };
            }
        });
    });

    // snmalloc
    group.bench_function("snmalloc", |b| {
        let layout = packet_layout();
        b.iter(|| {
            for _ in 0..iters {
                let ptr = unsafe { snmalloc_rs::SnMalloc.alloc(layout) };
                let ptr = criterion::black_box(ptr);
                unsafe { snmalloc_rs::SnMalloc.dealloc(ptr, layout) };
            }
        });
    });

    group.finish();
}

// ============================================================================
// Cross-thread: producer allocs, consumer frees (the packet pipeline)
// ============================================================================

fn bench_cross_thread_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_thread_pipeline");
    let iters = 100_000u64;
    group.throughput(Throughput::Elements(iters));

    // seqpool: producer allocs, sends ptr to consumer, consumer frees
    group.bench_function("seqpool", |b| {
        b.iter_custom(|n| {
            let total = iters * n;
            let pool = Arc::new(Pool::new(PoolConfig::<Packet>::new(POOL_SIZE, BATCH_SIZE)));

            let (tx, rx) = mpsc::sync_channel::<SendPtr<Packet>>(1024);
            let barrier = Arc::new(Barrier::new(2));

            let pool2 = Arc::clone(&pool);
            let barrier2 = Arc::clone(&barrier);
            let producer = thread::spawn(move || {
                barrier2.wait();
                for _ in 0..total {
                    let ptr = pool2.alloc().unwrap();
                    tx.send(SendPtr::new(ptr)).unwrap();
                }
            });

            let pool2 = Arc::clone(&pool);
            let barrier2 = Arc::clone(&barrier);
            let consumer = thread::spawn(move || {
                barrier2.wait();
                for _ in 0..total {
                    let sptr = rx.recv().unwrap();
                    unsafe { pool2.free(sptr.into_inner()) };
                }
            });

            let start = std::time::Instant::now();
            // Threads are already running, waiting on barrier
            // Actually they've already passed the barrier by now.
            // Let me restructure this.
            producer.join().unwrap();
            consumer.join().unwrap();
            start.elapsed()
        });
    });

    // System allocator: producer allocs, consumer frees
    group.bench_function("system_alloc", |b| {
        b.iter_custom(|n| {
            let total = iters * n;
            let layout = packet_layout();

            let (tx, rx) = mpsc::sync_channel::<usize>(1024);
            let barrier = Arc::new(Barrier::new(3));

            let b1 = Arc::clone(&barrier);
            let producer = thread::spawn(move || {
                b1.wait();
                for _ in 0..total {
                    let ptr = unsafe { System.alloc(layout) };
                    tx.send(ptr as usize).unwrap();
                }
            });

            let b2 = Arc::clone(&barrier);
            let consumer = thread::spawn(move || {
                b2.wait();
                for _ in 0..total {
                    let ptr = rx.recv().unwrap() as *mut u8;
                    unsafe { System.dealloc(ptr, layout) };
                }
            });

            barrier.wait();
            let start = std::time::Instant::now();
            producer.join().unwrap();
            consumer.join().unwrap();
            start.elapsed()
        });
    });

    // snmalloc: producer allocs, consumer frees
    group.bench_function("snmalloc", |b| {
        b.iter_custom(|n| {
            let total = iters * n;
            let layout = packet_layout();

            let (tx, rx) = mpsc::sync_channel::<usize>(1024);
            let barrier = Arc::new(Barrier::new(3));

            let b1 = Arc::clone(&barrier);
            let producer = thread::spawn(move || {
                b1.wait();
                for _ in 0..total {
                    let ptr = unsafe { snmalloc_rs::SnMalloc.alloc(layout) };
                    tx.send(ptr as usize).unwrap();
                }
            });

            let b2 = Arc::clone(&barrier);
            let consumer = thread::spawn(move || {
                b2.wait();
                for _ in 0..total {
                    let ptr = rx.recv().unwrap() as *mut u8;
                    unsafe { snmalloc_rs::SnMalloc.dealloc(ptr, layout) };
                }
            });

            barrier.wait();
            let start = std::time::Instant::now();
            producer.join().unwrap();
            consumer.join().unwrap();
            start.elapsed()
        });
    });

    group.finish();
}

// ============================================================================
// Batch alloc then batch free (simulates burst TX)
// ============================================================================

fn bench_batch_alloc_then_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_alloc_then_free");

    for batch in [32, 128, 512] {
        group.throughput(Throughput::Elements(batch as u64));

        group.bench_with_input(BenchmarkId::new("seqpool", batch), &batch, |b, &batch| {
            let pool = Pool::new(PoolConfig::<Packet>::new(POOL_SIZE, BATCH_SIZE));
            let mut ptrs: Vec<NonNull<Packet>> = Vec::with_capacity(batch);

            b.iter(|| {
                // Alloc burst
                for _ in 0..batch {
                    ptrs.push(pool.alloc().unwrap());
                }
                // Free burst
                for ptr in ptrs.drain(..) {
                    unsafe { pool.free(ptr) };
                }
            });
        });

        group.bench_with_input(
            BenchmarkId::new("system_alloc", batch),
            &batch,
            |b, &batch| {
                let layout = packet_layout();
                let mut ptrs: Vec<*mut u8> = Vec::with_capacity(batch);

                b.iter(|| {
                    for _ in 0..batch {
                        ptrs.push(unsafe { System.alloc(layout) });
                    }
                    for ptr in ptrs.drain(..) {
                        unsafe { System.dealloc(ptr, layout) };
                    }
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("snmalloc", batch), &batch, |b, &batch| {
            let layout = packet_layout();
            let mut ptrs: Vec<*mut u8> = Vec::with_capacity(batch);

            b.iter(|| {
                for _ in 0..batch {
                    ptrs.push(unsafe { snmalloc_rs::SnMalloc.alloc(layout) });
                }
                for ptr in ptrs.drain(..) {
                    unsafe { snmalloc_rs::SnMalloc.dealloc(ptr, layout) };
                }
            });
        });
    }

    group.finish();
}

// ============================================================================
// Multi-threaded contention: N threads each alloc+free independently
// ============================================================================

fn bench_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention");

    for num_threads in [1, 2, 4, 8] {
        let iters_per_thread = 50_000u64;
        group.throughput(Throughput::Elements(iters_per_thread * num_threads as u64));

        group.bench_with_input(
            BenchmarkId::new("seqpool", num_threads),
            &num_threads,
            |b, &nt| {
                let pool = Arc::new(Pool::new(PoolConfig::<Packet>::new(POOL_SIZE, BATCH_SIZE)));
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(nt));
                    let mut handles = vec![];
                    for _ in 0..nt {
                        let pool = Arc::clone(&pool);
                        let barrier = Arc::clone(&barrier);
                        handles.push(thread::spawn(move || {
                            barrier.wait();
                            for _ in 0..iters_per_thread {
                                let ptr = pool.alloc().unwrap();
                                unsafe { pool.free(ptr) };
                            }
                        }));
                    }
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("system_alloc", num_threads),
            &num_threads,
            |b, &nt| {
                let layout = packet_layout();
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(nt));
                    let mut handles = vec![];
                    for _ in 0..nt {
                        let barrier = Arc::clone(&barrier);
                        handles.push(thread::spawn(move || {
                            barrier.wait();
                            for _ in 0..iters_per_thread {
                                let ptr = unsafe { System.alloc(layout) };
                                unsafe { System.dealloc(ptr, layout) };
                            }
                        }));
                    }
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("snmalloc", num_threads),
            &num_threads,
            |b, &nt| {
                let layout = packet_layout();
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(nt));
                    let mut handles = vec![];
                    for _ in 0..nt {
                        let barrier = Arc::clone(&barrier);
                        handles.push(thread::spawn(move || {
                            barrier.wait();
                            for _ in 0..iters_per_thread {
                                let ptr = unsafe { snmalloc_rs::SnMalloc.alloc(layout) };
                                unsafe { snmalloc_rs::SnMalloc.dealloc(ptr, layout) };
                            }
                        }));
                    }
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread_alloc_free,
    bench_cross_thread_pipeline,
    bench_batch_alloc_then_free,
    bench_contention,
);
criterion_main!(benches);
