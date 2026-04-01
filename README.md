# mcu-inference-bench

Hardware benchmark suite for neural network inference on microcontrollers. Measures the raw capabilities that determine inference performance: memory bandwidth, SIMD throughput, popcount cost, INT4 unpack strategies, cache effects, and streaming pipeline overhead.

Designed to be portable across MCU platforms. The ESP32-P4 is the first target. The same benchmark structure will run on Renesas RA8M2 (Cortex-M85 + Helium) with only the SIMD kernel swapped.

## Why This Exists

When deploying quantized neural networks on MCUs, published specs are misleading. The ESP32-P4 PIE extension claims 16 INT8 MACs per instruction -- but each instruction takes 4.3 cycles, giving 3.7 MACs/cycle in practice. PSRAM is rated at 200 MHz x16 -- but actual sequential read throughput is 104 MB/s, not the 400 MB/s you might calculate. These gaps change every architectural decision: which quantization to use, whether to tile through SRAM or stream from PSRAM, whether INT4 saves time or wastes it.

This benchmark suite measures everything that matters, so design decisions are based on real silicon, not datasheets.

## Structure

```
esp32p4/
  main/
    main.c             Entry point, system info, runs all benchmarks
    bench.h            Common utilities, timing macros, cycle counter, helpers
    bench_memory.c     SRAM/PSRAM bandwidth and latency
    bench_pie.c        PIE XACC throughput, instruction timing, PSRAM access
    bench_popcount.c   Popcount strategy comparison, XNOR matmul, XNOR vs PIE
    bench_matmul.c     INT8 matmul from SRAM and PSRAM, access strategies
    bench_cache.c      Flash vs SRAM, stride patterns, cache boundary detection
    bench_dma.c        Double-buffer simulation, INT4 pipeline, copy/compute ratio
    bench_int4.c       INT4 unpack strategies, deinterleaved unpack, pipeline impact
    pie_kernels.S      PIE XACC assembly (from proven mcu-leworldmodel project)
  CMakeLists.txt
  sdkconfig.defaults
```

## Building

```bash
cd esp32p4
idf.py set-target esp32p4
idf.py build
idf.py flash monitor
```

Requires ESP-IDF v5.5+ with ESP32-P4 and PSRAM support.

## What Each Benchmark Measures

### bench_memory.c -- Memory Bandwidth and Latency

Measures the fundamental memory system performance that determines whether inference is compute-bound or memory-bound.

- **Sequential read/write bandwidth** at 4KB to 1MB buffer sizes, separately for SRAM and PSRAM. Reveals the cache hierarchy: small buffers hit cache and show ~360 MB/s regardless of backing memory, large buffers show true PSRAM throughput (~104 MB/s read, ~85 MB/s write).
- **memcpy throughput** in all four directions (SRAM/PSRAM to SRAM/PSRAM) at 64KB. Shows the copy cost for weight tiling.
- **memcpy vs transfer size** (64 bytes to 64KB, PSRAM to SRAM). Reveals optimal tile size where cache amortization peaks (~16-32KB at 560 MB/s) before dropping at 64KB (372 MB/s).
- **Random read latency** via pointer chasing at various working set sizes. Measures true memory latency without prefetcher help.
- **Row copy simulation** at row sizes matching real weight matrices (128-4096 bytes). Directly predicts the per-row cost of PSRAM weight tiling.

### bench_pie.c -- PIE XACC SIMD Throughput

Measures the actual throughput of the ESP32-P4's custom SIMD extension, which is the core compute primitive for INT8 inference.

- **Single dot product** (PIE vs scalar C) at 16 to 4096 elements, using the RISC-V mcycle CSR for cycle-accurate timing. Shows PIE speedup scaling from 2.5x (16 elements) to 39.8x (4096 elements).
- **PIE instruction timing** on a 256x256 matmul: measures the actual cycles per `esp.vmulas.s8.xacc` instruction (4.3 cycles) and effective MACs/cycle (3.7-3.9).
- **Multi-row matmul** at dimensions matching real model layers, with both cycle counts and GMAC/s.
- **Call overhead** quantification: N separate single-row PIE calls vs one batched N-row call. The overhead is 1.37-1.55x, meaning batched calls are always preferable.
- **PIE memory source comparison** at various dimensions: SRAM (single call) vs PSRAM direct (per-row PIE calls reading PSRAM) vs PSRAM row-copy (memcpy to SRAM then PIE). Reveals that PSRAM direct beats row-copy for short rows (cols <= 512) but both are 7-9x slower than SRAM for long rows.

### bench_popcount.c -- Popcount and XNOR Throughput

Measures the cost of software popcount (no MCU has hardware popcount) and the resulting XNOR binary matmul throughput.

- **Five popcount strategies** compared with cycle-accurate timing: naive bit loop (137 cyc), Wegner/Kernighan (82 cyc), GCC __builtin_popcount (34.5 cyc), 256-byte LUT (21 cyc), bit-parallel Hacker's Delight (19 cyc).
- **Full XNOR matmul** at realistic layer sizes with all five strategies. Shows that fast_popcount delivers ~0.49 binary GMAC/s.
- **XNOR vs INT8 PIE** head-to-head at identical dimensions from SRAM. PIE is 2.1-2.8x faster, settling the question of whether binary inference has any compute advantage on this chip (it does not).

### bench_matmul.c -- End-to-End Matmul

Measures complete matrix-vector multiply performance as experienced by real inference code.

- **INT8 PIE from SRAM** at various dimensions. The compute-bound baseline: 1.39 GMAC/s peak at 128x256, dropping to 0.88 at 1024x256 (larger matrices have more loop overhead per MAC).
- **INT8 from PSRAM** with three access strategies compared at the same dimensions: SRAM direct (baseline), PSRAM direct (PIE reads PSRAM through cache), PSRAM row-copy (memcpy then PIE), PSRAM batch-copy (memcpy N rows then batched PIE). Shows the real cost of weight streaming.

### bench_cache.c -- Cache and Flash Access Patterns

Characterizes the cache hierarchy to inform tiling strategy.

- **Flash (.rodata) vs SRAM** sequential read at matching sizes. Flash is 1-2% slower due to cache miss overhead.
- **Stride access patterns** on PSRAM: sequential (32 ns/access) vs stride-16+ (55 ns/access). Shows the cache line size effect.
- **Working set sweep** from 1KB to 512KB with pointer chasing. Detects cache boundaries: L1 at ~32KB (25 ns), L2 at ~128KB (37.5 ns), PSRAM uncached at 256KB+ (288-392 ns). This is the most important cache result -- it defines the optimal tile buffer size.

### bench_dma.c -- Double-Buffer and Pipeline Simulation

Measures the overlap potential between memory transfer and compute, which determines whether DMA double-buffering is worthwhile.

- **Double-buffer simulation** at batch sizes from 1 to 512 rows. Measures copy time and compute time separately. The copy/compute ratio peaks at ~11:1, meaning compute is 11x faster than the memory transfer. Even perfect overlap (real DMA) would only save ~10% of total time.
- **INT4 pipeline breakdown**: PSRAM copy + scalar unpack + PIE compute, each timed separately. Shows unpack is 74.5% of total time -- the critical bottleneck for INT4.
- **INT4 vs INT8 from PSRAM** direct comparison: INT4 is 0.46x the speed (slower, not faster) because unpack cost exceeds the bandwidth savings.

### bench_int4.c -- INT4 Unpack Strategies

Dedicated analysis of the INT4 unpacking bottleneck identified by the DMA benchmark.

- **Six unpack strategies** compared: naive byte-by-byte (5.5 cyc/val), unrolled x4 (4.0), word-level shift+mask (4.4), 256-byte LUT (4.5), SIMD-style bitmask (3.76), deinterleaved lo/hi split (1.75).
- **Full INT4 matmul pipeline** with each strategy, showing end-to-end impact: best interleaved (unrolled4) is 824 us vs INT8 at 49 us for 256x256. Still 16.8x slower.
- **Deinterleaved unpack** at 1.75 cyc/val is 2.9x faster than naive but produces separate lo/hi nibble arrays. Requires a 2-pass PIE matmul to use directly.

## ESP32-P4 Results

All measurements on ESP32-P4 rev 1.3, 360 MHz, 32 MB PSRAM at 200 MHz hex mode, ESP-IDF v5.5.3, -O2 optimization.

### Memory System

| Metric | Value | Notes |
|--------|-------|-------|
| SRAM total | 629 KB | Fragmented: 376KB max contiguous block |
| PSRAM total | 32 MB | 200 MHz x16 hex mode |
| SRAM seq read | 359 MB/s | Consistent across all sizes |
| PSRAM seq read (cached) | 359 MB/s | Buffer <= 64 KB (fits in cache) |
| PSRAM seq read (uncached) | 104 MB/s | Buffer > 64 KB (true PSRAM speed) |
| PSRAM seq write (uncached) | 85 MB/s | |
| SRAM->SRAM memcpy | 368 MB/s | 64 KB buffer |
| PSRAM->SRAM memcpy (16KB) | 561 MB/s | Optimal size, fully cached |
| PSRAM->SRAM memcpy (32KB) | 560 MB/s | Still in cache sweet spot |
| PSRAM->SRAM memcpy (64KB) | 372 MB/s | Cache spill, 34% drop |
| PSRAM->SRAM row copy (384B) | 102 MB/s | Per-row, matches uncached BW |
| SRAM random latency | 14 ns / 25 ns | Timer / pointer-chase |
| PSRAM random latency (256KB) | 288 ns | 11.5x SRAM pointer-chase |
| PSRAM random latency (512KB) | 392 ns | 15.7x SRAM |

### Cache Hierarchy

| Level | Size | Latency | Evidence |
|-------|------|---------|----------|
| L1 | ~32 KB | 25 ns (pointer chase) | Flat from 1-64 KB |
| L2 | ~128 KB | 37.5 ns | Step at 128 KB |
| PSRAM uncached | - | 288-392 ns | Step at 256 KB+ |

memcpy peaks at 560 MB/s for 16-32 KB transfers (L1 resident), drops to 372 MB/s at 64 KB (L1 spill), and to ~100 MB/s for row-level access patterns (uncached).

### PIE XACC (INT8 SIMD)

| Metric | Value | Notes |
|--------|-------|-------|
| Instruction | esp.vmulas.s8.xacc | 16 x int8 MAC per instruction |
| Cycles per instruction | 4.3 | NOT single-cycle |
| MACs per cycle | 3.7-3.9 | At optimal dimensions (128-256 cols) |
| Peak GMAC/s | 1.39 | At 128x256 from SRAM |
| GMAC/s at 1024x256 | 0.88 | More loop/call overhead per MAC |
| Scalar speedup | 17-40x | Depends on vector length |
| Call overhead | 1.37-1.55x | N calls vs 1 batched call |

Effective throughput decreases with larger matrices due to register reload overhead in the row loop. The fused load+MAC instructions (`esp.vmulas.s8.xacc.ld.ip`) help pipeline data movement but don't achieve single-cycle throughput.

### PIE Memory Source (512x384 matmul)

| Source | Time (us) | vs SRAM | Effective MB/s |
|--------|----------|---------|----------------|
| SRAM (single call) | 214 | 1.0x | 919 |
| PSRAM direct (per-row) | 1,615 | 7.5x | 122 |
| PSRAM row-copy (memcpy+PIE) | 1,939 | 9.0x | 101 |
| PSRAM batch-copy (16KB) | 1,826 | 8.5x | 108 |

For short rows (cols <= 512), PSRAM direct is only 1.1-1.9x slower than SRAM. For long rows (cols >= 1024), the gap jumps to 7-9x. The transition correlates with the cache line fill pattern.

### Popcount (Software)

| Strategy | Cycles/word | Notes |
|----------|-------------|-------|
| Naive (bit loop) | 137 | Baseline, worst case |
| Wegner (x &= x-1) | 82 | Data-dependent (avg 16 iterations) |
| __builtin_popcount | 34.5 | GCC software expansion (no Zbb) |
| LUT (256-byte table) | 21 | 4 table lookups per word |
| Fast (bit-parallel) | 19 | Hacker's Delight, best overall |

No MCU-class chip has hardware popcount. The ESP32-P4 does not implement the RISC-V Zbb extension. Best achievable is 19 cycles/word via the bit-parallel algorithm.

### XNOR Binary Matmul (from SRAM)

| Dimensions | XNOR (fast_popcount) | INT8 PIE | PIE advantage |
|-----------|---------------------|----------|---------------|
| 128x1024 | 266 us | 121 us | 2.2x |
| 256x256 | 137 us | 49 us | 2.8x |
| 512x384 | 404 us | 197 us | 2.1x |
| Peak GMAC/s | 0.49 (binary) | 1.39 (INT8) | 2.8x |

INT8 PIE is 2.1-2.8x faster than XNOR from SRAM. XNOR processes 32 binary MACs per popcount (19 cycles) = 1.68 binary MACs/cycle. PIE does 3.7-3.9 INT8 MACs/cycle. For SRAM-resident models, INT8 is strictly better.

### INT4 Unpack Cost

| Strategy | Cycles/value | Notes |
|----------|-------------|-------|
| Naive (byte-by-byte) | 5.5 | Baseline |
| Unrolled x4 | 4.0 | Loop unrolling helps ~27% |
| Word-level (shift+mask) | 4.4 | Word load amortizes slightly |
| LUT (256-byte) | 4.5 | Table lookup not faster here |
| SIMD-style (bitmask) | 3.76 | Best interleaved output |
| Deinterleaved (lo/hi split) | 1.75 | 2.9x faster, non-standard output |

### INT4 vs INT8 End-to-End (from PSRAM)

1024x512 matmul streaming from PSRAM:

| Phase | INT8 | INT4 (naive unpack) |
|-------|------|---------------------|
| PSRAM copy | 5,421 us (512 KB) | 2,625 us (256 KB) |
| Unpack | - | 8,807 us (74.5%) |
| PIE compute | included above | 398 us (3.4%) |
| **Total** | **5,421 us** | **11,830 us** |
| **Speed** | **1.0x** | **0.46x (slower)** |

INT4 halves the data transferred but the scalar unpack cost is 3.4x the copy savings. INT4 is currently 2.2x slower than INT8 on the ESP32-P4.

With the best interleaved unpack (unrolled4), the INT4 matmul pipeline at 256x256 takes 824 us vs INT8 at 49 us -- still 16.8x slower from SRAM.

The deinterleaved unpack at 1.75 cyc/val could theoretically bring INT4 closer to parity, but requires a 2-pass PIE matmul architecture that doubles the compute passes.

### Double-Buffer Copy/Compute Ratio

| Batch size | Copy (us) | Compute (us) | Ratio | Total (us) |
|-----------|----------|-------------|-------|-----------|
| 1 row | 3,419 | 1,086 | 3.1 | 4,505 |
| 4 rows | 2,487 | 414 | 6.0 | 2,901 |
| 16 rows | 2,321 | 264 | 8.8 | 2,585 |
| 32 rows | 2,286 | 231 | 9.9 | 2,517 |
| 64 rows | 2,275 | 226 | 10.1 | 2,501 |
| 128 rows | 2,371 | 215 | 11.0 | 2,586 |

For 1024x256 INT8 from PSRAM: compute is ~10% of total time. The memory bus is the bottleneck at every batch size. DMA double-buffering would save at most the 215 us compute overlap -- a ~8% improvement. Not transformative.

### Corrected Projections for Large Model Inference

Based on measured PSRAM bandwidth of 104 MB/s:

| Model (INT8) | Weight size | PSRAM time | Est. tok/s |
|-------------|------------|-----------|-----------|
| 1M params | 1 MB | 9.6 ms | ~100 |
| 5M params | 5 MB | 48 ms | ~20 |
| 10M params | 10 MB | 96 ms | ~10 |
| 20M params | 20 MB | 192 ms | ~5 |

These assume pure weight streaming with no activation or layer overhead. Real performance will be somewhat lower.

## Key Takeaways for ESP32-P4 Inference Engine Design

1. **Use INT8 + PIE XACC.** It is the fastest option on this chip at every model size. INT4 is slower due to unpack cost. XNOR is slower due to lack of hardware popcount.

2. **Keep activations and KV cache in SRAM.** The 32 KB L1 cache is the sweet spot for tile buffers. Activations for hidden_dim=512 fit in ~2 KB, well within L1.

3. **For PSRAM weight streaming, use PIE direct access for hidden_dim <= 512.** Skip memcpy entirely -- PIE reading through the cache controller is faster than explicit row-copy for short rows. For hidden_dim >= 1024, batch-copy to a 16-32 KB SRAM buffer.

4. **The PSRAM bandwidth wall limits large models to ~5-10 tokens/second** for 5-10M parameter INT8 models. This is a hardware constraint that no software optimization can overcome on this chip.

5. **The RA8M2 with SDRAM is the path for larger models.** Its ~5x higher external memory bandwidth (estimated ~500+ MB/s SDRAM vs 104 MB/s PSRAM) directly translates to 5x higher token throughput for memory-bound workloads.

## Porting to Other Chips

To run equivalent benchmarks on a new MCU:

1. Replace `pie_kernels.S` with the platform's SIMD kernel (Helium intrinsics for Cortex-M85, SMLAD for Cortex-M7)
2. Replace `MALLOC_CAP_SPIRAM` / `MALLOC_CAP_INTERNAL` with the platform's memory allocator
3. Replace `read_mcycle()` with the platform's cycle counter CSR
4. Replace `esp_timer_get_time()` with the platform's microsecond timer
5. Update `sdkconfig.defaults` / build system for the target
6. All benchmark logic in the C files remains unchanged
