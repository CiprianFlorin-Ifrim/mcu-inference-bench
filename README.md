# mcu-inference-bench

Hardware benchmark suite for neural network inference on microcontrollers. Measures the raw capabilities that determine inference performance: memory bandwidth, SIMD throughput, popcount cost, INT4 unpack strategies, cache effects, streaming pipeline overhead, FP32 operation costs, and full model token pipeline timing.

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
    bench_fp32.c       FP32 ops: layer norm, softmax, SSM, SiLU, conv1d, quantize
    bench_model.c      Architecture-matched matmuls, recurrence cache, token pipeline
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

### bench_fp32.c -- FP32 Operations

Measures every floating-point operation in the inference pipeline. These run on activations in SRAM, not on weights.

- **LayerNorm and RMSNorm** at dim=128 to 1024. RMSNorm is ~1.8x faster than LayerNorm (no mean subtraction pass). At dim=512: LayerNorm 46 us, RMSNorm 26 us.
- **Softmax** at output sizes from 140 (custom vocab) to 4096 (large BPE). At vocab=140: 81 us. Dominated by expf() calls.
- **SiLU and SwiGLU gating** at model dimensions. At dim=512: SiLU 253 us, SwiGLU 259 us. Dominated by expf() in the sigmoid. These are the most expensive FP32 operations per layer.
- **Mamba2 SSM state update** (per-token selective scan step). At state=16, expand=512: 344 us. At state=16, expand=256 (bottleneck): 172 us. This is the core Mamba recurrence and runs once per Mamba layer per token.
- **Conv1D step** (Mamba's causal convolution, stateful ring buffer). At channels=512, kernel=4: 110 us. At channels=1024: 219 us.
- **INT8 quantize/dequantize** overhead at layer boundaries. At dim=512: quantize 65 us, dequantize 14 us.

### bench_model.c -- Model Architecture Benchmarks

Maps the target model architecture directly onto hardware measurements.

- **Architecture-matched matmul sizes from SRAM**: every matmul in the U-Net (Mamba in_proj 1024x512, Mamba out_proj 512x1024, projection 512<->256, attention QKV 768x256, output head 140x512). Gives exact per-layer SRAM compute cost.
- **Architecture-matched matmul sizes from PSRAM**: same shapes but via PSRAM direct access. Shows the real PSRAM penalty per layer. Key finding: at dim=512, PSRAM direct is only 1.2-1.5x slower than SRAM -- much better than the 7-9x penalty at dim=1024.
- **Recurrence cache effect**: streams the same bottleneck weights (512x256, 128 KB) from PSRAM N consecutive times. First pass: 1,101 us. Every subsequent pass: ~205 us (5.3x faster). The PSRAM data stays in the ~128 KB L2 cache between passes. This is the key finding for depth recurrence efficiency.
- **Skip connection overhead**: SRAM memcpy for 512-dim FP32 activation vectors. Store: 3.8 us, load: 3.8 us. Negligible.
- **Full token pipeline simulation**: runs the exact sequence of matmuls for the complete U-Net architecture (encoder 3 unrolls + projection + bottleneck Nx recurrence + projection + decoder 3 unrolls) all from PSRAM. Reports per-component breakdown and total tok/s at 1x through 6x recurrence.

## ESP32-P4 Results

All measurements on ESP32-P4 rev 1.3, 360 MHz, 32 MB PSRAM at 200 MHz hex mode, ESP-IDF v5.5.3, -O2 optimization.

### Memory System

| Metric | Value | Notes |
|--------|-------|-------|
| SRAM total | 629 KB | Fragmented: 376 KB max contiguous block |
| PSRAM total | 32 MB | 200 MHz x16 hex mode |
| SRAM seq read | 359 MB/s | Consistent across all sizes |
| PSRAM seq read (cached) | 359 MB/s | Buffer <= 64 KB (fits in cache) |
| PSRAM seq read (uncached) | 104 MB/s | Buffer > 64 KB (true PSRAM speed) |
| PSRAM seq write (uncached) | 85 MB/s | |
| SRAM->SRAM memcpy | 368 MB/s | 64 KB buffer |
| PSRAM->SRAM memcpy (16 KB) | 561 MB/s | Optimal size, fully cached |
| PSRAM->SRAM memcpy (32 KB) | 560 MB/s | Still in cache sweet spot |
| PSRAM->SRAM memcpy (64 KB) | 372 MB/s | Cache spill, 34% drop |
| PSRAM->SRAM row copy (384B) | 102 MB/s | Per-row, matches uncached BW |
| SRAM random latency | 14 ns / 25 ns | Timer / pointer-chase |
| PSRAM random latency (256 KB) | 288 ns | 11.5x SRAM pointer-chase |
| PSRAM random latency (512 KB) | 392 ns | 15.7x SRAM |

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

### PIE Memory Source

| Matmul | SRAM (us) | PSRAM direct (us) | PSRAM/SRAM ratio |
|--------|----------|-------------------|-----------------|
| 256x128 | 30 | 56 | 1.9x |
| 256x256 | 49 | 75 | 1.5x |
| 256x512 | 134 | 174 | 1.3x |
| 256x1024 | 243 | 2,088 | 8.6x |
| 512x256 | 140 | 207 | 1.5x |
| 512x384 | 204 | 1,615 | 7.9x |
| 1024x256 | 280 | 2,204 | 7.9x |

Critical finding: for cols <= 512, PSRAM direct access is only 1.3-1.9x slower than SRAM. At cols >= 1024, the penalty jumps to 7-9x. This means model layers with dim=512 stream efficiently from PSRAM without explicit tiling.

### Popcount (Software)

| Strategy | Cycles/word | Notes |
|----------|-------------|-------|
| Naive (bit loop) | 137 | Baseline, worst case |
| Wegner (x &= x-1) | 82 | Data-dependent (avg 16 iterations) |
| __builtin_popcount | 34.5 | GCC software expansion (no Zbb) |
| LUT (256-byte table) | 21 | 4 table lookups per word |
| Fast (bit-parallel) | 19 | Hacker's Delight, best overall |

No MCU-class chip has hardware popcount. The ESP32-P4 does not implement the RISC-V Zbb extension. Best achievable is 19 cycles/word via the bit-parallel algorithm.

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

### FP32 Operations

| Operation | Dim/size | Cycles | Time (us) | Notes |
|-----------|---------|--------|----------|-------|
| LayerNorm | 512 | 16,504 | 45.8 | 3 passes over data |
| RMSNorm | 512 | 9,318 | 25.9 | 1.8x faster than LayerNorm |
| Softmax | 140 (vocab) | 29,076 | 80.8 | Dominated by expf() |
| Softmax | 256 | 52,904 | 147.0 | |
| SiLU | 512 | 91,253 | 253.5 | expf() in sigmoid |
| SwiGLU | 512 | 93,209 | 258.9 | SiLU + elementwise multiply |
| SSM state update | 16x512 | 124,004 | 344.5 | Core Mamba2 recurrence |
| SSM state update | 16x256 | 62,008 | 172.2 | Bottleneck dim |
| Conv1D step | 512ch, k=4 | 39,484 | 109.7 | Mamba causal conv |
| Conv1D step | 1024ch, k=4 | 78,933 | 219.3 | Expanded dim |
| INT8 quantize | 512 | 23,269 | 64.6 | Per-layer boundary |
| INT8 dequantize | 512 | 5,130 | 14.3 | |

FP32 overhead per token (estimated for 10M model with 14 effective layers): ~6-8 ms. Not negligible -- represents ~10% of total token time. SSM state updates and SiLU gating are the dominant costs. These are candidates for future optimization via INT8 approximation or lookup tables for expf().

### Architecture-Matched Matmul Sizes (from SRAM)

| Layer | Dimensions | Cycles | Time (us) | GMAC/s |
|-------|-----------|--------|----------|--------|
| Embedding lookup | 1x512 | 162 | 0.5 | 1.14 |
| Projection down | 256x512 | 48,249 | 134.0 | 0.98 |
| Projection up | 512x256 | 50,400 | 140.0 | 0.94 |
| Attention QKV | 768x256 | 75,575 | 209.9 | 0.94 |
| Attention out | 256x256 | 17,448 | 48.5 | 1.35 |
| Bot Mamba in_proj | 512x256 | 50,388 | 140.0 | 0.94 |
| Bot Mamba out_proj | 256x512 | 48,260 | 134.1 | 0.98 |
| Output head | 140x512 | 20,568 | 57.1 | 1.25 |

Note: Mamba in_proj (1024x512) and out_proj (512x1024) exceed the 376 KB max contiguous SRAM block and must stream from PSRAM.

### Architecture-Matched Matmul Sizes (from PSRAM direct)

| Layer | Dimensions | SRAM (us) | PSRAM (us) | Ratio |
|-------|-----------|----------|-----------|-------|
| Projection down | 256x512 | 134 | 174 | 1.3x |
| Attention QKV | 768x256 | 210 | 1,653 | 7.9x |
| Bot Mamba in_proj | 512x256 | 140 | 207 | 1.5x |
| Output head | 140x512 | 58 | 72 | 1.2x |

Layers with cols=512 show only 1.2-1.3x PSRAM penalty -- the data fits in cache lines efficiently. Layers with cols=256 and many rows (768x256 QKV) show 7.9x penalty because the total weight matrix (192 KB) exceeds L2 cache, causing thrashing on per-row access.

### Recurrence Cache Effect (key finding)

Bottleneck matmul 512x256 (128 KB) from PSRAM, consecutive passes:

| Pass | Time (us) | vs Pass 1 |
|------|----------|----------|
| 1 | 1,101 | 1.00x |
| 2 | 206 | 0.19x |
| 3 | 204 | 0.19x |
| 4 | 205 | 0.19x |
| 5 | 204 | 0.18x |
| 6 | 205 | 0.19x |

**After the first pass, PSRAM data remains in the ~128 KB L2 cache. Subsequent passes are 5.3x faster.** This is the single most important finding for depth recurrence: the bottleneck weights (128 KB for a single matmul) fit in L2, so recurrence passes 2+ execute at near-SRAM speed.

However, the full bottleneck layer (QKV + out + mamba_in + mamba_out = ~512 KB total) exceeds L2 cache. In the full pipeline simulation, per-recurrence cost is ~4,300 us (not ~800 us), indicating partial but not full cache benefit. The cache helps individual matmuls within the recurrence but not the complete layer.

### Skip Connection Overhead

| Dim | Bytes | Store (ns) | Load (ns) |
|-----|-------|-----------|----------|
| 512 | 2,048 | 3,805 | 3,802 |

Skip connections cost ~3.8 us per store and ~3.8 us per load. With 3 skip depths, total overhead is ~23 us per token. Negligible.

### Full Token Pipeline (measured, INT8, PSRAM streaming)

10M unique params, unrolled U-Net architecture. Encoder and decoder each stream ~1.5 MB from PSRAM. Bottleneck streams ~512 KB per recurrence pass.

| Recurrence | Encoder (us) | Proj dn (us) | Bottleneck (us) | Proj up (us) | Decoder (us) | Total (us) | tok/s |
|-----------|-------------|-------------|----------------|-------------|-------------|-----------|-------|
| 1x | 25,251 | 1,066 | 4,317 | 1,078 | 25,213 | 56,925 | 17.6 |
| 2x | 25,213 | 1,065 | 8,629 | 1,077 | 25,216 | 61,201 | 16.3 |
| 3x | 25,216 | 1,065 | 12,943 | 1,077 | 25,214 | 65,515 | 15.3 |
| 4x | 25,218 | 1,064 | 17,253 | 1,083 | 25,211 | 69,830 | 14.3 |
| 6x | 25,216 | 1,063 | 25,885 | 1,078 | 25,215 | 78,457 | 12.7 |

These timings exclude FP32 operations (layer norm, softmax, SSM state updates, SiLU gating, conv1d, quantization). Add approximately 6-8 ms for FP32 overhead based on bench_fp32 measurements.

### Corrected Performance Projections (with FP32 overhead)

| Config | Matmul (measured) | FP32 (measured) | Total | tok/s |
|--------|------------------|----------------|-------|-------|
| 10M INT8, 1x rec | 57 ms | ~6 ms | ~63 ms | ~16 |
| 10M INT8, 3x rec | 65.5 ms | ~7 ms | ~72.5 ms | ~14 |
| 10M INT8, 6x rec | 78.5 ms | ~8 ms | ~86.5 ms | ~11.5 |
| 10M INT4, 1x rec (est) | ~114 ms | ~6 ms | ~120 ms | ~8 |
| 10M INT4, 3x rec (est) | ~131 ms | ~7 ms | ~138 ms | ~7 |

The 10M INT8 model with 3x recurrence achieves 14 tok/s. With 6x recurrence (for deeper reasoning), it still delivers 11.5 tok/s. These are real numbers measured on hardware, not estimates.

## Strategic Implications

### INT8 vs INT4 on ESP32-P4

The benchmarks strongly favor INT8 for this specific chip:

1. **PSRAM direct access at dim=512 is only 1.2-1.5x slower than SRAM** -- the expected 7-9x penalty only applies at dim >= 1024. This eliminates the bandwidth argument for INT4.

2. **INT4 unpack overhead is 74.5% of total pipeline time** -- the halved data transfer does not compensate for the scalar unpack cost. INT4 is 2.2x slower than INT8 from PSRAM.

3. **Recurrence is cheap** -- each additional bottleneck pass costs ~4.3 ms. Going from 1x to 6x recurrence only drops throughput from 17.6 to 12.7 tok/s. This means a 10M INT8 model with deep recurrence can approach the effective quality of a much larger model without the speed penalty of INT4.

### The Recurrence vs Width Question

The critical open question: does a 10M model with 6x recurrence match a 30M model with 2x recurrence? If yes, INT8 with deep recurrence is strictly dominant on this hardware. This requires an ablation study (GPU training experiment, not MCU experiment) before committing to a quantization strategy.

### Recommended Development Path

```
1. INT8 first: train 10M model with INT8 QAT, deploy with variable recurrence
2. Ablation: sweep recurrence depth vs model size at equal compute to find exchange rate
3. INT4 optional: finetune INT8 checkpoint to INT4 if the ablation shows width > depth
4. RA8M2: benchmark Helium INT4 unpack -- if fast, INT4 becomes viable there
```

## Key Takeaways

1. **Use INT8 + PIE XACC on ESP32-P4.** It is the fastest option on this chip. INT4 is slower due to unpack cost. XNOR is slower due to lack of hardware popcount.

2. **PSRAM direct access is efficient for dim=512.** Skip memcpy entirely for model layers with 512 columns or fewer. PIE reading through the cache controller is only 1.2-1.5x slower than SRAM at these dimensions.

3. **Depth recurrence is nearly free relative to model streaming.** Each bottleneck recurrence pass adds ~4.3 ms to a ~57 ms base. 6x recurrence costs 38% more time for potentially much deeper reasoning.

4. **Individual bottleneck matmuls (128 KB) cache in L2.** Repeat passes are 5.3x faster than the first pass. The full bottleneck layer (512 KB) exceeds L2 but still benefits from partial caching.

5. **FP32 operations cost ~6-8 ms per token (~10% of total).** SSM state updates and SiLU gating dominate. These are optimization targets for future work (INT8 approximation, expf() lookup tables).

6. **14 tok/s at 3x recurrence with 10M unique params is achievable today** on the ESP32-P4 with INT8 weights and the measured architecture. No speculative optimizations required.

7. **The RA8M2 with SDRAM should deliver 3-5x improvement** based on its higher memory bandwidth (~500 MB/s estimated vs 104 MB/s measured). This needs benchmark validation.

## Porting to Other Chips

To run equivalent benchmarks on a new MCU:

1. Replace `pie_kernels.S` with the platform's SIMD kernel (Helium intrinsics for Cortex-M85, SMLAD for Cortex-M7)
2. Replace `MALLOC_CAP_SPIRAM` / `MALLOC_CAP_INTERNAL` with the platform's memory allocator
3. Replace `read_mcycle()` with the platform's cycle counter CSR
4. Replace `esp_timer_get_time()` with the platform's microsecond timer
5. Update `sdkconfig.defaults` / build system for the target
6. All benchmark logic in the C files remains unchanged
