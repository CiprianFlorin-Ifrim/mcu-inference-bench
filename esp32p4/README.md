# ESP32-P4 Hardware Benchmark Suite

Comprehensive microbenchmarks for neural network inference on the ESP32-P4.
Measures the raw hardware capabilities that determine inference performance.

## What It Measures

### 1. Memory Bandwidth (`bench_memory.c`)
- Sequential read/write: SRAM vs PSRAM at various buffer sizes
- memcpy throughput: all four direction combinations (SRAM/PSRAM -> SRAM/PSRAM)
- Random read latency: pointer chasing at various working set sizes
- Row copy: PSRAM -> SRAM at row sizes matching real weight matrices (128-4096 bytes)
- Batch copy: effect of batch size on total throughput (1 vs N rows per memcpy)

### 2. PIE XACC Throughput (`bench_pie.c`)
- Single dot product: PIE vs scalar C at 16 to 4096 elements
- Multi-row matmul: PIE vs scalar at dimensions matching real models
- Call overhead: N calls of 1 row vs 1 call of N rows (quantifies function call cost)
- PIE source comparison: SRAM direct vs PSRAM row-copy vs PSRAM batch-copy vs PSRAM direct
- Reports actual MACs/cycle and GMAC/s

### 3. Popcount Strategies (`bench_popcount.c`)
- Five implementations compared: naive bit loop, Wegner (x&=x-1), __builtin_popcount, 256-byte LUT, bit-parallel (Hacker's Delight)
- Cycles per 32-bit word for each
- Full XNOR matmul throughput at realistic layer sizes
- Direct XNOR vs INT8 PIE comparison at same dimensions

### 4. End-to-End Matmul (`bench_matmul.c`)
- INT8 PIE from SRAM (compute-bound baseline)
- INT4 unpack + PIE vs pure INT8 PIE (measures unpack overhead)
- INT4 batched unpack vs row-by-row
- PSRAM streaming: INT8 row-copy vs INT4 copy+unpack+PIE (the key comparison for large models)

### 5. Cache Effects (`bench_cache.c`)
- Flash (.rodata) sequential read throughput
- SRAM sequential read at matching sizes (quantifies flash cache penalty)
- Stride access patterns: measures cache line effects
- Working set sweep: detects L1 cache boundaries by measuring latency vs buffer size

### 6. DMA / Double-Buffer Simulation (`bench_dma.c`)
- Double-buffer pattern: copy and compute phases timed separately
- Overlap potential calculation (what speedup real DMA would give)
- Copy/compute ratio at various batch sizes
- INT4 pipeline breakdown: copy + unpack + compute as separate phases
- Direct INT8 vs INT4 PSRAM streaming comparison

## Building

```bash
# Set target
idf.py set-target esp32p4

# Build
idf.py build

# Flash and monitor
idf.py flash monitor
```

Requires ESP-IDF v5.5+ with ESP32-P4 support.

## Configuration

`sdkconfig.defaults` sets:
- CPU at 360 MHz
- PSRAM at 200 MHz hex mode
- Performance optimization
- Watchdog disabled (long benchmarks)
- 16 KB main task stack

## Output

The benchmark prints a structured report to serial console. Key numbers to extract:

- **PSRAM sequential read MB/s** -- actual external memory bandwidth
- **PSRAM -> SRAM memcpy MB/s** -- the row-copy cost for weight tiling
- **PIE XACC MACs/cycle** -- should be close to 16.0 for aligned SRAM data
- **fast_popcount cycles/word** -- the cost of software popcount
- **INT8 PIE vs XNOR speedup** -- quantifies the PIE advantage over binary
- **INT4 vs INT8 from PSRAM** -- the actual bandwidth savings of INT4
- **Copy/compute ratio** -- determines if double-buffering is worthwhile

## Project Structure

```
main/
  main.c           -- Entry point, system info, runs all benchmarks
  bench.h          -- Common utilities, timing macros, declarations
  bench_memory.c   -- SRAM/PSRAM bandwidth and latency
  bench_pie.c      -- PIE XACC throughput measurements
  bench_popcount.c -- Popcount strategy comparison
  bench_matmul.c   -- INT8/INT4/XNOR matmul comparison
  bench_cache.c    -- Cache and flash access patterns
  bench_dma.c      -- DMA double-buffer simulation
  pie_kernels.S    -- PIE XACC assembly (dot product + matmul)
```

## Porting

To run equivalent benchmarks on a different chip (e.g. Renesas RA8M2):
1. Replace `pie_kernels.S` with Helium intrinsic kernels
2. Replace `MALLOC_CAP_SPIRAM` with the appropriate external RAM allocator
3. Replace `esp_timer_get_time()` with the platform's cycle counter
4. Keep all benchmark logic in the C files unchanged
