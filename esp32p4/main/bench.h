#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_cache.h"

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

static inline int64_t bench_start(void) { return esp_timer_get_time(); }
static inline float bench_stop_us(int64_t t0) { return (float)(esp_timer_get_time() - t0); }

// Cycle-accurate timing via RISC-V mcycle CSR (1 cycle = 1/360MHz = 2.78ns)
static inline uint32_t read_mcycle(void)
{
    uint32_t val;
    __asm__ volatile("csrr %0, mcycle" : "=r"(val));
    return val;
}

#define CYCLES_TO_NS(cyc)  ((float)(cyc) * 1000.0f / (float)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ)

// Run a benchmark N times, return average microseconds
#define BENCH_AVG(name, n_iters, code_block) do { \
    /* warmup */ \
    for (int _w = 0; _w < 10; _w++) { code_block; } \
    int64_t _t0 = bench_start(); \
    for (int _i = 0; _i < (n_iters); _i++) { code_block; } \
    float _elapsed = bench_stop_us(_t0); \
    printf("  %-40s %10.1f us  (%d iters, %.1f us/iter)\n", \
           (name), _elapsed, (n_iters), _elapsed / (float)(n_iters)); \
} while(0)

// ---------------------------------------------------------------------------
// Separator
// ---------------------------------------------------------------------------

static inline void bench_separator(const char *title)
{
    printf("\n");
    printf("==============================================================\n");
    printf("  %s\n", title);
    printf("==============================================================\n\n");
}

static inline void bench_subsection(const char *title)
{
    printf("\n--- %s ---\n", title);
}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

static inline void *alloc_sram(size_t size, size_t align)
{
    void *p = heap_caps_aligned_alloc(align, size, MALLOC_CAP_INTERNAL);
    if (p) memset(p, 0, size);
    return p;
}

static inline void *alloc_psram(size_t size, size_t align)
{
    void *p = heap_caps_aligned_alloc(align, size, MALLOC_CAP_SPIRAM);
    if (p) memset(p, 0, size);
    return p;
}

// ---------------------------------------------------------------------------
// PRNG for filling buffers with deterministic data
// ---------------------------------------------------------------------------

static inline uint32_t bench_xorshift(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline void fill_random_s8(int8_t *buf, int n, uint32_t seed)
{
    uint32_t state = seed;
    for (int i = 0; i < n; i++) {
        buf[i] = (int8_t)(bench_xorshift(&state) & 0x7F) - 64;
    }
}

static inline void fill_random_u32(uint32_t *buf, int n, uint32_t seed)
{
    uint32_t state = seed;
    for (int i = 0; i < n; i++) {
        buf[i] = bench_xorshift(&state);
    }
}

static inline void fill_random_f32(float *buf, int n, uint32_t seed)
{
    uint32_t state = seed;
    for (int i = 0; i < n; i++) {
        buf[i] = ((float)(bench_xorshift(&state) & 0xFFFF) / 65536.0f) * 2.0f - 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Benchmark function declarations
// ---------------------------------------------------------------------------

// bench_memory.c -- SRAM/PSRAM bandwidth and latency
void bench_memory(void);

// bench_pie.c -- PIE XACC throughput, vector ops
void bench_pie(void);

// bench_popcount.c -- software popcount strategies
void bench_popcount(void);

// bench_matmul.c -- end-to-end matmul at various sizes and precisions
void bench_matmul(void);

// bench_cache.c -- cache effects, flash vs SRAM vs PSRAM code/data
void bench_cache(void);

// bench_dma.c -- DMA PSRAM-to-SRAM transfer rates
void bench_dma(void);

// bench_int4.c -- INT4 unpack strategies (the critical bottleneck)
void bench_int4(void);

// ---------------------------------------------------------------------------
// PIE assembly kernels (pie_kernels.S)
// ---------------------------------------------------------------------------

// Single dot product: y[0] += sum(W[i] * x[i]) for i in [0, cols)
// W and x must be 16-byte aligned, cols must be multiple of 16, in SRAM
extern void pie_dot_s8(const int8_t *W, const int8_t *x, int32_t *y, int cols);

// Multi-row: y[r] = dot(W + r*cols, x) for r in [0, rows)
extern void pie_matmul_s8_xacc(const int8_t *W, const int8_t *x, int32_t *y,
                                 int rows, int cols);
