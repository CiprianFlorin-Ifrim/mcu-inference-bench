#include "bench.h"

// ---------------------------------------------------------------------------
// Popcount strategies
// ---------------------------------------------------------------------------

static inline int naive_popcount(uint32_t x)
{
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}

static inline int wegner_popcount(uint32_t x)
{
    int c = 0;
    while (x) { x &= x - 1; c++; }
    return c;
}

static inline int fast_popcount(uint32_t x)
{
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

static uint8_t popcount_lut[256];

static void init_popcount_lut(void)
{
    for (int i = 0; i < 256; i++) {
        int c = 0, v = i;
        while (v) { c += v & 1; v >>= 1; }
        popcount_lut[i] = (uint8_t)c;
    }
}

static inline int lut_popcount(uint32_t x)
{
    return popcount_lut[x & 0xFF]
         + popcount_lut[(x >> 8) & 0xFF]
         + popcount_lut[(x >> 16) & 0xFF]
         + popcount_lut[(x >> 24) & 0xFF];
}

// ---------------------------------------------------------------------------
// XNOR matmul using each popcount strategy
// ---------------------------------------------------------------------------

#define DEFINE_XNOR_MATMUL(name, pcnt_fn) \
static int32_t xnor_matmul_##name(const uint32_t *W, const uint32_t *x, \
                                    int words_per_row, int rows) \
{ \
    int32_t total = 0; \
    for (int r = 0; r < rows; r++) { \
        const uint32_t *w_row = W + r * words_per_row; \
        int32_t agree = 0; \
        for (int w = 0; w < words_per_row; w++) \
            agree += pcnt_fn(~(w_row[w] ^ x[w])); \
        total += agree; \
    } \
    return total; \
}

DEFINE_XNOR_MATMUL(naive, naive_popcount)
DEFINE_XNOR_MATMUL(wegner, wegner_popcount)
DEFINE_XNOR_MATMUL(builtin, __builtin_popcount)
DEFINE_XNOR_MATMUL(fast, fast_popcount)
DEFINE_XNOR_MATMUL(lut, lut_popcount)

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

void bench_popcount(void)
{
    bench_separator("POPCOUNT / XNOR THROUGHPUT");

    init_popcount_lut();

    // --- Raw popcount: cycle-accurate ---

    bench_subsection("Raw Popcount: cycles per 32-bit word (mcycle CSR)");

    int n_words = 16384;  // 64 KB -- fits comfortably in SRAM
    uint32_t *data = (uint32_t *)alloc_sram(n_words * 4, 16);
    if (!data) {
        printf("  ALLOC FAILED\n");
        return;
    }
    printf("  Buffer: %d words (%d KB) in SRAM\n", n_words, n_words * 4 / 1024);

    fill_random_u32(data, n_words, 42);
    volatile int32_t sink_val;
    int reps = 20;

    printf("  Reps: %d (total %d M popcount ops per strategy)\n\n", reps, n_words * reps / 1000000);

    // Macro to benchmark each strategy
    #define BENCH_PCNT(label, fn) do { \
        int32_t sum = 0; \
        /* warmup */ \
        for (int i = 0; i < n_words; i++) sum += fn(data[i]); \
        sum = 0; \
        uint32_t c0 = read_mcycle(); \
        for (int rep = 0; rep < reps; rep++) { \
            sum = 0; \
            for (int i = 0; i < n_words; i++) sum += fn(data[i]); \
        } \
        uint32_t c1 = read_mcycle(); \
        sink_val = sum; \
        float cyc = (float)(c1 - c0) / (float)(n_words * reps); \
        float ns = CYCLES_TO_NS(cyc); \
        printf("  %-20s  %6.1f cycles/word  %6.1f ns/word\n", label, cyc, ns); \
    } while(0)

    BENCH_PCNT("Naive (bit loop)", naive_popcount);
    BENCH_PCNT("Wegner (x&=x-1)", wegner_popcount);
    BENCH_PCNT("__builtin_popcount", __builtin_popcount);
    BENCH_PCNT("LUT (256-byte)", lut_popcount);
    BENCH_PCNT("Fast (bit-parallel)", fast_popcount);

    #undef BENCH_PCNT
    (void)sink_val;
    heap_caps_free(data);

    // --- Full XNOR matmul at realistic sizes ---

    bench_subsection("XNOR Matmul: full binary layer throughput");

    int xnor_configs[][2] = {
        {128, 512}, {128, 1024}, {256, 256}, {256, 1024}, {512, 1024},
    };
    int n_xnor = sizeof(xnor_configs) / sizeof(xnor_configs[0]);
    int xnor_iters = 100;

    printf("  %-14s  %8s  %8s  %8s  %8s  %8s  %10s\n",
           "RxC (bits)", "Naive", "Wegner", "Builtin", "Fast", "LUT", "Best GMAC/s");
    printf("  %-14s  %8s  %8s  %8s  %8s  %8s  %10s\n",
           "", "us", "us", "us", "us", "us", "");
    printf("  ---------------------------------------------------------------------------------\n");

    for (int c = 0; c < n_xnor; c++) {
        int rows = xnor_configs[c][0];
        int cols = xnor_configs[c][1];
        int wpr = (cols + 31) / 32;

        uint32_t *W = (uint32_t *)alloc_sram(rows * wpr * 4, 16);
        uint32_t *x = (uint32_t *)alloc_sram(wpr * 4, 16);
        if (!W || !x) {
            if (W) heap_caps_free(W);
            if (x) heap_caps_free(x);
            continue;
        }
        fill_random_u32(W, rows * wpr, 42);
        fill_random_u32(x, wpr, 99);

        // Verify correctness
        int32_t rn = xnor_matmul_naive(W, x, wpr, rows);
        int32_t rw = xnor_matmul_wegner(W, x, wpr, rows);
        int32_t rb = xnor_matmul_builtin(W, x, wpr, rows);
        int32_t rf = xnor_matmul_fast(W, x, wpr, rows);
        int32_t rl = xnor_matmul_lut(W, x, wpr, rows);
        if (rn != rw || rn != rb || rn != rf || rn != rl) {
            printf("  MISMATCH at %dx%d\n", rows, cols);
        }

        float un = 0, uw = 0, ub = 0, uf = 0, ul = 0;

        #define TIME_XNOR(var, fn) do { \
            int64_t t0 = bench_start(); \
            for (int i = 0; i < xnor_iters; i++) fn(W, x, wpr, rows); \
            var = bench_stop_us(t0) / (float)xnor_iters; \
        } while(0)

        TIME_XNOR(un, xnor_matmul_naive);
        TIME_XNOR(uw, xnor_matmul_wegner);
        TIME_XNOR(ub, xnor_matmul_builtin);
        TIME_XNOR(uf, xnor_matmul_fast);
        TIME_XNOR(ul, xnor_matmul_lut);
        #undef TIME_XNOR

        float best = uf;
        if (ul < best) best = ul;
        if (ub < best) best = ub;
        float gmacs = (float)rows * cols / (best * 1000.0f);

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", rows, cols);
        printf("  %-14s  %8.1f  %8.1f  %8.1f  %8.1f  %8.1f  %10.2f\n",
               label, un, uw, ub, uf, ul, gmacs);

        heap_caps_free(W);
        heap_caps_free(x);
    }

    // --- XNOR vs INT8 PIE ---

    bench_subsection("XNOR (best) vs INT8 PIE: same dimensions from SRAM");

    int cmp[][2] = { {128, 1024}, {256, 256}, {512, 384} };
    int n_cmp = 3;
    int cmp_iters = 200;

    printf("  %-12s  %10s  %10s  %10s\n", "RxC", "XNOR us", "PIE us", "PIE faster");
    printf("  -------------------------------------------------\n");

    for (int c = 0; c < n_cmp; c++) {
        int rows = cmp[c][0], cols = cmp[c][1];
        int cp = (cols + 15) & ~15;
        int wpr = (cols + 31) / 32;

        uint32_t *Wb = (uint32_t *)alloc_sram(rows * wpr * 4, 16);
        uint32_t *xb = (uint32_t *)alloc_sram(wpr * 4, 16);
        int8_t *Wi = (int8_t *)alloc_sram(rows * cp, 16);
        int8_t *xi = (int8_t *)alloc_sram(cp, 16);
        int32_t *yi = (int32_t *)alloc_sram(rows * 4, 16);

        if (!Wb || !xb || !Wi || !xi || !yi) {
            if (Wb) heap_caps_free(Wb);
            if (xb) heap_caps_free(xb);
            if (Wi) heap_caps_free(Wi);
            if (xi) heap_caps_free(xi);
            if (yi) heap_caps_free(yi);
            continue;
        }
        fill_random_u32(Wb, rows * wpr, 42);
        fill_random_u32(xb, wpr, 99);
        fill_random_s8(Wi, rows * cp, 42);
        fill_random_s8(xi, cols, 99);

        int64_t t0 = bench_start();
        for (int i = 0; i < cmp_iters; i++)
            xnor_matmul_fast(Wb, xb, wpr, rows);
        float xnor_us = bench_stop_us(t0) / (float)cmp_iters;

        t0 = bench_start();
        for (int i = 0; i < cmp_iters; i++)
            pie_matmul_s8_xacc(Wi, xi, yi, rows, cp);
        float pie_us = bench_stop_us(t0) / (float)cmp_iters;

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", rows, cols);
        printf("  %-12s  %10.1f  %10.1f  %10.1fx\n",
               label, xnor_us, pie_us, xnor_us / pie_us);

        heap_caps_free(Wb);
        heap_caps_free(xb);
        heap_caps_free(Wi);
        heap_caps_free(xi);
        heap_caps_free(yi);
    }
}
