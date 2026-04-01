#include "bench.h"

// ---------------------------------------------------------------------------
// INT4 Unpack Strategies
//
// The first benchmark run showed INT4 unpack is 74.5% of total pipeline time.
// This file compares different unpack approaches to find the fastest.
// ---------------------------------------------------------------------------

// Strategy 1: Naive byte-by-byte (original)
static void unpack_naive(const uint8_t *packed, int8_t *out, int n_values)
{
    int n_bytes = n_values / 2;
    for (int i = 0; i < n_bytes; i++) {
        uint8_t b = packed[i];
        out[i * 2]     = (int8_t)((b & 0x0F) - 8);
        out[i * 2 + 1] = (int8_t)(((b >> 4) & 0x0F) - 8);
    }
}

// Strategy 2: Unrolled x4 (process 4 bytes = 8 values per iteration)
static void unpack_unrolled4(const uint8_t *packed, int8_t *out, int n_values)
{
    int n_bytes = n_values / 2;
    int i = 0;
    for (; i + 3 < n_bytes; i += 4) {
        uint8_t b0 = packed[i], b1 = packed[i+1], b2 = packed[i+2], b3 = packed[i+3];
        out[i*2]   = (int8_t)((b0 & 0x0F) - 8);
        out[i*2+1] = (int8_t)(((b0 >> 4) & 0x0F) - 8);
        out[i*2+2] = (int8_t)((b1 & 0x0F) - 8);
        out[i*2+3] = (int8_t)(((b1 >> 4) & 0x0F) - 8);
        out[i*2+4] = (int8_t)((b2 & 0x0F) - 8);
        out[i*2+5] = (int8_t)(((b2 >> 4) & 0x0F) - 8);
        out[i*2+6] = (int8_t)((b3 & 0x0F) - 8);
        out[i*2+7] = (int8_t)(((b3 >> 4) & 0x0F) - 8);
    }
    for (; i < n_bytes; i++) {
        uint8_t b = packed[i];
        out[i*2]   = (int8_t)((b & 0x0F) - 8);
        out[i*2+1] = (int8_t)(((b >> 4) & 0x0F) - 8);
    }
}

// Strategy 3: Word-level (process 4 bytes as uint32, extract with shifts)
static void unpack_wordlevel(const uint8_t *packed, int8_t *out, int n_values)
{
    int n_words = n_values / 8;  // 4 packed bytes = 8 values per word
    const uint32_t *wp = (const uint32_t *)packed;

    for (int w = 0; w < n_words; w++) {
        uint32_t word = wp[w];
        int8_t *dst = out + w * 8;
        dst[0] = (int8_t)((word & 0x0F) - 8);
        dst[1] = (int8_t)(((word >> 4) & 0x0F) - 8);
        dst[2] = (int8_t)(((word >> 8) & 0x0F) - 8);
        dst[3] = (int8_t)(((word >> 12) & 0x0F) - 8);
        dst[4] = (int8_t)(((word >> 16) & 0x0F) - 8);
        dst[5] = (int8_t)(((word >> 20) & 0x0F) - 8);
        dst[6] = (int8_t)(((word >> 24) & 0x0F) - 8);
        dst[7] = (int8_t)(((word >> 28) & 0x0F) - 8);
    }
}

// Strategy 4: LUT-based (256-entry table, each entry is 2 int8 values)
static int16_t unpack_lut[256];  // pairs of int8 packed as int16

static void init_unpack_lut(void)
{
    for (int i = 0; i < 256; i++) {
        int8_t lo = (int8_t)((i & 0x0F) - 8);
        int8_t hi = (int8_t)(((i >> 4) & 0x0F) - 8);
        // Store as pair: lo in low byte, hi in high byte
        unpack_lut[i] = (int16_t)((uint8_t)lo | ((uint8_t)hi << 8));
    }
}

static void unpack_lut_based(const uint8_t *packed, int8_t *out, int n_values)
{
    int n_bytes = n_values / 2;
    int16_t *out16 = (int16_t *)out;
    for (int i = 0; i < n_bytes; i++) {
        out16[i] = unpack_lut[packed[i]];
    }
}

// Strategy 5: SIMD-style with word packing (extract nibbles via masks, no branching)
static void unpack_simd_style(const uint8_t *packed, int8_t *out, int n_values)
{
    // Process 4 packed bytes at a time, produce 8 output bytes
    int n_words = n_values / 8;
    const uint32_t *src = (const uint32_t *)packed;
    uint32_t *dst = (uint32_t *)out;

    for (int w = 0; w < n_words; w++) {
        uint32_t v = src[w];

        // Extract low nibbles of each byte: bytes 0,1,2,3 -> lo nibbles
        uint32_t lo = v & 0x0F0F0F0Fu;
        // Extract high nibbles shifted down
        uint32_t hi = (v >> 4) & 0x0F0F0F0Fu;

        // Subtract 8 from each byte (INT4 zero-centering)
        // 0x08080808 subtracted from each byte
        lo = lo - 0x08080808u;
        hi = hi - 0x08080808u;

        // Interleave: lo[0],hi[0],lo[1],hi[1],lo[2],hi[2],lo[3],hi[3]
        // This requires byte interleave which is tricky without SIMD
        // Instead store as two words: even positions and odd positions
        // Output layout: lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3
        int8_t *d = (int8_t *)(dst + w * 2);
        int8_t *l = (int8_t *)&lo;
        int8_t *h = (int8_t *)&hi;
        d[0] = l[0]; d[1] = h[0];
        d[2] = l[1]; d[3] = h[1];
        d[4] = l[2]; d[5] = h[2];
        d[6] = l[3]; d[7] = h[3];
    }
}

// Strategy 6: Deinterleaved output (low nibbles first, then high nibbles)
// This avoids the interleave step and may be PIE-friendly
static void unpack_deinterleaved(const uint8_t *packed, int8_t *out_lo, int8_t *out_hi,
                                   int n_values)
{
    int n_words = n_values / 8;
    const uint32_t *src = (const uint32_t *)packed;
    uint32_t *dlo = (uint32_t *)out_lo;
    uint32_t *dhi = (uint32_t *)out_hi;

    for (int w = 0; w < n_words; w++) {
        uint32_t v = src[w];
        dlo[w] = (v & 0x0F0F0F0Fu) - 0x08080808u;
        dhi[w] = ((v >> 4) & 0x0F0F0F0Fu) - 0x08080808u;
    }
}

// ---------------------------------------------------------------------------
// Verify correctness
// ---------------------------------------------------------------------------

static int verify_unpack(const uint8_t *packed, int n_values)
{
    int8_t *ref = (int8_t *)alloc_sram(n_values, 16);
    int8_t *tst = (int8_t *)alloc_sram(n_values, 16);
    if (!ref || !tst) {
        if (ref) heap_caps_free(ref);
        if (tst) heap_caps_free(tst);
        return -1;
    }

    unpack_naive(packed, ref, n_values);

    int ok = 1;
    #define CHECK(name, fn) do { \
        fn(packed, tst, n_values); \
        for (int i = 0; i < n_values; i++) { \
            if (ref[i] != tst[i]) { \
                printf("  %s MISMATCH at %d: ref=%d got=%d\n", name, i, ref[i], tst[i]); \
                ok = 0; break; \
            } \
        } \
        if (ok) printf("  %s: OK\n", name); \
    } while(0)

    CHECK("Unrolled4", unpack_unrolled4);
    CHECK("Word-level", unpack_wordlevel);
    CHECK("LUT-based", unpack_lut_based);
    // simd_style and deinterleaved have different output ordering, skip verify
    #undef CHECK

    heap_caps_free(ref);
    heap_caps_free(tst);
    return ok;
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

void bench_int4(void)
{
    bench_separator("INT4 UNPACK STRATEGIES (critical bottleneck)");

    init_unpack_lut();

    // --- Verify correctness ---

    bench_subsection("Correctness verification (1024 values)");

    uint8_t *packed = (uint8_t *)alloc_sram(512, 16);  // 512 bytes = 1024 nibbles
    if (!packed) { printf("  ALLOC FAILED\n"); return; }
    uint32_t state = 42;
    for (int i = 0; i < 512; i++) packed[i] = (uint8_t)(bench_xorshift(&state) & 0xFF);
    verify_unpack(packed, 1024);
    heap_caps_free(packed);

    // --- Raw unpack throughput: cycle counter ---

    bench_subsection("Raw Unpack Throughput (cycles per output value)");

    int test_sizes[] = {256, 512, 1024, 2048, 4096};
    int n_tests = 5;

    printf("  %-10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
           "Values", "Naive", "Unroll4", "Word", "LUT", "SIMD-sty", "cyc/val best");
    printf("  %-10s  %10s  %10s  %10s  %10s  %10s\n",
           "", "cyc/val", "cyc/val", "cyc/val", "cyc/val", "cyc/val");
    printf("  --------------------------------------------------------------------------\n");

    for (int t = 0; t < n_tests; t++) {
        int n_val = test_sizes[t];
        int n_packed = n_val / 2;

        uint8_t *pk = (uint8_t *)alloc_sram(n_packed, 16);
        int8_t *out = (int8_t *)alloc_sram(n_val, 16);
        if (!pk || !out) {
            if (pk) heap_caps_free(pk);
            if (out) heap_caps_free(out);
            continue;
        }

        state = 42;
        for (int i = 0; i < n_packed; i++) pk[i] = (uint8_t)(bench_xorshift(&state) & 0xFF);

        int reps = 10000;

        #define TIME_UNPACK(fn) ({ \
            for (int w = 0; w < 10; w++) fn(pk, out, n_val); \
            uint32_t c0 = read_mcycle(); \
            for (int i = 0; i < reps; i++) fn(pk, out, n_val); \
            uint32_t c1 = read_mcycle(); \
            (float)(c1 - c0) / (float)(reps * n_val); \
        })

        float naive_cpv   = TIME_UNPACK(unpack_naive);
        float unroll_cpv  = TIME_UNPACK(unpack_unrolled4);
        float word_cpv    = TIME_UNPACK(unpack_wordlevel);
        float lut_cpv     = TIME_UNPACK(unpack_lut_based);
        float simd_cpv    = TIME_UNPACK(unpack_simd_style);

        #undef TIME_UNPACK

        float best = naive_cpv;
        if (unroll_cpv < best) best = unroll_cpv;
        if (word_cpv < best) best = word_cpv;
        if (lut_cpv < best) best = lut_cpv;
        if (simd_cpv < best) best = simd_cpv;

        printf("  %-10d  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f\n",
               n_val, naive_cpv, unroll_cpv, word_cpv, lut_cpv, simd_cpv, best);

        heap_caps_free(pk);
        heap_caps_free(out);
    }

    // --- Full INT4 matmul pipeline comparison ---

    bench_subsection("INT4 Matmul Pipeline: unpack strategy impact on total time");

    int mat_configs[][2] = {
        {256, 256}, {512, 384}, {1024, 256}, {128, 1024},
    };
    int n_mat = 4;
    int mat_iters = 100;

    printf("  %-12s  %10s  %10s  %10s  %10s  %10s\n",
           "RxC", "INT8 ref", "Naive", "Unroll4", "Word", "LUT");
    printf("  %-12s  %10s  %10s  %10s  %10s  %10s\n",
           "", "us", "us", "us", "us", "us");
    printf("  -------------------------------------------------------------------\n");

    for (int m = 0; m < n_mat; m++) {
        int rows = mat_configs[m][0], cols = mat_configs[m][1];
        int cp = (cols + 15) & ~15;
        int packed_per_row = cols / 2;

        int8_t *W8 = (int8_t *)alloc_sram(rows * cp, 16);
        uint8_t *W4 = (uint8_t *)alloc_sram(rows * packed_per_row, 16);
        int8_t *ubuf = (int8_t *)alloc_sram(cp, 16);
        int8_t *x = (int8_t *)alloc_sram(cp, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);

        if (!W8 || !W4 || !ubuf || !x || !y) {
            char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
            printf("  %-12s  ALLOC FAILED\n", l);
            if (W8) heap_caps_free(W8);
            if (W4) heap_caps_free(W4);
            if (ubuf) heap_caps_free(ubuf);
            if (x) heap_caps_free(x);
            if (y) heap_caps_free(y);
            continue;
        }

        fill_random_s8(W8, rows * cp, 42);
        state = 42;
        for (int i = 0; i < rows * packed_per_row; i++) W4[i] = (uint8_t)(bench_xorshift(&state) & 0xFF);
        fill_random_s8(x, cols, 99);

        // INT8 reference
        int64_t t0 = bench_start();
        for (int i = 0; i < mat_iters; i++)
            pie_matmul_s8_xacc(W8, x, y, rows, cp);
        float int8_us = bench_stop_us(t0) / (float)mat_iters;

        // Helper macro for INT4 matmul with specific unpack function
        #define TIME_INT4_MATMUL(fn) ({ \
            int64_t t0 = bench_start(); \
            for (int it = 0; it < mat_iters; it++) { \
                for (int r = 0; r < rows; r++) { \
                    fn(W4 + r * packed_per_row, ubuf, cols); \
                    for (int j = cols; j < cp; j++) ubuf[j] = 0; \
                    pie_matmul_s8_xacc(ubuf, x, y + r, 1, cp); \
                } \
            } \
            bench_stop_us(t0) / (float)mat_iters; \
        })

        float naive_us  = TIME_INT4_MATMUL(unpack_naive);
        float unroll_us = TIME_INT4_MATMUL(unpack_unrolled4);
        float word_us   = TIME_INT4_MATMUL(unpack_wordlevel);
        float lut_us    = TIME_INT4_MATMUL(unpack_lut_based);

        #undef TIME_INT4_MATMUL

        char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
        printf("  %-12s  %8.1f  %8.1f  %8.1f  %8.1f  %8.1f\n",
               l, int8_us, naive_us, unroll_us, word_us, lut_us);

        heap_caps_free(W8);
        heap_caps_free(W4);
        heap_caps_free(ubuf);
        heap_caps_free(x);
        heap_caps_free(y);
    }

    // --- Deinterleaved unpack + double-PIE concept ---

    bench_subsection("Deinterleaved Unpack: separate lo/hi nibbles (potential for 2-pass PIE)");

    int deint_vals = 4096;
    int deint_packed = deint_vals / 2;
    int reps = 5000;

    uint8_t *pk = (uint8_t *)alloc_sram(deint_packed, 16);
    int8_t *out_lo = (int8_t *)alloc_sram(deint_vals / 2, 16);
    int8_t *out_hi = (int8_t *)alloc_sram(deint_vals / 2, 16);
    int8_t *out_interleaved = (int8_t *)alloc_sram(deint_vals, 16);

    if (pk && out_lo && out_hi && out_interleaved) {
        state = 42;
        for (int i = 0; i < deint_packed; i++) pk[i] = (uint8_t)(bench_xorshift(&state) & 0xFF);

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < reps; i++)
            unpack_deinterleaved(pk, out_lo, out_hi, deint_vals);
        uint32_t c1 = read_mcycle();
        float deint_cpv = (float)(c1 - c0) / (float)(reps * deint_vals);

        c0 = read_mcycle();
        for (int i = 0; i < reps; i++)
            unpack_naive(pk, out_interleaved, deint_vals);
        c1 = read_mcycle();
        float naive_cpv = (float)(c1 - c0) / (float)(reps * deint_vals);

        printf("  Deinterleaved: %.2f cycles/value\n", deint_cpv);
        printf("  Naive:         %.2f cycles/value\n", naive_cpv);
        printf("  Speedup:       %.1fx\n", naive_cpv / deint_cpv);
        printf("\n  Note: deinterleaved output produces separate lo/hi arrays.\n");
        printf("  A 2-pass PIE matmul (lo_weight*lo_act + hi_weight*hi_act)\n");
        printf("  could use this directly without interleaving.\n");
    }

    if(pk) heap_caps_free(pk);
    if(out_lo) heap_caps_free(out_lo);
    if(out_hi) heap_caps_free(out_hi);
    if(out_interleaved) heap_caps_free(out_interleaved);
}
