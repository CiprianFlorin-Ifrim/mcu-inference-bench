#include "bench.h"

static volatile int32_t sink;

static float measure_seq_read_bw(const uint8_t *buf, size_t size, int iters)
{
    volatile uint32_t accum = 0;
    int64_t t0 = bench_start();
    for (int it = 0; it < iters; it++) {
        const uint32_t *p = (const uint32_t *)buf;
        int n = size / 4;
        uint32_t sum = 0;
        for (int i = 0; i < n; i++) sum += p[i];
        accum += sum;
    }
    float us = bench_stop_us(t0);
    sink = accum;
    return (float)size * iters / us;  // MB/s
}

static float measure_seq_write_bw(uint8_t *buf, size_t size, int iters)
{
    int64_t t0 = bench_start();
    for (int it = 0; it < iters; it++) {
        uint32_t *p = (uint32_t *)buf;
        int n = size / 4;
        for (int i = 0; i < n; i++) p[i] = (uint32_t)i;
    }
    return (float)size * iters / bench_stop_us(t0);
}

static float measure_memcpy_bw(void *dst, const void *src, size_t size, int iters)
{
    int64_t t0 = bench_start();
    for (int it = 0; it < iters; it++) memcpy(dst, src, size);
    return (float)size * iters / bench_stop_us(t0);
}

static float measure_random_lat_ns(uint32_t *buf, int n_words, int n_chases)
{
    uint32_t state = 12345;
    for (int i = 0; i < n_words; i++) buf[i] = bench_xorshift(&state) % n_words;
    uint32_t idx = 0;
    int64_t t0 = bench_start();
    for (int i = 0; i < n_chases; i++) idx = buf[idx];
    float us = bench_stop_us(t0);
    sink = idx;
    return (us * 1000.0f) / (float)n_chases;
}

void bench_memory(void)
{
    bench_separator("MEMORY BANDWIDTH AND LATENCY");

    printf("Heap:\n");
    printf("  SRAM  total=%6zuKB  free=%6zuKB  max_block=%6zuKB\n",
           heap_caps_get_total_size(MALLOC_CAP_INTERNAL) / 1024,
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
           heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024);
    printf("  PSRAM total=%6zuKB  free=%6zuKB\n",
           heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024,
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);

    // --- Sequential bandwidth ---

    bench_subsection("Sequential Read Bandwidth");

    size_t sizes[] = {4096, 16384, 65536, 262144, 1048576};
    int n_sz = 5;

    printf("  %-10s  %12s  %12s\n", "Size", "SRAM MB/s", "PSRAM MB/s");
    printf("  ------------------------------------------\n");

    for (int s = 0; s < n_sz; s++) {
        size_t sz = sizes[s];
        int iters = (sz < 65536) ? 1000 : (sz < 524288) ? 100 : 20;

        uint8_t *sb = (sz <= 262144) ? (uint8_t *)alloc_sram(sz, 16) : NULL;
        uint8_t *pb = (uint8_t *)alloc_psram(sz, 16);

        float s_bw = -1, p_bw = -1;
        if (sb) { memset(sb, 0x55, sz); s_bw = measure_seq_read_bw(sb, sz, iters); }
        if (pb) { memset(pb, 0x55, sz); p_bw = measure_seq_read_bw(pb, sz, iters); }

        printf("  %-10zu  ", sz);
        if (s_bw > 0) printf("%10.1f  ", s_bw); else printf("      N/A   ");
        if (p_bw > 0) printf("%10.1f", p_bw); else printf("      N/A");
        printf("\n");

        if (sb) heap_caps_free(sb);
        if (pb) heap_caps_free(pb);
    }

    bench_subsection("Sequential Write Bandwidth");

    printf("  %-10s  %12s  %12s\n", "Size", "SRAM MB/s", "PSRAM MB/s");
    printf("  ------------------------------------------\n");

    for (int s = 0; s < n_sz; s++) {
        size_t sz = sizes[s];
        int iters = (sz < 65536) ? 1000 : (sz < 524288) ? 100 : 20;

        uint8_t *sb = (sz <= 262144) ? (uint8_t *)alloc_sram(sz, 16) : NULL;
        uint8_t *pb = (uint8_t *)alloc_psram(sz, 16);

        float s_bw = -1, p_bw = -1;
        if (sb) s_bw = measure_seq_write_bw(sb, sz, iters);
        if (pb) p_bw = measure_seq_write_bw(pb, sz, iters);

        printf("  %-10zu  ", sz);
        if (s_bw > 0) printf("%10.1f  ", s_bw); else printf("      N/A   ");
        if (p_bw > 0) printf("%10.1f", p_bw); else printf("      N/A");
        printf("\n");

        if (sb) heap_caps_free(sb);
        if (pb) heap_caps_free(pb);
    }

    // --- memcpy: use 64 KB buffers so SRAM->SRAM fits ---

    bench_subsection("memcpy Bandwidth (64 KB buffers)");

    size_t cpy = 65536;
    int cpy_it = 500;

    uint8_t *s1 = (uint8_t *)alloc_sram(cpy, 16);
    uint8_t *s2 = (uint8_t *)alloc_sram(cpy, 16);
    uint8_t *p1 = (uint8_t *)alloc_psram(cpy, 16);
    uint8_t *p2 = (uint8_t *)alloc_psram(cpy, 16);

    if (s1) memset(s1, 0xAA, cpy);
    if (p1) memset(p1, 0xBB, cpy);

    if (s1 && s2) printf("  SRAM  -> SRAM   %10.1f MB/s\n", measure_memcpy_bw(s2, s1, cpy, cpy_it));
    if (p1 && s2) printf("  PSRAM -> SRAM   %10.1f MB/s\n", measure_memcpy_bw(s2, p1, cpy, cpy_it));
    if (s1 && p2) printf("  SRAM  -> PSRAM  %10.1f MB/s\n", measure_memcpy_bw(p2, s1, cpy, cpy_it));
    if (p1 && p2) printf("  PSRAM -> PSRAM  %10.1f MB/s\n", measure_memcpy_bw(p2, p1, cpy, cpy_it));

    if (s1) heap_caps_free(s1);
    if (s2) heap_caps_free(s2);
    if (p1) heap_caps_free(p1);
    if (p2) heap_caps_free(p2);

    // --- memcpy at various sizes (shows cache effects) ---

    bench_subsection("memcpy PSRAM->SRAM at various sizes");

    size_t cpy_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int n_cpy = sizeof(cpy_sizes) / sizeof(cpy_sizes[0]);

    printf("  %-10s  %10s  %10s\n", "Bytes", "us", "MB/s");
    printf("  ------------------------------------\n");

    for (int c = 0; c < n_cpy; c++) {
        size_t sz = cpy_sizes[c];
        int it = (sz < 1024) ? 10000 : (sz < 16384) ? 2000 : 500;

        uint8_t *src = (uint8_t *)alloc_psram(sz, 16);
        uint8_t *dst = (uint8_t *)alloc_sram(sz, 16);
        if (!src || !dst) {
            if (src) heap_caps_free(src);
            if (dst) heap_caps_free(dst);
            continue;
        }
        memset(src, 0xCC, sz);

        int64_t t0 = bench_start();
        for (int i = 0; i < it; i++) memcpy(dst, src, sz);
        float us = bench_stop_us(t0) / (float)it;
        float mb = (float)sz / us;

        printf("  %-10zu  %10.2f  %10.1f\n", sz, us, mb);

        heap_caps_free(src);
        heap_caps_free(dst);
    }

    // --- Random read latency ---

    bench_subsection("Random Read Latency (pointer chasing)");

    size_t lat_sizes[] = {1024, 4096, 16384, 65536, 262144};
    int n_lat = 5;
    int n_chases = 1000000;

    printf("  %-10s  %12s  %12s\n", "Buf size", "SRAM ns", "PSRAM ns");
    printf("  ------------------------------------------\n");

    for (int s = 0; s < n_lat; s++) {
        size_t sz = lat_sizes[s];
        int n_w = sz / 4;

        uint32_t *sb = (sz <= 262144) ? (uint32_t *)alloc_sram(sz, 16) : NULL;
        uint32_t *pb = (uint32_t *)alloc_psram(sz, 16);

        float s_ns = -1, p_ns = -1;
        if (sb) s_ns = measure_random_lat_ns(sb, n_w, n_chases);
        if (pb) p_ns = measure_random_lat_ns(pb, n_w, n_chases);

        printf("  %-10zu  ", sz);
        if (s_ns > 0) printf("%10.1f  ", s_ns); else printf("      N/A   ");
        if (p_ns > 0) printf("%10.1f", p_ns); else printf("      N/A");
        printf("\n");

        if (sb) heap_caps_free(sb);
        if (pb) heap_caps_free(pb);
    }

    // --- Row copy: PSRAM -> SRAM ---

    bench_subsection("Row Copy: PSRAM -> SRAM (weight tiling simulation)");

    int row_sizes[] = {128, 256, 384, 512, 1024, 2048, 4096};
    int n_rs = 7;
    int n_rows = 1024;

    printf("  %-10s  %10s  %10s  %10s\n", "Row bytes", "us/row", "MB/s", "Total ms");
    printf("  ------------------------------------------------\n");

    for (int r = 0; r < n_rs; r++) {
        int rsz = row_sizes[r];
        int total = n_rows * rsz;
        uint8_t *src = (uint8_t *)alloc_psram(total, 16);
        uint8_t *dst = (uint8_t *)alloc_sram(rsz, 16);
        if (!src || !dst) {
            if (src) heap_caps_free(src);
            if (dst) heap_caps_free(dst);
            continue;
        }
        memset(src, 0xAA, total);

        int64_t t0 = bench_start();
        for (int i = 0; i < n_rows; i++) memcpy(dst, src + i * rsz, rsz);
        float us = bench_stop_us(t0);

        printf("  %-10d  %10.2f  %10.1f  %10.2f\n",
               rsz, us / n_rows, (float)total / us, us / 1000.0f);

        heap_caps_free(src);
        heap_caps_free(dst);
    }
}
