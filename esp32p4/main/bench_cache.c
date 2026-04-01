#include "bench.h"

// ---------------------------------------------------------------------------
// Measure sequential read from flash-mapped data (const arrays in .rodata)
// ---------------------------------------------------------------------------

// Large const array that lives in flash (rodata)
static const uint32_t flash_data[16384] = {
    // Compiler will place this in flash. Fill with pattern.
    [0 ... 16383] = 0xDEADBEEF
};

static volatile int32_t cache_sink;

void bench_cache(void)
{
    bench_separator("CACHE AND FLASH ACCESS PATTERNS");

    // --- Flash sequential read (through cache) ---

    bench_subsection("Flash (.rodata) Sequential Read");

    int flash_sizes[] = {256, 1024, 4096, 16384};
    int n_fs = sizeof(flash_sizes) / sizeof(flash_sizes[0]);

    printf("  %-10s  %12s  %10s\n", "Words", "us", "MB/s");
    printf("  ----------------------------------------\n");

    for (int f = 0; f < n_fs; f++) {
        int n_words = flash_sizes[f];
        int iters = 1000;

        volatile uint32_t sum = 0;
        int64_t t0 = bench_start();
        for (int it = 0; it < iters; it++) {
            uint32_t s = 0;
            for (int i = 0; i < n_words; i++) {
                s += flash_data[i];
            }
            sum += s;
        }
        float us = bench_stop_us(t0);
        cache_sink = sum;

        float bytes = (float)(n_words * 4) * iters;
        float mb_s = bytes / us;
        printf("  %-10d  %10.1f  %10.1f\n", n_words, us / iters, mb_s);
    }

    // --- SRAM sequential read at matching sizes ---

    bench_subsection("SRAM Sequential Read (same sizes for comparison)");

    printf("  %-10s  %12s  %10s\n", "Words", "us", "MB/s");
    printf("  ----------------------------------------\n");

    for (int f = 0; f < n_fs; f++) {
        int n_words = flash_sizes[f];
        int iters = 1000;

        uint32_t *sram_data = (uint32_t *)alloc_sram(n_words * 4, 16);
        if (!sram_data) continue;
        for (int i = 0; i < n_words; i++) sram_data[i] = 0xDEADBEEF;

        volatile uint32_t sum = 0;
        int64_t t0 = bench_start();
        for (int it = 0; it < iters; it++) {
            uint32_t s = 0;
            for (int i = 0; i < n_words; i++) {
                s += sram_data[i];
            }
            sum += s;
        }
        float us = bench_stop_us(t0);
        cache_sink = sum;

        float bytes = (float)(n_words * 4) * iters;
        float mb_s = bytes / us;
        printf("  %-10d  %10.1f  %10.1f\n", n_words, us / iters, mb_s);

        heap_caps_free(sram_data);
    }

    // --- Stride access pattern (cache line effects) ---

    bench_subsection("Stride Access: cache line sensitivity");

    size_t buf_sz = 256 * 1024;  // 256 KB
    uint32_t *buf = (uint32_t *)alloc_sram(buf_sz, 16);
    if (!buf) {
        // Try PSRAM
        buf = (uint32_t *)alloc_psram(buf_sz, 16);
    }

    if (buf) {
        int n_words = buf_sz / 4;
        for (int i = 0; i < n_words; i++) buf[i] = (uint32_t)i;

        int strides[] = {1, 2, 4, 8, 16, 32, 64, 128};
        int n_strides = sizeof(strides) / sizeof(strides[0]);
        int accesses = 100000;

        printf("  Buffer in: %s\n", (uintptr_t)buf >= 0x48000000 ? "PSRAM" : "SRAM");
        printf("  %-10s  %10s  %10s\n", "Stride", "ns/access", "MB/s");
        printf("  ----------------------------------------\n");

        for (int s = 0; s < n_strides; s++) {
            int stride = strides[s];
            volatile uint32_t sum = 0;

            int64_t t0 = bench_start();
            uint32_t idx = 0;
            for (int i = 0; i < accesses; i++) {
                sum += buf[idx % n_words];
                idx += stride;
            }
            float us = bench_stop_us(t0);
            cache_sink = sum;

            float ns = (us * 1000.0f) / (float)accesses;
            float mb_s = ((float)accesses * 4.0f) / us;  // effective MB/s
            printf("  %-10d  %10.1f  %10.1f\n", stride, ns, mb_s);
        }

        heap_caps_free(buf);
    }

    // --- Working set sweep: L1 cache boundary detection ---

    bench_subsection("Working Set Sweep: detect cache boundaries");

    int ws_sizes_kb[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    int n_ws = sizeof(ws_sizes_kb) / sizeof(ws_sizes_kb[0]);
    int ws_accesses = 500000;

    printf("  %-10s  %12s  %12s\n", "Size KB", "SRAM ns/acc", "PSRAM ns/acc");
    printf("  ------------------------------------------\n");

    for (int w = 0; w < n_ws; w++) {
        size_t sz = (size_t)ws_sizes_kb[w] * 1024;
        int n_words = sz / 4;

        // SRAM
        float sram_ns = -1;
        uint32_t *s_buf = (uint32_t *)alloc_sram(sz, 16);
        if (s_buf) {
            for (int i = 0; i < n_words; i++) s_buf[i] = (uint32_t)(i * 7 + 3) % n_words;
            volatile uint32_t idx = 0;
            int64_t t0 = bench_start();
            for (int i = 0; i < ws_accesses; i++) idx = s_buf[idx];
            float us = bench_stop_us(t0);
            cache_sink = idx;
            sram_ns = (us * 1000.0f) / (float)ws_accesses;
            heap_caps_free(s_buf);
        }

        // PSRAM
        float psram_ns = -1;
        uint32_t *p_buf = (uint32_t *)alloc_psram(sz, 16);
        if (p_buf) {
            for (int i = 0; i < n_words; i++) p_buf[i] = (uint32_t)(i * 7 + 3) % n_words;
            volatile uint32_t idx = 0;
            int64_t t0 = bench_start();
            for (int i = 0; i < ws_accesses; i++) idx = p_buf[idx];
            float us = bench_stop_us(t0);
            cache_sink = idx;
            psram_ns = (us * 1000.0f) / (float)ws_accesses;
            heap_caps_free(p_buf);
        }

        printf("  %-10d  ", ws_sizes_kb[w]);
        if (sram_ns > 0) printf("%10.1f  ", sram_ns); else printf("      N/A   ");
        if (psram_ns > 0) printf("%10.1f", psram_ns); else printf("      N/A");
        printf("\n");
    }
}
