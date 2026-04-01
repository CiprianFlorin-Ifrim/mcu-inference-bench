#include "bench.h"

void bench_matmul(void)
{
    bench_separator("END-TO-END MATMUL COMPARISON");

    // --- INT8 PIE from SRAM ---

    bench_subsection("INT8 PIE from SRAM (compute-bound baseline)");

    int configs[][2] = {
        {128, 256}, {256, 256}, {512, 384}, {1024, 256}, {128, 1024},
    };
    int n_cfg = 5;
    int iters = 200;

    printf("  %-12s  %10s  %10s  %10s\n", "RxC", "us", "GMAC/s", "MAC/cyc");
    printf("  ------------------------------------------------\n");

    for (int c = 0; c < n_cfg; c++) {
        int rows = configs[c][0], cols = configs[c][1];
        int cp = (cols + 15) & ~15;

        int8_t *W = (int8_t *)alloc_sram(rows * cp, 16);
        int8_t *x = (int8_t *)alloc_sram(cp, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);
        if (!W || !x || !y) {
            if (W) heap_caps_free(W);
            if (x) heap_caps_free(x);
            if (y) heap_caps_free(y);
            continue;
        }
        fill_random_s8(W, rows * cp, 42);
        fill_random_s8(x, cols, 99);

        // Warmup
        for (int w = 0; w < 10; w++) pie_matmul_s8_xacc(W, x, y, rows, cp);

        int64_t t0 = bench_start();
        for (int i = 0; i < iters; i++) pie_matmul_s8_xacc(W, x, y, rows, cp);
        float us = bench_stop_us(t0) / (float)iters;

        float macs = (float)rows * cols;
        char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
        printf("  %-12s  %8.1f  %10.2f  %10.1f\n",
               l, us, macs / (us * 1000.0f), macs / (us * (float)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ));

        heap_caps_free(W);
        heap_caps_free(x);
        heap_caps_free(y);
    }

    // --- INT8 from PSRAM: direct vs row-copy vs batch ---

    bench_subsection("INT8 from PSRAM: access strategy comparison");

    int ps_configs[][2] = { {512, 384}, {1024, 256}, {512, 1024} };
    int n_ps = 3;
    int ps_iters = 50;

    printf("  %-12s  %10s  %10s  %10s  %10s\n",
           "RxC", "SRAM", "PSRAM dir", "PSRAM row", "PSRAM batch");
    printf("  %-12s  %10s  %10s  %10s  %10s\n", "", "us", "us", "us", "us");
    printf("  -----------------------------------------------------------\n");

    for (int p = 0; p < n_ps; p++) {
        int rows = ps_configs[p][0], cols = ps_configs[p][1];
        int cp = (cols + 15) & ~15;
        int total = rows * cp;

        int8_t *Ws = (int8_t *)alloc_sram(total, 16);
        int8_t *Wp = (int8_t *)alloc_psram(total, 16);
        int8_t *rb = (int8_t *)alloc_sram(cp, 16);
        // Batch buffer: 16 KB
        int batch_rows = 16384 / cp;
        if (batch_rows < 1) batch_rows = 1;
        int8_t *bb = (int8_t *)alloc_sram(batch_rows * cp, 16);
        int8_t *x = (int8_t *)alloc_sram(cp, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);

        if (!Ws || !Wp || !rb || !bb || !x || !y) {
            char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
            printf("  %-12s  ALLOC FAILED\n", l);
            if (Ws) heap_caps_free(Ws);
            if (Wp) heap_caps_free(Wp);
            if (rb) heap_caps_free(rb);
            if (bb) heap_caps_free(bb);
            if (x) heap_caps_free(x);
            if (y) heap_caps_free(y);
            continue;
        }
        fill_random_s8(Ws, total, 42);
        memcpy(Wp, Ws, total);
        fill_random_s8(x, cols, 99);

        // SRAM single call
        int64_t t0 = bench_start();
        for (int i = 0; i < ps_iters; i++) pie_matmul_s8_xacc(Ws, x, y, rows, cp);
        float sram_us = bench_stop_us(t0) / (float)ps_iters;

        // PSRAM direct (row-by-row PIE calls)
        t0 = bench_start();
        for (int i = 0; i < ps_iters; i++)
            for (int r = 0; r < rows; r++)
                pie_matmul_s8_xacc(Wp + r * cp, x, y + r, 1, cp);
        float dir_us = bench_stop_us(t0) / (float)ps_iters;

        // PSRAM row-copy
        t0 = bench_start();
        for (int i = 0; i < ps_iters; i++)
            for (int r = 0; r < rows; r++) {
                memcpy(rb, Wp + r * cp, cp);
                pie_matmul_s8_xacc(rb, x, y + r, 1, cp);
            }
        float row_us = bench_stop_us(t0) / (float)ps_iters;

        // PSRAM batch-copy
        t0 = bench_start();
        for (int i = 0; i < ps_iters; i++) {
            int r = 0;
            while (r < rows) {
                int b = rows - r;
                if (b > batch_rows) b = batch_rows;
                memcpy(bb, Wp + r * cp, b * cp);
                pie_matmul_s8_xacc(bb, x, y + r, b, cp);
                r += b;
            }
        }
        float batch_us = bench_stop_us(t0) / (float)ps_iters;

        char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
        printf("  %-12s  %8.1f  %8.1f  %8.1f  %10.1f\n",
               l, sram_us, dir_us, row_us, batch_us);

        heap_caps_free(Ws);
        heap_caps_free(Wp);
        heap_caps_free(rb);
        heap_caps_free(bb);
        heap_caps_free(x);
        heap_caps_free(y);
    }
}
