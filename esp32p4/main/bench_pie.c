#include "bench.h"

static int32_t scalar_dot_s8(const int8_t *a, const int8_t *b, int n)
{
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static void scalar_matmul_s8(const int8_t *W, const int8_t *x, int32_t *y,
                               int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        int32_t sum = 0;
        for (int c = 0; c < cols; c++) sum += (int32_t)W[r * cols + c] * (int32_t)x[c];
        y[r] = sum;
    }
}

void bench_pie(void)
{
    bench_separator("PIE XACC THROUGHPUT");

    // --- Cycle-accurate single dot product ---

    bench_subsection("Single Dot Product: PIE vs Scalar (cycle counter)");

    int dot_sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_dot = sizeof(dot_sizes) / sizeof(dot_sizes[0]);

    printf("  %-8s  %10s  %10s  %10s  %10s  %10s\n",
           "Cols", "Scalar cyc", "PIE cyc", "Speedup", "PIE cyc/el", "PIE MAC/c");
    printf("  -----------------------------------------------------------------------\n");

    for (int d = 0; d < n_dot; d++) {
        int cols = dot_sizes[d];
        int cp = (cols + 15) & ~15;

        int8_t *W = (int8_t *)alloc_sram(cp, 16);
        int8_t *x = (int8_t *)alloc_sram(cp, 16);
        int32_t y_s, y_p;

        if (!W || !x) {
            if (W) heap_caps_free(W);
            if (x) heap_caps_free(x);
            continue;
        }
        fill_random_s8(W, cols, 42);
        fill_random_s8(x, cols, 99);

        // Warmup
        for (int w = 0; w < 20; w++) {
            y_s = scalar_dot_s8(W, x, cols);
            pie_dot_s8(W, x, &y_p, cp);
        }

        // Verify
        y_s = scalar_dot_s8(W, x, cols);
        pie_dot_s8(W, x, &y_p, cp);
        if (y_s != y_p) {
            printf("  %-8d  MISMATCH scalar=%ld pie=%ld\n", cols, (long)y_s, (long)y_p);
        }

        // Benchmark with cycle counter, many reps to amortize measurement overhead
        int reps = 10000;

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < reps; i++) y_s = scalar_dot_s8(W, x, cols);
        uint32_t c1 = read_mcycle();
        float scalar_cyc = (float)(c1 - c0) / (float)reps;

        c0 = read_mcycle();
        for (int i = 0; i < reps; i++) pie_dot_s8(W, x, &y_p, cp);
        c1 = read_mcycle();
        float pie_cyc = (float)(c1 - c0) / (float)reps;

        float speedup = scalar_cyc / pie_cyc;
        float cyc_per_el = pie_cyc / (float)cols;
        float mac_per_cyc = (float)cols / pie_cyc;

        printf("  %-8d  %10.1f  %10.1f  %10.1fx  %10.3f  %10.1f\n",
               cols, scalar_cyc, pie_cyc, speedup, cyc_per_el, mac_per_cyc);

        heap_caps_free(W);
        heap_caps_free(x);
    }

    // --- PIE instruction-level timing ---

    bench_subsection("PIE Instruction Timing: cycles per esp.vmulas.s8.xacc");

    // Use large matmul to amortize loop overhead
    // At 256x256 from SRAM: total MACs = 65536, measured cycles tells us per-MAC cost
    {
        int rows = 256, cols = 256;
        int total = rows * cols;

        int8_t *W = (int8_t *)alloc_sram(total, 16);
        int8_t *x = (int8_t *)alloc_sram(cols, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);

        if (W && x && y) {
            fill_random_s8(W, total, 42);
            fill_random_s8(x, cols, 99);

            // Warmup
            for (int w = 0; w < 10; w++)
                pie_matmul_s8_xacc(W, x, y, rows, cols);

            int reps = 1000;
            uint32_t c0 = read_mcycle();
            for (int i = 0; i < reps; i++)
                pie_matmul_s8_xacc(W, x, y, rows, cols);
            uint32_t c1 = read_mcycle();

            float total_cyc = (float)(c1 - c0) / (float)reps;
            float total_macs = (float)rows * cols;
            float cyc_per_mac = total_cyc / total_macs;
            float macs_per_cyc = total_macs / total_cyc;

            // Each vmulas processes 16 elements, so cycles per instruction:
            float vmulas_count = total_macs / 16.0f;
            float cyc_per_vmulas = total_cyc / vmulas_count;

            printf("  Test: %dx%d matmul (%d reps)\n", rows, cols, reps);
            printf("  Total cycles/call:          %10.0f\n", total_cyc);
            printf("  Total MACs/call:            %10.0f\n", total_macs);
            printf("  Cycles/MAC:                 %10.3f\n", cyc_per_mac);
            printf("  MACs/cycle:                 %10.1f\n", macs_per_cyc);
            printf("  vmulas.s8.xacc count/call:  %10.0f\n", vmulas_count);
            printf("  Cycles/vmulas instruction:  %10.1f\n", cyc_per_vmulas);
            printf("  (16 MACs per vmulas, so %.1f MACs/cycle effective)\n", 16.0f / cyc_per_vmulas);
        }

        if (W) heap_caps_free(W);
        if (x) heap_caps_free(x);
        if (y) heap_caps_free(y);
    }

    // --- Multi-row matmul ---

    bench_subsection("Multi-Row Matmul: PIE vs Scalar (from SRAM, cycle counter)");

    int mat_configs[][2] = {
        {64, 64}, {64, 256}, {128, 256}, {256, 256},
        {512, 384}, {1024, 256}, {128, 1024},
    };
    int n_configs = sizeof(mat_configs) / sizeof(mat_configs[0]);

    printf("  %-12s  %10s  %10s  %10s  %10s  %10s\n",
           "RxC", "Scalar cyc", "PIE cyc", "Speedup", "GMAC/s", "MAC/cyc");
    printf("  -------------------------------------------------------------------\n");

    for (int c = 0; c < n_configs; c++) {
        int rows = mat_configs[c][0], cols = mat_configs[c][1];
        int cp = (cols + 15) & ~15;
        int total = rows * cp;

        int8_t *W = (int8_t *)alloc_sram(total, 16);
        int8_t *x = (int8_t *)alloc_sram(cp, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);

        if (!W || !x || !y) {
            char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
            printf("  %-12s  ALLOC FAILED\n", l);
            if (W) heap_caps_free(W);
            if (x) heap_caps_free(x);
            if (y) heap_caps_free(y);
            continue;
        }
        fill_random_s8(W, total, 42);
        fill_random_s8(x, cols, 99);

        int reps_s = (rows * cols < 100000) ? 200 : 50;
        int reps_p = (rows * cols < 100000) ? 1000 : 200;

        // Warmup
        for (int w = 0; w < 5; w++) { scalar_matmul_s8(W, x, y, rows, cols); pie_matmul_s8_xacc(W, x, y, rows, cp); }

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < reps_s; i++) scalar_matmul_s8(W, x, y, rows, cols);
        uint32_t c1 = read_mcycle();
        float scalar_cyc = (float)(c1 - c0) / (float)reps_s;

        c0 = read_mcycle();
        for (int i = 0; i < reps_p; i++) pie_matmul_s8_xacc(W, x, y, rows, cp);
        c1 = read_mcycle();
        float pie_cyc = (float)(c1 - c0) / (float)reps_p;

        float macs = (float)rows * cols;
        float speedup = scalar_cyc / pie_cyc;
        float mac_c = macs / pie_cyc;
        float pie_us = pie_cyc / (float)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
        float gmacs = macs / (pie_us * 1000.0f);

        char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
        printf("  %-12s  %10.0f  %10.0f  %10.1fx  %10.2f  %10.1f\n",
               l, scalar_cyc, pie_cyc, speedup, gmacs, mac_c);

        heap_caps_free(W);
        heap_caps_free(x);
        heap_caps_free(y);
    }

    // --- Call overhead ---

    bench_subsection("PIE Call Overhead: 1 call vs N calls (cycle counter)");

    int oh_cols = 256;
    int oh_rows_list[] = {16, 64, 256, 1024};
    int n_oh = 4;

    printf("  %-10s  %12s  %12s  %10s\n", "Rows", "N calls cyc", "1 call cyc", "Overhead x");
    printf("  -------------------------------------------------------\n");

    for (int o = 0; o < n_oh; o++) {
        int rows = oh_rows_list[o];
        int total = rows * oh_cols;

        int8_t *W = (int8_t *)alloc_sram(total, 16);
        int8_t *x = (int8_t *)alloc_sram(oh_cols, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);
        if (!W || !x || !y) {
            if (W) heap_caps_free(W);
            if (x) heap_caps_free(x);
            if (y) heap_caps_free(y);
            continue;
        }
        fill_random_s8(W, total, 42);
        fill_random_s8(x, oh_cols, 99);

        int reps = 200;

        uint32_t c0 = read_mcycle();
        for (int it = 0; it < reps; it++)
            for (int r = 0; r < rows; r++)
                pie_matmul_s8_xacc(W + r * oh_cols, x, y + r, 1, oh_cols);
        uint32_t c1 = read_mcycle();
        float n_cyc = (float)(c1 - c0) / (float)reps;

        c0 = read_mcycle();
        for (int it = 0; it < reps; it++)
            pie_matmul_s8_xacc(W, x, y, rows, oh_cols);
        c1 = read_mcycle();
        float one_cyc = (float)(c1 - c0) / (float)reps;

        printf("  %-10d  %12.0f  %12.0f  %10.2fx\n",
               rows, n_cyc, one_cyc, n_cyc / one_cyc);

        heap_caps_free(W);
        heap_caps_free(x);
        heap_caps_free(y);
    }

    // --- PIE from PSRAM at various row sizes ---

    bench_subsection("PIE Source: SRAM vs PSRAM direct vs PSRAM row-copy");

    int ps_configs[][2] = {
        {256, 128}, {256, 256}, {256, 512}, {256, 1024},
        {512, 256}, {512, 384}, {512, 512},
        {1024, 256},
    };
    int n_ps = sizeof(ps_configs) / sizeof(ps_configs[0]);

    printf("  %-12s  %10s  %10s  %10s  %10s  %10s\n",
           "RxC", "SRAM us", "PSRAM dir", "PSRAM copy", "Dir/SRAM", "Copy/SRAM");
    printf("  -------------------------------------------------------------------\n");

    for (int p = 0; p < n_ps; p++) {
        int rows = ps_configs[p][0], cols = ps_configs[p][1];
        int cp = (cols + 15) & ~15;
        int total = rows * cp;
        int iters = 50;

        int8_t *Ws = (int8_t *)alloc_sram(total, 16);
        int8_t *Wp = (int8_t *)alloc_psram(total, 16);
        int8_t *rb = (int8_t *)alloc_sram(cp, 16);
        int8_t *x = (int8_t *)alloc_sram(cp, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);

        if (!Ws || !Wp || !rb || !x || !y) {
            char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
            printf("  %-12s  ALLOC FAILED\n", l);
            if (Ws) heap_caps_free(Ws);
            if (Wp) heap_caps_free(Wp);
            if (rb) heap_caps_free(rb);
            if (x) heap_caps_free(x);
            if (y) heap_caps_free(y);
            continue;
        }
        fill_random_s8(Ws, total, 42);
        memcpy(Wp, Ws, total);
        fill_random_s8(x, cols, 99);

        // SRAM
        int64_t t0 = bench_start();
        for (int i = 0; i < iters; i++) pie_matmul_s8_xacc(Ws, x, y, rows, cp);
        float sram_us = bench_stop_us(t0) / (float)iters;

        // PSRAM direct
        t0 = bench_start();
        for (int i = 0; i < iters; i++)
            for (int r = 0; r < rows; r++)
                pie_matmul_s8_xacc(Wp + r * cp, x, y + r, 1, cp);
        float dir_us = bench_stop_us(t0) / (float)iters;

        // PSRAM row-copy
        t0 = bench_start();
        for (int i = 0; i < iters; i++)
            for (int r = 0; r < rows; r++) {
                memcpy(rb, Wp + r * cp, cp);
                pie_matmul_s8_xacc(rb, x, y + r, 1, cp);
            }
        float copy_us = bench_stop_us(t0) / (float)iters;

        char l[32]; snprintf(l, 32, "%dx%d", rows, cols);
        printf("  %-12s  %8.1f  %8.1f  %10.1f  %8.1fx  %8.1fx\n",
               l, sram_us, dir_us, copy_us, dir_us / sram_us, copy_us / sram_us);

        heap_caps_free(Ws);
        heap_caps_free(Wp);
        heap_caps_free(rb);
        heap_caps_free(x);
        heap_caps_free(y);
    }
}
