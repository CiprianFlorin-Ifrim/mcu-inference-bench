#include "bench.h"
#include "esp_dma_utils.h"

// ---------------------------------------------------------------------------
// DMA benchmarks are platform-specific and may not work on all ESP-IDF versions.
// This file provides memcpy-based simulation of DMA patterns and measures
// the overlap potential between copy and compute.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Double-buffer simulation: overlap memcpy with compute
// ---------------------------------------------------------------------------

static volatile int32_t dma_sink;

static void simulate_double_buffer(const int8_t *W_psram, const int8_t *x_sram,
                                     int8_t *buf_a, int8_t *buf_b,
                                     int32_t *y, int rows, int cols_padded,
                                     int batch_rows, float *total_us)
{
    // In real DMA double-buffering:
    //   - DMA fills buffer A while CPU processes buffer B
    //   - Then swap
    // Here we simulate sequentially but measure separately
    // to show the overlap potential

    int64_t t_copy = 0, t_compute = 0;

    int r = 0;
    int toggle = 0;
    int8_t *bufs[2] = {buf_a, buf_b};

    while (r < rows) {
        int batch = rows - r;
        if (batch > batch_rows) batch = batch_rows;

        int bytes = batch * cols_padded;

        // Copy phase
        int64_t t0 = bench_start();
        memcpy(bufs[toggle], W_psram + r * cols_padded, bytes);
        t_copy += esp_timer_get_time() - t0;

        // Compute phase
        t0 = bench_start();
        pie_matmul_s8_xacc(bufs[toggle], x_sram, y + r, batch, cols_padded);
        t_compute += esp_timer_get_time() - t0;

        toggle ^= 1;
        r += batch;
    }

    *total_us = (float)(t_copy + t_compute);

    printf("    Copy:    %8.1f us\n", (float)t_copy);
    printf("    Compute: %8.1f us\n", (float)t_compute);
    printf("    Total:   %8.1f us (sequential)\n", *total_us);
    printf("    Overlap potential: %.1f us (if fully overlapped)\n",
           (float)(t_copy > t_compute ? t_copy : t_compute));
    printf("    Copy/Compute ratio: %.2f\n",
           (float)t_copy / (float)(t_compute > 0 ? t_compute : 1));
}

void bench_dma(void)
{
    bench_separator("DMA / DOUBLE-BUFFER SIMULATION");

    // --- Double-buffer at various batch sizes ---

    bench_subsection("Double-Buffer: PSRAM->SRAM copy + PIE compute overlap");

    int rows = 1024;
    int cols = 256;
    int cols_padded = 256;
    int total_sz = rows * cols_padded;

    int8_t *W_psram = (int8_t *)alloc_psram(total_sz, 16);
    int8_t *x_sram = (int8_t *)alloc_sram(cols_padded, 16);
    int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);

    if (!W_psram || !x_sram || !y) {
        printf("  ALLOC FAILED\n");
        if (W_psram) heap_caps_free(W_psram);
        if (x_sram) heap_caps_free(x_sram);
        if (y) heap_caps_free(y);
        return;
    }

    fill_random_s8((int8_t *)W_psram, total_sz, 42);
    fill_random_s8(x_sram, cols, 99);

    int batch_sizes[] = {1, 4, 16, 32, 64, 128, 256, 512, 1024};
    int n_batch = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    for (int b = 0; b < n_batch; b++) {
        int batch = batch_sizes[b];
        if (batch > rows) continue;

        int buf_sz = batch * cols_padded;
        int8_t *buf_a = (int8_t *)alloc_sram(buf_sz, 16);
        int8_t *buf_b = (int8_t *)alloc_sram(buf_sz, 16);

        if (!buf_a || !buf_b) {
            printf("  Batch %d: ALLOC FAILED (need %d KB x2)\n", batch, buf_sz / 1024);
            if (buf_a) heap_caps_free(buf_a);
            if (buf_b) heap_caps_free(buf_b);
            continue;
        }

        printf("\n  Batch size: %d rows (%d KB buffer x2)\n", batch, buf_sz / 1024);
        float total_us;
        simulate_double_buffer(W_psram, x_sram, buf_a, buf_b, y,
                                rows, cols_padded, batch, &total_us);

        heap_caps_free(buf_a);
        heap_caps_free(buf_b);
    }

    // --- INT4 double-buffer simulation ---

    bench_subsection("INT4 Double-Buffer: PSRAM->SRAM copy + unpack + PIE");

    int i4_rows = 1024;
    int i4_cols = 512;
    int i4_cols_padded = 512;
    int i4_packed_per_row = i4_cols / 2;

    uint8_t *W4_psram = (uint8_t *)alloc_psram(i4_rows * i4_packed_per_row, 16);
    int8_t *x4_sram = (int8_t *)alloc_sram(i4_cols_padded, 16);
    int32_t *y4 = (int32_t *)alloc_sram(i4_rows * 4, 16);

    // Buffers: packed copy + unpacked for PIE
    int i4_batch = 32;
    uint8_t *i4_copy_buf = (uint8_t *)alloc_sram(i4_batch * i4_packed_per_row, 16);
    int8_t *i4_unpack_buf = (int8_t *)alloc_sram(i4_batch * i4_cols_padded, 16);

    if (W4_psram && x4_sram && y4 && i4_copy_buf && i4_unpack_buf) {
        uint32_t state = 42;
        for (int i = 0; i < i4_rows * i4_packed_per_row; i++)
            W4_psram[i] = (uint8_t)(bench_xorshift(&state) & 0xFF);
        fill_random_s8(x4_sram, i4_cols, 99);

        int64_t t_copy = 0, t_unpack = 0, t_compute = 0;

        int r = 0;
        while (r < i4_rows) {
            int batch = i4_rows - r;
            if (batch > i4_batch) batch = i4_batch;

            // Copy INT4 from PSRAM
            int64_t t0 = bench_start();
            memcpy(i4_copy_buf, W4_psram + r * i4_packed_per_row,
                   batch * i4_packed_per_row);
            t_copy += esp_timer_get_time() - t0;

            // Unpack INT4 -> INT8
            t0 = bench_start();
            for (int br = 0; br < batch; br++) {
                const uint8_t *src = i4_copy_buf + br * i4_packed_per_row;
                int8_t *dst = i4_unpack_buf + br * i4_cols_padded;
                for (int c = 0; c < i4_packed_per_row; c++) {
                    dst[c * 2]     = (int8_t)((src[c] & 0x0F) - 8);
                    dst[c * 2 + 1] = (int8_t)(((src[c] >> 4) & 0x0F) - 8);
                }
            }
            t_unpack += esp_timer_get_time() - t0;

            // PIE
            t0 = bench_start();
            pie_matmul_s8_xacc(i4_unpack_buf, x4_sram, y4 + r,
                                batch, i4_cols_padded);
            t_compute += esp_timer_get_time() - t0;

            r += batch;
        }

        printf("  INT4 pipeline (%dx%d, batch=%d):\n", i4_rows, i4_cols, i4_batch);
        printf("    PSRAM copy:  %8.1f us  (%d KB)\n",
               (float)t_copy, i4_rows * i4_packed_per_row / 1024);
        printf("    Unpack:      %8.1f us\n", (float)t_unpack);
        printf("    PIE compute: %8.1f us\n", (float)t_compute);
        printf("    Total:       %8.1f us\n", (float)(t_copy + t_unpack + t_compute));
        printf("    Copy is %.1f%% of total\n",
               100.0f * (float)t_copy / (float)(t_copy + t_unpack + t_compute));
        printf("    Unpack is %.1f%% of total\n",
               100.0f * (float)t_unpack / (float)(t_copy + t_unpack + t_compute));
        printf("    Compute is %.1f%% of total\n",
               100.0f * (float)t_compute / (float)(t_copy + t_unpack + t_compute));

        // Compare with INT8 at same dims
        int8_t *W8_psram = (int8_t *)alloc_psram(i4_rows * i4_cols_padded, 16);
        int8_t *row_buf = (int8_t *)alloc_sram(i4_cols_padded, 16);
        if (W8_psram && row_buf) {
            fill_random_s8(W8_psram, i4_rows * i4_cols_padded, 42);
            int64_t t0 = bench_start();
            for (int r = 0; r < i4_rows; r++) {
                memcpy(row_buf, W8_psram + r * i4_cols_padded, i4_cols_padded);
                pie_matmul_s8_xacc(row_buf, x4_sram, y4 + r, 1, i4_cols_padded);
            }
            float int8_us = bench_stop_us(t0);

            float int4_total = (float)(t_copy + t_unpack + t_compute);
            printf("\n  Comparison at %dx%d from PSRAM:\n", i4_rows, i4_cols);
            printf("    INT8 row-copy: %8.1f us  (%d KB transferred)\n",
                   int8_us, i4_rows * i4_cols_padded / 1024);
            printf("    INT4 pipeline: %8.1f us  (%d KB transferred)\n",
                   int4_total, i4_rows * i4_packed_per_row / 1024);
            printf("    INT4 speedup:  %.2fx\n", int8_us / int4_total);
        }
        if (W8_psram) heap_caps_free(W8_psram);
        if (row_buf) heap_caps_free(row_buf);
    }

    if (W4_psram) heap_caps_free(W4_psram);
    if (x4_sram) heap_caps_free(x4_sram);
    if (y4) heap_caps_free(y4);
    if (i4_copy_buf) heap_caps_free(i4_copy_buf);
    if (i4_unpack_buf) heap_caps_free(i4_unpack_buf);

    heap_caps_free(W_psram);
    heap_caps_free(x_sram);
    heap_caps_free(y);
}
