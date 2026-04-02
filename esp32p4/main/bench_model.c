#include "bench.h"

#define MODEL_DIM          512
#define BOTTLENECK_DIM     256
#define MAMBA_EXPAND       1024
#define SSM_STATE_DIM      16
#define MAMBA_CONV_KERNEL  4
#define VOCAB_SIZE         140
#define ATTN_HEADS         4
#define ATTN_HEAD_DIM      (BOTTLENECK_DIM / ATTN_HEADS)

void bench_model(void)
{
    bench_separator("MODEL ARCHITECTURE BENCHMARKS");

    bench_subsection("Architecture-Matched Matmul Sizes (INT8 PIE, from SRAM)");

    struct { const char *name; int rows; int cols; } arch_matmuls[] = {
        {"Emb lookup",          1,    MODEL_DIM},
        {"Mamba in_proj",       MAMBA_EXPAND, MODEL_DIM},
        {"Mamba gate_proj",     MAMBA_EXPAND, MODEL_DIM},
        {"Mamba out_proj",      MODEL_DIM, MAMBA_EXPAND},
        {"Proj down",           BOTTLENECK_DIM, MODEL_DIM},
        {"Proj up",             MODEL_DIM, BOTTLENECK_DIM},
        {"Attn QKV",            3 * BOTTLENECK_DIM, BOTTLENECK_DIM},
        {"Attn out",            BOTTLENECK_DIM, BOTTLENECK_DIM},
        {"Bot Mamba in_proj",   BOTTLENECK_DIM * 2, BOTTLENECK_DIM},
        {"Bot Mamba out_proj",  BOTTLENECK_DIM, BOTTLENECK_DIM * 2},
        {"Output head",         VOCAB_SIZE, MODEL_DIM},
    };
    int n_arch = sizeof(arch_matmuls) / sizeof(arch_matmuls[0]);
    int iters = 200;

    printf("  %-22s  %8s  %10s  %10s  %10s\n",
           "Layer", "RxC", "Cycles", "us", "GMAC/s");
    printf("  -----------------------------------------------------------------------\n");

    for (int a = 0; a < n_arch; a++) {
        int rows = arch_matmuls[a].rows;
        int cols = arch_matmuls[a].cols;
        int cp = (cols + 15) & ~15;

        int8_t *W = (int8_t *)alloc_sram(rows * cp, 16);
        int8_t *x = (int8_t *)alloc_sram(cp, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);

        if (!W || !x || !y) {
            printf("  %-22s  ALLOC FAILED\n", arch_matmuls[a].name);
            if (W) heap_caps_free(W);
            if (x) heap_caps_free(x);
            if (y) heap_caps_free(y);
            continue;
        }
        fill_random_s8(W, rows * cp, 42);
        fill_random_s8(x, cols, 99);

        for (int w = 0; w < 10; w++) {
            pie_matmul_s8_xacc(W, x, y, rows, cp);
        }

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < iters; i++) {
            pie_matmul_s8_xacc(W, x, y, rows, cp);
        }
        uint32_t c1 = read_mcycle();
        float cyc = (float)(c1 - c0) / (float)iters;
        float us = cyc / (float)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
        float macs = (float)rows * cols;
        float gmacs = macs / (us * 1000.0f);

        char dim_label[16];
        snprintf(dim_label, sizeof(dim_label), "%dx%d", rows, cols);
        printf("  %-22s  %8s  %10.0f  %10.1f  %10.2f\n",
               arch_matmuls[a].name, dim_label, cyc, us, gmacs);

        heap_caps_free(W);
        heap_caps_free(x);
        heap_caps_free(y);
    }

    bench_subsection("Architecture-Matched Matmul Sizes (INT8 PIE, from PSRAM direct)");

    struct { const char *name; int rows; int cols; } psram_matmuls[] = {
        {"Mamba in_proj",       MAMBA_EXPAND, MODEL_DIM},
        {"Mamba out_proj",      MODEL_DIM, MAMBA_EXPAND},
        {"Proj down",           BOTTLENECK_DIM, MODEL_DIM},
        {"Attn QKV",            3 * BOTTLENECK_DIM, BOTTLENECK_DIM},
        {"Bot Mamba in_proj",   BOTTLENECK_DIM * 2, BOTTLENECK_DIM},
        {"Output head",         VOCAB_SIZE, MODEL_DIM},
    };
    int n_psram = sizeof(psram_matmuls) / sizeof(psram_matmuls[0]);
    int ps_iters = 50;

    printf("  %-22s  %8s  %10s  %10s  %10s\n",
           "Layer", "RxC", "SRAM us", "PSRAM us", "Ratio");
    printf("  -----------------------------------------------------------------------\n");

    for (int a = 0; a < n_psram; a++) {
        int rows = psram_matmuls[a].rows;
        int cols = psram_matmuls[a].cols;
        int cp = (cols + 15) & ~15;

        int8_t *Ws = (int8_t *)alloc_sram(rows * cp, 16);
        int8_t *Wp = (int8_t *)alloc_psram(rows * cp, 16);
        int8_t *x = (int8_t *)alloc_sram(cp, 16);
        int32_t *y = (int32_t *)alloc_sram(rows * 4, 16);

        if (!Ws || !Wp || !x || !y) {
            printf("  %-22s  ALLOC FAILED\n", psram_matmuls[a].name);
            if (Ws) heap_caps_free(Ws);
            if (Wp) heap_caps_free(Wp);
            if (x) heap_caps_free(x);
            if (y) heap_caps_free(y);
            continue;
        }
        fill_random_s8(Ws, rows * cp, 42);
        memcpy(Wp, Ws, rows * cp);
        fill_random_s8(x, cols, 99);

        int64_t t0 = bench_start();
        for (int i = 0; i < ps_iters; i++) {
            pie_matmul_s8_xacc(Ws, x, y, rows, cp);
        }
        float sram_us = bench_stop_us(t0) / (float)ps_iters;

        t0 = bench_start();
        for (int i = 0; i < ps_iters; i++) {
            for (int r = 0; r < rows; r++) {
                pie_matmul_s8_xacc(Wp + r * cp, x, y + r, 1, cp);
            }
        }
        float psram_us = bench_stop_us(t0) / (float)ps_iters;

        char dim_label[16];
        snprintf(dim_label, sizeof(dim_label), "%dx%d", rows, cols);
        printf("  %-22s  %8s  %8.1f  %8.1f  %8.1fx\n",
               psram_matmuls[a].name, dim_label, sram_us, psram_us,
               psram_us / sram_us);

        heap_caps_free(Ws);
        heap_caps_free(Wp);
        heap_caps_free(x);
        heap_caps_free(y);
    }

    bench_subsection("Recurrence Cache Effect: same weights streamed N times from PSRAM");

    int rec_rows = BOTTLENECK_DIM * 2;
    int rec_cols = BOTTLENECK_DIM;
    int rec_cp = (rec_cols + 15) & ~15;
    int rec_total = rec_rows * rec_cp;

    int8_t *Wrec = (int8_t *)alloc_psram(rec_total, 16);
    int8_t *xrec = (int8_t *)alloc_sram(rec_cp, 16);
    int32_t *yrec = (int32_t *)alloc_sram(rec_rows * 4, 16);

    if (Wrec && xrec && yrec) {
        fill_random_s8((int8_t *)Wrec, rec_total, 42);
        fill_random_s8(xrec, rec_cols, 99);

        printf("  Bottleneck matmul %dx%d from PSRAM, consecutive passes:\n\n", rec_rows, rec_cols);
        printf("  %-10s  %10s  %10s\n", "Pass", "us", "vs Pass 1");
        printf("  ----------------------------------------\n");

        for (int pass = 1; pass <= 6; pass++) {
            int8_t *flush = (int8_t *)alloc_psram(256 * 1024, 16);
            if (flush) {
                volatile int8_t sink = 0;
                for (int i = 0; i < 256 * 1024; i += 64) sink += flush[i];
                (void)sink;
                heap_caps_free(flush);
            }

            float pass_times[6];
            for (int p = 0; p < pass; p++) {
                int64_t t0 = bench_start();
                for (int r = 0; r < rec_rows; r++) {
                    pie_matmul_s8_xacc(Wrec + r * rec_cp, xrec, yrec + r, 1, rec_cp);
                }
                pass_times[p] = bench_stop_us(t0);
            }
            printf("  %-10d  %8.1f  ", pass, pass_times[pass - 1]);
            if (pass == 1) {
                printf("    1.00x\n");
            } else {
                printf("    %.2fx\n", pass_times[pass - 1] / pass_times[0]);
            }
        }
    } else {
        printf("  ALLOC FAILED\n");
    }
    if (Wrec) heap_caps_free(Wrec);
    if (xrec) heap_caps_free(xrec);
    if (yrec) heap_caps_free(yrec);

    bench_subsection("Skip Connection Overhead (SRAM store + load)");

    int skip_dims[] = {MODEL_DIM, MODEL_DIM, MODEL_DIM};
    int n_skips = 3;
    int skip_reps = 100000;

    printf("  %-8s  %10s  %12s  %12s\n",
           "Dim", "Bytes", "Store ns", "Load ns");
    printf("  ------------------------------------------------\n");

    for (int s = 0; s < n_skips; s++) {
        int dim = skip_dims[s];
        int bytes = dim * 4;

        float *src = (float *)alloc_sram(bytes, 16);
        float *skip_buf = (float *)alloc_sram(bytes, 16);
        float *dst = (float *)alloc_sram(bytes, 16);

        if (!src || !skip_buf || !dst) {
            if (src) heap_caps_free(src);
            if (skip_buf) heap_caps_free(skip_buf);
            if (dst) heap_caps_free(dst);
            continue;
        }
        fill_random_f32(src, dim, 42);

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < skip_reps; i++) memcpy(skip_buf, src, bytes);
        uint32_t c1 = read_mcycle();
        float store_ns = CYCLES_TO_NS((float)(c1 - c0) / (float)skip_reps);

        c0 = read_mcycle();
        for (int i = 0; i < skip_reps; i++) memcpy(dst, skip_buf, bytes);
        c1 = read_mcycle();
        float load_ns = CYCLES_TO_NS((float)(c1 - c0) / (float)skip_reps);

        printf("  %-8d  %10d  %10.0f  %10.0f\n", dim, bytes, store_ns, load_ns);

        heap_caps_free(src);
        heap_caps_free(skip_buf);
        heap_caps_free(dst);
    }

    bench_subsection("Full Token Pipeline Simulation (INT8, PSRAM streaming)");

    int enc_unrolls = 3;
    int dec_unrolls = 3;
    int bot_recurrence[] = {1, 2, 3, 4, 6};
    int n_rec = 5;

    int cp512 = (MODEL_DIM + 15) & ~15;
    int cp256 = (BOTTLENECK_DIM + 15) & ~15;
    int cp1024 = (MAMBA_EXPAND + 15) & ~15;

    int enc_in_sz = MAMBA_EXPAND * cp512;
    int enc_out_sz = MODEL_DIM * cp1024;
    int bot_qkv_sz = 3 * BOTTLENECK_DIM * cp256;
    int bot_aout_sz = BOTTLENECK_DIM * cp256;
    int bot_min_sz = 2 * BOTTLENECK_DIM * cp256;
    int bot_mout_sz = BOTTLENECK_DIM * ((2 * BOTTLENECK_DIM + 15) & ~15);
    int proj_dn_sz = BOTTLENECK_DIM * cp512;
    int proj_up_sz = MODEL_DIM * cp256;
    int head_sz = VOCAB_SIZE * cp512;

    int8_t *enc_in_w = (int8_t *)alloc_psram(enc_in_sz, 16);
    int8_t *enc_out_w = (int8_t *)alloc_psram(enc_out_sz, 16);
    int8_t *bot_qkv_w = (int8_t *)alloc_psram(bot_qkv_sz, 16);
    int8_t *bot_aout_w = (int8_t *)alloc_psram(bot_aout_sz, 16);
    int8_t *bot_min_w = (int8_t *)alloc_psram(bot_min_sz, 16);
    int8_t *bot_mout_w = (int8_t *)alloc_psram(bot_mout_sz, 16);
    int8_t *proj_dn_w = (int8_t *)alloc_psram(proj_dn_sz, 16);
    int8_t *proj_up_w = (int8_t *)alloc_psram(proj_up_sz, 16);
    int8_t *head_w = (int8_t *)alloc_psram(head_sz, 16);

    int8_t *act = (int8_t *)alloc_sram(cp1024, 16);
    int32_t *acc = (int32_t *)alloc_sram(MAMBA_EXPAND * 4, 16);

    int all_ok = enc_in_w && enc_out_w && bot_qkv_w && bot_aout_w &&
                 bot_min_w && bot_mout_w && proj_dn_w && proj_up_w &&
                 head_w && act && acc;

    if (all_ok) {
        fill_random_s8(enc_in_w, enc_in_sz, 1);
        fill_random_s8(enc_out_w, enc_out_sz, 2);
        fill_random_s8(bot_qkv_w, bot_qkv_sz, 3);
        fill_random_s8(bot_aout_w, bot_aout_sz, 4);
        fill_random_s8(bot_min_w, bot_min_sz, 5);
        fill_random_s8(bot_mout_w, bot_mout_sz, 6);
        fill_random_s8(proj_dn_w, proj_dn_sz, 7);
        fill_random_s8(proj_up_w, proj_up_sz, 8);
        fill_random_s8(head_w, head_sz, 9);
        fill_random_s8(act, MODEL_DIM, 10);

        printf("  Weight sizes (INT8):\n");
        printf("    Encoder (shared, per-unroll):  in_proj %dKB + gate %dKB + out_proj %dKB\n",
               enc_in_sz/1024, enc_in_sz/1024, enc_out_sz/1024);
        printf("    Bottleneck (shared, per-rec):  QKV %dKB + out %dKB + mamba %dKB + %dKB\n",
               bot_qkv_sz/1024, bot_aout_sz/1024, bot_min_sz/1024, bot_mout_sz/1024);
        printf("    Projections: down %dKB + up %dKB\n", proj_dn_sz/1024, proj_up_sz/1024);
        printf("    Output head: %dKB\n", head_sz/1024);

        int total_enc = enc_in_sz + enc_in_sz + enc_out_sz;
        int total_bot = bot_qkv_sz + bot_aout_sz + bot_min_sz + bot_mout_sz;
        int total_proj = proj_dn_sz + proj_up_sz;
        printf("    Total streamed per token (1x rec): %dKB\n",
               (total_enc + total_proj + total_bot + head_sz) / 1024);
        printf("\n");

        printf("  %-12s  %10s  %10s  %10s  %10s  %10s  %10s\n",
               "Recurrence", "Encoder", "Proj dn", "Bottleneck", "Proj up", "Decoder", "Total");
        printf("  %-12s  %10s  %10s  %10s  %10s  %10s  %10s\n",
               "(passes)", "us", "us", "us", "us", "us", "us (tok/s)");
        printf("  -------------------------------------------------------------------------------\n");

        int cp_2x256 = (2 * BOTTLENECK_DIM + 15) & ~15;

        for (int ri = 0; ri < n_rec; ri++) {
            int rec = bot_recurrence[ri];

            float enc_us = 0, proj_dn_us = 0, bot_us = 0, proj_up_us = 0, dec_us = 0;
            int sim_reps = 5;

            for (int rep = 0; rep < sim_reps; rep++) {
                int64_t t0;

                t0 = bench_start();
                for (int u = 0; u < enc_unrolls; u++) {
                    for (int r = 0; r < MAMBA_EXPAND; r++) {
                        pie_matmul_s8_xacc(enc_in_w + r * cp512, act, acc + r, 1, cp512);
                    }
                    for (int r = 0; r < MODEL_DIM; r++) {
                        pie_matmul_s8_xacc(enc_out_w + r * cp1024, act, acc + r, 1, cp1024);
                    }
                }
                enc_us += bench_stop_us(t0);

                t0 = bench_start();
                for (int r = 0; r < BOTTLENECK_DIM; r++) {
                    pie_matmul_s8_xacc(proj_dn_w + r * cp512, act, acc + r, 1, cp512);
                }
                proj_dn_us += bench_stop_us(t0);

                t0 = bench_start();
                for (int p = 0; p < rec; p++) {
                    for (int r = 0; r < 3 * BOTTLENECK_DIM; r++) {
                        pie_matmul_s8_xacc(bot_qkv_w + r * cp256, act, acc + r, 1, cp256);
                    }
                    for (int r = 0; r < BOTTLENECK_DIM; r++) {
                        pie_matmul_s8_xacc(bot_aout_w + r * cp256, act, acc + r, 1, cp256);
                    }
                    for (int r = 0; r < 2 * BOTTLENECK_DIM; r++) {
                        pie_matmul_s8_xacc(bot_min_w + r * cp256, act, acc + r, 1, cp256);
                    }
                    for (int r = 0; r < BOTTLENECK_DIM; r++) {
                        pie_matmul_s8_xacc(bot_mout_w + r * cp_2x256, act, acc + r, 1, cp_2x256);
                    }
                }
                bot_us += bench_stop_us(t0);

                t0 = bench_start();
                for (int r = 0; r < MODEL_DIM; r++) {
                    pie_matmul_s8_xacc(proj_up_w + r * cp256, act, acc + r, 1, cp256);
                }
                proj_up_us += bench_stop_us(t0);

                t0 = bench_start();
                for (int u = 0; u < dec_unrolls; u++) {
                    for (int r = 0; r < MAMBA_EXPAND; r++) {
                        pie_matmul_s8_xacc(enc_in_w + r * cp512, act, acc + r, 1, cp512);
                    }
                    for (int r = 0; r < MODEL_DIM; r++) {
                        pie_matmul_s8_xacc(enc_out_w + r * cp1024, act, acc + r, 1, cp1024);
                    }
                }
                dec_us += bench_stop_us(t0);
            }

            enc_us /= (float)sim_reps;
            proj_dn_us /= (float)sim_reps;
            bot_us /= (float)sim_reps;
            proj_up_us /= (float)sim_reps;
            dec_us /= (float)sim_reps;

            float total = enc_us + proj_dn_us + bot_us + proj_up_us + dec_us;
            float tok_s = 1e6f / total;

            char rec_label[16];
            snprintf(rec_label, sizeof(rec_label), "%dx", rec);
            printf("  %-12s  %8.0f  %8.0f  %10.0f  %8.0f  %8.0f  %8.0f (%.1f)\n",
                   rec_label, enc_us, proj_dn_us, bot_us, proj_up_us, dec_us, total, tok_s);
        }

        printf("\n  Note: excludes FP32 ops (norm, softmax, SSM, conv1d, gating).\n");
        printf("  See bench_fp32 for those timings. Add ~2-5ms for FP32 overhead.\n");
        printf("  Output head not included (runs once, ~0.1ms from SRAM).\n");
    } else {
        printf("  ALLOC FAILED -- not enough PSRAM or SRAM\n");
    }

    if (enc_in_w) heap_caps_free(enc_in_w);
    if (enc_out_w) heap_caps_free(enc_out_w);
    if (bot_qkv_w) heap_caps_free(bot_qkv_w);
    if (bot_aout_w) heap_caps_free(bot_aout_w);
    if (bot_min_w) heap_caps_free(bot_min_w);
    if (bot_mout_w) heap_caps_free(bot_mout_w);
    if (proj_dn_w) heap_caps_free(proj_dn_w);
    if (proj_up_w) heap_caps_free(proj_up_w);
    if (head_w) heap_caps_free(head_w);
    if (act) heap_caps_free(act);
    if (acc) heap_caps_free(acc);
}