#include "bench.h"

static void layernorm_f32(const float *x, float *out, const float *gamma,
                           const float *beta, int dim)
{
    float mean = 0;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;

    float var = 0;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)dim;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

static void rmsnorm_f32(const float *x, float *out, const float *gamma, int dim)
{
    float ss = 0;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)dim + 1e-5f);

    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * inv_rms * gamma[i];
    }
}

static void softmax_f32(float *x, int n)
{
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

static void silu_f32(const float *x, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void swiglu_f32(const float *gate, const float *x, float *out, int n)
{
    for (int i = 0; i < n; i++) {
        float silu = gate[i] / (1.0f + expf(-gate[i]));
        out[i] = silu * x[i];
    }
}

static void ssm_state_update_f32(float *state, const float *A_diag,
                                   const float *B_col, const float *C_row,
                                   const float *x_expand, float *y_out,
                                   int state_dim, int expand_dim)
{
    for (int e = 0; e < expand_dim; e++) {
        float y = 0;
        for (int s = 0; s < state_dim; s++) {
            int idx = s * expand_dim + e;
            state[idx] = A_diag[s] * state[idx] + B_col[s] * x_expand[e];
            y += C_row[s] * state[idx];
        }
        y_out[e] = y;
    }
}

static void conv1d_step_f32(float *ring_buf, int *ring_pos,
                              const float *kernel, const float *new_input,
                              float *output, int channels, int kernel_size)
{
    int pos = *ring_pos;
    for (int c = 0; c < channels; c++) {
        ring_buf[pos * channels + c] = new_input[c];
    }

    for (int c = 0; c < channels; c++) {
        float sum = 0;
        for (int k = 0; k < kernel_size; k++) {
            int idx = (pos - k + kernel_size) % kernel_size;
            sum += ring_buf[idx * channels + c] * kernel[k * channels + c];
        }
        output[c] = sum;
    }

    *ring_pos = (pos + 1) % kernel_size;
}

static void quantize_f32_to_s8(const float *in, int8_t *out, float *scale_out, int n)
{
    float absmax = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(in[i]);
        if (a > absmax) absmax = a;
    }
    float scale = absmax / 127.0f;
    float inv_scale = (absmax > 0) ? 127.0f / absmax : 0;
    *scale_out = scale;
    for (int i = 0; i < n; i++) {
        float v = in[i] * inv_scale;
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        out[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }
}

static void dequantize_s8_to_f32(const int8_t *in, float *out, float scale, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = (float)in[i] * scale;
    }
}

void bench_fp32(void)
{
    bench_separator("FP32 OPERATIONS (layer norm, softmax, SSM, gating)");

    volatile float fp_sink;

    bench_subsection("LayerNorm and RMSNorm (cycle counter)");

    int dims[] = {128, 256, 384, 512, 1024};
    int n_dims = 5;
    int reps = 10000;

    printf("  %-8s  %12s  %12s  %12s  %12s\n",
           "Dim", "LayerNorm", "RMSNorm", "LN ns", "RMS ns");
    printf("  %-8s  %12s  %12s  %12s  %12s\n",
           "", "cycles", "cycles", "", "");
    printf("  -----------------------------------------------------------------\n");

    for (int d = 0; d < n_dims; d++) {
        int dim = dims[d];
        float *x = (float *)alloc_sram(dim * 4, 16);
        float *out = (float *)alloc_sram(dim * 4, 16);
        float *gamma = (float *)alloc_sram(dim * 4, 16);
        float *beta = (float *)alloc_sram(dim * 4, 16);
        if (!x || !out || !gamma || !beta) {
            if (x) heap_caps_free(x);
            if (out) heap_caps_free(out);
            if (gamma) heap_caps_free(gamma);
            if (beta) heap_caps_free(beta);
            continue;
        }
        fill_random_f32(x, dim, 42);
        fill_random_f32(gamma, dim, 99);
        fill_random_f32(beta, dim, 77);

        for (int w = 0; w < 100; w++) {
            layernorm_f32(x, out, gamma, beta, dim);
            rmsnorm_f32(x, out, gamma, dim);
        }

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < reps; i++) layernorm_f32(x, out, gamma, beta, dim);
        uint32_t c1 = read_mcycle();
        float ln_cyc = (float)(c1 - c0) / (float)reps;

        c0 = read_mcycle();
        for (int i = 0; i < reps; i++) rmsnorm_f32(x, out, gamma, dim);
        c1 = read_mcycle();
        float rms_cyc = (float)(c1 - c0) / (float)reps;

        fp_sink = out[0];

        printf("  %-8d  %10.0f  %10.0f  %10.0f  %10.0f\n",
               dim, ln_cyc, rms_cyc, CYCLES_TO_NS(ln_cyc), CYCLES_TO_NS(rms_cyc));

        heap_caps_free(x);
        heap_caps_free(out);
        heap_caps_free(gamma);
        heap_caps_free(beta);
    }

    bench_subsection("Softmax (cycle counter)");

    int sm_sizes[] = {140, 256, 512, 1024, 4096};
    int n_sm = 5;

    printf("  %-8s  %12s  %12s\n", "Size", "Cycles", "ns");
    printf("  ------------------------------------\n");

    for (int s = 0; s < n_sm; s++) {
        int n = sm_sizes[s];
        float *x = (float *)alloc_sram(n * 4, 16);
        if (!x) continue;
        fill_random_f32(x, n, 42);

        for (int w = 0; w < 100; w++) {
            fill_random_f32(x, n, 42);
            softmax_f32(x, n);
        }

        fill_random_f32(x, n, 42);
        uint32_t c0 = read_mcycle();
        for (int i = 0; i < reps; i++) {
            fill_random_f32(x, n, 42 + i);
            softmax_f32(x, n);
        }
        uint32_t c1 = read_mcycle();
        float cyc = (float)(c1 - c0) / (float)reps;

        fp_sink = x[0];
        printf("  %-8d  %10.0f  %10.0f\n", n, cyc, CYCLES_TO_NS(cyc));
        heap_caps_free(x);
    }

    bench_subsection("SiLU and SwiGLU gating (cycle counter)");

    printf("  %-8s  %12s  %12s  %12s  %12s\n",
           "Dim", "SiLU cyc", "SwiGLU cyc", "SiLU ns", "SwiGLU ns");
    printf("  -----------------------------------------------------------------\n");

    for (int d = 0; d < n_dims; d++) {
        int dim = dims[d];
        float *x = (float *)alloc_sram(dim * 4, 16);
        float *gate = (float *)alloc_sram(dim * 4, 16);
        float *out = (float *)alloc_sram(dim * 4, 16);
        if (!x || !gate || !out) {
            if (x) heap_caps_free(x);
            if (gate) heap_caps_free(gate);
            if (out) heap_caps_free(out);
            continue;
        }
        fill_random_f32(x, dim, 42);
        fill_random_f32(gate, dim, 99);

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < reps; i++) silu_f32(x, out, dim);
        uint32_t c1 = read_mcycle();
        float silu_cyc = (float)(c1 - c0) / (float)reps;

        c0 = read_mcycle();
        for (int i = 0; i < reps; i++) swiglu_f32(gate, x, out, dim);
        c1 = read_mcycle();
        float swi_cyc = (float)(c1 - c0) / (float)reps;

        fp_sink = out[0];
        printf("  %-8d  %10.0f  %10.0f  %10.0f  %10.0f\n",
               dim, silu_cyc, swi_cyc, CYCLES_TO_NS(silu_cyc), CYCLES_TO_NS(swi_cyc));

        heap_caps_free(x);
        heap_caps_free(gate);
        heap_caps_free(out);
    }

    bench_subsection("Mamba2 SSM State Update (per-token, cycle counter)");

    int ssm_configs[][2] = {
        {16, 512},
        {16, 1024},
        {64, 512},
        {16, 256},
    };
    int n_ssm = 4;
    int ssm_reps = 5000;

    printf("  %-16s  %12s  %12s  %10s\n",
           "State x Expand", "Cycles", "ns", "KB state");
    printf("  --------------------------------------------------------\n");

    for (int s = 0; s < n_ssm; s++) {
        int sd = ssm_configs[s][0];
        int ed = ssm_configs[s][1];
        int state_sz = sd * ed;

        float *state = (float *)alloc_sram(state_sz * 4, 16);
        float *A = (float *)alloc_sram(sd * 4, 16);
        float *B = (float *)alloc_sram(sd * 4, 16);
        float *C = (float *)alloc_sram(sd * 4, 16);
        float *x_in = (float *)alloc_sram(ed * 4, 16);
        float *y_out = (float *)alloc_sram(ed * 4, 16);

        if (!state || !A || !B || !C || !x_in || !y_out) {
            if (state) heap_caps_free(state);
            if (A) heap_caps_free(A);
            if (B) heap_caps_free(B);
            if (C) heap_caps_free(C);
            if (x_in) heap_caps_free(x_in);
            if (y_out) heap_caps_free(y_out);
            continue;
        }

        fill_random_f32(state, state_sz, 42);
        fill_random_f32(A, sd, 10);
        fill_random_f32(B, sd, 20);
        fill_random_f32(C, sd, 30);
        fill_random_f32(x_in, ed, 40);

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < ssm_reps; i++) {
            ssm_state_update_f32(state, A, B, C, x_in, y_out, sd, ed);
        }
        uint32_t c1 = read_mcycle();
        float cyc = (float)(c1 - c0) / (float)ssm_reps;

        fp_sink = y_out[0];

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", sd, ed);
        printf("  %-16s  %10.0f  %10.0f  %10.1f\n",
               label, cyc, CYCLES_TO_NS(cyc), (float)(state_sz * 4) / 1024.0f);

        heap_caps_free(state);
        heap_caps_free(A);
        heap_caps_free(B);
        heap_caps_free(C);
        heap_caps_free(x_in);
        heap_caps_free(y_out);
    }

    bench_subsection("Conv1D Step (Mamba causal conv, per-token, cycle counter)");

    int conv_channels[] = {256, 512, 1024};
    int n_conv = 3;
    int kernel_size = 4;
    int conv_reps = 10000;

    printf("  %-10s  %10s  %12s  %12s\n",
           "Channels", "Kernel", "Cycles", "ns");
    printf("  --------------------------------------------------\n");

    for (int c = 0; c < n_conv; c++) {
        int ch = conv_channels[c];

        float *ring = (float *)alloc_sram(kernel_size * ch * 4, 16);
        float *kern = (float *)alloc_sram(kernel_size * ch * 4, 16);
        float *inp = (float *)alloc_sram(ch * 4, 16);
        float *outp = (float *)alloc_sram(ch * 4, 16);

        if (!ring || !kern || !inp || !outp) {
            if (ring) heap_caps_free(ring);
            if (kern) heap_caps_free(kern);
            if (inp) heap_caps_free(inp);
            if (outp) heap_caps_free(outp);
            continue;
        }

        memset(ring, 0, kernel_size * ch * 4);
        fill_random_f32(kern, kernel_size * ch, 42);
        fill_random_f32(inp, ch, 99);
        int ring_pos = 0;

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < conv_reps; i++) {
            conv1d_step_f32(ring, &ring_pos, kern, inp, outp, ch, kernel_size);
        }
        uint32_t c1 = read_mcycle();
        float cyc = (float)(c1 - c0) / (float)conv_reps;

        fp_sink = outp[0];
        printf("  %-10d  %10d  %10.0f  %10.0f\n",
               ch, kernel_size, cyc, CYCLES_TO_NS(cyc));

        heap_caps_free(ring);
        heap_caps_free(kern);
        heap_caps_free(inp);
        heap_caps_free(outp);
    }

    bench_subsection("INT8 Quantize / Dequantize (cycle counter)");

    printf("  %-8s  %12s  %12s  %12s  %12s\n",
           "Dim", "Quant cyc", "Dequant cyc", "Quant ns", "Dequant ns");
    printf("  -----------------------------------------------------------------\n");

    for (int d = 0; d < n_dims; d++) {
        int dim = dims[d];
        float *f_buf = (float *)alloc_sram(dim * 4, 16);
        int8_t *i_buf = (int8_t *)alloc_sram(dim, 16);
        float *f_out = (float *)alloc_sram(dim * 4, 16);
        float scale;

        if (!f_buf || !i_buf || !f_out) {
            if (f_buf) heap_caps_free(f_buf);
            if (i_buf) heap_caps_free(i_buf);
            if (f_out) heap_caps_free(f_out);
            continue;
        }
        fill_random_f32(f_buf, dim, 42);

        uint32_t c0 = read_mcycle();
        for (int i = 0; i < reps; i++) quantize_f32_to_s8(f_buf, i_buf, &scale, dim);
        uint32_t c1 = read_mcycle();
        float q_cyc = (float)(c1 - c0) / (float)reps;

        c0 = read_mcycle();
        for (int i = 0; i < reps; i++) dequantize_s8_to_f32(i_buf, f_out, scale, dim);
        c1 = read_mcycle();
        float dq_cyc = (float)(c1 - c0) / (float)reps;

        fp_sink = f_out[0];
        printf("  %-8d  %10.0f  %10.0f  %10.0f  %10.0f\n",
               dim, q_cyc, dq_cyc, CYCLES_TO_NS(q_cyc), CYCLES_TO_NS(dq_cyc));

        heap_caps_free(f_buf);
        heap_caps_free(i_buf);
        heap_caps_free(f_out);
    }

    (void)fp_sink;
}