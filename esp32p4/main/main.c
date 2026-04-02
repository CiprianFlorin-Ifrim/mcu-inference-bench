#include "bench.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"

static const char *TAG = "bench";

static void print_system_info(void)
{
    bench_separator("SYSTEM INFORMATION");

    esp_chip_info_t chip;
    esp_chip_info(&chip);

    printf("  Chip:       ESP32-P4 rev %d.%d\n", chip.revision / 100, chip.revision % 100);
    printf("  Cores:      %d\n", chip.cores);
    printf("  CPU freq:   %d MHz\n", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);

    printf("\n  Memory:\n");
    printf("  %-14s  %10s  %10s  %10s\n", "Region", "Total", "Free", "Max Block");
    printf("  -------------------------------------------------------\n");

    size_t t, f, m;
    t = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    f = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    m = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    printf("  %-14s  %8zuKB  %8zuKB  %8zuKB\n", "SRAM", t/1024, f/1024, m/1024);

    t = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    f = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    m = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    printf("  %-14s  %8zuKB  %8zuKB  %8zuKB\n", "PSRAM", t/1024, f/1024, m/1024);

    printf("\n  SRAM fragmentation probe:\n");
    size_t probes[] = {384*1024, 256*1024, 192*1024, 128*1024, 64*1024, 32*1024};
    for (int i = 0; i < 6; i++) {
        void *p = heap_caps_aligned_alloc(16, probes[i], MALLOC_CAP_INTERNAL);
        printf("    %6zuKB alloc: %s\n", probes[i]/1024, p ? "OK" : "FAILED");
        if (p) heap_caps_free(p);
    }

    printf("\n  ISA: rv32imafc_zicsr_zifencei_xesppie\n");
    printf("  PIE: 128-bit vector regs, XACC 40-bit accumulator\n");
    printf("  No Zbb (no hardware cpop/clz/ctz)\n");
}

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-P4 Hardware Benchmarks v2");
    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("\n\n");
    printf("################################################################\n");
    printf("#          ESP32-P4 Hardware Benchmark Suite v3                #\n");
    printf("#                                                              #\n");
    printf("#  v3: added FP32 ops (norm, softmax, SSM, gating, quant),    #\n");
    printf("#  architecture-matched matmuls, recurrence cache effects,     #\n");
    printf("#  skip connection overhead, full token pipeline simulation    #\n");
    printf("################################################################\n");

    print_system_info();

    bench_memory();
    bench_pie();
    bench_popcount();
    bench_matmul();
    bench_cache();
    bench_dma();
    bench_int4();
    bench_fp32();
    bench_model();

    bench_separator("ALL BENCHMARKS COMPLETE");

    printf("  Key numbers to extract:\n");
    printf("    - PSRAM seq read MB/s (beyond cache)\n");
    printf("    - PSRAM->SRAM memcpy MB/s at various sizes\n");
    printf("    - PIE actual MACs/cycle (via mcycle CSR)\n");
    printf("    - Cycles per esp.vmulas.s8.xacc instruction\n");
    printf("    - Popcount cycles/word (each strategy)\n");
    printf("    - INT4 unpack cycles/value (each strategy)\n");
    printf("    - XNOR vs INT8 PIE ratio from SRAM\n");
    printf("    - INT8 PSRAM: direct vs row-copy vs batch\n");
    printf("    - Copy/compute ratio for double-buffering\n");
    printf("    - Best INT4 pipeline vs INT8 from PSRAM\n");
    printf("    --- NEW in v3 ---\n");
    printf("    - LayerNorm/RMSNorm cycles at dim=256/512\n");
    printf("    - Softmax cycles at vocab=140/256\n");
    printf("    - SSM state update cycles (16x512, 16x256)\n");
    printf("    - Conv1d step cycles at expand=512/1024\n");
    printf("    - SiLU/SwiGLU gating cycles\n");
    printf("    - Quantize/dequantize overhead\n");
    printf("    - Architecture-matched matmul times (SRAM + PSRAM)\n");
    printf("    - Recurrence cache warmth effect\n");
    printf("    - Skip connection store/load overhead\n");
    printf("    - Full token time at 1x/2x/3x/4x/6x recurrence\n");

    size_t min_free = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    printf("\n  Min SRAM free during benchmarks: %zuKB\n", min_free / 1024);

    ESP_LOGI(TAG, "Done.");
}
