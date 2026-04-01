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
    printf("#          ESP32-P4 Hardware Benchmark Suite v2                #\n");
    printf("#                                                              #\n");
    printf("#  Fixes from v1: cycle counter for sub-us ops, fixed allocs,  #\n");
    printf("#  added INT4 unpack strategies, PSRAM burst patterns          #\n");
    printf("################################################################\n");

    print_system_info();

    bench_memory();
    bench_pie();
    bench_popcount();
    bench_matmul();
    bench_cache();
    bench_dma();
    bench_int4();

    bench_separator("ALL BENCHMARKS COMPLETE");

    size_t min_free = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    printf("\n  Min SRAM free during benchmarks: %zuKB\n", min_free / 1024);

    ESP_LOGI(TAG, "Done.");
}
