idf_component_register(
    SRCS
        "main.c"
        "bench_memory.c"
        "bench_pie.c"
        "bench_popcount.c"
        "bench_matmul.c"
        "bench_cache.c"
        "bench_dma.c"
        "bench_int4.c"
        "bench_fp32.c"
        "bench_model.c"
        "pie_kernels.S"
    INCLUDE_DIRS "."
)
