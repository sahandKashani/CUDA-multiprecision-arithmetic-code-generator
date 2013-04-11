#ifndef MEMORY_LAYOUT_BENCHMARKS_CUH
#define MEMORY_LAYOUT_BENCHMARKS_CUH

#include <stdint.h>

__global__ void coalesced_normal_addition(uint32_t* c, uint32_t* a, uint32_t* b);

void coalesced_normal_memory_layout_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, uint32_t threads_per_block, uint32_t blocks_per_grid);

#endif
