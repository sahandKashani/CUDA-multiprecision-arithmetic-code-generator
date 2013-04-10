#ifndef MEMORY_LAYOUT_BENCHMARKS_CUH
#define MEMORY_LAYOUT_BENCHMARKS_CUH

#include "bignum_types.h"

__global__ void normal_addition(bignum* c, bignum* a, bignum* b);

__global__ void interleaved_addition(bignum* c, interleaved_bignum* ops);

__global__ void coalesced_normal_addition(coalesced_bignum* c,
                                          coalesced_bignum* a,
                                          coalesced_bignum* b);

__global__ void coalesced_interleaved_addition(coalesced_bignum* c,
                                               coalesced_interleaved_bignum* ops);

void normal_memory_layout_benchmark(bignum* host_c,
                                    bignum* host_a,
                                    bignum* host_b,
                                    uint32_t threads_per_block,
                                    uint32_t blocks_per_grid);


void interleaved_memory_layout_benchmark(bignum* host_c,
                                         bignum* host_a,
                                         bignum* host_b,
                                         uint32_t threads_per_block,
                                         uint32_t blocks_per_grid);

void coalesced_normal_memory_layout_benchmark(bignum* host_c,
                                              bignum* host_a,
                                              bignum* host_b,
                                              uint32_t threads_per_block,
                                              uint32_t blocks_per_grid);

void coalesced_interleaved_memory_layout_benchmark(bignum* host_c,
                                                   bignum* host_a,
                                                   bignum* host_b,
                                                   uint32_t threads_per_block,
                                                   uint32_t blocks_per_grid);

#endif
