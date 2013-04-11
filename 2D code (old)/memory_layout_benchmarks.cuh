#ifndef MEMORY_LAYOUT_BENCHMARKS_CUH
#define MEMORY_LAYOUT_BENCHMARKS_CUH

#include "bignum_types.h"

__global__ void normal_addition(bignum* c, bignum* a, bignum* b);

__global__ void interleaved_addition(bignum* c, interleaved_bignum* ops);

__global__ void coalesced_normal_addition(coalesced_bignum* c, coalesced_bignum* a, coalesced_bignum* b);

__global__ void coalesced_interleaved_addition(coalesced_bignum* c, coalesced_interleaved_bignum* ops);

__global__ void coalesced_normal_addition_with_local_memory(coalesced_bignum* c, coalesced_bignum* a, coalesced_bignum* b);

void normal_memory_layout_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid);


void interleaved_memory_layout_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid);

void coalesced_normal_memory_layout_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid);

void coalesced_interleaved_memory_layout_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid);

void coalesced_normal_memory_layout_with_local_memory_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid);

// void andrea_hardcoded_test(uint32_t threads_per_block, uint32_t blocks_per_grid);
// __global__ void andrea_hardcoded_kernel(coalesced_bignum* c);

// void andrea_hardcoded_local_test(uint32_t threads_per_block, uint32_t blocks_per_grid);
// __global__ void andrea_hardcoded_local_kernel(coalesced_bignum* c);

// void coalesced_normal_memory_layout_with_cudaMallocPitch(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid);

// __global__ void coalesced_normal_addition_with_cudaMallocPitch(coalesced_bignum* c, coalesced_bignum* a, coalesced_bignum* b, size_t pitch_c, size_t pitch_a, size_t pitch_b);

#endif
