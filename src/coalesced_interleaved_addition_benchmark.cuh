#ifndef COALESCED_INTERLEAVED_ADDITION_BENCHMARK_CUH
#define COALESCED_INTERLEAVED_ADDITION_BENCHMARK_CUH

#include "bignum_type.h"
#include "coalesced_interleaved_bignum_type.h"
#include "coalesced_bignum_type.h"

__global__ void coalesced_interleaved_addition(
    coalesced_bignum* dev_coalesced_results,
    coalesced_interleaved_bignum* dev_coalesced_interleaved_operands);

void execute_coalesced_interleaved_addition_on_device(bignum* host_c,
                                                      bignum* host_a,
                                                      bignum* host_b,
                                                      uint32_t threads_per_block,
                                                      uint32_t blocks_per_grid);

void check_coalesced_interleaved_addition_results(bignum* host_c,
                                                  bignum* host_a,
                                                  bignum* host_b);

#endif
