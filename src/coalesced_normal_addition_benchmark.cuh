#ifndef COALESCED_NORMAL_ADDITION_BENCHMARK_CUH
#define COALESCED_NORMAL_ADDITION_BENCHMARK_CUH

#include "bignum_type.h"
#include "coalesced_bignum_type.h"

__global__ void coalesced_normal_addition(coalesced_bignum* dev_coalesced_c,
                                          coalesced_bignum* dev_coalesced_a,
                                          coalesced_bignum* dev_coalesced_b);

void execute_coalesced_normal_addition_on_device(bignum* host_c, bignum* host_a,
                                                 bignum* host_b,
                                                 int threads_per_block,
                                                 int blocks_per_grid);

void check_coalesced_normal_addition_results(bignum* host_c, bignum* host_a,
                                             bignum* host_b);

#endif