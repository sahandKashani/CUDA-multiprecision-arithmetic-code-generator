#ifndef COALESCED_ADDITION_BENCHMARK_CUH
#define COALESCED_ADDITION_BENCHMARK_CUH

#include "bignum_type.h"

__global__ void coalesced_addition(bignum* dev_c, bignum* dev_a, bignum* dev_b);

void execute_coalesced_addition_on_device(bignum* host_c, bignum* host_a,
                                          bignum* host_b);

void check_coalesced_addition_results(bignum* host_c, bignum* host_a,
                                      bignum* host_b);

#endif
