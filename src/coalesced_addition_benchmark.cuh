#ifndef COALESCED_ADDITION_BENCHMARK_CUH
#define COALESCED_ADDITION_BENCHMARK_CUH

#include "bignum_type.h"
#include "coalesced_bignum_type.h"
#include "coalesced_bignum_result_type.h"

__global__ void coalesced_addition(coalesced_bignum_result* dev_results,
                                   coalesced_bignum* dev_coalesced_operands);

void execute_coalesced_addition_on_device(bignum* host_c, bignum* host_a,
                                          bignum* host_b);

void check_coalesced_addition_results(bignum* host_c, bignum* host_a,
                                      bignum* host_b);

#endif
