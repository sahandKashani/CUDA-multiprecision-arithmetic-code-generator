#ifndef NORMAL_ADDITION_BENCHMARK_CUH
#define NORMAL_ADDITION_BENCHMARK_CUH

#include "bignum_type.h"

__global__ void normal_addition(bignum* dev_c, bignum* dev_a, bignum* dev_b);

void execute_normal_addition_on_device(bignum* host_c, bignum* host_a,
                                       bignum* host_b);

#endif
