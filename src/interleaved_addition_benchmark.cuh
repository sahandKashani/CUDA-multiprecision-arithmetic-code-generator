#ifndef INTERLEAVED_ADDITION_BENCHMARK_CUH
#define INTERLEAVED_ADDITION_BENCHMARK_CUH

#include "bignum_type.h"
#include "interleaved_bignum_type.h"

__global__ void interleaved_addition(bignum* dev_results,
                                     interleaved_bignum* dev_interleaved_operands);

void execute_interleaved_addition_on_device(bignum* host_c, bignum* host_a,
                                            bignum* host_b);

#endif
