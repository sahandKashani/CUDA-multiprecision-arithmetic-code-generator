#ifndef INTERLEAVED_ADDITION_BENCHMARK_CUH
#define INTERLEAVED_ADDITION_BENCHMARK_CUH

__global__ void interleaved_addition(bignum* dev_results,
                                     interleaved_bignum* dev_interleaved_operands);

void execute_interleaved_addition_on_device(bignum* host_c, bignum* host_a,
                                            bignum* host_b,
                                            uint32_t threads_per_block,
                                            uint32_t blocks_per_grid);

void check_interleaved_addition_results(bignum* host_c, bignum* host_a,
                                        bignum* host_b);

#endif
