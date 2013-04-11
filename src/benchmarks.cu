#include "bignum_types.h"
#include "random_bignum_generator.h"
#include "constants.h"
#include "operation_check.h"
#include "memory_layout_benchmarks.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void generate_operands(uint32_t* host_a, uint32_t* host_b);

int main(void)
{
    // host operands (host_a, host_b) and results (host_c)
    uint32_t* host_a = (uint32_t*) calloc(TOTAL_NUMBER_OF_THREADS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(TOTAL_NUMBER_OF_THREADS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(TOTAL_NUMBER_OF_THREADS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    // generate random numbers for the tests
    generate_operands(host_a, host_b);

    uint32_t blocks = 256;
    uint32_t threads = 256;

    coalesced_normal_memory_layout_benchmark(host_c, host_a, host_b, blocks, threads);
    addition_check(host_c, host_a, host_b);

    free(host_a);
    free(host_b);
    free(host_c);

    // for leak detection when using cuda-memcheck
    cudaDeviceReset();
}

/**
 * Generates random numbers and assigns them to the 2 bignum arrays passed as a
 * parameter.
 * @param host_a first array to populate
 * @param host_b second array to populate
 */
void generate_operands(uint32_t* host_a, uint32_t* host_b)
{
    printf("generating operands ... ");
    fflush(stdout);

    start_random_number_generator();

    for (uint32_t i = 0; i < TOTAL_NUMBER_OF_THREADS; i++)
    {
        generate_random_bignum(&host_a[IDX(i, 0)]);
        generate_random_bignum(&host_b[IDX(i, 0)]);
    }

    stop_random_number_generator();

    printf("done\n");
    fflush(stdout);
}
