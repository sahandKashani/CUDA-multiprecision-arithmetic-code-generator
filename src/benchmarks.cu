#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "bignum_types.h"
#include "random_bignum_generator.h"

void generate_operands(bignum* host_a, bignum* host_b);

int main(void)
{
    printf("Benchmarking PTX\n");
    fflush(stdout);

    // host operands (host_a, host_b) and results (host_c)
    bignum* host_a = (bignum*) calloc(TOTAL_NUMBER_OF_THREADS, sizeof(bignum));
    bignum* host_b = (bignum*) calloc(TOTAL_NUMBER_OF_THREADS, sizeof(bignum));
    bignum* host_c = (bignum*) calloc(TOTAL_NUMBER_OF_THREADS, sizeof(bignum));

    // generate random numbers for the tests
    generate_operands(host_a, host_b);

    uint32_t blocks = 256;
    uint32_t threads = 256;

    // execute_normal_addition_on_device(host_c, host_a, host_b, blocks, threads);
    // addition_check(host_c, host_a, host_b);

    execute_coalesced_normal_addition_on_device(host_c, host_a, host_b, blocks, threads);
    addition_check(host_c, host_a, host_b);

    // execute_interleaved_addition_on_device(host_c, host_a, host_b, blocks, threads);
    // addition_check(host_c, host_a, host_b);

    // execute_coalesced_interleaved_addition_on_device(host_c, host_a, host_b, blocks, threads);
    // addition_check(host_c, host_a, host_b);

    free(host_a);
    free(host_b);
    free(host_c);
}

/**
 * Generates random numbers and assigns them to the 2 bignum arrays passed as a
 * parameter.
 * @param host_a first array to populate
 * @param host_b second array to populate
 */
void generate_operands(bignum* host_a, bignum* host_b)
{
    printf("generating operands ... ");
    fflush(stdout);

    start_random_number_generator();

    for (uint32_t i = 0; i < TOTAL_NUMBER_OF_THREADS; i++)
    {
        generate_random_bignum(host_a[i]);
        generate_random_bignum(host_b[i]);
    }

    stop_random_number_generator();

    printf("done\n");
    fflush(stdout);
}
