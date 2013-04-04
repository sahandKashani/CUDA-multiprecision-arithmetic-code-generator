#include "bignum_type.h"
#include "interleaved_bignum_type.h"
#include "random_bignum_generator.h"
#include "test_constants.h"
#include "normal_addition_benchmark.cuh"
#include "interleaved_addition_benchmark.cuh"

#include <stdio.h>
#include <stdlib.h>

void generate_operands(bignum* host_a, bignum* host_b);

int main(void)
{
    printf("Benchmarking PTX\n");

    // host operands (host_a, host_b) and results (host_c)
    bignum* host_a = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));
    bignum* host_b = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));
    bignum* host_c = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));

    // generate random numbers for the tests
    generate_operands(host_a, host_b);

    // once the operations are executed on the device, we need to extract the
    // results and put them in host_c. This is done by the code which calls the
    // kernels. They are the "execute_xxx_on_device" functions.

    execute_normal_addition_on_device(host_c, host_a, host_b);
    check_normal_addition_results(host_c, host_a, host_b);

    execute_interleaved_addition_on_device(host_c, host_a, host_b);
    check_interleaved_addition_results(host_c, host_a, host_b);

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
    start_random_number_generator();

    for (int i = 0; i < NUMBER_OF_TESTS; i++)
    {
        generate_random_bignum(host_a[i]);
        generate_random_bignum(host_b[i]);
    }

    stop_random_number_generator();
}
