#include "bignum_type.h"
#include "interleaved_bignum_type.h"
#include "random_bignum_generator.h"
#include "bignum_conversions.h"
#include "test_constants.h"
#include "normal_addition_benchmark.cuh"
#include "interleaved_addition_benchmark.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

void generate_operands(bignum* host_a, bignum* host_b);
void check_results(bignum* host_c, bignum* host_a, bignum* host_b);

int main(void)
{
    printf("Testing PTX\n");

    // host operands (host_a, host_b) and results (host_c)
    bignum* host_a = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));
    bignum* host_b = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));
    bignum* host_c = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));

    // generate random numbers for the tests
    generate_operands(host_a, host_b);

    execute_normal_addition_on_device(host_c, host_a, host_b);
    // execute_coalescing_addition_on_device(host_c, host_a, host_b);

    check_results(host_c, host_a, host_b);

    free(host_a);
    free(host_b);
    free(host_c);
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// General //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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

/**
 * Checks if host_a op host_b == host_c, where host_c is to be tested against
 * values computed by gmp. If you have data in any other formats than these, you
 * will have to "rearrange" them to meet this pattern for the check to work.
 * @param host_c Values we have computed with our algorithms.
 * @param host_a First operands.
 * @param host_b Second operands.
 */
void check_results(bignum* host_c, bignum* host_a, bignum* host_b)
{
    bool results_correct = true;

    for (int i = 0; results_correct && i < NUMBER_OF_TESTS; i++)
    {
        char* bignum_a_str = bignum_to_string(host_a[i]);
        char* bignum_b_str = bignum_to_string(host_b[i]);
        char* bignum_c_str = bignum_to_string(host_c[i]);

        mpz_t gmp_bignum_a;
        mpz_t gmp_bignum_b;
        mpz_t gmp_bignum_c;

        mpz_init_set_str(gmp_bignum_a, bignum_a_str, 2);
        mpz_init_set_str(gmp_bignum_b, bignum_b_str, 2);
        mpz_init(gmp_bignum_c);

        // CHOOSE GMP FUNCTION TO EXECUTE HERE
        mpz_add(gmp_bignum_c, gmp_bignum_a, gmp_bignum_b);

        // get binary string result
        char* gmp_bignum_c_str = mpz_get_str(NULL, 2, gmp_bignum_c);
        pad_string_with_zeros(&gmp_bignum_c_str);

        if (strcmp(gmp_bignum_c_str, bignum_c_str) != 0)
        {
            printf("incorrect calculation at iteration %d\n", i);
            results_correct = false;
            printf("own\n%s +\n%s =\n%s\n", bignum_a_str, bignum_b_str,
                   bignum_c_str);
            printf("gmp\n%s +\n%s =\n%s\n", bignum_a_str, bignum_b_str,
                   gmp_bignum_c_str);
        }

        free(bignum_a_str);
        free(bignum_b_str);
        free(bignum_c_str);
        free(gmp_bignum_c_str);

        mpz_clear(gmp_bignum_a);
        mpz_clear(gmp_bignum_b);
        mpz_clear(gmp_bignum_c);
    }

    if (results_correct)
    {
        printf("all correct\n");
    }
    else
    {
        printf("something wrong\n");
    }
}
