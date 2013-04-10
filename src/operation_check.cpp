#include "operation_check.h"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include "constants.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <gmp.h>

void binary_operator_check(bignum* host_c, bignum* host_a, bignum* host_b,
                           void (*op)(mpz_t rop, const mpz_t op1, const mpz_t op2))
{
    bool results_correct = true;

    for (uint32_t i = 0; results_correct && i < TOTAL_NUMBER_OF_THREADS; i++)
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

        // GMP function which will calculate what our algorithm is supposed to
        // calculate
        op(gmp_bignum_c, gmp_bignum_a, gmp_bignum_b);

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
        printf("additions correct\n");
    }
    else
    {
        printf("additions incorrect\n");
    }
}

/**
 * Checks if host_c == host_a + host_b. host_c is a value calculated with our
 * gpu algorithm. This function will use GMP to calculate the value that host_c
 * should have after the addition of host_a and host_b. If any differences are
 * found with the values computed by our gpu algorithm, an error is reported. If
 * you have data in any other formats than bignum*, you will have to rearrange
 * them to meet this pattern for the check to work.
 * @param host_c Result of the addition we have computed with our gpu algorithm.
 * @param host_a First operands.
 * @param host_b Second operands.
 */
void addition_check(bignum* host_c, bignum* host_a, bignum* host_b)
{
    binary_operator_check(host_c, host_a, host_b, mpz_add);
}

/**
 * Checks if host_c == host_a - host_b. host_c is a value calculated with our
 * gpu algorithm. This function will use GMP to calculate the value that host_c
 * should have after the subtraction of host_b from host_a. If any differences
 * are found with the values computed by our gpu algorithm, an error is
 * reported. If you have data in any other formats than bignum*, you will have
 * to rearrange them to meet this pattern for the check to work.
 * @param host_c Result of the subtraction we have computed with our gpu
 *               algorithm.
 * @param host_a First operands.
 * @param host_b Second operands.
 */
void addition_check(bignum* host_c, bignum* host_a, bignum* host_b)
{
    binary_operator_check(host_c, host_a, host_b, mpz_sub);
}
