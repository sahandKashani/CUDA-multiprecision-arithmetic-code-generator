#include "operation_check.h"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include "constants.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <gmp.h>

void binary_operator_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*op)(mpz_t rop, const mpz_t op1, const mpz_t op2), char op_character, const char* operation_name);
char* mpz_t_to_binary_2s_complement_string(mpz_t number);
char* twos_complement_binary_string_of_negative_number(mpz_t negative_number);

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// OPERATOR CHECKS //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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
void addition_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    binary_operator_check(host_c, host_a, host_b, mpz_add, '+', "addition");
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
void subtraction_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    binary_operator_check(host_c, host_a, host_b, mpz_sub, '-', "subtraction");
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// GENERIC CHECKS ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void binary_operator_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*op)(mpz_t rop, const mpz_t op1, const mpz_t op2), char op_character, const char* operation_name)
{
    assert(host_c != NULL);
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(op != NULL);
    assert(operation_name != NULL);

    printf("checking \"%s\" with gmp ... ", operation_name);
    fflush(stdout);

    bool results_correct = true;
    for (uint32_t i = 0; results_correct && i < NUMBER_OF_BIGNUMS; i++)
    {
        mpz_t gmp_a;
        mpz_t gmp_b;
        mpz_t gmp_c;

        mpz_init(gmp_a);
        mpz_init(gmp_b);
        mpz_init(gmp_c);

        bignum_to_mpz_t(&host_a[IDX(i, 0)], gmp_a);
        bignum_to_mpz_t(&host_b[IDX(i, 0)], gmp_b);
        bignum_to_mpz_t(&host_c[IDX(i, 0)], gmp_c);

        // GMP function which will calculate what our algorithm is supposed to
        // calculate
        op(gmp_c, gmp_a, gmp_b);

        char* a_str = bignum_to_binary_string(&host_a[IDX(i, 0)]);
        char* b_str = bignum_to_binary_string(&host_b[IDX(i, 0)]);
        char* c_str = bignum_to_binary_string(&host_c[IDX(i, 0)]);

        // get binary string result in 2's complement (same format as what the
        // gpu is calculating)
        char* gmp_c_str = mpz_t_to_binary_2s_complement_string(gmp_c);

        if (strcmp(gmp_c_str, c_str) != 0)
        {
            printf("incorrect calculation at iteration %d\n", i);
            printf("gpu algorithm:   %s\n               %c %s\n               = %s\n", a_str, op_character, b_str, c_str);
            printf("\n");
            printf("gmp algorithm:   %s\n               %c %s\n               = %s\n", a_str, op_character, b_str, gmp_c_str);

            fflush(stdout);
            results_correct = false;
        }

        free(a_str);
        free(b_str);
        free(c_str);

        free(gmp_c_str);

        mpz_clear(gmp_a);
        mpz_clear(gmp_b);
        mpz_clear(gmp_c);
    }

    printf("done => ");
    results_correct ? printf("correct\n") : printf("errors\n");
    fflush(stdout);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// TWOS COMPLEMENT TOOLS ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Returns a binary string of length TOTAL_BIT_LENGTH containing the 2's
 * complement representation of the number given as a parameter. If the number
 * is positive, the binary string is just padded with zeros. If the number is
 * negative, then we use the well-known formula -B = \bar{B} + 1 to get the 2's
 * complement representation of a negative number where abs(number) = B.
 * @param  number number to be represented in 2's complement.
 * @return        binary 2's complement string representation of number.
 */
char* mpz_t_to_binary_2s_complement_string(mpz_t number)
{
    // if number >= 0
    if (mpz_cmp_ui(number, 0) >= 0)
    {
        // get binary string representation (does not contain any symbol in
        // front of the string, since it is a positive number)
        return mpz_t_to_binary_string(number);
    }
    else
    {
        return twos_complement_binary_string_of_negative_number(number);
    }
}

/**
 * This function takes a negative number as a parameter and returns a binary 2's
 * complement string representation of the negative number. The length of the
 * string which is returned is TOTAL_BIT_LENGTH.
 * @param  negative_number Gmp negative number to represent as a binary string
 *                         in 2's complement notation.
 * @return                 Binary string representation of negative_number in
 *                         2's complement.
 */
char* twos_complement_binary_string_of_negative_number(mpz_t negative_number)
{
    // number must be negative
    assert(mpz_cmp_ui(negative_number, 0) < 0);

    // get absolute value of the negative number
    mpz_t abs_number;
    mpz_init(abs_number);
    mpz_abs(abs_number, negative_number);

    // get binary string representation of the absolute value.
    char* abs_number_str = mpz_t_to_binary_string(abs_number);

    // Then, get the twos complement representation of the binary string using
    // -B = \bar{B} + 1

    // find the index of the right-most bit set to 1
    uint32_t right_most_1_index = 0;
    bool right_most_1_not_found = true;
    for (uint32_t i = 0; right_most_1_not_found && i < TOTAL_BIT_LENGTH; i++)
    {
        if (abs_number_str[TOTAL_BIT_LENGTH - i - 1] == '1')
        {
            right_most_1_index = TOTAL_BIT_LENGTH - i - 1;
            right_most_1_not_found = false;
        }
    }

    // invert all bits to the left of the right-most bit set to 1
    for (uint32_t i = 0; i < right_most_1_index; i++)
    {
        char bit = abs_number_str[i];
        abs_number_str[i] = (bit == '1' ? '0' : '1');
    }

    mpz_clear(abs_number);

    return abs_number_str;
}
