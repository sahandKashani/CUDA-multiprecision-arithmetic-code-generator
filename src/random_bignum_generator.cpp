#include "random_bignum_generator.h"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gmp.h>

bool precise_enough(mpz_t number, uint32_t precision);
bool bignum_less_than_bignum(uint32_t* smaller, uint32_t* bigger);
char* generate_exact_precision_bignum_string(uint32_t precision);
char* generate_bignum_string_less_than_bignum(uint32_t* bigger);

// current state of the random number generator
gmp_randstate_t random_state;

/**
 * Initializes the state of the random number generator. You must call this
 * function before using any of the other functions in this file.
 */
void start_random_number_generator()
{
    gmp_randinit_default(random_state);
    gmp_randseed_ui(random_state, SEED);

    printf("Started random number generator\n");
    fflush(stdout);
}

/**
 * Stops and resets the state of the random number generator. After a call to
 * this function, you can no longer reuse the number generator unless you call
 * start_random_number_generator() first. You can only use this function if a
 * prior call to start_random_number_generator() has been made, or else this
 * function will fail
 */
void stop_random_number_generator()
{
    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);

    printf("Stopped random number generator\n");
    fflush(stdout);
}

/**
 * Decides if a number has enough precision in bits.
 * @param  number    number to be tested.
 * @param  precision precision wanted in bits.
 * @return           true if the number has the given precision, or false
 *                   otherwise.
 */
bool precise_enough(mpz_t number, uint32_t precision)
{
    assert(precision > BITS_PER_WORD);

    // get binary string version
    char* str_number = mpz_t_to_exact_precision_binary_string(number);

    uint32_t str_number_length = strlen(str_number);
    free(str_number);

    return str_number_length == precision;
}

/**
 * Generates a binary string representation of a bignum with bit precision given
 * by the precision parameter. The binary string returned is of length
 * TOTAL_BIT_LENGTH.
 * @param  precision precision wanted in bits.
 * @return           binary string representation of the generated bignum.
 */
char* generate_exact_precision_bignum_string(uint32_t precision)
{
    assert(precision > BITS_PER_WORD);

    mpz_t number;
    mpz_init(number);

    do
    {
        // generate random number of at most precision bits
        mpz_urandomb(number, random_state, precision);
    }
    while (!precise_enough(number, precision));

    // get binary string version of number
    char* str_number = mpz_t_to_binary_string(number);

    mpz_clear(number);

    assert(strlen(str_number) == TOTAL_BIT_LENGTH);
    return str_number;
}

/**
 * Generates a bignum with bit precision given by the precision parameter and
 * stores the result in "number".
 * @param number    bignum in which the generated bignum is to be set.
 * @param precision precision wanted in bits.
 */
void generate_exact_precision_bignum(uint32_t* number, uint32_t precision)
{
    assert(number != NULL);
    assert(precision > BITS_PER_WORD);

    char* number_str = generate_exact_precision_bignum_string(precision);
    binary_string_to_bignum(number_str, number);
    free(number_str);
}

/**
 * Tests if a bignum is less than another bignum.
 * @param  smaller the supposedly smaller bignum.
 * @param  bigger  the supposedly bigger bignum.
 * @return         true if smaller < bigger, and false otherwise.
 */
bool bignum_less_than_bignum(uint32_t* smaller, uint32_t* bigger)
{
    assert(smaller != NULL);
    assert(bigger != NULL);

    mpz_t smaller_gmp;
    mpz_t bigger_gmp;

    mpz_init(smaller_gmp);
    mpz_init(bigger_gmp);

    bignum_to_mpz_t(smaller, smaller_gmp);
    bignum_to_mpz_t(bigger, bigger_gmp);

    // return true if smaller is strictly less than bigger
    bool is_smaller = mpz_cmp(smaller_gmp, bigger_gmp) < 0;

    mpz_clear(smaller_gmp);
    mpz_clear(bigger_gmp);

    return is_smaller;
}

/**
 * Generates a binary string representation of a bignum which is less than the
 * "bigger" bignum given as a parameter. The returned string is of length
 * TOTAL_BIT_LENGTH.
 * @param  bigger bignum which represents the upper bound on the generated
 *                binary bignum string.
 * @return        binary string representation of a bignum of value at most
 *                bigger - 1.
 */
char* generate_bignum_string_less_than_bignum(uint32_t* bigger)
{
    uint32_t precision = get_bignum_precision(bigger);

    mpz_t bigger_gmp;
    mpz_t smaller_gmp;

    mpz_init(bigger_gmp);
    mpz_init(smaller_gmp);

    bignum_to_mpz_t(bigger, bigger_gmp);

    do
    {
        // generate random number of at most precision bits
        mpz_urandomb(smaller_gmp, random_state, precision);
    }
    while (mpz_cmp(smaller_gmp, bigger_gmp) >= 0);

    char* smaller_str = mpz_t_to_binary_string(smaller_gmp);

    assert(strlen(smaller_str) == TOTAL_BIT_LENGTH);

    mpz_clear(bigger_gmp);
    mpz_clear(smaller_gmp);

    return smaller_str;
}

/**
 * Generates a bignum which is less than the "bigger" bignum given as a
 * parameter. The generated bignum is stored in the "number" bignum.
 * @param bigger bignum which represents the upper bound on the generated
 *               bignum.
 * @param number bignum in which the generated bignum is to be set.
 */
void generate_bignum_less_than_bignum(uint32_t* bigger, uint32_t* number)
{
    assert(bigger != NULL);
    assert(number != NULL);

    char* number_str = generate_bignum_string_less_than_bignum(bigger);
    binary_string_to_bignum(number_str, number);
    free(number_str);
}
