#include "random_bignum_generator.h"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include <stdlib.h>
#include <gmp.h>

// current state of the random number generator
gmp_randstate_t random_state;

/**
 * Generates the next random number and stores it in the bignum given as a
 * parameter
 * @param number bignum to set
 */
void generate_random_bignum(bignum number)
{
    char* number_str = generate_random_bignum_str();
    string_to_bignum(number_str, number);
    free(number_str);
}

/**
 * Initializes the state of the random number generator. You must call this
 * function before using any of the other functions in this file.
 */
void start_random_number_generator()
{
    gmp_randinit_default(random_state);
    gmp_randseed_ui(random_state, SEED);
}

/**
 * Stops and resets the state of the random number generator. After a call to
 * this function, you can no longer reuse the number generator unless you call
 * start_random_number_generator() first.
 */
void stop_random_number_generator()
{
    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);
}

/**
 * Generates a string representation of the next random number. The number is
 * padded with 0s up until a length of TOTAL_BIT_LENGTH
 * @return string representing the next random number.
 */
char* generate_random_bignum_str()
{
    mpz_t number;
    mpz_init(number);

    // generate random number
    mpz_urandomb(number, random_state, BIT_RANGE);

    // get binary string version
    char* str_number = mpz_get_str(NULL, 2, number);
    pad_string_with_zeros(&str_number);

    mpz_clear(number);

    return str_number;
}
