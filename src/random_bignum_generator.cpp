#include "random_bignum_generator.h"
#include "bignum_conversions.h"

#include <stdlib.h>
#include <gmp.h>

gmp_randstate_t random_state;

void generate_random_bignum(bignum number)
{
    char* number_str = generate_random_bignum_str();
    string_to_bignum(number_str, number);
    free(number_str);
}

void start_random_number_generator()
{
    gmp_randinit_default(random_state);
    gmp_randseed_ui(random_state, SEED);
}

void stop_random_number_generator()
{
    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);
}

char* generate_random_bignum_str()
{
    mpz_t number;
    mpz_init(number);

    // generate random number
    mpz_urandomb(number, random_state, RANDOM_NUMBER_BIT_RANGE);

    // get binary string version
    char* str_number = mpz_get_str(NULL, 2, number);
    pad_string_with_zeros(&str_number);

    mpz_clear(number);

    return str_number;
}
