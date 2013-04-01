#include "random_bignum_generator.h"
#include "bignum_conversions.h"

#include <stdlib.h>
#include <gmp.h>

/**
 * Generates the i'th random number from the seed, where "i" is the "index"
 * value passed as a parameter.
 * @param  index  "Index" of the random number.
 * @param  seed   Seed of the random number generator.
 * @param  bits   Bit precision requested.
 * @param  base   Base of the number returned in the string (2 until 62)
 * @param  number bignum to hold the generated random number.
 */
void generate_random_bignum(unsigned int index, unsigned int seed,
                            unsigned int bits, unsigned int base, bignum number)
{
    char* number_str = generate_random_bignum_str(index, seed, bits, base);
    string_to_bignum(number_str, number);
    free(number_str);
}

/**
 * Generates the i'th random number from the seed, where "i" is the "index"
 * value passed as a parameter. Remember to call free() on the returned string
 * once you don't need it anymore.
 * @param  index "Index" of the random number.
 * @param  seed  Seed of the random number generator.
 * @param  bits  Bit precision requested.
 * @param  base  Base of the number returned in the string (2 until 62)
 * @return       String representing the binary version of the number.
 */
char* generate_random_bignum_str(unsigned int index, unsigned int seed,
                                 unsigned int bits, unsigned int base)
{
    // random number generator initialization
    gmp_randstate_t random_state;
    gmp_randinit_default(random_state);
    // incorporated seed in generator
    gmp_randseed_ui(random_state, seed);

    // initialize test vector operands and result
    mpz_t number;
    mpz_init(number);

    // generate random number
    mpz_urandomb(number, random_state, bits);
    for (int i = 0; i < index; i++)
    {
        mpz_urandomb(number, random_state, bits);
    }

    // get binary string version
    char* str_number = mpz_get_str(NULL, base, number);
    pad_string_with_zeros(&str_number);

    // get memory back from operands and results
    mpz_clear(number);

    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);

    return str_number;
}
