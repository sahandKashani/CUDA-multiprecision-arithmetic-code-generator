#include "random_bignum_generator.h"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include <stdlib.h>
#include <stdio.h>
#include <gmp.h>

// current state of the random number generator
gmp_randstate_t random_state;

/**
 * Generates the next random number and stores it in the bignum given as a
 * parameter.
 * @param number Bignum to set.
 */
void generate_random_bignum(uint32_t* number)
{
    if (number != NULL)
    {
        char* number_str = generate_random_bignum_str();

        if (number_str != NULL)
        {
            string_to_bignum(number_str, number);
            free(number_str);
        }
        else
        {
            printf("Error: \"number_str\" is NULL\n");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        printf("Error: bignum \"number\" is NULL\n");
        exit(EXIT_FAILURE);
    }
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
 * start_random_number_generator() first. You can only use this function if a
 * prior call to start_random_number_generator() has been made, or else this
 * function will fail
 */
void stop_random_number_generator()
{
    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);
}

/**
 * Generates a binary string representation of the next random number. The
 * number is padded with zeros up until a length of TOTAL_BIT_LENGTH.
 * @return Binary string representing the next random number.
 */
char* generate_random_bignum_str()
{
    mpz_t number;
    mpz_init(number);

    // generate random number
    mpz_urandomb(number, random_state, BIT_RANGE);

    // get binary string version
    char* str_number = mpz_get_str(NULL, 2, number);

    if (str_number != NULL)
    {
        pad_string_with_zeros(&str_number);
        mpz_clear(number);
    }
    else
    {
        printf("Error: \"str_number\" is NULL\n");
        exit(EXIT_FAILURE);
    }

    return str_number;
}
