#include "random_bignum_generator.h"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gmp.h>

// current state of the random number generator
gmp_randstate_t random_state;

void generate_generic_random_bignum(uint32_t* number, char* (*f)(void))
{
    if (number != NULL)
    {
        char* number_str = f();

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

void generate_random_bignum_modulus(uint32_t* number)
{
    generate_generic_random_bignum(number, generate_random_bignum_modulus_str);
}

void generate_random_bignum(uint32_t* number)
{
    generate_generic_random_bignum(number, generate_random_bignum_str);
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

char* generate_random_bignum_modulus_str()
{
    mpz_t number;
    mpz_init(number);

    char* str_number;

    // generate a random number of exactly BIT_RANGE size (at least with a '1'
    // at most significant bit which is at position BIT_RANGE)
    do
    {
        // generate random number
        mpz_urandomb(number, random_state, BIT_RANGE);

        // get binary string version
        str_number = mpz_get_str(NULL, 2, number);

        if (str_number == NULL)
        {
            printf("Error: \"str_number\" is NULL\n");
            exit(EXIT_FAILURE);
        }
    }
    while (strlen(str_number) != BIT_RANGE);

    pad_string_with_zeros(&str_number);
    mpz_clear(number);

    return str_number;
}
