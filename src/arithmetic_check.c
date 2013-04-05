#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <gmp.h>

#define NUMBER_OF_TESTS ((uint32_t) 3)
#define SEED ((uint32_t) 12345)
#define RANDOM_NUMBER_BIT_RANGE ((uint32_t) 131)
#define MODULO ((uint32_t) 12)

void operator_test(void (*function)(mpz_t rop, const mpz_t op1, const mpz_t op2))
{
    // random number generator initialization
    gmp_randstate_t random_state;
    gmp_randinit_default(random_state);
    // incorporated seed in generator
    gmp_randseed_ui(random_state, SEED);

    // initialize test vector operands and result
    mpz_t op1;
    mpz_t op2;
    mpz_t rop;
    mpz_init(op1);
    mpz_init(op2);
    mpz_init(rop);

    for(uint32_t i = 0; i < NUMBER_OF_TESTS; i += 1)
    {
        // generate 2 random numbers as inputs
        mpz_urandomb(op1, random_state, RANDOM_NUMBER_BIT_RANGE);
        mpz_urandomb(op2, random_state, RANDOM_NUMBER_BIT_RANGE);

        // apply function
        function(rop, op1, op2);
    }

    // get memory back from operands and results
    mpz_clear(op1);
    mpz_clear(op2);
    mpz_clear(rop);

    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);
}

void addition(mpz_t rop, const mpz_t op1, const mpz_t op2)
{
    mpz_add(rop, op1, op2);
    gmp_printf("%Zd + %Zd = %Zd\n", op1, op2, rop);

    char* op1_binary_string = mpz_get_str(NULL, 2, op1);
    char* op2_binary_string = mpz_get_str(NULL, 2, op2);
    char* rop_binary_string = mpz_get_str(NULL, 2, rop);
    printf("op1 = \"%s\" (length = %d)\n", op1_binary_string,
           (uint32_t) strlen(op1_binary_string));
    printf("op2 = \"%s\" (length = %d)\n", op2_binary_string,
           (uint32_t) strlen(op2_binary_string));
    printf("rop = \"%s\" (length = %d)\n", rop_binary_string,
           (uint32_t) strlen(rop_binary_string));

    free(op1_binary_string);
    free(op2_binary_string);
    free(rop_binary_string);

    printf("\n");
}

void subtraction(mpz_t rop, const mpz_t op1, const mpz_t op2)
{
    mpz_sub(rop, op1, op2);
    gmp_printf("%Zd - %Zd = %Zd\n", op1, op2, rop);
}

void modular_addition(mpz_t rop, const mpz_t op1, const mpz_t op2)
{
    mpz_t mod;
    mpz_init_set_ui(mod, MODULO);

    // perform modular addition
    mpz_add(rop, op1, op2);
    mpz_cdiv_r(rop, rop, mod);

    // might have to adjust the remainder to be positive, because gmp only
    // guarantees that n = q*d + r, with 0 <= |r| <= |d|
    uint32_t positive_remainder = mpz_cmp_ui(rop, 0);
    if(positive_remainder == -1)
    {
        mpz_add(rop, rop, mod);
    }

    gmp_printf("(%Zd + %Zd) mod %Zd = %Zd\n", op1, op2, mod, rop);

    mpz_clear(mod);
}

void modular_subtraction(mpz_t rop, const mpz_t op1, const mpz_t op2)
{
    mpz_t mod;
    mpz_init_set_ui(mod, MODULO);

    // perform modular addition
    mpz_sub(rop, op1, op2);
    mpz_cdiv_r(rop, rop, mod);

    // might have to adjust the remainder to be positive, because gmp only
    // guarantees that n = q*d + r, with 0 <= |r| <= |d|
    uint32_t positive_remainder = mpz_cmp_ui(rop, 0);
    if(positive_remainder == -1)
    {
        mpz_add(rop, rop, mod);
    }

    gmp_printf("(%Zd - %Zd) mod %Zd = %Zd\n", op1, op2, mod, rop);

    mpz_clear(mod);
}

uint32_t main(void)
{
    operator_test(&addition);
    // operator_test(&subtraction);
    // operator_test(&modular_addition);
    // operator_test(&modular_subtraction);
    return 0;
}
