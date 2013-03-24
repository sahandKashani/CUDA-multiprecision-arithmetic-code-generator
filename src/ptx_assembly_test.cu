#include "conversions.h"
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

// __global__ void test_kernel(int* dev_c, int a, int b);
char* generate_random_number(unsigned int index, unsigned int seed,
                             unsigned int bits, unsigned int base);

int main(void)
{
    printf("Testing inline PTX\n");

    int i = 0;
    char* number_str;

    number_str = generate_random_number(i++, SEED, RANDOM_NUMBER_BIT_RANGE, BASE);

    bignum a;
    string_to_bignum(number_str, a);
    free(number_str);

    // cudaMalloc((void**) &dev_c, sizeof(int));
    // test_kernel<<<1, 1>>>(dev_c, a, b);
    // cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // printf("%d + %d = %d\n", a, b, c);
    // cudaFree(dev_c);
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
char* generate_random_number(unsigned int index, unsigned int seed,
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

// __global__ void test_kernel(int* dev_c, int a, int b)
// {
//     int c;

//     asm("{"
//         "    add.u32 %0, %1, %2;"
//         "}"
//         : "=r"(c) : "r"(a), "r"(b)
//         );

//     *dev_c = c;
// }
