#include <stdio.h>
#include <gmp.h>

#define NUMBER_OF_TEST_VECTORS ((unsigned long int) 1e5)
#define SEED 12345
#define RANDOM_NUMBER_BIT_RANGE 131

typedef struct
{
    mpz_t input_1;
    mpz_t input_2;
    mpz_t output;
} test_vector;

test_vector cpu_test_vectors[NUMBER_OF_TEST_VECTORS];

int main(void)
{
    gmp_randstate_t random_state;
    gmp_randinit_default(random_state);
    gmp_randseed_ui(random_state, SEED);

    for(int i = 0; i < NUMBER_OF_TEST_VECTORS; i += 1)
    {
        mpz_init(cpu_test_vectors[i].input_1);
        mpz_init(cpu_test_vectors[i].input_2);
        mpz_init(cpu_test_vectors[i].output);
    }

    for(int i = 0; i < NUMBER_OF_TEST_VECTORS; i += 1)
    {
        mpz_urandomb(cpu_test_vectors[i].input_1,
                     random_state,
                     RANDOM_NUMBER_BIT_RANGE);

        mpz_urandomb(cpu_test_vectors[i].input_2,
                     random_state,
                     RANDOM_NUMBER_BIT_RANGE);

        mpz_add(cpu_test_vectors[i].output,
                cpu_test_vectors[i].input_1,
                cpu_test_vectors[i].input_2);
    }

    for(int i = 0; i < NUMBER_OF_TEST_VECTORS; i += 1)
    {
        gmp_printf("%Zd + %Zd = %Zd\n",
                   cpu_test_vectors[i].input_1,
                   cpu_test_vectors[i].input_2,
                   cpu_test_vectors[i].output);
    }

    for(int i = 0; i < NUMBER_OF_TEST_VECTORS; i += 1)
    {
        mpz_clear(cpu_test_vectors[i].input_1);
        mpz_clear(cpu_test_vectors[i].input_2);
        mpz_clear(cpu_test_vectors[i].output);
    }

    return 0;
}
