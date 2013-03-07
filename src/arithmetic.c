#include <stdio.h>
#include <gmp.h>

#define NUMBER_OF_TEST_VECTORS ((unsigned int) 1e6)
#define SEED 12345
#define RANDOM_NUMBER_BIT_RANGE 131

/**
 * Structure to hold a test vector. A test vector contains 2 inputs and 1
 * output. This can be used for testing addition and subtraction (with or
 * without modulo arithmetic)
 */
typedef struct
{
    mpz_t op1; // first  operand
    mpz_t op2; // second operand
    mpz_t rop; // result operand
} test_vector;

test_vector cpu_test_vectors[NUMBER_OF_TEST_VECTORS];

/**
 * Function which tests a binary operator over mpz_t (integers). It expects a
 * pointer towards a binary function, and a char representing the printed form
 * of the operation (for representation purposes).
 */
void binary_operator_test(void (*function)(mpz_t, const mpz_t, const mpz_t),
                          char operator)
{
    // random number generator initialization
    gmp_randstate_t random_state;
    gmp_randinit_default(random_state);
    // incorporated seed in generator
    gmp_randseed_ui(random_state, SEED);

    for(int i = 0; i < NUMBER_OF_TEST_VECTORS; i += 1)
    {
        // initialize test vector operands and result
        mpz_init(cpu_test_vectors[i].op1);
        mpz_init(cpu_test_vectors[i].op2);
        mpz_init(cpu_test_vectors[i].rop);

        // generate 2 random numbers as inputs
        mpz_urandomb(cpu_test_vectors[i].op1,
                     random_state,
                     RANDOM_NUMBER_BIT_RANGE);
        mpz_urandomb(cpu_test_vectors[i].op2,
                     random_state,
                     RANDOM_NUMBER_BIT_RANGE);

        // apply function
        function(cpu_test_vectors[i].rop,
                cpu_test_vectors[i].op1,
                cpu_test_vectors[i].op2);

        // gmp_printf("%Zd %c %Zd = %Zd\n",
        //            cpu_test_vectors[i].op1,
        //            operator,
        //            cpu_test_vectors[i].op2,
        //            cpu_test_vectors[i].rop);

        // get memory back from test vectors
        mpz_clear(cpu_test_vectors[i].op1);
        mpz_clear(cpu_test_vectors[i].op2);
        mpz_clear(cpu_test_vectors[i].rop);
    }

    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);
}

/**
 * Tests the addition operator
 */
void addition_test()
{
    binary_operator_test(&mpz_add, '+');
}

/**
 * Tests the subtraction operator
 */
void subtraction_test()
{
    binary_operator_test(&mpz_sub, '-');
}

int main(void)
{
    addition_test();
    subtraction_test();

    return 0;
}
