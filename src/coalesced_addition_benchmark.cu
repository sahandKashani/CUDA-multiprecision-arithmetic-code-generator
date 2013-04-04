#include "coalesced_addition_benchmark.cuh"
#include "test_constants.h"
#include "bignum_conversions.h"

#include <gmp.h>

void execute_coalesced_addition_on_device(bignum* host_c, bignum* host_a,
                                          bignum* host_b)
{
    // for this coalesced addition, we are going to store the values of the 2
    // operands host_a and host_b in a special way such that each thread in a
    // block can access its iterative operands with a single global memory
    // operation. Our operands will look like the following:

    // assuming N = 2 * NUMBER_OF_TESTS
    // a[0][0], b[0][0], a[1][0], b[1][0], ..., a[N-1][0], b[N-1][0]
    // a[0][1], b[0][1], a[1][1], b[1][1], ..., a[N-1][1], b[N-1][1]
    // a[0][2], b[0][2], a[1][2], b[1][2], ..., a[N-1][2], b[N-1][2]
    // a[0][3], b[0][3], a[1][3], b[1][3], ..., a[N-1][3], b[N-1][3]
    // a[0][4], b[0][4], a[1][4], b[1][4], ..., a[N-1][4], b[N-1][4]

    // Therefore, for best performance, you must choose a TOTAL number of
    // threads close to NUMBER_OF_TESTS

    // our results will be stocked sequentially as for normal addition.

    void* host_ops = calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(coalesced_bignum));
    coalesced_bignum* coalesced_operands = (coalesced_bignum*) host_ops;
    host_ops = NULL;

    // arrange values of host_a and host_b in coalesced_operands.
    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        for (int j = 0; j < COALESCED_BIGNUM_NUMBER_OF_WORDS; j += 2)
        {
            coalesced_operands[i][j] = host_a[j/2][i];
            coalesced_operands[i][j + 1] = host_b[j/2][i];
        }
    }

    // device operands (dev_coalesced_operands) and results (dev_results)
    coalesced_bignum* dev_coalesced_operands;
    coalesced_bignum_result* dev_results;

    cudaMalloc((void**) &dev_coalesced_operands,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
    cudaMalloc((void**) &dev_results,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum_result));

    // copy operands to device memory
    cudaMemcpy(dev_coalesced_operands, coalesced_operands,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyHostToDevice);

    free(coalesced_operands);

    printf("executing coalesced addition ... ");
    coalesced_addition<<<256, 256>>>(dev_results, dev_coalesced_operands);
    printf("done\n");

    void* host_results_tmp = calloc(BIGNUM_NUMBER_OF_WORDS,
                                    sizeof(coalesced_bignum_result));
    coalesced_bignum_result* host_results = (coalesced_bignum_result*) host_results_tmp;
    host_results_tmp = NULL;

    // copy results back to host
    cudaMemcpy(host_results, dev_results, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // rearrange result values into host_c
    for (int i = 0; i < COALESCED_BIGNUM_RESULT_NUMBER_OF_WORDS; i++)
    {
        for (int j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
        {
            host_c[i][j] = host_results[j][i];
        }
    }

    free(host_results);

    // free device memory
    cudaFree(dev_coalesced_operands);
    cudaFree(dev_results);
}

__global__ void coalesced_addition(coalesced_bignum_result* dev_results,
                                   coalesced_bignum* dev_coalesced_operands)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < NUMBER_OF_TESTS)
    {
        int col = 2 * tid;

        asm("add.cc.u32  %0, %1, %2;"
            : "=r"(dev_results[0][tid])
            : "r"(dev_coalesced_operands[0][col]),
              "r"(dev_coalesced_operands[0][col + 1])
            );

        asm("addc.cc.u32 %0, %1, %2;"
            : "=r"(dev_results[1][tid])
            : "r"(dev_coalesced_operands[1][col]),
              "r"(dev_coalesced_operands[1][col + 1])
            );

        asm("addc.cc.u32 %0, %1, %2;"
            : "=r"(dev_results[2][tid])
            : "r"(dev_coalesced_operands[2][col]),
              "r"(dev_coalesced_operands[2][col + 1])
            );

        asm("addc.cc.u32 %0, %1, %2;"
            : "=r"(dev_results[3][tid])
            : "r"(dev_coalesced_operands[3][col]),
              "r"(dev_coalesced_operands[3][col + 1])
            );

        asm("addc.u32    %0, %1, %2;"
            : "=r"(dev_results[4][tid])
            : "r"(dev_coalesced_operands[4][col]),
              "r"(dev_coalesced_operands[4][col + 1])
            );

        tid += blockDim.x * gridDim.x;
    }
}

/**
 * Checks if host_a op host_b == host_c, where host_c is to be tested against
 * values computed by gmp. If you have data in any other formats than these, you
 * will have to "rearrange" them to meet this pattern for the check to work.
 * @param host_c Values we have computed with our algorithms.
 * @param host_a First operands.
 * @param host_b Second operands.
 */
void check_coalesced_addition_results(bignum* host_c, bignum* host_a,
                                      bignum* host_b)
{
    printf("checking results ... ");
    bool results_correct = true;

    for (int i = 0; results_correct && i < NUMBER_OF_TESTS; i++)
    {
        char* bignum_a_str = bignum_to_string(host_a[i]);
        char* bignum_b_str = bignum_to_string(host_b[i]);
        char* bignum_c_str = bignum_to_string(host_c[i]);

        mpz_t gmp_bignum_a;
        mpz_t gmp_bignum_b;
        mpz_t gmp_bignum_c;

        mpz_init_set_str(gmp_bignum_a, bignum_a_str, 2);
        mpz_init_set_str(gmp_bignum_b, bignum_b_str, 2);
        mpz_init(gmp_bignum_c);

        // GMP function which will calculate what our algorithm is supposed to
        // calculate
        mpz_add(gmp_bignum_c, gmp_bignum_a, gmp_bignum_b);

        // get binary string result
        char* gmp_bignum_c_str = mpz_get_str(NULL, 2, gmp_bignum_c);
        pad_string_with_zeros(&gmp_bignum_c_str);

        if (strcmp(gmp_bignum_c_str, bignum_c_str) != 0)
        {
            printf("incorrect calculation at iteration %d\n", i);
            results_correct = false;
            printf("own\n%s +\n%s =\n%s\n", bignum_a_str, bignum_b_str,
                   bignum_c_str);
            printf("gmp\n%s +\n%s =\n%s\n", bignum_a_str, bignum_b_str,
                   gmp_bignum_c_str);
        }

        free(bignum_a_str);
        free(bignum_b_str);
        free(bignum_c_str);
        free(gmp_bignum_c_str);

        mpz_clear(gmp_bignum_a);
        mpz_clear(gmp_bignum_b);
        mpz_clear(gmp_bignum_c);
    }

    if (results_correct)
    {
        printf("all correct\n");
    }
    else
    {
        printf("something wrong\n");
    }
}