#include "coalesced_normal_addition_benchmark.cuh"
#include "test_constants.h"
#include "bignum_conversions.h"

#include <gmp.h>

void execute_coalesced_normal_addition_on_device(bignum* host_c, bignum* host_a,
                                                 bignum* host_b,
                                                 uint32_t threads_per_block,
                                                 uint32_t blocks_per_grid)
{
    // for this coalesced normal addition, we are going to store the values of
    // host_a, host_b and host_c in a coalesced way, but each in their own
    // arrays. No interleaving here. Each array will look like the following:

    // assuming N = NUMBER_OF_TESTS
    // c[0][0], c[1][0], ..., c[N-1][0]
    // c[0][1], c[1][1], ..., c[N-1][1]
    // c[0][2], c[1][2], ..., c[N-1][2]
    // c[0][3], c[1][3], ..., c[N-1][3]
    // c[0][4], c[1][4], ..., c[N-1][4]

    coalesced_bignum* host_coalesced_a =
        (coalesced_bignum*) calloc(BIGNUM_NUMBER_OF_WORDS,
                                   sizeof(coalesced_bignum));
    coalesced_bignum* host_coalesced_b =
        (coalesced_bignum*) calloc(BIGNUM_NUMBER_OF_WORDS,
                                   sizeof(coalesced_bignum));

    printf("arranging values on cpu ... ");
    fflush(stdout);
    // arrange values of each of the arrays in a coalesced way
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        for (uint32_t j = 0; j < COALESCED_BIGNUM_NUMBER_OF_WORDS; j++)
        {
            host_coalesced_a[i][j] = host_a[j][i];
            host_coalesced_b[i][j] = host_b[j][i];
        }
    }
    printf("done\n");
    fflush(stdout);

    // device operands (dev_coalesced_a, dev_coalesced_b) and results
    // (dev_coalesced_c)
    coalesced_bignum* dev_coalesced_a;
    coalesced_bignum* dev_coalesced_b;
    coalesced_bignum* dev_coalesced_c;

    cudaMalloc((void**) &dev_coalesced_a,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
    cudaMalloc((void**) &dev_coalesced_b,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
    cudaMalloc((void**) &dev_coalesced_c,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));

    // copy operands to device memory
    cudaMemcpy(dev_coalesced_a, host_coalesced_a,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_coalesced_b, host_coalesced_b,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyHostToDevice);

    free(host_coalesced_a);
    free(host_coalesced_b);

    coalesced_normal_addition<<<blocks_per_grid, threads_per_block>>>(
        dev_coalesced_c, dev_coalesced_a, dev_coalesced_b);

    coalesced_bignum* host_coalesced_c =
        (coalesced_bignum*) calloc(BIGNUM_NUMBER_OF_WORDS,
                                   sizeof(coalesced_bignum));

    // copy results back to host
    cudaMemcpy(host_coalesced_c, dev_coalesced_c,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyDeviceToHost);

    // rearrange results into host_c
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        for (uint32_t j = 0; j < COALESCED_BIGNUM_NUMBER_OF_WORDS; j++)
        {
            host_c[j][i] = host_coalesced_c[i][j];
        }
    }

    free(host_coalesced_c);

    // free device memory
    cudaFree(dev_coalesced_a);
    cudaFree(dev_coalesced_b);
    cudaFree(dev_coalesced_c);
}

__global__ void coalesced_normal_addition(coalesced_bignum* dev_coalesced_c,
                                          coalesced_bignum* dev_coalesced_a,
                                          coalesced_bignum* dev_coalesced_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid_increment = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_TESTS)
    {
        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(dev_coalesced_c[0][tid])
            : "r" (dev_coalesced_a[0][tid]),
              "r" (dev_coalesced_b[0][tid]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(dev_coalesced_c[i][tid])
                : "r" (dev_coalesced_a[i][tid]),
                  "r" (dev_coalesced_b[i][tid]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(dev_coalesced_c[BIGNUM_NUMBER_OF_WORDS - 1][tid])
            : "r" (dev_coalesced_a[BIGNUM_NUMBER_OF_WORDS - 1][tid]),
              "r" (dev_coalesced_b[BIGNUM_NUMBER_OF_WORDS - 1][tid]));

        tid += tid_increment;
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
void check_coalesced_normal_addition_results(bignum* host_c, bignum* host_a,
                                             bignum* host_b)
{
    bool results_correct = true;

    for (uint32_t i = 0; results_correct && i < NUMBER_OF_TESTS; i++)
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
