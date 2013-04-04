#include "interleaved_addition_benchmark.cuh"
#include "test_constants.h"
#include "bignum_conversions.h"

#include <gmp.h>

void execute_interleaved_addition_on_device(bignum* host_c, bignum* host_a,
                                            bignum* host_b)
{
    // for this interleaved addition, we are going to interleave the values of
    // the 2 operands host_a and host_b.
    // Our operands will look like the following:

    // host_a[0][0], host_b[0][0], host_a[0][1], host_b[0][1],
    // host_a[0][2], host_b[0][2], host_a[0][3], host_b[0][3],
    // host_a[0][4], host_b[0][4], host_a[1][0], host_b[1][0], ...

    // our results will be stocked sequentially as for normal addition.

    void* host_ops = calloc(NUMBER_OF_TESTS, sizeof(interleaved_bignum));
    interleaved_bignum* host_interleaved_operands = (interleaved_bignum*) host_ops;
    host_ops = NULL;

    // interleave values of host_a and host_b in host_interleaved_operands.
    for (int i = 0; i < NUMBER_OF_TESTS; i++)
    {
        for (int j = 0; j < INTERLEAVED_BIGNUM_NUMBER_OF_WORDS; j++)
        {
            if (j % 2 == 0)
            {
                host_interleaved_operands[i][j] = host_a[i][j / 2];
            }
            else
            {
                host_interleaved_operands[i][j] = host_b[i][j / 2];
            }
        }
    }

    // device operands (dev_interleaved_operands) and results (dev_results)
    interleaved_bignum* dev_interleaved_operands;
    bignum* dev_results;

    cudaMalloc((void**) &dev_interleaved_operands,
               NUMBER_OF_TESTS * sizeof(interleaved_bignum));
    cudaMalloc((void**) &dev_results, NUMBER_OF_TESTS * sizeof(bignum));

    // copy operands to device memory
    cudaMemcpy(dev_interleaved_operands, host_interleaved_operands,
               NUMBER_OF_TESTS * sizeof(interleaved_bignum),
               cudaMemcpyHostToDevice);

    // free host_interleaved_operands which we no longer need.
    free(host_interleaved_operands);

    printf("executing interleaved addition ... ");
    interleaved_addition<<<256, 256>>>(dev_results, dev_interleaved_operands);
    printf("done\n");

    // copy results back to host
    cudaMemcpy(host_c, dev_results, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_interleaved_operands);
    cudaFree(dev_results);
}

__global__ void interleaved_addition(bignum* dev_results,
                                    interleaved_bignum* dev_interleaved_operands)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < NUMBER_OF_TESTS)
    {
        asm("add.cc.u32  %0, %1, %2;"
            : "=r"(dev_results[tid][0])
            : "r"(dev_interleaved_operands[tid][0]),
              "r"(dev_interleaved_operands[tid][1])
            );

        asm("addc.cc.u32 %0, %1, %2;"
            : "=r"(dev_results[tid][1])
            : "r"(dev_interleaved_operands[tid][2]),
              "r"(dev_interleaved_operands[tid][3])
            );

        asm("addc.cc.u32 %0, %1, %2;"
            : "=r"(dev_results[tid][2])
            : "r"(dev_interleaved_operands[tid][4]),
              "r"(dev_interleaved_operands[tid][5])
            );

        asm("addc.cc.u32 %0, %1, %2;"
            : "=r"(dev_results[tid][3])
            : "r"(dev_interleaved_operands[tid][6]),
              "r"(dev_interleaved_operands[tid][7])
            );

        asm("addc.u32    %0, %1, %2;"
            : "=r"(dev_results[tid][4])
            : "r"(dev_interleaved_operands[tid][8]),
              "r"(dev_interleaved_operands[tid][9])
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
void check_interleaved_addition_results(bignum* host_c, bignum* host_a,
                                        bignum* host_b)
{
    printf("checking resuts ... ");
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
