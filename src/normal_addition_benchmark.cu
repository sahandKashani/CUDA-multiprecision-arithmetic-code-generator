#include "normal_addition_benchmark.cuh"
#include "test_constants.h"
#include "bignum_conversions.h"

#include <gmp.h>

void execute_normal_addition_on_device(bignum* host_c, bignum* host_a,
                                       bignum* host_b)
{
    // device operands (dev_a, dev_b) and results (dev_c)
    bignum* dev_a;
    bignum* dev_b;
    bignum* dev_c;

    cudaMalloc((void**) &dev_a, NUMBER_OF_TESTS * sizeof(bignum));
    cudaMalloc((void**) &dev_b, NUMBER_OF_TESTS * sizeof(bignum));
    cudaMalloc((void**) &dev_c, NUMBER_OF_TESTS * sizeof(bignum));

    // copy operands to device memory
    cudaMemcpy(dev_a, host_a, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyHostToDevice);

    printf("executing normal addition ... ");
    normal_addition<<<256, 256>>>(dev_c, dev_a, dev_b);
    printf("done\n");

    // copy results back to host
    cudaMemcpy(host_c, dev_c, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ void normal_addition(bignum* dev_c, bignum* dev_a, bignum* dev_b)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < NUMBER_OF_TESTS)
    {
        asm("{"
            "    add.cc.u32  %0, %5, %10;"
            "    addc.cc.u32 %1, %6, %11;"
            "    addc.cc.u32 %2, %7, %12;"
            "    addc.cc.u32 %3, %8, %13;"
            "    addc.u32    %4, %9, %14;"
            "}"

            : "=r"(dev_c[tid][0]),
              "=r"(dev_c[tid][1]),
              "=r"(dev_c[tid][2]),
              "=r"(dev_c[tid][3]),
              "=r"(dev_c[tid][4])

            : "r"(dev_a[tid][0]),
              "r"(dev_a[tid][1]),
              "r"(dev_a[tid][2]),
              "r"(dev_a[tid][3]),
              "r"(dev_a[tid][4]),

              "r"(dev_b[tid][0]),
              "r"(dev_b[tid][1]),
              "r"(dev_b[tid][2]),
              "r"(dev_b[tid][3]),
              "r"(dev_b[tid][4])
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
void check_normal_addition_results(bignum* host_c, bignum* host_a,
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
