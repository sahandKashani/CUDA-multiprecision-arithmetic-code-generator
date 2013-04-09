#include "coalesced_normal_addition_benchmark.cuh"
#include "test_constants.h"
#include "bignum_conversions.h"

#include <gmp.h>

void execute_coalesced_normal_addition_on_device(bignum* host_c, bignum* host_a,
                                                 bignum* host_b,
                                                 uint32_t threads_per_block,
                                                 uint32_t blocks_per_grid)
{
    coalesced_bignum* host_coalesced_a =
        (coalesced_bignum*) calloc(BIGNUM_NUMBER_OF_WORDS,
                                   sizeof(coalesced_bignum));
    coalesced_bignum* host_coalesced_b =
        (coalesced_bignum*) calloc(BIGNUM_NUMBER_OF_WORDS,
                                   sizeof(coalesced_bignum));

    // arrange values of each of the arrays in a coalesced way
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        for (uint32_t j = 0; j < NUMBER_OF_TESTS; j++)
        {
            host_coalesced_a[i][j] = host_a[j][i];
            host_coalesced_b[i][j] = host_b[j][i];
        }
    }

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
        for (uint32_t j = 0; j < NUMBER_OF_TESTS; j++)
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
