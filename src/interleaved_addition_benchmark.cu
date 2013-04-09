#include "interleaved_addition_benchmark.cuh"
#include "test_constants.h"
#include "bignum_conversions.h"

#include <gmp.h>

void execute_interleaved_addition_on_device(bignum* host_c, bignum* host_a,
                                            bignum* host_b,
                                            uint32_t threads_per_block,
                                            uint32_t blocks_per_grid)
{
    interleaved_bignum* host_interleaved_operands =
        (interleaved_bignum*) calloc(NUMBER_OF_TESTS, sizeof(interleaved_bignum));

    // interleave values of host_a and host_b in host_interleaved_operands.
    for (uint32_t i = 0; i < NUMBER_OF_TESTS; i++)
    {
        for (uint32_t j = 0; j < 2 * BIGNUM_NUMBER_OF_WORDS; j++)
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

    interleaved_addition<<<blocks_per_grid, threads_per_block>>>(
        dev_results, dev_interleaved_operands);

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
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid_increment = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_TESTS)
    {
        uint32_t i = 0;
        uint32_t col = 2 * i;

        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(dev_results[tid][i])
            : "r" (dev_interleaved_operands[tid][col]),
              "r" (dev_interleaved_operands[tid][col + 1]));

        #pragma unroll
        for (i = 1, col = 2 * i; i < BIGNUM_NUMBER_OF_WORDS - 1; i++, col = 2 * i)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(dev_results[tid][i])
                : "r" (dev_interleaved_operands[tid][col]),
                  "r" (dev_interleaved_operands[tid][col + 1]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(dev_results[tid][BIGNUM_NUMBER_OF_WORDS - 1])
            : "r" (dev_interleaved_operands[tid][col]),
              "r" (dev_interleaved_operands[tid][col + 1]));

        tid += tid_increment;
    }
}
