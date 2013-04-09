#include "normal_addition_benchmark.cuh"
#include "test_constants.h"
#include "bignum_conversions.h"

#include <gmp.h>

void execute_normal_addition_on_device(bignum* host_c, bignum* host_a,
                                       bignum* host_b,
                                       uint32_t threads_per_block,
                                       uint32_t blocks_per_grid)
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

    normal_addition<<<blocks_per_grid, threads_per_block>>>(dev_c, dev_a, dev_b);

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
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid_increment = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_TESTS)
    {

        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(dev_c[tid][0])
            : "r" (dev_a[tid][0]),
              "r" (dev_b[tid][0]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(dev_c[tid][i])
                : "r" (dev_a[tid][i]),
                  "r" (dev_b[tid][i]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(dev_c[tid][BIGNUM_NUMBER_OF_WORDS - 1])
            : "r" (dev_a[tid][BIGNUM_NUMBER_OF_WORDS - 1]),
              "r" (dev_b[tid][BIGNUM_NUMBER_OF_WORDS - 1]));

        tid += tid_increment;
    }
}
