#include "interleaved_addition_benchmark.cuh"
#include "test_constants.h"

void execute_interleaved_addition_on_device(bignum* host_c, bignum* host_a,
                                            bignum* host_b)
{
    // for this coalescing addition, we are going to interleave the values of
    // the 2 operands host_a and host_b.
    // Our operands will look like the following:

    // host_a[0][0], host_b[0][0], host_a[0][1], host_b[0][1],
    // host_a[0][2], host_b[0][2], host_a[0][3], host_b[0][3],
    // host_a[0][4], host_b[0][4], host_a[1][0], host_b[1][0], ...

    // our results will be stocked sequentially as for normal addition.

    void* host_ops = calloc(NUMBER_OF_TESTS, sizeof(interleaved_bignum));
    interleaved_bignum* interleaved_operands = (interleaved_bignum*) host_ops;
    host_ops = NULL;

    // interleave values of host_a and host_b in interleaved_operands.
    for (int i = 0; i < NUMBER_OF_TESTS; i++)
    {
        for (int j = 0; j < INTERLEAVED_BIGNUM_NUMBER_OF_WORDS; j++)
        {
            if (j % 2 == 0)
            {
                interleaved_operands[i][j] = host_a[i][j / 2];
            }
            else
            {
                interleaved_operands[i][j] = host_b[i][j / 2];
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
    cudaMemcpy(dev_interleaved_operands, interleaved_operands,
               NUMBER_OF_TESTS * sizeof(interleaved_bignum),
               cudaMemcpyHostToDevice);

    // free interleaved_operands which we no longer need.
    free(interleaved_operands);

    interleaved_addition<<<256, 256>>>(dev_results, dev_interleaved_operands);

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
        asm("{"
            "    add.cc.u32  %0, %5, %10;"
            "    addc.cc.u32 %1, %6, %11;"
            "    addc.cc.u32 %2, %7, %12;"
            "    addc.cc.u32 %3, %8, %13;"
            "    addc.u32    %4, %9, %14;"
            "}"

            : "=r"(dev_results[tid][0]),
              "=r"(dev_results[tid][1]),
              "=r"(dev_results[tid][2]),
              "=r"(dev_results[tid][3]),
              "=r"(dev_results[tid][4])

            : "r"(dev_interleaved_operands[tid][0]),
              "r"(dev_interleaved_operands[tid][2]),
              "r"(dev_interleaved_operands[tid][4]),
              "r"(dev_interleaved_operands[tid][6]),
              "r"(dev_interleaved_operands[tid][8]),

              "r"(dev_interleaved_operands[tid][1]),
              "r"(dev_interleaved_operands[tid][3]),
              "r"(dev_interleaved_operands[tid][5]),
              "r"(dev_interleaved_operands[tid][7]),
              "r"(dev_interleaved_operands[tid][9])
            );

        tid += blockDim.x * gridDim.x;
    }
}
