#include "normal_addition_benchmark.cuh"
#include "test_constants.h"

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

    normal_addition<<<256, 256>>>(dev_c, dev_a, dev_b);

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
