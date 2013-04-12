#include "benchmarks.cuh"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include "constants.h"
#include "operation_check.h"
#include <stdint.h>

void benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    addition_benchmark(host_c, host_a, host_b);
    // subtraction_benchmark(host_c, host_a, host_b);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// ADDITION ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void addition_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    // arrange data in coalesced form
    bignum_array_to_coalesced_bignum_array(host_a);
    bignum_array_to_coalesced_bignum_array(host_b);
    bignum_array_to_coalesced_bignum_array(host_c);

    // device operands (dev_a, dev_b) and results (dev_c)
    uint32_t* dev_a;
    uint32_t* dev_b;
    uint32_t* dev_c;

    // allocate gpu memory
    cudaMalloc((void**) &dev_a, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaMalloc((void**) &dev_b, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));

    // copy operands to device memory
    cudaMemcpy(dev_a, host_a, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // execute kernel
    addition_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

    // copy results back to host
    cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // put data back to non-coalesced form
    coalesced_bignum_array_to_bignum_array(host_a);
    coalesced_bignum_array_to_bignum_array(host_b);
    coalesced_bignum_array_to_bignum_array(host_c);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // check if the results of the addition are correct by telling gmp to do
    // them on the cpu as a verification.
    addition_check(host_c, host_a, host_b);
}

__global__ void addition_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    add(dev_c, dev_a, dev_b);
}

__device__ void add(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_BIGNUMS)
    {
        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(dev_c[COAL_IDX(0, tid)])
            : "r" (dev_a[COAL_IDX(0, tid)]),
              "r" (dev_b[COAL_IDX(0, tid)]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(dev_c[COAL_IDX(i, tid)])
                : "r" (dev_a[COAL_IDX(i, tid)]),
                  "r" (dev_b[COAL_IDX(i, tid)]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(dev_c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])
            : "r" (dev_a[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]),
              "r" (dev_b[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));

        tid += stride;
    }
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// SUBTRACTION //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void subtraction_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    // arrange data in coalesced form
    bignum_array_to_coalesced_bignum_array(host_a);
    bignum_array_to_coalesced_bignum_array(host_b);
    bignum_array_to_coalesced_bignum_array(host_c);

    // device operands (dev_a, dev_b) and results (dev_c)
    uint32_t* dev_a;
    uint32_t* dev_b;
    uint32_t* dev_c;

    // allocate gpu memory
    cudaMalloc((void**) &dev_a, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaMalloc((void**) &dev_b, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));

    // copy operands to device memory
    cudaMemcpy(dev_a, host_a, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // execute kernel
    subtraction_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

    // copy results back to host
    cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // put data back to non-coalesced form
    coalesced_bignum_array_to_bignum_array(host_a);
    coalesced_bignum_array_to_bignum_array(host_b);
    coalesced_bignum_array_to_bignum_array(host_c);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // check if the results of the subtraction are correct by telling gmp to do
    // them on the cpu as a verification.
    subtraction_check(host_c, host_a, host_b);
}

__global__ void subtraction_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    subtract(dev_c, dev_a, dev_b);
}

__device__ void subtract(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_BIGNUMS)
    {
        asm("sub.cc.u32 %0, %1, %2;"
            : "=r"(dev_c[COAL_IDX(0, tid)])
            : "r" (dev_a[COAL_IDX(0, tid)]),
              "r" (dev_b[COAL_IDX(0, tid)]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("subc.cc.u32 %0, %1, %2;"
                : "=r"(dev_c[COAL_IDX(i, tid)])
                : "r" (dev_a[COAL_IDX(i, tid)]),
                  "r" (dev_b[COAL_IDX(i, tid)]));
        }

        asm("subc.u32 %0, %1, %2;"
            : "=r"(dev_c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])
            : "r" (dev_a[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]),
              "r" (dev_b[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));

        tid += stride;
    }
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// MODULAR ADDITION ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// MODULAR SUBTRACTION /////////////////////////////
////////////////////////////////////////////////////////////////////////////////
