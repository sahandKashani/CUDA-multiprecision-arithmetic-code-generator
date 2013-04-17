#include "benchmarks.cuh"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include "constants.h"
#include "operation_check.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), void (*checking_function)(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b), char* operation_name);

void addition_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
__global__ void addition_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__device__ void addition(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

void subtraction_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
__global__ void subtraction_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__device__ void subtraction(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// BENCHMARKS /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    addition_benchmark(host_c, host_a, host_b);
    subtraction_benchmark(host_c, host_a, host_b);
}

void addition_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    binary_operator_benchmark(host_c, host_a, host_b, addition_kernel, addition_check, "addition");
}

void subtraction_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    binary_operator_benchmark(host_c, host_a, host_b, subtraction_kernel, subtraction_check, "subtraction");
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// KERNELS ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void addition_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    addition(dev_c, dev_a, dev_b);
}

__global__ void subtraction_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    subtraction(dev_c, dev_a, dev_b);
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DEVICE FUNCTIONS //////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__device__ void addition(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
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

__device__ void subtraction(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
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

__device__ void modular_addition(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_BIGNUMS)
    {
        // addition
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

        // subtraction
        asm("sub.cc.u32 %0, %0, %1;"
            : "+r"(dev_c[COAL_IDX(0, tid)])
            : "r" (dev_m[COAL_IDX(0, tid)]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("subc.cc.u32 %0, %0, %1;"
                : "+r"(dev_c[COAL_IDX(i, tid)])
                : "r" (dev_m[COAL_IDX(i, tid)]));
        }

        // we want the borrow bit (unlike for normal subtraction)
        asm("subc.cc.u32 %0, %0, %1;"
            : "+r"(dev_c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])
            : "r" (dev_m[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));

        uint32_t borrow[BIGNUM_NUMBER_OF_WORDS];
        asm("subc.u32 %0, 0, ");
        for

        uint32_t mask[BIGNUM_NUMBER_OF_WORDS];
        for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
        {
            mask[i] = 0;
        }

        // mask = 0 until now
        // now do mask = mask - mask

        // subtraction
        asm("subc.cc.u32 %0, %0, %1;"
            : "+r"(dev_c[COAL_IDX(0, tid)])
            : "r" (dev_m[COAL_IDX(0, tid)]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("subc.cc.u32 %0, %0, %1;"
                : "+r"(dev_c[COAL_IDX(i, tid)])
                : "r" (dev_m[COAL_IDX(i, tid)]));
        }

        // we want the borrow bit (unlike for normal subtraction)
        asm("subc.cc.u32 %0, %0, %1;"
            : "+r"(dev_c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])
            : "r" (dev_m[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));

        tid += stride;
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////// GENERIC LAUNCH CONFIGURATION ////////////////////////
////////////////////////////////////////////////////////////////////////////////

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), void (*checking_function)(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b), char* operation_name)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);
    assert(kernel != NULL);
    assert(checking_function != NULL);
    assert(operation_name != NULL);

    // arrange data in coalesced form
    bignum_array_to_coalesced_bignum_array(host_a);
    bignum_array_to_coalesced_bignum_array(host_b);
    bignum_array_to_coalesced_bignum_array(host_c);

    // device operands (dev_a, dev_b) and results (dev_c)
    uint32_t* dev_a;
    uint32_t* dev_b;
    uint32_t* dev_c;

    // allocate gpu memory
    cudaError dev_a_malloc_success = cudaMalloc((void**) &dev_a, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_malloc_success = cudaMalloc((void**) &dev_b, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_malloc_success = cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));

    assert(dev_a_malloc_success == cudaSuccess);
    assert(dev_b_malloc_success == cudaSuccess);
    assert(dev_c_malloc_success == cudaSuccess);

    // copy operands to device memory
    cudaError dev_a_memcpy_succes = cudaMemcpy(dev_a, host_a, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError dev_b_memcpy_succes = cudaMemcpy(dev_b, host_b, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    assert(dev_a_memcpy_succes == cudaSuccess);
    assert(dev_b_memcpy_succes == cudaSuccess);

    // execute kernel
    printf("Performing \"%s\" on GPU ... ", operation_name);
    fflush(stdout);

    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

    printf("done\n");
    fflush(stdout);

    // copy results back to host
    cudaError dev_c_memcpy_success = cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    assert(dev_c_memcpy_success == cudaSuccess);

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
    checking_function(host_c, host_a, host_b);
}
