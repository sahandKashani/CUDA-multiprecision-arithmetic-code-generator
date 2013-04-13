#include "benchmarks.cuh"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include "constants.h"
#include "operation_check.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    if (host_a != NULL && host_b != NULL && host_c != NULL)
    {
        addition_benchmark(host_c, host_a, host_b);
        subtraction_benchmark(host_c, host_a, host_b);
        // modular_subtraction_benchmark(host_c, host_a, host_b);
    }
    else
    {
        if (host_a == NULL)
        {
            printf("Error: bignum array \"host_a\" is NULL\n");
        }

        if (host_b == NULL)
        {
            printf("Error: bignum array \"host_b\" is NULL\n");
        }

        if (host_c == NULL)
        {
            printf("Error: bignum array \"host_c\" is NULL\n");
        }

        exit(EXIT_FAILURE);
    }
}

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), void (*checking_function)(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b))
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
    cudaError dev_a_malloc_success = cudaMalloc((void**) &dev_a, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_malloc_success = cudaMalloc((void**) &dev_b, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_malloc_success = cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));

    if (dev_a_malloc_success == cudaSuccess && dev_b_malloc_success == cudaSuccess && dev_c_malloc_success == cudaSuccess)
    {
        // copy operands to device memory
        cudaError dev_a_memcpy_succes = cudaMemcpy(dev_a, host_a, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaError dev_b_memcpy_succes = cudaMemcpy(dev_b, host_b, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);

        if (dev_a_memcpy_succes == cudaSuccess && dev_b_memcpy_succes == cudaSuccess)
        {
            // execute kernel
            kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

            // copy results back to host
            cudaError dev_c_memcpy_success = cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (dev_c_memcpy_success == cudaSuccess)
            {
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
            else
            {
                printf("Error: could not copy \"dev_c\" to \"host_c\"\n");
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            if (dev_a_memcpy_succes != cudaSuccess)
            {
                printf("Error: could not copy \"host_a\" to \"dev_a\"\n");
            }

            if (dev_b_memcpy_succes != cudaSuccess)
            {
                printf("Error: could not copy \"host_b\" to \"dev_b\"\n");
            }

            exit(EXIT_FAILURE);
        }
    }
    else
    {
        if (dev_a_malloc_success != cudaSuccess)
        {
            printf("Error: could not allocate device memory for \"dev_a\"");
        }

        if (dev_b_malloc_success != cudaSuccess)
        {
            printf("Error: could not allocate device memory for \"dev_b\"");
        }

        if (dev_c_malloc_success != cudaSuccess)
        {
            printf("Error: could not allocate device memory for \"dev_c\"");
        }

        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// ADDITION ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void addition_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, addition_kernel, addition_check);
}

__global__ void addition_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    addition(dev_c, dev_a, dev_b);
}

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

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// SUBTRACTION //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void subtraction_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, subtraction_kernel, subtraction_check);
}

__global__ void subtraction_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    subtraction(dev_c, dev_a, dev_b);
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

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// MODULAR ADDITION ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// MODULAR SUBTRACTION /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void modular_subtraction_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, modular_subtraction_kernel, subtraction_check);
}

__global__ void modular_subtraction_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    modular_subtraction(dev_c, dev_a, dev_b);
}

__device__ void modular_subtraction(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
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
///////////////////////////////// MULTIPLICATION ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// MODULAR MULTIPLICATION ///////////////////////////
////////////////////////////////////////////////////////////////////////////////
