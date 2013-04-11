#include "memory_layout_benchmarks.cuh"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include "constants.h"
#include <stdint.h>

void benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, uint32_t threads_per_block, uint32_t blocks_per_grid)
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

    // execute benchmark kernels
    kernel_1<<<blocks_per_grid, threads_per_block>>>(dev_c, dev_a, dev_b);

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
}

__global__ void kernel_1(uint32_t* c, uint32_t* a, uint32_t* b)
{
    // add(c, a, b);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_BIGNUMS)
    {
        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(c[COAL_IDX(0, tid)])
            : "r" (a[COAL_IDX(0, tid)]),
              "r" (b[COAL_IDX(0, tid)]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(c[COAL_IDX(i, tid)])
                : "r" (a[COAL_IDX(i, tid)]),
                  "r" (b[COAL_IDX(i, tid)]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])
            : "r" (a[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]),
              "r" (b[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));

        tid += stride;
    }
}

// __device__ void add(uint32_t* c, uint32_t* a, uint32_t* b)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {
//         asm("add.cc.u32 %0, %1, %2;"
//             : "=r"(c[COAL_IDX(0, tid)])
//             : "r" (a[COAL_IDX(0, tid)]),
//               "r" (b[COAL_IDX(0, tid)]));

//         #pragma unroll
//         for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
//         {
//             asm("addc.cc.u32 %0, %1, %2;"
//                 : "=r"(c[COAL_IDX(i, tid)])
//                 : "r" (a[COAL_IDX(i, tid)]),
//                   "r" (b[COAL_IDX(i, tid)]));
//         }

//         asm("addc.u32 %0, %1, %2;"
//             : "=r"(c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])
//             : "r" (a[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]),
//               "r" (b[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));

//         tid += stride;
//     }
// }

// __device__ void subtract(uint32_t* c, uint32_t* a, uint32_t* b)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {
//         asm("sub.cc.u32 %0, %1, %2;"
//             : "=r"(c[COAL_IDX(0, tid)])
//             : "r" (a[COAL_IDX(0, tid)]),
//               "r" (b[COAL_IDX(0, tid)]));

//         #pragma unroll
//         for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
//         {
//             asm("subc.cc.u32 %0, %1, %2;"
//                 : "=r"(c[COAL_IDX(i, tid)])
//                 : "r" (a[COAL_IDX(i, tid)]),
//                   "r" (b[COAL_IDX(i, tid)]));
//         }

//         asm("subc.u32 %0, %1, %2;"
//             : "=r"(c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])
//             : "r" (a[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]),
//               "r" (b[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));

//         tid += stride;
//     }
// }
