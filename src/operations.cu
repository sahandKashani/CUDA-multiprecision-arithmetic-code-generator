#include "operations.cuh"
#include "bignum_types.h"
#include "constants.h"
#include <stdint.h>

__device__ void add(uint32_t* c, uint32_t* a, uint32_t* b)
{
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

__device__ void subtract(uint32_t* c, uint32_t* a, uint32_t* b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_BIGNUMS)
    {
        asm("sub.cc.u32 %0, %1, %2;"
            : "=r"(c[COAL_IDX(0, tid)])
            : "r" (a[COAL_IDX(0, tid)]),
              "r" (b[COAL_IDX(0, tid)]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("subc.cc.u32 %0, %1, %2;"
                : "=r"(c[COAL_IDX(i, tid)])
                : "r" (a[COAL_IDX(i, tid)]),
                  "r" (b[COAL_IDX(i, tid)]));
        }

        asm("subc.u32 %0, %1, %2;"
            : "=r"(c[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)])
            : "r" (a[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]),
              "r" (b[COAL_IDX(BIGNUM_NUMBER_OF_WORDS - 1, tid)]));

        tid += stride;
    }
}
