#include "operations.cuh"
#include "bignum_types.h"
#include "constants.h"
#include <stdint.h>

__device__ void add(coalesced_bignum* c,
                    coalesced_bignum* a,
                    coalesced_bignum* b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid_increment = blockDim.x * gridDim.x;

    while (tid < TOTAL_NUMBER_OF_THREADS)
    {
        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(c[0][tid])
            : "r" (a[0][tid]),
              "r" (b[0][tid]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(c[i][tid])
                : "r" (a[i][tid]),
                  "r" (b[i][tid]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(c[BIGNUM_NUMBER_OF_WORDS - 1][tid])
            : "r" (a[BIGNUM_NUMBER_OF_WORDS - 1][tid]),
              "r" (b[BIGNUM_NUMBER_OF_WORDS - 1][tid]));

        tid += tid_increment;
    }
}
