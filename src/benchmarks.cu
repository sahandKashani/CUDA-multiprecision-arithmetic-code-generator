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

void add_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
void sub_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
void mul_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);

__global__ void add_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void add_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void sub_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void sub_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void mul_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

__device__ void add_glo(uint32_t* c_glo, uint32_t* a_glo, uint32_t* b_glo, uint32_t tid);
__device__ void add_loc(uint32_t* c_loc, uint32_t* a_loc, uint32_t* b_loc);
__device__ void sub_glo(uint32_t* c_glo, uint32_t* a_glo, uint32_t* b_glo, uint32_t tid);
__device__ void sub_loc(uint32_t* c_loc, uint32_t* a_loc, uint32_t* b_loc);
__device__ void mul_loc(uint32_t* c_glo, uint32_t* a_glo, uint32_t* b_glo);
__device__ void mul_with_one_word_loc(uint32_t* c_loc, uint32_t* a_loc, uint32_t b_loc, uint32_t shift);

// To remove
__device__ void dev_print_bignum(uint32_t* in)
{
    assert(in != NULL);

    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        printf("%u    ", in[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// BENCHMARKS /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    // add_benchmark(host_c, host_a, host_b);
    // sub_benchmark(host_c, host_a, host_b);
    mul_benchmark(host_c, host_a, host_b);
}

void add_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, add_glo_kernel, add_check, "global addition");
    binary_operator_benchmark(host_c, host_a, host_b, add_loc_kernel, add_check, "local addition");
}

void sub_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, sub_glo_kernel, sub_check, "global subtraction");
    binary_operator_benchmark(host_c, host_a, host_b, sub_loc_kernel, sub_check, "local subtraction");
}

void mul_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, mul_loc_kernel, mul_check, "multiplication");
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// KERNELS ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void add_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    add_glo(dev_c, dev_a, dev_b, tid);
}

__global__ void add_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    add_loc(c, a, b);

    #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

__global__ void sub_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    sub_glo(dev_c, dev_a, dev_b, tid);
}

__global__ void sub_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    sub_loc(c, a, b);

    #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

__global__ void mul_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    mul_loc(c, a, b);

    #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// OPERATIONS /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__device__ void add_glo(uint32_t* c_glo, uint32_t* a_glo, uint32_t* b_glo, uint32_t tid)
{
    asm("add.cc.u32 %0, %1, %2;"
        : "=r"(c_glo[COAL_IDX(0, tid)])
        : "r" (a_glo[COAL_IDX(0, tid)]),
          "r" (b_glo[COAL_IDX(0, tid)]));

    #pragma unroll
    for (uint32_t i = 1; i < MAX_BIGNUM_NUMBER_OF_WORDS - 1; i++)
    {
        asm("addc.cc.u32 %0, %1, %2;"
            : "=r"(c_glo[COAL_IDX(i, tid)])
            : "r" (a_glo[COAL_IDX(i, tid)]),
              "r" (b_glo[COAL_IDX(i, tid)]));
    }

    asm("addc.u32 %0, %1, %2;"
        : "=r"(c_glo[COAL_IDX(MAX_BIGNUM_NUMBER_OF_WORDS - 1, tid)])
        : "r" (a_glo[COAL_IDX(MAX_BIGNUM_NUMBER_OF_WORDS - 1, tid)]),
          "r" (b_glo[COAL_IDX(MAX_BIGNUM_NUMBER_OF_WORDS - 1, tid)]));
}

__device__ void add_loc(uint32_t* c_loc, uint32_t* a_loc, uint32_t* b_loc)
{
    asm("add.cc.u32  %0, %9 , %18;"
        "addc.cc.u32 %1, %10, %19;"
        "addc.cc.u32 %2, %11, %20;"
        "addc.cc.u32 %3, %12, %21;"
        "addc.cc.u32 %4, %13, %22;"
        "addc.cc.u32 %5, %14, %23;"
        "addc.cc.u32 %6, %15, %24;"
        "addc.cc.u32 %7, %16, %25;"
        "addc.u32    %8, %17, %26;"
        : "=r"(c_loc[0]),
          "=r"(c_loc[1]),
          "=r"(c_loc[2]),
          "=r"(c_loc[3]),
          "=r"(c_loc[4]),
          "=r"(c_loc[5]),
          "=r"(c_loc[6]),
          "=r"(c_loc[7]),
          "=r"(c_loc[8])
        : "r" (a_loc[0]),
          "r" (a_loc[1]),
          "r" (a_loc[2]),
          "r" (a_loc[3]),
          "r" (a_loc[4]),
          "r" (a_loc[5]),
          "r" (a_loc[6]),
          "r" (a_loc[7]),
          "r" (a_loc[8]),
          "r" (b_loc[0]),
          "r" (b_loc[1]),
          "r" (b_loc[2]),
          "r" (b_loc[3]),
          "r" (b_loc[4]),
          "r" (b_loc[5]),
          "r" (b_loc[6]),
          "r" (b_loc[7]),
          "r" (b_loc[8])
          );

    // asm("add.cc.u32 %0, %1, %2;"
    //     : "=r"(c_loc[0])
    //     : "r" (a_loc[0]),
    //       "r" (b_loc[0]));

    // #pragma unroll
    // for (uint32_t i = 1; i < MAX_BIGNUM_NUMBER_OF_WORDS - 1; i++)
    // {
    //     asm("addc.cc.u32 %0, %1, %2;"
    //         : "=r"(c_loc[i])
    //         : "r" (a_loc[i]),
    //           "r" (b_loc[i]));
    // }

    // asm("addc.u32 %0, %1, %2;"
    //     : "=r"(c_loc[MAX_BIGNUM_NUMBER_OF_WORDS - 1])
    //     : "r" (a_loc[MAX_BIGNUM_NUMBER_OF_WORDS - 1]),
    //       "r" (b_loc[MAX_BIGNUM_NUMBER_OF_WORDS - 1]));
}

__device__ void sub_glo(uint32_t* c_glo, uint32_t* a_glo, uint32_t* b_glo, uint32_t tid)
{
    asm("sub.cc.u32 %0, %1, %2;"
        : "=r"(c_glo[COAL_IDX(0, tid)])
        : "r" (a_glo[COAL_IDX(0, tid)]),
          "r" (b_glo[COAL_IDX(0, tid)]));

    #pragma unroll
    for (uint32_t i = 1; i < MAX_BIGNUM_NUMBER_OF_WORDS - 1; i++)
    {
        asm("subc.cc.u32 %0, %1, %2;"
            : "=r"(c_glo[COAL_IDX(i, tid)])
            : "r" (a_glo[COAL_IDX(i, tid)]),
              "r" (b_glo[COAL_IDX(i, tid)]));
    }

    asm("subc.u32 %0, %1, %2;"
        : "=r"(c_glo[COAL_IDX(MAX_BIGNUM_NUMBER_OF_WORDS - 1, tid)])
        : "r" (a_glo[COAL_IDX(MAX_BIGNUM_NUMBER_OF_WORDS - 1, tid)]),
          "r" (b_glo[COAL_IDX(MAX_BIGNUM_NUMBER_OF_WORDS - 1, tid)]));
}

__device__ void sub_loc(uint32_t* c_loc, uint32_t* a_loc, uint32_t* b_loc)
{
    asm("sub.cc.u32 %0, %1, %2;"
        : "=r"(c_loc[0])
        : "r" (a_loc[0]),
          "r" (b_loc[0]));

    #pragma unroll
    for (uint32_t i = 1; i < MAX_BIGNUM_NUMBER_OF_WORDS - 1; i++)
    {
        asm("subc.cc.u32 %0, %1, %2;"
            : "=r"(c_loc[i])
            : "r" (a_loc[i]),
              "r" (b_loc[i]));
    }

    asm("subc.u32 %0, %1, %2;"
        : "=r"(c_loc[MAX_BIGNUM_NUMBER_OF_WORDS - 1])
        : "r" (a_loc[MAX_BIGNUM_NUMBER_OF_WORDS - 1]),
          "r" (b_loc[MAX_BIGNUM_NUMBER_OF_WORDS - 1]));
}

__device__ void mul_loc(uint32_t* c_loc, uint32_t* a_loc, uint32_t* b_loc)
{
    // ATTENTION: Assuming "a" and "b" are n-bit bignums, their multiplication
    // can give a bignum of length 2n-bits. Since we are coding a generic
    // multiplication, we will use this information to do less loops, so we use
    // MIN_BIGNUM_NUMBER_OF_WORDS to represent "a" and "b", and
    // MAX_BIGNUM_NUMBER_OF_WORDS to represent "c".

    // Example of the schoolbook multiplication algorithm we will use:
    //
    //                                      A[4]---A[3]---A[2]---A[1]---A[0]
    //                                    * B[4]---B[3]---B[2]---B[1]---B[0]
    // -----------------------------------------------------------------------
    // |      |      |      |      |      |      |      |      | B[0] * A[0] |
    // |      |      |      |      |      |      |      | B[0] * A[1] |      |
    // |      |      |      |      |      |      | B[0] * A[2] |      |      |
    // |      |      |      |      |      | B[0] * A[3] |      |      |      |
    // |      |      |      |      | B[0] * A[4] |      |      |      |      |
    // |      |      |      |      |      |      |      | B[1] * A[0] |      |
    // |      |      |      |      |      |      | B[1] * A[1] |      |      |
    // |      |      |      |      |      | B[1] * A[2] |      |      |      |
    // |      |      |      |      | B[1] * A[3] |      |      |      |      |
    // |      |      |      | B[1] * A[4] |      |      |      |      |      |
    // |      |      |      |      |      |      | B[2] * A[0] |      |      |
    // |      |      |      |      |      | B[2] * A[1] |      |      |      |
    // |      |      |      |      | B[2] * A[2] |      |      |      |      |
    // |      |      |      | B[2] * A[3] |      |      |      |      |      |
    // |      |      | B[2] * A[4] |      |      |      |      |      |      |
    // |      |      |      |      |      | B[3] * A[0] |      |      |      |
    // |      |      |      |      | B[3] * A[1] |      |      |      |      |
    // |      |      |      | B[3] * A[2] |      |      |      |      |      |
    // |      |      | B[3] * A[3] |      |      |      |      |      |      |
    // |      | B[3] * A[4] |      |      |      |      |      |      |      |
    // |      |      |      |      | B[4] * A[0] |      |      |      |      |
    // |      |      |      | B[4] * A[1] |      |      |      |      |      |
    // |      |      | B[4] * A[2] |      |      |      |      |      |      |
    // |      | B[4] * A[3] |      |      |      |      |      |      |      |
    // + B[4] * A[4] |      |      |      |      |      |      |      |      |
    // -----------------------------------------------------------------------
    // | C[9] | C[8] | C[7] | C[6] | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |

    // Because of CUDA carry propagation problems (the carry flag is only kept
    // for the next assembly instruction), we will have to order the steps in
    // the following way:
    //
    //                                      A[4]---A[3]---A[2]---A[1]---A[0]
    //                                    * B[4]---B[3]---B[2]---B[1]---B[0]
    // -----------------------------------------------------------------------
    // |      |      |      |      |      |      |      |      | B[0] * A[0] |
    // |      |      |      |      |      |      |      | B[0] * A[1] |      |
    // |      |      |      |      |      |      |      | B[1] * A[0] |      |
    // |      |      |      |      |      |      | B[0] * A[2] |      |      |
    // |      |      |      |      |      |      | B[1] * A[1] |      |      |
    // |      |      |      |      |      |      | B[2] * A[0] |      |      |
    // |      |      |      |      |      | B[0] * A[3] |      |      |      |
    // |      |      |      |      |      | B[1] * A[2] |      |      |      |
    // |      |      |      |      |      | B[2] * A[1] |      |      |      |
    // |      |      |      |      |      | B[3] * A[0] |      |      |      |
    // |      |      |      |      | B[0] * A[4] |      |      |      |      |
    // |      |      |      |      | B[1] * A[3] |      |      |      |      |
    // |      |      |      |      | B[2] * A[2] |      |      |      |      |
    // |      |      |      |      | B[3] * A[1] |      |      |      |      |
    // |      |      |      |      | B[4] * A[0] |      |      |      |      |
    // |      |      |      | B[1] * A[4] |      |      |      |      |      |
    // |      |      |      | B[2] * A[3] |      |      |      |      |      |
    // |      |      |      | B[3] * A[2] |      |      |      |      |      |
    // |      |      |      | B[4] * A[1] |      |      |      |      |      |
    // |      |      | B[2] * A[4] |      |      |      |      |      |      |
    // |      |      | B[3] * A[3] |      |      |      |      |      |      |
    // |      |      | B[4] * A[2] |      |      |      |      |      |      |
    // |      | B[3] * A[4] |      |      |      |      |      |      |      |
    // |      | B[4] * A[3] |      |      |      |      |      |      |      |
    // + B[4] * A[4] |      |      |      |      |      |      |      |      |
    // -----------------------------------------------------------------------
    // | C[9] | C[8] | C[7] | C[6] | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |

    // But these loops are very difficult to code in a generic way for them to
    // work for any value of MIN_BIGNUM_NUMBER_OF_WORDS and
    // MAX_BIGNUM_NUMBER_OF_WORDS. So we will use the following derivative:

    // Alternatively, we could use our addition algorithm, and do bignum
    // multiplication with bigger chunks, then add the intermediary results in
    // the following:
    //
    //                                      A[4]---A[3]---A[2]---A[1]---A[0]
    //                                    * B[4]---B[3]---B[2]---B[1]---B[0]
    // -----------------------------------------------------------------------
    // |      |      |      |      | A[4]---A[3]---A[2]---A[1]---A[0] * B[0] |
    // |      |      |      | A[4]---A[3]---A[2]---A[1]---A[0] * B[1] |      |
    // |      |      | A[4]---A[3]---A[2]---A[1]---A[0] * B[2] |      |      |
    // |      | A[4]---A[3]---A[2]---A[1]---A[0] * B[3] |      |      |      |
    // + A[4]---A[3]---A[2]---A[1]---A[0] * B[4] |      |      |      |      |
    // -----------------------------------------------------------------------
    // | C[9] | C[8] | C[7] | C[6] | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |

    // we are multiplying 2 bignums, each of which are held on
    // MAX_BIGNUM_NUMBER_OF_WORDS, but we know the data is actually on
    // MIN_BIGNUM_NUMBER_OF_WORDS.

    uint32_t tmp[MAX_BIGNUM_NUMBER_OF_WORDS];
    // mul_with_one_word_loc(c_loc, a_loc, b_loc[0], 0);

    #pragma unroll
    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        mul_with_one_word_loc(tmp, a_loc, b_loc[i], i);
        add_loc(c_loc, c_loc, tmp);
    }
}

__device__ void mul_with_one_word_loc(uint32_t* c_loc, uint32_t* a_loc, uint32_t b_loc, uint32_t shift)
{
    // Example algorithm execution on 5-word bignums with shift = 0:
    //
    //                                      A[4]---A[3]---A[2]---A[1]---A[0]
    //                                    *                               B
    // -----------------------------------------------------------------------
    // |      |      |      |      |      |      |      |      |   B  * A[0] |
    // |      |      |      |      |      |      |      |   B  * A[1] |      |
    // |      |      |      |      |      |      |   B  * A[2] |      |      |
    // |      |      |      |      |      |   B  * A[3] |      |      |      |
    // |      |      |      |      +   B  * A[4] |      |      |      |      |
    // -----------------------------------------------------------------------
    // |   0  |   0  |   0  |   0  | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |

    // Example algorithm execution on 5-word bignums with shift = 2
    //
    //                                      A[4]---A[3]---A[2]---A[1]---A[0]
    //                                    *                 B ---  0 ---  0
    // -----------------------------------------------------------------------
    // |      |      |      |      |      |      |   B  * A[0] |      |      |
    // |      |      |      |      |      |   B  * A[1] |      |      |      |
    // |      |      |      |      |   B  * A[2] |      |      |      |      |
    // |      |      |      |   B  * A[3] |      |      |      |      |      |
    // |      |      +   B  * A[4] |      |      |      |      |      |      |
    // -----------------------------------------------------------------------
    // |   0  |   0  | C[7] | C[6] | C[5] | C[4] | C[3] | C[2] |   0  |   0  |

    // Example assembly for 5-word bignum multiplication:
    //
    // mad.lo.u32    C[shift]    , B, A[0], 0;
    // mad.hi.u32    C[shift + 1], B, A[0], 0;
    //
    // mad.lo.cc.u32 C[shift + 1], B, A[1], C[shift + 1];
    // madc.hi.u32   C[shift + 2], B, A[1], 0;
    //
    // mad.lo.cc.u32 C[shift + 2], B, A[2], C[shift + 2];
    // madc.hi.u32   C[shift + 3], B, A[2], 0;
    //
    // mad.lo.cc.u32 C[shift + 3], B, A[3], C[shift + 3];
    // madc.hi.u32   C[shift + 4], B, A[3], 0;
    //
    // mad.lo.cc.u32 C[shift + 4], B, A[4], C[shift + 4];
    // madc.hi.u32   C[shift + 5], B, A[4], 0;

    // set leading and trailing values of c_loc to 0
    #pragma unroll
    for (uint32_t i = 0; i < shift; i++)
    {
        c_loc[i] = 0;
        c_loc[MAX_BIGNUM_NUMBER_OF_WORDS - i] = 0;
    }

    asm("mad.lo.u32    %0, %6, %7 ,  0;"
        "mad.hi.u32    %1, %6, %7 ,  0;"
        "mad.lo.cc.u32 %1, %6, %8 , %1;"
        "madc.hi.u32   %2, %6, %8 ,  0;"
        "mad.lo.cc.u32 %2, %6, %9 , %2;"
        "madc.hi.u32   %3, %6, %9 ,  0;"
        "mad.lo.cc.u32 %3, %6, %10, %3;"
        "madc.hi.u32   %4, %6, %10,  0;"
        "mad.lo.cc.u32 %4, %6, %11, %4;"
        "madc.hi.u32   %5, %6, %11,  0;"
        : "=r"(c_loc[shift]),
          "=r"(c_loc[shift + 1]),
          "=r"(c_loc[shift + 2]),
          "=r"(c_loc[shift + 3]),
          "=r"(c_loc[shift + 4]),
          "=r"(c_loc[shift + 5])
        : "r" (b_loc),
          "r" (a_loc[0]),
          "r" (a_loc[1]),
          "r" (a_loc[2]),
          "r" (a_loc[3]),
          "r" (a_loc[4]));

    // // ATTENTION: here we loop until MIN_BIGNUM_NUMBER_OF_WORDS (included),
    // // because we are trying to optimize the number of loops we use. We can do
    // // this because we know the actual bit-length of the operands.
    // #pragma unroll
    // for (uint32_t i = 1; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    // {
    //     asm("mad.lo.cc.u32 %0, %1, %2, %0;"
    //         : "=r"(c_loc[shift + i])
    //         : "r" (b_loc),
    //           "r" (a_loc[i]));

    //     asm("madc.hi.u32 %0, %1, %2, 0;"
    //         : "=r"(c_loc[shift + i + 1])
    //         : "r" (b_loc),
    //           "r" (a_loc[i]));
    // }
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
    cudaError dev_a_malloc_success = cudaMalloc((void**) &dev_a, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_malloc_success = cudaMalloc((void**) &dev_b, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_malloc_success = cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    assert(dev_a_malloc_success == cudaSuccess);
    assert(dev_b_malloc_success == cudaSuccess);
    assert(dev_c_malloc_success == cudaSuccess);

    // copy operands to device memory
    cudaError dev_a_memcpy_succes = cudaMemcpy(dev_a, host_a, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError dev_b_memcpy_succes = cudaMemcpy(dev_b, host_b, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    assert(dev_a_memcpy_succes == cudaSuccess);
    assert(dev_b_memcpy_succes == cudaSuccess);

    // set result values to 0
    cudaError dev_c_memset_success = cudaMemset(dev_c, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    assert(dev_c_memset_success == cudaSuccess);

    // execute kernel
    printf("Performing \"%s\" on GPU ... ", operation_name);
    fflush(stdout);

    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

    printf("done\n");
    fflush(stdout);

    // copy results back to host
    cudaError dev_c_memcpy_success = cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    assert(dev_c_memcpy_success == cudaSuccess);

    // set all values to 0 before freeing
    cudaError dev_a_cleanup_memset_success = cudaMemset(dev_a, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_cleanup_memset_success = cudaMemset(dev_b, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_cleanup_memset_success = cudaMemset(dev_c, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    assert(dev_a_cleanup_memset_success == cudaSuccess);
    assert(dev_b_cleanup_memset_success == cudaSuccess);
    assert(dev_c_cleanup_memset_success == cudaSuccess);

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
