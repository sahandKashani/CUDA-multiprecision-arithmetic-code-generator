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

__global__ void add_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void sub_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void mul_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

__device__ void add(uint32_t* c, uint32_t* a, uint32_t* b, uint32_t tid);
__device__ void sub(uint32_t* c, uint32_t* a, uint32_t* b, uint32_t tid);
__device__ void mul(uint32_t* c, uint32_t* a, uint32_t* b, uint32_t tid);

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// BENCHMARKS /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    add_benchmark(host_c, host_a, host_b);
    sub_benchmark(host_c, host_a, host_b);
    mul_benchmark(host_c, host_a, host_b);
}

void add_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, add_kernel, add_check, "addition");
}

void sub_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, sub_kernel, sub_check, "subtraction");
}

void mul_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b)
{
    binary_operator_benchmark(host_c, host_a, host_b, mul_kernel, mul_check, "multiplication");
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// KERNELS ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void add_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    add(dev_c, dev_a, dev_b, tid);
}

__global__ void sub_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    sub(dev_c, dev_a, dev_b, tid);
}

__global__ void mul_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    mul(dev_c, dev_a, dev_b, tid);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// OPERATIONS /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__device__ void add(uint32_t* c_glo, uint32_t* a_glo, uint32_t* b_glo, uint32_t tid)
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

__device__ void sub(uint32_t* c_glo, uint32_t* a_glo, uint32_t* b_glo, uint32_t tid)
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

__device__ void mul(uint32_t* c_glo, uint32_t* a_glo, uint32_t* b_glo, uint32_t tid)
{
    // ATTENTION: Assuming "a" and "b" are n-bit bignums, their multiplication
    // can give a bignum of length 2n-bits. Since we are coding a generic
    // multiplication, we will use this information to do less loops, so we use
    // MIN_BIGNUM_NUMBER_OF_WORDS to represent "a" and "b", and
    // MAX_BIGNUM_NUMBER_OF_WORDS to represent "c".
    uint32_t a_loc[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b_loc[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c_loc[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t carry[MAX_BIGNUM_NUMBER_OF_WORDS];

    // Example of the schoolbook multiplication algorithm we will use:
    //
    //                                      A[4]   A[3]   A[2]   A[1]   A[0]
    //                                    * B[4]   B[3]   B[2]   B[1]   B[0]
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
    //                                      A[4]   A[3]   A[2]   A[1]   A[0]
    //                                    * B[4]   B[3]   B[2]   B[1]   B[0]
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

    // A = operand 1, B = operand 2, D = carry, C = result
    //
    // mad.lo.u32    C[0], B[0], A[0], 0
    //
    // mad.hi.u32    C[1], B[0], A[1], 0
    // mad.lo.cc.u32 C[1], B[0], A[1], C[1]
    // addc.u32      D[2], D[2], 0
    // mad.lo.cc.u32 C[1], B[1], A[0], C[1]
    // addc.u32      D[2], D[2], 0
    //
    // mad.hi.u32    C[2], B[0], A[1], 0
    // mad.hi.u32    j


    // extended-precision multiply: [C[3],C[2],C[1],C[0]] = [A[1],A[0]] * [B[1],B[0]]
    // mul.lo.u32     C[0], B[0], A[0]      ; // C[0]  = (A[0]*B[0]).[31:0]             , no  carry-out
    // mul.hi.u32     C[1], B[0], A[0]      ; // C[1]  = (A[0]*B[0]).[63:32]            , no  carry-out
    // mad.lo.cc.u32  C[1], B[0], A[1], C[1]; // C[1] += (A[1]*B[0]).[31:0]             , may carry-out
    // madc.hi.u32    C[2], B[0], A[1], 0   ; // C[2]  = (A[1]*B[0]).[63:32] + carry-in , no  carry-out
    // mad.lo.cc.u32  C[1], B[1], A[0], C[1]; // C[1] += (A[0]*B[1]).[31:0]             , may carry-out
    // madc.hi.cc.u32 C[2], B[1], A[0], C[2]; // C[2] += (A[0]*B[1]).[63:32] + carry-in , may carry-out
    // addc.u32       C[3], 0   , 0         ; // C[3]  = carry-in                       , no  carry-out
    // mad.lo.cc.u32  C[2], B[1], A[1], C[2]; // C[2] += (A[1]*B[1]).[31:0]             , may carry-out
    // madc.hi.u32    C[3], B[1], A[1], C[3]; // C[3] += (A[1]*B[1]).[63:32] + carry-in
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

    // execute kernel
    printf("Performing \"%s\" on GPU ... ", operation_name);
    fflush(stdout);

    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

    printf("done\n");
    fflush(stdout);

    // copy results back to host
    cudaError dev_c_memcpy_success = cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

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
