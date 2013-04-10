#include "memory_layout_benchmarks.cuh"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include "constants.h"
#include <stdint.h>

void normal_memory_layout_benchmark(bignum** host_c,
                                    bignum** host_a,
                                    bignum** host_b,
                                    uint32_t threads_per_block,
                                    uint32_t blocks_per_grid)
{
    // device operands (dev_a, dev_b) and results (dev_c)
    bignum* dev_a;
    bignum* dev_b;
    bignum* dev_c;

    // allocate gpu memory
    cudaMalloc((void**) &dev_a, TOTAL_NUMBER_OF_THREADS * sizeof(bignum));
    cudaMalloc((void**) &dev_b, TOTAL_NUMBER_OF_THREADS * sizeof(bignum));
    cudaMalloc((void**) &dev_c, TOTAL_NUMBER_OF_THREADS * sizeof(bignum));

    // copy operands to device memory
    cudaMemcpy(dev_a, *host_a, TOTAL_NUMBER_OF_THREADS * sizeof(bignum),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, *host_b, TOTAL_NUMBER_OF_THREADS * sizeof(bignum),
               cudaMemcpyHostToDevice);

    // execute addition
    normal_addition<<<blocks_per_grid, threads_per_block>>>(dev_c, dev_a, dev_b);

    // copy results back to host
    cudaMemcpy(*host_c, dev_c, TOTAL_NUMBER_OF_THREADS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ void normal_addition(bignum* c, bignum* a, bignum* b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid_increment = blockDim.x * gridDim.x;

    while (tid < TOTAL_NUMBER_OF_THREADS)
    {

        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(c[tid][0])
            : "r" (a[tid][0]),
              "r" (b[tid][0]));

        #pragma unroll
        for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(c[tid][i])
                : "r" (a[tid][i]),
                  "r" (b[tid][i]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(c[tid][BIGNUM_NUMBER_OF_WORDS - 1])
            : "r" (a[tid][BIGNUM_NUMBER_OF_WORDS - 1]),
              "r" (b[tid][BIGNUM_NUMBER_OF_WORDS - 1]));

        tid += tid_increment;
    }
}

void interleaved_memory_layout_benchmark(bignum** host_c,
                                         bignum** host_a,
                                         bignum** host_b,
                                         uint32_t threads_per_block,
                                         uint32_t blocks_per_grid)
{
    // arrange data in interleaved form
    interleaved_bignum* host_ops = bignums_to_interleaved_bignum(host_a, host_b);

    // device operands (dev_ops) and results (dev_c)
    interleaved_bignum* dev_ops;
    bignum* dev_c;

    // allocate gpu memory
    cudaMalloc((void**) &dev_ops,
               TOTAL_NUMBER_OF_THREADS * sizeof(interleaved_bignum));
    cudaMalloc((void**) &dev_c, TOTAL_NUMBER_OF_THREADS * sizeof(bignum));

    // copy operands to device memory
    cudaMemcpy(dev_ops, host_ops,
               TOTAL_NUMBER_OF_THREADS * sizeof(interleaved_bignum),
               cudaMemcpyHostToDevice);

    // execute addition
    interleaved_addition<<<blocks_per_grid, threads_per_block>>>(dev_c, dev_ops);

    // copy results back to host
    cudaMemcpy(*host_c, dev_c, TOTAL_NUMBER_OF_THREADS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // re-arrange data in non-interleaved form
    interleaved_bignum_to_bignums(host_a, host_b, &host_ops);

    // free device memory
    cudaFree(dev_ops);
    cudaFree(dev_c);
}

__global__ void interleaved_addition(bignum* c, interleaved_bignum* ops)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid_increment = blockDim.x * gridDim.x;

    while (tid < TOTAL_NUMBER_OF_THREADS)
    {
        uint32_t i = 0;
        uint32_t col = 2 * i;

        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(c[tid][i])
            : "r" (ops[tid][col]),
              "r" (ops[tid][col + 1]));

        #pragma unroll
        for (i = 1, col = 2 * i; i < BIGNUM_NUMBER_OF_WORDS - 1; i++, col = 2 * i)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(c[tid][i])
                : "r" (ops[tid][col]),
                  "r" (ops[tid][col + 1]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(c[tid][BIGNUM_NUMBER_OF_WORDS - 1])
            : "r" (ops[tid][col]),
              "r" (ops[tid][col + 1]));

        tid += tid_increment;
    }
}

void coalesced_normal_memory_layout_benchmark(bignum** host_c,
                                              bignum** host_a,
                                              bignum** host_b,
                                              uint32_t threads_per_block,
                                              uint32_t blocks_per_grid)
{
    // arrange data in coalesced form
    coalesced_bignum* host_c_a = bignum_to_coalesced_bignum(host_a);
    coalesced_bignum* host_c_b = bignum_to_coalesced_bignum(host_b);
    coalesced_bignum* host_c_c = bignum_to_coalesced_bignum(host_c);

    // device operands (dev_c_a, dev_c_b) and results (dev_c_c)
    coalesced_bignum* dev_c_a;
    coalesced_bignum* dev_c_b;
    coalesced_bignum* dev_c_c;

    // allocate gpu memory
    cudaMalloc((void**) &dev_c_a,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
    cudaMalloc((void**) &dev_c_b,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
    cudaMalloc((void**) &dev_c_c,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));

    // copy operands to device memory
    cudaMemcpy(dev_c_a, host_c_a,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c_b, host_c_b,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyHostToDevice);

    // execute addition
    coalesced_normal_addition<<<blocks_per_grid, threads_per_block>>>(
        dev_c_c, dev_c_a, dev_c_b);

    // copy results back to host
    cudaMemcpy(host_c_c, dev_c_c,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyDeviceToHost);

    // put data back to non-coalesced form
    *host_a = coalesced_bignum_to_bignum(&host_c_a);
    *host_b = coalesced_bignum_to_bignum(&host_c_b);
    *host_c = coalesced_bignum_to_bignum(&host_c_c);

    // free device memory
    cudaFree(dev_c_a);
    cudaFree(dev_c_b);
    cudaFree(dev_c_c);
}

__global__ void coalesced_normal_addition(coalesced_bignum* c,
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

void coalesced_interleaved_memory_layout_benchmark(bignum** host_c,
                                                   bignum** host_a,
                                                   bignum** host_b,
                                                   uint32_t threads_per_block,
                                                   uint32_t blocks_per_grid)
{
    // arrange data in coalesced interleaved form
    coalesced_interleaved_bignum* host_c_ops =
        bignums_to_coalesced_interleaved_bignum(host_a, host_b);
    coalesced_bignum* host_c_c = bignum_to_coalesced_bignum(host_c);

    // device operands (dev_c_ops) and results (dev_c_c)
    coalesced_interleaved_bignum* dev_c_ops;
    coalesced_bignum* dev_c_c;

    // allocate gpu memory
    cudaMalloc((void**) &dev_c_ops,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_interleaved_bignum));
    cudaMalloc((void**) &dev_c_c,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));

    // copy operands to device memory
    cudaMemcpy(dev_c_ops, host_c_ops,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_interleaved_bignum),
               cudaMemcpyHostToDevice);

    // execute addition
    coalesced_interleaved_addition<<<blocks_per_grid, threads_per_block>>>(
        dev_c_c, dev_c_ops);

    // copy results back to host
    cudaMemcpy(host_c_c, dev_c_c,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyDeviceToHost);

    // rearrange data back to non-coalesced form
    coalesced_interleaved_bignum_to_bignums(host_a, host_b, &host_c_ops);
    *host_c = coalesced_bignum_to_bignum(&host_c_c);

    // free device memory
    cudaFree(dev_c_ops);
    cudaFree(dev_c_c);
}

__global__ void coalesced_interleaved_addition(coalesced_bignum* c,
                                               coalesced_interleaved_bignum* ops)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid_increment = blockDim.x * gridDim.x;

    while (tid < TOTAL_NUMBER_OF_THREADS)
    {
        uint32_t i = 0;
        uint32_t col = 2 * tid;

        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(c[i][tid])
            : "r" (ops[i][col]),
              "r" (ops[i][col + 1]));

        #pragma unroll
        for (i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(c[i][tid])
                : "r" (ops[i][col]),
                  "r" (ops[i][col + 1]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(c[i][tid])
            : "r" (ops[i][col]),
              "r" (ops[i][col + 1]));

        tid += tid_increment;
    }
}
