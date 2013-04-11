#include "memory_layout_benchmarks.cuh"
#include "bignum_types.h"
#include "bignum_conversions.h"
#include "constants.h"
#include <stdint.h>

// void normal_memory_layout_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid)
// {
//     // device operands (dev_a, dev_b) and results (dev_c)
//     bignum* dev_a;
//     bignum* dev_b;
//     bignum* dev_c;

//     // allocate gpu memory
//     cudaMalloc((void**) &dev_a, NUMBER_OF_BIGNUMS * sizeof(bignum));
//     cudaMalloc((void**) &dev_b, NUMBER_OF_BIGNUMS * sizeof(bignum));
//     cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * sizeof(bignum));

//     // copy operands to device memory
//     cudaMemcpy(dev_a, *host_a, NUMBER_OF_BIGNUMS * sizeof(bignum), cudaMemcpyHostToDevice);
//     cudaMemcpy(dev_b, *host_b, NUMBER_OF_BIGNUMS * sizeof(bignum), cudaMemcpyHostToDevice);

//     // execute addition
//     normal_addition<<<blocks_per_grid, threads_per_block>>>(dev_c, dev_a, dev_b);

//     // copy results back to host
//     cudaMemcpy(*host_c, dev_c, NUMBER_OF_BIGNUMS * sizeof(bignum), cudaMemcpyDeviceToHost);

//     // free device memory
//     cudaFree(dev_a);
//     cudaFree(dev_b);
//     cudaFree(dev_c);
// }

// __global__ void normal_addition(bignum* c, bignum* a, bignum* b)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {

//         asm("add.cc.u32 %0, %1, %2;"
//             : "=r"(c[tid][0])
//             : "r" (a[tid][0]),
//               "r" (b[tid][0]));

//         #pragma unroll
//         for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
//         {
//             asm("addc.cc.u32 %0, %1, %2;"
//                 : "=r"(c[tid][i])
//                 : "r" (a[tid][i]),
//                   "r" (b[tid][i]));
//         }

//         asm("addc.u32 %0, %1, %2;"
//             : "=r"(c[tid][BIGNUM_NUMBER_OF_WORDS - 1])
//             : "r" (a[tid][BIGNUM_NUMBER_OF_WORDS - 1]),
//               "r" (b[tid][BIGNUM_NUMBER_OF_WORDS - 1]));

//         tid += stride;
//     }
// }

// void interleaved_memory_layout_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid)
// {
//     // arrange data in interleaved form
//     interleaved_bignum* host_ops = bignums_to_interleaved_bignum(host_a, host_b);

//     // device operands (dev_ops) and results (dev_c)
//     interleaved_bignum* dev_ops;
//     bignum* dev_c;

//     // allocate gpu memory
//     cudaMalloc((void**) &dev_ops, NUMBER_OF_BIGNUMS * sizeof(interleaved_bignum));
//     cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * sizeof(bignum));

//     // copy operands to device memory
//     cudaMemcpy(dev_ops, host_ops, NUMBER_OF_BIGNUMS * sizeof(interleaved_bignum), cudaMemcpyHostToDevice);

//     // execute addition
//     interleaved_addition<<<blocks_per_grid, threads_per_block>>>(dev_c, dev_ops);

//     // copy results back to host
//     cudaMemcpy(*host_c, dev_c, NUMBER_OF_BIGNUMS * sizeof(bignum), cudaMemcpyDeviceToHost);

//     // re-arrange data in non-interleaved form
//     interleaved_bignum_to_bignums(host_a, host_b, &host_ops);

//     // free device memory
//     cudaFree(dev_ops);
//     cudaFree(dev_c);
// }

// __global__ void interleaved_addition(bignum* c, interleaved_bignum* ops)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {
//         uint32_t i = 0;
//         uint32_t col = 2 * i;

//         asm("add.cc.u32 %0, %1, %2;"
//             : "=r"(c[tid][i])
//             : "r" (ops[tid][col]),
//               "r" (ops[tid][col + 1]));

//         #pragma unroll
//         for (i = 1, col = 2 * i; i < BIGNUM_NUMBER_OF_WORDS - 1; i++, col = 2 * i)
//         {
//             asm("addc.cc.u32 %0, %1, %2;"
//                 : "=r"(c[tid][i])
//                 : "r" (ops[tid][col]),
//                   "r" (ops[tid][col + 1]));
//         }

//         asm("addc.u32 %0, %1, %2;"
//             : "=r"(c[tid][BIGNUM_NUMBER_OF_WORDS - 1])
//             : "r" (ops[tid][col]),
//               "r" (ops[tid][col + 1]));

//         tid += stride;
//     }
// }

void coalesced_normal_memory_layout_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid)
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
    cudaMalloc((void**) &dev_c_a, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
    cudaMalloc((void**) &dev_c_b, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
    cudaMalloc((void**) &dev_c_c, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));

    // copy operands to device memory
    cudaMemcpy(dev_c_a, host_c_a, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c_b, host_c_b, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum), cudaMemcpyHostToDevice);

    // execute addition
    coalesced_normal_addition<<<blocks_per_grid, threads_per_block>>>(dev_c_c, dev_c_a, dev_c_b);

    // copy results back to host
    cudaMemcpy(host_c_c, dev_c_c, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum), cudaMemcpyDeviceToHost);

    // put data back to non-coalesced form
    *host_a = coalesced_bignum_to_bignum(&host_c_a);
    *host_b = coalesced_bignum_to_bignum(&host_c_b);
    *host_c = coalesced_bignum_to_bignum(&host_c_c);

    // free device memory
    cudaFree(dev_c_a);
    cudaFree(dev_c_b);
    cudaFree(dev_c_c);
}

__global__ void coalesced_normal_addition(coalesced_bignum* c, coalesced_bignum* a, coalesced_bignum* b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    while (tid < NUMBER_OF_BIGNUMS)
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

        tid += stride;
    }
}

// void coalesced_interleaved_memory_layout_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid)
// {
//     // arrange data in coalesced interleaved form
//     coalesced_interleaved_bignum* host_c_ops = bignums_to_coalesced_interleaved_bignum(host_a, host_b);
//     coalesced_bignum* host_c_c = bignum_to_coalesced_bignum(host_c);

//     // device operands (dev_c_ops) and results (dev_c_c)
//     coalesced_interleaved_bignum* dev_c_ops;
//     coalesced_bignum* dev_c_c;

//     // allocate gpu memory
//     cudaMalloc((void**) &dev_c_ops, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_interleaved_bignum));
//     cudaMalloc((void**) &dev_c_c, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));

//     // copy operands to device memory
//     cudaMemcpy(dev_c_ops, host_c_ops, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_interleaved_bignum), cudaMemcpyHostToDevice);

//     // execute addition
//     coalesced_interleaved_addition<<<blocks_per_grid, threads_per_block>>>(dev_c_c, dev_c_ops);

//     // copy results back to host
//     cudaMemcpy(host_c_c, dev_c_c, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum), cudaMemcpyDeviceToHost);

//     // rearrange data back to non-coalesced form
//     coalesced_interleaved_bignum_to_bignums(host_a, host_b, &host_c_ops);
//     *host_c = coalesced_bignum_to_bignum(&host_c_c);

//     // free device memory
//     cudaFree(dev_c_ops);
//     cudaFree(dev_c_c);
// }

// __global__ void coalesced_interleaved_addition(coalesced_bignum* c, coalesced_interleaved_bignum* ops)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {
//         uint32_t i = 0;
//         uint32_t col = 2 * tid;

//         asm("add.cc.u32 %0, %1, %2;"
//             : "=r"(c[i][tid])
//             : "r" (ops[i][col]),
//               "r" (ops[i][col + 1]));

//         #pragma unroll
//         for (i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
//         {
//             asm("addc.cc.u32 %0, %1, %2;"
//                 : "=r"(c[i][tid])
//                 : "r" (ops[i][col]),
//                   "r" (ops[i][col + 1]));
//         }

//         asm("addc.u32 %0, %1, %2;"
//             : "=r"(c[i][tid])
//             : "r" (ops[i][col]),
//               "r" (ops[i][col + 1]));

//         tid += stride;
//     }
// }

// void coalesced_normal_memory_layout_with_local_memory_benchmark(bignum** host_c, bignum** host_a, bignum** host_b, uint32_t threads_per_block, uint32_t blocks_per_grid)
// {
//     // arrange data in coalesced form
//     coalesced_bignum* host_c_a = bignum_to_coalesced_bignum(host_a);
//     coalesced_bignum* host_c_b = bignum_to_coalesced_bignum(host_b);
//     coalesced_bignum* host_c_c = bignum_to_coalesced_bignum(host_c);

//     // device operands (dev_c_a, dev_c_b) and results (dev_c_c)
//     coalesced_bignum* dev_c_a;
//     coalesced_bignum* dev_c_b;
//     coalesced_bignum* dev_c_c;

//     // allocate gpu memory
//     cudaMalloc((void**) &dev_c_a, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
//     cudaMalloc((void**) &dev_c_b, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
//     cudaMalloc((void**) &dev_c_c, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));

//     // copy operands to device memory
//     cudaMemcpy(dev_c_a, host_c_a, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum), cudaMemcpyHostToDevice);
//     cudaMemcpy(dev_c_b, host_c_b, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum), cudaMemcpyHostToDevice);

//     // execute addition
//     coalesced_normal_addition_with_local_memory<<<blocks_per_grid, threads_per_block>>>(dev_c_c, dev_c_a, dev_c_b);

//     // copy results back to host
//     cudaMemcpy(host_c_c, dev_c_c, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum), cudaMemcpyDeviceToHost);

//     // put data back to non-coalesced form
//     *host_a = coalesced_bignum_to_bignum(&host_c_a);
//     *host_b = coalesced_bignum_to_bignum(&host_c_b);
//     *host_c = coalesced_bignum_to_bignum(&host_c_c);

//     // free device memory
//     cudaFree(dev_c_a);
//     cudaFree(dev_c_b);
//     cudaFree(dev_c_c);
// }

// __global__ void coalesced_normal_addition_with_local_memory(coalesced_bignum* c, coalesced_bignum* a, coalesced_bignum* b)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     bignum local_a;
//     bignum local_b;
//     bignum local_c;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {
//         // read the thread's operands to per-thread local memory in advance as a
//         // form of prefetching.
//         for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
//         {
//             local_a[i] = a[i][tid];
//             local_b[i] = b[i][tid];
//         }

//         asm("add.cc.u32 %0, %1, %2;"
//             : "=r"(local_c[0])
//             : "r" (local_a[0]),
//               "r" (local_b[0]));

//         #pragma unroll
//         for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
//         {
//             asm("addc.cc.u32 %0, %1, %2;"
//                 : "=r"(local_c[i])
//                 : "r" (local_a[i]),
//                   "r" (local_b[i]));
//         }

//         asm("addc.u32 %0, %1, %2;"
//             : "=r"(local_c[BIGNUM_NUMBER_OF_WORDS - 1])
//             : "r" (local_a[BIGNUM_NUMBER_OF_WORDS - 1]),
//               "r" (local_b[BIGNUM_NUMBER_OF_WORDS - 1]));

//         // store the thread's result from per-thread local memory to global
//         // memory once all calculations are finished.
//         for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
//         {
//             c[i][tid] = local_c[i];
//         }

//         tid += stride;
//     }
// }

// void andrea_hardcoded_test(uint32_t threads_per_block, uint32_t blocks_per_grid)
// {
//     coalesced_bignum* dev_c;
//     cudaMalloc((void**) &dev_c, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
//     andrea_hardcoded_kernel<<<threads_per_block, blocks_per_grid>>>(dev_c);
//     cudaFree(dev_c);
// }

// __global__ void andrea_hardcoded_kernel(coalesced_bignum* c)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {
//         asm("add.cc.u32 %0, %0, 3;"
//             : "+r"(c[0][tid])
//             :);

//         #pragma unroll
//         for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
//         {
//             asm("addc.cc.u32 %0, %0, 0;"
//                 : "+r"(c[i][tid])
//                 :);
//         }

//         asm("addc.u32 %0, %0, 0;"
//             : "+r"(c[BIGNUM_NUMBER_OF_WORDS - 1][tid])
//             :);

//         tid += stride;
//     }
// }

// void andrea_hardcoded_local_test(uint32_t threads_per_block, uint32_t blocks_per_grid)
// {
//     coalesced_bignum* dev_c;
//     cudaMalloc((void**) &dev_c, BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));
//     andrea_hardcoded_local_kernel<<<threads_per_block, blocks_per_grid>>>(dev_c);
//     cudaFree(dev_c);
// }

// __global__ void andrea_hardcoded_local_kernel(coalesced_bignum* c)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     bignum local_c;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {
//         // read the thread's operands to per-thread local memory in advance as a
//         // form of prefetching.
//         for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
//         {
//             local_c[i] = c[i][tid];
//         }

//         asm("add.cc.u32 %0, %0, 3;"
//             : "+r"(local_c[0])
//             :);

//         #pragma unroll
//         for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
//         {
//             asm("addc.cc.u32 %0, %0, 3;"
//                 : "+r"(local_c[i])
//                 :);
//         }

//         asm("addc.u32 %0, %0, 3;"
//             : "+r"(local_c[BIGNUM_NUMBER_OF_WORDS - 1])
//             :);

//         // store the thread's result from per-thread local memory to global
//         // memory once all calculations are finished.
//         for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
//         {
//             c[i][tid] = local_c[i];
//         }

//         tid += stride;
//     }
// }

// void coalesced_normal_memory_layout_with_cudaMallocPitch(bignum** host_c,
//                                                          bignum** host_a,
//                                                          bignum** host_b,
//                                                          uint32_t threads_per_block,
//                                                          uint32_t blocks_per_grid)
// {
//     // arrange data in coalesced form
//     coalesced_bignum* host_c_a = bignum_to_coalesced_bignum(host_a);
//     coalesced_bignum* host_c_b = bignum_to_coalesced_bignum(host_b);
//     coalesced_bignum* host_c_c = bignum_to_coalesced_bignum(host_c);

//     // device operands (dev_c_a, dev_c_b) and results (dev_c_c)
//     coalesced_bignum* dev_c_a;
//     coalesced_bignum* dev_c_b;
//     coalesced_bignum* dev_c_c;

//     // allocate gpu memory
//     size_t pitch_a;
//     size_t pitch_b;
//     size_t pitch_c;

//     cudaMallocPitch((void**) &dev_c_a, &pitch_a, sizeof(coalesced_bignum), BIGNUM_NUMBER_OF_WORDS);
//     cudaMallocPitch((void**) &dev_c_b, &pitch_b, sizeof(coalesced_bignum), BIGNUM_NUMBER_OF_WORDS);
//     cudaMallocPitch((void**) &dev_c_c, &pitch_c, sizeof(coalesced_bignum), BIGNUM_NUMBER_OF_WORDS);

//     // copy operands to device memory
//     cudaMemcpy2D(dev_c_a, pitch_a, host_c_a, sizeof(coalesced_bignum), sizeof(coalesced_bignum), BIGNUM_NUMBER_OF_WORDS, cudaMemcpyHostToDevice);
//     cudaMemcpy2D(dev_c_b, pitch_b, host_c_b, sizeof(coalesced_bignum), sizeof(coalesced_bignum), BIGNUM_NUMBER_OF_WORDS, cudaMemcpyHostToDevice);

//     // execute addition
//     coalesced_normal_addition_with_cudaMallocPitch<<<blocks_per_grid, threads_per_block>>>(dev_c_c, dev_c_a, dev_c_b, pitch_c, pitch_a, pitch_b);

//     // copy results back to host
//     cudaMemcpy2D(host_c_c, sizeof(coalesced_bignum), dev_c_c, pitch_c, sizeof(coalesced_bignum), BIGNUM_NUMBER_OF_WORDS, cudaMemcpyDeviceToHost);

//     // put data back to non-coalesced form
//     *host_a = coalesced_bignum_to_bignum(&host_c_a);
//     *host_b = coalesced_bignum_to_bignum(&host_c_b);
//     *host_c = coalesced_bignum_to_bignum(&host_c_c);

//     // free device memory
//     cudaFree(dev_c_a);
//     cudaFree(dev_c_b);
//     cudaFree(dev_c_c);
// }

// __device__ uint32_t* get_element_from_coalesced_bignum_array(uint32_t* base_address,
//                                                             uint32_t row,
//                                                             uint32_t col,
//                                                             size_t pitch)
// {
//     uint32_t* element = (uint32_t*) (((char*) base_address + row * pitch) + col);
//     return element;
// }

// __global__ void coalesced_normal_addition_with_cudaMallocPitch(coalesced_bignum* c,
//                                                                coalesced_bignum* a,
//                                                                coalesced_bignum* b,
//                                                                size_t pitch_c,
//                                                                size_t pitch_a,
//                                                                size_t pitch_b)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;

//     while (tid < NUMBER_OF_BIGNUMS)
//     {
//         asm("add.cc.u32 %0, %1, %2;"
//             : "=r"(get_element_from_coalesced_bignum_array(c, 0, tid, pitch_c))
//             : "r" (get_element_from_coalesced_bignum_array(a, 0, tid, pitch_a)),
//               "r" (get_element_from_coalesced_bignum_array(b, 0, tid, pitch_b)));

//         #pragma unroll
//         for (uint32_t i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
//         {
//             asm("addc.cc.u32 %0, %1, %2;"
//                 : "=r"(get_element_from_coalesced_bignum_array(c, i, tid, pitch_c))
//                 : "r" (get_element_from_coalesced_bignum_array(a, i, tid, pitch_a)),
//                   "r" (get_element_from_coalesced_bignum_array(b, i, tid, pitch_b)));
//         }

//         asm("addc.u32 %0, %1, %2;"
//             : "=r"(get_element_from_coalesced_bignum_array(c, BIGNUM_NUMBER_OF_WORDS - 1, tid, pitch_c))
//             : "r" (get_element_from_coalesced_bignum_array(a, BIGNUM_NUMBER_OF_WORDS - 1, tid, pitch_a)),
//               "r" (get_element_from_coalesced_bignum_array(b, BIGNUM_NUMBER_OF_WORDS - 1, tid, pitch_b)));

//         tid += stride;
//     }
// }
