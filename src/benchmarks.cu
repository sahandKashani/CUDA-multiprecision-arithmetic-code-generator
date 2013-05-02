#include "benchmarks.h"
#include "bignum_types.h"
#include "input_output.h"
#include "operations.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), char* operation_name);

void add_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name);
void sub_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name);
void mul_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name);

__global__ void add_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void add_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void sub_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void sub_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void mul_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void mul_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// BENCHMARKS /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void add_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name)
{
    binary_operator_benchmark(host_c, host_a, host_b, add_glo_kernel, "add_glo");
    binary_operator_benchmark(host_c, host_a, host_b, add_loc_kernel, "add_loc");

    write_coalesced_bignums_to_file(output_file_name, host_c);
}

void sub_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name)
{
    binary_operator_benchmark(host_c, host_a, host_b, sub_glo_kernel, "sub_glo");
    binary_operator_benchmark(host_c, host_a, host_b, sub_loc_kernel, "sub_loc");

    write_coalesced_bignums_to_file(output_file_name, host_c);
}

void mul_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name)
{
    binary_operator_benchmark(host_c, host_a, host_b, mul_glo_kernel, "mul_glo");
    binary_operator_benchmark(host_c, host_a, host_b, mul_loc_kernel, "mul_loc");

    write_coalesced_bignums_to_file(output_file_name, host_c);
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// KERNELS ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void add_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // #pragma unroll
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        add_glo(dev_c, dev_a, dev_b, tid);
    }
}

__global__ void add_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    // #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    // #pragma unroll
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        add_loc(c, a, b);
    }

    // #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

__global__ void sub_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // #pragma unroll
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        sub_glo(dev_c, dev_a, dev_b, tid);
    }
}

__global__ void sub_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    // #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    // #pragma unroll
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        sub_loc(c, a, b);
    }

    // #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

__global__ void mul_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // #pragma unroll
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        mul_glo(dev_c, dev_a, dev_b, tid);
    }
}

__global__ void mul_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    // #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    // #pragma unroll
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        mul_loc(c, a, b);
    }

    // #pragma unroll
    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////// GENERIC LAUNCH CONFIGURATIONS ///////////////////////
////////////////////////////////////////////////////////////////////////////////

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), char* operation_name)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);
    assert(kernel != NULL);
    assert(operation_name != NULL);

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

    // make sure gpu memory is clean before our calculations (you never know ...)
    cudaError dev_a_cleanup_memset_success = cudaMemset(dev_a, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_cleanup_memset_success = cudaMemset(dev_b, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_cleanup_memset_success = cudaMemset(dev_c, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    assert(dev_a_cleanup_memset_success == cudaSuccess);
    assert(dev_b_cleanup_memset_success == cudaSuccess);
    assert(dev_c_cleanup_memset_success == cudaSuccess);

    // copy operands to device memory
    cudaError dev_a_memcpy_succes = cudaMemcpy(dev_a, host_a, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError dev_b_memcpy_succes = cudaMemcpy(dev_b, host_b, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    assert(dev_a_memcpy_succes == cudaSuccess);
    assert(dev_b_memcpy_succes == cudaSuccess);

    printf("Benchmarking \"%s\" on GPU ... ", operation_name);
    fflush(stdout);

    // execute kernel
    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

    printf("done\n");
    fflush(stdout);

    // copy results back to host
    cudaError dev_c_memcpy_success = cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    assert(dev_c_memcpy_success == cudaSuccess);

    // clean up gpu memory after our calculations
    dev_a_cleanup_memset_success = cudaMemset(dev_a, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_b_cleanup_memset_success = cudaMemset(dev_b, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_c_cleanup_memset_success = cudaMemset(dev_c, 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    assert(dev_a_cleanup_memset_success == cudaSuccess);
    assert(dev_b_cleanup_memset_success == cudaSuccess);
    assert(dev_c_cleanup_memset_success == cudaSuccess);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
