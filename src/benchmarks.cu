#include "benchmarks.h"
#include "bignum_types.h"
#include "input_output.h"
#include "operations.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), char* operation_name, bool is_long_result);
void modular_binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, uint32_t* host_m, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m), char* operation_name);

void add_benchmark();
void sub_benchmark();
void mul_benchmark();
void add_m_benchmark();

// local data kernels
__global__ void add_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void sub_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void mul_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void add_m_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m);

// global data kernels
__global__ void add_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void sub_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void mul_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// BENCHMARKS /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void add_benchmark()
{
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, false);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, false);

    binary_operator_benchmark(host_c, host_a, host_b, add_loc_kernel, "add_loc", false);
    binary_operator_benchmark(host_c, host_a, host_b, add_glo_kernel, "add_glo", false);

    write_coalesced_bignums_to_file(ADD_RESULTS_FILE_NAME, host_c, false);

    free(host_a);
    free(host_b);
    free(host_c);
}

void sub_benchmark()
{
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, false);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, false);

    binary_operator_benchmark(host_c, host_a, host_b, sub_loc_kernel, "sub_loc", false);
    binary_operator_benchmark(host_c, host_a, host_b, sub_glo_kernel, "sub_glo", false);

    write_coalesced_bignums_to_file(SUB_RESULTS_FILE_NAME, host_c, false);

    free(host_a);
    free(host_b);
    free(host_c);
}

void mul_benchmark()
{
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, false);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, false);

    binary_operator_benchmark(host_c, host_a, host_b, mul_loc_kernel, "mul_loc", true);
    binary_operator_benchmark(host_c, host_a, host_b, mul_glo_kernel, "mul_glo", true);

    write_coalesced_bignums_to_file(MUL_RESULTS_FILE_NAME, host_c, true);

    free(host_a);
    free(host_b);
    free(host_c);
}

void add_m_benchmark()
{
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_m = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);
    assert(host_m != NULL);

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, false);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, false);
    read_coalesced_bignums_from_file(COALESCED_M_FILE_NAME, host_m, false);

    modular_binary_operator_benchmark(host_c, host_a, host_b, host_m, add_m_loc_kernel, "add_m_loc");

    write_coalesced_bignums_to_file(ADD_M_RESULTS_FILE_NAME, host_c, false);

    free(host_a);
    free(host_b);
    free(host_c);
    free(host_m);
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// KERNELS ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void add_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    // uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // // 10 iterations
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
    // add_glo(dev_c, dev_a, dev_b, tid);
}

__global__ void add_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    // uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
    // uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
    // uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];

    // for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    // {
    //     a[i] = dev_a[COAL_IDX(i, tid)];
    //     b[i] = dev_b[COAL_IDX(i, tid)];
    // }

    // // 10 iterations
    // add_loc(c, a, b);
    // add_loc(c, a, b);
    // add_loc(c, a, b);
    // add_loc(c, a, b);
    // add_loc(c, a, b);
    // add_loc(c, a, b);
    // add_loc(c, a, b);
    // add_loc(c, a, b);
    // add_loc(c, a, b);
    // add_loc(c, a, b);

    // for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    // {
    //     dev_c[COAL_IDX(i, tid)] = c[i];
    // }
}

__global__ void sub_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    // uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // // 10 iterations
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
    // sub_glo(dev_c, dev_a, dev_b, tid);
}

__global__ void sub_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    // uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
    // uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
    // uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];

    // for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    // {
    //     a[i] = dev_a[COAL_IDX(i, tid)];
    //     b[i] = dev_b[COAL_IDX(i, tid)];
    // }

    // // 10 iterations
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);
    // sub_loc(c, a, b);

    // for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    // {
    //     dev_c[COAL_IDX(i, tid)] = c[i];
    // }
}

__global__ void mul_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    // uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // // 10 iterations
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
    // mul_glo(dev_c, dev_a, dev_b, tid);
}

__global__ void mul_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    // uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
    // uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
    // uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    // for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    // {
    //     a[i] = dev_a[COAL_IDX(i, tid)];
    //     b[i] = dev_b[COAL_IDX(i, tid)];
    // }

    // // 10 iterations
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);
    // mul_loc(c, a, b);

    // for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    // {
    //     dev_c[COAL_IDX(i, tid)] = c[i];
    // }
}

__global__ void add_m_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t m[MIN_BIGNUM_NUMBER_OF_WORDS];

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
        m[i] = dev_m[COAL_IDX(i, tid)];
    }

    // 10 iterations
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);
    add_m_loc(c, a, b, m);

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////// GENERIC LAUNCH CONFIGURATIONS ///////////////////////
////////////////////////////////////////////////////////////////////////////////

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), char* operation_name, bool is_long_result)
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

    uint32_t result_number_of_words = is_long_result ? MAX_BIGNUM_NUMBER_OF_WORDS : MIN_BIGNUM_NUMBER_OF_WORDS;

    // allocate gpu memory
    cudaError dev_a_malloc_success = cudaMalloc((void**) &dev_a, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_malloc_success = cudaMalloc((void**) &dev_b, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_malloc_success = cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * result_number_of_words     * sizeof(uint32_t));
    assert(dev_a_malloc_success == cudaSuccess);
    assert(dev_b_malloc_success == cudaSuccess);
    assert(dev_c_malloc_success == cudaSuccess);

    // make sure gpu memory is clean before our calculations (you never know ...)
    cudaError dev_a_cleanup_memset_success = cudaMemset(dev_a, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_cleanup_memset_success = cudaMemset(dev_b, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_cleanup_memset_success = cudaMemset(dev_c, 0, NUMBER_OF_BIGNUMS * result_number_of_words     * sizeof(uint32_t));
    assert(dev_a_cleanup_memset_success == cudaSuccess);
    assert(dev_b_cleanup_memset_success == cudaSuccess);
    assert(dev_c_cleanup_memset_success == cudaSuccess);

    // copy operands to device memory
    cudaError dev_a_memcpy_succes = cudaMemcpy(dev_a, host_a, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError dev_b_memcpy_succes = cudaMemcpy(dev_b, host_b, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    assert(dev_a_memcpy_succes == cudaSuccess);
    assert(dev_b_memcpy_succes == cudaSuccess);

    printf("Benchmarking \"%s\" on GPU ... ", operation_name);
    fflush(stdout);

    // execute kernel
    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b);

    printf("done\n");
    fflush(stdout);

    // copy results back to host
    cudaError dev_c_memcpy_success = cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * result_number_of_words    * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    assert(dev_c_memcpy_success == cudaSuccess);

    // clean up gpu memory after our calculations
    dev_a_cleanup_memset_success = cudaMemset(dev_a, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_b_cleanup_memset_success = cudaMemset(dev_b, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_c_cleanup_memset_success = cudaMemset(dev_c, 0, NUMBER_OF_BIGNUMS * result_number_of_words     * sizeof(uint32_t));
    assert(dev_a_cleanup_memset_success == cudaSuccess);
    assert(dev_b_cleanup_memset_success == cudaSuccess);
    assert(dev_c_cleanup_memset_success == cudaSuccess);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void modular_binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, uint32_t* host_m, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m), char* operation_name)
{
    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);
    assert(host_m != NULL);
    assert(kernel != NULL);
    assert(operation_name != NULL);

    // device operands (dev_a, dev_b, dev_m) and results (dev_c)
    uint32_t* dev_a;
    uint32_t* dev_b;
    uint32_t* dev_c;
    uint32_t* dev_m;

    // allocate gpu memory
    cudaError dev_a_malloc_success = cudaMalloc((void**) &dev_a, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_malloc_success = cudaMalloc((void**) &dev_b, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_malloc_success = cudaMalloc((void**) &dev_c, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_m_malloc_success = cudaMalloc((void**) &dev_m, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    assert(dev_a_malloc_success == cudaSuccess);
    assert(dev_b_malloc_success == cudaSuccess);
    assert(dev_c_malloc_success == cudaSuccess);
    assert(dev_m_malloc_success == cudaSuccess);

    // make sure gpu memory is clean before our calculations (you never know ...)
    cudaError dev_a_cleanup_memset_success = cudaMemset(dev_a, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_b_cleanup_memset_success = cudaMemset(dev_b, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_cleanup_memset_success = cudaMemset(dev_c, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_m_cleanup_memset_success = cudaMemset(dev_m, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    assert(dev_a_cleanup_memset_success == cudaSuccess);
    assert(dev_b_cleanup_memset_success == cudaSuccess);
    assert(dev_c_cleanup_memset_success == cudaSuccess);
    assert(dev_m_cleanup_memset_success == cudaSuccess);

    // copy operands to device memory
    cudaError dev_a_memcpy_succes = cudaMemcpy(dev_a, host_a, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError dev_b_memcpy_succes = cudaMemcpy(dev_b, host_b, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError dev_m_memcpy_succes = cudaMemcpy(dev_m, host_m, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    assert(dev_a_memcpy_succes == cudaSuccess);
    assert(dev_b_memcpy_succes == cudaSuccess);
    assert(dev_m_memcpy_succes == cudaSuccess);

    printf("Benchmarking \"%s\" on GPU ... ", operation_name);
    fflush(stdout);

    // execute kernel
    kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_a, dev_b, dev_m);

    printf("done\n");
    fflush(stdout);

    // copy results back to host
    cudaError dev_c_memcpy_success = cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    assert(dev_c_memcpy_success == cudaSuccess);

    // clean up gpu memory after our calculations
    dev_a_cleanup_memset_success = cudaMemset(dev_a, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_b_cleanup_memset_success = cudaMemset(dev_b, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_c_cleanup_memset_success = cudaMemset(dev_c, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_m_cleanup_memset_success = cudaMemset(dev_m, 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    assert(dev_a_cleanup_memset_success == cudaSuccess);
    assert(dev_b_cleanup_memset_success == cudaSuccess);
    assert(dev_c_cleanup_memset_success == cudaSuccess);
    assert(dev_m_cleanup_memset_success == cudaSuccess);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_m);
}
