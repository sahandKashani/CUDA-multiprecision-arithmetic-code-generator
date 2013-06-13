#include "benchmarks.h"
#include "bignum_types.h"
#include "input_output.h"
#include "operations.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), char* operation_name, uint32_t result_number_of_words);
void modular_binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, uint32_t* host_m, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m), char* operation_name);

void add_benchmark();
void sub_benchmark();
void mul_benchmark();
void mul_karatsuba_benchmark();
void add_m_benchmark();
void sub_m_benchmark();
void montgomery_reduction_benchmark();

// local data kernels
__global__ void add_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void sub_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void mul_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void mul_karatsuba_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__global__ void add_m_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m);
__global__ void sub_m_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m);
__global__ void montgomery_reduction_loc_kernel(uint32_t* dev_c, uint32_t* dev_T, uint32_t* dev_m, uint32_t* dev_m_prime);

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

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, MIN_BIGNUM_NUMBER_OF_WORDS);

    binary_operator_benchmark(host_c, host_a, host_b, add_loc_kernel, "add_loc", MIN_BIGNUM_NUMBER_OF_WORDS);
    binary_operator_benchmark(host_c, host_a, host_b, add_glo_kernel, "add_glo", MIN_BIGNUM_NUMBER_OF_WORDS);

    write_coalesced_bignums_to_file(ADD_RESULTS_FILE_NAME, host_c, MIN_BIGNUM_NUMBER_OF_WORDS);

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

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, MIN_BIGNUM_NUMBER_OF_WORDS);

    binary_operator_benchmark(host_c, host_a, host_b, sub_loc_kernel, "sub_loc", MIN_BIGNUM_NUMBER_OF_WORDS);
    binary_operator_benchmark(host_c, host_a, host_b, sub_glo_kernel, "sub_glo", MIN_BIGNUM_NUMBER_OF_WORDS);

    write_coalesced_bignums_to_file(SUB_RESULTS_FILE_NAME, host_c, MIN_BIGNUM_NUMBER_OF_WORDS);

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

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, MIN_BIGNUM_NUMBER_OF_WORDS);

    binary_operator_benchmark(host_c, host_a, host_b, mul_loc_kernel, "mul_loc", MAX_BIGNUM_NUMBER_OF_WORDS);
    binary_operator_benchmark(host_c, host_a, host_b, mul_glo_kernel, "mul_glo", MAX_BIGNUM_NUMBER_OF_WORDS);

    write_coalesced_bignums_to_file(MUL_RESULTS_FILE_NAME, host_c, MAX_BIGNUM_NUMBER_OF_WORDS);

    free(host_a);
    free(host_b);
    free(host_c);
}

void mul_karatsuba_benchmark()
{
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, MIN_BIGNUM_NUMBER_OF_WORDS);

    binary_operator_benchmark(host_c, host_a, host_b, mul_karatsuba_loc_kernel, "mul_karatsuba_loc", MAX_BIGNUM_NUMBER_OF_WORDS);

    write_coalesced_bignums_to_file(MUL_KARATSUBA_RESULTS_FILE_NAME, host_c, MAX_BIGNUM_NUMBER_OF_WORDS);

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

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_M_FILE_NAME, host_m, MIN_BIGNUM_NUMBER_OF_WORDS);

    modular_binary_operator_benchmark(host_c, host_a, host_b, host_m, add_m_loc_kernel, "add_m_loc");

    write_coalesced_bignums_to_file(ADD_M_RESULTS_FILE_NAME, host_c, MIN_BIGNUM_NUMBER_OF_WORDS);

    free(host_a);
    free(host_b);
    free(host_c);
    free(host_m);
}

void sub_m_benchmark()
{
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_m = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);
    assert(host_m != NULL);

    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_M_FILE_NAME, host_m, MIN_BIGNUM_NUMBER_OF_WORDS);

    modular_binary_operator_benchmark(host_c, host_a, host_b, host_m, sub_m_loc_kernel, "sub_m_loc");

    write_coalesced_bignums_to_file(SUB_M_RESULTS_FILE_NAME, host_c, MIN_BIGNUM_NUMBER_OF_WORDS);

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
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 10 iterations
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
    add_glo(dev_c, dev_a, dev_b, tid);
}

__global__ void add_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    // 10 iterations
    add_loc(c, a, b);
    add_loc(c, a, b);
    add_loc(c, a, b);
    add_loc(c, a, b);
    add_loc(c, a, b);
    add_loc(c, a, b);
    add_loc(c, a, b);
    add_loc(c, a, b);
    add_loc(c, a, b);
    add_loc(c, a, b);

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

__global__ void sub_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 10 iterations
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
    sub_glo(dev_c, dev_a, dev_b, tid);
}

__global__ void sub_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    // 10 iterations
    sub_loc(c, a, b);
    sub_loc(c, a, b);
    sub_loc(c, a, b);
    sub_loc(c, a, b);
    sub_loc(c, a, b);
    sub_loc(c, a, b);
    sub_loc(c, a, b);
    sub_loc(c, a, b);
    sub_loc(c, a, b);
    sub_loc(c, a, b);

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

__global__ void mul_glo_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 10 iterations
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
    mul_glo(dev_c, dev_a, dev_b, tid);
}

__global__ void mul_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    // 10 iterations
    mul_loc(c, a, b);
    mul_loc(c, a, b);
    mul_loc(c, a, b);
    mul_loc(c, a, b);
    mul_loc(c, a, b);
    mul_loc(c, a, b);
    mul_loc(c, a, b);
    mul_loc(c, a, b);
    mul_loc(c, a, b);
    mul_loc(c, a, b);

    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

__global__ void mul_karatsuba_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MAX_BIGNUM_NUMBER_OF_WORDS];

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        a[i] = dev_a[COAL_IDX(i, tid)];
        b[i] = dev_b[COAL_IDX(i, tid)];
    }

    // 10 iterations
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);
    mul_karatsuba_loc(c, a, b);

    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
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

__global__ void sub_m_loc_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m)
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
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);
    sub_m_loc(c, a, b, m);

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

__global__ void montgomery_reduction_loc_kernel(uint32_t* dev_c, uint32_t* dev_T, uint32_t* dev_m, uint32_t* dev_m_prime)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t T[MAX_BIGNUM_NUMBER_OF_WORDS];
    uint32_t m[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];
    uint32_t m_prime;

    for (uint32_t i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        T[i] = dev_T[COAL_IDX(i, tid)];
    }

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        m[i] = dev_m    [COAL_IDX(i, tid)];
    }

    m_prime = dev_m_prime[COAL_IDX(0, tid)];

    // 10 iterations
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);
    montgomery_reduction(c, T, m, m_prime);

    for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        dev_c[COAL_IDX(i, tid)] = c[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////// GENERIC LAUNCH CONFIGURATIONS ///////////////////////
////////////////////////////////////////////////////////////////////////////////

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), char* operation_name, uint32_t result_number_of_words)
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

void montgomery_reduction_benchmark()
{
    // host
    uint32_t* host_T       = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_m       = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c       = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_m_prime = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * 1                         , sizeof(uint32_t));

    assert(host_T       != NULL);
    assert(host_m       != NULL);
    assert(host_c       != NULL);
    assert(host_m_prime != NULL);

    read_coalesced_bignums_from_file(COALESCED_T_MON_FILE_NAME, host_T      , MAX_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(COALESCED_M_FILE_NAME    , host_m      , MIN_BIGNUM_NUMBER_OF_WORDS);
    read_coalesced_bignums_from_file(M_PRIME_FILE_NAME        , host_m_prime, 1                         );

    // device operands (dev_a, dev_b, dev_m) and results (dev_c)
    uint32_t* dev_T;
    uint32_t* dev_m;
    uint32_t* dev_c;
    uint32_t* dev_m_prime;

    // allocate gpu memory
    cudaError dev_T_malloc_success       = cudaMalloc((void**) &dev_T      , NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_m_malloc_success       = cudaMalloc((void**) &dev_m      , NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_malloc_success       = cudaMalloc((void**) &dev_c      , NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_m_prime_malloc_success = cudaMalloc((void**) &dev_m_prime, NUMBER_OF_BIGNUMS * 1                          * sizeof(uint32_t));
    assert(dev_T_malloc_success       == cudaSuccess);
    assert(dev_m_malloc_success       == cudaSuccess);
    assert(dev_c_malloc_success       == cudaSuccess);
    assert(dev_m_prime_malloc_success == cudaSuccess);

    // make sure gpu memory is clean before our calculations (you never know ...)
    cudaError dev_T_cleanup_memset_success       = cudaMemset(dev_T      , 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_m_cleanup_memset_success       = cudaMemset(dev_m      , 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_c_cleanup_memset_success       = cudaMemset(dev_c      , 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    cudaError dev_m_prime_cleanup_memset_success = cudaMemset(dev_m_prime, 0, NUMBER_OF_BIGNUMS * 1                          * sizeof(uint32_t));
    assert(dev_T_cleanup_memset_success       == cudaSuccess);
    assert(dev_m_cleanup_memset_success       == cudaSuccess);
    assert(dev_c_cleanup_memset_success       == cudaSuccess);
    assert(dev_m_prime_cleanup_memset_success == cudaSuccess);

    // copy operands to device memory
    cudaError dev_T_memcpy_succes       = cudaMemcpy(dev_T      , host_T      , NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError dev_m_memcpy_succes       = cudaMemcpy(dev_m      , host_m      , NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaError dev_m_prime_memcpy_succes = cudaMemcpy(dev_m_prime, host_m_prime, NUMBER_OF_BIGNUMS * 1                          * sizeof(uint32_t), cudaMemcpyHostToDevice);
    assert(dev_T_memcpy_succes       == cudaSuccess);
    assert(dev_m_memcpy_succes       == cudaSuccess);
    assert(dev_m_prime_memcpy_succes == cudaSuccess);

    printf("Benchmarking \"montgomery_reduction\" on GPU ... ");
    fflush(stdout);

    // execute kernel
    montgomery_reduction_loc_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_c, dev_T, dev_m, dev_m_prime);

    printf("done\n");
    fflush(stdout);

    // copy results back to host
    cudaError dev_c_memcpy_success = cudaMemcpy(host_c, dev_c, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    assert(dev_c_memcpy_success == cudaSuccess);

    // clean up gpu memory after our calculations
    dev_T_cleanup_memset_success       = cudaMemset(dev_T      , 0, NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_m_cleanup_memset_success       = cudaMemset(dev_m      , 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_c_cleanup_memset_success       = cudaMemset(dev_c      , 0, NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS * sizeof(uint32_t));
    dev_m_prime_cleanup_memset_success = cudaMemset(dev_m_prime, 0, NUMBER_OF_BIGNUMS * 1                          * sizeof(uint32_t));
    assert(dev_T_cleanup_memset_success       == cudaSuccess);
    assert(dev_m_cleanup_memset_success       == cudaSuccess);
    assert(dev_c_cleanup_memset_success       == cudaSuccess);
    assert(dev_m_prime_cleanup_memset_success == cudaSuccess);

    // free device memory
    cudaFree(dev_T);
    cudaFree(dev_m);
    cudaFree(dev_c);
    cudaFree(dev_m_prime);

    write_coalesced_bignums_to_file(MONTGOMERY_REDUCTION_RESULTS_FILE_NAME, host_c, MIN_BIGNUM_NUMBER_OF_WORDS);

    free(host_T);
    free(host_m);
    free(host_c);
    free(host_m_prime);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// ASSEMBLY vs. C //////////////////////////////
////////////////////////////////////////////////////////////////////////////////


// __global__ void add_loc_assembly_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

//     uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];

//     for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//     {
//         a[i] = dev_a[COAL_IDX(i, tid)];
//         b[i] = dev_b[COAL_IDX(i, tid)];
//     }

//     // 10 iterations
//     add_loc(c, a, b);
//     add_loc(c, a, b);
//     add_loc(c, a, b);
//     add_loc(c, a, b);
//     add_loc(c, a, b);
//     add_loc(c, a, b);
//     add_loc(c, a, b);
//     add_loc(c, a, b);
//     add_loc(c, a, b);
//     add_loc(c, a, b);

//     for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//     {
//         dev_c[COAL_IDX(i, tid)] = c[i];
//     }
// }

// __global__ void add_loc_C_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

//     uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];

//     for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//     {
//         a[i] = dev_a[COAL_IDX(i, tid)];
//         b[i] = dev_b[COAL_IDX(i, tid)];
//     }

//     // 10 iterations
//     for (uint32_t j = 0; j < 10; j++)
//     {
//         c[0] = a[0] + b[0];
//         for (uint32_t i = 1; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//         {
//             c[i] = a[i] + b[i] + (c[i-1] < max(a[i-1], b[i-1]));
//         }
//     }

//     for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//     {
//         dev_c[COAL_IDX(i, tid)] = c[i];
//     }
// }

// __global__ void add_m_loc_assembly_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

//     uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t m[MIN_BIGNUM_NUMBER_OF_WORDS];

//     for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//     {
//         a[i] = dev_a[COAL_IDX(i, tid)];
//         b[i] = dev_b[COAL_IDX(i, tid)];
//         m[i] = dev_m[COAL_IDX(i, tid)];
//     }

//     // 10 iterations
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);
//     add_m_loc(c, a, b, m);

//     for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//     {
//         dev_c[COAL_IDX(i, tid)] = c[i];
//     }
// }

// __global__ void add_m_loc_C_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b, uint32_t* dev_m)
// {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

//     uint32_t a[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t b[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t c[MIN_BIGNUM_NUMBER_OF_WORDS];
//     uint32_t m[MIN_BIGNUM_NUMBER_OF_WORDS];

//     for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//     {
//         a[i] = dev_a[COAL_IDX(i, tid)];
//         b[i] = dev_b[COAL_IDX(i, tid)];
//         m[i] = dev_m[COAL_IDX(i, tid)];
//     }

//     // 10 iterations
//     for (uint32_t j = 0; j < 10; j++)
//     {
//         // c    = a + b
//         c[0] = a[0] + b[0];
//         for (uint32_t i = 1; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//         {
//             c[i] = a[i] + b[i] + (c[i-1] < max(a[i-1], b[i-1]));
//         }

//         // c    = c - m
//         c[0] = c[0] - m[0];
//         for (uint32_t i = 1; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//         {
//             c[i] = c[i] - m[i] + (c[i-1] < max(a[i-1], b[i-1]));
//         }

//         // mask = 0 - borrow = mask - mask - borrow
//         // mask = mask & m
//         // c    = c + mask
//     }

//     for (uint32_t i = 0; i < MIN_BIGNUM_NUMBER_OF_WORDS; i++)
//     {
//         dev_c[COAL_IDX(i, tid)] = c[i];
//     }
// }

// void assembly_vs_C_addition_benchmark()
// {
//     uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
//     uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
//     uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

//     assert(host_a != NULL);
//     assert(host_b != NULL);
//     assert(host_c != NULL);

//     read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, MIN_BIGNUM_NUMBER_OF_WORDS);
//     read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, MIN_BIGNUM_NUMBER_OF_WORDS);

//     binary_operator_benchmark(host_c, host_a, host_b, add_loc_assembly_kernel, "add_loc_assembly", MIN_BIGNUM_NUMBER_OF_WORDS);
//     binary_operator_benchmark(host_c, host_a, host_b, add_loc_C_kernel, "add_loc_C", MIN_BIGNUM_NUMBER_OF_WORDS);

//     write_coalesced_bignums_to_file(ADD_RESULTS_FILE_NAME, host_c, MIN_BIGNUM_NUMBER_OF_WORDS);

//     free(host_a);
//     free(host_b);
//     free(host_c);
// }

// void assembly_vs_C_modular_addition_benchmark()
// {
//     uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
//     uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
//     uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
//     uint32_t* host_m = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MIN_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

//     assert(host_a != NULL);
//     assert(host_b != NULL);
//     assert(host_c != NULL);
//     assert(host_m != NULL);

//     read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a, MIN_BIGNUM_NUMBER_OF_WORDS);
//     read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b, MIN_BIGNUM_NUMBER_OF_WORDS);
//     read_coalesced_bignums_from_file(COALESCED_M_FILE_NAME, host_m, MIN_BIGNUM_NUMBER_OF_WORDS);

//     modular_binary_operator_benchmark(host_c, host_a, host_b, host_m, add_m_loc_assembly_kernel, "add_m_loc_assembly");
//     modular_binary_operator_benchmark(host_c, host_a, host_b, host_m, add_m_loc_C_kernel, "add_m_loc_C");

//     write_coalesced_bignums_to_file(ADD_M_RESULTS_FILE_NAME, host_c, MIN_BIGNUM_NUMBER_OF_WORDS);

//     free(host_a);
//     free(host_b);
//     free(host_c);
//     free(host_m);
// }
