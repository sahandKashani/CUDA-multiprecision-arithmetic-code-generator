#ifndef BENCHMARKS_CUH
#define BENCHMARKS_CUH

#include <stdint.h>

void benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);

void binary_operator_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*kernel)(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b), void (*checking_function)(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b));

void addition_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
__global__ void addition_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__device__ void addition(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

void subtraction_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
__global__ void subtraction_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__device__ void subtraction(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

void modular_subtraction_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
__global__ void modular_subtraction_kernel(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);
__device__ void modular_subtraction(uint32_t* dev_c, uint32_t* dev_a, uint32_t* dev_b);

#endif


