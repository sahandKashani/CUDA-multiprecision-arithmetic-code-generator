#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <stdint.h>

void add_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name);
void sub_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name);
void mul_benchmark(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, const char* output_file_name);

#endif


