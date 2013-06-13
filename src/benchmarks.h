#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <stdint.h>

void add_benchmark();
void sub_benchmark();
void mul_benchmark();
void mul_karatsuba_benchmark();
void add_m_benchmark();
void sub_m_benchmark();
void montgomery_reduction_benchmark();

// void assembly_vs_C_addition_benchmark();
// void assembly_vs_C_modular_addition_benchmark();

#endif
