#include "benchmarks.h"
#include <stdlib.h>

int main(void)
{
    add_benchmark();
    sub_benchmark();
    mul_benchmark();
    mul_karatsuba_benchmark();
    add_m_benchmark();
    sub_m_benchmark();
    // montgomery_reduction_benchmark();

    assembly_vs_C_addition_benchmark();
    assembly_vs_C_modular_addition_benchmark();

    // for leak detection when using cuda-memcheck
    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
