#include "benchmarks.h"
#include "constants.h"
#include <stdlib.h>

int main(void)
{
    // add_benchmark();
    // sub_benchmark();
    // mul_benchmark();
    add_m_benchmark();

    // for leak detection when using cuda-memcheck
    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
