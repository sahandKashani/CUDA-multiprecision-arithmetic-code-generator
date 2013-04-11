#include "bignum_types.h"
#include "constants.h"
#include "benchmarks.cuh"
#include "io_interface.h"
#include <stdlib.h>
#include <stdint.h>

int main(void)
{
    const char* bignum_file_1 = "../data/bignum_array_1.txt";
    const char* bignum_file_2 = "../data/bignum_array_2.txt";

    // host operands (host_a, host_b) and results (host_c)
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    // generate_random_bignum_arrays_to_files(bignum_file_1, bignum_file_2);
    read_bignum_arrays_from_files(host_a, host_b, bignum_file_1, bignum_file_2);

    benchmark(host_c, host_a, host_b);

    free(host_a);
    free(host_b);
    free(host_c);

    // for leak detection when using cuda-memcheck
    cudaDeviceReset();
}
