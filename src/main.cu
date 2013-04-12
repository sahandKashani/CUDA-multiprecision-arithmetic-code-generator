#include "bignum_types.h"
#include "constants.h"
#include "benchmarks.cuh"
#include "io_interface.h"
#include <stdio.h>
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

    if (host_a != NULL && host_b != NULL && host_c != NULL)
    {
        // generate_random_bignum_arrays_to_files(bignum_file_1, bignum_file_2);
        read_bignum_arrays_from_files(host_a, host_b, bignum_file_1, bignum_file_2);

        benchmark(host_c, host_a, host_b);

        free(host_a);
        free(host_b);
        free(host_c);
    }
    else
    {
        if (host_a == NULL)
        {
            printf("Error: could not allocate memory for \"host_a\"\n");
        }

        if (host_b == NULL)
        {
            printf("Error: could not allocate memory for \"host_b\"\n");
        }

        if (host_c == NULL)
        {
            printf("Error: could not allocate memory for \"host_c\"\n");
        }

        exit(EXIT_FAILURE);
    }


    // for leak detection when using cuda-memcheck
    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
