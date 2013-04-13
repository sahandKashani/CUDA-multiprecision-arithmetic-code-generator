#include "bignum_types.h"
#include "constants.h"
#include "benchmarks.cuh"
#include "io_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(void)
{
    const char* host_a_file_name = "../data/host_a.txt";
    const char* host_b_file_name = "../data/host_b.txt";
    const char* host_m_file_name = "../data/host_m.txt";

    // host operands (host_a, host_b) and results (host_c)
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_m = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    if (host_a != NULL && host_b != NULL && host_c != NULL)
    {
        // generate operands to files
        generate_random_bignum_modulus_and_operand_arrays_to_files(host_m_file_name, host_a_file_name, host_b_file_name);

        // read operands from files back to memory
        read_bignum_array_from_file(host_a_file_name, host_a, NUMBER_OF_BIGNUMS);
        read_bignum_array_from_file(host_b_file_name, host_b, NUMBER_OF_BIGNUMS);
        read_bignum_array_from_file(host_m_file_name, host_m, NUMBER_OF_BIGNUMS);

        benchmark(host_c, host_a, host_b);

        free(host_a);
        free(host_b);
        free(host_c);
        free(host_m);
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
