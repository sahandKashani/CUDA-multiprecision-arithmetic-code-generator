#include "bignum_types.h"
#include "input_output.h"
#include "constants.h"
// #include "benchmarks.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

int main(void)
{
    const char* host_a_file_name = "../data/coalesced_a.txt";
    const char* host_b_file_name = "../data/coalesced_b.txt";
    const char* host_m_file_name = "../data/coalesced_m.txt";

    // host operands (host_a, host_b) and results (host_c)
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_m = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * MAX_BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_c != NULL);
    assert(host_m != NULL);

    // read operands from files back to memory
    read_coalesced_bignums_from_file(host_a_file_name, host_a);
    read_coalesced_bignums_from_file(host_b_file_name, host_b);
    read_coalesced_bignums_from_file(host_m_file_name, host_m);

    for (int i = 0; i < MAX_BIGNUM_NUMBER_OF_WORDS; i++)
    {
        for (int j = 0; j < NUMBER_OF_BIGNUMS; j++)
        {
            printf("%x ", host_a[COAL_IDX(i, j)]);
        }
        printf("\n");
    }

    // benchmark(host_c, host_a, host_b);

    free(host_a);
    free(host_b);
    free(host_c);
    free(host_m);

    // for leak detection when using cuda-memcheck
    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
