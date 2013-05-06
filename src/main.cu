#include "bignum_types.h"
#include "input_output.h"
#include "benchmarks.h"
#include "constants.h"
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

int main(void)
{
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
    read_coalesced_bignums_from_file(COALESCED_A_FILE_NAME, host_a);
    read_coalesced_bignums_from_file(COALESCED_B_FILE_NAME, host_b);
    read_coalesced_bignums_from_file(COALESCED_M_FILE_NAME, host_m);

    // benchmarks
    // add_benchmark(host_c, host_a, host_b, ADD_RESULTS_FILE_NAME);
    // sub_benchmark(host_c, host_a, host_b, SUB_RESULTS_FILE_NAME);
    mul_benchmark(host_c, host_a, host_b, MUL_RESULTS_FILE_NAME);

    free(host_a);
    free(host_b);
    free(host_c);
    free(host_m);

    // for leak detection when using cuda-memcheck
    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
