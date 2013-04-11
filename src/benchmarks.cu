#include "bignum_types.h"
#include "random_bignum_generator.h"
#include "constants.h"
#include "operation_check.h"
#include "memory_layout_benchmarks.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void generate_random_bignum_arrays_to_files();
void generate_random_bignum_array_to_file(const char* file_name);
void read_bignum_arrays_from_files(uint32_t* a, uint32_t* b);
void read_bignum_array_from_file(const char* file_name, uint32_t* bignum, uint32_t amount_to_read);
void write_bignum_array_to_file(const char* file_name, uint32_t* bignum);

int main(void)
{
    // host operands (host_a, host_b) and results (host_c)
    uint32_t* host_a = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_c = (uint32_t*) calloc(NUMBER_OF_BIGNUMS * BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    // generate_random_bignum_arrays_to_files();
    read_bignum_arrays_from_files(host_a, host_b);

    benchmark(host_c, host_a, host_b, BLOCKS_PER_GRID, THREADS_PER_BLOCK);
    addition_check(host_c, host_a, host_b);

    free(host_a);
    free(host_b);
    free(host_c);

    // for leak detection when using cuda-memcheck
    cudaDeviceReset();
}

void read_bignum_arrays_from_files(uint32_t* a, uint32_t* b)
{
    printf("reading bignums from files ... ");
    fflush(stdout);
    read_bignum_array_from_file("../data/bignum_array_1.txt", a, NUMBER_OF_BIGNUMS);
    read_bignum_array_from_file("../data/bignum_array_2.txt", b, NUMBER_OF_BIGNUMS);
    printf("done\n");
    fflush(stdout);
}

void generate_random_bignum_arrays_to_files()
{
    start_random_number_generator();

    generate_random_bignum_array_to_file("../data/bignum_array_1.txt");
    generate_random_bignum_array_to_file("../data/bignum_array_2.txt");

    stop_random_number_generator();
}

void generate_random_bignum_array_to_file(const char* file_name)
{
    uint32_t* bignum = (uint32_t*) calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    FILE* file = fopen(file_name, "w");

    for (uint32_t i = 0; i < NUMBER_OF_BIGNUMS; i++)
    {
        generate_random_bignum(bignum);

        for (uint32_t j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
        {
            fprintf(file, "%u ", bignum[j]);
        }

        fprintf(file, "\n");
    }
}

void read_bignum_array_from_file(const char* file_name, uint32_t* bignum, uint32_t amount_to_read)
{
    FILE* file = fopen(file_name, "r");

    for (uint32_t i = 0; i < amount_to_read; i++)
    {
        for (uint32_t j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
        {
            fscanf(file, "%u", &bignum[IDX(i, j)]);
        }

        fprintf(file, "\n");
    }

    fclose(file);
}

void write_bignum_array_to_file(const char* file_name, uint32_t* bignum)
{
    FILE* file = fopen(file_name, "w");

    for (uint32_t i = 0; i < NUMBER_OF_BIGNUMS; i++)
    {
        for (uint32_t j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
        {
            fprintf(file, "%u ", bignum[IDX(i, j)]);
        }

        fprintf(file, "\n");
    }

    fclose(file);
}
