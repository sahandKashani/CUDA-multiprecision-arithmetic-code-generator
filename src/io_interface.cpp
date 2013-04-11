#include "io_interface.h"
#include "bignum_types.h"
#include "random_bignum_generator.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void read_bignum_arrays_from_files(uint32_t* a, uint32_t* b, const char* file_name_1, const char* file_name_2)
{
    printf("reading bignums from files ... ");
    fflush(stdout);
    read_bignum_array_from_file(file_name_1, a, NUMBER_OF_BIGNUMS);
    read_bignum_array_from_file(file_name_2, b, NUMBER_OF_BIGNUMS);
    printf("done\n");
    fflush(stdout);
}

void generate_random_bignum_arrays_to_files(const char* file_name_1, const char* file_name_2)
{
    start_random_number_generator();

    generate_random_bignum_array_to_file(file_name_1);
    generate_random_bignum_array_to_file(file_name_2);

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
