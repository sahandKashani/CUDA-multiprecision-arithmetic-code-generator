#include "input_output.h"
#include "bignum_types.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

void read_coalesced_bignum_array_from_file(const char* file_name, uint32_t* bignum);
void write_coalesced_bignum_array_to_file(const char* file_name, uint32_t* bignum);

void read_coalesced_bignum_array_from_file(const char* file_name, uint32_t* bignum)
{
    assert(file_name != NULL);
    assert(bignum != NULL);

    FILE* file = fopen(file_name, "r");
    assert(file != NULL);

    printf("Reading coalesced bignum array from file \"%s\" ... ", file_name);
    fflush(stdout);

    for (uint32_t i = 0; i < NUMBER_OF_BIGNUMS; i++)
    {
        for (uint32_t j = 0; j < MAX_BIGNUM_NUMBER_OF_WORDS; j++)
        {
            fscanf(file, "%x", &bignum[IDX(i, j)]);
        }
        fprintf(file, "\n");
    }
    fclose(file);

    printf("done\n");
    fflush(stdout);
}

void write_coalesced_bignum_array_to_file(const char* file_name, uint32_t* bignum)
{
    assert(file_name != NULL);
    assert(bignum != NULL);

    FILE* file = fopen(file_name, "w");
    assert(file != NULL);

    printf("Writing coalesced bignum array to file \"%s\" ... ", file_name);
    fflush(stdout);

    for (uint32_t i = 0; i < NUMBER_OF_BIGNUMS; i++)
    {
        for (uint32_t j = 0; j < MAX_BIGNUM_NUMBER_OF_WORDS; j++)
        {
            fprintf(file, "%x ", bignum[IDX(i, j)]);
        }
        fprintf(file, "\n");
    }
    fclose(file);

    printf("done\n");
    fflush(stdout);
}
