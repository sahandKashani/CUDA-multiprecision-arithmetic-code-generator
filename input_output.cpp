#include "input_output.h"
#include "bignum_types.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

void read_coalesced_bignums_from_file(const char* file_name, uint32_t* bignums, uint32_t word_count);
void write_coalesced_bignums_to_file(const char* file_name, uint32_t* bignums, uint32_t word_count);

void read_coalesced_bignums_from_file(const char* file_name, uint32_t* bignums, uint32_t word_count)
{
    assert(file_name != NULL);
    assert(bignums != NULL);

    FILE* file = fopen(file_name, "r");
    assert(file != NULL);

    printf("Reading coalesced bignums from file \"%s\" ... ", file_name);
    fflush(stdout);

    for (uint32_t i = 0; i < word_count; i++)
    {
        for (uint32_t j = 0; j < NUMBER_OF_BIGNUMS_IN_FILES; j++)
        {
            if (j < NUMBER_OF_BIGNUMS)
            {
                fscanf(file, "%x", &bignums[COAL_IDX(i, j)]);
            }
            else
            {
                uint32_t dummy = 0;
                fscanf(file, "%x", &dummy);
            }
        }
        fprintf(file, "\n");
    }

    printf("done\n");
    fflush(stdout);

    fclose(file);
}

void write_coalesced_bignums_to_file(const char* file_name, uint32_t* bignums, uint32_t word_count)
{
    assert(file_name != NULL);
    assert(bignums != NULL);

    FILE* file = fopen(file_name, "w");
    assert(file != NULL);

    printf("Writing coalesced bignums to   file \"%s\" ... ", file_name);
    fflush(stdout);

    for (uint32_t i = 0; i < word_count; i++)
    {
        for (uint32_t j = 0; j < NUMBER_OF_BIGNUMS; j++)
        {
            fprintf(file, "%08x ", bignums[COAL_IDX(i, j)]);
        }
        fprintf(file, "\n");
    }

    printf("done\n");
    fflush(stdout);

    fclose(file);
}
