#include "input_output.h"
#include "bignum_types.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

void read_coalesced_bignums_from_file(const char* file_name, uint32_t* bignums, bool is_long_number);
void write_coalesced_bignums_to_file(const char* file_name, uint32_t* bignums, bool is_long_number);

void read_coalesced_bignums_from_file(const char* file_name, uint32_t* bignums, bool is_long_number)
{
    assert(file_name != NULL);
    assert(bignums != NULL);

    FILE* file = fopen(file_name, "r");
    assert(file != NULL);

    printf("Reading coalesced bignums from file \"%s\" ... ", file_name);
    fflush(stdout);

    uint32_t number_of_words = is_long_number ? MAX_BIGNUM_NUMBER_OF_WORDS : MIN_BIGNUM_NUMBER_OF_WORDS;
    for (uint32_t i = 0; i < number_of_words; i++)
    {
        for (uint32_t j = 0; j < NUMBER_OF_BIGNUMS; j++)
        {
            fscanf(file, "%x", &bignums[COAL_IDX(i, j)]);
        }
        fprintf(file, "\n");
    }

    printf("done\n");
    fflush(stdout);

    fclose(file);
}

void write_coalesced_bignums_to_file(const char* file_name, uint32_t* bignums, bool is_long_number)
{
    assert(file_name != NULL);
    assert(bignums != NULL);

    FILE* file = fopen(file_name, "w");
    assert(file != NULL);

    printf("Writing coalesced bignums to   file \"%s\" ... ", file_name);
    fflush(stdout);

    uint32_t number_of_words = is_long_number ? MAX_BIGNUM_NUMBER_OF_WORDS : MIN_BIGNUM_NUMBER_OF_WORDS;
    for (uint32_t i = 0; i < number_of_words; i++)
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
