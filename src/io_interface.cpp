#include "io_interface.h"
#include "bignum_types.h"
#include "random_bignum_generator.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>

#include "bignum_conversions.h" // TO REMOVE

void generate_random_bignum_modulus_and_operand_arrays_to_files(const char* host_m_file_name, const char* host_a_file_name, const char* host_b_file_name)
{
    assert(host_a_file_name != NULL);
    assert(host_b_file_name != NULL);
    assert(host_m_file_name != NULL);

    start_random_number_generator();

    FILE* host_a_file = fopen(host_a_file_name, "w");
    FILE* host_b_file = fopen(host_b_file_name, "w");
    FILE* host_m_file = fopen(host_m_file_name, "w");

    assert(host_a_file != NULL);
    assert(host_b_file != NULL);
    assert(host_m_file != NULL);

    uint32_t* host_a = (uint32_t*) calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_b = (uint32_t*) calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
    uint32_t* host_m = (uint32_t*) calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

    assert(host_a != NULL);
    assert(host_b != NULL);
    assert(host_m != NULL);

    for (uint32_t i = 0; i < NUMBER_OF_BIGNUMS; i++)
    {
        // generate modulus
        // generate_exact_precision_bignum(host_m, BIT_RANGE);

        // generate operands which are smaller than the modulus
        // generate_bignum_less_than_bignum(host_m, host_a);
        // generate_bignum_less_than_bignum(host_m, host_b);

        char* host_a_str = bignum_to_binary_string(host_a);
        // char* host_b_str = bignum_to_binary_string(host_b);
        // char* host_m_str = bignum_to_binary_string(host_m);

        // fprintf(host_a_file, "%s", host_a_str);
        // fprintf(host_b_file, "%s", host_b_str);
        // fprintf(host_m_file, "%s", host_m_str);

        free(host_a_str);
        // free(host_b_str);
        // free(host_m_str);

        // for (uint32_t j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
        // {
        //     fprintf(host_a_file, "%u ", host_a[j]);
        //     fprintf(host_b_file, "%u ", host_b[j]);
        //     fprintf(host_m_file, "%u ", host_m[j]);
        // }

        // fprintf(host_a_file, "\n");
        // fprintf(host_b_file, "\n");
        // fprintf(host_m_file, "\n");
    }

    fclose(host_a_file);
    fclose(host_b_file);
    fclose(host_m_file);

    free(host_a);
    free(host_b);
    free(host_m);

    stop_random_number_generator();
}

void read_bignum_array_from_file(const char* file_name, uint32_t* bignum, uint32_t amount_to_read)
{
    assert(file_name != NULL);
    assert(bignum != NULL);
    assert(amount_to_read > 0);

    FILE* file = fopen(file_name, "r");
    assert(file != NULL);

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
    assert(file_name != NULL);
    assert(bignum != NULL);

    FILE* file = fopen(file_name, "w");
    assert(file != NULL);

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
