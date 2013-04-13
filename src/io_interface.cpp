#include "io_interface.h"
#include "bignum_types.h"
#include "random_bignum_generator.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void generate_random_bignum_modulus_and_operand_arrays_to_files(const char* host_m_file_name, const char* host_a_file_name, const char* host_b_file_name)
{
    start_random_number_generator();

    if (host_m_file_name != NULL && host_a_file_name != NULL && host_b_file_name != NULL)
    {
        FILE* host_m_file = fopen(host_m_file_name, "w");
        FILE* host_a_file = fopen(host_a_file_name, "w");
        FILE* host_b_file = fopen(host_b_file_name, "w");

        if (host_m_file != NULL && host_a_file != NULL && host_b_file != NULL)
        {
            uint32_t* host_m = (uint32_t*) calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
            uint32_t* host_a = (uint32_t*) calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));
            uint32_t* host_b = (uint32_t*) calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

            if (host_m != NULL && host_a != NULL && host_b != NULL)
            {
                for (uint32_t i = 0; i < NUMBER_OF_BIGNUMS; i++)
                {
                    generate_random_bignum_modulus(host_m);
                    // generate_random_bignum_less_than(host_a, host_m);
                    // generate_random_bignum_less_than(host_b, host_m);

                    for (uint32_t j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
                    {
                        fprintf(host_m_file, "%u ", host_m[j]);
                        fprintf(host_a_file, "%u ", host_a[j]);
                        fprintf(host_b_file, "%u ", host_b[j]);
                    }

                    fprintf(host_m_file, "\n");
                    fprintf(host_a_file, "\n");
                    fprintf(host_b_file, "\n");
                }

                fclose(host_m_file);
                fclose(host_a_file);
                fclose(host_b_file);
            }
            else
            {
                if (host_m == NULL)
                {
                    printf("Error: could not allocate memory for bignum \"host_m\"\n");
                }

                if (host_a == NULL)
                {
                    printf("Error: could not allocate memory for bignum \"host_a\"\n");
                }

                if (host_b == NULL)
                {
                    printf("Error: could not allocate memory for bignum \"host_b\"\n");
                }

                exit(EXIT_FAILURE);
            }
        }
        else
        {
            if (host_m_file == NULL)
            {
                printf("Error: \"host_m_file\" is NULL\n");
            }

            if (host_a_file == NULL)
            {
                printf("Error: \"host_a_file\" is NULL\n");
            }

            if (host_b_file == NULL)
            {
                printf("Error: \"host_b_file\" is NULL\n");
            }

            exit(EXIT_FAILURE);
        }
    }
    else
    {
        if (host_m_file_name == NULL)
        {
            printf("Error: \"host_m_file_name\" is NULL\n");
            exit(EXIT_FAILURE);
        }

        if (host_a_file_name == NULL)
        {
            printf("Error: \"host_a_file_name\" is NULL\n");
            exit(EXIT_FAILURE);
        }

        if (host_b_file_name == NULL)
        {
            printf("Error: \"host_b_file_name\" is NULL\n");
            exit(EXIT_FAILURE);
        }
    }

    stop_random_number_generator();
}

void read_bignum_array_from_file(const char* file_name, uint32_t* bignum, uint32_t amount_to_read)
{
    if (file_name != NULL && bignum != NULL && amount_to_read > 0)
    {
        FILE* file = fopen(file_name, "r");

        if (file != NULL)
        {
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
        else
        {
            printf("Error: could not open file \"%s\" for reading\n", file_name);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        if (file_name == NULL)
        {
            printf("Error: \"file_name\" is NULL\n");
        }

        if (bignum == NULL)
        {
            printf("Error: bignum array \"bignum\" is NULL\n");
        }

        if (amount_to_read <= 0)
        {
            printf("Error: \"amount_to_read\" is negative\n");
        }

        exit(EXIT_FAILURE);
    }
}

void write_bignum_array_to_file(const char* file_name, uint32_t* bignum)
{
    if (file_name != NULL && bignum != NULL)
    {
        FILE* file = fopen(file_name, "w");

        if (file != NULL)
        {
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
        else
        {
            printf("Error: could not open file \"%s\" for writing\n", file_name);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        if (file_name == NULL)
        {
            printf("Error: \"file_name\" is NULL\n");
        }

        if (bignum == NULL)
        {
            printf("Error: bignum array \"bignum\" is NULL\n");
        }

        exit(EXIT_FAILURE);
    }
}
