#include "io_interface.h"
#include "bignum_types.h"
#include "random_bignum_generator.h"
#include "constants.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void read_bignum_arrays_from_files(uint32_t* a, uint32_t* b, const char* file_name_1, const char* file_name_2)
{
    if (a != NULL && b != NULL && file_name_1 != NULL && file_name_2 != NULL)
    {
        printf("reading bignums from files ... ");
        fflush(stdout);
        read_bignum_array_from_file(file_name_1, a, NUMBER_OF_BIGNUMS);
        read_bignum_array_from_file(file_name_2, b, NUMBER_OF_BIGNUMS);
        printf("done\n");
        fflush(stdout);
    }
    else
    {
        if (a == NULL)
        {
            printf("Error: bignum array \"a\" is NULL\n");
        }

        if (b == NULL)
        {
            printf("Error: bignum array \"b\" is NULL\n");
        }

        if (file_name_1 == NULL)
        {
            printf("Error: \"file_name_1\" is NULL\n");
        }

        if (file_name_2 == NULL)
        {
            printf("Error: \"file_name_2\" is NULL\n");
        }

        exit(EXIT_FAILURE);
    }
}

void generate_random_bignum_arrays_to_files(const char* file_name_1, const char* file_name_2)
{
    if (file_name_1 != NULL && file_name_2 != NULL)
    {
        start_random_number_generator();

        generate_random_bignum_array_to_file(file_name_1);
        generate_random_bignum_array_to_file(file_name_2);

        stop_random_number_generator();
    }
    else
    {
        if (file_name_1 == NULL)
        {
            printf("Error: \"file_name_1\" is NULL\n");
        }

        if (file_name_2 == NULL)
        {
            printf("Error: \"file_name_2\" is NULL\n");
        }

        exit(EXIT_FAILURE);
    }
}

void generate_random_bignum_array_to_file(const char* file_name)
{
    if (file_name != NULL)
    {
        FILE* file = fopen(file_name, "w");

        if (file != NULL)
        {
            uint32_t* bignum = (uint32_t*) calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(uint32_t));

            if (bignum != NULL)
            {
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
            else
            {
                printf("Error: could not allocate memory for generated bignum\n");
                exit(EXIT_FAILURE);
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
        printf("Error: \"file_name\" is NULL\n");
        exit(EXIT_FAILURE);
    }
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
