#include "bignum_conversions.h"
#include "bignum_types.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Returns an binary string representation of a uint32_t. The string returned is
 * of length BITS_PER_WORD.
 * @param  number Number to be converted.
 * @return        String with the binary representation of number.
 */
char* uint32_t_to_string(uint32_t number)
{
    char* str = (char*) calloc(BITS_PER_WORD + 1, sizeof(char));

    if (str != NULL)
    {
        str[BITS_PER_WORD] = '\0';

        for (uint32_t i = 0; i < BITS_PER_WORD; i++)
        {
            uint32_t masked_number = number & (1 << i);
            str[BITS_PER_WORD - 1 - i] = (masked_number != 0) ? '1' : '0';
        }

        return str;
    }
    else
    {
        printf("Error: could not allocate memory for \"str\"\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Converts a bignum to a binary string. The returned string will have a length
 * of TOTAL_BIT_LENGTH. The bignum passed as a parameter must not be NULL, or
 * else the function will fail.
 * @param  number Bignum to be converted.
 * @return        Binary string representation of the bignum.
 */
char* bignum_to_string(uint32_t* number)
{
    if (number != NULL)
    {
        // make an array of strings which will each contain 1 of the
        // BIGNUM_NUMBER_OF_WORDS words in the bignum
        char** words = (char**) calloc(BIGNUM_NUMBER_OF_WORDS + 1, sizeof(char*));

        if (words != NULL)
        {
            words[BIGNUM_NUMBER_OF_WORDS] = NULL;

            for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
            {
                words[i] = (char*) calloc(BITS_PER_WORD + 1, sizeof(char));

                if (words[i] != NULL)
                {
                    words[i][BITS_PER_WORD] = '\0';

                    // convert each bignum element to a string
                    words[i] = uint32_t_to_string(number[i]);
                }
                else
                {
                    printf("Error: could not allocate memory for \"words[%u]\"\n", i);
                    exit(EXIT_FAILURE);
                }
            }

            // concatenate the words together to form a TOTAL_BIT_LENGTH long
            // string
            char* final_str = (char*) calloc(TOTAL_BIT_LENGTH + 1, sizeof(char));

            if (final_str != NULL)
            {
                final_str[TOTAL_BIT_LENGTH] = '\0';

                char* src;
                char* dest = final_str;
                for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
                {
                    src = words[BIGNUM_NUMBER_OF_WORDS - i - 1];
                    strncpy(dest, src, BITS_PER_WORD);

                    dest += BITS_PER_WORD;
                }

                free_string_words(&words);

                // return concatenated string
                return final_str;
            }
            else
            {
                printf("Error: could not allocate memory for \"final_str\"\n");
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            printf("Error: could not allocate memory for \"words\"\n");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        printf("Error: bignum \"number\" is NULL\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Pads the binary string with zeros until it is TOTAL_BIT_LENGTH long.
 * @param old_str String to be padded with zeros.
 */
void pad_string_with_zeros(char** old_str)
{
    if (old_str != NULL)
    {
        char* new_str = (char*) calloc(TOTAL_BIT_LENGTH + 1, sizeof(char));

        if (new_str != NULL)
        {
            new_str[TOTAL_BIT_LENGTH] = '\0';
            for (uint32_t i = 0; i < TOTAL_BIT_LENGTH; i++)
            {
                new_str[i] = '0';
            }

            uint32_t old_str_length = strlen(*old_str);

            for (uint32_t i = 0; i < old_str_length; i++)
            {
                new_str[(TOTAL_BIT_LENGTH - old_str_length) + i] = (*old_str)[i];
            }

            free(*old_str);
            *old_str = new_str;
        }
        else
        {
            printf("Error: could not allocate memory for \"new_str\"\n");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        printf("Error: \"old_str\" is NULL");
        exit(EXIT_FAILURE);
    }
}

/**
 * Separates a TOTAL_BIT_LENGTH long binary string into an array of binary
 * strings, each of length BITS_PER_WORD. The returned array contains the most
 * significant bits at its last index, and its least significant bits at its
 * first position. Ex: str_words[0] contains bits (BITS_PER_WORD downto 0).
 * @param  str Binary string to be decomposed.
 * @return     Decomposed string.
 */
char** cut_string_to_multiple_words(char* str)
{
    if (str != NULL)
    {
        // cut str into BIGNUM_NUMBER_OF_WORDS pieces, each of which is
        // BITS_PER_WORD long

        // array of BITS_PER_WORD length strings
        char** str_words = (char**) calloc(BIGNUM_NUMBER_OF_WORDS + 1, sizeof(char*));

        if (str_words != NULL)
        {
            str_words[BIGNUM_NUMBER_OF_WORDS] = NULL;

            // allocate each one of the strings and fill them up
            for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
            {
                str_words[i] = (char*) calloc(BITS_PER_WORD + 1, sizeof(char));

                if (str_words[i] != NULL)
                {
                    str_words[i][BITS_PER_WORD] = '\0';

                    for (uint32_t j = 0; j < BITS_PER_WORD; j++)
                    {
                        str_words[i][j] = str[i * BITS_PER_WORD + j];
                    }
                }
                else
                {
                    printf("Error: could not allocate memory for \"str_words[%u]\"\n", i);
                    exit(EXIT_FAILURE);
                }
            }

            // until now, the strings have been cut in big-endian form, but we
            // want little-endian for indexing, so we have to invert the array.
            char* tmp;
            uint32_t middle_of_array = CEILING(BIGNUM_NUMBER_OF_WORDS, 2);
            for (uint32_t i = 0; i < middle_of_array; i++)
            {
                tmp = str_words[i];
                str_words[i] = str_words[BIGNUM_NUMBER_OF_WORDS - 1 - i];
                str_words[BIGNUM_NUMBER_OF_WORDS - 1 - i] = tmp;
            }

            return str_words;
        }
        else
        {
            printf("Error: could not allocate memory for \"str_words\"\n");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        printf("Error: \"str\" is NULL");
        exit(EXIT_FAILURE);
    }
}

/**
 * Frees an array containing BIGNUM_NUMBER_OF_WORDS strings.
 * @param words Pointer to the array of strings to be freed.
 */
void free_string_words(char*** words)
{
    if (*words != NULL)
    {
        for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
        {
            if ((*words)[i] != NULL)
            {
                free((*words)[i]);
                (*words)[i] = NULL;
            }
        }

        // free the char** pointing to the words
        free(*words);
        *words = NULL;
    }
}

/**
 * Returns a uint32_t representation of the binary string of length
 * BITS_PER_WORD passed as a parameter.
 * @param  str Binary string to be converted.
 * @return     Converted value.
 */
uint32_t string_to_uint32_t(char* str)
{
    if (str != NULL)
    {
        uint32_t number = 0;

        for (uint32_t i = 0; i < BITS_PER_WORD; i++)
        {
            uint32_t bit_value = str[BITS_PER_WORD - 1 - i] == '1' ? 1 : 0;
            number |= bit_value << i;
        }

        return number;
    }
    else
    {
        printf("Error: \"str\" is NULL\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Converts the "str" binary string of length TOTAL_BITS_LENGTH into a bignum
 * and stores the result in the bignum passed as a parameter. The bignum must
 * not be NULL, or else the function will fail.
 * @param str    Binary string to be converted.
 * @param number Number which will be assigned the value of the binary string
 * str.
 */
void string_to_bignum(char* str, uint32_t* number)
{
    if (str != NULL && number != NULL)
    {
        char** words = cut_string_to_multiple_words(str);

        // set the number
        for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
        {
            number[i] = string_to_uint32_t(words[i]);
        }

        free_string_words(&words);
    }
    else
    {
        if (str == NULL)
        {
            printf("Error: \"str\" is NULL\n");
        }

        if (number == NULL)
        {
            printf("Error: bignum \"number\" is NULL\n");
        }

        exit(EXIT_FAILURE);
    }
}

/**
 * Transforms an array of bignums from its non-coalesced representation to its
 * coalesced representation.
 * @param  in Bignum array to transform from its non-coalesced representation.
 */
void bignum_array_to_coalesced_bignum_array(uint32_t* in)
{
    if (in != NULL)
    {
        transpose(in, BIGNUM_NUMBER_OF_WORDS, NUMBER_OF_BIGNUMS);
    }
    else
    {
        printf("Error: bignum array \"in\" is NULL\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Transforms an array of bignums from its coalesced representation to its
 * non-coalesced representation.
 * @param  in Bignum array to transform from its coalesced representation.
 */
void coalesced_bignum_array_to_bignum_array(uint32_t* in)
{
    if (in != NULL)
    {
        transpose(in, NUMBER_OF_BIGNUMS, BIGNUM_NUMBER_OF_WORDS);
    }
    else
    {
        printf("Error: bignum array \"in\" is NULL\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Prints a bignum array.
 * @param in Bignum array to be printed.
 */
void print_bignum_array(uint32_t* in)
{
    if (in != NULL)
    {
        for (uint32_t i = 0; i < NUMBER_OF_BIGNUMS; i++)
        {
            for (uint32_t j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
            {
                printf("%11u    ", in[IDX(i, j)]);
            }

            printf("\n");
        }
    }
    else
    {
        printf("Error: bignum array \"in\" is NULL\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Prints a coalesced bignum array.
 * @param in Coalesced bignum array to be printed.
 */
void print_coalesced_bignum_array(uint32_t* in)
{
    if (in != NULL)
    {
        for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
        {
            for (uint32_t j = 0; j < NUMBER_OF_BIGNUMS; j++)
            {
                printf("%11u    ", in[COAL_IDX(i, j)]);
            }

            printf("\n");
        }
    }
    else
    {
        printf("Error: bignum array \"in\" is NULL\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Transposes a 2D matrix of width "w" and height "h" in-place. All credit goes
 * to "http://rosettacode.org/wiki/Matrix_transposition#C" for the following
 * in-place matrix transposition code.
 * @param m Matrix to be transposed.
 * @param w Width of the matrix.
 * @param h Height of the matrix.
 */
void transpose(uint32_t* m, int w, int h)
{
    int start, next, i;
    uint32_t tmp;

    for (start = 0; start <= w * h - 1; start++)
    {
        next = start;
        i = 0;
        do
        {
            i++;
            next = (next % h) * w + next / h;
        }
        while (next > start);

        if (next < start || i == 1)
        {
            continue;
        }

        tmp = m[next = start];

        do
        {
            i = (next % h) * w + next / h;
            m[next] = (i == start) ? tmp : m[i];
            next = i;
        }
        while (next > start);
    }
}
