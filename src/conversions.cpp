#include "conversions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Returns an binary string representation of an unsigned integer. The string
 * returned is of length BITS_PER_WORD.
 * @param  number Unsigned integer to be converted.
 * @return        String with the binary representation of number
 */
char* unsigned_int_to_string(unsigned int number)
{
    char* str = (char*) calloc(BITS_PER_WORD + 1, sizeof(char));
    str[BITS_PER_WORD] = '\0';

    for (int i = 0; i < BITS_PER_WORD; i++)
    {
        unsigned int masked_number = number & (1 << i);
        str[BITS_PER_WORD - 1 - i] = (masked_number != 0) ? '1' : '0';
    }

    return str;
}

/**
 * Converts a bignum to a binary string. The returned string will have a length
 * of TOTAL_BIT_LENGTH.
 * @param  number Number to be converted.
 * @return        Binary string representation of the bignum.
 */
char* bignum_to_string(bignum number)
{
    // make an array of strings which will each contain 1 of the
    // BIGNUM_NUMBER_OF_WORDS words in the bignum
    char** words = (char**) calloc(BIGNUM_NUMBER_OF_WORDS + 1, sizeof(char*));
    words[BIGNUM_NUMBER_OF_WORDS] = NULL;

    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        words[i] = (char*) calloc(BITS_PER_WORD + 1, sizeof(char));
        words[i][BITS_PER_WORD] = '\0';

        // convert each bignum element to a string
        words[i] = unsigned_int_to_string(number[i]);
    }

    // concatenate the words together to form a TOTAL_BIT_LENGTH long string
    char* final_str = (char*) calloc(TOTAL_BIT_LENGTH + 1, sizeof(char));
    final_str[TOTAL_BIT_LENGTH] = '\0';

    char* src;
    char* dest = final_str;
    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        src = words[BIGNUM_NUMBER_OF_WORDS - i - 1];
        strncpy(dest, src, BITS_PER_WORD);

        dest += BITS_PER_WORD;
    }

    free_string_words(&words);

    // return concatenated string
    return final_str;
}

/**
 * Pads the binary string with zeros until it is TOTAL_BIT_LENGTH long.
 * @param old_str [description]
 */
void pad_string_with_zeros(char** old_str)
{
    char* new_str = (char*) calloc(TOTAL_BIT_LENGTH + 1, sizeof(char));
    new_str[TOTAL_BIT_LENGTH] = '\0';
    for (int i = 0; i < TOTAL_BIT_LENGTH; i++)
    {
        new_str[i] = '0';
    }

    unsigned int old_str_length = strlen(*old_str);

    for (int i = 0; i < old_str_length; i++)
    {
        new_str[(TOTAL_BIT_LENGTH - old_str_length) + i] = (*old_str)[i];
    }

    free(*old_str);
    *old_str = new_str;
}

/**
 * Separates a TOTAL_BIT_LENGTH long binary string into an array of binary
 * strings of length BITS_PER_WORD. The returned array contains the most
 * significant bits at its last index, and its least significant bits at its
 * first position. Ex: str_words[0] contains bits (BITS_PER_WORD downto 0)
 * @param  str String to be decomposed.
 * @return     Decomposed string.
 */
char** cut_string_to_multiple_words(char* str)
{
    // cut str into BIGNUM_NUMBER_OF_WORDS pieces, each of which is
    // BITS_PER_WORD long

    // array of BITS_PER_WORD length strings
    char** str_words = (char**) calloc(BIGNUM_NUMBER_OF_WORDS + 1,
                                       sizeof(char*));
    str_words[BIGNUM_NUMBER_OF_WORDS] = NULL;

    // allocate each one of the strings and fill them up
    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        str_words[i] = (char*) calloc(BITS_PER_WORD + 1, sizeof(char));
        str_words[i][BITS_PER_WORD] = '\0';

        for (int j = 0; j < BITS_PER_WORD; j++)
        {
            str_words[i][j] = str[i * BITS_PER_WORD + j];
        }
    }

    // until now, the strings have been cut in big-endian form, but we want
    // little endian for indexing, so we have to invert the array.
    char* tmp;
    int middle_of_array = ceil(BIGNUM_NUMBER_OF_WORDS / 2);
    for (int i = 0; i < middle_of_array; i++)
    {
        tmp = str_words[i];
        str_words[i] = str_words[BIGNUM_NUMBER_OF_WORDS - 1 - i];
        str_words[BIGNUM_NUMBER_OF_WORDS - 1 - i] = tmp;
    }

    return str_words;
}

/**
 * Frees an array containing BIGNUM_NUMBER_OF_WORDS strings.
 * @param words Pointer to the array of strings to be freed.
 */
void free_string_words(char*** words)
{
    if (*words != NULL)
    {
        for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
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
 * Returns an unsigned integer representation of the binary string of length
 * BITS_PER_WORD passed as a parameter.
 * @param  str String to be converted.
 * @return     Converted value.
 */
unsigned int string_to_unsigned_int(char* str)
{
    unsigned int number = 0;

    for (int i = 0; i < BITS_PER_WORD; i++)
    {
        unsigned int bit_value = str[BITS_PER_WORD - 1 - i] == '1' ? 1 : 0;
        number |= bit_value << i;
    }

    return number;
}

/**
 * Converts the "str" binary string of length TOTAL_BITS_LENGTH into a bignum.
 * @param str    String to be converted.
 * @param number Number which will be assigned the value of the binary string
 *               str.
 */
void string_to_bignum(char* str, bignum number)
{
    char** words = cut_string_to_multiple_words(str);

    // set the number
    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        number[i] = string_to_unsigned_int(words[i]);
    }

    free_string_words(&words);
}
