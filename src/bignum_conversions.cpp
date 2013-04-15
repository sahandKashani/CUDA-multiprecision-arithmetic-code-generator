#include "bignum_conversions.h"
#include "bignum_types.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gmp.h>

bool is_binary_string(char* str);

char* uint32_t_to_binary_string(uint32_t number);
char* concatenate_binary_string_array(char** words);

char** bignum_to_binary_string_array(uint32_t* bignum);
char** binary_string_to_binary_string_array(char* str);

void pad_binary_string_with_char(char** old_str, char pad_character);
void flip_string_array(char** array);
void free_binary_string_words(char** words);
void transpose(uint32_t* m, int w, int h);

uint32_t number_of_elements_in_binary_string_array(char** array);
uint32_t binary_string_to_uint32_t(char* str);

/**
 * Checks if a string is a binary string.
 * @param  str string to be tested.
 * @return     true if the string is a binary string, and false otherwise.
 */
bool is_binary_string(char* str)
{
    assert(str != NULL);

    uint32_t str_length = strlen(str);
    for (uint32_t i = 0; i < str_length; i++)
    {
        if (str[i] != '0' && str[i] != '1')
        {
            return false;
        }
    }

    return true;
}

/**
 * Counts the number of elements in a NULL-terminated string array.
 * @param  array NULL-terminated array to be tested.
 * @return       number of elements in the array.
 */
uint32_t number_of_elements_in_binary_string_array(char** array)
{
    assert(array != NULL);

    uint32_t word_count = 0;

    for (char** element = array; *element != NULL; element++)
    {
        word_count++;
    }

    return word_count;
}

/**
 * Converts a uint32_t to a binary string. The returned string will always be
 * non-NULL and of length BITS_PER_WORD.
 * @param  number uint32_t to convert.
 * @return        binary string of length BITS_PER_WORD representing number.
 */
char* uint32_t_to_binary_string(uint32_t number)
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
    }
    else
    {
        printf("Error: could not allocate enough memory\n");
        exit(EXIT_FAILURE);
    }

    assert(str !=  NULL);
    assert(strlen(str) == BITS_PER_WORD);
    assert(is_binary_string(str));
    return str;
}

/**
 * Converts a bignum to a NULL-terminated array of binary strings, each of
 * length BITS_PER_WORD. The returned array will have a length of
 * (BIGNUM_NUMBER_OF_WORDS + 1), with the least significant bits of the bignum
 * at index 0, and the most significant bits at index (BIGNUM_NUMBER_OF_WORDS -
 * 1). Index BIGNUM_NUMBER_OF_WORDS contains the NULL terminator symbol.
 * @param  bignum bignum to convert.
 * @return        NULL-terminated binary string array representing the bignum.
 */
char** bignum_to_binary_string_array(uint32_t* bignum)
{
    assert(bignum != NULL);

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
                words[i] = uint32_t_to_binary_string(bignum[i]);
            }
            else
            {
                printf("Error: could not allocate enough memory\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        printf("Error: could not allocate enough memory\n");
        exit(EXIT_FAILURE);
    }

    assert(words != NULL);
    assert(number_of_elements_in_binary_string_array(words) == BIGNUM_NUMBER_OF_WORDS);
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        assert(is_binary_string(words[i]));
        assert(strlen(words[i]) == BITS_PER_WORD);
    }

    return words;
}

/**
 * Concatenates each of the binary strings in the NULL-terminated string array
 * "words" to form a single binary string of length TOTAL_BIT_LENGTH. The
 * "words" array must contain BIGNUM_NUMBER_OF_WORDS binary strings, each of
 * which is BITS_PER_WORD characters long.
 * @param  words binary string array to concatenate together.
 * @return       concatenated binary string.
 */
char* concatenate_binary_string_array(char** words)
{
    assert(words != NULL);
    assert(number_of_elements_in_binary_string_array(words) == BIGNUM_NUMBER_OF_WORDS);
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        assert(is_binary_string(words[i]));
        assert(strlen(words[i]) == BITS_PER_WORD);
    }

    // concatenate the words together to form a TOTAL_BIT_LENGTH long
    // string
    char* str = (char*) calloc(TOTAL_BIT_LENGTH + 1, sizeof(char));

    if (str != NULL)
    {
        str[TOTAL_BIT_LENGTH] = '\0';

        char* src;
        char* dest = str;
        for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
        {
            src = words[BIGNUM_NUMBER_OF_WORDS - i - 1];
            strncpy(dest, src, BITS_PER_WORD);

            dest += BITS_PER_WORD;
        }
    }
    else
    {
        printf("Error: could not allocate enough memory\n");
        exit(EXIT_FAILURE);
    }

    assert(str != NULL);
    assert(strlen(str) == TOTAL_BIT_LENGTH);
    assert(is_binary_string(str));
    return str;
}

/**
 * Converts a bignum to a binary string.
 * @param  bignum bignum to convert.
 * @return        binary string representation of the bignum.
 */
char* bignum_to_binary_string(uint32_t* bignum)
{
    assert(bignum != NULL);

    char** words = bignum_to_binary_string_array(bignum);
    char* str = concatenate_binary_string_array(words);

    free_binary_string_words(words);
    free(words);
    words = NULL;

    return str;
}

/**
 * Pads a binary string with '0's until it has length TOTAL_BIT_LENGTH. This
 * function reallocates the memory of "old_str" to have enough space for
 * TOTAL_BIT_LENGTH elements.
 * @param old_str string to pad with '0's.
 */
void pad_binary_string_with_zeros(char** old_str)
{
    assert(old_str != NULL);
    assert(*old_str != NULL);

    pad_binary_string_with_char(old_str, '0');
}

/**
 * Pads the binary string given as a parameter with pad_character until it has
 * length TOTAL_BIT_LENGTH. This function reallocates the memory of "old_str" to
 * have enough space for TOTAL_BIT_LENGTH elements. "old_str" must have <=
 * TOTAL_BIT_LENGTH characters when this function is called.
 * @param old_str       string to be padded with pad_character.
 * @param pad_character character to use as padding.
 */
void pad_binary_string_with_char(char** old_str, char pad_character)
{
    assert(old_str != NULL);
    assert(*old_str != NULL);
    assert(is_binary_string(*old_str));
    assert(strlen(*old_str) <= TOTAL_BIT_LENGTH);

    char* new_str = (char*) calloc(TOTAL_BIT_LENGTH + 1, sizeof(char));

    if (new_str != NULL)
    {
        new_str[TOTAL_BIT_LENGTH] = '\0';

        // write the pad character on the whole length
        for (uint32_t i = 0; i < TOTAL_BIT_LENGTH; i++)
        {
            new_str[i] = pad_character;
        }

        // copy the old string's elements over the padded string.
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
        printf("Error: could not allocate enough memory\n");
        exit(EXIT_FAILURE);
    }

    assert(*old_str != NULL);
    assert(strlen(*old_str) == TOTAL_BIT_LENGTH);
}

/**
 * Flips the string array.
 * @param array array to be flipped.
 */
void flip_string_array(char** array)
{
    assert(array != NULL);

    char* tmp;
    uint32_t middle_of_array = CEILING(BIGNUM_NUMBER_OF_WORDS, 2);

    for (uint32_t i = 0; i < middle_of_array; i++)
    {
        tmp = array[i];
        array[i] = array[BIGNUM_NUMBER_OF_WORDS - 1 - i];
        array[BIGNUM_NUMBER_OF_WORDS - 1 - i] = tmp;
    }
}

/**
 * Converts a binary string of length TOTAL_BIT_LENGTH to a NULL-terminated
 * binary string array with BIGNUM_NUMBER_OF_WORDS elements, each of length
 * BITS_PER_WORD.
 * @param  str binary string to be converted.
 * @return     binary string array after conversion.
 */
char** binary_string_to_binary_string_array(char* str)
{
    assert(str != NULL);
    assert(is_binary_string(str));
    assert(strlen(str) == TOTAL_BIT_LENGTH);

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

                // copy substring from str to str_words
                for (uint32_t j = 0; j < BITS_PER_WORD; j++)
                {
                    str_words[i][j] = str[i * BITS_PER_WORD + j];
                }
            }
            else
            {
                printf("Error: could not allocate enough memory\n");
                exit(EXIT_FAILURE);
            }
        }

        // until now, the strings have been cut in big-endian form, but we
        // want little-endian for indexing, so we have to invert the array.
        flip_string_array(str_words);
    }
    else
    {
        printf("Error: could not allocate enough memory\n");
        exit(EXIT_FAILURE);
    }

    assert(number_of_elements_in_binary_string_array(str_words) == BIGNUM_NUMBER_OF_WORDS);
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        assert(strlen(str_words[i]) == BITS_PER_WORD);
    }

    return str_words;
}

/**
 * Frees the NULL-terminated binary string array containing
 * BIGNUM_NUMBER_OF_WORDS elements, each of which is BITS_PER_WORD characters
 * long. This array is returned by a call to
 * bignum_to_binary_string_array(uint32_t* bignum).
 * @param words NULL-terminated binary string array to free.
 */
void free_binary_string_words(char** words)
{
    assert(words != NULL);
    assert(number_of_elements_in_binary_string_array(words) == BIGNUM_NUMBER_OF_WORDS);
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        assert(words[i] != NULL);
        assert(is_binary_string(words[i]));
        assert(strlen(words[i]) == BITS_PER_WORD);
    }

    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        free(words[i]);
        words[i] = NULL;
    }
}

/**
 * Converts a binary string of length BITS_PER_WORD to a uint32_t.
 * @param  str binary string to be converted.
 * @return     converted value.
 */
uint32_t binary_string_to_uint32_t(char* str)
{
    assert(str != NULL);
    assert(is_binary_string(str));
    assert(strlen(str) == BITS_PER_WORD);

    uint32_t number = 0;

    for (uint32_t i = 0; i < BITS_PER_WORD; i++)
    {
        uint32_t bit_value = str[BITS_PER_WORD - 1 - i] == '1' ? 1 : 0;
        number |= bit_value << i;
    }

    return number;
}

/**
 * Converts a binary string of length TOTAL_BIT_LENGTH to a bignum.
 * @param str    string to be converted.
 * @param number converted value.
 */
void binary_string_to_bignum(char* str, uint32_t* number)
{
    assert(str != NULL);
    assert(number != NULL);
    assert(is_binary_string(str));
    assert(strlen(str) == TOTAL_BIT_LENGTH);

    char** words = binary_string_to_binary_string_array(str);

    // set the number
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        number[i] = binary_string_to_uint32_t(words[i]);
    }

    free_binary_string_words(words);
    free(words);
    words = NULL;
}

/**
 * Transforms a bignum array's internal organization to the one of a coalesced
 * bignum array.
 * @param in bignum array to be transformed.
 */
void bignum_array_to_coalesced_bignum_array(uint32_t* in)
{
    assert(in != NULL);
    transpose(in, BIGNUM_NUMBER_OF_WORDS, NUMBER_OF_BIGNUMS);
}

/**
 * Transforms a coalesced bignum array's internal organization to the one of a
 * bignum array.
 * @param in coalesced bignum array to be transformed.
 */
void coalesced_bignum_array_to_bignum_array(uint32_t* in)
{
    assert(in != NULL);
    transpose(in, NUMBER_OF_BIGNUMS, BIGNUM_NUMBER_OF_WORDS);
}

/**
 * Prints the contents of a bignum array on the standard output.
 * @param in bignum array to be printed.
 */
void print_bignum_array(uint32_t* in)
{
    assert(in != NULL);

    uint32_t print_width = ceil(log10(pow(2, BITS_PER_WORD) - 1));

    for (uint32_t i = 0; i < NUMBER_OF_BIGNUMS; i++)
    {
        for (uint32_t j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
        {
            printf("%*u    ", print_width, in[IDX(i, j)]);
        }

        printf("\n");
    }
}

/**
 * Prints the contents of a coalesced bignum array on the standard output.
 * @param in coalesced bignum array to be printed.
 */
void print_coalesced_bignum_array(uint32_t* in)
{
    assert(in != NULL);

    uint32_t print_width = ceil(log10(pow(2, BITS_PER_WORD) - 1));

    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        for (uint32_t j = 0; j < NUMBER_OF_BIGNUMS; j++)
        {
            printf("%*u    ", print_width, in[COAL_IDX(i, j)]);
        }

        printf("\n");
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
    assert(m != NULL);

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

/**
 * Converts a positive mpz_t to a binary string. The length of the string is the
 * number's precision in bits. No padding has been done.
 * @param  number number to be converted.
 * @return        binary string representation of number with exact precision,
 *                with no leading 0s are kept.
 */
char* mpz_t_to_exact_precision_binary_string(mpz_t number)
{
    // number has to be positive
    assert(mpz_cmp_ui(number, 0) > 0);

    char* number_str = mpz_get_str(NULL, 2, number);
    if (number_str != NULL)
    {
        assert(strlen(number_str) < TOTAL_BIT_LENGTH);
        assert(is_binary_string(number_str));

        return number_str;
    }
    else
    {
        printf("Error: could not allocate enough memory\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * Converts a mpz_t to a binary string of length TOTAL_BIT_LENGTH. The number is
 * padded with 0s until it is TOTAL_BIT_LENGTH long.
 * @param  number number to be converted.
 * @return        binary string representation of the string padded with 0s to
 *                have length TOTAL_BIT_LENGTH.
 */
char* mpz_t_to_binary_string(mpz_t number)
{
    // number has to be positive
    assert(mpz_cmp_ui(number, 0) > 0);

    char* number_str = mpz_t_to_exact_precision_binary_string(number);
    pad_binary_string_with_zeros(&number_str);

    assert(strlen(number_str) == TOTAL_BIT_LENGTH);
    assert(is_binary_string(number_str));

    return number_str;
}

/**
 * Converts a mpz_t to a bignum.
 * @param number mpz_t to convert.
 * @param bignum converted value.
 */
void mpz_t_to_bignum(mpz_t number, uint32_t* bignum)
{
    assert(bignum != NULL);

    char* number_str = mpz_t_to_binary_string(number);
    binary_string_to_bignum(number_str, bignum);
}

/**
 * Converts a bignum to a mpz_t. The mpz_t given as a parameter MUST be
 * initialized before calling this function.
 * @param  bignum bignum to convert.
 * @return        converted value.
 */
void bignum_to_mpz_t(uint32_t* bignum, mpz_t number)
{
    assert(bignum != NULL);

    char* bignum_str = bignum_to_binary_string(bignum);

    uint32_t conversion_success = mpz_set_str(number, bignum_str, 2);
    if (conversion_success != 0)
    {
        printf("Error: gmp could not convert bignum string to gmp format\n");
        exit(EXIT_FAILURE);
    }

    free(bignum_str);
}

/**
 * Returns the precision of a bignum.
 * @param  bignum bignum to retrieve precision of.
 * @return        precision of the bignum.
 */
uint32_t get_bignum_precision(uint32_t* bignum)
{
    assert(bignum != NULL);

    mpz_t bignum_gmp;
    mpz_init(bignum_gmp);
    bignum_to_mpz_t(bignum, bignum_gmp);

    uint32_t precision = get_mpz_t_precision(bignum_gmp);

    mpz_clear(bignum_gmp);
    return precision;
}

/**
 * Returns the precision of a mpz_t.
 * @param  bignum mpz_t to retrieve precision of.
 * @return        precision of the mpz_t.
 */
uint32_t get_mpz_t_precision(mpz_t bignum)
{
    // bignum must be positive
    assert(mpz_cmp_ui(bignum, 0) > 0);

    char* bignum_str = mpz_t_to_exact_precision_binary_string(bignum);
    uint32_t precision = strlen(bignum_str);

    assert(precision > BITS_PER_WORD);

    free(bignum_str);
    return precision;
}
