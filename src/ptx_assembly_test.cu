#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gmp.h>

#define BITS_PER_WORD 32
#define BIGNUM_NUMBER_OF_WORDS 5
#define TOTAL_BIT_LENGTH BIGNUM_NUMBER_OF_WORDS * BITS_PER_WORD
#define SEED ((unsigned int) 12345)
#define RANDOM_NUMBER_BIT_RANGE ((unsigned int) 131)
#define BASE 2

// Note : strings are written as they are read, from left to right with MSB on
// the left and LSB on the right. They are not divided into
// BIGNUM_NUMBER_OF_WORDS parts, each of which is BITS_PER_WORD bits long. The
// strings are actually TOTAL_BIT_LENGTH in length.

// little endian: most significant bits come in bignum[4] and least significant
// bits come in bignum[0]
typedef unsigned int bignum[BIGNUM_NUMBER_OF_WORDS];

/////////////////////////
// Function Prototypes //
/////////////////////////
// __global__ void test_kernel(int* dev_c, int a, int b);
char* bignum_to_string(bignum number);
void print_bignum(bignum number);
void pad_string_with_zeros(char** old_str);
char** cut_string_to_multiple_words(char* str);
void free_string_words(char*** words);
void string_to_bignum(char* str, bignum number);
char* unsigned_int_to_string(unsigned int number);
unsigned int string_to_unsigned_int(char* str);
char* generate_random_number(unsigned int index, unsigned int seed,
                             unsigned int bits, unsigned int base);

//////////////
// Launcher //
//////////////
int main(void)
{
    printf("Testing inline PTX\n");

    int i = 0;
    char* number_str;

    number_str = generate_random_number(i++, SEED, RANDOM_NUMBER_BIT_RANGE, BASE);

    bignum a;
    string_to_bignum(number_str, a);
    free(number_str);

    // cudaMalloc((void**) &dev_c, sizeof(int));
    // test_kernel<<<1, 1>>>(dev_c, a, b);
    // cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // printf("%d + %d = %d\n", a, b, c);
    // cudaFree(dev_c);
}

char* unsigned_int_to_string(unsigned int number)
{
    char* str = calloc(BITS_PER_WORD + 1, sizeof(char));
    str[BITS_PER_WORD] = '\0';

    for (int i = 0; i < BITS_PER_WORD; i++)
    {
        unsigned int masked_number = number & (1 << i);
        str[BITS_PER_WORD - 1 - i] = (masked_number != 0) ? '1' : '0';
    }

    return str;
}

char* bignum_to_string(bignum number)
{
    char** words = calloc(BIGNUM_NUMBER_OF_WORDS + 1, sizeof(char*));
    words[BIGNUM_NUMBER_OF_WORDS] = NULL;

    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        words[i] = calloc(BITS_PER_WORD + 1, sizeof(char));
        words[i][BITS_PER_WORD] = '\0';

        words[i] = unsigned_int_to_string(number[i]);
    }

    // concatenate the words together to form a TOTAL_BIT_LENGTH long string
    char* final_str = calloc(TOTAL_BIT_LENGTH + 1, sizeof(char));
    final_str[TOTAL_BIT_LENGTH] = '\0';

    char* src;
    char* dest = final_str;
    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        src = words[BIGNUM_NUMBER_OF_WORDS - i - 1];
        strncpy(dest, src, BITS_PER_WORD);

        dest += i * BITS_PER_WORD;
    }

    free_string_words(&words);

    return final_str;
}

void pad_string_with_zeros(char** old_str)
{
    char* new_str = calloc(TOTAL_BIT_LENGTH + 1, sizeof(char));
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

char** cut_string_to_multiple_words(char* str)
{
    // cut str into BIGNUM_NUMBER_OF_WORDS pieces, each of which is
    // BITS_PER_WORD long

    // array of BITS_PER_WORD length strings
    char** str_words = calloc(BIGNUM_NUMBER_OF_WORDS + 1, sizeof(char*));
    str_words[BIGNUM_NUMBER_OF_WORDS] = NULL;

    // allocate each one of the strings and fill them up
    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        str_words[i] = calloc(BITS_PER_WORD + 1, sizeof(char));
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

void free_string_words(char*** words)
{
    for (int i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        // free each word
        free((*words)[i]);
        (*words)[i] = NULL;
    }

    // free the char** pointing to the words
    free(*words);
    *words = NULL;
}

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

/**
 * Generates the i'th random number from the seed, where "i" is the "index"
 * value passed as a parameter. Remember to call free() on the returned string
 * once you don't need it anymore.
 * @param  index "Index" of the random number.
 * @param  seed  Seed of the random number generator.
 * @param  bits  Bit precision requested.
 * @param  base  Base of the number returned in the string (2 until 62)
 * @return       String representing the binary version of the number.
 */
char* generate_random_number(unsigned int index, unsigned int seed,
                             unsigned int bits, unsigned int base)
{
    // random number generator initialization
    gmp_randstate_t random_state;
    gmp_randinit_default(random_state);
    // incorporated seed in generator
    gmp_randseed_ui(random_state, seed);

    // initialize test vector operands and result
    mpz_t number;
    mpz_init(number);

    // generate random number
    mpz_urandomb(number, random_state, bits);
    for (int i = 0; i < index; i++)
    {
        mpz_urandomb(number, random_state, bits);
    }

    // get binary string version
    char* str_number = mpz_get_str(NULL, base, number);
    pad_string_with_zeros(&str_number);

    // get memory back from operands and results
    mpz_clear(number);

    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);

    return str_number;
}

// __global__ void test_kernel(int* dev_c, int a, int b)
// {
//     int c;

//     asm("{"
//         "    add.u32 %0, %1, %2;"
//         "}"
//         : "=r"(c) : "r"(a), "r"(b)
//         );

//     *dev_c = c;
// }
