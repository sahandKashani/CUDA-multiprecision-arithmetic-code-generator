#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#define BITS_PER_WORD 32
#define BIGNUM_NUMBER_OF_WORDS 5
#define TOTAL_BIT_LENGTH BIGNUM_NUMBER_OF_WORDS * BITS_PER_WORD
#define SEED ((unsigned int) 12345)
#define RANDOM_NUMBER_BIT_RANGE ((unsigned int) 131)
#define BASE 2

// Note: all functions work on little endian word representations (not byte
// representation).

// Note: we only use words in our functions, not bytes. When we talk about
// endianness, we are talking about word endianness, and not byte endianness.

// Note: strings are written as they are read, from left to right with MSB on
// the left and LSB on the right. They are not divided into
// BIGNUM_NUMBER_OF_WORDS parts, each of which is BITS_PER_WORD bits long. The
// strings are actually TOTAL_BIT_LENGTH in length.

// little endian: most significant bits come in bignum[4] and least significant
// bits come in bignum[0]
typedef unsigned int bignum[BIGNUM_NUMBER_OF_WORDS];

char* unsigned_int_to_string(unsigned int number);
char* bignum_to_string(bignum number);

char** cut_string_to_multiple_words(char* str);

void pad_string_with_zeros(char** old_str);
void free_string_words(char*** words);
void string_to_bignum(char* str, bignum number);

unsigned int string_to_unsigned_int(char* str);

#endif
