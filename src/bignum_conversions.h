#ifndef BIGNUM_CONVERSIONS_H
#define BIGNUM_CONVERSIONS_H

#include <stdint.h>

#include "bignum_type.h"
#include "coalesced_bignum_type.h"

// Note: all functions work on little endian word representations (not byte
// representation).

// Note: we only use words in our functions, not bytes. When we talk about
// endianness, we are talking about word endianness, and not byte endianness.

// Note: strings are written as they are read, from left to right with MSB on
// the left and LSB on the right. They are not divided into
// BIGNUM_NUMBER_OF_WORDS parts, each of which is BITS_PER_WORD bits long. The
// strings are actually TOTAL_BIT_LENGTH in length.

char* uint32_t_to_string(uint32_t number);
char* bignum_to_string(bignum number);

char** cut_string_to_multiple_words(char* str);

void pad_string_with_zeros(char** old_str);
void free_string_words(char*** words);
void string_to_bignum(char* str, bignum number);

uint32_t string_to_uint32_t(char* str);

bignum* coalesced_bignum_to_bignum(coalesced_bignum** a);
coalesced_bignum* bignum_to_coalesced_bignum(bignum** a);

#endif
