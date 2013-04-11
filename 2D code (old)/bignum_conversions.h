#ifndef BIGNUM_CONVERSIONS_H
#define BIGNUM_CONVERSIONS_H

#include "bignum_types.h"
#include <stdint.h>

// Note: all functions work on little endian word representations (not byte
// representation).

// Note: we only use words in our functions, not bytes. When we talk about
// endianness, we are talking about word endianness, and not byte endianness.

// Note: strings are written as they are read, from left to right with MSB on
// the left and LSB on the right. The strings have length TOTAL_BIT_LENGTH

char* uint32_t_to_string(uint32_t number);
char* bignum_to_string(bignum number);

char** cut_string_to_multiple_words(char* str);

void pad_string_with_zeros(char** old_str);
void free_string_words(char*** words);
void string_to_bignum(char* str, bignum number);

uint32_t string_to_uint32_t(char* str);

coalesced_bignum* bignum_to_coalesced_bignum(bignum** in);
interleaved_bignum* bignums_to_interleaved_bignum(bignum** in_1, bignum** in_2);
coalesced_interleaved_bignum* bignums_to_coalesced_interleaved_bignum(bignum** in_1, bignum** in_2);

bignum* coalesced_bignum_to_bignum(coalesced_bignum** in);
void interleaved_bignum_to_bignums(bignum** out_1, bignum** out_2, interleaved_bignum** in);
void coalesced_interleaved_bignum_to_bignums(bignum** out_1, bignum** out_2, coalesced_interleaved_bignum** in);

#endif
