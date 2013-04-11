#ifndef BIGNUM_CONVERSIONS_H
#define BIGNUM_CONVERSIONS_H

#include <stdint.h>

// Note: strings are written as they are read, from left to right with MSB on
// the left and LSB on the right. The strings have length TOTAL_BIT_LENGTH.

char* uint32_t_to_string(uint32_t number);
char* bignum_to_string(uint32_t* number);

char** cut_string_to_multiple_words(char* str);

uint32_t string_to_uint32_t(char* str);

void pad_string_with_zeros(char** old_str);
void free_string_words(char*** words);
void string_to_bignum(char* str, uint32_t* number);
void bignum_array_to_coalesced_bignum_array(uint32_t** in);
void coalesced_bignum_array_to_bignum_array(uint32_t** in);
void print_bignum_array(uint32_t* in);
void print_coalesced_bignum_array(uint32_t* in);

#endif
