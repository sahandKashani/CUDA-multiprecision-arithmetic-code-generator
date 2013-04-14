#ifndef BIGNUM_CONVERSIONS_H
#define BIGNUM_CONVERSIONS_H

#include <stdint.h>

// Note: strings are written as they are read, from left to right with MSB on
// the left and LSB on the right. The strings have length TOTAL_BIT_LENGTH.

char* bignum_to_binary_string(uint32_t* bignum);

void pad_binary_string_with_zeros(char** old_str);
void binary_string_to_bignum(char* str, uint32_t* number);
void bignum_array_to_coalesced_bignum_array(uint32_t* in);
void coalesced_bignum_array_to_bignum_array(uint32_t* in);
void print_bignum_array(uint32_t* in);
void print_coalesced_bignum_array(uint32_t* in);

#endif
