#ifndef BIGNUM_TYPE_H
#define BIGNUM_TYPE_H

#include <stdint.h>

#define BITS_PER_WORD 32
#define BIGNUM_NUMBER_OF_WORDS 5
#define TOTAL_BIT_LENGTH (BIGNUM_NUMBER_OF_WORDS * BITS_PER_WORD)

// Little endian: most significant bits come in bignum[BIGNUM_NUMBER_OF_WORDS-1]
// and least significant bits come in bignum[0].

// The radix is 2^BITS_PER_WORD

// Assume you have a bignum array "c", then the data would be represented as:

//  c[0][0]   c[0][1]  ...  c[0][H-1]
//  c[1][0]   c[1][1]  ...  c[1][H-1]
//     .         .     .        .
//     .         .      .       .
//     .         .       .      .
// c[N-1][0] c[N-1][1] ... c[N-1][H-1]

// with N = number of bignums in the array
//      H = BIGNUM_NUMBER_OF_WORDS

// A bignum is written "horizontally". The data on one "line" of a bignum
// consists of the BIGNUM_NUMBER_OF_WORDS elements of the bignum.

typedef uint32_t bignum[BIGNUM_NUMBER_OF_WORDS];

#endif
