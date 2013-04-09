#ifndef BIGNUM_TYPES_H
#define BIGNUM_TYPES_H

#include "constants.h"
#include <stdint.h>

#define BITS_PER_WORD 32
#define BIGNUM_NUMBER_OF_WORDS 5
#define TOTAL_BIT_LENGTH (BIGNUM_NUMBER_OF_WORDS * BITS_PER_WORD)

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// BIGNUM /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Little endian: most significant bits come in bignum[BIGNUM_NUMBER_OF_WORDS-1]
// and least significant bits come in bignum[0]. The radix is 2^BITS_PER_WORD

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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// COALESCED_BIGNUM ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Assume you have a bignum array "c", then the data in coalesced_bignum "c"
// would be:

//  c[0][0]   c[1][0]  ...  c[N-1][0]
//  c[0][1]   c[1][1]  ...  c[N-1][1]
//     .         .     .        .
//     .         .      .       .
//     .         .       .      .
// c[0][H-1] c[1][H-1] ... c[N-1][H-1]

// with N = number of bignums in the array
//      H = BIGNUM_NUMBER_OF_WORDS

// A bignum is written "vertically" instead of "horizontally" with this
// representation. Each column represents one bignum. The data on one "line" of
// a coalesced_bignum is a mix of the i'th element of N different bignums.

// We can see that coalesced_bignum* is actually a transposed version of bignum*

typedef uint32_t coalesced_bignum[TOTAL_NUMBER_OF_THREADS];

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// INTERLEAVED_BIGNUM ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Assume you have 2 bignum arrays "a" and "b", then the data in
// interleaved_bignum would be:

//  a[0][0]   b[0][0]   a[0][1]   b[0][1]  ...  a[0][H-1]   b[0][H-1]
//  a[1][0]   b[1][0]   a[1][1]   b[1][1]  ...  a[1][H-1]   b[1][H-1]
//     .         .         .         .     .        .           .
//     .         .         .         .      .       .           .
//     .         .         .         .       .      .           .
// a[N-1][0] b[N-1][0] a[N-1][1] b[N-1][1] ... a[N-1][H-1] b[N-1][H-1]

// with N = number of bignums in the array
//      H = BIGNUM_NUMBER_OF_WORDS

typedef uint32_t interleaved_bignum[2 * BIGNUM_NUMBER_OF_WORDS];

////////////////////////////////////////////////////////////////////////////////
//////////////////////// COALESCED_INTERLEAVED_BIGNUM //////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Assume you have 2 bignum arrays "a" and "b", then the data in
// coalesced_interleaved_bignum would be:

//  a[0][0]   b[0][0]   a[1][0]   b[1][0]  ...  a[N-1][0]   b[N-1][0]
//  a[0][1]   b[0][1]   a[1][1]   b[1][1]  ...  a[N-1][1]   b[N-1][1]
//     .         .         .         .     .        .           .
//     .         .         .         .      .       .           .
//     .         .         .         .       .      .           .
// a[0][H-1] b[0][H-1] a[1][H-1] b[1][H-1] ... a[N-1][H-1] b[N-1][H-1]

// with N = number of bignums in the array
//      H = BIGNUM_NUMBER_OF_WORDS

typedef uint32_t coalesced_interleaved_bignum[2 * TOTAL_NUMBER_OF_THREADS];

#endif
