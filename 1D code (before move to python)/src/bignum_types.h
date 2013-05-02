#ifndef BIGNUM_TYPES_H
#define BIGNUM_TYPES_H

#include "constants.h"
#include <stdint.h>

#define BITS_PER_WORD 32
#define BIT_RANGE 33
#define MIN_BIGNUM_NUMBER_OF_WORDS 2
#define MAX_BIGNUM_NUMBER_OF_WORDS 3
#define TOTAL_BIT_LENGTH 96

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// BIGNUM /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// A bignum is represented as the following data structure:
// uint32_t[MAX_BIGNUM_NUMBER_OF_WORDS]

// In the code of this project, there will be no "bignum" type. It will only be
// referred to as a uint32_t*. This is needed, because having direct access to
// the inner representation of a bignum will be useful for efficient operations
// such as matrix transpositions, ...

// The code of this project will not have a bignum's size as a parameter to
// functions. This value is accessible throught the macros of this header file.

// A bignum is represented in "little endian" format: the most significant bits
// come in bignum[MAX_BIGNUM_NUMBER_OF_WORDS - 1] and the least significant bits
// come in bignum[0].

// A bignum's radix is 2^BITS_PER_WORD.

// Assume you have a bignum array "c", then the data would be conceptually
// represented as:

//  c[0][0]   c[0][1]  ...  c[0][H-1]
//  c[1][0]   c[1][1]  ...  c[1][H-1]
//     .         .     .        .
//     .         .      .       .
//     .         .       .      .
// c[N-1][0] c[N-1][1] ... c[N-1][H-1]

// with N = NUMBER_OF_BIGNUMS
//      H = MAX_BIGNUM_NUMBER_OF_WORDS

// A bignum is written "horizontally". The data on one "line" of a bignum
// consists of the MAX_BIGNUM_NUMBER_OF_WORDS elements of the bignum.

// For memory alignment issues, an array of bignums will not be represented as a
// 2D array like uint32_t[N][H], but rather as a flattened 1D array like
// uint32_t[N * H]. Index manipulation will be needed to access the array like a
// 2D array.

// Assuming the human readable 2D standard bignum array representation above,
// the following macro returns the index of the "j"th element of the "i"th
// bignum from a 1D array of size N * H (N and H defined as below).

// 0 < i < N = NUMBER_OF_BIGNUMS
// 0 < j < H = MAX_BIGNUM_NUMBER_OF_WORDS
#define IDX(i, j) (((i) * (MAX_BIGNUM_NUMBER_OF_WORDS)) + (j))

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// COALESCED_BIGNUM ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// For efficient access to operands in gpu global memory, data needs to be
// accessed in a coalesced way. This is easily achieved by transposing an array
// of bignums to have the following representation:

// Assume you have a bignum array "c", then the data in a coalesced bignum "c"
// would be:

//  c[0][0]   c[1][0]  ...  c[N-1][0]
//  c[0][1]   c[1][1]  ...  c[N-1][1]
//     .         .     .        .
//     .         .      .       .
//     .         .       .      .
// c[0][H-1] c[1][H-1] ... c[N-1][H-1]

// with N = NUMBER_OF_BIGNUMS
//      H = MAX_BIGNUM_NUMBER_OF_WORDS

// A bignum is written "vertically" instead of "horizontally" with this
// representation. Each column represents one bignum. The data on one "line" of
// a coalesced bignum is a mix of the j'th element of N different bignums.

// As for normal bignums, a coalesced bignum will be represented as a flattened
// 1D array like uint32_t[N * H], and index manipulation would be neeeded to
// access the array like a 2D array.

// Assuming the human readable 2D coalesced bignum array representation above,
// the following macro returns the index of the "i"th element of the "j"th
// bignum from a 1D array of size N * H (N and H defined as below).

// 0 < i < H = MAX_BIGNUM_NUMBER_OF_WORDS
// 0 < j < N = NUMBER_OF_BIGNUMS
#define COAL_IDX(i, j) (((i) * (NUMBER_OF_BIGNUMS)) + (j))

#endif