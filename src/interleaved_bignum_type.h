#ifndef INTERLEAVED_BIGNUM_TYPE_H
#define INTERLEAVED_BIGNUM_TYPE_H

#include "bignum_type.h"

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

#endif
