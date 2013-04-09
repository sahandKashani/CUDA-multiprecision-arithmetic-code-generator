#ifndef COALESCED_BIGNUM_TYPE_H
#define COALESCED_BIGNUM_TYPE_H

#include "test_constants.h"

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

// A bignum is written "vertically" instead of "horizontally". Each column
// represents one bignum. The data on one "line" of a coalesced_bignum is a mix
// of the i'th element of N different bignums.

// We can see that coalesced_bignum* is actually a transposed version of bignum*

typedef uint32_t coalesced_bignum[NUMBER_OF_TESTS];

#endif
