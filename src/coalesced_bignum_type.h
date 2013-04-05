#ifndef COALESCED_BIGNUM_TYPE_H
#define COALESCED_BIGNUM_TYPE_H

#include "test_constants.h"

#define COALESCED_BIGNUM_NUMBER_OF_WORDS NUMBER_OF_TESTS

// assume you have a bignum ARRAY "c", then the data in coalesced_bignum_result
// "c" would be:

// The "height" of the array should be BIGNUM_NUMBER_OF_WORDS

// assuming N = COALESCED_BIGNUM_NUMBER_OF_WORDS
// c[0][0], c[1][0], ..., c[N-1][0]
// c[0][1], c[1][1], ..., c[N-1][1]
// c[0][2], c[1][2], ..., c[N-1][2]
// c[0][3], c[1][3], ..., c[N-1][3]
// c[0][4], c[1][4], ..., c[N-1][4]

// coalesced_bignum would represent one "line" of the array given above.

typedef uint32_t coalesced_bignum[COALESCED_BIGNUM_NUMBER_OF_WORDS];

#endif
