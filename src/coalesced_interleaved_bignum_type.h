#ifndef COALESCED_INTERLEAVED_BIGNUM_TYPE_H
#define COALESCED_INTERLEAVED_BIGNUM_TYPE_H

#include "test_constants.h"

#define COALESCED_INTERLEAVED_BIGNUM_NUMBER_OF_WORDS (2 * NUMBER_OF_TESTS)

// assume you have 2 bignum ARRAYS "a" and "b", then the data in
// coalesced_bignum "c" would be:

// The "height" of the array should be BIGNUM_NUMBER_OF_WORDS

// assuming N = COALESCED_INTERLEAVED_BIGNUM_NUMBER_OF_WORDS
// a[0][0], b[0][0], a[1][0], b[1][0], ..., a[N-1][0], b[N-1][0]
// a[0][1], b[0][1], a[1][1], b[1][1], ..., a[N-1][1], b[N-1][1]
// a[0][2], b[0][2], a[1][2], b[1][2], ..., a[N-1][2], b[N-1][2]
// a[0][3], b[0][3], a[1][3], b[1][3], ..., a[N-1][3], b[N-1][3]
// a[0][4], b[0][4], a[1][4], b[1][4], ..., a[N-1][4], b[N-1][4]

// coalesced_bignum would represent one "line" of the array given above.

typedef uint32_t coalesced_interleaved_bignum[COALESCED_INTERLEAVED_BIGNUM_NUMBER_OF_WORDS];

#endif
