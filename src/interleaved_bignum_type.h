#ifndef INTERLEAVED_BIGNUM_TYPE_H
#define INTERLEAVED_BIGNUM_TYPE_H

#include "bignum_type.h"

#define INTERLEAVED_BIGNUM_NUMBER_OF_WORDS (2 * BIGNUM_NUMBER_OF_WORDS)

// "little" endian derivative form like the following: assume you have 2 bignums
// ""a" and "b", then the data in interleaved_bignum "c" would be:
//
// a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3], a[4], b[4]
typedef unsigned int interleaved_bignum[INTERLEAVED_BIGNUM_NUMBER_OF_WORDS];

#endif
