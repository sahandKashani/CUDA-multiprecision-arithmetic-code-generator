#ifndef BIGNUM_TYPE_H
#define BIGNUM_TYPE_H

#include <stdint.h>

#define BITS_PER_WORD 32
#define BIGNUM_NUMBER_OF_WORDS 5
#define TOTAL_BIT_LENGTH (BIGNUM_NUMBER_OF_WORDS * BITS_PER_WORD)

// little endian: most significant bits come in bignum[4] and least significant
// bits come in bignum[0]
typedef uint32_t bignum[BIGNUM_NUMBER_OF_WORDS];

#endif
