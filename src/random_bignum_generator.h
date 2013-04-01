#ifndef RANDOM_BIGNUM_GENERATOR_H
#define RANDOM_BIGNUM_GENERATOR_H

#include "bignum_type.h"

#define SEED ((unsigned int) 12345)
#define RANDOM_NUMBER_BIT_RANGE ((unsigned int) 131)
#define BASE 2

char* generate_random_bignum_str(unsigned int index, unsigned int seed,
                                 unsigned int bits, unsigned int base);

void generate_random_bignum(unsigned int index, unsigned int seed,
                            unsigned int bits, unsigned int base,
                            bignum number);

#endif
