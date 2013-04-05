#ifndef RANDOM_BIGNUM_GENERATOR_H
#define RANDOM_BIGNUM_GENERATOR_H

#include "bignum_type.h"

#define SEED ((uint32_t) 12345)
#define RANDOM_NUMBER_BIT_RANGE ((uint32_t) 131)

char* generate_random_bignum_str();
void generate_random_bignum(bignum number);
void start_random_number_generator();
void stop_random_number_generator();

#endif
