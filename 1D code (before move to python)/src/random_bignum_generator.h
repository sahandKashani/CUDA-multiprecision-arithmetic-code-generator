#ifndef RANDOM_BIGNUM_GENERATOR_H
#define RANDOM_BIGNUM_GENERATOR_H

#include <stdint.h>

#define SEED 12345

void start_random_number_generator();
void stop_random_number_generator();
void generate_exact_precision_bignum(uint32_t* number, uint32_t precision);
void generate_bignum_less_than_bignum(uint32_t* bigger, uint32_t* number);

#endif
