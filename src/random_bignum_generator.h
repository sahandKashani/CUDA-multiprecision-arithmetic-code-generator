#ifndef RANDOM_BIGNUM_GENERATOR_H
#define RANDOM_BIGNUM_GENERATOR_H

#include <stdint.h>

#define SEED ((uint32_t) 12345)

void generate_generic_random_bignum(uint32_t* number, char* (*f)(void));
void generate_random_bignum_modulus(uint32_t* number);
void generate_random_bignum(uint32_t* number);
void start_random_number_generator();
void stop_random_number_generator();

char* generate_random_bignum_str();
char* generate_random_bignum_modulus_str();

#endif
