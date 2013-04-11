#ifndef OPERATION_CHECK_H
#define OPERATION_CHECK_H

#include <stdint.h>
#include <gmp.h>

void binary_operator_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b, void (*op)(mpz_t rop, const mpz_t op1, const mpz_t op2));
void addition_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
void subtraction_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);

#endif
