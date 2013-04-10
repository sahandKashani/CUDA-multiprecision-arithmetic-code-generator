#ifndef OPERATION_CHECK_H
#define OPERATION_CHECK_H

#include "bignum_types.h"
#include <gmp.h>

void binary_operator_check(bignum* host_c, bignum* host_a, bignum* host_b,
                           void (*op)(mpz_t rop, const mpz_t op1, const mpz_t op2));

void addition_check(bignum* host_c, bignum* host_a, bignum* host_b);
void subtraction_check(bignum* host_c, bignum* host_a, bignum* host_b);

#endif
