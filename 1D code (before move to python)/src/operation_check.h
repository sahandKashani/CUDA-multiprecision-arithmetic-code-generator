#ifndef OPERATION_CHECK_H
#define OPERATION_CHECK_H

#include <stdint.h>
#include <gmp.h>

void add_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
void sub_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);
void mul_check(uint32_t* host_c, uint32_t* host_a, uint32_t* host_b);

#endif
