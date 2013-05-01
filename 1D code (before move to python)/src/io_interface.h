#ifndef IO_INTERFACE_H
#define IO_INTERFACE_H

#include <stdint.h>

void generate_modulus_and_operands_to_files(const char* host_m_file_name, const char* host_a_file_name, const char* host_b_file_name);
void read_bignum_array_from_file(const char* file_name, uint32_t* bignum, uint32_t amount_to_read);
void write_bignum_array_to_file(const char* file_name, uint32_t* bignum);

#endif
