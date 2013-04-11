#ifndef IO_INTERFACE_H
#define IO_INTERFACE_H

#include <stdint.h>

void generate_random_bignum_arrays_to_files(const char* file_name_1, const char* file_name_2);
void generate_random_bignum_array_to_file(const char* file_name);
void read_bignum_arrays_from_files(uint32_t* a, uint32_t* b, const char* file_name_1, const char* file_name_2);
void read_bignum_array_from_file(const char* file_name, uint32_t* bignum, uint32_t amount_to_read);
void write_bignum_array_to_file(const char* file_name, uint32_t* bignum);

#endif
