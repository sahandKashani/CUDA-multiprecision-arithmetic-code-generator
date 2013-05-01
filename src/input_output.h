#ifndef INPUT_OUTPUT_H
#define INPUT_OUTPUT_H

#include <stdint.h>

void read_coalesced_bignum_array_from_file(const char* file_name, uint32_t* bignum);
void write_coalesced_bignum_array_to_file(const char* file_name, uint32_t* bignum);

#endif
