#ifndef INPUT_OUTPUT_H
#define INPUT_OUTPUT_H

#include <stdint.h>

void read_coalesced_bignums_from_file(const char* file_name, uint32_t* bignums);
void write_coalesced_bignums_to_file(const char* file_name, uint32_t* bignums);

#endif