#ifndef OPERATIONS_CUH
#define OPERATIONS_CUH

#include <stdint.h>

__device__ void add(uint32_t* c, uint32_t* a, uint32_t* b);
__device__ void subtract(uint32_t* c, uint32_t* a, uint32_t* b);

#endif
