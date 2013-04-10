#ifndef OPERATIONS_CUH
#define OPERATIONS_CUH

#include "bignum_types.h"

__device__ void add(coalesced_bignum* c,
                    coalesced_bignum* a,
                    coalesced_bignum* b);

#endif
