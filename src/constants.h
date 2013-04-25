#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>

#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 512

#define NUMBER_OF_BIGNUMS ((uint32_t) (THREADS_PER_BLOCK) * (BLOCKS_PER_GRID))

#endif
