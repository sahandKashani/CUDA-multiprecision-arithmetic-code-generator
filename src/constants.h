#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>

#define THREADS_PER_BLOCK 32
#define BLOCKS_PER_GRID 32
#define NUMBER_OF_BIGNUMS 1024
#define COALESCED_A_FILE_NAME "../data/coalesced_a.txt"
#define COALESCED_B_FILE_NAME "../data/coalesced_b.txt"
#define COALESCED_M_FILE_NAME "../data/coalesced_m.txt"
#define ADD_RESULTS_FILE_NAME "../data/add_results.txt"
#define SUB_RESULTS_FILE_NAME "../data/sub_results.txt"
#define MUL_RESULTS_FILE_NAME "../data/mul_results.txt"
#define ADD_M_RESULTS_FILE_NAME "../data/add_m_results.txt"
#define SUB_M_RESULTS_FILE_NAME "../data/sub_m_results.txt"

#endif
