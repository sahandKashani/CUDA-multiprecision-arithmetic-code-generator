#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdint.h>

#define NUMBER_OF_BIGNUMS_IN_FILES 4194304
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 1024
#define NUMBER_OF_BIGNUMS 1048576
#define COALESCED_A_FILE_NAME "../data/coalesced_a.txt"
#define COALESCED_B_FILE_NAME "../data/coalesced_b.txt"
#define COALESCED_M_FILE_NAME "../data/coalesced_m.txt"
#define ADD_RESULTS_FILE_NAME "../data/add_results.txt"
#define SUB_RESULTS_FILE_NAME "../data/sub_results.txt"
#define MUL_RESULTS_FILE_NAME "../data/mul_results.txt"
#define MUL_KARATSUBA_RESULTS_FILE_NAME "../data/mul_karatsuba_results.txt"
#define ADD_M_RESULTS_FILE_NAME "../data/add_m_results.txt"
#define SUB_M_RESULTS_FILE_NAME "../data/sub_m_results.txt"
#define COALESCED_A_MON_FILE_NAME "../data/coalesced_a_mon.txt"
#define COALESCED_B_MON_FILE_NAME "../data/coalesced_b_mon.txt"
#define MONTGOMERY_REDUCTION_RESULTS_FILE_NAME "../data/montgomery_reduction_results.txt"
#define COALESCED_T_MON_FILE_NAME "../data/coalesced_T_mon.txt"
#define INVERSE_R_FILE_NAME "../data/inverse_R.txt"
#define M_PRIME_FILE_NAME "../data/m_prime.txt"

#endif
