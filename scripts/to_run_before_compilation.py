from constants import precision
from constants import blocks_per_grid
from constants import threads_per_block
from constants import number_of_bignums
from constants import coalesced_m_file_name
from constants import coalesced_a_file_name
from constants import coalesced_b_file_name
from constants import min_bignum_number_of_words
from constants import max_bignum_number_of_words
from constants import add_results_file_name
from constants import sub_results_file_name
from constants import mul_results_file_name

from random_number_generator import k_bit_rand_int
from random_number_generator import k_bit_rand_int_less_than

from input_output import write_numbers_to_file_coalesced

from operation_generator import generate_operations

import re

# generate random numbers to files #############################################
m = []
a = []
b = []

print("Generating random numbers ... ", end = '')
for i in range(number_of_bignums):
    m.append(k_bit_rand_int(precision))
for i in m:
    a.append(k_bit_rand_int_less_than(i, precision))
    b.append(k_bit_rand_int_less_than(i, precision))
print("done")

write_numbers_to_file_coalesced(m, False, coalesced_m_file_name)
write_numbers_to_file_coalesced(a, False, coalesced_a_file_name)
write_numbers_to_file_coalesced(b, False, coalesced_b_file_name)

# set constants in C code ######################################################
with open('../src/bignum_types.h', 'r') as input_file:
    contents = [line for line in input_file]
    contents = [re.sub(r"#define MIN_BIGNUM_NUMBER_OF_WORDS \d+", r"#define MIN_BIGNUM_NUMBER_OF_WORDS " + str(min_bignum_number_of_words), line) for line in contents]
    contents = [re.sub(r"#define MAX_BIGNUM_NUMBER_OF_WORDS \d+", r"#define MAX_BIGNUM_NUMBER_OF_WORDS " + str(max_bignum_number_of_words), line) for line in contents]
with open('../src/bignum_types.h', 'w') as output_file:
    output_file.write("".join(contents))

with open('../src/constants.h', 'r') as input_file:
    contents = [line for line in input_file]
    contents = [re.sub(r"#define THREADS_PER_BLOCK \d+"       , r"#define THREADS_PER_BLOCK "      + str(threads_per_block)          , line) for line in contents]
    contents = [re.sub(r"#define BLOCKS_PER_GRID \d+"         , r"#define BLOCKS_PER_GRID "        + str(blocks_per_grid)            , line) for line in contents]
    contents = [re.sub(r"#define NUMBER_OF_BIGNUMS \d+"       , r"#define NUMBER_OF_BIGNUMS "      + str(number_of_bignums)          , line) for line in contents]
    contents = [re.sub(r'#define COALESCED_M_FILE_NAME "(.*)"', r'#define COALESCED_M_FILE_NAME "' + coalesced_m_file_name     + r'"', line) for line in contents]
    contents = [re.sub(r'#define COALESCED_A_FILE_NAME "(.*)"', r'#define COALESCED_A_FILE_NAME "' + coalesced_a_file_name     + r'"', line) for line in contents]
    contents = [re.sub(r'#define COALESCED_B_FILE_NAME "(.*)"', r'#define COALESCED_B_FILE_NAME "' + coalesced_b_file_name     + r'"', line) for line in contents]
    contents = [re.sub(r'#define ADD_RESULTS_FILE_NAME "(.*)"', r'#define ADD_RESULTS_FILE_NAME "' + add_results_file_name     + r'"', line) for line in contents]
    contents = [re.sub(r'#define SUB_RESULTS_FILE_NAME "(.*)"', r'#define SUB_RESULTS_FILE_NAME "' + sub_results_file_name     + r'"', line) for line in contents]
    contents = [re.sub(r'#define MUL_RESULTS_FILE_NAME "(.*)"', r'#define MUL_RESULTS_FILE_NAME "' + mul_results_file_name     + r'"', line) for line in contents]
with open('../src/constants.h', 'w') as output_file:
    output_file.write("".join(contents))

# generate macros for gpu operations ###########################################
generate_operations()
