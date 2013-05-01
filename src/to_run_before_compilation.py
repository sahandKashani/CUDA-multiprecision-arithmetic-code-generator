#!/usr/bin/env python3

from constants import precision
from constants import file_name_m
from constants import file_name_a
from constants import file_name_b
from constants import bits_per_word
from constants import blocks_per_grid
from constants import total_bit_length
from constants import threads_per_block
from constants import number_of_bignums
from constants import min_bignum_number_of_words
from constants import max_bignum_number_of_words

from random_number_generator import k_bit_rand_int
from random_number_generator import k_bit_rand_int_less_than

from input_output import write_numbers_to_file_coalesced

from operation_generator import generate_operations

import re

# generate random numbers to files #############################################
m = []
a = []
b = []

for i in range(number_of_bignums):
    m.append(k_bit_rand_int(precision))
    a.append(k_bit_rand_int_less_than(m[i], precision))
    b.append(k_bit_rand_int_less_than(m[i], precision))

write_numbers_to_file_coalesced(m, file_name_m)
write_numbers_to_file_coalesced(a, file_name_a)
write_numbers_to_file_coalesced(b, file_name_b)

# set constants in c code ######################################################
with open('../src/bignum_types.h', 'r') as input_file:
    contents = [line for line in input_file]
    contents = [re.sub(r"#define BITS_PER_WORD \d+", r"#define BITS_PER_WORD " + str(bits_per_word), line) for line in contents]
    contents = [re.sub(r"#define BIT_RANGE \d+", r"#define BIT_RANGE " + str(precision), line) for line in contents]
    contents = [re.sub(r"#define MIN_BIGNUM_NUMBER_OF_WORDS \d+", r"#define MIN_BIGNUM_NUMBER_OF_WORDS " + str(min_bignum_number_of_words), line) for line in contents]
    contents = [re.sub(r"#define MAX_BIGNUM_NUMBER_OF_WORDS \d+", r"#define MAX_BIGNUM_NUMBER_OF_WORDS " + str(max_bignum_number_of_words), line) for line in contents]
    contents = [re.sub(r"#define TOTAL_BIT_LENGTH \d+", r"#define TOTAL_BIT_LENGTH " + str(total_bit_length), line) for line in contents]
with open('../src/bignum_types.h', 'w') as output_file:
    output_file.write("".join(contents))

with open('../src/constants.h', 'r') as input_file:
    contents = [line for line in input_file]
    contents = [re.sub(r"#define THREADS_PER_BLOCK \d+", r"#define THREADS_PER_BLOCK " + str(threads_per_block), line) for line in contents]
    contents = [re.sub(r"#define BLOCKS_PER_GRID \d+", r"#define BLOCKS_PER_GRID " + str(blocks_per_grid), line) for line in contents]
    contents = [re.sub(r"#define NUMBER_OF_BIGNUMS \d+", r"#define NUMBER_OF_BIGNUMS " + str(number_of_bignums), line) for line in contents]
with open('../src/constants.h', 'w') as output_file:
    output_file.write("".join(contents))

with open('../src/main.cu', 'r') as input_file:
    contents = [line for line in input_file]
    contents = [re.sub(r'host_a_file_name = "(.*)"', r'host_a_file_name = "' + file_name_a + r'"', line) for line in contents]
    contents = [re.sub(r'host_b_file_name = "(.*)"', r'host_b_file_name = "' + file_name_b + r'"', line) for line in contents]
    contents = [re.sub(r'host_m_file_name = "(.*)"', r'host_m_file_name = "' + file_name_m + r'"', line) for line in contents]
with open('../src/main.cu', 'w') as output_file:
    output_file.write("".join(contents))

# generate operations ##########################################################
generate_operations()
