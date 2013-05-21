import math
import re

# change anything you want here
precision = 131
threads_per_block = 16
blocks_per_grid = 16
coalesced_m_file_name = r'../data/coalesced_m.txt'
coalesced_a_file_name = r'../data/coalesced_a.txt'
coalesced_b_file_name = r'../data/coalesced_b.txt'
add_results_file_name = r'../data/add_results.txt'
sub_results_file_name = r'../data/sub_results.txt'
mul_results_file_name = r'../data/mul_results.txt'
mul_karatsuba_results_file_name = r'../data/mul_karatsuba_results.txt'
add_m_results_file_name = r'../data/add_m_results.txt'
sub_m_results_file_name = r'../data/sub_m_results.txt'

# don't touch anything here
seed = 12345
bits_per_word = 32
hex_digits_per_word = bits_per_word // 4
min_bignum_number_of_words = math.ceil(precision / bits_per_word)
max_bignum_number_of_words = math.ceil((2 * precision) / bits_per_word)
min_bit_length = min_bignum_number_of_words * bits_per_word
max_bit_length = max_bignum_number_of_words * bits_per_word
min_hex_length = min_bignum_number_of_words * hex_digits_per_word
max_hex_length = max_bignum_number_of_words * hex_digits_per_word
number_of_bignums = threads_per_block * blocks_per_grid
file_name_operations_h = r'../src/operations.h'

# The number of words needed to hold "precision" bits MUST be the same as the
# number of words needed to hold "precision + 1" bits. This is needed, because
# the addition of two n-bit numbers can give a (n + 1)-bit number, an our
# algorithms go by the principle that this (n + 1)-bit number is representable
# on the same number of bits as the n-bit number.
assert min_bignum_number_of_words == math.ceil((precision + 1) / bits_per_word)

# set constants in C code ######################################################
with open('../src/bignum_types.h', 'r') as input_file:
    contents = [line for line in input_file]
    contents = [re.sub(r"#define MIN_BIGNUM_NUMBER_OF_WORDS \d+", r"#define MIN_BIGNUM_NUMBER_OF_WORDS " + str(min_bignum_number_of_words), line) for line in contents]
    contents = [re.sub(r"#define MAX_BIGNUM_NUMBER_OF_WORDS \d+", r"#define MAX_BIGNUM_NUMBER_OF_WORDS " + str(max_bignum_number_of_words), line) for line in contents]
with open('../src/bignum_types.h', 'w') as output_file:
    output_file.write("".join(contents))

with open('../src/constants.h', 'r') as input_file:
    contents = [line for line in input_file]
    contents = [re.sub(r"#define THREADS_PER_BLOCK \d+"                  , r"#define THREADS_PER_BLOCK "                + str(threads_per_block)                , line) for line in contents]
    contents = [re.sub(r"#define BLOCKS_PER_GRID \d+"                    , r"#define BLOCKS_PER_GRID "                  + str(blocks_per_grid)                  , line) for line in contents]
    contents = [re.sub(r"#define NUMBER_OF_BIGNUMS \d+"                  , r"#define NUMBER_OF_BIGNUMS "                + str(number_of_bignums)                , line) for line in contents]
    contents = [re.sub(r'#define COALESCED_M_FILE_NAME "(.*)"'           , r'#define COALESCED_M_FILE_NAME "'           + coalesced_m_file_name           + r'"', line) for line in contents]
    contents = [re.sub(r'#define COALESCED_A_FILE_NAME "(.*)"'           , r'#define COALESCED_A_FILE_NAME "'           + coalesced_a_file_name           + r'"', line) for line in contents]
    contents = [re.sub(r'#define COALESCED_B_FILE_NAME "(.*)"'           , r'#define COALESCED_B_FILE_NAME "'           + coalesced_b_file_name           + r'"', line) for line in contents]
    contents = [re.sub(r'#define ADD_RESULTS_FILE_NAME "(.*)"'           , r'#define ADD_RESULTS_FILE_NAME "'           + add_results_file_name           + r'"', line) for line in contents]
    contents = [re.sub(r'#define SUB_RESULTS_FILE_NAME "(.*)"'           , r'#define SUB_RESULTS_FILE_NAME "'           + sub_results_file_name           + r'"', line) for line in contents]
    contents = [re.sub(r'#define MUL_RESULTS_FILE_NAME "(.*)"'           , r'#define MUL_RESULTS_FILE_NAME "'           + mul_results_file_name           + r'"', line) for line in contents]
    contents = [re.sub(r'#define MUL_KARATSUBA_RESULTS_FILE_NAME "(.*)"' , r'#define MUL_KARATSUBA_RESULTS_FILE_NAME "' + mul_karatsuba_results_file_name + r'"', line) for line in contents]
    contents = [re.sub(r'#define ADD_M_RESULTS_FILE_NAME "(.*)"'         , r'#define ADD_M_RESULTS_FILE_NAME "'         + add_m_results_file_name         + r'"', line) for line in contents]
    contents = [re.sub(r'#define SUB_M_RESULTS_FILE_NAME "(.*)"'         , r'#define SUB_M_RESULTS_FILE_NAME "'         + sub_m_results_file_name         + r'"', line) for line in contents]
with open('../src/constants.h', 'w') as output_file:
    output_file.write("".join(contents))
