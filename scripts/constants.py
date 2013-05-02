import math

# change anything you want here
precision = 131
threads_per_block = 4
blocks_per_grid = 4
benchmark_iterations = 256
coalesced_m_file_name = r'../data/coalesced_m.txt'
coalesced_a_file_name = r'../data/coalesced_a.txt'
coalesced_b_file_name = r'../data/coalesced_b.txt'
add_results_file_name = r'../data/add_results.txt'
sub_results_file_name = r'../data/sub_results.txt'
mul_results_file_name = r'../data/mul_results.txt'

# don't touch anything here
seed = 12345
bits_per_word = 32
min_bignum_number_of_words = math.ceil(precision / bits_per_word)
max_bignum_number_of_words = math.ceil((2 * precision) / bits_per_word)
total_bit_length = max_bignum_number_of_words * bits_per_word
number_of_bignums = threads_per_block * blocks_per_grid
file_name_operations_h = r'../src/operations.h'
