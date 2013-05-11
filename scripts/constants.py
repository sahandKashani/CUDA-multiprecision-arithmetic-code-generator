import math

# change anything you want here
precision = 131
threads_per_block = 1
blocks_per_grid = 1
coalesced_m_file_name = r'../data/coalesced_m.txt'
coalesced_a_file_name = r'../data/coalesced_a.txt'
coalesced_b_file_name = r'../data/coalesced_b.txt'
add_results_file_name = r'../data/add_results.txt'
sub_results_file_name = r'../data/sub_results.txt'
mul_results_file_name = r'../data/mul_results.txt'

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
