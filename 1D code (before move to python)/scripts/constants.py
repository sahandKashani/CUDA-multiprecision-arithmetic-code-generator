import math;
import re;

# change anything you want here
precision = 33;
threads_per_block = 1;
blocks_per_grid = 1;

file_name_m = r'../data/bignum_array_m.txt';
file_name_a = r'../data/bignum_array_a.txt';
file_name_b = r'../data/bignum_array_b.txt';

# don't touch anything here
bits_per_word = 32;
min_bignum_number_of_words = math.ceil(precision / bits_per_word);
max_bignum_number_of_words = math.ceil((2 * precision) / bits_per_word);
total_bit_length = max_bignum_number_of_words * bits_per_word;
random_bignum_generator_seed = 12345;
number_of_bignums = threads_per_block * blocks_per_grid;
file_name_operations_h = r'../src/operations.h';

assert(precision > 32);

def set_constants():
    # bignum_types.h ###########################################################
    with open('../src/bignum_types.h', 'r') as input_file:
        contents = [line for line in input_file];
        contents = [re.sub(r"#define BITS_PER_WORD \d+", r"#define BITS_PER_WORD " + str(bits_per_word), line) for line in contents];
        contents = [re.sub(r"#define BIT_RANGE \d+", r"#define BIT_RANGE " + str(precision), line) for line in contents];
        contents = [re.sub(r"#define MIN_BIGNUM_NUMBER_OF_WORDS \d+", r"#define MIN_BIGNUM_NUMBER_OF_WORDS " + str(min_bignum_number_of_words), line) for line in contents];
        contents = [re.sub(r"#define MAX_BIGNUM_NUMBER_OF_WORDS \d+", r"#define MAX_BIGNUM_NUMBER_OF_WORDS " + str(max_bignum_number_of_words), line) for line in contents];
        contents = [re.sub(r"#define TOTAL_BIT_LENGTH \d+", r"#define TOTAL_BIT_LENGTH " + str(total_bit_length), line) for line in contents];
    with open('../src/bignum_types.h', 'w') as output_file:
        output_file.write("".join(contents));

    # random_bignum_generator.h ################################################
    with open('../src/random_bignum_generator.h', 'r') as input_file:
        contents = [line for line in input_file];
        contents = [re.sub(r"#define SEED \d+", r"#define SEED " + str(random_bignum_generator_seed), line) for line in contents];
    with open('../src/random_bignum_generator.h', 'w') as output_file:
        output_file.write("".join(contents));

    # constants.h ##############################################################
    with open('../src/constants.h', 'r') as input_file:
        contents = [line for line in input_file];
        contents = [re.sub(r"#define THREADS_PER_BLOCK \d+", r"#define THREADS_PER_BLOCK " + str(threads_per_block), line) for line in contents];
        contents = [re.sub(r"#define BLOCKS_PER_GRID \d+", r"#define BLOCKS_PER_GRID " + str(blocks_per_grid), line) for line in contents];
        contents = [re.sub(r"#define NUMBER_OF_BIGNUMS \d+", r"#define NUMBER_OF_BIGNUMS " + str(number_of_bignums), line) for line in contents];
    with open('../src/constants.h', 'w') as output_file:
        output_file.write("".join(contents));

    # main.cu ##################################################################
    with open('../src/main.cu', 'r') as input_file:
        contents = [line for line in input_file];
        contents = [re.sub(r'host_a_file_name = "(.*)"', r'host_a_file_name = "' + file_name_a + r'"', line) for line in contents];
        contents = [re.sub(r'host_b_file_name = "(.*)"', r'host_b_file_name = "' + file_name_b + r'"', line) for line in contents];
        contents = [re.sub(r'host_m_file_name = "(.*)"', r'host_m_file_name = "' + file_name_m + r'"', line) for line in contents];
    with open('../src/main.cu', 'w') as output_file:
        output_file.write("".join(contents));
