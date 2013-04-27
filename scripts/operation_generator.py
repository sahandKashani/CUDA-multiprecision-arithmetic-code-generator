import math;
import re;

bit_range = 131;
threads_per_block = 64;
blocks_per_grid = 64;

bits_per_word = 32;
min_bignum_number_of_words = math.ceil(bit_range / bits_per_word);
max_bignum_number_of_words = math.ceil((2 * bit_range) / bits_per_word);
total_bit_length = max_bignum_number_of_words * bits_per_word;
random_bignum_generator_seed = 12345;
number_of_bignums = threads_per_block * blocks_per_grid;

def set_constants():
    # bignum_types.h ###########################################################
    with open('../src/bignum_types.h', 'r') as input_file:
        contents = [line for line in input_file];
        contents = [re.sub(r"#define BITS_PER_WORD \d+", r"#define BITS_PER_WORD " + str(bits_per_word), line) for line in contents];
        contents = [re.sub(r"#define BIT_RANGE \d+", r"#define BIT_RANGE " + str(bit_range), line) for line in contents];
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

def add_backslash_to_end_of_elements_in_list(list):
    for i in range(len(list)):
        list[i] = list[i] + "\\";

def add_loc():
    asm = [];

    # header ###################################################################
    asm.append("#define add_loc(c_loc, a_loc, b_loc) {");
    asm.append("asm(\"{\"");

    # assembly statements ######################################################
    c_idx = 0;
    a_idx = max_bignum_number_of_words;
    b_idx = 2 * a_idx;

    asm.append("\"add.cc.u32 %" + str(c_idx) + ", %" + str(a_idx) + ", %" + str(b_idx) + ";\"");
    c_idx += 1;
    a_idx += 1;
    b_idx += 1;

    for i in range(max_bignum_number_of_words - 2):
        asm.append("\"addc.cc.u32 %" + str(c_idx) + ", %" + str(a_idx) + ", %" + str(b_idx) + ";\"");
        c_idx += 1;
        a_idx += 1;
        b_idx += 1;

    asm.append("\"addc.u32 %" + str(c_idx) + ", %" + str(a_idx) + ", %" + str(b_idx) + ";\"");

    asm.append("\"}\"");

    # assembly operands ########################################################
    asm.append(":");
    for i in range(max_bignum_number_of_words - 1):
        asm.append("\"=r\"(c_loc[" + str(i) + "]),");
    asm.append("\"=r\"(c_loc[" + str(max_bignum_number_of_words - 1) + "])");

    asm.append(":");
    for i in range(max_bignum_number_of_words):
        asm.append("\"r\"(a_loc[" + str(i) + "]),");

    for i in range(max_bignum_number_of_words - 1):
        asm.append("\"r\"(b_loc[" + str(i) + "]),");
    asm.append("\"r\"(b_loc[" + str(max_bignum_number_of_words - 1) + "])");

    # close asm statement
    asm.append(");");

    add_backslash_to_end_of_elements_in_list(asm);

    # footer ###################################################################
    asm.append("}");

    return asm;

def add_glo():
    asm = add_loc();
    asm = [line.replace('add_loc(c_loc, a_loc, b_loc)', 'add_glo(c_glo, a_glo, b_glo, tid)') for line in asm];
    asm = [re.sub(r"_loc\[(\d+)\]", r"_glo[COAL_IDX(\1, tid)]", line) for line in asm];
    return asm;

def sub_loc():
    asm = add_loc();
    asm = [line.replace('add', 'sub') for line in asm];
    return asm;

def sub_glo():
    asm = add_glo();
    asm = [line.replace('add', 'sub') for line in asm];
    return asm;

# MAIN #########################################################################
set_constants();

# needed for COAL_IDX
print("#include \"bignum_types.h\"\n");

macros_to_print = [add_loc, add_glo, sub_loc, sub_glo];
for func in macros_to_print:
    print("\n".join(func()) + "\n");
