import math;

bit_width = 131;
bits_per_word = 32;
min_bignum_number_of_words = math.ceil(bit_width / bits_per_word);
max_bignum_number_of_words = math.ceil((2 * bit_width) / bits_per_word);

# print("bit width           =", bit_width);
# print("bits per word       =", bits_per_word);
# print("min words in bignum =", min_bignum_number_of_words);
# print("max words in bignum =", max_bignum_number_of_words);

def add_backslash_to_end_of_elements_in_list(list):
    for i in range(len(list)):
        list[i] = list[i] + "\\";

def add_loc():
    asm = [];


    # header ###################################################################
    asm.append("#define add_loc(c_loc, a_loc, b_loc) {");
    asm.append("asm(");

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

    print('\n'.join(asm));

add_loc();
