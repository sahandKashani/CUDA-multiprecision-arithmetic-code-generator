import math;
import re;

bit_range = 131;
bits_per_word = 32;
min_bignum_number_of_words = math.ceil(bit_range / bits_per_word);
max_bignum_number_of_words = math.ceil((2 * bit_range) / bits_per_word);

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

# needed for COAL_IDX
print("#include \"bignum_types.h\"\n");

macros_to_print = [add_loc, add_glo, sub_loc, sub_glo];
for func in macros_to_print:
    print("\n".join(func()) + "\n");
