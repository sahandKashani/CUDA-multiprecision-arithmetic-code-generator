from constants import *
import math
import re

# ATTENTION: all "_generic()" functions do NOT create macros. They just paste
# the assembly code that does the wanted operation for the specified operand
# precisions. All "_generic()" function also have curly brackets around them to
# avoid any scoping conflicts with the callers variables.

################################################################################
########################## GENERAL PURPOSE FUNCTIONS ###########################
################################################################################

def number_of_words_needed_for_precision(precision):
    return math.ceil(precision / bits_per_word)

def add_res_precision(op1_precision, op2_precision):
    res_precision = max(op1_precision, op2_precision) + 1
    return res_precision

def mul_res_precision(op1_precision, op2_precision):
    res_precision = op1_precision + op2_precision

    # res_precision = op1_precision + op2_precision does not hold if one of the
    # operands has precision 1. In that case, you need to reduce the precision
    # of the result by 1 bit.
    if (op1_precision == 1) or (op2_precision == 1):
        res_precision -= 1

    return res_precision

################################################################################
################################## DOCUMENTATION ###############################
################################################################################

def add_doc():
    doc = """
// Example of the schoolbook addition algorithm we will use if bignums were
// represented on 5 words:
//
//   A[4]---A[3]---A[2]---A[1]---A[0]
// + B[4]---B[3]---B[2]---B[1]---B[0]
// ------------------------------------
// | A[4] | A[3] | A[2] | A[1] | A[0] |
// |  +   |  +   |  +   |  +   |  +   |
// | B[4] | B[3] | B[2] | B[1] | B[0] |
// |  +   |  +   |  +   |  +   |      |
// | c_in | c_in | c_in | c_in |      |
// ------------------------------------
// | C[4] | C[3] | C[2] | C[1] | C[0] |"""
    doc_list = doc.split('\n')
    for i in range(len(doc_list)):
        doc_list[i] = doc_list[i].strip()
    return doc_list

def sub_doc():
    doc = """
// Example of the schoolbook subtraction algorithm we will use if bignums were
// represented on 5 words:
//
//   A[4]---A[3]---A[2]---A[1]---A[0]
// - B[4]---B[3]---B[2]---B[1]---B[0]
// ------------------------------------
// | A[4] | A[3] | A[2] | A[1] | A[0] |
// |  -   |  -   |  -   |  -   |  -   |
// | B[4] | B[3] | B[2] | B[1] | B[0] |
// |  -   |  -   |  -   |  -   |      |
// | b_in | b_in | b_in | b_in |      |
// ------------------------------------
// | C[4] | C[3] | C[2] | C[1] | C[0] |"""
    doc_list = doc.split('\n')
    for i in range(len(doc_list)):
        doc_list[i] = doc_list[i].strip()
    return doc_list

def mul_doc():
    doc = """
// Example of the schoolbook multiplication algorithm we will use if bignums
// were represented on 5 words:
//
//                                      A[4]---A[3]---A[2]---A[1]---A[0]
//                                    * B[4]---B[3]---B[2]---B[1]---B[0]
// -----------------------------------------------------------------------
// |      |      |      |      |      |      |      |      | B[0] * A[0] |
// |      |      |      |      |      |      |      | B[0] * A[1] |      |
// |      |      |      |      |      |      | B[0] * A[2] |      |      |
// |      |      |      |      |      | B[0] * A[3] |      |      |      |
// |      |      |      |      | B[0] * A[4] |      |      |      |      |
// |      |      |      |      |      |      |      | B[1] * A[0] |      |
// |      |      |      |      |      |      | B[1] * A[1] |      |      |
// |      |      |      |      |      | B[1] * A[2] |      |      |      |
// |      |      |      |      | B[1] * A[3] |      |      |      |      |
// |      |      |      | B[1] * A[4] |      |      |      |      |      |
// |      |      |      |      |      |      | B[2] * A[0] |      |      |
// |      |      |      |      |      | B[2] * A[1] |      |      |      |
// |      |      |      |      | B[2] * A[2] |      |      |      |      |
// |      |      |      | B[2] * A[3] |      |      |      |      |      |
// |      |      | B[2] * A[4] |      |      |      |      |      |      |
// |      |      |      |      |      | B[3] * A[0] |      |      |      |
// |      |      |      |      | B[3] * A[1] |      |      |      |      |
// |      |      |      | B[3] * A[2] |      |      |      |      |      |
// |      |      | B[3] * A[3] |      |      |      |      |      |      |
// |      | B[3] * A[4] |      |      |      |      |      |      |      |
// |      |      |      |      | B[4] * A[0] |      |      |      |      |
// |      |      |      | B[4] * A[1] |      |      |      |      |      |
// |      |      | B[4] * A[2] |      |      |      |      |      |      |
// |      | B[4] * A[3] |      |      |      |      |      |      |      |
// + B[4] * A[4] |      |      |      |      |      |      |      |      |
// -----------------------------------------------------------------------
// | C[9] | C[8] | C[7] | C[6] | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |
//
// Because of CUDA carry propagation problems (the carry flag is only kept for
// the next assembly instruction), we will have to order the steps in the
// following way:
//
//                                      A[4]---A[3]---A[2]---A[1]---A[0]
//                                    * B[4]---B[3]---B[2]---B[1]---B[0]
// -----------------------------------------------------------------------
// |      |      |      |      |      |      |      |      | B[0] * A[0] |
// |      |      |      |      |      |      |      | B[0] * A[1] |      |
// |      |      |      |      |      |      |      | B[1] * A[0] |      |
// |      |      |      |      |      |      | B[0] * A[2] |      |      |
// |      |      |      |      |      |      | B[1] * A[1] |      |      |
// |      |      |      |      |      |      | B[2] * A[0] |      |      |
// |      |      |      |      |      | B[0] * A[3] |      |      |      |
// |      |      |      |      |      | B[1] * A[2] |      |      |      |
// |      |      |      |      |      | B[2] * A[1] |      |      |      |
// |      |      |      |      |      | B[3] * A[0] |      |      |      |
// |      |      |      |      | B[0] * A[4] |      |      |      |      |
// |      |      |      |      | B[1] * A[3] |      |      |      |      |
// |      |      |      |      | B[2] * A[2] |      |      |      |      |
// |      |      |      |      | B[3] * A[1] |      |      |      |      |
// |      |      |      |      | B[4] * A[0] |      |      |      |      |
// |      |      |      | B[1] * A[4] |      |      |      |      |      |
// |      |      |      | B[2] * A[3] |      |      |      |      |      |
// |      |      |      | B[3] * A[2] |      |      |      |      |      |
// |      |      |      | B[4] * A[1] |      |      |      |      |      |
// |      |      | B[2] * A[4] |      |      |      |      |      |      |
// |      |      | B[3] * A[3] |      |      |      |      |      |      |
// |      |      | B[4] * A[2] |      |      |      |      |      |      |
// |      | B[3] * A[4] |      |      |      |      |      |      |      |
// |      | B[4] * A[3] |      |      |      |      |      |      |      |
// + B[4] * A[4] |      |      |      |      |      |      |      |      |
// -----------------------------------------------------------------------
// | C[9] | C[8] | C[7] | C[6] | C[5] | C[4] | C[3] | C[2] | C[1] | C[0] |
//
// Note: it is possible that C[9] will not be calculated if we are sure that the
// product of the 2 bignums will never require 2 * min_bignum_number_of_words
// words."""
    doc_list = doc.split('\n')
    for i in range(len(doc_list)):
        doc_list[i] = doc_list[i].strip()
    return doc_list

################################################################################
################################ GENERIC ADDITION ##############################
################################################################################

def add_loc_exact_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    res_precision = add_res_precision(op1_precision, op2_precision)
    op1_number_of_words = number_of_words_needed_for_precision(op1_precision)
    op2_number_of_words = number_of_words_needed_for_precision(op2_precision)
    res_number_of_words = number_of_words_needed_for_precision(res_precision)

    smaller_number_of_words = min(op1_number_of_words, op2_number_of_words)
    bigger_number_of_words = max(op1_number_of_words, op2_number_of_words)

    if bigger_number_of_words == op1_number_of_words:
        bigger_name = 'a_loc'
        bigger_shift = op1_shift
    else:
        bigger_name = 'b_loc'
        bigger_shift = op2_shift

    asm = []

    if res_number_of_words == 1:
        asm.append('asm("add.u32     %0, %1, %2;" : "=r"(c_loc[' + str(res_shift) + ']) : "r"(a_loc[' + str(op1_shift) + ']), "r"(b_loc[' + str(op2_shift) + ']));\\')

    elif res_number_of_words > 1:
        asm.append('asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[' + str(res_shift) + ']) : "r"(a_loc[' + str(op1_shift) + ']), "r"(b_loc[' + str(op2_shift) + ']));\\')

        if smaller_number_of_words == bigger_number_of_words == res_number_of_words:
            for i in range(1, res_number_of_words):
                if i < res_number_of_words - 1:
                    asm.append('asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(a_loc[' + str(i + op1_shift) + ']), "r"(b_loc[' + str(i + op2_shift) + ']));\\')
                elif i == res_number_of_words - 1:
                    asm.append('asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(a_loc[' + str(i + op1_shift) + ']), "r"(b_loc[' + str(i + op2_shift) + ']));\\')

        elif smaller_number_of_words < bigger_number_of_words == res_number_of_words:
            for i in range(1, res_number_of_words):
                if i < smaller_number_of_words:
                    asm.append('asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(a_loc[' + str(i + op1_shift) + ']), "r"(b_loc[' + str(i + op2_shift) + ']));\\')
                elif i < res_number_of_words - 1:
                    asm.append('asm("addc.cc.u32 %0, %1,  0;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(' + bigger_name + '[' + str(i + bigger_shift) + ']));\\')
                elif i == res_number_of_words - 1:
                    asm.append('asm("addc.u32    %0, %1,  0;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(' + bigger_name + '[' + str(i + bigger_shift) + ']));\\')

        # special case in like 32-bit + 32-bit = 33-bit
        elif smaller_number_of_words <= bigger_number_of_words < res_number_of_words:
            for i in range(1, res_number_of_words):
                if i < smaller_number_of_words:
                    asm.append('asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(a_loc[' + str(i + op1_shift) + ']), "r"(b_loc[' + str(i + op2_shift) + ']));\\')
                elif i < bigger_number_of_words:
                    asm.append('asm("addc.cc.u32 %0, %1,  0;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(' + bigger_name + '[' + str(i + bigger_shift) + ']));\\')

                # res_number_of_words can be at most 1 bigger than
                # bigger_number_of_words, so we can just check if we have
                # reached (res_number_of_words - 1) instead of having to check
                # for (i < res_number_of_words)
                elif i == res_number_of_words - 1:
                    asm.append('asm("addc.u32    %0,  0,  0;" : "=r"(c_loc[' + str(i + res_shift) + ']) : );\\')


    # replace all occurrences of a_loc, b_loc and c_loc by their appropriate
    # names, as provided by the user.
    for i in range(len(asm)):
        asm[i] = " " * 4 * indent + asm[i].replace('a_loc', op1_name).replace('b_loc', op2_name).replace('c_loc', res_name)

    return asm

def add_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    op1_number_of_words = number_of_words_needed_for_precision(op1_precision)
    op2_number_of_words = number_of_words_needed_for_precision(op2_precision)

    smaller_number_of_words = min(op1_number_of_words, op2_number_of_words)
    bigger_number_of_words = max(op1_number_of_words, op2_number_of_words)

    if bigger_number_of_words == op1_number_of_words:
        bigger_name = 'a_loc'
        bigger_shift = op1_shift
    else:
        bigger_name = 'b_loc'
        bigger_shift = op2_shift

    asm = []

    if bigger_number_of_words == 1:
        asm.append('asm("add.u32     %0, %1, %2;" : "=r"(c_loc[' + str(res_shift) + ']) : "r"(a_loc[' + str(op1_shift) + ']), "r"(b_loc[' + str(op2_shift) + ']));\\')

    elif bigger_number_of_words > 1:
        asm.append('asm("add.cc.u32  %0, %1, %2;" : "=r"(c_loc[' + str(res_shift) + ']) : "r"(a_loc[' + str(op1_shift) + ']), "r"(b_loc[' + str(op2_shift) + ']));\\')

        if smaller_number_of_words == bigger_number_of_words:
            for i in range(1, bigger_number_of_words):
                if i < bigger_number_of_words - 1:
                    asm.append('asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(a_loc[' + str(i + op1_shift) + ']), "r"(b_loc[' + str(i + op2_shift) + ']));\\')
                elif i == bigger_number_of_words - 1:
                    asm.append('asm("addc.u32    %0, %1, %2;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(a_loc[' + str(i + op1_shift) + ']), "r"(b_loc[' + str(i + op2_shift) + ']));\\')

        elif smaller_number_of_words < bigger_number_of_words:
            for i in range(1, bigger_number_of_words):
                if i < smaller_number_of_words:
                    asm.append('asm("addc.cc.u32 %0, %1, %2;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(a_loc[' + str(i + op1_shift) + ']), "r"(b_loc[' + str(i + op2_shift) + ']));\\')
                elif i < bigger_number_of_words - 1:
                    asm.append('asm("addc.cc.u32 %0, %1,  0;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(' + bigger_name + '[' + str(i + bigger_shift) + ']));\\')
                elif i == bigger_number_of_words - 1:
                    asm.append('asm("addc.u32    %0, %1,  0;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(' + bigger_name + '[' + str(i + bigger_shift) + ']));\\')


    # replace all occurrences of a_loc, b_loc and c_loc by their appropriate
    # names, as provided by the user.
    for i in range(len(asm)):
        asm[i] = " " * 4 * indent + asm[i].replace('a_loc', op1_name).replace('b_loc', op2_name).replace('c_loc', res_name)

    return asm

def addc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    asm = add_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)
    if number_of_words_needed_for_precision(max(op1_precision, op2_precision)) == 1:
        asm[0] = asm[0].replace("add.u32     ", "addc.u32    ")
    else:
        asm[0] = asm[0].replace("add.cc.u32  ", "addc.cc.u32 ")
    return asm

def add_cc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    asm = add_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)
    if number_of_words_needed_for_precision(max(op1_precision, op2_precision)) == 1:
        asm[0] = asm[0].replace("add.u32     ", "add.cc.u32  ")
    else:
        last_index = len(asm) - 1
        asm[last_index] = asm[last_index].replace("addc.u32    ", "addc.cc.u32 ")
    return asm

def addc_cc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    asm = add_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)
    if number_of_words_needed_for_precision(max(op1_precision, op2_precision)) == 1:
        asm[0] = asm[0].replace("add.u32     ", "addc.cc.u32 ")
    else:
        last_index = len(asm) - 1
        asm[0] = asm[0].replace("add.cc.u32  ", "addc.cc.u32 ")
        asm[last_index] = asm[last_index].replace("addc.u32    ", "addc.cc.u32 ")
    return asm

################################################################################
############################### GENERIC SUBTRACTION ############################
################################################################################

def sub_loc_exact_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    return [line.replace("add", "sub") for line in add_loc_exact_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)]

def sub_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    return [line.replace("add", "sub") for line in add_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)]

def subc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    return [line.replace("add", "sub") for line in addc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)]

def sub_cc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    return [line.replace("add", "sub") for line in add_cc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)]

def subc_cc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    return [line.replace("add", "sub") for line in addc_cc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)]

################################################################################
################################### GENERIC AND ################################
################################################################################

def and_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    return [line.replace("addc.cc.u32 ", "and.b32     ") for line in addc_cc_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)]

################################################################################
############################# GENERIC MULTIPLICATION ###########################
################################################################################

def mul_loc_generic(op1_precision, op2_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    res_precision = mul_res_precision(op1_precision, op2_precision)
    op1_number_of_words = number_of_words_needed_for_precision(op1_precision)
    op2_number_of_words = number_of_words_needed_for_precision(op2_precision)
    res_number_of_words = number_of_words_needed_for_precision(res_precision)

    asm = []

    if res_number_of_words == 1:
        asm.append('asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c_loc[' + str(res_shift) + ']) : "r"(b_loc[' + str(op2_shift) + ']), "r"(a_loc[' + str(op1_shift) + ']));\\')
    elif res_number_of_words == 2:
        asm.append('asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c_loc[' + str(res_shift) + ']) : "r"(b_loc[' + str(op2_shift) + ']), "r"(a_loc[' + str(op1_shift) + ']));\\')
        asm.append('asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c_loc[' + str(1 + res_shift) + ']) : "r"(b_loc[' + str(op2_shift) + ']), "r"(a_loc[' + str(op1_shift) + ']));\\')
    else:
        # sum until op1_number_of_words and op2_number_of_words, because we know
        # that a number is actually represented on those number of words, but
        # the result is on res_number_of_words.

        # generate tuples of indexes in arrays A and B that are to be multiplied
        # together. "i" represents "b_loc" and "j" represents "a_loc"
        mul_index_tuples = []

        # min shifting value is 0 and max is (res_number_of_words - 1)
        for shift_index in range(res_number_of_words):
            shift_index_tuples = []
            for j in range(op1_number_of_words):
                for i in range(op2_number_of_words):
                    if i + j == shift_index:
                        shift_index_tuples.append((i, j))
            if shift_index_tuples != []:
                mul_index_tuples.append(shift_index_tuples)

        # sort each tuple by its "b_loc" index, from smallest to biggest
        for i in range(len(mul_index_tuples)):
            mul_index_tuples[i] = list(sorted(mul_index_tuples[i], key = lambda tup: tup[0]))

        asm.append('{\\')
        asm.append('uint32_t carry = 0;\\')

        asm.append('asm("mul.lo.u32    %0, %1, %2    ;" : "=r"(c_loc[' + str(res_shift) + ']) : "r"(b_loc[' + str(op2_shift) + ']), "r"(a_loc[' + str(op1_shift) + ']));\\')

        for i in range(1, len(mul_index_tuples)):
            c_index = i

            # There is no carry to add to c_loc[1] in the very first iteration.
            # We don't need to set carry to 0 either if we are in this case.
            if i != 1:
                asm.append('asm("add.u32       %0, %1,  0    ;" : "=r"(c_loc[' + str(c_index + res_shift) + ']) : "r"(carry));\\')
                asm.append('asm("add.u32       %0,  0,  0    ;" : "=r"(carry));\\')

            # .hi bit operations
            for k in range(len(mul_index_tuples[i - 1])):
                b_index = mul_index_tuples[c_index - 1][k][0]
                a_index = mul_index_tuples[c_index - 1][k][1]

                # in the first iteration, we don't have any carry-out, or any
                # older value of c_loc[1] to add, so we just do a normal mul
                # instead of mad.
                if (c_index - 1) == 0:
                    asm.append('asm("mul.hi.u32    %0, %1, %2    ;" : "=r"(c_loc[' + str(c_index + res_shift) + ']) : "r"(b_loc[' + str(b_index + op2_shift) + ']), "r"(a_loc[' + str(a_index + op1_shift) + ']));\\')
                else:
                    # multiply add, with carry-out this time.
                    asm.append('asm("mad.hi.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[' + str(c_index + res_shift) + ']) : "r"(b_loc[' + str(b_index + op2_shift) + ']), "r"(a_loc[' + str(a_index + op1_shift) + ']));\\')
                    asm.append('asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\\')

            # .lo bit operations
            for j in range(len(mul_index_tuples[i])):
                b_index = mul_index_tuples[c_index][j][0]
                a_index = mul_index_tuples[c_index][j][1]

                asm.append('asm("mad.lo.cc.u32 %0, %1, %2, %0;" : "+r"(c_loc[' + str(c_index + res_shift) + ']) : "r"(b_loc[' + str(b_index + op2_shift) + ']), "r"(a_loc[' + str(a_index + op1_shift) + ']));\\')

                # in the second last shift iteration of the multiplication, if
                # we are at the last step, we no longer need to add the carry
                # unless if the result is indeed on (op1_number_of_words +
                # op2_number_of_words) words.
                if not ((i == len(mul_index_tuples) - 1) and (j == len(mul_index_tuples[i]) - 1)) or (res_number_of_words == (op1_number_of_words + op2_number_of_words)):
                    asm.append('asm("addc.u32      %0, %0,  0    ;" : "+r"(carry));\\')

        # if it is possible for the multiplication of 2 bignums to give a result
        # of size (op1_number_of_words + op2_number_of_words), then calculate
        # the final index of C
        if res_number_of_words == (op1_number_of_words + op2_number_of_words):
            asm.append('asm("mad.hi.u32    %0, %1, %2, %3;" : "=r"(c_loc[' + str(res_number_of_words - 1 + res_shift) + ']) : "r"(b_loc[' + str(op2_number_of_words - 1 + op2_shift) + ']), "r"(a_loc[' + str(op1_number_of_words - 1 + op1_shift) + ']), "r"(carry));\\')

        asm.append('}\\')

    # replace all occurrences of a_loc, b_loc and c_loc by their appropriate
    # names, as provided by the user.
    for i in range(len(asm)):
        asm[i] = " " * 4 * indent + asm[i].replace('a_loc', op1_name).replace('b_loc', op2_name).replace('c_loc', res_name)

    return asm

def mul_karatsuba_loc_generic(op_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent = 0):
    asm = []

    if op_precision <= bits_per_word + 1:
        asm += mul_loc_generic(op_precision, op_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, 0)
    else:
        asm.append('{\\')

        op_word_count = number_of_words_needed_for_precision(op_precision)
        lo_precision = bits_per_word * math.ceil(op_word_count / 2)
        hi_precision = op_precision - lo_precision
        if lo_precision - hi_precision > bits_per_word:
            lo_precision -= bits_per_word
            hi_precision += bits_per_word

        # The low part of the cut will always have full precision, and will
        # therefore NEED 2 times the precision for the multiplication result
        # storage
        lo_word_count = number_of_words_needed_for_precision(lo_precision)
        c0_precision = mul_res_precision(lo_precision, lo_precision)
        c0_word_count = number_of_words_needed_for_precision(c0_precision)

        # The hi part could have optimizations to save storage, since it might
        # most likely be shorter than the low part. We can do smaller
        # multiplications by taking this into account.
        hi_word_count = number_of_words_needed_for_precision(hi_precision)
        c2_precision = mul_res_precision(hi_precision, hi_precision)
        c2_word_count = number_of_words_needed_for_precision(c2_precision)

        # For c1, the situation is different from the other parts. We are going
        # to need to add 2 numbers which do not have the same precision, and we
        # need the complete result, so we are going to be using
        # add_loc_exact_generic() to do the calculation, instead of using
        # add_loc_generic(), because we want the full result, not just the
        # result that holds on the number of words of the bigger of the 2 words.
        lo_plus_hi_precision = add_res_precision(lo_precision, hi_precision)
        lo_plus_hi_word_count = number_of_words_needed_for_precision(lo_plus_hi_precision)
        c1_precision = mul_res_precision(lo_plus_hi_precision, lo_plus_hi_precision)
        c1_word_count = number_of_words_needed_for_precision(c1_precision)

        asm.append('uint32_t c0[' + str(c0_word_count) + '] = ' + str([0] * c0_word_count).replace('[', '{').replace(']', '}') + ';\\')
        asm.append('uint32_t c1[' + str(c1_word_count) + '] = ' + str([0] * c1_word_count).replace('[', '{').replace(']', '}') + ';\\')
        asm.append('uint32_t c2[' + str(c2_word_count) + '] = ' + str([0] * c2_word_count).replace('[', '{').replace(']', '}') + ';\\')

        asm.append('uint32_t a0_plus_a1[' + str(lo_plus_hi_word_count) + '] = ' + str([0] * lo_plus_hi_word_count).replace('[', '{').replace(']', '}') + ';\\')
        asm.append('uint32_t b0_plus_b1[' + str(lo_plus_hi_word_count) + '] = ' + str([0] * lo_plus_hi_word_count).replace('[', '{').replace(']', '}') + ';\\')

        # Low part multiplication (always the "bigger" multiplication of the 2
        # parts).
        # asm += mul_karatsuba_loc_generic(lo_precision, 'a_loc', 'b_loc', 'c0', 0 + op1_shift, 0 + op2_shift, 0, indent)
        asm += mul_loc_generic(lo_precision, lo_precision, 'a_loc', 'b_loc', 'c0', 0 + op1_shift, 0 + op2_shift, 0, indent)

        # Hi part multiplication (possibly the "smaller" multiplication of the 2
        # parts).
        # asm += mul_karatsuba_loc_generic(hi_precision, 'a_loc', 'b_loc', 'c2', lo_word_count + op1_shift, lo_word_count + op2_shift, 0, indent)
        asm += mul_loc_generic(hi_precision, hi_precision, 'a_loc', 'b_loc', 'c2', lo_word_count + op1_shift, lo_word_count + op2_shift, 0, indent)

        # c1 calculation
        # (a0 + a1) and (b0 + b1) has to be done with _exact_ function
        asm += add_loc_exact_generic(lo_precision, hi_precision, 'a_loc', 'a_loc', 'a0_plus_a1', 0 + op1_shift, lo_word_count + op1_shift, 0, indent)
        asm += add_loc_exact_generic(lo_precision, hi_precision, 'b_loc', 'b_loc', 'b0_plus_b1', 0 + op2_shift, lo_word_count + op2_shift, 0, indent)

        # (a0 + a1) * (b0 + b1)
        # asm += mul_karatsuba_loc_generic(lo_plus_hi_precision, 'a0_plus_a1', 'b0_plus_b1', 'c1', 0, 0, 0, indent)
        asm += mul_loc_generic(lo_plus_hi_precision, lo_plus_hi_precision, 'a0_plus_a1', 'b0_plus_b1', 'c1', 0, 0, 0, indent)

        # c1 = (a0 + a1) * (b0 + b1) - c0 - c2 = c1 - c0 - c2
        # Needs to be done with _exact_ function
        asm += sub_loc_exact_generic(c1_precision, c0_precision, 'c1', 'c0', 'c1', 0, 0, 0, indent)
        asm += sub_loc_exact_generic(c1_precision, c2_precision, 'c1', 'c2', 'c1', 0, 0, 0, indent)

        # final stage:
        # step = c0_word_count * bits_per_word
        # c_loc = c2 * 2^(2*step) + c1 * 2^(step) + c0

        # Example of overlap addition if precision = 131-bits
        #
        #                                   | c0[5] | c0[4] | c0[3] | c0[2] | c0[1] | c0[0] |   =>   c0
        # + | c1[6] | c1[5] | c1[4] | c1[3] | c1[2] | c1[1] | c1[0] |                           =>   c1
        # +         | c2[2] | c2[1] | c2[0] |                                                   =>   c2
        # -----------------------------------------------------------------------------------
        #   | re[9] | re[8] | re[7] | re[6] | re[5] | re[4] | re[3] | re[2] | re[1] | re[0] |   =>   result

        # Note: It is possible that re[9] will not be calculated if we are sure
        # that the result of the multiplication will never need that storage
        # location. For 131-bits, re[9] will not be calculated.

        # we always know that the first lo_word_count words of the result are
        # going to be unchanged by the addition, so we assign them directly from
        # the values of c0[0 .. lo_word_count]
        for i in range(lo_word_count):
            asm.append('asm("add.u32     %0, %1,  0;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(c0[' + str(i) + ']));\\')

        # now, we have to do the addition between c0[lo_word_count ..
        # c0_word_count] and c1[0 .. lo_word_count]

        # we know that c1 has at least 1 word more than c0, so we don't need to
        # deal with special cases where one is shorter than another. The
        # overlapped part between c0 and c1 will always be an addc.cc operation
        # on lo_word_count words.
        overlap_1_precision = bits_per_word * lo_word_count
        asm += add_cc_loc_generic(overlap_1_precision, overlap_1_precision, 'c0', 'c1', 'c_loc', lo_word_count, 0, lo_word_count + res_shift, indent)

        # finally, we have to do the addition between c1[lo_word_count ..
        # c1_word_count] and c2[0 .. c2_word_count] by taking the carry_in into
        # account, because of the first overlap addition that may have
        # overflowed.
        result_precision = mul_res_precision(op_precision, op_precision)
        overlap_2_precision = result_precision - c0_precision

        # we may not need to add the full c1 precision or c2 precision, because
        # we may know that c_loc would not need certain parts of c1 or c2. The
        # upper bound on c1's and c2's words are provided by result_precision,
        # which is the total number of words that the complete multiplication
        # algorithm is to yield. We can thus restrain c1's and c2's words to
        # that upper bound.
        c1_overlap_precision = c1_precision - lo_word_count * bits_per_word
        if c1_overlap_precision >= overlap_2_precision:
            c1_overlap_precision = overlap_2_precision

        c2_overlap_precision = c2_precision
        if c2_overlap_precision >= overlap_2_precision:
            c2_overlap_precision = overlap_2_precision

        asm += addc_loc_generic(c1_overlap_precision, c2_overlap_precision, 'c1', 'c2', 'c_loc', lo_word_count, 0, c0_word_count + res_shift, indent)

        asm.append('}\\')

    # replace all occurrences of a_loc, b_loc and c_loc by their appropriate
    # names, as provided by the user.
    for i in range(len(asm)):
        asm[i] = " " * 4 * indent + asm[i].replace('a_loc', op1_name).replace('b_loc', op2_name).replace('c_loc', res_name)

    return asm

################################################################################
################################ EXPORTED MACROS ###############################
################################################################################

# Note: for all the additions and subtractions below, we never need to call any
# of the "_exact_generic" functions, but only need the normal "_generic"
# functions. This is possible, because we have asserted in constants.py that
# min_bignum_number_of_words == math.ceil((precision + 1) / bits_per_word).

# This implies that the "_exact_generic" functions would return the exact same
# output as the normal "_generic" functions for these SPECIAL cases.

# addition #####################################################################
def add_loc():
    indent = 1

    asm = []
    asm.append('#define add_loc(c_loc, a_loc, b_loc)\\')
    asm.append('{\\')
    asm += add_loc_generic(precision, precision, 'a_loc', 'b_loc', 'c_loc', 0, 0, 0, indent)
    asm.append('}' + '\n')
    return asm

def add_glo():
    asm = add_loc()
    asm = [line.replace(r'add_loc(c_loc, a_loc, b_loc)', r'add_glo(c_glo, a_glo, b_glo, tid)') for line in asm]
    asm = [re.sub(r'_loc\[(\d+)\]', r'_glo[COAL_IDX(\1, tid)]', line) for line in asm]
    return asm

# subtraction ##################################################################
def sub_loc():
    indent = 1

    asm = []
    asm.append('#define sub_loc(c_loc, a_loc, b_loc)\\')
    asm.append('{\\')
    asm += sub_loc_generic(precision, precision, 'a_loc', 'b_loc', 'c_loc', 0, 0, 0, indent)
    asm.append('}' + '\n')
    return asm

def sub_glo():
    asm = sub_loc()
    asm = [line.replace(r'sub_loc(c_loc, a_loc, b_loc)', r'sub_glo(c_glo, a_glo, b_glo, tid)') for line in asm]
    asm = [re.sub(r'_loc\[(\d+)\]', r'_glo[COAL_IDX(\1, tid)]', line) for line in asm]
    return asm

# multiplication ###############################################################
def mul_loc():
    indent = 1

    asm = []
    asm.append('#define mul_loc(c_loc, a_loc, b_loc)\\')
    asm.append('{\\')
    asm += mul_loc_generic(precision, precision, 'a_loc', 'b_loc', 'c_loc', 0, 0, 0, indent)
    asm.append('}' + '\n')
    return asm

def mul_glo():
    asm = mul_loc()
    asm = [line.replace(r'mul_loc(c_loc, a_loc, b_loc)', r'mul_glo(c_glo, a_glo, b_glo, tid)') for line in asm]
    asm = [re.sub(r'_loc\[(\d+)\]', r'_glo[COAL_IDX(\1, tid)]', line) for line in asm]
    return asm

def mul_karatsuba_loc():
    indent = 1

    asm = []
    asm.append('#define mul_karatsuba_loc(c_loc, a_loc, b_loc)\\')
    asm.append('{\\')
    asm += mul_karatsuba_loc_generic(precision, 'a_loc', 'b_loc', 'c_loc', 0, 0, 0, indent)
    asm.append('}' + '\n')
    asm.append('')
    return asm

# modular addition #############################################################
def add_m_loc():
    # algorithm:
    # c    = a + b
    # c    = c - m
    # mask = 0 - borrow = mask - mask - borrow
    # mask = mask & m
    # c    = c + mask

    indent = 1

    asm = []
    asm.append('#define add_m_loc(c_loc, a_loc, b_loc, m_loc)\\')
    asm.append('{\\')
    asm.append(" " * 4 * indent + 'uint32_t mask[' + str(min_bignum_number_of_words) + '] = ' + str([0] * min_bignum_number_of_words).replace('[', '{').replace(']', '}') + ';\\')

    asm += add_loc_generic(precision, precision, 'a_loc', 'b_loc', 'c_loc', 0, 0, 0, indent)
    asm += sub_cc_loc_generic(precision, precision, 'c_loc', 'm_loc', 'c_loc', 0, 0, 0, indent)
    asm += subc_loc_generic(precision, precision, 'mask', 'mask', 'mask', 0, 0, 0, indent)
    asm += and_loc_generic(precision, precision, 'mask', 'm_loc', 'mask', 0, 0, 0, indent)
    asm += add_loc_generic(precision, precision, 'c_loc', 'mask', 'c_loc', 0, 0, 0, indent)

    asm.append('}' + '\n')
    return asm

# modular subtraction ##########################################################
def sub_m_loc():
    # algorithm:
    # c    = a - b
    # mask = 0 - borrow = mask - mask - borrow
    # mask = mask & m
    # c    = c + mask

    indent = 1

    asm = []
    asm.append('#define sub_m_loc(c_loc, a_loc, b_loc, m_loc)\\')
    asm.append('{\\')
    asm.append(" " * 4 * indent + 'uint32_t mask[' + str(min_bignum_number_of_words) + '] = ' + str([0] * min_bignum_number_of_words).replace('[', '{').replace(']', '}') + ';\\')

    asm += sub_cc_loc_generic(precision, precision, 'a_loc', 'b_loc', 'c_loc', 0, 0, 0, indent)
    asm += subc_loc_generic(precision, precision, 'mask', 'mask', 'mask', 0, 0, 0, indent)
    asm += and_loc_generic(precision, precision, 'mask', 'm_loc', 'mask', 0, 0, 0, indent)
    asm += add_loc_generic(precision, precision, 'c_loc', 'mask', 'c_loc', 0, 0, 0, indent)

    asm.append('}' + '\n')
    return asm

################################################################################
################################# CODE GENERATOR ###############################
################################################################################

macros_to_print = [add_doc,
                   add_loc,
                   add_glo,

                   sub_doc,
                   sub_loc,
                   sub_glo,

                   mul_doc,
                   mul_loc,
                   mul_karatsuba_loc,
                   mul_glo,

                   add_m_loc,
                   sub_m_loc]

all_lines = []

# needed for COAL_IDX
all_lines.append(r'#include "bignum_types.h"' + '\n')

for func in macros_to_print:
    all_lines.extend(func())

with open(file_name_operations_h, 'w') as operations_h:
    operations_h.write('\n'.join(all_lines))
