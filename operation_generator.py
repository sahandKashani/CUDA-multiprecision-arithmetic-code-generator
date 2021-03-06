#!/usr/bin/env python3

import math
import re

# change anything you want here
precision = 131
file_name_operations_h = r'operations.h'

# don't touch anything here
bits_per_word = 32
hex_digits_per_word = bits_per_word // 4
min_bignum_number_of_words = math.ceil(precision / bits_per_word)
max_bignum_number_of_words = math.ceil((2 * precision) / bits_per_word)
min_bit_length = min_bignum_number_of_words * bits_per_word
max_bit_length = max_bignum_number_of_words * bits_per_word
min_hex_length = min_bignum_number_of_words * hex_digits_per_word
max_hex_length = max_bignum_number_of_words * hex_digits_per_word

# The number of words needed to hold "precision" bits MUST be the same as the
# number of words needed to hold "precision + 1" bits. This is needed, because
# the addition of two n-bit numbers can give a (n + 1)-bit number, and our
# algorithms go by the principle that this (n + 1)-bit number is representable
# on the same number of bits as the n-bit number.
assert min_bignum_number_of_words == math.ceil((precision + 1) / bits_per_word)

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

def bignum_macro():
    doc = """////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// BIGNUM /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// A bignum is represented as the following 2 data structures depending on its
// size:
// uint32_t[MIN_BIGNUM_NUMBER_OF_WORDS]
// uint32_t[MAX_BIGNUM_NUMBER_OF_WORDS]

// In the code of this project, there will be no "bignum" type. It will only be
// referred to as a uint32_t*. This is needed, because having direct access to
// the inner representation of a bignum will be useful for efficient operations
// such as matrix transpositions, ...

// The code of this project will not have a bignum's size as a parameter to
// functions. This value is accessible throught the macros of this header file.

// A bignum is represented in "little endian" format: the most significant bits
// come in bignum[MAX_BIGNUM_NUMBER_OF_WORDS - 1] and the least significant bits
// come in bignum[0].

// A bignum's radix is 2^BITS_PER_WORD (words are 32 bits on our architecture).

// Assume you have an array of bignums "c", then the data would be conceptually
// represented as:

//  c[0][0]   c[0][1]  ...  c[0][H-1]
//  c[1][0]   c[1][1]  ...  c[1][H-1]
//     .         .     .        .
//     .         .      .       .
//     .         .       .      .
// c[N-1][0] c[N-1][1] ... c[N-1][H-1]

// with N = NUMBER_OF_BIGNUMS
//      H = MIN_BIGNUM_NUMBER_OF_WORDS or MAX_BIGNUM_NUMBER_OF_WORDS

// A bignum is written "horizontally". The data on one "line" of a bignum
// consists of the MIN_BIGNUM_NUMBER_OF_WORDS or MAX_BIGNUM_NUMBER_OF_WORDS
// elements of the bignum.

// For memory alignment issues, an array of bignums will not be represented as a
// 2D array like uint32_t[N][H], but rather as a flattened 1D array like
// uint32_t[N * H]. Index manipulation will be needed to access the array like a
// 2D array.

// Assuming the human readable 2D standard array of bignums representation
// above, the following macro returns the index of the "j"th element of the
// "i"th bignum from a 1D array of size N * H (N and H defined as below).

// 0 <= i < N = NUMBER_OF_BIGNUMS
// 0 <= j < H = MIN_BIGNUM_NUMBER_OF_WORDS or MAX_BIGNUM_NUMBER_OF_WORDS
#define IDX(i, j, is_long_number) (((i) * ((is_long_number) ? (MAX_BIGNUM_NUMBER_OF_WORDS) : (MIN_BIGNUM_NUMBER_OF_WORDS))) + (j))
"""
    doc_list = doc.split('\n')
    for i in range(len(doc_list)):
        doc_list[i] = doc_list[i].strip()
    return doc_list

def coalesced_bignum_macro():
    doc = """////////////////////////////////////////////////////////////////////////////////
////////////////////////////// COALESCED_BIGNUM ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// For efficient access to operands in gpu global memory, data needs to be
// accessed in a coalesced way. This is easily achieved by transposing an array
// of bignums to have the following representation:

// Assume you have an array of bignums "c", then the data in a coalesced array
// of bignums "c" would be:

//  c[0][0]   c[1][0]  ...  c[N-1][0]
//  c[0][1]   c[1][1]  ...  c[N-1][1]
//     .         .     .        .
//     .         .      .       .
//     .         .       .      .
// c[0][H-1] c[1][H-1] ... c[N-1][H-1]

// with N = NUMBER_OF_BIGNUMS
//      H = MIN_BIGNUM_NUMBER_OF_WORDS or MAX_BIGNUM_NUMBER_OF_WORDS

// A bignum is written "vertically" instead of "horizontally" with this
// representation. Each column represents one bignum. The data on one "line" of
// a coalesced bignum is a mix of the j'th element of N different bignums.

// As for normal bignums, a coalesced array of bignums will be represented as a
// flattened 1D array like uint32_t[N * H], and index manipulation would be
// neeeded to access the array like a 2D array.

// Assuming the human readable 2D coalesced bignum array representation above,
// the following macro returns the index of the "i"th element of the "j"th
// bignum from a 1D array of size N * H (N and H defined as below).

// 0 <= i < H = MIN_BIGNUM_NUMBER_OF_WORDS or MAX_BIGNUM_NUMBER_OF_WORDS
// 0 <= j < N = NUMBER_OF_BIGNUMS
#define COAL_IDX(i, j) (((i) * (NUMBER_OF_BIGNUMS)) + (j))"""
    doc_list = doc.split('\n')
    for i in range(len(doc_list)):
        doc_list[i] = doc_list[i].strip()
    return doc_list

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
        asm += mul_loc_generic(op_precision, op_precision, op1_name, op2_name, res_name, op1_shift, op2_shift, res_shift, indent)
    else:
        asm.append(" " * 4 * indent + '{\\')

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

        asm.append(" " * 4 * indent + 'uint32_t c0[' + str(c0_word_count) + '] = ' + str([0] * c0_word_count).replace('[', '{').replace(']', '}') + ';\\')
        asm.append(" " * 4 * indent + 'uint32_t c1[' + str(c1_word_count) + '] = ' + str([0] * c1_word_count).replace('[', '{').replace(']', '}') + ';\\')
        asm.append(" " * 4 * indent + 'uint32_t c2[' + str(c2_word_count) + '] = ' + str([0] * c2_word_count).replace('[', '{').replace(']', '}') + ';\\')

        asm.append(" " * 4 * indent + 'uint32_t a0[' + str(lo_word_count) + '] = ' + str(['a_loc!$' + str(i + op1_shift) + '$!' for i in range(lo_word_count)]).replace('[', '{').replace(']', '}').replace('!$', '[').replace('$!', ']').replace('\'', '') + ';\\')
        asm.append(" " * 4 * indent + 'uint32_t b0[' + str(lo_word_count) + '] = ' + str(['b_loc!$' + str(i + op2_shift) + '$!' for i in range(lo_word_count)]).replace('[', '{').replace(']', '}').replace('!$', '[').replace('$!', ']').replace('\'', '') + ';\\')

        asm.append(" " * 4 * indent + 'uint32_t a1[' + str(hi_word_count) + '] = ' + str(['a_loc!$' + str(i + op1_shift) + '$!' for i in range(lo_word_count, lo_word_count + hi_word_count)]).replace('[', '{').replace(']', '}').replace('!$', '[').replace('$!', ']').replace('\'', '') + ';\\')
        asm.append(" " * 4 * indent + 'uint32_t b1[' + str(hi_word_count) + '] = ' + str(['b_loc!$' + str(i + op2_shift) + '$!' for i in range(lo_word_count, lo_word_count + hi_word_count)]).replace('[', '{').replace(']', '}').replace('!$', '[').replace('$!', ']').replace('\'', '') + ';\\')

        asm.append(" " * 4 * indent + 'uint32_t a0_plus_a1[' + str(lo_plus_hi_word_count) + '] = ' + str([0] * lo_plus_hi_word_count).replace('[', '{').replace(']', '}') + ';\\')
        asm.append(" " * 4 * indent + 'uint32_t b0_plus_b1[' + str(lo_plus_hi_word_count) + '] = ' + str([0] * lo_plus_hi_word_count).replace('[', '{').replace(']', '}') + ';\\')

        # Low part multiplication (always the "full precision" multiplication of
        # the 2 parts).
        asm += mul_loc_generic(lo_precision, lo_precision, 'a0', 'b0', 'c0', 0, 0, 0, indent)

        # Hi part multiplication (possibly the "lesser precision" multiplication
        # of the 2 parts).
        asm += mul_loc_generic(hi_precision, hi_precision, 'a1', 'b1', 'c2', 0, 0, 0, indent)

        # c1 calculation
        # (a0 + a1) and (b0 + b1) has to be done with _exact_ function
        asm += add_loc_exact_generic(lo_precision, hi_precision, 'a0', 'a1', 'a0_plus_a1', 0, 0, 0, indent)
        asm += add_loc_exact_generic(lo_precision, hi_precision, 'b0', 'b1', 'b0_plus_b1', 0, 0, 0, indent)

        # (a0 + a1) * (b0 + b1)
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
        #                                                   | c0[3] | c0[2] | c0[1] | c0[0] | => c0
        # +                         | c1[4] | c1[3] | c1[2] | c1[1] | c1[0] |                 => c1
        # +         | c2[4] | c2[3] | c2[2] | c2[1] | c2[0] |                                 => c2
        # -----------------------------------------------------------------------------------
        #   | re[9] | re[8] | re[7] | re[6] | re[5] | re[4] | re[3] | re[2] | re[1] | re[0] | => result

        # Note: It is possible that re[9] will not be calculated if we are sure
        # that the result of the multiplication will never need that storage
        # location. For 131-bits, re[9] will not be calculated.

        # we always know that the first lo_word_count words of the result are
        # going to be unchanged by the addition, so we assign them directly from
        # the values of c0[0 .. lo_word_count]
        for i in range(lo_word_count):
            asm.append(" " * 4 * indent + 'asm("add.u32     %0, %1,  0;" : "=r"(c_loc[' + str(i + res_shift) + ']) : "r"(c0[' + str(i) + ']));\\')

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

        asm.append(" " * 4 * indent + '}\\')

    # replace all occurrences of a_loc, b_loc and c_loc by their appropriate
    # names, as provided by the user.
    for i in range(len(asm)):
        asm[i] = asm[i].replace('a_loc', op1_name).replace('b_loc', op2_name).replace('c_loc', res_name)

    return asm

def montgomery_reduction_generic(indent):
    asm = []

    u_i_m_precision = mul_res_precision(bits_per_word, precision)
    u_i_m_word_count = number_of_words_needed_for_precision(u_i_m_precision)

    T_precision = mul_res_precision(precision, precision)
    A_precision = add_res_precision(T_precision, u_i_m_precision + (min_bignum_number_of_words - 1) * bits_per_word)
    A_word_count = number_of_words_needed_for_precision(A_precision)

    # We need to declare u_i as an array of 1 element, because the _generic_
    # functions expect arrays as inputs.
    asm.append('uint32_t A[' + str(A_word_count) + '] = {' + str(['T_loc!$' + str(i) + '$!' for i in range(max_bignum_number_of_words)]).replace('[', '').replace(']', '').replace('!$', '[').replace('$!', ']').replace('\'', '') + ', ' + str([0] * (A_word_count - max_bignum_number_of_words)).replace('[', '').replace(']', '') + '};\\')
    asm.append('uint32_t u_i_m[' + str(u_i_m_word_count) + '] = ' + str([0] * u_i_m_word_count).replace('[', '{').replace(']', '}') + ';\\')
    asm.append('uint32_t u_i[1] = {0};\\')

    # asm.append("""for(uint32_t i = 0; i < 10; i++)\\
    #            {\\
    #                printf("%08x ", A[10 - i - 1]);\\
    #            }\\
    #            printf("\\n");\\""")

    # for (i = 0, i < min_bignum_number_of_words, i++)
    # {
    #     u_i = a_i * m_prime mod 2^32
    #     A = A + u_i * m * 2^(32 * i)
    # }
    for i in range(min_bignum_number_of_words):
        asm.append('asm("mul.lo.u32 %0, %1, %2;" : "=r"(u_i[0]) : "r"(A[' + str(i) + ']), "r"(m_prime));\\')
        # asm.append('printf("%08x\\n", u_i[0]);\\')
        asm += mul_loc_generic(bits_per_word, precision, 'u_i', 'm_loc', 'u_i_m', 0, 0, 0, 0)
        # asm.append("""for(uint32_t i = 0; i < 6; i++)\\
        #            {\\
        #                printf("%08x", u_i_m[6 - i - 1]);\\
        #            }\\""")
        asm += add_loc_generic(A_precision - i * bits_per_word, u_i_m_precision, 'A', 'u_i_m', 'A', i, 0, i, 0)
        # asm.append("""for(uint32_t i = 0; i < 10; i++)\\
        #            {\\
        #                printf("%08x ", A[10 - i - 1]);\\
        #            }\\
        #            printf("\\n");\\""")

    # A = A >> precision
    complete_words_with_zeros_count = min_bignum_number_of_words
    # if min_bit_length != precision:
    #     complete_words_with_zeros_count -= 1

    # extra_0_bit_count = precision - complete_words_with_zeros_count * bits_per_word
    # lo_mask = '0x' + hex((2**extra_0_bit_count) - 1)[2:].rjust(hex_digits_per_word, '0')
    # hi_mask = '0x' + hex((2**bits_per_word - 1) - int(lo_mask, 16))[2:].rjust(hex_digits_per_word, '0')

    # asm.append('uint32_t upper = 0;\\')

    # asm.append("""for(uint32_t i = 0; i < 10; i++)\\
    #                {\\
    #                    printf("%08x ", A[10 - i - 1]);\\
    #                }\\
    #                printf("\\n");\\""")

    # for i in range(complete_words_with_zeros_count, complete_words_with_zeros_count + min_bignum_number_of_words + 1):
    #     asm.append('asm("and.b32 %0, %0, ' + hi_mask + ';" : "+r"(A[' + str(i) + ']) : );\\')
    #     asm.append('asm("shr.b32 %0, %0, ' + str(extra_0_bit_count) + ';" : "+r"(A[' + str(i) + ']) : );\\')
    #     # the very last word does not have any word after it to have to get an
    #     # "upper" part
    #     if i != complete_words_with_zeros_count + min_bignum_number_of_words:
    #         asm.append('asm("and.b32 %0, %1, ' + lo_mask + ';" : "=r"(upper) : "r"(A[' + str(i + 1) + ']));\\')
    #         asm.append('asm("shl.b32 %0, %0, ' + str(bits_per_word - extra_0_bit_count) + ';" : "+r"(upper) : );\\')
    #         asm.append('asm("or.b32  %0, %0, %1;" : "+r"(A[' + str(i) + ']) : "r"(upper));\\')

    # asm.append("""for(uint32_t i = 0; i < 10; i++)\\
    #            {\\
    #                printf("%08x ", A[10 - i - 1]);\\
    #            }\\
    #            printf("\\n");\\""")

    # asm.append('}' + '\n')
    # return asm

    # if A >= m
    # {
    #     A = A - m
    # }
    asm.append('uint32_t mask[' + str(A_word_count) + '] = ' + str([0] * A_word_count).replace('[', '{').replace(']', '}') + ';\\')
    asm += sub_cc_loc_generic(A_precision, precision, 'A', 'm_loc', 'A', 0, 0, 0, 0)
    asm += subc_loc_generic(A_precision, A_precision, 'mask', 'mask', 'mask', 0, 0, 0, 0)
    asm += and_loc_generic(A_precision, precision, 'mask', 'm_loc', 'mask', 0, 0, 0, 0)
    asm += add_loc_generic(A_precision, A_precision, 'A', 'mask', 'A', 0, 0, 0, 0)

    # c_loc = A
    for i in range(min_bignum_number_of_words):
        asm.append('asm("add.u32 %0, %1,  0;" : "=r"(c_loc[' + str(i) + ']) : "r"(A[' + str(i + complete_words_with_zeros_count) + ']));\\')

    # replace all occurrences of a_loc, b_loc and c_loc by their appropriate
    # names, as provided by the user.
    for i in range(len(asm)):
        asm[i] = " " * 4 * indent + asm[i]

    return asm

def montgomery_reduction():
    indent = 1

    asm = []
    asm.append('#define montgomery_reduction(c_loc, T_loc, m_loc, m_prime)\\')
    asm.append('{\\')
    asm += montgomery_reduction_generic(indent)
    asm.append('}' + '\n')
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

def mul_karatsuba_glo():
    asm = mul_karatsuba_loc()
    asm = [line.replace(r'mul_karatsuba_loc(c_loc, a_loc, b_loc)', r'mul_karatsuba_glo(c_glo, a_glo, b_glo, tid)') for line in asm]
    asm = [re.sub(r'_loc\[(\d+)\]', r'_glo[COAL_IDX(\1, tid)]', line) for line in asm]
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

def add_m_glo():
    asm = add_m_loc()
    asm = [line.replace(r'add_m_loc(c_loc, a_loc, b_loc, m_loc)', r'add_m_glo(c_glo, a_glo, b_glo, m_glo, tid)') for line in asm]
    asm = [re.sub(r'_loc\[(\d+)\]', r'_glo[COAL_IDX(\1, tid)]', line) for line in asm]
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

def sub_m_glo():
    asm = sub_m_loc()
    asm = [line.replace(r'sub_m_loc(c_loc, a_loc, b_loc, m_loc)', r'sub_m_glo(c_glo, a_glo, b_glo, m_glo, tid)') for line in asm]
    asm = [re.sub(r'_loc\[(\d+)\]', r'_glo[COAL_IDX(\1, tid)]', line) for line in asm]
    return asm

################################################################################
################################# CODE GENERATOR ###############################
################################################################################

macros_functions_to_print = [bignum_macro,
                             coalesced_bignum_macro,

                             add_doc,
                             add_loc,
                             add_glo,

                             sub_doc,
                             sub_loc,
                             sub_glo,

                             mul_doc,
                             mul_loc,
                             mul_glo,

                             mul_karatsuba_loc,
                             mul_karatsuba_glo,

                             add_m_loc,
                             add_m_glo,

                             sub_m_loc,
                             sub_m_glo
                             ]

all_lines = []

# header guard
all_lines.append(r'#ifndef OPERATIONS_H')
all_lines.append(r'#define OPERATIONS_H' + '\n')

# info the application programmer will need
all_lines.append(r'#define MIN_BIGNUM_NUMBER_OF_WORDS (' + str(min_bignum_number_of_words) + ')')
all_lines.append(r'#define MAX_BIGNUM_NUMBER_OF_WORDS (' + str(max_bignum_number_of_words) + ')' + '\n')

for func in macros_functions_to_print:
    all_lines.extend(func())

all_lines.append(r'#endif' + '\n')

with open(file_name_operations_h, 'w') as operations_h:
    operations_h.write('\n'.join(all_lines))
