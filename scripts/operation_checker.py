#!/usr/bin/env python3

from input_output import read_numbers_from_file_coalesced

from constants import total_bit_length
from constants import total_hex_length
from constants import coalesced_a_file_name
from constants import coalesced_b_file_name
from constants import add_results_file_name
from constants import sub_results_file_name
from constants import mul_results_file_name

def real_bin_str(integer):
    bit_string = ""
    if integer >= 0:
        return '0b' + bin(integer)[2:].rjust(total_bit_length, '0')
    else:
        # to get abs value, you have to remove '-0x' from binary version
        abs_value = bin(integer)[3:].rjust(total_bit_length, '0')

        complement = "".join(['1' if c == '0' else '0' for c in abs_value])
        twos_complement = bin(int(complement, 2) + 1)[2:].rjust(total_bit_length, '0')
        return '0b' + twos_complement

def real_hex_str(integer):
    # remove '0b'
    bin_str = real_bin_str(integer)[2:]

    indices = [i * 4 for i in range(total_hex_length)]
    hex_parts = [hex(int(bin_str[i:(i + 4)], 2))[2:] for i in indices]
    hex_str = "".join(hex_parts)

    return hex_str


sub_results = read_numbers_from_file_coalesced(sub_results_file_name)
for r in sub_results:
    print(real_bin_str(r))
    print(real_hex_str(r))
    print()

# # operands
# a = read_numbers_from_file_coalesced(coalesced_a_file_name)
# b = read_numbers_from_file_coalesced(coalesced_b_file_name)

# # check addition results
# print("checking add results => ", end='')
# add_results = read_numbers_from_file_coalesced(add_results_file_name)

# for (op1, op2, result) in zip(a, b, add_results):
#     expected_result = op1 + op2
#     if expected_result != result:
#         print("\nadd error:")
#         print("op1             = " + str(op1))
#         print("op2             = " + str(op2))
#         print("result          = " + str(result))
#         print("expected_result = " + str(expected_result))
#         break

# print("done")

# # check subtraction results
# print("checking sub results => ", end='')
# sub_results = read_numbers_from_file_coalesced(sub_results_file_name)

# for (op1, op2, result) in zip(a, b, sub_results):
#     expected_result = op1 - op2
#     if expected_result != result:
#         print("\nsub error:")
#         print("op1             = " + str(op1))
#         print("op2             = " + str(op2))
#         print("result          = " + str(result))
#         print("expected_result = " + str(expected_result))
#         break

# print("done")

# # check subtraction results
# print("checking mul results => ", end='')
# mul_results = read_numbers_from_file_coalesced(mul_results_file_name)

# for (op1, op2, result) in zip(a, b, mul_results):
#     expected_result = op1 * op2
#     if expected_result != result:
#         print("\nmul error:")
#         print("op1             = " + str(op1))
#         print("op2             = " + str(op2))
#         print("result          = " + str(result))
#         print("expected_result = " + str(expected_result))
#         break

# print("done")
