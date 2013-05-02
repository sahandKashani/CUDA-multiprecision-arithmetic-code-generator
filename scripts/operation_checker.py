#!/usr/bin/env python3

from input_output import read_numbers_from_file_coalesced

from types_and_conversions import to_binary_string

from constants import precision
from constants import coalesced_a_file_name
from constants import coalesced_b_file_name
from constants import add_results_file_name
from constants import sub_results_file_name
from constants import mul_results_file_name

# operands
a = read_numbers_from_file_coalesced(coalesced_a_file_name)
b = read_numbers_from_file_coalesced(coalesced_b_file_name)

# check addition results
print("checking add results => ", end='')
add_results = read_numbers_from_file_coalesced(add_results_file_name)

for i in add_results:
    print(i)

# for (op1, op2, result) in zip(a, b, add_results):
#     expected_result = op1 + op2
#     if expected_result != result:
#         print("\nadd error:")
#         print(str(op1))
#         print(str(op2))
#         print(str(expected_result))
#         break
# print("done")

# # check subtraction results
# sub_results = read_numbers_from_file_coalesced(sub_results_file_name)
# for i in sub_results:
#     print(str(i))

# for (op1, op2, result) in zip(a, b, sub_results):
#     expected_result = op1 - op2
#     if expected_result != result:
#         print("\nsub error:")
#         print(str(op1))
#         print(str(op2))
#         print(str(result))
#         print(str(expected_result))
#         # print(str(expected_result % (1 << precision)) + "=")
#         break
# print("done")

# # check multiplication results
# mul_results = read_numbers_from_file_coalesced(mul_results_file_name)
# for (op1, op2, result) in zip(a, b, mul_results):
#     expected_result = op1 * op2
#     if expected_result != result:
#         print("\nmul error:")
#         print(to_binary_string(op1))
#         print(to_binary_string(op2))
#         print(to_binary_string(expected_result))
#         break
# print("done")
