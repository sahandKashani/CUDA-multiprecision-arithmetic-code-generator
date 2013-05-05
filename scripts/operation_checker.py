#!/usr/bin/env python3

from input_output import read_numbers_from_file_coalesced

from constants import coalesced_a_file_name
from constants import coalesced_b_file_name
from constants import add_results_file_name
from constants import sub_results_file_name
from constants import mul_results_file_name

# operands
a = read_numbers_from_file_coalesced(coalesced_a_file_name)
b = read_numbers_from_file_coalesced(coalesced_b_file_name)

# check addition results
add_results = read_numbers_from_file_coalesced(add_results_file_name)

print("checking add results => ", end='')
error = False
for (op1, op2, result) in zip(a, b, add_results):
    expected_result = op1 + op2
    if expected_result != result:
        error = True
        print("error")
        print("\nadd error:")
        print("op1             = " + str(op1))
        print("op2             = " + str(op2))
        print("result          = " + str(result))
        print("expected_result = " + str(expected_result))
        break

if not error:
    print("ok")

# check subtraction results
print("checking sub results => ", end='')
sub_results = read_numbers_from_file_coalesced(sub_results_file_name)

for (op1, op2, result) in zip(a, b, sub_results):
    expected_result = op1 - op2
    if expected_result != result:
        error = True
        print("error")
        print("\nsub error:")
        print("op1             = " + str(op1))
        print("op2             = " + str(op2))
        print("result          = " + str(result))
        print("expected_result = " + str(expected_result))
        break

if not error:
    print("ok")

# check multiplication results
print("checking mul results => ", end='')
mul_results = read_numbers_from_file_coalesced(mul_results_file_name)

for (op1, op2, result) in zip(a, b, mul_results):
    expected_result = op1 * op2
    if expected_result != result:
        error = True
        print("error")
        print("\nmul error:")
        print("op1             = " + str(op1))
        print("op2             = " + str(op2))
        print("result          = " + str(result))
        print("expected_result = " + str(expected_result))
        break

if not error:
    print("ok")
