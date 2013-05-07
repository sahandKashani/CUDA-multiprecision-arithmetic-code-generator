from input_output import read_numbers_from_file_coalesced

from constants import coalesced_a_file_name
from constants import coalesced_b_file_name
from constants import add_results_file_name
from constants import sub_results_file_name
from constants import mul_results_file_name

from conversions import int_to_hex_str

def binary_operator_check(result_file_name, op1, op2, op_name, py_op_symbol):
    res = read_numbers_from_file_coalesced(result_file_name)
    print("Checking \"" + op_name + "\" results => ", end = '')

    error = False
    for (i, a, b, c) in zip(range(len(res)), op1, op2, res):
        expected = eval('a' + py_op_symbol + 'b')
        if c != expected:
            error = True
            print("error")
            print("\n" + op_name + " error from thread " + str(i) + ":")
            print("a        = " + int_to_hex_str(a))
            print("b        = " + int_to_hex_str(b))
            print("c        = " + int_to_hex_str(c))
            print("expected = " + int_to_hex_str(expected))
            break

    if not error:
        print("ok")

print("Checking operation results:")

# operands
a = read_numbers_from_file_coalesced(coalesced_a_file_name)
b = read_numbers_from_file_coalesced(coalesced_b_file_name)

binary_operator_check(add_results_file_name, a, b, "add", "+")
binary_operator_check(sub_results_file_name, a, b, "sub", "-")
binary_operator_check(mul_results_file_name, a, b, "mul", "*")
