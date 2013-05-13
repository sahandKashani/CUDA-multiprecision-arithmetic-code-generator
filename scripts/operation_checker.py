from input_output import read_numbers_from_file_coalesced

from constants import coalesced_a_file_name
from constants import coalesced_b_file_name
from constants import coalesced_m_file_name
from constants import add_results_file_name
from constants import sub_results_file_name
from constants import mul_results_file_name
from constants import add_m_results_file_name
from constants import sub_m_results_file_name
from constants import mul_karatsuba_results_file_name

from conversions import int_to_hex_str

def binary_operator_check(result_file_name, op1, op2, op_name, py_op_symbol, result_is_long_number):
    res = read_numbers_from_file_coalesced(result_file_name)
    print("Checking \"" + op_name + "\" results => ", end = '')

    error = False
    for (i, a, b, c) in zip(range(len(res)), op1, op2, res):
        expected = eval('a' + py_op_symbol + 'b')
        if c != expected:
            error = True
            print("error")
            print("\n" + op_name + " error from thread " + str(i) + ":")
            print("a        = " + int_to_hex_str(a, False))
            print("b        = " + int_to_hex_str(b, False))
            print("c        = " + int_to_hex_str(c, result_is_long_number))
            print("expected = " + int_to_hex_str(expected, result_is_long_number))
            break

    if not error:
        print("ok")

def modular_binary_operator_check(result_file_name, op1, op2, op3, op_name, py_op_symbol):
    res = read_numbers_from_file_coalesced(result_file_name)
    print("Checking \"" + op_name + "\" results => ", end = '')

    error = False
    for (i, a, b, m, c) in zip(range(len(res)), op1, op2, op3, res):
        expected = eval('(a' + py_op_symbol + 'b) % m')
        if c != expected:
            error = True
            print("error")
            print("\n" + op_name + " error from thread " + str(i) + ":")
            print("a        = " + int_to_hex_str(a, False))
            print("b        = " + int_to_hex_str(b, False))
            print("m        = " + int_to_hex_str(m, False))
            print("c        = " + int_to_hex_str(c, False))
            print("expected = " + int_to_hex_str(expected, False))
            break

    if not error:
        print("ok")

print("Checking operation results:")

# operands
a = read_numbers_from_file_coalesced(coalesced_a_file_name)
b = read_numbers_from_file_coalesced(coalesced_b_file_name)
m = read_numbers_from_file_coalesced(coalesced_m_file_name)

binary_operator_check(add_results_file_name, a, b, "add", "+", False)
binary_operator_check(sub_results_file_name, a, b, "sub", "-", False)
binary_operator_check(mul_results_file_name, a, b, "mul", "*", True)
binary_operator_check(mul_karatsuba_results_file_name, a, b, "mul_karatsuba", "*", True)
modular_binary_operator_check(add_m_results_file_name, a, b, m, "add_m", "+")
modular_binary_operator_check(sub_m_results_file_name, a, b, m, "sub_m", "-")
