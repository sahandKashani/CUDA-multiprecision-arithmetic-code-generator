from input_output import *
from conversions import *
from constants import *

def binary_operator_check(result_file_name, op1, op2, op_name, py_op_symbol, result_is_long_number):
    res = read_numbers_from_file_coalesced(result_file_name)
    print("Checking \"" + op_name + "\" results => ", end = '')

    error = False
    for (i, a, b, c) in zip(range(len(res)), op1, op2, res):
        expected = eval('a' + py_op_symbol + 'b')

        a_hex_str        = int_to_hex_str(a, False)
        b_hex_str        = int_to_hex_str(b, False)
        c_hex_str        = int_to_hex_str(c, result_is_long_number)
        expected_hex_str = int_to_hex_str(expected, result_is_long_number)

        if c_hex_str != expected_hex_str:
            error = True
            print("error")
            print("\n" + op_name + " error from thread " + str(i) + ":")
            print("a        = " + a_hex_str)
            print("b        = " + b_hex_str)
            print("c        = " + c_hex_str)
            print("expected = " + expected_hex_str)
            print()
            break

    if not error:
        print("ok")

def modular_binary_operator_check(result_file_name, op1, op2, op3, op_name, py_op_symbol):
    res = read_numbers_from_file_coalesced(result_file_name)
    print("Checking \"" + op_name + "\" results => ", end = '')

    error = False
    for (i, a, b, m, c) in zip(range(len(res)), op1, op2, op3, res):
        expected = eval('(a' + py_op_symbol + 'b) % m')

        a_hex_str        = int_to_hex_str(a, False)
        b_hex_str        = int_to_hex_str(b, False)
        m_hex_str        = int_to_hex_str(m, False)
        c_hex_str        = int_to_hex_str(c, False)
        expected_hex_str = int_to_hex_str(expected, False)

        if c_hex_str != expected_hex_str:
            error = True
            print("error")
            print("\n" + op_name + " error from thread " + str(i) + ":")
            print("a        = " + a_hex_str)
            print("b        = " + b_hex_str)
            print("m        = " + m_hex_str)
            print("c        = " + c_hex_str)
            print("expected = " + expected_hex_str)
            print()
            break

    if not error:
        print("ok")

def montgomery_reduction_check(result_file_name, inv_R, op1, op2, op3):
    res = read_numbers_from_file_coalesced(result_file_name)
    print("Checking \"montgomery reduction\" results => ", end = '')

    error = False
    for (i, a_mon, b_mon, m, c_mon, inverse_R) in zip(range(len(res)), op1, op2, op3, res, inv_R):
        T = a_mon * b_mon
        expected_mon = (T * inverse_R) % m

        a_mon_hex_str        = int_to_hex_str(a_mon, False)
        b_mon_hex_str        = int_to_hex_str(b_mon, False)
        m_hex_str            = int_to_hex_str(m, False)
        c_mon_hex_str        = int_to_hex_str(c_mon, False)
        expected_mon_hex_str = int_to_hex_str(expected, False)

        if c_mon_hex_str != expected_mon_hex_str:
            error = True
            print("error")
            print("\n" + op_name + " error from thread " + str(i) + ":")
            print("a_mon        = " + a_mon_hex_str)
            print("b_mon        = " + b_mon_hex_str)
            print("m            = " + m_hex_str)
            print("c_mon        = " + c_mon_hex_str)
            print("expected_mon = " + expected_mon_hex_str)
            print()
            break

    if not error:
        print("ok")

print("Checking operation results:")

# operands
a = read_numbers_from_file_coalesced(coalesced_a_file_name)
b = read_numbers_from_file_coalesced(coalesced_b_file_name)
m = read_numbers_from_file_coalesced(coalesced_m_file_name)
a_mon = read_numbers_from_file_coalesced(coalesced_a_mon_file_name)
b_mon = read_numbers_from_file_coalesced(coalesced_b_mon_file_name)
inv_R = read_numbers_from_file_coalesced(inverse_R_file_name)

# binary_operator_check(add_results_file_name, a, b, "add", "+", False)
# binary_operator_check(sub_results_file_name, a, b, "sub", "-", False)
# binary_operator_check(mul_results_file_name, a, b, "mul", "*", True)
# binary_operator_check(mul_karatsuba_results_file_name, a, b, "mul_karatsuba", "*", True)
# modular_binary_operator_check(add_m_results_file_name, a, b, m, "add_m", "+")
# modular_binary_operator_check(sub_m_results_file_name, a, b, m, "sub_m", "-")
montgomery_reduction_check(montgomery_reduction_results_file_name, inv_R, a_mon, b_mon, m)
