from input_output import *
from conversions import *
from constants import *

def binary_operator_check(c_file_name, a_file_name, b_file_name, op_name, py_op_symbol, result_is_long_number):
    a_data = read_numbers_from_file_coalesced(a_file_name)
    b_data = read_numbers_from_file_coalesced(b_file_name)
    c_data = read_numbers_from_file_coalesced(c_file_name)

    print("Checking \"" + op_name + "\" results => ", end = '')

    error = False
    for (i, a, b, c) in zip(range(len(c_data)), a_data, b_data, c_data):
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

def modular_binary_operator_check(c_file_name, a_file_name, b_file_name, m_file_name, op_name, py_op_symbol):
    a_data = read_numbers_from_file_coalesced(a_file_name)
    b_data = read_numbers_from_file_coalesced(b_file_name)
    m_data = read_numbers_from_file_coalesced(m_file_name)
    c_data = read_numbers_from_file_coalesced(c_file_name)

    print("Checking \"" + op_name + "\" results => ", end = '')

    error = False
    for (i, a, b, m, c) in zip(range(len(c_data)), a_data, b_data, m_data, c_data):
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

def montgomery_reduction_check(c_file_name, T_mon_file_name, inverse_R_file_name, m_file_name):
    c_data     = read_numbers_from_file_coalesced(c_file_name)
    T_mon_data = read_numbers_from_file_coalesced(T_mon_file_name)
    inv_R_data = read_numbers_from_file_coalesced(inverse_R_file_name)
    m_data     = read_numbers_from_file_coalesced(m_file_name)

    print("Checking \"montgomery_reduction\" results => ", end = '')

    error = False
    for (i, T_mon, inverse_R, m, c_mon) in zip(range(len(c_data)), T_mon_data, inv_R_data, m_data, c_data):
        expected_mon = (T_mon * inverse_R) % m

        T_mon_hex_str        = int_to_hex_str(T_mon, True)
        inverse_R_hex_str    = int_to_hex_str(inverse_R, False)
        m_hex_str            = int_to_hex_str(m, False)
        c_mon_hex_str        = int_to_hex_str(c_mon, False)
        expected_mon_hex_str = int_to_hex_str(expected_mon, False)

        if c_mon_hex_str != expected_mon_hex_str:
            error = True
            print("error")
            print("\nmontgomery reduction error from thread " + str(i) + ":")
            print("T_mon        = " + T_mon_hex_str)
            print("inverse_R    = " + inverse_R_hex_str)
            print("m            = " + m_hex_str)
            print("c_mon        = " + c_mon_hex_str)
            print("expected_mon = " + expected_mon_hex_str)
            print()
            break

    if not error:
        print("ok")

print("Checking operation results:")

# binary_operator_check(add_results_file_name          , coalesced_a_file_name, coalesced_b_file_name, "add"          , "+", False)
# binary_operator_check(sub_results_file_name          , coalesced_a_file_name, coalesced_b_file_name, "sub"          , "-", False)
# binary_operator_check(mul_results_file_name          , coalesced_a_file_name, coalesced_b_file_name, "mul"          , "*", True)
# binary_operator_check(mul_karatsuba_results_file_name, coalesced_a_file_name, coalesced_b_file_name, "mul_karatsuba", "*", True)

# modular_binary_operator_check(add_m_results_file_name, coalesced_a_file_name, coalesced_b_file_name, coalesced_m_file_name, "add_m", "+")
# modular_binary_operator_check(sub_m_results_file_name, coalesced_a_file_name, coalesced_b_file_name, coalesced_m_file_name, "sub_m", "-")

montgomery_reduction_check(montgomery_reduction_results_file_name, coalesced_T_mon_file_name, inverse_R_file_name, coalesced_m_file_name)
