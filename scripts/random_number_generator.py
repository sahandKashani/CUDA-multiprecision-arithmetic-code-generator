from input_output import *
from constants import *
import random
import gmpy2

# initialize the random number generator
random.seed(seed)

def k_bit_rand_int(precision):
    assert precision > 0
    while True:
        number = random.getrandbits(precision)
        if len(bin(number)[2:]) == precision and gmpy2.invert(R, number) != 0:
            assert number > 0
            return number

def k_bit_rand_int_less_than(upper_bound, precision):
    assert upper_bound > 0
    assert precision > 0
    while True:
        number = k_bit_rand_int(precision)
        if number != 0 and number < upper_bound and gmpy2.invert(R, number) != 0:
            assert number > 0
            return number

# generate random numbers to files #############################################
m = []
a = []
b = []
a_mon = []
b_mon = []
T_mon = []
inverse_R = []

print("Generating random numbers ... ", end = '')
for i in range(number_of_bignums):
    m.append(k_bit_rand_int(precision))
for i in m:
    a_normal = k_bit_rand_int_less_than(i, precision)
    b_normal = k_bit_rand_int_less_than(i, precision)
    a_montgo = (a_normal * R) % i
    b_montgo = (b_normal * R) % i
    T_montgo = (a_montgo * b_montgo)
    inv_R = gmpy2.invert(R, i)

    a.append(a_normal)
    b.append(b_normal)
    a_mon.append(a_montgo)
    b_mon.append(b_montgo)
    T_mon.append(T_montgo)
    inverse_R.append(inv_R)
print("done")

write_numbers_to_file_coalesced(m, False, coalesced_m_file_name)
write_numbers_to_file_coalesced(a, False, coalesced_a_file_name)
write_numbers_to_file_coalesced(b, False, coalesced_b_file_name)
write_numbers_to_file_coalesced(a_mon, False, coalesced_a_mon_file_name)
write_numbers_to_file_coalesced(b_mon, False, coalesced_b_mon_file_name)
write_numbers_to_file_coalesced(T_mon, True, coalesced_T_mon_file_name)
write_numbers_to_file_coalesced(inverse_R, False, inverse_R_file_name)
