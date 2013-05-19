from input_output import *
from constants import *
import random

# initialize the random number generator
random.seed(seed)

def k_bit_rand_int(precision):
    assert precision > 0
    while True:
        number = random.getrandbits(precision)
        if len(bin(number)[2:]) == precision:
            assert number > 0
            return number

def k_bit_rand_int_less_than(upper_bound, precision):
    assert upper_bound > 0
    assert precision > 0
    while True:
        number = k_bit_rand_int(precision)
        if number != 0 and number < upper_bound:
            assert number > 0
            return number

# generate random numbers to files #############################################
m = []
a = []
b = []

print("Generating random numbers ... ", end = '')
for i in range(number_of_bignums):
    m.append(k_bit_rand_int(precision))
for i in m:
    a.append(k_bit_rand_int_less_than(i, precision))
    b.append(k_bit_rand_int_less_than(i, precision))
print("done")

write_numbers_to_file_coalesced(m, False, coalesced_m_file_name)
write_numbers_to_file_coalesced(a, False, coalesced_a_file_name)
write_numbers_to_file_coalesced(b, False, coalesced_b_file_name)
