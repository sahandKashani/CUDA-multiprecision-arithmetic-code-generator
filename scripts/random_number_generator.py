from constants import seed
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
