from constants import seed

from types_and_conversions import to_binary_string
from types_and_conversions import to_int

import random

# initialize the random number generator
random.seed(seed)

def get_precision(element):
    return len(to_binary_string(element)[2:])

def k_bit_rand_int(precision):
    while True:
        number = to_int(random.getrandbits(precision))
        if get_precision(number) == precision:
            return number

def k_bit_rand_int_less_than(upper_bound, precision):
    upper_bound = to_int(upper_bound)
    number = 0

    while number == 0 or number >= upper_bound:
        number = k_bit_rand_int(precision)

    return number
