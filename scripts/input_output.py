from types_and_conversions import to_int
from types_and_conversions import to_hex_string
from types_and_conversions import to_binary_string

from constants import bits_per_word
from constants import total_bit_length
from constants import number_of_bignums
from constants import max_bignum_number_of_words

def to_int_parts(number):
    number = to_binary_string(number)[2:].zfill(total_bit_length)

    parts = []
    indexes = [bits_per_word * k for k in range(max_bignum_number_of_words)]

    for i in indexes:
        part = '0b' + number[i:(i + bits_per_word)]
        parts.append(to_int(part))

    parts.reverse()
    return parts

def write_numbers_to_file_coalesced(numbers, file_name):
    numbers = [to_int_parts(number) for number in numbers]

    with open(file_name, 'w') as f:
        for j in range(max_bignum_number_of_words):
            for i in range(number_of_bignums):
                f.write(to_hex_string(numbers[i][j])[2:] + " ")
            f.write('\n')
