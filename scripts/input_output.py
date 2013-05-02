from types_and_conversions import to_int
from types_and_conversions import to_hex_string
from types_and_conversions import to_binary_string

from constants import bits_per_word
from constants import total_bit_length
from constants import number_of_bignums
from constants import max_bignum_number_of_words

def real_int_value(element):
    binary = list(reversed(to_binary_string(element)[2:]))

    value = 0
    for i in range(len(binary)):
        power = int(binary[i], 2) * (2 ** i)
        if i != len(binary):
            value += power
        else:
            value -= power

    return value

def to_int_parts(number):
    number = to_binary_string(number)[2:].rjust(total_bit_length, '0')

    parts = []
    indexes = [bits_per_word * k for k in range(max_bignum_number_of_words)]

    for i in indexes:
        part = number[i:(i + bits_per_word)]
        parts.append(int(part, 2))

    parts.reverse()
    return parts

def from_number_parts(parts):
    padded = []
    for i in range(len(parts)):
        binary_part = to_binary_string('0x' + parts[i])[2:]
        pad_char = '0'
        if i == len(parts) - 1:
            pad_char = binary_part[0]
        padded.append(binary_part.rjust(bits_per_word, pad_char))

    return real_int_value('0b' + "".join(reversed(padded)))

def write_numbers_to_file_coalesced(numbers, file_name):
    for n in numbers:
        assert n > 0

    hex_numbers = [hex(n).rjust(total_bit_length / 4, '0') for n in numbers]
    split_indexes = [i * (bits_per_word / 4) for i in range()]
    hex_numbers_split = [hex_numbers[]

    numbers = [to_int_parts(number) for number in numbers]

    with open(file_name, 'w') as f:
        for j in range(max_bignum_number_of_words):
            for i in range(number_of_bignums):
                f.write(to_hex_string(numbers[i][j])[2:].rjust(8, '0') + " ")
            f.write('\n')

def read_numbers_from_file_coalesced(file_name):
    with open(file_name, 'r') as f:
        return list(map(from_number_parts, zip(*[line.split() for line in f])))
