from constants import total_hex_length
from constants import hex_digits_per_word

from conversions import hex_str_to_int
from conversions import int_to_hex_str

def write_positive_numbers_to_file_coalesced(numbers, file_name):
    # no '0x'
    full_hex_numbers = [int_to_hex_str(n)[2:] for n in numbers]

    split_indexes = [i * hex_digits_per_word for i in range(total_hex_length // hex_digits_per_word)]

    hex_parts_normal = []
    for hex_n in full_hex_numbers:
        parts = [hex_n[i:(i + hex_digits_per_word)] for i in split_indexes]

        # "little" endian
        hex_parts_normal.append(reversed(parts))

    # "transpose" the matrix of numbers to have them in coalesced form
    hex_parts_coalesced = zip(*hex_parts_normal)

    with open(file_name, 'w') as f:
        for line in hex_parts_coalesced:
            for col in line:
                f.write(col + " ")
            f.write("\n")

def read_numbers_from_file_coalesced(file_name):
    with open(file_name, 'r') as f:
        # coalesced version (numbers on a column)
        hex_parts_coalesced = [line.split() for line in f]

    # transpose values to get non-coalesced version (number is on a line)
    hex_parts_normal = zip(*hex_parts_coalesced)

    int_values = []
    for hex_parts in hex_parts_normal:
        # remove "little endianness" by reversing the values, then convert to int
        int_values.append(hex_str_to_int('0x' + "".join(reversed(hex_parts))))

    return int_values
