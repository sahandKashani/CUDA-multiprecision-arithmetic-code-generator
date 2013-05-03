from constants import total_hex_length
from constants import total_bit_length
from constants import hex_digits_per_word

def real_int_value(binary):
    assert len(binary) == total_bit_length

    # real binary string is given, so have to flip it for (2 ** i) to work
    value = 0
    for i, b in enumerate(reversed(binary)):
        power = int(b, 2) * (2 ** i)
        if i != total_bit_length - 1:
            value += power
        else:
            value -= power

    return value

def write_positive_numbers_to_file_coalesced(numbers, file_name):
    for n in numbers:
        assert n > 0

    # no '0x'
    full_hex_numbers = [hex(n)[2:].rjust(total_hex_length, '0') for n in numbers]

    hex_parts_normal = []
    split_indexes = [i * hex_digits_per_word for i in range(total_hex_length // hex_digits_per_word)]
    for hex_n in full_hex_numbers:
        parts = [hex_n[i:(i + hex_digits_per_word)] for i in split_indexes]

        # "little" endian
        parts.reverse()
        hex_parts_normal.append(parts)

    # "transpose" the matrix of numbers to have them in coalesced form
    hex_parts_coalesced = zip(*hex_parts_normal)

    with open(file_name, 'w') as f:
        for line in hex_parts_coalesced:
            for col in line:
                f.write(col + " ")
            f.write("\n")

def read_numbers_from_file_coalesced(file_name):
    with open(file_name, 'r') as f:
        hex_parts_coalesced = [line.split() for line in f]

    hex_parts_normal = zip(*hex_parts_coalesced)

    int_values = []
    for hex_parts in hex_parts_normal:
        # no '0x'
        # remove "little endianness"
        full_hex_number = "".join(reversed(hex_parts))

        # no '0b'
        full_bin_number = bin(int(full_hex_number, 16))[2:].rjust(4 * total_hex_length, '0')

        int_value = real_int_value(full_bin_number)
        int_values.append(int_value)

    return int_values
