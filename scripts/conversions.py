from constants import min_bit_length
from constants import max_bit_length
from constants import min_hex_length
from constants import max_hex_length

def bin_str_to_hex_str(bin_str):
    # no '0b'
    assert (len(bin_str[2:]) == min_bit_length) or (len(bin_str[2:]) == max_bit_length)

    if len(bin_str[2:]) == min_bit_length:
        total_hex_length = min_hex_length
    else:
        total_hex_length = max_hex_length

    indices = [i * 4 for i in range(total_hex_length)]
    hex_parts = [hex(int(bin_str[2:][i:(i + 4)], 2))[2:] for i in indices]

    return '0x' + "".join(hex_parts)

def bin_str_to_int(bin_str):
    # no '0b'
    assert (len(bin_str[2:]) == min_bit_length) or (len(bin_str[2:]) == max_bit_length)

    if len(bin_str[2:]) == min_bit_length:
        total_bit_length = min_bit_length
    else:
        total_bit_length = max_bit_length

    # real binary string is given, so have to flip it for (2 ** i) to work
    value = 0
    for i, b in enumerate(reversed(bin_str[2:])):
        power = int(b, 2) * (2 ** i)
        if i != total_bit_length - 1:
            value += power
        else:
            value -= power

    return value

def int_to_bin_str(integer, is_long_number):
    if is_long_number:
        total_bit_length = max_bit_length
    else:
        total_bit_length = min_bit_length

    if integer >= 0:
        return '0b' + bin(integer)[2:].rjust(total_bit_length, '0')
    else:
        # to get abs value, you have to remove '-0b' from binary version
        abs_value = bin(integer)[3:].rjust(total_bit_length, '0')

        complement = "".join(['1' if c == '0' else '0' for c in abs_value])
        twos_complement = bin(int(complement, 2) + 1)[2:]
        return '0b' + twos_complement

def hex_str_to_bin_str(hex_str):
    # no '0x'
    assert (len(hex_str[2:]) == min_hex_length) or (len(hex_str[2:]) == max_hex_length)

    return '0b' + "".join([bin(int(h, 16))[2:].rjust(4, '0') for h in hex_str[2:]])

def hex_str_to_int(hex_str):
    return bin_str_to_int(hex_str_to_bin_str(hex_str))

def int_to_hex_str(integer, is_long_number):
    return bin_str_to_hex_str(int_to_bin_str(integer, is_long_number))
