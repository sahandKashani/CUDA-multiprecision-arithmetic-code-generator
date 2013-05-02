import re

def is_binary_string(element):
    return type(element) is str and bool(re.match(r"^0b[01]+$", element))

def is_hex_string(element):
    return type(element) is str and bool(re.match(r"^0x[0-9a-fA-F]+$", element))

def is_int(element):
    return type(element) is int

def to_binary_string(element):
    if is_binary_string(element):
        return element
    elif is_hex_string(element):
        return bin(int(element, 16))
    elif is_int(element):
        return bin(element)
    else:
        assert False

def to_hex_string(element):
    return hex(int(to_binary_string(element), 2))

def to_int(element):
    return int(to_binary_string(element), 2)
