from constants import *;
import random;

random.seed(random_bignum_generator_seed);

def get_precision(bin_str):
    return len(bin_str[2:]);

def k_bit_rand_bin_str(precision):
    while True:
        bin_str = bin(random.getrandbits(precision));
        if get_precision(bin_str) == precision:
            return bin_str;

def k_bit_rand_bin_str_less_than(precision, upper_bound_bin_str):
    upper_bound = int(upper_bound_bin_str, 2);
    number = 0;
    number_bin_str = 0x0;

    while number == 0 or number >= upper_bound:
        number_bin_str = k_bit_rand_bin_str(precision);
        number = int(number_bin_str, 2);

    return number_bin_str;

def pad_with_zeros(bignum_bin_str):
    return bignum_bin_str[2:].zfill(total_bit_length);

def split_bignum_to_parts(bignum_bin_str):
    bignum_bin_str = pad_with_zeros(bignum_bin_str);

    indices = [bits_per_word * i for i in range(max_bignum_number_of_words)];
    parts = [];
    for i in indices:
        parts.append(str(int(bignum_bin_str[i:(i + bits_per_word)], 2)));

    parts.reverse();
    return parts;

def generate_modulus_and_operands_to_files():
    with open(file_name_m, 'w') as file_m, open(file_name_a, 'w') as file_a, open(file_name_b, 'w') as file_b:
        for i in range(number_of_bignums):
            m_bin_str = k_bit_rand_bin_str(precision);
            a_bin_str = k_bit_rand_bin_str_less_than(precision, m_bin_str);
            b_bin_str = k_bit_rand_bin_str_less_than(precision, m_bin_str);

            m_split = split_bignum_to_parts(m_bin_str);
            a_split = split_bignum_to_parts(a_bin_str);
            b_split = split_bignum_to_parts(b_bin_str);

            for j in range(len(m_split)):
                file_m.write(m_split[j] + " ");
                file_a.write(a_split[j] + " ");
                file_b.write(b_split[j] + " ");

            file_m.write('\n');
            file_a.write('\n');
            file_b.write('\n');
