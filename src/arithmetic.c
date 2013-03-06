#include <stdio.h>

/**
 * Structure to represent our 131-bit number. the letters form the number when
 * read alphabetically from left to right.
 *
 * Example: "abcde" is the number made by concatenating the string
 * representation of "a", "b", "c", "d" and "e". Therefore "e" contains the
 * least significant bits of the number, and "a" contains the most significant
 * bits.
 */
typedef struct
{
    unsigned int a;
    unsigned int b;
    unsigned int c;
    unsigned int d;
    unsigned int e;
} uint_131;

uint_131 add_uint_131(uint_131 operand_1, uint_131 operand_2)
{
    uint_131 result;

    result.e = operand_1.e + operand_2.e;
    result.d = operand_1.d + operand_2.d + (result.e < operand_1.e);
    result.c = operand_1.c + operand_2.c + (result.d < operand_1.d);
    result.b = operand_1.b + operand_2.b + (result.c < operand_1.c);
    result.a = operand_1.a + operand_2.a + (result.b < operand_1.b);

    return result;
}

unsigned int number_digits_length_base_10(unsigned int number)
{
    unsigned int i = 1;

    while(number >= 10)
    {
        number /= 10;
        i += 1;
    }

    return i;
}

void print_uint_131(uint_131 number, char* end_of_line)
{
    unsigned int padding_a = 10 - number_digits_length_base_10(number.a);
    unsigned int padding_b = 10 - number_digits_length_base_10(number.b);
    unsigned int padding_c = 10 - number_digits_length_base_10(number.c);
    unsigned int padding_d = 10 - number_digits_length_base_10(number.d);
    unsigned int padding_e = 10 - number_digits_length_base_10(number.e);

    printf("%0*u%0*u%0*u%0*u%0*u%s",
           padding_a, number.a,
           padding_b, number.b,
           padding_c, number.c,
           padding_d, number.d,
           padding_e, number.e,
           end_of_line);
}

uint_131 str_to_uint_131(char* str)
{
    uint_131 number;
}

int main(void)
{
    uint_131 bignum   = {0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    uint_131 smallnum = {0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};

    uint_131 result = add_uint_131(bignum, smallnum);

    print_uint_131(result, "\n");
}
