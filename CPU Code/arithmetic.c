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
    unsigned int e;
    unsigned int d;
    unsigned int c;
    unsigned int b;
    unsigned int a;
} int_131_u;

int_131_u add_int_131_u(int_131_u operand_a,
                        int_131_u operand_b)
{
    int_131_u result;
    return result;
}

int main(void)
{
    int si = 4294967295;
    unsigned int ui = 4294967295;

    printf("signed   integer: %d\n", si);
    printf("unsigned integer: %u\n", ui);
    printf("signed   integer: %x\n", si);
    printf("unsigned integer: %x\n", ui);
}

// typedef struct {

// unsigned long long int lo;

// unsigned long long int hi;

// } my_uint128;



// my_uint128 add_uint128 (my_uint128 a, my_uint128 b)

// {

// my_uint128 res;

// res.lo = a.lo + b.lo;

// res.hi = a.hi + b.hi + (res.lo < a.lo);

// return res;

// }
