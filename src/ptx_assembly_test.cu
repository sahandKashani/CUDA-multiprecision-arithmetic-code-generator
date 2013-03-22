#include <stdio.h>
#include <gmp.h>

// bignum array size in words
#define BIGNUM_SIZE 5
#define SEED ((unsigned int) 12345)
#define RANDOM_NUMBER_BIT_RANGE ((unsigned int) 131)

// most significant bits come in bignum[4] and least significant bits come in
// bignum[0]
typedef unsigned int bignum[BIGNUM_SIZE];

/////////////////////////
// Function Prototypes //
/////////////////////////
__global__ void test_kernel(int* dev_c, int a, int b);
void string_to_bignum(char* str, bignum number);
void print_bignum(bignum number);
char* generate_random_number();

//////////////
// Launcher //
//////////////
int main(void)
{
    printf("Testing inline PTX\n");

    // // 846668913323474690677881083138300645367 (length = 130-bit)
    // char op1[] = "00000000000000000000000000000010"
    //              "01111100111101101000000001010110"
    //              "00110000100001100001011011010000"
    //              "10111101100000101011011100011000"
    //              "11111100101100110000111111110111";

    // // 2029881613101810887805297702190481787852 (length = 131-bit)
    // char op2[] = "00000000000000000000000000000101"
    //              "11110111000111001111101001101100"
    //              "11011001000000111101101101110100"
    //              "00100100111000111111001100100000"
    //              "10001100010110011100101111001100";

    // // 2876550526425285578483178785328782433219 (length = 132-bit)
    // char rop[] = "00000000000000000000000000001000"
    //              "01110100000100110111101011000011"
    //              "00001001100010011111001001000100"
    //              "11100010011001101010101000111001"
    //              "10001001000011001101101111000011";

    // bignum a;
    // bignum b;
    // bignum c;
    // string_to_bignum(op1, a);
    // string_to_bignum(op2, b);
    // string_to_bignum(rop, c);

    // print_bignum(a);

    generate_random_number();

    // cudaMalloc((void**) &dev_c, sizeof(int));
    // test_kernel<<<1, 1>>>(dev_c, a, b);
    // cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    // printf("%d + %d = %d\n", a, b, c);
    // cudaFree(dev_c);
}

void print_bignum(bignum number)
{
    for (int i = 0; i < BIGNUM_SIZE; i++)
    {
        printf("%u", number[i]);
    }

    printf("\n");
}

void string_to_bignum(char* str, bignum number)
{
    unsigned int length = strlen(str);

    // clear the number
    for (int i = 0; i < BIGNUM_SIZE; i++)
    {
        number[i] = 0;
    }

    // fill the number
    for (int i = 0; i < length; i++)
    {
        // "bit" to be added to number
        unsigned int to_add = (unsigned int) str[length - 1 - i] << (i % 32);

        if (i < 32)
        {
            number[4] += to_add;
        }

        if (i < 64)
        {
            number[3] += to_add;
        }

        if (i < 96)
        {
            number[2] += to_add;
        }

        if (i < 128)
        {
            number[1] += to_add;
        }

        if (i < 160)
        {
            number[0] += to_add;
        }
    }
}

char* generate_random_number()
{
    // random number generator initialization
    gmp_randstate_t random_state;
    gmp_randinit_default(random_state);
    // incorporated seed in generator
    gmp_randseed_ui(random_state, SEED);

    // initialize test vector operands and result
    mpz_t number;
    mpz_init(number);

    mpz_urandomb(number, random_state, RANDOM_NUMBER_BIT_RANGE);

    gmp_print("random number = %Zd\n", number);

    // get memory back from operands and results
    mpz_clear(number);

    // get memory back from gmp_randstate_t
    gmp_randclear(random_state);

    return NULL;
}

__global__ void test_kernel(int* dev_c, int a, int b)
{
    // *dev_c = a + b;

    int c;

    asm("{"
        "    add.u32 %0, %1, %2;"
        "}"
        : "=r"(c) : "r"(a), "r"(b)
        );

    *dev_c = c;
}
