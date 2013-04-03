#include "bignum_type.h"
#include "random_bignum_generator.h"
#include "bignum_conversions.h"

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

#define NUMBER_OF_TESTS ((unsigned long int) 1e4)

void generate_operands(bignum* host_a, bignum* host_b);
void check_results(bignum* host_c, bignum* host_a, bignum* host_b);
void execute_normal_addition_on_device();
__global__ void normal_addition(bignum* dev_c, bignum* dev_a, bignum* dev_b);

int main(void)
{
    printf("Testing PTX\n");

    // host operands
    bignum* host_a = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));
    bignum* host_b = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));
    // host results
    bignum* host_c = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));

    // generate random numbers for the tests
    generate_operands(host_a, host_b);

    check_results(host_c, host_a, host_b);

    free(host_a);
    free(host_b);
    free(host_c);
}

void execute_normal_addition_on_device()
{
    // device operands
    bignum* dev_a;
    bignum* dev_b;
    // device results
    bignum* dev_c;

    // allocate memory on device
    cudaMalloc((void**) &dev_a, NUMBER_OF_TESTS * sizeof(bignum));
    cudaMalloc((void**) &dev_b, NUMBER_OF_TESTS * sizeof(bignum));
    cudaMalloc((void**) &dev_c, NUMBER_OF_TESTS * sizeof(bignum));

    // copy operands to device memory
    cudaMemcpy(dev_a, host_a, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyHostToDevice);

    normal_addition<<<1, 1>>>(dev_c, dev_a, dev_b);

    // copy results back to host
    cudaMemcpy(host_c, dev_c, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

void generate_operands(bignum* host_a, bignum* host_b)
{
    start_random_number_generator();

    for (int i = 0; i < NUMBER_OF_TESTS; i++)
    {
        generate_random_bignum(host_a[i]);
        generate_random_bignum(host_b[i]);
    }

    stop_random_number_generator();
}

void check_results(bignum* host_c, bignum* host_a, bignum* host_b)
{
    bool results_correct = true;

    for (int i = 0; results_correct && i < NUMBER_OF_TESTS; i++)
    {
        char* bignum_a_str = bignum_to_string(host_a[i]);
        char* bignum_b_str = bignum_to_string(host_b[i]);
        char* bignum_c_str = bignum_to_string(host_c[i]);

        mpz_t gmp_bignum_a;
        mpz_t gmp_bignum_b;
        mpz_t gmp_bignum_c;

        mpz_init_set_str(gmp_bignum_a, bignum_a_str, 2);
        mpz_init_set_str(gmp_bignum_b, bignum_b_str, 2);
        mpz_init(gmp_bignum_c);

        mpz_add(gmp_bignum_c, gmp_bignum_a, gmp_bignum_b);

        // get binary string result
        char* gmp_bignum_c_str = mpz_get_str(NULL, 2, gmp_bignum_c);
        pad_string_with_zeros(&gmp_bignum_c_str);

        if (strcmp(gmp_bignum_c_str, bignum_c_str) != 0)
        {
            printf("incorrect calculation at iteration %d\n", i);
            results_correct = false;
            printf("own\n%s +\n%s =\n%s\n", bignum_a_str, bignum_b_str,
                   bignum_c_str);
            printf("gmp\n%s +\n%s =\n%s\n", bignum_a_str, bignum_b_str,
                   gmp_bignum_c_str);
        }

        free(bignum_a_str);
        free(bignum_b_str);
        free(bignum_c_str);
        free(gmp_bignum_c_str);

        mpz_clear(gmp_bignum_a);
        mpz_clear(gmp_bignum_b);
        mpz_clear(gmp_bignum_c);
    }

    if (results_correct)
    {
        printf("all correct\n");
    }
    else
    {
        printf("something wrong\n");
    }
}

__global__ void normal_addition(bignum* dev_c, bignum* dev_a, bignum* dev_b)
{
    for (int i = 0; i < NUMBER_OF_TESTS; i++)
    {
        asm("{"
            "    add.cc.u32  %0, %5, %10;"
            "    addc.cc.u32 %1, %6, %11;"
            "    addc.cc.u32 %2, %7, %12;"
            "    addc.cc.u32 %3, %8, %13;"
            "    addc.u32    %4, %9, %14;"
            "}"

            : "=r"(dev_c[i][0]), "=r"(dev_c[i][1]), "=r"(dev_c[i][2]),
              "=r"(dev_c[i][3]), "=r"(dev_c[i][4])

            : "r"(dev_a[i][0]), "r"(dev_a[i][1]), "r"(dev_a[i][2]),
              "r"(dev_a[i][3]), "r"(dev_a[i][4]),

              "r"(dev_b[i][0]), "r"(dev_b[i][1]), "r"(dev_b[i][2]),
              "r"(dev_b[i][3]), "r"(dev_b[i][4])
            );
    }
}
