#include "bignum_type.h"
#include "random_bignum_generator.h"
#include "bignum_conversions.h"

#include <stdio.h>
#include <gmp.h>

#define NUMBER_OF_TESTS ((unsigned long int) 1e3)

__global__ void test_kernel(bignum* dev_c, bignum* dev_a, bignum* dev_b);

int main(void)
{
    printf("Testing inline PTX\n");

    // operands
    bignum host_a[NUMBER_OF_TESTS];
    bignum host_b[NUMBER_OF_TESTS];
    // results
    bignum host_c[NUMBER_OF_TESTS];

    // these are binary operators, so we have to generate 2 * NUMBER_OF_TESTS
    // random numbers to test NUMBER_OF_TESTS tests
    for (int i = 0; i < 2 * NUMBER_OF_TESTS; i += 2)
    {
        generate_random_bignum(i, SEED, RANDOM_NUMBER_BIT_RANGE, BASE,
                               host_a[i]);
        generate_random_bignum(i + 1, SEED, RANDOM_NUMBER_BIT_RANGE, BASE,
                               host_b[i]);
    }

    // pointers to device memory for arrays "host_a" and "host_b"
    // operands
    bignum* dev_a;
    bignum* dev_b;
    // results
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

    test_kernel<<<1, 1>>>(dev_c, dev_a, dev_b);

    // copy results back to host
    cudaMemcpy(host_c, dev_c, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    bool results_correct = true;
    for (int i = 0; results_correct && i < NUMBER_OF_TESTS; i++)
    {
        char* bignum_a_str = bignum_to_string(host_a[i]);
        char* bignum_b_str = bignum_to_string(host_b[i]);
        char* bignum_c_str = bignum_to_string(host_c[i]);

        mpz_t gmp_bignum_a;
        mpz_t gmp_bignum_b;
        mpz_t gmp_bignum_c;

        mpz_init_set_str(gmp_bignum_a, bignum_a_str, BASE);
        mpz_init_set_str(gmp_bignum_b, bignum_b_str, BASE);
        mpz_init(gmp_bignum_c);

        mpz_add(gmp_bignum_c, gmp_bignum_a, gmp_bignum_b);

        // get binary string result
        char* gmp_bignum_c_str = mpz_get_str(NULL, BASE, gmp_bignum_c);
        pad_string_with_zeros(&gmp_bignum_c_str);

        if(strcmp(gmp_bignum_c_str, bignum_c_str) != 0)
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
}

__global__ void test_kernel(bignum* dev_c, bignum* dev_a, bignum* dev_b)
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

        // asm("{"
        //     "    add.cc.u32 %0, %1, %2;"
        //     "}"
        //     : "=r"(dev_c[i][0]) : "r"(dev_a[i][0]), "r"(dev_b[i][0])
        //     );

        // for (int j = 1; j < BIGNUM_NUMBER_OF_WORDS - 1; j++)
        // {
        //     asm("{"
        //         "    addc.cc.u32 %0, %1, %2;"
        //         "}"
        //         : "=r"(dev_c[i][j]) : "r"(dev_a[i][j]), "r"(dev_b[i][j])
        //         );
        // }

        // asm("{"
        //     "    addc.u32 %0, %1, %2;"
        //     "}"
        //     : "=r"(dev_c[i][BIGNUM_NUMBER_OF_WORDS - 1])
        //     : "r"(dev_a[i][BIGNUM_NUMBER_OF_WORDS - 1])
        //     , "r"(dev_b[i][BIGNUM_NUMBER_OF_WORDS - 1])
        //     );
    }
}
