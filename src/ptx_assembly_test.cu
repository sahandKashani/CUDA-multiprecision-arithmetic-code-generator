#include "bignum_type.h"
#include "interleaved_bignum_type.h"
#include "random_bignum_generator.h"
#include "bignum_conversions.h"

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

#define NUMBER_OF_TESTS ((unsigned long int) 65536) // 2^16

void execute_normal_addition_on_device(bignum* host_c, bignum* host_a,
                                       bignum* host_b);
__global__ void normal_addition(bignum* dev_c, bignum* dev_a, bignum* dev_b);


void execute_coalescing_addition_on_device(bignum* host_c, bignum* host_a,
                                           bignum* host_b);
__global__ void coalescing_addition(bignum* dev_results,
                                    interleaved_bignum* dev_interleaved_operands);


void generate_operands(bignum* host_a, bignum* host_b);
void check_results(bignum* host_c, bignum* host_a, bignum* host_b);

int main(void)
{
    printf("Testing PTX\n");

    // host operands (host_a, host_b) and results (host_c)
    bignum* host_a = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));
    bignum* host_b = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));
    bignum* host_c = (bignum*) calloc(NUMBER_OF_TESTS, sizeof(bignum));

    // generate random numbers for the tests
    generate_operands(host_a, host_b);

    execute_normal_addition_on_device(host_c, host_a, host_b);
    // execute_coalescing_addition_on_device(host_c, host_a, host_b);

    check_results(host_c, host_a, host_b);

    free(host_a);
    free(host_b);
    free(host_c);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Normal Addition ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void execute_normal_addition_on_device(bignum* host_c, bignum* host_a,
                                       bignum* host_b)
{
    // device operands (dev_a, dev_b) and results (dev_c)
    bignum* dev_a;
    bignum* dev_b;
    bignum* dev_c;

    cudaMalloc((void**) &dev_a, NUMBER_OF_TESTS * sizeof(bignum));
    cudaMalloc((void**) &dev_b, NUMBER_OF_TESTS * sizeof(bignum));
    cudaMalloc((void**) &dev_c, NUMBER_OF_TESTS * sizeof(bignum));

    // copy operands to device memory
    cudaMemcpy(dev_a, host_a, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyHostToDevice);

    normal_addition<<<256, 256>>>(dev_c, dev_a, dev_b);

    // copy results back to host
    cudaMemcpy(host_c, dev_c, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

__global__ void normal_addition(bignum* dev_c, bignum* dev_a, bignum* dev_b)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < NUMBER_OF_TESTS)
    {
        asm("{"
            "    add.cc.u32  %0, %5, %10;"
            "    addc.cc.u32 %1, %6, %11;"
            "    addc.cc.u32 %2, %7, %12;"
            "    addc.cc.u32 %3, %8, %13;"
            "    addc.u32    %4, %9, %14;"
            "}"

            : "=r"(dev_c[tid][0]),
              "=r"(dev_c[tid][1]),
              "=r"(dev_c[tid][2]),
              "=r"(dev_c[tid][3]),
              "=r"(dev_c[tid][4])

            : "r"(dev_a[tid][0]),
              "r"(dev_a[tid][1]),
              "r"(dev_a[tid][2]),
              "r"(dev_a[tid][3]),
              "r"(dev_a[tid][4]),

              "r"(dev_b[tid][0]),
              "r"(dev_b[tid][1]),
              "r"(dev_b[tid][2]),
              "r"(dev_b[tid][3]),
              "r"(dev_b[tid][4])
            );

        tid += blockDim.x * gridDim.x;
    }
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Coalescing Addition ////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void execute_coalescing_addition_on_device(bignum* host_c, bignum* host_a,
                                           bignum* host_b)
{
    // for this coalescing addition, we are going to interleave the values of
    // the 2 operands host_a and host_b.
    // Our operands will look like the following:

    // host_a[0][0], host_b[0][0], host_a[0][1], host_b[0][1],
    // host_a[0][2], host_b[0][2], host_a[0][3], host_b[0][3],
    // host_a[0][4], host_b[0][4], host_a[1][0], host_b[1][0], ...

    // our results will be stocked sequentially as for normal addition.

    void* host_ops = calloc(NUMBER_OF_TESTS, sizeof(interleaved_bignum));
    interleaved_bignum* interleaved_operands = (interleaved_bignum*) host_ops;
    host_ops = NULL;

    // interleave values of host_a and host_b in interleaved_operands.
    for (int i = 0; i < NUMBER_OF_TESTS; i++)
    {
        for (int j = 0; j < INTERLEAVED_BIGNUM_NUMBER_OF_WORDS; j++)
        {
            if (j % 2 == 0)
            {
                interleaved_operands[i][j] = host_a[i][j / 2];
            }
            else
            {
                interleaved_operands[i][j] = host_b[i][j / 2];
            }
        }
    }

    // device operands (dev_interleaved_operands) and results (dev_results)
    interleaved_bignum* dev_interleaved_operands;
    bignum* dev_results;

    cudaMalloc((void**) &dev_interleaved_operands,
               NUMBER_OF_TESTS * sizeof(interleaved_bignum));
    cudaMalloc((void**) &dev_results, NUMBER_OF_TESTS * sizeof(bignum));

    // copy operands to device memory
    cudaMemcpy(dev_interleaved_operands, interleaved_operands,
               NUMBER_OF_TESTS * sizeof(interleaved_bignum),
               cudaMemcpyHostToDevice);

    // free interleaved_operands which we no longer need.
    free(interleaved_operands);

    coalescing_addition<<<256, 256>>>(dev_results, dev_interleaved_operands);

    // copy results back to host
    cudaMemcpy(host_c, dev_results, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_interleaved_operands);
    cudaFree(dev_results);
}

__global__ void coalescing_addition(bignum* dev_results,
                                    interleaved_bignum* dev_interleaved_operands)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < NUMBER_OF_TESTS)
    {
        asm("{"
            "    add.cc.u32  %0, %5, %10;"
            "    addc.cc.u32 %1, %6, %11;"
            "    addc.cc.u32 %2, %7, %12;"
            "    addc.cc.u32 %3, %8, %13;"
            "    addc.u32    %4, %9, %14;"
            "}"

            : "=r"(dev_results[tid][0]),
              "=r"(dev_results[tid][1]),
              "=r"(dev_results[tid][2]),
              "=r"(dev_results[tid][3]),
              "=r"(dev_results[tid][4])

            : "r"(dev_interleaved_operands[tid][0]),
              "r"(dev_interleaved_operands[tid][2]),
              "r"(dev_interleaved_operands[tid][4]),
              "r"(dev_interleaved_operands[tid][6]),
              "r"(dev_interleaved_operands[tid][8]),

              "r"(dev_interleaved_operands[tid][1]),
              "r"(dev_interleaved_operands[tid][3]),
              "r"(dev_interleaved_operands[tid][5]),
              "r"(dev_interleaved_operands[tid][7]),
              "r"(dev_interleaved_operands[tid][9])
            );

        tid += blockDim.x * gridDim.x;
    }
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// General //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
 * Generates random numbers and assigns them to the 2 bignum arrays passed as a
 * parameter.
 * @param host_a first array to populate
 * @param host_b second array to populate
 */
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

/**
 * Checks if host_a op host_b == host_c, where host_c is to be tested against
 * values computed by gmp. If you have data in any other formats than these, you
 * will have to "rearrange" them to meet this pattern for the check to work.
 * @param host_c Values we have computed with our algorithms.
 * @param host_a First operands.
 * @param host_b Second operands.
 */
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

        // CHOOSE GMP FUNCTION TO EXECUTE HERE
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
