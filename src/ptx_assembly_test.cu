#include "bignum_type.h"
#include "random_bignum_generator.h"

#include <stdio.h>

#define NUMBER_OF_TESTS 2

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

    // test_kernel<<<1, 1>>>(dev_c, dev_a, dev_b);

    // copy results back to host
    cudaMemcpy(host_c, dev_c, NUMBER_OF_TESTS * sizeof(bignum),
               cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

// __global__ void test_kernel(int* dev_number, int a, int host_b)
// {
//     int c;

//     asm("{"
//         "    add.u32 %0, %1, %2;"
//         "}"
//         : "=r"(c) : "r"(a), "r"(host_b)
//         );

//     *dev_c = c;
// }
