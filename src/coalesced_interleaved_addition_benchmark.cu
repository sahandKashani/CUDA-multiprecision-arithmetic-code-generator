void execute_coalesced_interleaved_addition_on_device(bignum* host_c,
                                                      bignum* host_a,
                                                      bignum* host_b,
                                                      uint32_t threads_per_block,
                                                      uint32_t blocks_per_grid)
{
    coalesced_interleaved_bignum* coalesced_interleaved_operands =
        (coalesced_interleaved_bignum*)
            calloc(BIGNUM_NUMBER_OF_WORDS, sizeof(coalesced_interleaved_bignum));

    // arrange values of host_a and host_b in coalesced_interleaved_operands.
    for (uint32_t i = 0; i < BIGNUM_NUMBER_OF_WORDS; i++)
    {
        for (uint32_t j = 0; j < 2 * TOTAL_NUMBER_OF_THREADS; j += 2)
        {
            coalesced_interleaved_operands[i][j]     = host_a[j / 2][i];
            coalesced_interleaved_operands[i][j + 1] = host_b[j / 2][i];
        }
    }

    // device operands (dev_coalesced_interleaved_operands) and results
    // (dev_coalesced_results)
    coalesced_interleaved_bignum* dev_coalesced_interleaved_operands;
    coalesced_bignum* dev_coalesced_results;

    cudaMalloc((void**) &dev_coalesced_interleaved_operands,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_interleaved_bignum));
    cudaMalloc((void**) &dev_coalesced_results,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum));

    // copy operands to device memory
    cudaMemcpy(dev_coalesced_interleaved_operands,
               coalesced_interleaved_operands,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_interleaved_bignum),
               cudaMemcpyHostToDevice);

    free(coalesced_interleaved_operands);

    coalesced_interleaved_addition<<<blocks_per_grid, threads_per_block>>>(
        dev_coalesced_results, dev_coalesced_interleaved_operands);

    coalesced_bignum* host_coalesced_results =
        (coalesced_bignum*) calloc(BIGNUM_NUMBER_OF_WORDS,
                                   sizeof(coalesced_bignum));

    // copy results back to host
    cudaMemcpy(host_coalesced_results, dev_coalesced_results,
               BIGNUM_NUMBER_OF_WORDS * sizeof(coalesced_bignum),
               cudaMemcpyDeviceToHost);

    // rearrange result values into host_c
    for (uint32_t i = 0; i < TOTAL_NUMBER_OF_THREADS; i++)
    {
        for (uint32_t j = 0; j < BIGNUM_NUMBER_OF_WORDS; j++)
        {
            host_c[i][j] = host_coalesced_results[j][i];
        }
    }

    free(host_coalesced_results);

    // free device memory
    cudaFree(dev_coalesced_interleaved_operands);
    cudaFree(dev_coalesced_results);
}

__global__ void coalesced_interleaved_addition(
    coalesced_bignum* dev_coalesced_results,
    coalesced_interleaved_bignum* dev_coalesced_interleaved_operands)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid_increment = blockDim.x * gridDim.x;

    while (tid < TOTAL_NUMBER_OF_THREADS)
    {
        uint32_t i = 0;
        uint32_t col = 2 * tid;

        asm("add.cc.u32 %0, %1, %2;"
            : "=r"(dev_coalesced_results[i][tid])
            : "r" (dev_coalesced_interleaved_operands[i][col]),
              "r" (dev_coalesced_interleaved_operands[i][col + 1]));

        #pragma unroll
        for (i = 1; i < BIGNUM_NUMBER_OF_WORDS - 1; i++)
        {
            asm("addc.cc.u32 %0, %1, %2;"
                : "=r"(dev_coalesced_results[i][tid])
                : "r" (dev_coalesced_interleaved_operands[i][col]),
                  "r" (dev_coalesced_interleaved_operands[i][col + 1]));
        }

        asm("addc.u32 %0, %1, %2;"
            : "=r"(dev_coalesced_results[i][tid])
            : "r" (dev_coalesced_interleaved_operands[i][col]),
              "r" (dev_coalesced_interleaved_operands[i][col + 1]));

        tid += tid_increment;
    }
}
