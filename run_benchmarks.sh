#!/bin/bash

if [ "$#" -eq 2 ] && [ "$1" -eq 20 ] || [ "$1" -eq 30 ] && [ "$2" = "ns" ] || [ "$2" = "us" ] || [ "$2" = "ms" ] || [ "$2" = "s" ]; then
    cd scripts/
    for blocks in 1 2 4 8 16 32 64 128 256 512 1024
    do
        for threads in 1 2 4 8 16 32 64 128 256 512 1024
            for bits in 109 131 163
            do
                perl -i -pe "s/precision = \d+/precision = $bits/g" constants.py
                perl -i -pe "s/threads_per_block = \d+/threads_per_block = $threads/g" constants.py
                perl -i -pe "s/blocks_per_grid = \d+/blocks_per_grid = $blocks/g" constants.py
                python3 constants.py
                python3 operation_generator.py
                ./nvcc_compile.sh $1
                ./nvcc_run.sh $2
            done
        done
    done
else
    echo "usage: ./compile_and_run_on_server.sh <20|30> <ns|us|ms|s>"
fi;
