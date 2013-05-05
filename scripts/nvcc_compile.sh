#!/bin/sh

if [ "$#" -eq 1 ]; then
    python3 -u to_run_before_compilation.py
    cd "../src/"
    nvcc -arch=sm_"$1" main.cu benchmarks.cu input_output.cpp -o ../bin/benchmarks
else
    echo "Please input architecture <20|30>"
fi;
