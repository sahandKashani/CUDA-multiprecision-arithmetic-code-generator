#!/bin/sh

if [ "$#" -eq 1 ]; then
    cd "../src/"
    # nvcc -arch=sm_"$1" --ptxas-options=-v main.cu benchmarks.cu input_output.cpp -o ../bin/benchmarks
    nvcc -arch=sm_"$1" main.cu benchmarks.cu input_output.cpp -o ../bin/benchmarks
else
    echo "Please input architecture <20|30>"
fi;
