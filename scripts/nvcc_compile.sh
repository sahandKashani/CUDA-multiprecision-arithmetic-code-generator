#!/bin/sh

if [ "$#" -eq 1 ]; then
    ./to_run_before_compilation.py;

    cd "../src/";
    nvcc -arch=sm_"$1" main.cu input_output.cpp -o ../bin/benchmarks;
else
    echo "Please input architecture <20|30>";
fi;
