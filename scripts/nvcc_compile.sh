#!/bin/sh

if [ "$#" -eq 1 ]; then
    python3 operation_generator.py > ../src/operations.h;
    cd "../src/";
    nvcc -arch=sm_"$1" *.cu *.cpp -o ../bin/benchmarks -lgmp;
else
    echo "Please input architecture <20|30>";
fi;
