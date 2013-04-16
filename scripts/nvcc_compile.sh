#!/bin/sh

if [ "$#" -eq 1 ]; then
    cd "../src/";
    nvcc -arch=sm_"$1" benchmarks.cu bignum_conversions.cpp io_interface.cpp main.cu operation_check.cpp random_bignum_generator.cpp -o ../bin/benchmarks -lgmp;
else
    echo "Please input architecture <20|30>";
fi;
