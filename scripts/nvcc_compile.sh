#!/bin/sh

if [ "$#" -eq 1 ]; then
    cd "../src/";
    nvcc -arch=sm_"$1" *.cu *.cpp -o ../bin/benchmarks -lgmp;
else
    echo "Please input architecture <20|30>";
fi;
