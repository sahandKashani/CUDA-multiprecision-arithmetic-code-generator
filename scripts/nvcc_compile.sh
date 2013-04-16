#!/bin/sh

cd "../src/";
nvcc benchmarks.cu bignum_conversions.cpp io_interface.cpp main.cu operation_check.cpp random_bignum_generator.cpp -o ../bin/benchmarks -lgmp;
