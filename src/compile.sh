#!/bin/sh

g++ -g bignum_conversions.cpp io_interface.cpp main.cpp operation_check.cpp random_bignum_generator.cpp -o ../bin/benchmarks -lgmp;
