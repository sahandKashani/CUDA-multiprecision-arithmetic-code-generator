#!/bin/sh

if [ "$#" -eq 1 ]; then
    cd "../bin/"
    nvprof --print-gpu-trace --normalized-time-unit "$1" ../bin/benchmarks
    # nvprof --normalized-time-unit "$1" ../bin/benchmarks
    cd "../scripts/"
    python3 -u operation_checker.py
else
    echo "Please input units <ns|us|ms|s>"
fi;
