#!/bin/sh

if [ "$#" -eq 2 ] && [ "$1" -eq 20 ] || [ "$1" -eq 30 ] && [ "$2" = "ns" ] || [ "$2" = "us" ] || [ "$2" = "ms" ] || [ "$2" = "s" ]; then
    rm -rf "scripts/" "src/";
    tar -xpzf zipped_files.tar.gz;
    rm -f zipped_files.tar.gz;

    cd scripts/;
    python3 random_number_generator.py;
    ./nvcc_compile.sh $1;
    ./nvcc_run.sh $2;
    python3 operation_checker.py
    cd ..;
else
    echo "usage: ./compile_and_run_on_server.sh <20|30> <ns|us|ms|s>"
fi;
