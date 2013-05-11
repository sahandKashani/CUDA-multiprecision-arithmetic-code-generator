#!/bin/sh

if [ "$#" -eq 2 ] && [ "$1" -eq 20 ] || [ "$1" -eq 30 ] && [ "$2" = "ns" ] || [ "$2" = "us" ] || [ "$2" = "ms" ] || [ "$2" = "s" ]; then
    rm -rf "scripts/" "src/";
    tar -xpzf zipped_files.tar.gz;
    rm -f zipped_files.tar.gz;

    cd scripts/;
    ./nvcc_compile.sh 30;
    ./nvcc_run.sh us;
    cd ..;
else
    echo "usage: ./compile_on_server.sh <20|30> <ns|us|ms|s>"
fi;
