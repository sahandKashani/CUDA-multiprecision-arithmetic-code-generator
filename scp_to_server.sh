#!/bin/sh

if [ "$#" -eq 1 ]; then
    cd scripts/;
    python3 constants.py;
    python3 operation_generator.py;
    cd ..;
    rm -rf "scripts/__pycache__";
    tar -zcpf zipped_files.tar.gz scripts/ src/;
    scp zipped_files.tar.gz pollardrho@"$1":Pollard-Rho-CUDA/;
    rm -f zipped_files.tar.gz;
else
    echo "Please input server address";
fi;
