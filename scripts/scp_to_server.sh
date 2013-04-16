#!/bin/sh

if [ "$#" -eq 1 ]; then
    scp -r ../src/ pollardrho@"$1":Pollard-Rho-CUDA/;
else
    echo "Please input server address";
fi;
