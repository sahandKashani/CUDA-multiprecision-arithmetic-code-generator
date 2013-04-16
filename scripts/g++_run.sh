#!/bin/sh

cd "../bin/";
valgrind --leak-check=full --show-reachable=yes --track-origins=yes ./../bin/benchmarks;
