#!/bin/bash

if [ "$1" == "test" ]; then
    echo "=== Running Tests ==="
    cmake -S . -B build 
    cmake --build build
    ./build/tests/ml_lib_tests
elif [ "$1" == "run" ]; then
    echo "=== Building Project ==="
    cmake -S . -B build
    cmake --build build
    ./build/ml_test_app
else
    echo "Usage: $0 {test|run}"
    exit 1
fi