#!/bin/bash

mkdir -p build
cd build && rm rf *
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)  # Or use: cmake --build . --config Release