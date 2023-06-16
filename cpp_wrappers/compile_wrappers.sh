#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
python3 setup.py build_ext --inplace
cd ../
cd cpp_neighbors
python3 setup.py build_ext --inplace
cd ../
cd cpp_pcf_kernel
python3 setup.py install
cd ../
