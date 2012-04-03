#!/bin/bash

SDK_HOME="/opt/NVIDIA_GPU_Computing_SDK/"
ARCH="i386"

nvcc -I $SDK_HOME/C/common/inc/ \
    -L $SDK_HOME/C/lib/ -l cutil_$ARCH \
    -lcudart \
    -O4 \
    "$@"
