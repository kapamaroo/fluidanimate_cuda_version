#!/bin/bash

CC="g++"

if [ `which my_nvcc.sh` ]; then
    NVCC="my_nvcc.sh"
    NVCCFLAGS=""
    echo "my_nvcc.sh found, use that"
else
    NVCC="nvcc"

    SDK_HOME="/opt/NVIDIA_GPU_Computing_SDK"

    ARCH="`uname -i`"

    if [ $ARCH = 'unknown' ]; then
        echo "unkown platform, set to i686"
        ARCH="i686"
    fi

    NVCCFLAGS="-I $SDK_HOME/C/common/inc/ \
    -L $SDK_HOME/C/lib/ -l cutil_$ARCH \
    -lcudart \
    -O4"
fi

CPPFLAGS="-lpthread -O4"
NVCCFLAGS="$NVCCFLAGS -arch=sm_20"

BINDIR="build"

CPU_BIN="fluidanimate_cpu"
GPU_BIN="fluidanimate_gpu"

CPU_SRC_CODE="./src/pthreads.cpp"
GPU_SRC_CODE="./src/cuda.cu"

mkdir -p $BINDIR

$CC $CPPFLAGS $CPU_SRC_CODE -o ./$BINDIR/$CPU_BIN

$NVCC $NVCCFLAGS $GPU_SRC_CODE -o ./$BINDIR/$GPU_BIN
