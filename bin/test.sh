#!/bin/bash

BUILDDIR="build"
LOGDIR="log"

CPU_BIN="fluidanimate_cpu"
GPU_BIN="fluidanimate_gpu"

THREADS="256"
FRAMES="1"

INPUT="./inputs/in_100K.fluid"
CPU_OUTPUT="./out_100K_cpu.fluid"
GPU_OUTPUT="./out_100K_gpu.fluid"

mkdir -p $LOGDIR

#./$BUILDDIR/$CPU_BIN $THREADS $FRAMES $INPUT ./$LOGDIR/$CPU_OUTPUT
./$BUILDDIR/$GPU_BIN $THREADS $FRAMES $INPUT ./$LOGDIR/$GPU_OUTPUT
