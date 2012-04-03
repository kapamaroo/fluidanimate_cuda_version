#!/bin/bash

BUILDDIR="bin"
LOGDIR="log"

CPU_BIN="fluidanimate_cpu"
GPU_BIN="fluidanimate_gpu"

THREADS="256"
FRAMES="$@"

INPUT="inputs/in_100K.fluid"
CPU_OUTPUT="out_100K_cpu.fluid"
GPU_OUTPUT="out_100K_gpu.fluid"

mkdir -p $LOGDIR

echo "###"
echo "#"
echo "# Start Test for $@ frames"
echo "#"
echo "# CPU Test"
echo "#"

./$BUILDDIR/$CPU_BIN $THREADS $FRAMES ./$INPUT ./$LOGDIR/$CPU_OUTPUT

echo "#"
echo "# GPU Test"
echo "#"

./$BUILDDIR/$GPU_BIN $THREADS $FRAMES ./$INPUT ./$LOGDIR/$GPU_OUTPUT

echo "#"
echo "# Test Finished."
