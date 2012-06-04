#!/bin/bash

BUILDDIR="bin"
LOGDIR="log"
LOGFILE="benchmark.log"

CPU_BIN="fluidanimate_cpu"
GPU_BIN="fluidanimate_gpu"

CHECK_BIN="./tools/checkfiles"

CPU_THREADS="8"
GPU_THREADS="256"
FRAMES="$@"

INPUT="inputs/in_100K.fluid"
CPU_OUTPUT="out_100K_cpu.fluid"
GPU_OUTPUT="out_100K_gpu.fluid"

mkdir -p $LOGDIR

echo
echo
echo
echo "######## Start Test for $@ frames #############"
echo
echo "############### CPU Test #####################"
echo

for i in 1 2 4 8; do
    echo "$i threads"
    #command time -f '%e' -a -o $LOGFILE ./$BUILDDIR/$CPU_BIN $i $FRAMES ./$INPUT ./$LOGDIR/$CPU_OUTPUT
    time ./$BUILDDIR/$CPU_BIN $i $FRAMES ./$INPUT ./$LOGDIR/$CPU_OUTPUT
    echo
done

echo
echo "############### GPU Test #####################"
echo

time ./$BUILDDIR/$GPU_BIN $GPU_THREADS $FRAMES ./$INPUT ./$LOGDIR/$GPU_OUTPUT

echo
echo "############ Compare Results #################"
$CHECK_BIN ./$LOGDIR/$CPU_OUTPUT ./$LOGDIR/$GPU_OUTPUT
echo
echo "############## Test Finished #################"
echo
echo
echo

