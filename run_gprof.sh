#!/bin/bash

CPPFLAGS="-g -pg -fno-omit-frame-pointer -lpthread"

SRC_CODE="./src/pthreads.cpp"
PROGNAME="fluidanimate"

THREADS="16"
FRAMES="16"
INPUT="./inputs/in_100K.fluid"
OUTPUT="./out_100K.log"
OUTPUT_GRAPH="graph.png"

g++ $CPPFLAGS $SRC_CODE -o $PROGNAME

./$PROGNAME $THREADS $FRAMES $INPUT $OUTPUT

gprof ./$PROGNAME |gprof2dot.py |dot -Tpng -o $OUTPUT_GRAPH

