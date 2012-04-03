#!/bin/bash

CC="g++"
CPPFLAGS="-g -pg -fno-omit-frame-pointer -lpthread"

TOOLSDIR="tools"
BUILDDIR="bin"

SRC_CODE="./src/pthreads.cpp"
PROGNAME="fluidanimate_gprof"

THREADS="16"
FRAMES="16"
INPUT="./inputs/in_100K.fluid"
OUTPUT_GRAPH="graph.png"

mkdir -p $BUILDDIR

$CC $CPPFLAGS $SRC_CODE -o ./$BUILDDIR/$PROGNAME

./$BUILDDIR/$PROGNAME $THREADS $FRAMES $INPUT

gprof ./$BUILDDIR/$PROGNAME |./$TOOLSDIR/gprof2dot.py |dot -Tpng -o $OUTPUT_GRAPH

#rm ./$BUILDDIR/$PROGNAME
