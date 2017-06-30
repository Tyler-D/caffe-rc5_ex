#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe test --model=examples/xnor/cifar10_xnor.proto \
    --weights=examples/xnor/cifar10_xnor_iter_40000.caffemodel --iterations=100 --gpu 0 \
           
