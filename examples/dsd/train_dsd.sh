#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/dsd/cifar10_dsd_solver.prototxt \
  --weights=examples/dsd/cifar10_quick_iter_5000.caffemodel \
  2>&1 | tee ./log_sparse
# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=examples/dsd/cifar10_dsd_solver_dense.prototxt \
  --weights=examples/dsd/cifar10_dsd_iter_3000.caffemodel \
  2>&1 | tee ./log_dense
