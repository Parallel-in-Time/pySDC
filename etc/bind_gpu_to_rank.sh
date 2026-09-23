#!/bin/bash
#
# Give each MPI rank a GPU of its own, then run the command that follows.
#
# NCCL wants one GPU per rank. Nothing in pySDC selects a device, and on a batch system it does
# not have to: the scheduler hands each task its own, so every rank taking device 0 is correct.
# Inside one container with several GPUs attached it is not, and NCCL refuses the duplicate.
#
# This is a script rather than an export in the caller because the rank is only known after the
# launcher has forked, and `run_mpi_tests.sh` expands `$PYTEST` in the launching shell.
set -eu

export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK:-0}"

exec "$@"
