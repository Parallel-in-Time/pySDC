#!/bin/bash
#
# Give each MPI rank a GPU of its own, then run the command that follows.
#
# NCCL wants exactly one GPU per rank. Nothing in pySDC selects a device -- there is no
# `cupy.cuda.Device(rank).use()` anywhere -- because on a batch system the scheduler already
# handed each task its own, so every rank simply takes device 0 and is right. Inside one
# container with several GPUs attached that assumption breaks: every rank takes device 0, the
# same one, and NCCL either refuses the duplicate or hangs.
#
# Restricting the visible devices per rank reproduces what the scheduler does, and keeps the
# "device 0" assumption true, so nothing in pySDC has to change.
#
# Usage: bind_gpu_to_rank.sh <command> [args...]
set -eu

# OpenMPI sets the local rank; without it (a serial pass, or a different launcher) fall back to
# the first device rather than leaving every GPU visible, so the choice is explicit either way.
export CUDA_VISIBLE_DEVICES="${OMPI_COMM_WORLD_LOCAL_RANK:-0}"

exec "$@"
