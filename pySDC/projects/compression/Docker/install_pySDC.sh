#!/bin/bash

# store current working directory to return here later
current_dir=$(pwd)

# load the spack environment variables
source /opt/spack/share/spack/setup-env.sh

# load libpressio in spack to make sure we are using the correct Python
spack load libpressio

which mpicc
mpicc --version

# install local version of pySDC and other dependencies
python -m pip install --upgrade pip
cd /pySDC
pip install -e .
python -m pip install pytest
python -m pip install coverage
python -m pip install --no-binary :all: mpi4py

python -c "from mpi4py import MPI"

# go back to original working directory
cd $current_dir
