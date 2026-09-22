#!/bin/bash

# store current working directory to return here later
current_dir=$(pwd)

# load the spack environment variables
source /opt/spack/share/spack/setup-env.sh

# load libpressio in spack to make sure we are using the correct Python
spack load libpressio

# install local version of pySDC and other dependencies
python -m pip install --upgrade pip
cd /pySDC
# numpy<2 is this image's constraint, not pySDC's: libpressio here is a spack build whose `_pressio`
# extension is compiled against NumPy 1.x, and it fails to import the moment anything replaces numpy
# underneath it. Held explicitly rather than relying on the image's preinstalled numpy happening to
# satisfy the resolver.
pip install -e . "numpy<2"
python -m pip install pytest
python -m pip install coverage
python -m pip install "mpi4py<4.1.0"

# go back to original working directory
cd $current_dir
