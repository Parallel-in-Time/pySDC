#!/bin/bash

# load spack environment variables
source /opt/spack/share/spack/setup-env.sh

allparams=("$@")

echo "Install pySDC"
source /pySDC/pySDC/projects/compression/Docker/install_pySDC.sh

echo "Done"
 
# open a new shell to keep the container running
/bin/bash "${allparams[@]}"
