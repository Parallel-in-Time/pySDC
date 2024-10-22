#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

source "${ABSOLUTE_PATH}"/activate.sh

FFTW_LIBRARY_DIR="/p/software/jusuf/stages/2024/software/FFTW/3.3.10-GCC-12.3.0/lib64" python3 -m pip install -e /p/project/ccstma/baumann7/mpi4py-fft
python3 -m pip install -e /p/project1/ccstma/baumann7/qmat
python3 -m pip install -r "${ABSOLUTE_PATH}"/requirements.txt
python3 -m pip install -e /p/project1/ccstma/baumann7/pySDC/

