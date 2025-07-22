#!/bin/bash

# See https://stackoverflow.com/a/28336473
SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

source "${ABSOLUTE_PATH}"/activate.sh

python3 -m pip install --upgrade pip

python3 -m pip install -r "${ABSOLUTE_PATH}"/requirements.txt

FFTW_LIBRARY_DIR="/p/software/juwels/stages/2025/software/FFTW/3.3.10-GCC-13.3.0/lib64/" python3 -m pip install git+https://github.com/brownbaerchen/mpi4py-fft.git@cupy_implementation

python3 -m pip install -e "${ABSOLUTE_PATH}"/../../

