#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

python3 -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"
python3 -m pip install --upgrade pip

source "${ABSOLUTE_PATH}"/activate.sh


FFTW_LIBRARY_DIR="/p/software/juwels/stages/2025/software/FFTW/3.3.10-GCC-13.3.0/lib64/" python3 -m pip install git+https://github.com/brownbaerchen/mpi4py-fft.git@cupy_implementation

python3 -m pip install -r "${ABSOLUTE_PATH}"/requirements.txt

python3 -m pip install -e "${ABSOLUTE_PATH}"/../../

