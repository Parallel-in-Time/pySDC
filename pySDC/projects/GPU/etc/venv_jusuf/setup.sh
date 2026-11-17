#!/bin/bash
# Create the micromamba environment from the same files as pySDC's GPU CI (etc/environment-cupy.yml and
# etc/environment-tests.yml), but with conda-forge's external MPICH in place of openmpi: mpi4py and mpi4py-fft are
# built against the MPICH interface and load ParaStationMPI from the module at run time (both use libmpi.so.12).
set -euo pipefail

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
# the pySDC checkout this file lives in
REPO="$(realpath "${ABSOLUTE_PATH}"/../../../../..)"

source "${ABSOLUTE_PATH}"/config.sh
# before anything imports mpi4py, which needs the system libmpi.so.12
source "${ABSOLUTE_PATH}"/modules.sh

[ -x "${MAMBA_EXE}" ] || { echo "micromamba not found: put it on PATH or set MAMBA_EXE"; exit 1; }
[ -n "${MAMBA_ROOT_PREFIX}" ] || { echo "set MAMBA_ROOT_PREFIX, or SCRATCH with \`jutil env activate -p <project>\`"; exit 1; }
MAMBA=("${MAMBA_EXE}" --root-prefix "${MAMBA_ROOT_PREFIX}")

# The external MPI placeholders live on their own label, which has to come first, or strict priority hides them.
# Python 3.13 and CUDA 12 to match the driver; a compiler, cython and setuptools to build the mpi4py-fft fork below.
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
grep -q '^  - openmpi$' "${REPO}"/etc/environment-cupy.yml || { echo "no openmpi to replace in environment-cupy.yml"; exit 1; }
sed -e 's|^  - openmpi$|  - "mpich=4.3.*=external_*"|' \
    -e 's|^dependencies:$|dependencies:\n  - python=3.13\n  - "cuda-version=12.*"\n  - c-compiler\n  - cython\n  - setuptools|' \
    "${REPO}"/etc/environment-cupy.yml > "${WORK}"/environment.yml

"${MAMBA[@]}" env remove -y -n "${ENV_NAME}" > /dev/null 2>&1 || true
# one create with both files: installing the test file afterwards lets conda overwrite what pip installed
"${MAMBA[@]}" create -y -n "${ENV_NAME}" -f "${WORK}"/environment.yml -f "${REPO}"/etc/environment-tests.yml \
    --override-channels -c conda-forge/label/mpi-external -c conda-forge --strict-channel-priority

PIP=("${MAMBA[@]}" run -n "${ENV_NAME}" python -m pip install -q)
# The GPU spectral code needs an mpi4py-fft that no release provides: the CuPy FFT backends, `DistArrayCuPy` and the
# NCCL `comm_backend` only exist on this fork, pinned to the commit CI uses (etc/modal_gpu_tests.py). Built against
# conda's NumPy, mpi4py and FFTW without replacing them; it reports the same version as the release it replaces.
"${PIP[@]}" --no-deps --no-build-isolation --force-reinstall \
    https://github.com/brownbaerchen/mpi4py-fft/archive/a7aeec6ace99dd49561625c866c605ed0b337c18.tar.gz
# provides the `parallel(n)` marker of the MPI tests
"${PIP[@]}" "mpi-pytest>=2026.0"
"${PIP[@]}" --no-deps -e "${REPO}"
"${MAMBA[@]}" clean -y --all > /dev/null
