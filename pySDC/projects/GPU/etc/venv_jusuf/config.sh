SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

## Check if this script is sourced
[[ "$0" != "${SOURCE_PATH}" ]] && echo "Setting vars" || ( echo "Vars script must be sourced." && exit 1) ;
## Determine location of this file
RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
####################################

### User Configuration
export ENV_NAME="${ENV_NAME:-pySDC-GPU}"
# micromamba is a single binary, see https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
export MAMBA_EXE="${MAMBA_EXE:-$(command -v micromamba)}"
# All packages and environments, tens of thousands of files: on scratch, because project storage is short of inodes.
# Scratch is purged after a while, which is fine for an environment `setup.sh` rebuilds in about a minute. JSC sets
# $SCRATCH for the active project; without it there is no default, rather than tens of thousands of files in $HOME.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${SCRATCH:+${SCRATCH}/${USER}/micromamba}}"
