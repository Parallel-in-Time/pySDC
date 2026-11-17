#!/bin/bash

# See https://stackoverflow.com/a/28336473
SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

[[ "$0" != "${SOURCE_PATH}" ]] && echo "The activation script must be sourced, otherwise the environment will not work." || ( echo "Vars script must be sourced." && exit 1) ;

source "${ABSOLUTE_PATH}"/config.sh
[ -n "${MAMBA_ROOT_PREFIX}" ] || { echo "set MAMBA_ROOT_PREFIX, or SCRATCH with \`jutil env activate -p <project>\`"; return 1; }
# the modules before the environment, so that its Python comes first on PATH
source "${ABSOLUTE_PATH}"/modules.sh

eval "$("${MAMBA_EXE}" shell hook --shell bash --root-prefix "${MAMBA_ROOT_PREFIX}")"
micromamba activate "${ENV_NAME}"
