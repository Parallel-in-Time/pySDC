#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
source "${ABSOLUTE_PATH}"/config.sh

KERNELFILE="${ENV_DIR}"/kernel.sh

echo the name is "$ENV_NAME"

echo "Setting up the kernel script in the following dir: " "${KERNELFILE}"
echo "If you use multiprocessing, edit this file to remove the srun call from the kernel and run it again."
echo '#!/bin/bash

source "'"${ABSOLUTE_PATH}"'"/activate.sh

hostname=$(hostname)

if [[ $hostname == *"login"* || $hostname == *"jsfl"* ]]; then
   exec python -m ipykernel "$@"
else
   srun python -m ipykernel "$@"
fi
' > "${KERNELFILE}"

chmod a+x "${KERNELFILE}"

mkdir -p ~/.local/share/jupyter/kernels/"${ENV_NAME}"
echo '{
 "argv": [
  "'"${KERNELFILE}"'",
  "-f",
  "{connection_file}"
 ],
 "display_name": "'"${ENV_NAME}"'",
 "language": "python"
}' > ~/.local/share/jupyter/kernels/"${ENV_NAME}"/kernel.json
