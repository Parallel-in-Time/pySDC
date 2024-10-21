#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"
source "${ABSOLUTE_PATH}"/config.sh
PYTHONWRAPPER="${ABSOLUTE_PATH}"/python

echo '#!/bin/bash
module purge 2> /dev/null
deactivate 2> /dev/null
source '"'${ABSOLUTE_PATH}'"'/activate.sh
python "$@"
' > "${PYTHONWRAPPER}"

chmod a+x "${PYTHONWRAPPER}"
