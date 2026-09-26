#!/bin/bash

# Assuring we are running in the project's root
[[ -d "${PWD}/docs" && "./docs/update_apidocs.sh" == "$0" ]] ||
    {
        echo "ERROR: You must be in the project root."
        exit 1
    }

SPHINX_APIDOC="`which sphinx-apidoc`"
[[ -x "$SPHINX_APIDOC" ]] ||
    {
        echo "ERROR: sphinx-apidoc not found."
        exit 1
    }

echo "removing existing .rst files ..."
rm -f ${PWD}/docs/source/pySDC/*.rst

echo ""
echo "generating new .rst files ..."
# One run over the whole package, so that modules are documented under their import names (pySDC.core...).
${SPHINX_APIDOC} -o docs/source/pySDC pySDC pySDC/tutorial pySDC/projects pySDC/playgrounds pySDC/tests --force -T -d 2 -e
# The package page would be a second, orphaned entry point next to api.rst.
rm docs/source/pySDC/pySDC.rst

./docs/convert_markdown.py

echo ""
echo "building the wheels the tutorials install when they run in the browser ..."
# From this checkout, so the browser runs the code the pages were built from. qmat has no wheel on PyPI.
rm -rf docs/source/_static/wheels
python -m pip wheel . qmat dill --no-deps --quiet -w docs/source/_static/wheels
