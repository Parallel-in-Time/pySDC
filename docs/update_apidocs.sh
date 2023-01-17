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
rm ${PWD}/docs/source/pySDC/*.rst
#rm -r ${PWD}/target/doc/build
#rm -r ${PWD}/*_out.txt ${PWD}/*.png run_*.log

echo ""
echo "generating new .rst files ..."
${SPHINX_APIDOC} -o docs/source/pySDC pySDC/core --force -T -d 2 -e
${SPHINX_APIDOC} -o docs/source/pySDC pySDC/implementations --force -T -d 2 -e
${SPHINX_APIDOC} -o docs/source/pySDC pySDC/helpers --force -T -d 2 -e
#rm docs/source/pySDC/pySDC.rst

echo "Running : pip install sphinxemoji m2r2"
pip install sphinxemoji m2r2 --quiet
./docs/convert_markdown.py
