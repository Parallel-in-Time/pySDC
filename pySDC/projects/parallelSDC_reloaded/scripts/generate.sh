#!/bin/bash

# Run all python scripts
for script in fig*.py
do
    echo "------------------------------------------------------------"
    echo "Running ${script} python script ..."
    python "${script}"
done

# Trim all figures
for fig in *.pdf
do
    pdfcrop "${fig}" "${fig}"
done