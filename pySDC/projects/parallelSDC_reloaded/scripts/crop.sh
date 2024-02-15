#!/bin/bash

# Trim all figures
echo "------------------------------------------------------------"
echo "Croping figures ..."
for fig in *.pdf
do
    pdfcrop "${fig}" "${fig}"
done