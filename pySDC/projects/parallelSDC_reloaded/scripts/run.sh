#!/bin/bash

# Run all python scripts
for script in fig*.py
do
    echo "------------------------------------------------------------"
    echo "Running ${script} python script ..."
    python "${script}"
done