#!/bin/bash -x
#SBATCH --account=${JUWELS_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --output=sbatch.out
#SBATCH --error=sbatch.err

python -m pytest --continue-on-collection-errors -v pySDC/tests -m "benchmark" --benchmark-json=benchmarks/output.json
