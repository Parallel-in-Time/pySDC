#!/bin/bash -x
#SBATCH --account=cstma
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --output=sbatch.out
#SBATCH --error=sbatch.err

srun python -m pytest --continue-on-collection-errors -v pySDC/tests -m "benchmark" --benchmark-json=benchmarks.json
