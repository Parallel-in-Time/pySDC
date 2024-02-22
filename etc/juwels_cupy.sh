#!/bin/bash -x
#SBATCH --account=cstma
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=devel-gpu
#SBATCH --output=sbatch.out
#SBATCH --error=sbatch.err

srun coverage run -m pytest --continue-on-collection-errors -v pySDC/tests -m "cupy"
