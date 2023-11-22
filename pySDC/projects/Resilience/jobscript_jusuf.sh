#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=1:00:00
#SBATCH --output=out/out%j.txt
#SBATCH --error=out/err%j.txt
#SBATCH -A cstma
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.baumann@fz-juelich.de
#SBATCH -p batch
#SBATCH -J MarsDirect2023

module --force purge
module load Stages/2023
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load mpi4py/3.1.4

cd /p/project/ccstma/baumann7/pySDC/pySDC/projects/Resilience

source /p/project/ccstma/baumann7/miniconda/bin/activate pySDC

srun -n 64 python ${1}
