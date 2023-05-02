#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --output=out/out%j.txt
#SBATCH --error=out/err%j.txt
#SBATCH -A cstma
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.baumann@fz-juelich.de
#SBATCH -p batch
#SBATCH -J red_wedding

module --force purge
module load Stages/2023
module load Intel/2022.1.0
module load ParaStationMPI/5.8.0-1

cd /p/project/ccstma/baumann7/pySDC/pySDC/projects/Resilience

source /p/project/ccstma/baumann7/miniconda/bin/activate pySDC

srun -n 64 python ${1}
