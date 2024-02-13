#!/bin/bash -l
#SBATCH --job-name="monodomain"
#SBATCH --account="u0"
#SBATCH --time=00:05:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=./output.log
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
#SBATCH --partition=debug
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
srun python3 playground_parallelization.py
