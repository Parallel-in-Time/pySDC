#!/bin/bash -l                    
#SBATCH --job-name="monodomain"                    
#SBATCH --account="s1074"                    
#SBATCH --time=00:30:00                    
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12                    
#SBATCH --output=./PFASST_MPI.log                    
#SBATCH --cpus-per-task=1                    
#SBATCH --ntasks-per-core=1                    
#SBATCH --constraint=gpu                    
#SBATCH --hint=nomultithread                    
#SBATCH --partition=debug
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                    
export CRAY_CUDA_MPS=1                    
srun python3 run_MonodomainSystem_FEniCSx_MPI.py
