#!/bin/bash -x
#SBATCH -n #PROCS#
#SBATCH --tasks-per-node=#PROCS_PER_NODE#
#SBATCH -p #PARTITION#
#SBATCH --time=#WALLTIME#
#SBATCH -e #ERROR_FILEPATH#
#SBATCH -o #OUT_FILEPATH#
#SBATCH --job-name=benchmark 
#SBATCH -A jureap27
#SBATCH --mail-user=t.baumann@fz-juelich.de
#SBATCH --mail-type=BEGIN,END,FAIL

### start of jobscript

source /e/scratch/jureap1/baumann7/pySDC/pySDC/projects/GPU/etc/venv_jupiter/activate.sh

#EXEC#
touch #READY#
