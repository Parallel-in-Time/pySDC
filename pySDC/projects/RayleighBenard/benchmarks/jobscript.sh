#!/bin/bash -x
#SBATCH -n #PROCS#
#SBATCH --tasks-per-node=#PROCS_PER_NODE#
#SBATCH -p #PARTITION#
#SBATCH --time=#WALLTIME#
#SBATCH -A #ACCOUNT#
#SBATCH -e #ERROR_FILEPATH#
#SBATCH -o #OUT_FILEPATH#
#SBATCH --job-name=benchmark 

### start of jobscript

source /p/project1/ccstma/baumann7/pySDC/pySDC/projects/GPU/etc/#VENVNAME#/activate.sh

#EXEC#
touch #READY#
