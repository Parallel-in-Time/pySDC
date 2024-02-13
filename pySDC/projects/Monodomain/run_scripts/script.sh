#!/bin/bash -l                    
#SBATCH --job-name="PFASST"                    
#SBATCH --account="u0"                    
#SBATCH --time=00:10:00                    
#SBATCH --ntasks=1                    
#SBATCH --ntasks-per-node=72                    
#SBATCH --output=/home/giacomo/Dropbox/Ricerca/Codes/Research_Codes/pySDC_and_Stabilized_for_Monodomain/pySDC/pySDC/projects/Monodomain/run_scripts/../../../../data/Monodomain/results_stability/cuboid_1D_very_large/ref_2/TTP/test.log                    
#SBATCH --cpus-per-task=1                    
#SBATCH --ntasks-per-core=1                    
#SBATCH --constraint=mc                    
#SBATCH --hint=nomultithread                                        
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                    
export CRAY_CUDA_MPS=1                    
srun python3 run_MonodomainODE_cli.py --integrator IMEXEXP_EXPRK --num_nodes 6,4 --num_sweeps 1 --mass_rhs none --max_iter 100 --dt 0.05 --restol 5e-08 --space_disc FD --domain_name cuboid_1D_very_large --pre_refinements 2 --order 4 --lin_solv_max_iter 1000 --lin_solv_rtol 1e-08 --ionic_model_name TTP --read_init_val --init_time 0.0 --no-enable_output --no-write_as_reference_solution --end_time 0.05 --output_file_name test --output_root results_stability --mass_lumping --truly_time_parallel --n_time_ranks 1 --print_stats                    
