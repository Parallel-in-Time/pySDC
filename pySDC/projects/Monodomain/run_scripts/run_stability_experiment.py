import os
import numpy as np
from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE_cli_slurm_wrapper import execute_with_dependencies, options_command

"""
Finished experiments:
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-1024
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,4, dt = 0.05, n_time_ranks = 1-1024
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-1024
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,4, dt = 0.2, n_time_ranks = 1-1024

    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-1024
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,4,2, dt = 0.05, n_time_ranks = 1-1024
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-1024
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,4,2, dt = 0.2, n_time_ranks = 1-1024

    - TTP, domain_name = cube_2D, pre_refinements = 0, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = 0, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-128

    - TTP, domain_name = cube_2D, pre_refinements = 0, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = 0, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-128

    - TTP, domain_name = cube_2D, pre_refinements = 0,-1, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = 0,-1, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-128

    - TTP, domain_name = cube_2D, pre_refinements = 0,-1,-2, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = 0,-1,-2, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-128

    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4, dt = 0.05, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4, dt = 0.2, n_time_ranks = 1-128

    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4,2, dt = 0.05, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4,2, dt = 0.2, n_time_ranks = 1-128

    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,3, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,3, dt = 0.05, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,3, dt = 0.1, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,3, dt = 0.2, n_time_ranks = 1-128

    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,3,1, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,3,1, dt = 0.05, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,3,1, dt = 0.1, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,3,1, dt = 0.2, n_time_ranks = 1-32  (more than 32 time ranks takes more than 100 iterations)

    - TTP, domain_name = cube_2D, pre_refinements = -1,-2, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1,-2, num nodes = 6,4, dt = 0.05, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1,-2, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-128
    - TTP, domain_name = cube_2D, pre_refinements = -1,-2, num nodes = 6,4, dt = 0.2, n_time_ranks = 1-128

    - TTP, domain_name = cube_2D, pre_refinements = -1,_2, num nodes = 6,3, dt = 0.025, n_time_ranks = 1-32 (more than 32 time ranks takes too many iterations)    
    - TTP, domain_name = cube_2D, pre_refinements = -1,_2, num nodes = 6,3, dt = 0.05, n_time_ranks = 1-32
    - TTP, domain_name = cube_2D, pre_refinements = -1,_2, num nodes = 6,3, dt = 0.1, n_time_ranks = 1-8

    - TTP, domain_name = cube_2D, pre_refinements = -1,-2,-3, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-32
    - TTP, domain_name = cube_2D, pre_refinements = -1,-2,-3, num nodes = 6,4,2, dt = 0.05, n_time_ranks = 1-32
    - TTP, domain_name = cube_2D, pre_refinements = -1,-2,-3, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-32
    - TTP, domain_name = cube_2D, pre_refinements = -1,-2,-3, num nodes = 6,4,2, dt = 0.2, n_time_ranks = 1-32

    - TTP, domain_name = cube_2D, pre_refinements = -1,_2,_3, num nodes = 6,3,1, dt = 0.025, n_time_ranks = 1-32
    - TTP, domain_name = cube_2D, pre_refinements = -1,_2,_3, num nodes = 6,3,1, dt = 0.05, n_time_ranks = 1-32
    - TTP, domain_name = cube_2D, pre_refinements = -1,_2,_3, num nodes = 6,3,1, dt = 0.1, n_time_ranks = 1-8

    - CRN, domain_name = cube_2D, pre_refinements = 0, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-128    
    - CRN, domain_name = cube_2D, pre_refinements = 0, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-128

    - CRN, domain_name = cube_2D, pre_refinements = 0, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-128    
    - CRN, domain_name = cube_2D, pre_refinements = 0, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-128

    - CRN, domain_name = cube_2D, pre_refinements = 0,-1, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-128    
    - CRN, domain_name = cube_2D, pre_refinements = 0,-1, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-128

    - CRN, domain_name = cube_2D, pre_refinements = 0,-1,-2, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-128    
    - CRN, domain_name = cube_2D, pre_refinements = 0,-1,-2, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-128    

    - CRN, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-128    
    - CRN, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-128

    - CRN, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-128    
    - CRN, domain_name = cube_2D, pre_refinements = -1, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-128

    - CRN, domain_name = cube_2D, pre_refinements = -1,-2, num nodes = 6,4, dt = 0.025, n_time_ranks = 1-128    
    - CRN, domain_name = cube_2D, pre_refinements = -1,-2, num nodes = 6,4, dt = 0.1, n_time_ranks = 1-128  

    - CRN, domain_name = cube_2D, pre_refinements = -1,-2,-3, num nodes = 6,4,2, dt = 0.025, n_time_ranks = 1-128    
    - CRN, domain_name = cube_2D, pre_refinements = -1,-2,-3, num nodes = 6,4,2, dt = 0.1, n_time_ranks = 1-128   
        

Running experiments:
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2, num nodes = 6,3, dt = 0.0250,0.05,0.1,0.2, n_time_ranks = 1-1024   
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 2,1, num nodes = 6,3, dt = 0.0250,0.05,0.1,0.2, n_time_ranks = 1-1024   
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 3, num nodes = 6,3, dt = 0.0250,0.05,0.1,0.2, n_time_ranks = 1-1024   
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 3,2, num nodes = 6,3, dt = 0.0250,0.05,0.1,0.2, n_time_ranks = 1-1024   
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 4, num nodes = 6,3, dt = 0.0250,0.05,0.1,0.2, n_time_ranks = 1-1024   
    - TTP, domain_name = cuboid_1D_very_large, pre_refinements = 4,3, num nodes = 6,3, dt = 0.0250,0.05,0.1,0.2, n_time_ranks = 1-1024   
    
"""


def main():
    local = False
    list_n_time_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    slurm_options = dict()
    slurm_options["cluster"] = "eiger"

    # define sweeper parameters
    # integrator = "IMEXEXP"
    options = dict()
    options["integrator"] = "IMEXEXP_EXPRK"
    options["num_nodes"] = [6, 3, 1]
    options["num_sweeps"] = [1]

    # set step parameters
    options["max_iter"] = 100

    # set space discretization
    options["space_disc"] = "DCT"

    # set level parameters
    # options["dt"] = 0.2
    dt_list = [0.025, 0.05, 0.1, 0.2]
    options["restol"] = 5e-8

    # set time parallelism to True or emulated (False)
    options["truly_time_parallel"] = True

    # set problem parameters
    options["domain_name"] = "cube_1D"
    options["pre_refinements"] = [4, 3, 2]
    options["order"] = 4
    options["lin_solv_max_iter"] = 1000000
    options["lin_solv_rtol"] = 1e-8
    ionic_models_list = ["HH"]
    # options["ionic_model_name"] = "BS"
    options["read_init_val"] = True
    options["init_time"] = 1000.0
    options["enable_output"] = False
    options["write_as_reference_solution"] = False
    options["output_root"] = "results_stability_DCT"
    options["mass_lumping"] = True
    options["mass_rhs"] = "none"

    options["print_stats"] = True

    n_space_ranks = 1

    slurm_options["dry_run"] = False
    slurm_options["overwrite_existing_results"] = False
    slurm_options["ntaskspernode"] = 72 if slurm_options["cluster"] == "eiger" else 12
    minutes = 30
    hours = 0
    slurm_options["run_time"] = 3600 * hours + 60 * minutes

    dependencies = False
    job_number = 0

    base_python_command = "python3 run_MonodomainODE_cli.py"
    local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_container_monodomain_new "

    for ionic_model_name in ionic_models_list:
        options["ionic_model_name"] = ionic_model_name
        for dt in dt_list:
            options["dt"] = dt
            for n_time_ranks in list_n_time_ranks:
                options["n_time_ranks"] = n_time_ranks
                options["end_time"] = options["dt"] * options["n_time_ranks"]
                num_nodes_str = "-".join([str(num_node) for num_node in options["num_nodes"]])
                pre_refinements_str = "-".join([str(pre_refinement) for pre_refinement in options["pre_refinements"]])
                options["output_file_name"] = (
                    "pre_refinements_" + pre_refinements_str + "_num_nodes_" + num_nodes_str + "_dt_" + str(options["dt"]).replace(".", "p") + "_n_time_ranks_" + str(n_time_ranks)
                )
                n_tasks = n_space_ranks * n_time_ranks
                if local:
                    os.system(local_docker_command + f"mpirun -n {n_tasks} " + base_python_command + " " + options_command(options))
                else:
                    slurm_options["n_tasks"] = n_tasks
                    merged_opts = options.copy()
                    merged_opts.update(slurm_options)
                    job_number = execute_with_dependencies(base_python_command, merged_opts, job_number, dependencies)


if __name__ == "__main__":
    main()
