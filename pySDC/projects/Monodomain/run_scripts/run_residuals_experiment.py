import os
import numpy as np
from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE_cli_slurm_wrapper import execute_with_dependencies, options_command

"""
Finished experiments:

Running experiments:
  
To be run:

"""


def main():
    local = False
    compute_reference_solution = True

    list_procs = [1, 1, 8, 256]
    list_nodes = [[8]] + [[8, 4]] * len(list_procs[1:])

    # list_procs = [1, 8, 256]
    # list_nodes = [[8, 4]] * len(list_procs)

    # define sweeper parameters
    options = dict()
    options["integrator"] = "IMEXEXP_EXPRK"
    options["num_sweeps"] = [1]

    options["dt"] = 0.025

    # set step parameters
    options["max_iter"] = 100

    # set space discretization
    options["space_disc"] = "DCT"

    # set level parameters
    options["restol"] = 1e-10

    # set time parallelism to True or emulated (False)
    options["truly_time_parallel"] = True

    options["end_time"] = options["dt"] * 256

    # set problem parameters
    options["domain_name"] = "cube_2D"
    options["pre_refinements"] = [0]
    options["order"] = 4
    options["lin_solv_max_iter"] = int(1e9)
    options["lin_solv_rtol"] = 1e-12
    options["ionic_model_name"] = "TTP"
    options["read_init_val"] = True
    options["init_time"] = 2500.0
    options["enable_output"] = False
    options["write_as_reference_solution"] = False
    options["output_root"] = "results_residuals"
    options["mass_lumping"] = True
    options["mass_rhs"] = "none"
    options["skip_res"] = True

    options["print_stats"] = False

    slurm_options = dict()
    slurm_options["cluster"] = "eiger"
    slurm_options["dry_run"] = False
    slurm_options["overwrite_existing_results"] = False
    slurm_options["ntaskspernode"] = 8 if slurm_options["cluster"] == "eiger" else 12
    minutes = 0
    hours = 12
    slurm_options["run_time"] = 3600 * hours + 60 * minutes

    dependencies = False
    job_number = 0
    ref_sol_job_number = 0
    n_space_ranks = 12 if options["space_disc"] == "FEM" else 1

    base_python_command = "python3 run_MonodomainODE_cli.py"
    local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_container_monodomain_new "
    # local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_monodomain "

    nodes_ref_sol = list_nodes[0][0]
    pre_refinement_ref_sol = options["pre_refinements"][0]
    ref_sol_name = "ref_sol_pre_refinements_" + str(pre_refinement_ref_sol) + "_num_nodes_" + str(nodes_ref_sol) + "_dt_" + str(options["dt"]).replace(".", "p")

    options["ref_sol"] = ref_sol_name

    if compute_reference_solution and ref_sol_job_number == 0:
        options_ref_sol = options.copy()
        options_ref_sol["output_file_name"] = ref_sol_name
        options_ref_sol["num_nodes"] = [nodes_ref_sol]
        options_ref_sol["pre_refinements"] = [pre_refinement_ref_sol]
        options_ref_sol["n_time_ranks"] = 1
        options_ref_sol["write_as_reference_solution"] = True
        n_tasks = n_space_ranks * options_ref_sol["n_time_ranks"]

        if local:
            os.system(local_docker_command + f"mpirun -n {n_tasks} " + base_python_command + " " + options_command(options_ref_sol))
        else:
            slurm_options_ref_sol = slurm_options.copy()
            slurm_options_ref_sol["n_tasks"] = n_tasks
            merged_opts = options_ref_sol.copy()
            merged_opts.update(slurm_options_ref_sol)
            ref_sol_job_number = execute_with_dependencies(base_python_command, merged_opts, job_number, dependencies)
            if dependencies:
                job_number = ref_sol_job_number

    for nodes, procs in zip(list_nodes, list_procs):
        options["num_nodes"] = nodes
        options["n_time_ranks"] = procs
        n_tasks = n_space_ranks * options["n_time_ranks"]

        num_nodes_str = "-".join([str(num_node) for num_node in options["num_nodes"]])
        pre_refinements_str = "-".join([str(pre_refinement) for pre_refinement in options["pre_refinements"]])
        options["output_file_name"] = (
            "pre_refinements_"
            + pre_refinements_str
            + "_num_nodes_"
            + num_nodes_str
            + "_max_iter_"
            + str(options["max_iter"])
            + "_dt_"
            + str(options["dt"]).replace(".", "p")
            + "_n_time_ranks_"
            + str(options["n_time_ranks"])
        )

        if local:
            print(local_docker_command + f"mpirun -n {n_tasks} " + base_python_command + " " + options_command(options))
            os.system(local_docker_command + f"mpirun -n {n_tasks} " + base_python_command + " " + options_command(options))
        else:
            slurm_options["n_tasks"] = n_tasks
            merged_opts = options.copy()
            merged_opts.update(slurm_options)
            job_number = execute_with_dependencies(
                base_python_command,
                merged_opts,
                ref_sol_job_number if (not dependencies and compute_reference_solution) else job_number,
                compute_reference_solution or dependencies,
            )


if __name__ == "__main__":
    main()
