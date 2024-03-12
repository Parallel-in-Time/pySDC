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
    max_dt = 1.0
    min_dt_pow = 1
    max_dt_pow = 6
    list_dts = max_dt * np.array([1.0 / 2**i for i in range(min_dt_pow, max_dt_pow + 1)])

    # define sweeper parameters
    options = dict()
    options["integrator"] = "IMEXEXP_EXPRK"
    # options["integrator"] = "IMEXEXP"
    options["num_nodes"] = [6, 3]
    options["num_sweeps"] = [1]

    # set step parameters
    options["max_iter"] = 100

    # set space discretization
    options["space_disc"] = "DCT"

    # set level parameters
    options["restol"] = 5e-14

    # set time parallelism to True or emulated (False)
    options["truly_time_parallel"] = True
    options["n_time_ranks"] = 64

    options["end_time"] = 16.0

    # set problem parameters
    options["domain_name"] = "cube_2D"
    options["pre_refinements"] = [-1]
    options["order"] = 4
    options["lin_solv_max_iter"] = int(1e9)
    options["lin_solv_rtol"] = 1e-12
    options["ionic_model_name"] = "TTP_SMOOTH"
    options["read_init_val"] = True
    options["init_time"] = 2500.0
    options["enable_output"] = False
    options["write_as_reference_solution"] = False
    options["output_root"] = "results_convergence_parallel"
    options["mass_lumping"] = True
    options["mass_rhs"] = "none"
    options["skip_res"] = False

    options["print_stats"] = False

    slurm_options = dict()
    slurm_options["cluster"] = "eiger"
    slurm_options["dry_run"] = False
    slurm_options["overwrite_existing_results"] = False
    slurm_options["ntaskspernode"] = 72 if slurm_options["cluster"] == "eiger" else 12
    minutes = 0
    hours = 4
    slurm_options["run_time"] = 3600 * hours + 60 * minutes

    dependencies = False
    job_number = 0
    ref_sol_job_number = 0
    n_space_ranks = 12 if options["space_disc"] == "FEM" else 1
    n_time_ranks = options["n_time_ranks"]
    n_tasks = n_space_ranks * n_time_ranks

    base_python_command = "python3 run_MonodomainODE_cli.py"
    # local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_container_monodomain_new "
    local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_monodomain "

    dt_ref_sol = list_dts[-1] / 2.0
    nodes_ref_sol = options["num_nodes"][0]
    pre_refinement_ref_sol = options["pre_refinements"][0]
    ref_sol_name = "ref_sol_pre_refinements_" + str(pre_refinement_ref_sol) + "_num_nodes_" + str(nodes_ref_sol) + "_dt_" + str(dt_ref_sol).replace(".", "p")

    options["ref_sol"] = ref_sol_name

    if compute_reference_solution and ref_sol_job_number == 0:
        options_ref_sol = options.copy()
        options_ref_sol["output_file_name"] = ref_sol_name
        options_ref_sol["dt"] = dt_ref_sol
        options_ref_sol["num_nodes"] = [nodes_ref_sol]
        options_ref_sol["pre_refinements"] = [pre_refinement_ref_sol]
        options_ref_sol["n_time_ranks"] = 1
        options_ref_sol["restol"] = options["restol"]
        options_ref_sol["lin_solv_rtol"] = options["lin_solv_rtol"]
        options_ref_sol["write_as_reference_solution"] = True
        options_ref_sol["max_iter"] = options_ref_sol["num_nodes"][0] + 0

        if local:
            os.system(local_docker_command + f"mpirun -n {n_tasks} " + base_python_command + " " + options_command(options_ref_sol))
        else:
            slurm_options_ref_sol = slurm_options.copy()
            slurm_options_ref_sol["n_tasks"] = n_space_ranks
            merged_opts = options_ref_sol.copy()
            merged_opts.update(slurm_options_ref_sol)
            ref_sol_job_number = execute_with_dependencies(base_python_command, merged_opts, job_number, dependencies)
            if dependencies:
                job_number = ref_sol_job_number

    for dt in list_dts:
        options["dt"] = dt
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
            + str(n_time_ranks)
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


def dt_inv_prop_tasks():
    local = False
    compute_reference_solution = True
    max_dt = 1.0
    min_dt_pow = 1
    max_dt_pow = 6
    list_dts = max_dt * np.array([1.0 / 2**i for i in range(min_dt_pow, max_dt_pow + 1)])

    # define sweeper parameters
    options = dict()
    options["integrator"] = "IMEXEXP_EXPRK"
    # options["integrator"] = "IMEXEXP"
    options["num_nodes"] = [6, 3]
    options["num_sweeps"] = [1]

    # set step parameters
    options["max_iter"] = 100

    # set space discretization
    options["space_disc"] = "DCT"

    # set level parameters
    options["restol"] = 5e-14

    # set time parallelism to True or emulated (False)
    options["truly_time_parallel"] = True

    options["end_time"] = 16.0

    # set problem parameters
    options["domain_name"] = "cube_2D"
    options["pre_refinements"] = [-1]
    options["order"] = 4
    options["lin_solv_max_iter"] = int(1e9)
    options["lin_solv_rtol"] = 1e-12
    options["ionic_model_name"] = "TTP_SMOOTH"
    options["read_init_val"] = True
    options["init_time"] = 2500.0
    options["enable_output"] = False
    options["write_as_reference_solution"] = False
    options["output_root"] = "results_convergence_parallel"
    options["mass_lumping"] = True
    options["mass_rhs"] = "none"
    options["skip_res"] = False

    options["print_stats"] = False

    slurm_options = dict()
    slurm_options["cluster"] = "eiger"
    slurm_options["dry_run"] = False
    slurm_options["overwrite_existing_results"] = False
    slurm_options["ntaskspernode"] = 72 if slurm_options["cluster"] == "eiger" else 12
    minutes = 0
    hours = 4
    slurm_options["run_time"] = 3600 * hours + 60 * minutes

    dependencies = False
    job_number = 0
    ref_sol_job_number = 2917708
    n_space_ranks = 12 if options["space_disc"] == "FEM" else 1

    base_python_command = "python3 run_MonodomainODE_cli.py"
    # local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_container_monodomain_new "
    local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_monodomain "

    dt_ref_sol = list_dts[-1] / 4.0
    nodes_ref_sol = options["num_nodes"][0]
    pre_refinement_ref_sol = options["pre_refinements"][0]
    ref_sol_name = "ref_sol_pre_refinements_" + str(pre_refinement_ref_sol) + "_num_nodes_" + str(nodes_ref_sol) + "_dt_" + str(dt_ref_sol).replace(".", "p")

    options["ref_sol"] = ref_sol_name

    if compute_reference_solution and ref_sol_job_number == 0:
        options_ref_sol = options.copy()
        options_ref_sol["output_file_name"] = ref_sol_name
        options_ref_sol["dt"] = dt_ref_sol
        options_ref_sol["num_nodes"] = [nodes_ref_sol]
        options_ref_sol["pre_refinements"] = [pre_refinement_ref_sol]
        options_ref_sol["n_time_ranks"] = 1
        options_ref_sol["restol"] = options["restol"]
        options_ref_sol["lin_solv_rtol"] = options["lin_solv_rtol"]
        options_ref_sol["write_as_reference_solution"] = True
        options_ref_sol["max_iter"] = options_ref_sol["num_nodes"][0] + 0

        n_time_ranks = 1
        n_tasks = n_space_ranks * n_time_ranks

        if local:
            os.system(local_docker_command + f"mpirun -n {n_tasks} " + base_python_command + " " + options_command(options_ref_sol))
        else:
            slurm_options_ref_sol = slurm_options.copy()
            slurm_options_ref_sol["n_tasks"] = n_space_ranks
            merged_opts = options_ref_sol.copy()
            merged_opts.update(slurm_options_ref_sol)
            ref_sol_job_number = execute_with_dependencies(base_python_command, merged_opts, job_number, dependencies)
            if dependencies:
                job_number = ref_sol_job_number

    for dt in list_dts:
        options["dt"] = dt
        options["n_time_ranks"] = int(np.round(options["end_time"] / dt))
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
    # main()
    dt_inv_prop_tasks()
