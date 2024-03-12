import os
import numpy as np
from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE_cli_slurm_wrapper import execute_with_dependencies, options_command


def diff_dt():
    local = False
    max_dt = 1.0
    min_dt_pow = 2
    max_dt_pow = 7
    list_dts = max_dt * np.array([1.0 / 2**i for i in range(min_dt_pow, max_dt_pow + 1)])

    # define sweeper parameters
    options = dict()
    options["integrator"] = "IMEXEXP_EXPRK"
    options["num_nodes"] = [8, 2]
    options["num_sweeps"] = [1]

    # set step parameters
    options["max_iter"] = 100

    # set space discretization
    options["space_disc"] = "DCT"

    # set level parameters
    options["restol"] = 5e-8

    # set time parallelism to True or emulated (False)
    options["truly_time_parallel"] = False
    options["n_time_ranks"] = 1

    options["end_time"] = 1.0

    # set problem parameters
    options["domain_name"] = "cube_2D"
    options["pre_refinements"] = [0]
    options["order"] = 4
    options["lin_solv_max_iter"] = int(1e9)
    options["lin_solv_rtol"] = 1e-12
    ionic_models_names = ["HH", "CRN", "TTP"]
    options["read_init_val"] = True
    options["init_time"] = 2500.0
    options["enable_output"] = False
    options["write_as_reference_solution"] = False
    options["output_root"] = "results_iterations"
    options["mass_lumping"] = True
    options["mass_rhs"] = "none"
    options["skip_res"] = True

    options["print_stats"] = False

    slurm_options = dict()
    slurm_options["cluster"] = "eiger"
    slurm_options["dry_run"] = False
    slurm_options["overwrite_existing_results"] = False
    slurm_options["ntaskspernode"] = 32 if slurm_options["cluster"] == "eiger" else 12
    minutes = 0
    hours = 2
    slurm_options["run_time"] = 3600 * hours + 60 * minutes

    dependencies = False
    job_number = 0
    n_space_ranks = 12 if options["space_disc"] == "FEM" else 1
    n_time_ranks = options["n_time_ranks"]
    n_tasks = n_space_ranks * n_time_ranks

    base_python_command = "python3 run_MonodomainODE_cli.py"
    # local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_container_monodomain_new "
    local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_monodomain "

    for ionic_model in ionic_models_names:
        options["ionic_model_name"] = ionic_model
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
                if not slurm_options["dry_run"]:
                    os.system(local_docker_command + f"mpirun -n {n_tasks} " + base_python_command + " " + options_command(options))
            else:
                slurm_options["n_tasks"] = n_tasks
                merged_opts = options.copy()
                merged_opts.update(slurm_options)
                job_number = execute_with_dependencies(base_python_command, merged_opts, job_number, dependencies)


def diff_nodes():
    local = False
    min_nodes = 1
    max_nodes = 12

    # define sweeper parameters
    options = dict()
    options["integrator"] = "IMEXEXP_EXPRK"
    options["dt"] = 0.05
    list_nodes = list(range(min_nodes, max_nodes + 1))
    options["num_sweeps"] = [1]

    # set step parameters
    options["max_iter"] = 100

    # set space discretization
    options["space_disc"] = "DCT"

    # set level parameters
    options["restol"] = 5e-8

    # set time parallelism to True or emulated (False)
    options["truly_time_parallel"] = False
    options["n_time_ranks"] = 1

    options["end_time"] = 1.0

    # set problem parameters
    options["domain_name"] = "cube_2D"
    options["pre_refinements"] = [0]
    options["order"] = 4
    options["lin_solv_max_iter"] = int(1e9)
    options["lin_solv_rtol"] = 1e-12
    ionic_models_names = ["HH", "CRN", "TTP"]
    options["read_init_val"] = True
    options["init_time"] = 2500.0
    options["enable_output"] = False
    options["write_as_reference_solution"] = False
    options["output_root"] = "results_iterations"
    options["mass_lumping"] = True
    options["mass_rhs"] = "none"
    options["skip_res"] = True

    options["print_stats"] = False

    slurm_options = dict()
    slurm_options["cluster"] = "eiger"
    slurm_options["dry_run"] = False
    slurm_options["overwrite_existing_results"] = True
    slurm_options["ntaskspernode"] = 32 if slurm_options["cluster"] == "eiger" else 12
    minutes = 0
    hours = 2
    slurm_options["run_time"] = 3600 * hours + 60 * minutes

    dependencies = False
    job_number = 0
    n_space_ranks = 12 if options["space_disc"] == "FEM" else 1
    n_time_ranks = options["n_time_ranks"]
    n_tasks = n_space_ranks * n_time_ranks

    base_python_command = "python3 run_MonodomainODE_cli.py"
    # local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_container_monodomain_new "
    local_docker_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_monodomain "

    for ionic_model in ionic_models_names:
        options["ionic_model_name"] = ionic_model
        for num_nodes in list_nodes:
            options["num_nodes"] = [num_nodes]

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
                job_number = execute_with_dependencies(base_python_command, merged_opts, job_number, dependencies)


if __name__ == "__main__":
    # diff_dt()
    diff_nodes()
