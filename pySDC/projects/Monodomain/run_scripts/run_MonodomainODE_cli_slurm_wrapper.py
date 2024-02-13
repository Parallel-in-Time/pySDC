import os
import numpy as np
import subprocess
import re
import argparse


def list_of_ints(arg):
    arg = arg.replace(' ', '')
    arg = arg.replace('_', '-')
    arg = arg.split(',')
    return list(map(int, arg))


def options_command(options):
    cmd = ""
    for key, val in options.items():
        if type(val) is list:
            opt = key
            if type(val[0]) is int:
                arg = ",".join([str(v).replace("-", "_") for v in val])
            else:
                arg = ",".join([map(str, val)])
        elif type(val) is bool:
            if not val:
                opt = "no-" + key
            else:
                opt = key
            arg = ""
        else:
            opt = key
            arg = str(val)
        cmd = cmd + " --" + opt + (" " + arg if arg != "" else "")
    return cmd


def get_time_str(seconds):
    hours = int(np.floor(seconds / 3600.0))
    minutes = int(np.ceil(60 * (seconds / 3600.0 - hours)))
    hours_str = ("0" if hours < 10 else "") + str(hours)
    minutes_str = ("0" if minutes < 10 else "") + str(minutes)
    time_str = hours_str + ":" + minutes_str + ":" + "00"
    hours_exact = seconds / 3600.0
    return time_str


def execute_with_dependencies(base_python_command, options, job_number, dependencies):
    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    pre_ref = options["pre_refinements"][0] if type(options["pre_refinements"]) is list else options["pre_refinements"]
    base_dir = executed_file_dir + "/../../../../data/Monodomain/" + options["output_root"] + "/" + options["domain_name"] + "/" + "ref_" + str(pre_ref) + "/" + options["ionic_model_name"] + "/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    output_file = base_dir + options["output_file_name"] + ".db"
    file_exists = os.path.isfile(output_file)
    if options["overwrite_existing_results"] or not file_exists:
        ntasks = options["n_tasks"]
        ntaskspernode = options["ntaskspernode"]

        dry_run = options["dry_run"]
        MonodomainODE_options = options.copy()
        del MonodomainODE_options["n_tasks"]
        del MonodomainODE_options["ntaskspernode"]
        del MonodomainODE_options["dry_run"]
        del MonodomainODE_options["overwrite_existing_results"]
        del MonodomainODE_options["run_time"]
        del MonodomainODE_options["cluster"]

        account_name = "u0" if options["cluster"] == "eiger" else "s1074"
        constraint = "mc" if options["cluster"] == "eiger" else "gpu"
        time_str = get_time_str(options["run_time"])
        opts = options_command(MonodomainODE_options)
        print(f"Slurm options: ntasks = {ntasks}, ntaskspernode = {ntaskspernode}, time = {time_str}")
        print(f"Simulation options: {opts}")
        log_file = base_dir + options["output_file_name"] + ".log"
        dependency_str = "" if (job_number == 0 or not dependencies) else f"\n#SBATCH --dependency=afterany:{job_number}"
        prev_job_number = job_number
        script = f'#!/bin/bash -l\
                    \n#SBATCH --job-name="PFASST"\
                    \n#SBATCH --account="{account_name}"\
                    \n#SBATCH --time={time_str}\
                    \n#SBATCH --ntasks={ntasks}\
                    \n#SBATCH --ntasks-per-node={ntaskspernode}\
                    \n#SBATCH --output={log_file}\
                    \n#SBATCH --cpus-per-task=1\
                    \n#SBATCH --ntasks-per-core=1\
                    \n#SBATCH --constraint={constraint}\
                    \n#SBATCH --hint=nomultithread\
                    {dependency_str}\
                    \nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\
                    \nexport CRAY_CUDA_MPS=1\
                    \nMPI4PY_RC_RECV_MPROBE=0 srun {base_python_command+opts}\
                    \n'

        if not dry_run:
            with open("script.sh", "w") as file:
                file.write(script)
            res = subprocess.check_output("sbatch script.sh", shell=True)  # output similar to: Submitted batch job 48953310
            match = re.search(r"\d+", res.decode())
            if match:
                job_number = int(match.group())
            else:
                raise Exception("Could not find the job number")
            os.system("rm script.sh")
            print(f"Submitted batch job {job_number}, runtime estimated {time_str}" + (f", dependent on job {prev_job_number}" if prev_job_number != 0 and dependencies else ""))

        return job_number
    else:
        return job_number


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # define sweeper parameters
    parser.add_argument("--integrator", default="IMEXEXP_EXPRK", type=str, help="sweeper name")
    parser.add_argument("--num_nodes", default="5", type=list_of_ints, help="list of ints (as '5,3', i.e. no brakets): number of collocation nodes per level")
    parser.add_argument("--num_sweeps", default="1", type=list_of_ints, help="list of ints: number of sweeps per level")
    parser.add_argument("--mass_rhs", default="none", type=str, help="if rhs is already multiplied by mass matrix: none, one (only potential), all (also ionic model states)")
    # set step parameters
    parser.add_argument("--max_iter", default=100, type=int, help="maximal number of iterations")
    # set level parameters
    parser.add_argument("--dt", default=0.1, type=float, help="step size")
    parser.add_argument("--restol", default=5e-8, type=float, help="residual tolerance")
    # problem args
    parser.add_argument("--space_disc", default="FD", type=str, help="space discretization method: FEM, FD")
    parser.add_argument("--domain_name", default="cuboid_1D_very_large", type=str, help="cuboid_2D, cuboid_3D, truncated_ellipsoid,...")
    parser.add_argument("--pre_refinements", default="2", type=list_of_ints, help="list of ints: loads a mesh which has already been pre-refined pre_refinements times.")
    parser.add_argument("--order", default="4", type=list_of_ints, help="list of ints: order of FEM or FD discretization")
    parser.add_argument("--lin_solv_max_iter", default=1000, type=int, help="maximal number of iterations in iterative linear solver")
    parser.add_argument("--lin_solv_rtol", default=1e-8, type=float, help="residual tolerance in iterative linear solver")
    parser.add_argument("--ionic_model_name", default="TTP", type=str, help="ionic_model: HH, CRN, TTP")
    parser.add_argument("--read_init_val", default=False, action=argparse.BooleanOptionalAction, help="read the initial value from file")
    parser.add_argument("--init_time", default=0.0, type=float, help="duration of stimulus. -1 means default")
    parser.add_argument("--enable_output", default=False, action=argparse.BooleanOptionalAction, help="activate or deactivate xdmf output: True or False")
    parser.add_argument("--write_as_reference_solution", default=False, action=argparse.BooleanOptionalAction, help="write as reference solution: True or False")
    parser.add_argument("--end_time", default=1.0, type=float, help="end time. If negative, a default one is used")
    parser.add_argument("--output_file_name", default="monodomain", type=str, help="output file name")
    parser.add_argument("--ref_sol", default="ref_sol", type=str, help="reference solution file name")
    parser.add_argument("--output_root", default="results_tmp/", type=str, help="output root folder")
    parser.add_argument("--mass_lumping", default=True, action=argparse.BooleanOptionalAction, help="with or without mass lumping")
    # controller args
    parser.add_argument("--truly_time_parallel", default=True, action=argparse.BooleanOptionalAction, help="truly time parallel or emulated")
    parser.add_argument("--n_time_ranks", default=1, type=int, help="number of time ranks")
    parser.add_argument("--print_stats", default=True, action=argparse.BooleanOptionalAction, help="print stats or not")

    # other
    parser.add_argument("--cluster", default="eiger", type=str, help="cluster name: eiger or daint")
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction, help="dry run or not")
    parser.add_argument("--overwrite_existing_results", default=True, action=argparse.BooleanOptionalAction, help="overwrite existing results or skip simulation if they exist")
    parser.add_argument("--n_tasks", default=1, type=int, help="number of mpi tasks")
    parser.add_argument("--ntaskspernode", default=72, type=int, help="number of mpi tasks per node")
    parser.add_argument("--run_time", default=600, type=int, help="estimated run time in seconds (for slurm)")
    args = vars(parser.parse_args())

    base_python_command = "python3 run_MonodomainODE_cli.py"
    # base_python_command = "docker exec -w /src/pySDC/pySDC/projects/Monodomain/run_scripts -it my_dolfinx_daint_container_monodomain_new mpirun -n 1 python3 run_MonodomainODE_cli.py"

    job_number = 0
    dependencies = False
    job_number = execute_with_dependencies(base_python_command, args, job_number, dependencies)


if __name__ == "__main__":
    main()
