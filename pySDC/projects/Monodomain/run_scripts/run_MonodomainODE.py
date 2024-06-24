from pathlib import Path
import numpy as np
from mpi4py import MPI
import logging
import os

from pySDC.core.errors import ParameterError

from pySDC.projects.Monodomain.problem_classes.MonodomainODE import MultiscaleMonodomainODE
from pySDC.projects.Monodomain.hooks.HookClass_pde import pde_hook
from pySDC.projects.Monodomain.hooks.HookClass_post_iter_info import post_iter_info_hook

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import (
    imexexp_1st_order as imexexp_1st_order_ExpRK,
)
from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.imexexp_1st_order import imexexp_1st_order

from pySDC.projects.Monodomain.transfer_classes.TransferVectorOfDCTVectors import TransferVectorOfDCTVectors

from pySDC.projects.Monodomain.utils.data_management import database


def set_logger(controller_params):
    logging.basicConfig(level=controller_params["logger_level"])
    hooks_logger = logging.getLogger("hooks")
    hooks_logger.setLevel(controller_params["logger_level"])


def get_controller(controller_params, description, time_comm, n_time_ranks, truly_time_parallel):
    if truly_time_parallel:
        controller = controller_MPI(controller_params=controller_params, description=description, comm=time_comm)
    else:
        controller = controller_nonMPI(
            num_procs=n_time_ranks, controller_params=controller_params, description=description
        )
    return controller


def print_dofs_stats(time_rank, controller, P, uinit):
    tot_dofs = uinit.size
    mesh_dofs = uinit.shape[1]
    if time_rank == 0:
        controller.logger.info(f"Total dofs: {tot_dofs}, mesh dofs = {mesh_dofs}")


def get_P_data(controller, truly_time_parallel):
    if truly_time_parallel:
        P = controller.S.levels[0].prob
    else:
        P = controller.MS[0].levels[0].prob
    # set time parameters
    t0 = P.t0
    Tend = P.Tend
    uinit = P.initial_value()
    return t0, Tend, uinit, P


def get_comms(n_time_ranks, truly_time_parallel):
    if truly_time_parallel:
        time_comm = MPI.COMM_WORLD
        time_rank = time_comm.Get_rank()
        assert time_comm.Get_size() == n_time_ranks, "Number of time ranks does not match the number of MPI ranks"
    else:
        time_comm = None
        time_rank = 0
    return time_comm, time_rank


def get_base_transfer_params(finter):
    base_transfer_params = dict()
    base_transfer_params["finter"] = finter
    return base_transfer_params


def get_controller_params(problem_params, n_time_ranks):
    controller_params = dict()
    controller_params["predict_type"] = "pfasst_burnin" if n_time_ranks > 1 else None
    controller_params["log_to_file"] = False
    controller_params["fname"] = problem_params["output_root"] + "controller"
    controller_params["logger_level"] = 20
    controller_params["dump_setup"] = False
    if n_time_ranks == 1:
        controller_params["hook_class"] = [post_iter_info_hook, pde_hook]
    else:
        controller_params["hook_class"] = [post_iter_info_hook]
    return controller_params


def get_description(
    integrator, problem_params, sweeper_params, level_params, step_params, base_transfer_params, space_transfer_class
):
    description = dict()

    problem = MultiscaleMonodomainODE

    if integrator == "IMEXEXP":
        # implicit-explicit-exponential integrators in the preconditioner and standard SDC
        description["sweeper_class"] = imexexp_1st_order
    elif integrator == "IMEXEXP_EXPRK":
        # implicit-explicit-exponential integrators in the preconditioner and exponential SDC
        description["sweeper_class"] = imexexp_1st_order_ExpRK
    else:
        raise ParameterError("Unknown integrator.")

    description["problem_class"] = problem
    description["problem_params"] = problem_params
    description["sweeper_params"] = sweeper_params
    description["level_params"] = level_params
    description["step_params"] = step_params
    description["base_transfer_params"] = base_transfer_params
    description["space_transfer_class"] = space_transfer_class

    return description


def get_step_params(maxiter):
    step_params = dict()
    step_params["maxiter"] = maxiter
    return step_params


def get_level_params(dt, nsweeps, restol, n_time_ranks):
    # initialize level parameters
    level_params = dict()
    level_params["restol"] = restol
    level_params["dt"] = dt
    level_params["nsweeps"] = nsweeps
    level_params["residual_type"] = "full_rel"
    level_params["parallel"] = n_time_ranks > 1

    return level_params


def get_sweeper_params(num_nodes, skip_residual_computation):
    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params["initial_guess"] = "spread"
    sweeper_params["quad_type"] = "RADAU-RIGHT"
    sweeper_params["num_nodes"] = num_nodes
    sweeper_params["QI"] = "IE"
    if skip_residual_computation:
        sweeper_params["skip_residual_computation"] = ("IT_FINE", "IT_COARSE", "IT_DOWN", "IT_UP")

    return sweeper_params


def get_space_tranfer_params():

    space_transfer_class = TransferVectorOfDCTVectors

    return space_transfer_class


def get_problem_params(
    domain_name,
    refinements,
    ionic_model_name,
    read_init_val,
    init_time,
    enable_output,
    end_time,
    order,
    output_root,
    output_file_name,
    ref_sol,
):
    # initialize problem parameters
    problem_params = dict()
    problem_params["order"] = order  # order of the spatial discretization
    problem_params["refinements"] = refinements  # number of refinements with respect to a baseline
    problem_params["domain_name"] = (
        domain_name  # name of the domain: cube_1D, cube_2D, cube_3D, cuboid_1D, cuboid_2D, cuboid_3D, cuboid_1D_small, cuboid_2D_small, cuboid_3D_small
    )
    problem_params["ionic_model_name"] = (
        ionic_model_name  # name of the ionic model: HH, CRN, TTP, TTP_SMOOTH for Hodgkin-Huxley, Courtemanche-Ramirez-Nattel, Ten Tusscher-Panfilov and a smoothed version of Ten Tusscher-Panfilov
    )
    problem_params["read_init_val"] = (
        read_init_val  # read the initial value from file (True) or initiate an action potential with a stimulus (False)
    )
    problem_params["init_time"] = (
        init_time  # stimulus happpens at t=0 and t=1000 and lasts 2ms. If init_time>2 nothing happens up to t=1000. If init_time>1002 nothing happens, never.
    )
    problem_params["init_val_name"] = "init_val_DCT"  # name of the file containing the initial value
    problem_params["enable_output"] = (
        enable_output  # activate or deactivate output (that can be visualized with visualization/show_monodomain_sol.py)
    )
    problem_params["output_V_only"] = (
        True  # output only the transmembrane potential (V) and not the ionic model variables
    )
    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    problem_params["output_root"] = (
        executed_file_dir + "/../../../../data/" + output_root
    )  # output root folder. A hierarchy of folders is created in this folder, as root/domain_name/ref_+str(refinements)/ionic_model_name. Initial values are put here
    problem_params["output_file_name"] = output_file_name
    problem_params["ref_sol"] = ref_sol  # reference solution file name
    problem_params["end_time"] = end_time
    Path(problem_params["output_root"]).mkdir(parents=True, exist_ok=True)

    return problem_params


def setup_and_run(
    integrator,
    num_nodes,
    skip_residual_computation,
    num_sweeps,
    max_iter,
    dt,
    restol,
    domain_name,
    refinements,
    order,
    ionic_model_name,
    read_init_val,
    init_time,
    enable_output,
    write_as_reference_solution,
    write_all_variables,
    output_root,
    output_file_name,
    ref_sol,
    end_time,
    truly_time_parallel,
    n_time_ranks,
    finter,
    write_database,
):

    # get time communicator
    time_comm, time_rank = get_comms(n_time_ranks, truly_time_parallel)

    # get time integration parameters
    # set maximum number of iterations in ESDC/MLESDC/PFASST
    step_params = get_step_params(maxiter=max_iter)
    # set number of collocation nodes in each level
    sweeper_params = get_sweeper_params(num_nodes=num_nodes, skip_residual_computation=skip_residual_computation)
    # set step size, number of sweeps per iteration, and residual tolerance for the stopping criterion
    level_params = get_level_params(
        dt=dt,
        nsweeps=num_sweeps,
        restol=restol,
        n_time_ranks=n_time_ranks,
    )

    # fix enable output to that only finest level has output
    n_levels = max(len(refinements), len(num_nodes))
    enable_output = [enable_output] + [False] * (n_levels - 1)
    # get problem parameters
    problem_params = get_problem_params(
        domain_name=domain_name,
        refinements=refinements,
        ionic_model_name=ionic_model_name,
        read_init_val=read_init_val,
        init_time=init_time,
        enable_output=enable_output,
        end_time=end_time,
        order=order,
        output_root=output_root,
        output_file_name=output_file_name,
        ref_sol=ref_sol,
    )

    space_transfer_class = get_space_tranfer_params()

    # get remaining prams
    base_transfer_params = get_base_transfer_params(finter)
    controller_params = get_controller_params(problem_params, n_time_ranks)
    description = get_description(
        integrator,
        problem_params,
        sweeper_params,
        level_params,
        step_params,
        base_transfer_params,
        space_transfer_class,
    )
    set_logger(controller_params)
    controller = get_controller(controller_params, description, time_comm, n_time_ranks, truly_time_parallel)

    # get PDE data
    t0, Tend, uinit, P = get_P_data(controller, truly_time_parallel)

    # print dofs stats
    print_dofs_stats(time_rank, controller, P, uinit)

    # run
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # write reference solution, to be used later for error computation
    if write_as_reference_solution:
        P.write_reference_solution(uend, write_all_variables)

    # compute errors, if a reference solution is found
    error_availabe, error_L2, rel_error_L2 = P.compute_errors(uend)

    # get some stats
    iter_counts = get_sorted(stats, type="niter", sortby="time")
    residuals = get_sorted(stats, type="residual_post_iteration", sortby="time")
    if time_comm is not None:
        iter_counts = time_comm.gather(iter_counts, root=0)
        residuals = time_comm.gather(residuals, root=0)
        if time_rank == 0:
            iter_counts = [item for sublist in iter_counts for item in sublist]
            residuals = [item for sublist in residuals for item in sublist]
        iter_counts = time_comm.bcast(iter_counts, root=0)
        residuals = time_comm.bcast(residuals, root=0)

    iter_counts.sort()
    times = [item[0] for item in iter_counts]
    niters = [item[1] for item in iter_counts]

    residuals.sort()
    residuals_new = [residuals[0][1]]
    t = residuals[0][0]
    for i in range(1, len(residuals)):
        if residuals[i][0] > t + dt / 2.0:
            residuals_new.append(residuals[i][1])
            t = residuals[i][0]
    residuals = residuals_new

    avg_niters = np.mean(niters)
    if time_rank == 0:
        controller.logger.info("Mean number of iterations: %4.2f" % avg_niters)
        controller.logger.info(
            "Std and var for number of iterations: %4.2f -- %4.2f" % (float(np.std(niters)), float(np.var(niters)))
        )

    if write_database and time_rank == 0:
        errors = dict()
        errors["error_L2"] = error_L2
        errors["rel_error_L2"] = rel_error_L2
        iters_info = dict()
        iters_info["avg_niters"] = avg_niters
        iters_info["times"] = times
        iters_info["niters"] = niters
        iters_info["residuals"] = residuals
        file_name = P.output_folder / Path(P.output_file_name)
        if file_name.with_suffix('.db').is_file():
            os.remove(file_name.with_suffix('.db'))
        data_man = database(file_name)
        data_man.write_dictionary("errors", errors)
        data_man.write_dictionary("iters_info", iters_info)

    return error_L2, rel_error_L2, avg_niters, times, niters, residuals


def main():
    # define sweeper parameters
    # integrator = "IMEXEXP"
    integrator = "IMEXEXP_EXPRK"
    num_nodes = [4]
    num_sweeps = [1]

    # set step parameters
    max_iter = 100

    # set level parameters
    dt = 0.05
    restol = 5e-8

    # set problem parameters
    domain_name = "cube_2D"
    refinements = [-1]
    order = 4  # 2 or 4
    ionic_model_name = "TTP"
    read_init_val = True
    init_time = 3.0
    enable_output = False
    write_as_reference_solution = False
    write_all_variables = False
    write_database = False
    end_time = 0.05
    output_root = "results_tmp"
    output_file_name = "ref_sol" if write_as_reference_solution else "monodomain"
    ref_sol = "ref_sol"
    skip_residual_computation = False

    finter = False

    # set time parallelism to True or emulated (False)
    truly_time_parallel = False
    n_time_ranks = 1

    error_L2, rel_error_L2, avg_niters, times, niters, residuals = setup_and_run(
        integrator,
        num_nodes,
        skip_residual_computation,
        num_sweeps,
        max_iter,
        dt,
        restol,
        domain_name,
        refinements,
        order,
        ionic_model_name,
        read_init_val,
        init_time,
        enable_output,
        write_as_reference_solution,
        write_all_variables,
        output_root,
        output_file_name,
        ref_sol,
        end_time,
        truly_time_parallel,
        n_time_ranks,
        finter,
        write_database,
    )


if __name__ == "__main__":
    main()
