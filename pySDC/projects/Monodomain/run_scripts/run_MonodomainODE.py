from pathlib import Path
import numpy as np
from mpi4py import MPI
import logging
import os

from pySDC.core.Errors import ParameterError

from pySDC.projects.Monodomain.problem_classes.MonodomainODE import MonodomainODE, MultiscaleMonodomainODE
from pySDC.projects.Monodomain.problem_classes.space_discretizazions.Parabolic_FEniCSx import Parabolic_FEniCSx
from pySDC.projects.Monodomain.problem_classes.space_discretizazions.Parabolic_FD import Parabolic_FD
from pySDC.projects.Monodomain.transfer_classes.TransferVectorOfFEniCSxVectors import TransferVectorOfFEniCSxVectors
from pySDC.projects.Monodomain.transfer_classes.TransferVectorOfFDVectors import TransferVectorOfFDVectors
from pySDC.projects.Monodomain.hooks.HookClass_pde_MPI import pde_hook as pde_hook_MPI
from pySDC.projects.Monodomain.hooks.HookClass_pde import pde_hook
from pySDC.projects.Monodomain.hooks.HookClass_post_iter_info import post_iter_info_hook

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import imexexp_1st_order as imexexp_1st_order_ExpRK
from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.imexexp_1st_order import imexexp_1st_order

# from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.exponential_multirate_explicit_stabilized import (
#     exponential_multirate_explicit_stabilized as exponential_multirate_explicit_stabilized_ExpRK,
# )

# from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.multirate_explicit_stabilized import multirate_explicit_stabilized
# from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.exponential_multirate_explicit_stabilized import exponential_multirate_explicit_stabilized
# from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.explicit_stabilized import explicit_stabilized


def set_logger(controller_params):
    logging.basicConfig(level=controller_params["logger_level"])
    hooks_logger = logging.getLogger("hooks")
    hooks_logger.setLevel(controller_params["logger_level"])


def get_controller(controller_params, description, time_comm, n_time_ranks, truly_time_parallel):
    if truly_time_parallel:
        controller = controller_MPI(controller_params=controller_params, description=description, comm=time_comm)
    else:
        controller = controller_nonMPI(num_procs=n_time_ranks, controller_params=controller_params, description=description)
    return controller


def print_statistics(stats, controller, problem_params, space_rank, time_comm, n_time_ranks):
    iter_counts = get_sorted(stats, type="niter", sortby="time")
    niters = [item[1] for item in iter_counts]
    timing = get_sorted(stats, type="timing_run", sortby="time")
    timing = timing[0][1]
    if time_comm is not None:
        niters = time_comm.gather(niters, root=0)
        timing = time_comm.gather(timing, root=0)
    if time_comm is None or time_comm.rank == 0:
        niters = np.array(niters).flatten()
        controller.logger.info("Mean number of iterations: %4.2f" % np.mean(niters))
        controller.logger.info("Std and var for number of iterations: %4.2f -- %4.2f" % (float(np.std(niters)), float(np.var(niters))))
        timing = np.mean(np.array(timing))
        controller.logger.info(f"Time to solution: {timing:6.4f} sec.")

    from pySDC.projects.Monodomain.utils.visualization_tools import show_residual_across_simulation

    fname = problem_params["output_root"] + f"residuals_{n_time_ranks}_time_ranks.png"
    if space_rank == 0:
        show_residual_across_simulation(stats=stats, fname=fname, comm=time_comm)


def print_dofs_stats(space_rank, time_rank, controller, P, uinit):
    data, avg_data = P.parabolic.get_dofs_stats()
    if space_rank == 0 and time_rank == 0:
        controller.logger.info(f"Total dofs: {uinit.getSize()}, mesh dofs = {uinit[0].getSize()}")
        for i, d in enumerate(data):
            controller.logger.info(f"Processor {i}: tot_mesh_dofs = {d[0]:.2e}, n_loc_mesh_dofs = {d[1]:.2e}, n_mesh_ghost_dofs = {d[2]:.2e}, %ghost = {100*d[3]:.2f}")


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
        # set MPI communicator
        comm = MPI.COMM_WORLD
        world_rank = comm.Get_rank()
        world_size = comm.Get_size()
        assert world_size % n_time_ranks == 0, "Total number of ranks must be a multiple of n_time_ranks"
        n_space_ranks = int(world_size / n_time_ranks)
        # split world communicator to create space-communicators
        color = int(world_rank / n_space_ranks)
        space_comm = comm.Split(color=color)
        space_rank = space_comm.Get_rank()
        # split world communicator to create time-communicators
        color = int(world_rank % n_space_ranks)
        time_comm = comm.Split(color=color)
        time_rank = time_comm.Get_rank()
    else:
        space_comm = MPI.COMM_WORLD
        space_rank = space_comm.Get_rank()
        time_comm = None
        time_rank = 0
    return space_comm, time_comm, space_rank, time_rank


def get_base_transfer_params():
    base_transfer_params = dict()
    base_transfer_params["finter"] = False
    return base_transfer_params


def get_controller_params(problem_params, space_rank, truly_time_parallel):
    controller_params = dict()
    controller_params["predict_type"] = "pfasst_burnin"
    controller_params["log_to_file"] = False
    controller_params["fname"] = problem_params["output_root"] + "controller"
    controller_params["logger_level"] = 20 if space_rank == 0 else 99  # set level depending on rank
    controller_params["dump_setup"] = False
    if truly_time_parallel:
        controller_params["hook_class"] = [pde_hook_MPI, post_iter_info_hook]
    else:
        controller_params["hook_class"] = [pde_hook, post_iter_info_hook]
    return controller_params


def get_description(integrator, problem_params, sweeper_params, level_params, step_params, base_transfer_params, space_transfer_class, space_transfer_params):
    description = dict()

    if integrator != "ES":
        problem = MultiscaleMonodomainODE
    else:
        problem = MonodomainODE

    if integrator != 'mES':
        problem_params["splitting"] = "exp_nonstiff"
    else:
        problem_params["splitting"] = "stiff_nonstiff"

    if integrator == "IMEXEXP":
        description["sweeper_class"] = imexexp_1st_order
    elif integrator == "IMEXEXP_EXPRK":
        description["sweeper_class"] = imexexp_1st_order_ExpRK
    # elif integrator == "ES":
    #     description["sweeper_class"] = explicit_stabilized
    # elif integrator == "mES":
    #     description["sweeper_class"] = multirate_explicit_stabilized
    # elif integrator == "exp_mES":
    #     description["sweeper_class"] = exponential_multirate_explicit_stabilized
    # elif integrator == "exp_mES_EXPRK":
    #     description["sweeper_class"] = exponential_multirate_explicit_stabilized_ExpRK
    else:
        raise ParameterError("Unknown integrator.")

    description["problem_class"] = problem
    description["problem_params"] = problem_params
    description["sweeper_params"] = sweeper_params
    description["level_params"] = level_params
    description["step_params"] = step_params
    description["base_transfer_params"] = base_transfer_params
    description["space_transfer_class"] = space_transfer_class
    description["space_transfer_params"] = space_transfer_params
    return description


def get_step_params(maxiter):
    step_params = dict()
    step_params["maxiter"] = maxiter
    return step_params


def get_level_params(dt, nsweeps, restol):
    # initialize level parameters
    level_params = dict()
    level_params["restol"] = restol
    level_params["dt"] = dt
    level_params["nsweeps"] = nsweeps
    level_params["residual_type"] = "full_rel"
    return level_params


def get_sweeper_params(num_nodes):
    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params["initial_guess"] = "spread"
    sweeper_params["quad_type"] = "RADAU-RIGHT"
    sweeper_params["num_nodes"] = num_nodes
    sweeper_params["QI"] = "IE"
    # specific for explicit stabilized methods
    # sweeper_params["es_class"] = "RKW1"
    # sweeper_params["es_class_outer"] = "RKW1"
    # sweeper_params["es_class_inner"] = "RKW1"
    # # sweeper_params['es_s_outer'] = 0 # if given, or not zero, then the algorithm fixes s of the outer stabilized scheme to this value.
    # # sweeper_params['es_s_inner'] = 0
    # # sweeper_params['res_comp'] = 'f_eta'
    # sweeper_params["damping"] = 0.05
    # sweeper_params["safe_add"] = 0
    # sweeper_params["rho_freq"] = 100
    return sweeper_params


def get_space_tranfer_params(problem_params, iorder, rorder):
    if problem_params['parabolic_class'] is Parabolic_FEniCSx:
        space_transfer_class = TransferVectorOfFEniCSxVectors
        space_transfer_params = dict()
    elif problem_params['parabolic_class'] is Parabolic_FD:
        space_transfer_class = TransferVectorOfFDVectors
        space_transfer_params = dict()
        space_transfer_params["iorder"] = iorder
        space_transfer_params["rorder"] = rorder
        space_transfer_params["periodic"] = False
        space_transfer_params["equidist_nested"] = False

    return space_transfer_class, space_transfer_params


def get_problem_params(space_comm, domain_name, parabolic_class, pre_refinements, ionic_model_name, read_init_val, enable_output, end_time):
    # initialize problem parameters
    problem_params = dict()
    problem_params["comm"] = space_comm
    problem_params["family"] = "CG"
    if problem_params["family"] == "CG":
        problem_params["order"] = 1
        problem_params["mass_lumping"] = True  # has effect for family=CG and order=1
    elif problem_params["family"] == "DG":
        problem_params["order"] = [2, 1]
        problem_params["mass_lumping"] = False
    problem_params["pre_refinements"] = pre_refinements
    problem_params["post_refinements"] = [0]
    problem_params["fibrosis"] = False
    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    problem_params["meshes_fibers_root_folder"] = executed_file_dir + "/../../../../../meshes_fibers_fibrosis/results"
    problem_params["domain_name"] = domain_name
    problem_params["solver_rtol"] = 1e-8
    problem_params["parabolic_class"] = parabolic_class
    problem_params["ionic_model_name"] = ionic_model_name
    problem_params["read_init_val"] = read_init_val
    problem_params["init_val_name"] = "init_val"
    problem_params["istim_dur"] = 0.0 if problem_params["read_init_val"] else -1.0
    problem_params["enable_output"] = enable_output
    problem_params["output_V_only"] = True
    problem_params["output_root"] = executed_file_dir + "/../../../../data/Monodomain/results_tmp"
    problem_params["output_file_name"] = "monodomain"
    problem_params["ref_sol"] = "ref_sol"
    problem_params["end_time"] = end_time
    Path(problem_params["output_root"]).mkdir(parents=True, exist_ok=True)
    return problem_params


def main():
    # define integration method
    # integrators = ['ES']
    # integrators = ['mES']
    # integrators = ["exp_mES"]
    # integrators = ["exp_mES_EXPRK"]

    # integrator = "IMEXEXP"
    integrator = "IMEXEXP_EXPRK"

    space_disc = 'FD'

    # set time parallelism to True or emulated (False)
    truly_time_parallel = True
    # number of time ranks. If truly_parallel, space ranks chosen accoding to world_size/n_time_ranks, else space_ranks = world_size
    n_time_ranks = 4

    # get space-time communicators
    space_comm, time_comm, space_rank, time_rank = get_comms(n_time_ranks, truly_time_parallel)
    # get time integration parameters
    # set maximum number of iterations in SDC/ESDC/MLSDC/etc
    step_params = get_step_params(maxiter=50)
    # set number of collocation nodes in each level
    sweeper_params = get_sweeper_params(num_nodes=[4,2,1])
    # set step size, number of sweeps per iteration, and residual tolerance for the stopping criterion
    level_params = get_level_params(
        dt=0.025,
        nsweeps=[1],
        restol=5e-8,
    )
    # get problem parameters
    problem_params = get_problem_params(
        space_comm=space_comm,
        domain_name="cuboid_2D_small",
        parabolic_class=Parabolic_FEniCSx if space_disc == 'FEM' else Parabolic_FD,
        pre_refinements=[3,2,1],
        ionic_model_name="TTP",
        read_init_val=True,
        enable_output=False,
        end_time=0.1,
    )

    space_transfer_class, space_transfer_params = get_space_tranfer_params(
        problem_params,
        iorder=6,
        rorder=2,
    )

    # Usually do not modify below this line ------------------
    # get remaining prams
    base_transfer_params = get_base_transfer_params()
    controller_params = get_controller_params(problem_params, space_rank, truly_time_parallel)
    description = get_description(integrator, problem_params, sweeper_params, level_params, step_params, base_transfer_params, space_transfer_class, space_transfer_params)
    set_logger(controller_params)
    controller = get_controller(controller_params, description, time_comm, n_time_ranks, truly_time_parallel)

    # get PDE data
    t0, Tend, uinit, P = get_P_data(controller, truly_time_parallel)
    # print dofs stats (dofs per processor, etc.)
    print_dofs_stats(space_rank, time_rank, controller, P, uinit)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute errors, if a reference solution is available
    error_availabe, error_L2, rel_error_L2 = P.compute_errors(uend)

    # print statistics (number of iterations, time to solution, etc.)
    print_statistics(stats, controller, problem_params, space_rank, time_comm, n_time_ranks)

    # free communicators
    if truly_time_parallel:
        space_comm.Free()
        time_comm.Free()


if __name__ == "__main__":
    main()
