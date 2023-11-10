from pathlib import Path
import numpy as np
from mpi4py import MPI
import logging

from pySDC.core.Errors import ParameterError

from pySDC.projects.Monodomain.problem_classes.MonodomainSystem_FEniCSx_vec import monodomain_system, monodomain_system_exp_expl_impl
from pySDC.projects.Monodomain.transfer_classes.TransferFenicsxMeshVec import mesh_to_mesh_fenicsx
from pySDC.projects.Monodomain.hooks.HookClass_pde_MPI import pde_hook
from pySDC.projects.Monodomain.hooks.HookClass_post_iter_info import post_iter_info_hook

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import imexexp_1st_order as imexexp_1st_order_ExpRK
from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.exponential_multirate_explicit_stabilized import (
    exponential_multirate_explicit_stabilized as exponential_multirate_explicit_stabilized_ExpRK,
)

from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.imexexp_1st_order import imexexp_1st_order
from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.multirate_explicit_stabilized import multirate_explicit_stabilized
from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.exponential_multirate_explicit_stabilized import exponential_multirate_explicit_stabilized
from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.explicit_stabilized import explicit_stabilized


def main():
    # define integration methods
    # integrators = ['ES']
    # integrators = ["IMEXEXP"]
    # integrators = ['mES']
    # integrators = ["exp_mES"]

    integrators = ["IMEXEXP_EXPRK"]
    # integrators = ["exp_mES_EXPRK"]

    n_time_ranks = 4  # number of time ranks. Space ranks chosen accoding to world_size/n_time_ranks

    # cuboid_2D with 24 procs in time, 2 in space: time to sol 186.4860 sec.
    # the same but with 1 proc in time took 736.8081 sec

    ref = 2

    # initialize level parameters
    level_params = dict()
    level_params["restol"] = 5e-8
    level_params["dt"] = 0.1 / 2**ref
    level_params["nsweeps"] = [1]
    level_params["residual_type"] = "full_rel"

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params["initial_guess"] = "spread"
    sweeper_params["quad_type"] = "RADAU-RIGHT"
    sweeper_params["num_nodes"] = [4, 2, 1]
    sweeper_params["QI"] = "IE"
    # specific for explicit stabilized methods
    sweeper_params["es_class"] = "RKW1"
    sweeper_params["es_class_outer"] = "RKW1"
    sweeper_params["es_class_inner"] = "RKW1"
    # sweeper_params['es_s_outer'] = 0 # if given, or not zero, then the algorithm fixes s of the outer stabilized scheme to this value.
    # sweeper_params['es_s_inner'] = 0
    # sweeper_params['res_comp'] = 'f_eta'
    sweeper_params["damping"] = 0.05
    sweeper_params["safe_add"] = 0
    sweeper_params["rho_freq"] = 100

    # initialize step parameters
    step_params = dict()
    step_params["maxiter"] = 50

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
    time_size = time_comm.Get_size()

    # initialize problem parameters
    problem_params = dict()
    problem_params["communicator"] = space_comm
    problem_params["family"] = "CG"
    if problem_params["family"] == "CG":
        problem_params["order"] = [1]
        problem_params["mass_lumping"] = True  # has effect for family=CG and order=1
    elif problem_params["family"] == "DG":
        problem_params["order"] = [2, 1]
        problem_params["mass_lumping"] = False
    problem_params["domain_name"] = "cuboid_2D_small"
    problem_params["refinements"] = [2, 1, 0]
    problem_params["ionic_model"] = "HH"
    problem_params["ionic_model_eval"] = "c++"
    problem_params["fibrosis"] = False
    problem_params["meshes_fibers_root_folder"] = "../../../../../meshes_fibers_fibrosis/results"
    problem_params["output_root"] = "../../../../data/ExplicitStabilized/results_tmp"
    problem_params["output_file_name"] = "monodomain"
    problem_params["enable_output"] = False
    problem_params["output_V_only"] = True
    problem_params["ref_sol"] = "ref_sol"
    problem_params["solver_rtol"] = 1e-8
    problem_params["read_init_val"] = True
    problem_params["init_val_name"] = "init_val"
    problem_params["istim_dur"] = 0.0 if problem_params["read_init_val"] else -1.0

    # base transfer parameters
    base_transfer_params = dict()
    base_transfer_params["finter"] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params["predict_type"] = "pfasst_burnin"
    controller_params["log_to_file"] = False
    controller_params["fname"] = problem_params["output_root"] + "controller"
    controller_params["logger_level"] = 20 if space_rank == 0 else 99  # set level depending on rank
    controller_params["dump_setup"] = False
    controller_params["hook_class"] = [pde_hook, post_iter_info_hook]

    logging.basicConfig(level=controller_params["logger_level"])
    hooks_logger = logging.getLogger("hooks")
    hooks_logger.setLevel(controller_params["logger_level"])

    Path(problem_params["output_root"]).mkdir(parents=True, exist_ok=True)

    for integrator in integrators:
        description = dict()
        if integrator == "IMEXEXP":
            problem = monodomain_system_exp_expl_impl
            problem_params["splitting"] = "exp_nonstiff"
            description["sweeper_class"] = imexexp_1st_order
        elif integrator == "IMEXEXP_EXPRK":
            problem = monodomain_system_exp_expl_impl
            problem_params["splitting"] = "exp_nonstiff"
            description["sweeper_class"] = imexexp_1st_order_ExpRK
        elif integrator == "ES":
            problem = monodomain_system
            description["sweeper_class"] = explicit_stabilized
        elif integrator == "mES":
            problem_params["splitting"] = "stiff_nonstiff"
            problem = monodomain_system_exp_expl_impl
            description["sweeper_class"] = multirate_explicit_stabilized
        elif integrator == "exp_mES":
            problem_params["splitting"] = "exp_nonstiff"
            problem = monodomain_system_exp_expl_impl
            description["sweeper_class"] = exponential_multirate_explicit_stabilized
        elif integrator == "exp_mES_EXPRK":
            problem_params["splitting"] = "exp_nonstiff"
            problem = monodomain_system_exp_expl_impl
            description["sweeper_class"] = exponential_multirate_explicit_stabilized_ExpRK
        else:
            raise ParameterError("Unknown integrator.")

        description["problem_class"] = problem
        description["problem_params"] = problem_params
        description["sweeper_params"] = sweeper_params
        description["level_params"] = level_params
        description["step_params"] = step_params
        description["base_transfer_params"] = base_transfer_params
        description["space_transfer_class"] = mesh_to_mesh_fenicsx

        # instantiate the controller
        controller = controller_MPI(controller_params=controller_params, description=description, comm=time_comm)

        # get initial values on finest level
        P = controller.S.levels[0].prob
        # set time parameters
        t0 = P.t0
        Tend = P.Tend
        uinit = P.initial_value()

        data = (uinit.n_loc_dofs + uinit.n_ghost_dofs, uinit.n_loc_dofs, uinit.n_ghost_dofs, uinit.n_ghost_dofs / (uinit.n_loc_dofs + uinit.n_loc_dofs))
        data = P.domain.comm.gather(data, root=0)
        if space_rank == 0 and time_rank == 0:
            controller.logger.info(f"Total dofs: {uinit.getSize()}")
            for i, d in enumerate(data):
                controller.logger.info(f"Processor {i}: tot_dofs = {d[0]:.2e}, n_loc_dofs = {d[1]:.2e}, n_ghost_dofs = {d[2]:.2e}, %ghost = {100*d[3]:.2f}")

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        error_availabe, error_L2, rel_error_L2 = P.compute_errors(uend)

        # filter statistics by type (number of iterations)
        iter_counts = get_sorted(stats, type="niter", sortby="time")

        niters = np.array([item[1] for item in iter_counts])
        out = "Mean number of iterations: %4.2f" % np.mean(niters)
        controller.logger.info(out)
        out = "Std and var for number of iterations: %4.2f -- %4.2f" % (float(np.std(niters)), float(np.var(niters)))
        controller.logger.info(out)

        timing = get_sorted(stats, type="timing_run", sortby="time")
        out = f"Time to solution: {timing[0][1]:6.4f} sec."
        controller.logger.info(out)

        from pySDC.projects.ExplicitStabilized.utils.visualization_tools import show_residual_across_simulation

        fname = problem_params["output_root"] + f"residuals_{time_size}_time_ranks.png"
        if space_rank == 0:
            show_residual_across_simulation(stats=stats, fname=fname, comm=time_comm)

    space_comm.Free()
    time_comm.Free()


if __name__ == "__main__":
    main()
