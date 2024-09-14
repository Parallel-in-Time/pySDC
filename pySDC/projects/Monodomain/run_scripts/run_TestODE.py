from pathlib import Path
import numpy as np
import logging
import os

from tqdm import tqdm

from pySDC.core.errors import ParameterError

from pySDC.projects.Monodomain.problem_classes.TestODE import MultiscaleTestODE
from pySDC.projects.Monodomain.transfer_classes.TransferVectorOfDCTVectors import TransferVectorOfDCTVectors

from pySDC.projects.Monodomain.hooks.HookClass_post_iter_info import post_iter_info_hook

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import (
    imexexp_1st_order as imexexp_1st_order_ExpRK,
)
from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.imexexp_1st_order import imexexp_1st_order

"""
Run the multirate Dahlquist test equation and plot the stability domain of the method.
We vary only the exponential term and the stiff term, while the non stiff term is kept constant (to allow 2D plots).
"""


def set_logger(controller_params):
    logging.basicConfig(level=controller_params["logger_level"])
    hooks_logger = logging.getLogger("hooks")
    hooks_logger.setLevel(controller_params["logger_level"])


def get_controller(controller_params, description, n_time_ranks):
    controller = controller_nonMPI(num_procs=n_time_ranks, controller_params=controller_params, description=description)
    return controller


def get_P_data(controller):
    P = controller.MS[0].levels[0].prob
    # set time parameters
    t0 = P.t0
    Tend = P.Tend
    uinit = P.initial_value()
    return t0, Tend, uinit, P


def get_base_transfer_params():
    base_transfer_params = dict()
    base_transfer_params["finter"] = False
    return base_transfer_params


def get_controller_params(output_root, logger_level):
    controller_params = dict()
    controller_params["predict_type"] = "pfasst_burnin"
    controller_params["log_to_file"] = False
    controller_params["fname"] = output_root + "controller"
    controller_params["logger_level"] = logger_level
    controller_params["dump_setup"] = False
    controller_params["hook_class"] = [post_iter_info_hook]
    return controller_params


def get_description(
    integrator, problem_params, sweeper_params, level_params, step_params, base_transfer_params, space_transfer_class
):
    description = dict()

    problem = MultiscaleTestODE

    if integrator == "IMEXEXP":
        description["sweeper_class"] = imexexp_1st_order
    elif integrator == "IMEXEXP_EXPRK":
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

    return sweeper_params


def get_output_root():
    executed_file_dir = os.path.dirname(os.path.realpath(__file__))
    output_root = executed_file_dir + "/../../../../data/Monodomain/results_tmp"
    return output_root


def get_problem_params(lmbda_laplacian, lmbda_gating, lmbda_others, end_time):
    # initialize problem parameters
    problem_params = dict()
    problem_params["output_file_name"] = "monodomain"
    problem_params["output_root"] = get_output_root()
    problem_params["end_time"] = end_time
    problem_params["lmbda_laplacian"] = lmbda_laplacian
    problem_params["lmbda_gating"] = lmbda_gating
    problem_params["lmbda_others"] = lmbda_others
    Path(problem_params["output_root"]).mkdir(parents=True, exist_ok=True)
    return problem_params


def plot_stability_domain(lmbda_laplacian_list, lmbda_gating_list, R, integrator, num_nodes, n_time_ranks):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import pySDC.helpers.plot_helper as plt_helper

    plt_helper.setup_mpl()

    # fig, ax = plt_helper.newfig(textwidth=400, scale=0.89, ratio=0.5)
    # fig, ax = plt_helper.newfig(textwidth=238.96, scale=0.89)
    fig, ax = plt_helper.plt.subplots(
        figsize=plt_helper.figsize(textwidth=400, scale=1.0, ratio=0.78), layout='constrained'
    )

    fs_label = 14
    fs_ticks = 12
    fs_title = 16
    X, Y = np.meshgrid(lmbda_gating_list, lmbda_laplacian_list)
    R = np.abs(R)
    CS = ax.contourf(X, Y, R, cmap=plt.cm.viridis, levels=np.logspace(-6, 0, 13), norm=LogNorm())
    ax.plot(lmbda_gating_list, 0 * lmbda_gating_list, 'k--', linewidth=1.0)
    ax.plot(0 * lmbda_laplacian_list, lmbda_laplacian_list, 'k--', linewidth=1.0)
    ax.contour(CS, levels=CS.levels, colors='black')
    ax.set_xlabel(r'$z_{e}$', fontsize=fs_label, labelpad=-5)
    ax.set_ylabel(r'$z_{I}$', fontsize=fs_label, labelpad=-10)
    ax.tick_params(axis='x', labelsize=fs_ticks)
    ax.tick_params(axis='y', labelsize=fs_ticks)
    if len(num_nodes) == 1 and n_time_ranks == 1:
        prefix = ""
    elif len(num_nodes) > 1 and n_time_ranks == 1:
        prefix = "ML"
    elif len(num_nodes) > 1 and n_time_ranks > 1:
        prefix = "PFASST "
    if integrator == "IMEXEXP":
        ax.set_title(prefix + "SDC stability domain", fontsize=fs_title)
    elif integrator == "IMEXEXP_EXPRK":
        ax.set_title(prefix + "ESDC stability domain", fontsize=fs_title)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel(r'$|R(z_e,z_{I})|$', fontsize=fs_label, labelpad=-20)
    cbar.set_ticks([cbar.vmin, cbar.vmax])  # keep only the ticks at the ends
    cbar.ax.tick_params(labelsize=fs_ticks)
    # plt_helper.plt.show()
    plt_helper.savefig("data/stability_domain_" + integrator, save_pdf=False, save_pgf=False, save_png=True)


def main(integrator, dl, l_min, openmp, n_time_ranks, end_time, num_nodes, check_stability):

    # get time integration parameters
    # set maximum number of iterations in SDC/ESDC/MLSDC/etc
    step_params = get_step_params(maxiter=5)
    # set number of collocation nodes in each level
    sweeper_params = get_sweeper_params(num_nodes=num_nodes)
    # set step size, number of sweeps per iteration, and residual tolerance for the stopping criterion
    level_params = get_level_params(dt=1.0, nsweeps=[1], restol=5e-8)
    # set space transfer parameters
    # space_transfer_class = Transfer_myfloat
    space_transfer_class = TransferVectorOfDCTVectors
    base_transfer_params = get_base_transfer_params()
    controller_params = get_controller_params(get_output_root(), logger_level=40)

    # set stability test parameters
    lmbda_others = -1.0  # the non stiff term
    lmbda_laplacian_min = l_min  # the stiff term
    lmbda_laplacian_max = 0.0
    lmbda_gating_min = l_min  # the exponential term
    lmbda_gating_max = 0.0

    # define the grid for the stability domain
    n_lmbda_laplacian = np.round((lmbda_laplacian_max - lmbda_laplacian_min) / dl).astype(int) + 1
    n_lmbda_gating = np.round((lmbda_gating_max - lmbda_gating_min) / dl).astype(int) + 1
    lmbda_laplacian_list = np.linspace(lmbda_laplacian_min, lmbda_laplacian_max, n_lmbda_laplacian)
    lmbda_gating_list = np.linspace(lmbda_gating_min, lmbda_gating_max, n_lmbda_gating)

    if not openmp:
        R = np.zeros((n_lmbda_laplacian, n_lmbda_gating))
        for i in tqdm(range(n_lmbda_gating)):
            for j in range(n_lmbda_laplacian):
                lmbda_gating = lmbda_gating_list[i]
                lmbda_laplacian = lmbda_laplacian_list[j]

                problem_params = get_problem_params(
                    lmbda_laplacian=lmbda_laplacian,
                    lmbda_gating=lmbda_gating,
                    lmbda_others=lmbda_others,
                    end_time=end_time,
                )
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
                controller = get_controller(controller_params, description, n_time_ranks)

                t0, Tend, uinit, P = get_P_data(controller)
                uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                R[j, i] = abs(uend)
    else:
        import pymp

        R = pymp.shared.array((n_lmbda_laplacian, n_lmbda_gating), dtype=float)
        with pymp.Parallel(12) as p:
            for i in tqdm(p.range(0, n_lmbda_gating)):
                for j in range(n_lmbda_laplacian):
                    lmbda_gating = lmbda_gating_list[i]
                    lmbda_laplacian = lmbda_laplacian_list[j]

                    problem_params = get_problem_params(
                        lmbda_laplacian=lmbda_laplacian,
                        lmbda_gating=lmbda_gating,
                        lmbda_others=lmbda_others,
                        end_time=end_time,
                    )
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
                    controller = get_controller(controller_params, description, n_time_ranks)

                    t0, Tend, uinit, P = get_P_data(controller)
                    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                    R[j, i] = abs(uend)

    plot_stability_domain(lmbda_laplacian_list, lmbda_gating_list, R, integrator, num_nodes, n_time_ranks)

    if check_stability:
        assert (
            np.max(np.abs(R.ravel())) <= 1.0
        ), "The maximum absolute value of the stability function is greater than 1.0."


if __name__ == "__main__":
    # Plot stability for exponential SDC coupled with the implicit-explicit-exponential integrator as preconditioner
    main(
        integrator="IMEXEXP_EXPRK",
        dl=2,
        l_min=-100,
        openmp=True,
        n_time_ranks=1,
        end_time=1.0,
        num_nodes=[5, 3],
        check_stability=True,  # check that the stability function is bounded by 1.0
    )
    # Plot stability for standard SDC coupled with the implicit-explicit-exponential integrator as preconditioner
    main(
        integrator="IMEXEXP",
        dl=2,
        l_min=-100,
        openmp=True,
        n_time_ranks=1,
        end_time=1.0,
        num_nodes=[5, 3],
        check_stability=False,  # do not check for stability since we already know that the method is not stable
    )
