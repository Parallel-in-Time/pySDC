from pathlib import Path
import numpy as np
import logging
import os

from tqdm import tqdm

from pySDC.core.Errors import ParameterError

# from pySDC.projects.Monodomain.problem_classes.TestODE import MultiscaleTestODE
# from pySDC.projects.Monodomain.transfer_classes.Transfer_myfloat import Transfer_myfloat
from pySDC.projects.Monodomain.problem_classes.TestODEnew import MultiscaleTestODE
from pySDC.projects.Monodomain.transfer_classes.TransferVectorOfDCTVectors import TransferVectorOfDCTVectors

from pySDC.projects.Monodomain.hooks.HookClass_post_iter_info import post_iter_info_hook

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import (
    imexexp_1st_order as imexexp_1st_order_ExpRK,
)
from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.imexexp_1st_order import imexexp_1st_order


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


def plot_stability_domain(lmbda_laplacian_list, lmbda_gating_list, R):
    import matplotlib.pyplot as plt

    plt.rc("text", usetex=True)

    fs_label = 16
    fs_ticks = 16
    fig, ax = plt.subplots(layout='constrained')
    X, Y = np.meshgrid(lmbda_gating_list, lmbda_laplacian_list)
    R = np.abs(R)
    CS = ax.contourf(X, Y, R, cmap=plt.cm.bone, levels=np.array([0.0, 1.0]))
    ax.plot(lmbda_gating_list, 0 * lmbda_gating_list, 'k--', linewidth=1.0)
    ax.plot(0 * lmbda_laplacian_list, lmbda_laplacian_list, 'k--', linewidth=1.0)
    ax.contour(CS, levels=CS.levels, colors='black')
    ax.set_xlabel(r'$z_{g}$', fontsize=fs_label)
    ax.set_ylabel(r'$z_{\Delta}$', fontsize=fs_label)
    ax.tick_params(axis='x', labelsize=fs_ticks)
    ax.tick_params(axis='y', labelsize=fs_ticks)
    # ax.set_title(r'$R(z_g,z_{\Delta})$')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    # cbar = fig.colorbar(CS)
    # cbar.ax.set_ylabel(r'$R(z_{\Delta},z_g)$')
    plt.show()


def main(dl, openmp):

    # integrator = "IMEXEXP"
    integrator = "IMEXEXP_EXPRK"

    # number of time ranks.
    n_time_ranks = 1

    end_time = float(n_time_ranks)

    # get time integration parameters
    # set maximum number of iterations in SDC/ESDC/MLSDC/etc
    step_params = get_step_params(maxiter=5)
    # set number of collocation nodes in each level
    sweeper_params = get_sweeper_params(num_nodes=[5, 3])
    # set step size, number of sweeps per iteration, and residual tolerance for the stopping criterion
    level_params = get_level_params(dt=1.0, nsweeps=[1], restol=5e-8)
    # set space transfer parameters
    # space_transfer_class = Transfer_myfloat
    space_transfer_class = TransferVectorOfDCTVectors
    base_transfer_params = get_base_transfer_params()
    controller_params = get_controller_params(get_output_root(), logger_level=40)

    # set stability test parameters
    lmbda_others = -1.0
    lmbda_laplacian_min = -1000.0
    lmbda_laplacian_max = 0.0
    lmbda_gating_min = -1000.0
    lmbda_gating_max = 0.0
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

    plot_stability_domain(lmbda_laplacian_list, lmbda_gating_list, R)

    assert np.max(np.abs(R.ravel())) <= 1.0, "The maximum absolute value of the stability function is greater than 1.0."


if __name__ == "__main__":
    main(10, True)
