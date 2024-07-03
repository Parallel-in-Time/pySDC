import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.playgrounds.DAE.genericImplicitDAE import genericImplicitConstrained, genericImplicitEmbedded, genericImplicitOriginal
from pySDC.playgrounds.DAE.LinearTestDAEMinion import LinearTestDAEMinionConstrained, LinearTestDAEMinionEmbedded
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.playgrounds.DAE.log_residual_components import LogResidualComponentsPostIter

from pySDC.helpers.stats_helper import get_sorted


def flatten_nested_list(nested_list):
    flattened_list = [sublist for sublist in nested_list]
    result_list = []
    for sublist in flattened_list:
        extracted_elements = [element[1] for element in sublist]
        if len(extracted_elements) > 0:
            result_list.append([element[1] for element in sublist])
    result_list = [element[0] for element in result_list]
    return result_list


def run(dt, restol, maxiter, M, QI, problem, sweeper, t0, Tend, hook_class):
    r"""
    It simply generates a description without any unnecessary stuff. Then, a run is simulated for a problem.

    Parameters
    dt : float
        Time step size.
    restol : float
        Residual tolerance used to execute.
    maxiter : int
        Number of maximum iterations.
    M : int
        Number of collocation nodes.
    QI : np.2darray
        Matrix :math:`Q_\Delta`.
    problem : pySDC.projects.DAE.misc.ptype_dae
        Problem class to be solved.
    sweeper : pySDC.core.Sweeper
        Sweeper used for simulation.
    t0 : float
        Staring time.
    Tend : float
        End time
    """

    # initialize level parameters
    level_params = {
        'restol': restol,
        'dt': dt,
    }

    # initialize problem parameters
    problem_params = {
        'newton_tol': 1e-12,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': QI,
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': maxiter,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class' : hook_class
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': problem,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = t0
    Tend = Tend

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    Path("data").mkdir(parents=True, exist_ok=True)

    # call main function to get things done...
    _, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


def main():
    r"""
    In this main function all the things are done to plot the components for the residual. Here, also the original generic_implicit sweeper
    is applied in the way to the problem so that quadrature is not applied to the algebraic parts. This can be don as long as the
    ``solve_system`` does the correct things.
    """

    problems = [LinearTestDAEMinionConstrained, LinearTestDAEMinionEmbedded, LinearTestDAEMinionConstrained]
    sweepers = [genericImplicitConstrained, genericImplicitEmbedded, genericImplicitOriginal]

    hook_class = [LogResidualComponentsPostIter]

    M = 5
    QI = 'LU'  # try also 'IE' to see how the residual component in algebraic equation behaves!
    restol = -1
    maxiter = 40

    t0 = 0.0
    dtValues = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    color = ['firebrick', 'dodgerblue', 'forestgreen', 'black']

    for i, dt in enumerate(dtValues):
        fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
        fig2, ax2 = plt.subplots(1, 1, figsize=(9.5, 9.5))

        for ind, (problem, sweeper) in enumerate(zip(problems, sweepers)):
            ax_wrapper = ax[ind] if ind < 2 else ax2

            Tend = t0 + dt
            stats = run(dt, restol, maxiter, M, QI, problem, sweeper, t0, Tend, hook_class)

            residual_components_iter = [get_sorted(stats, iter=k, type='residual_comp_post_iter', sortby='time') for k in range(1, maxiter + 1)]
            residual_components_iter = flatten_nested_list(residual_components_iter)

            if sweeper.__name__ == 'genericImplicitEmbedded':
                linestyle = 'solid'
                title = r"Residual components of $\mathtt{SDC-E}$"
            elif sweeper.__name__ == 'genericImplicitConstrained':
                linestyle = 'dashed'
                title = r"Residual components of $\mathtt{SDC-C}$"
            else:
                linestyle = 'dashed'
                title = "Residual components of original SDC-ODE sweeper"

            solid_capstyle = 'round' if linestyle == 'solid' else None
            dash_capstyle = 'round' if linestyle == 'dashed' else None

            ax_wrapper.set_title(title)
            for diff_ind in range(len(residual_components_iter[0].diff)):
                ax_wrapper.semilogy(
                    np.arange(1, len(residual_components_iter) + 1),
                    [abs(comp[0][diff_ind]) for comp in residual_components_iter],
                    color=color[diff_ind],
                    linewidth=4.0,
                    linestyle=linestyle,
                    solid_capstyle=solid_capstyle,
                    dash_capstyle=dash_capstyle,
                    label=rf"$u_{diff_ind + 1}$",
                )

            ax_wrapper.semilogy(
                np.arange(1, len(residual_components_iter) + 1),
                [abs(comp[1][0]) for comp in residual_components_iter],
                color=color[-1],
                linewidth=4.0,
                linestyle=linestyle,
                marker='o',
                markersize=10.0,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=r"$u_4$",
            )

            ax_wrapper.tick_params(axis='both', which='major', labelsize=14)
            ax_wrapper.set_xlabel(r'Iteration $k$', fontsize=20)
            ax_wrapper.set_xlim(0, maxiter + 1)
            ax_wrapper.set_yscale('symlog', linthresh=1e-17)
            ax_wrapper.set_ylim(-1e-17, 1e1)
            ax_wrapper.minorticks_off()

        ax[0].set_ylabel(r"$||r_{u_{1, 2, 3, 4}}^k||_\infty$", fontsize=20)
        ax[0].legend(frameon=False, fontsize=12, loc='upper right', ncols=2)

        ax2.set_ylabel(r"$||r_{u_{1, 2, 3, 4}}^k||_\infty$", fontsize=20)
        ax2.legend(frameon=False, fontsize=12, loc='upper right', ncols=2)

        fig.savefig(f"data/{i}_plotResidualComponentsEmbedding_dt={dt}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig2.savefig(f"data/{i}_plotResidualComponentsOriginal_dt={dt}.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)


if __name__ == "__main__":
    main()
