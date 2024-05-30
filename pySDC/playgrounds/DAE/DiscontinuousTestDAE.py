import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.hooks.log_solution import LogSolution


def simulateDAE():
    r"""
    Main function where things will be done. Here, the problem class ``DiscontinuousTestDAE`` is simulated using
    the ``SemiImplicitDAE`` sweeper, where only the differential variable is integrated using spectral quadrature.
    The problem usually contains a discrete event, but the simulation interval is chosen so that the state function
    does not have a sign change yet (the discrete event does not yet occur). Thus, the solution is smooth.
    """
    # initialize level parameters
    level_params = {
        'restol': 1e-12,
        'dt': 0.1,
    }

    # initialize problem parameters
    problem_params = {
        'newton_tol': 1e-6,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 3,
        'QI': 'IE',
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 60,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogSolution],
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': DiscontinuousTestDAE,
        'problem_params': problem_params,
        'sweeper_class': SemiImplicitDAE,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 1.0
    Tend = 3.0

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    Path("data").mkdir(parents=True, exist_ok=True)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    plotSolution(stats)


def plotSolution(stats, file_name='data/solution.png'):
    r"""
    Here, the solution of the DAE is plotted.

    Parameters
    ----------
    stats : dict
        Contains all the statistics of the simulation.
    """

    u_val = get_sorted(stats, type='u', sortby='time')
    t = np.array([me[0] for me in u_val])
    y = np.array([me[1].diff[0] for me in u_val])
    z = np.array([me[1].alg[0] for me in u_val])

    plt.figure(figsize=(8.5, 8.5))
    plt.plot(t, y, label='Differential variable y')
    plt.plot(t, z, label='Algebraic variable z')
    plt.legend(frameon=False, fontsize=12, loc='upper left')
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Solution $y(t)$, $z(t)$")

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    simulateDAE()
