import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.heat import run_heat
from pySDC.projects.Resilience.hook import LogData
from pySDC.implementations.convergence_controller_classes.collocation_inexactness import CollocationInexactness
from pySDC.core.Hooks import hooks


CMAP = list(TABLEAU_COLORS.values())


class LogSweeperParams(hooks):
    """
    Log the sweeper parameters after every iteration to check if the collocation inexactness is doing what it's supposed
    to.
    """

    def post_iteration(self, step, level_number):
        """
        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_iteration(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='sweeper_params',
            value=L.sweep.params.__dict__,
        )


class LogLocalErrorPostIter(hooks):
    """
    Log the local error after each iteration
    """

    def post_iteration(self, step, level_number):
        """
        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_local_post_iter',
            value=abs(L.prob.u_exact(t=L.time + L.dt) - L.uend),
        )


def compare_collocation_inexactness(prob):
    """
    Run a problem with various modes of collocation inexactness.

    Args:
        prob (function): A problem from the resilience project to run

    Returns:
        None
    """
    fig, ax = plt.subplots()
    node_ax = ax.twinx()

    Tend = 0.015
    base_params = {
        'step_params': {'maxiter': 99},
        'sweeper_params': {
            'QI': 'LU',
            'num_nodes': 4,
        },
        'level_params': {'restol': 1e-9, 'dt': Tend},
    }

    coll_params_inexact = {
        'num_nodes': [2, 3, 4],
        'restol': [1e-4, 1e-7, 1e-9],
    }
    coll_params_refinement = {
        'num_nodes': [3, 4],
    }
    coll_params_reduce = {
        'num_nodes': [4, 3],
    }
    coll_params_type = {
        # 'quad_type': ['RADAU-RIGHT', 'GAUSS'],
        'quad_type': ['RADAU-RIGHT', 'LOBATTO'],
    }

    special_params = {
        'inexact': {CollocationInexactness: coll_params_inexact},
        'refinement': {CollocationInexactness: coll_params_refinement},
        'reduce': {CollocationInexactness: coll_params_reduce},
        'standard': {},
        'type': {CollocationInexactness: coll_params_type},
    }

    for i in range(len(special_params.keys())):
        key = list(special_params.keys())[i]
        custom_description = {**base_params, 'convergence_controllers': special_params[key]}
        custom_controller_parameters = {'logger_level': 30}
        stats, _, _ = prob(
            Tend=Tend,
            custom_description=custom_description,
            custom_controller_params=custom_controller_parameters,
            hook_class=[LogData, LogSweeperParams],
        )

        plot_residual(stats, ax, node_ax, label=key, color=CMAP[i])


def plot_residual(stats, ax, node_ax, **kwargs):
    """
    Plot residual and nodes vs. iteration.
    Also a test is performed to see if we can reproduce previously obtained results.

    Args:
        stats (pySDC.stats): The stats object of the run
        ax (Matplotlib.pyplot.axes): Somewhere to plot
        node_ax (Matplotlib.pyplot.axes): Somewhere to plot

    Returns:
        None
    """
    sweeper_params = get_sorted(stats, type='sweeper_params', sortby='iter')
    residual = get_sorted(stats, type='residual_post_iteration', sortby='iter')

    # determine when the number of collocation nodes increased
    nodes = [me[1]['num_nodes'] for me in sweeper_params]

    # test if the expected outcome was achieved
    label = kwargs['label']
    expect = {
        'inexact': [2, 2, 3, 3, 4, 4],
        'refinement': [3, 3, 3, 3, 3, 3, 4, 4],
        'reduce': [4, 4, 4, 4, 4, 4, 3, 3],
        'standard': [4, 4, 4, 4, 4, 4],
        'type': [4, 4, 4, 4, 4, 4, 4],
    }
    assert np.allclose(
        nodes, expect[label]
    ), f"Unexpected distribution of nodes vs. iteration in {label}! Expected {expect[label]}, got {nodes}"

    ax.plot([me[0] for me in residual], [me[1] for me in residual], **kwargs)
    ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'residual')

    node_ax.plot([me[0] for me in sweeper_params], nodes, **kwargs, ls='--')
    node_ax.set_ylabel(r'nodes')


def main():
    compare_collocation_inexactness(run_vdp)


if __name__ == "__main__":
    main()
    plt.show()
