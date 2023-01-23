import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.heat import run_heat
from pySDC.projects.Resilience.hook import LogData
from pySDC.projects.Resilience.accuracy_check import get_accuracy_order
from pySDC.implementations.convergence_controller_classes.collocation_inexactness import CollocationInexactness
from pySDC.core.Hooks import hooks


# define global parameters for running problems and plotting
CMAP = list(TABLEAU_COLORS.values())


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
    'num_nodes': [1, 2, 3, 4],
}
coll_params_reduce = {
    'num_nodes': [4, 3, 2, 1],
}
coll_params_type = {
    # 'quad_type': ['RADAU-RIGHT', 'GAUSS'],
    'quad_type': ['GAUSS', 'RADAU-RIGHT', 'LOBATTO'],
}

special_params = {
    'inexact': {CollocationInexactness: coll_params_inexact},
    'refinement': {CollocationInexactness: coll_params_refinement},
    'reduce': {CollocationInexactness: coll_params_reduce},
    'standard': {},
    'type': {CollocationInexactness: coll_params_type},
}


def get_collocation_order(quad_type, num_nodes, node_type):
    """
    Compute the maximal order achievable by a given collocation method
    """
    pass


# define a few hooks
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
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='coll_order',
            value=L.sweep.coll.order,
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


# plotting functions
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
        'refinement': [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4],
        'reduce': [4, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 1],
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


def check_order(prob, coll_name, ax, k_ax):
    """
    Make plot of the order of the collocation problems and check if they are as expected.

    Args:
        prob (function): A problem from the resilience project to run
        coll_name (str): The name of the collocation refinement strategy
        ax (Matplotlib.pyplot.axes): Somewhere to plot
        k_ax (Matplotlib.pyplot.axes): Somewhere to plot

    Returns:
        None
    """
    dt_range = [2.0 ** (-i) for i in range(2, 11)]

    res = []

    label_keys = {
        'type': 'quad_type',
    }

    for i in range(len(dt_range)):
        new_params = {
            'level_params': {'restol': 1e-9, 'dt': dt_range[i]},
            'sweeper_params': {'num_nodes': 2, 'QI': 'IE'},
        }
        custom_description = {**base_params, 'convergence_controllers': special_params[coll_name], **new_params}
        custom_controller_parameters = {'logger_level': 30}
        stats, _, _ = prob(
            Tend=dt_range[i],
            custom_description=custom_description,
            custom_controller_params=custom_controller_parameters,
            hook_class=[LogData, LogSweeperParams, LogLocalErrorPostIter],
        )

        sweeper_params = get_sorted(stats, type='sweeper_params', sortby='iter')
        converged_solution = [
            sweeper_params[i][1] != sweeper_params[i + 1][1] for i in range(len(sweeper_params) - 1)
        ] + [True]
        idx = np.arange(len(converged_solution))[converged_solution]
        labels = [sweeper_params[i][1][label_keys.get(coll_name, 'num_nodes')] for i in idx]
        e_loc = np.array([me[1] for me in get_sorted(stats, type='e_local_post_iter', sortby='iter')])[
            converged_solution
        ]
        coll_order = np.array([me[1] for me in get_sorted(stats, type='coll_order', sortby='iter')])[converged_solution]

        res += [(dt_range[i], e_loc, idx[1:] - idx[:-1], labels, coll_order)]
        # res += [(dt_range[i], np.array([me[1] for me in e_loc])[converged_solution], (idx[1:]-idx[:-1])/(idx[:-1]+1)*100, labels)]

    # assemble sth we can compute the order from
    result = {'dt': [me[0] for me in res]}
    num_sols = len(res[0][1])
    for i in range(num_sols):
        result[i] = [me[1][i] for me in res]
        label = res[0][3][i]
        expected_order = res[0][4][i] + 1

        order = get_accuracy_order(result, key=i, thresh=1e-9)
        assert np.isclose(
            np.mean(order), expected_order, atol=0.3
        ), f"Expected order: {expected_order}, got {order:.2f}!"
        ax.loglog(result['dt'], result[i], label=f'{label} nodes: order: {np.mean(order):.1f}', color=CMAP[i])

        if i > 0:
            extra_iter = [me[2][i - 1] for me in res]
            k_ax.plot(result['dt'], extra_iter, ls='--', color=CMAP[i])
    ax.legend(frameon=False)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$e_\mathrm{local}$')
    k_ax.set_ylabel(r'extra iterations')


def order_stuff(prob):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    k_axs = []
    modes = ['type', 'refinement', 'reduce']
    for i in range(len(modes)):
        k_axs += [axs.flatten()[i].twinx()]
        check_order(prob, modes[i], axs.flatten()[i], k_axs[-1])

    for i in range(2):
        k_axs[i].set_ylabel('')

    for ax in axs[1:]:
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.tight_layout()


def main():
    order_stuff(run_advection)
    compare_collocation_inexactness(run_vdp)


if __name__ == "__main__":
    main()
    plt.show()
