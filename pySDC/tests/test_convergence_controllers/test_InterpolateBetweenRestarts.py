import pytest

from pySDC.core.Hooks import hooks
from pySDC.core.Lagrange import LagrangeApproximation
from pySDC.core.Collocation import CollBase
import numpy as np


class LogInterpolation(hooks):
    """
    Log the solution when a step is supposed to be restarted as well as the interpolated solution to the new nodes and
    the solution that ends up at the nodes after the restart.
    """

    def __init__(self):
        super().__init__()
        self.log_u_now = False

    def pre_iteration(self, step, level_number):
        if self.log_u_now:
            L = step.levels[level_number]

            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='u_inter',
                value=L.u.copy(),
            )
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='nodes_inter',
                value=L.sweep.coll.nodes * L.params.dt,
            )
            self.log_u_now = False

    def post_step(self, step, level_number):
        super().post_iteration(step, level_number)

        L = step.levels[level_number]

        if step.status.restart:
            self.log_u_now = True
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='u_before_interpolation',
                value=L.u.copy(),
            )
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='nodes',
                value=L.sweep.coll.nodes * L.params.dt,
            )

            # double check
            nodes_old = L.sweep.coll.nodes.copy()
            nodes_new = L.sweep.coll.nodes.copy() * L.status.dt_new / L.params.dt
            interpolator = LagrangeApproximation(points=np.append(0, nodes_old))
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='u_inter_double_check',
                value=(interpolator.getInterpolationMatrix(np.append(0, nodes_new)) @ L.u[:])[:],
            )


@pytest.mark.base
def test_InterpolateBetweenRestarts(plotting=False):
    """
    Check that the solution is interpolated to the new nodes correctly and ends up the next step the way we want it to.
    We also check that the residual at the end of the step after the restart is smaller than before.
    """
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
    from pySDC.implementations.convergence_controller_classes.interpolate_between_restarts import (
        InterpolateBetweenRestarts,
    )

    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted, filter_stats
    from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'spread'

    problem_params = {
        'mu': 5.0,
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
        'u0': np.array([2.0, 0.0]),
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # convergence controllers
    convergence_controllers = {
        Adaptivity: {'e_tol': 1e-7},
        InterpolateBetweenRestarts: {},
    }

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = LogInterpolation
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=1e-2)

    u = {
        'before': get_sorted(stats, type='u_before_interpolation'),
        'after': get_sorted(stats, type='u_inter'),
        'double_check': get_sorted(stats, type='u_inter_double_check'),
    }

    nodes = {
        'before': get_sorted(stats, type='nodes'),
        'after': get_sorted(stats, type='nodes_inter'),
        'double_check': get_sorted(stats, type='nodes_inter'),
    }

    residual = get_sorted(stats, type='residual_post_step')
    for t in np.unique([me[0] for me in residual]):
        _res = np.array([me[1] for me in residual if me[0] == t])
        if len(_res) > 1:
            contraction = _res[1:] / _res[:-1]
            assert all(
                contraction < 6e-3
            ), f"Residual was not decreased as much as expected! Got {max(contraction):.2e}. Without interpolation we expect about 0.15, but with interpolation we want about 6e-3!"

    for i in range(len(u['before'])):
        # check the nodes
        assert nodes['after'][i][1][-1] < nodes['before'][i][1][-1], "Step size was not reduced!"

        # check the solution
        for j in range(len(u['before'][i][1])):
            assert (
                abs(u['double_check'][i][1][j] - u['after'][i][1][j]) < 1e-12
            ), f"The interpolated solution from the convergence controller is not right! Expected {u['double_check'][i][1][j]}, got {u['after'][i][1][j]}"

    if plotting:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True)

        colors = {
            'before': 'teal',
            'after': 'violet',
            'double_check': 'black',
        }

        ls = {'before': '-', 'after': '--', 'double_check': '-.'}
        for i in [0, 1]:
            for key in nodes.keys():
                axs[0].plot(
                    np.append([0], nodes[key][i][1]), [me[1] for me in u[key][i][1]], color=colors[key], ls=ls[key]
                )
                axs[1].plot(
                    np.append([0], nodes[key][i][1]), [me[0] for me in u[key][i][1]], color=colors[key], ls=ls[key]
                )
        axs[1].set_xlabel('$t$')
        axs[0].set_ylabel('$u_t$')
        axs[1].set_ylabel('$u$')
        plt.show()


if __name__ == "__main__":
    test_InterpolateBetweenRestarts(plotting=True)
