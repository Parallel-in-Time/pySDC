import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted

from pySDC.projects.Resilience.advection import run_advection


class log_residual(hooks):
    def post_iteration(self, step, level_number):
        '''
        Log the residual after every iteration
        '''
        super(log_residual, self).post_iteration(step, level_number)
        L = step.levels[0]
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='res', value=L.status.residual)


def contraction_rate(ax, precon, desc=None, label=None):
    custom_problem_params = {'freq': -1}  # Gaussian

    step_params = {'maxiter': 50}
    sweeper_params = {'QI': precon}
    level_params = {}  # {'dt': 1e-3}

    custom_description = {'step_params': step_params, 'sweeper_params': sweeper_params, 'level_params': level_params}
    if desc is not None:
        custom_description = {**custom_description, **desc}
    stats, _, _ = run_advection(hook_class=log_residual, custom_problem_params=custom_problem_params,
                                custom_description=custom_description, Tend=1e-2)

    res = get_sorted(stats, type='res')

    # get the time of the first step and the residual from all iterations in that step
    t0 = res[0][0]
    r = [res[i][1] for i in range(len(res)) if res[i][0] == t0]

    k = np.arange(len(r)) + 1
    ax.plot(k, r, label=precon if label is None else label)
    ax.set_yscale('log')
    ax.legend(frameon=False)


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    precons = ['IE', 'LU', 'MIN', 'MIN3']
    [contraction_rate(ax=ax, precon=precon) for precon in precons]
    plt.show()
