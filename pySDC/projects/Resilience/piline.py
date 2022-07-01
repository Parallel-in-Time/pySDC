import types
import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.Resilience.fault_injection import FaultInjector


class log_data(FaultInjector):

    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='u', value=L.uend)
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='k', value=step.status.iter)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='v1', value=L.uend[0])
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='v2', value=L.uend[1])
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='p3', value=L.uend[2])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_embedded', value=L.status.error_embedded_estimate)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_extrapolated', value=L.status.error_extrapolation_estimate)
        self.increment_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='restart', value=int(step.status.restart))
        self.increment_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='sweeps', value=step.status.iter)


def run_piline(strategy, rng, faults, force_params=types.MappingProxyType({}), num_procs=1):
    """
    A simple test program to do PFASST runs for the Pi-line equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 5e-2
    level_params['e_tol'] = 5e-7

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'Vs': 100.,
        'Rs': 1.,
        'C1': 1.,
        'Rpi': 0.2,
        'C2': 1.,
        'Lpi': 1.,
        'Rl': 5.,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data
    controller_params['use_HotRod'] = False
    controller_params['use_adaptivity'] = False
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 2e+1

    if strategy == 'adaptivity':
        controller_params['use_adaptivity'] = True
    elif strategy == 'HotRod':
        controller_params['use_HotRod'] = True
    elif strategy == 'iterate':
        step_params['maxiter'] = 99
        level_params['restol'] = 2.3e-8

    # check if we want to change some parameters
    for k in force_params.keys():
        for j in force_params[k].keys():
            if k == 'controller_params':
                controller_params[j] = force_params[k][j]
            else:
                description[k][j] = force_params[k][j]

    # instantiate controller
    controller_class = controller_nonMPI
    controller = controller_class(num_procs=num_procs, controller_params=controller_params,
                                  description=description)

    controller.hooks.random_generator = rng
    if faults:
        controller.hooks.add_random_fault(time=2.5, rnd_args={'iteration': 4}, args={'target': 0})

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller, Tend


def plot(stats, controller):
    v1 = [me[1] for me in get_sorted(stats, type='v1', recomputed=False)]
    v2 = [me[1] for me in get_sorted(stats, type='v2', recomputed=False)]
    p3 = [me[1] for me in get_sorted(stats, type='p3', recomputed=False)]
    t = [me[0] for me in get_sorted(stats, type='v1', recomputed=False)]
    k = get_sorted(stats, type='k', recomputed=False)
    dt = get_sorted(stats, type='dt', recomputed=False)
    restarts = get_sorted(stats, type='restart')

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, v1, label=r'$v_1$')
    ax.plot(t, v2, label=r'$v_2$')
    ax.plot(t, p3, label=r'$p_3$')
    ax.legend(frameon=False)

    k_ax = ax.twinx()
    k_ax.axhline(4, color='grey')
    k_ax.plot([me[0] for me in k], [me[1] for me in k])
    k_ax.plot([me[0] for me in dt], [me[1] / dt[0][1] for me in dt])

    for t_r in [me[0] for me in restarts if me[1]]:
        k_ax.axvline(t_r, color='grey', ls=':')
    fig.savefig('data/Piline_sol.pdf', transparent=True)

    exact = controller.MS[0].levels[0].prob.u_exact(t=t[-1])
    e = abs(exact - np.array([v1[-1], v2[-1], p3[-1]]))
    k_tot = sum([me[1] for me in k])
    print(f'e={e:.2e} (2e-8), k_tot={k_tot} (1600)')


if __name__ == "__main__":
    strategy = 'iterate'
    force_params = {'controller_params': {'logger_level': 20}}
    stats, controller, Tend = run_piline(strategy, np.random.RandomState(16), False, force_params)
    plot(stats, controller)
