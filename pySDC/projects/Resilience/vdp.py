import types
import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
import pySDC.helpers.plot_helper as plt_helper
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Errors import ProblemError

from fault_injection import FaultInjector

num_procs = 1
plt_helper.setup_mpl()


def test_float_conversion():
    # Try the conversion between floats and bytes
    injector = FaultInjector()
    exp = [-1, 2, 256]
    bit = [0, 11, 8]
    nan_counter = 0
    num_tests = int(1e3)
    for i in range(num_tests):
        # generate a random number almost between the full range of python float
        rand = np.random.uniform(low=-1.797693134862315e+307, high=1.797693134862315e+307, size=1)[0]
        # convert to bytes and back
        res = injector.to_float(injector.to_binary(rand))
        assert np.isclose(res, rand), f"Conversion between bytes and float failed for {rand}: result: {res}"

        # flip some bits
        for i in range(len(exp)):
            res = injector.flip_bit(rand, bit[i]) / rand
            if np.isfinite(res):
                assert exp[i] in [res, 1. / res], f'Bitflip failed: expected ratio: {exp[i]}, got: {res:.2e} or \
{1./res:.2e}'
            else:
                nan_counter += 1
    if nan_counter > 0:
        print(f'When flipping bits, we got nan {nan_counter} times out of {num_tests} tests')


class log_data(FaultInjector):

    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='u', value=L.uend)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_em', value=L.status.error_embedded_estimate)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_ex', value=L.status.error_extrapolation_estimate)
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='k', value=step.status.iter)
        self.increment_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='restart', value=int(step.status.restart))


def run_vdp(strategy, rng, faults, force_params=types.MappingProxyType({})):
    """
    A simple test program to do PFASST runs for the van der Pol equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-2
    level_params['e_tol'] = 3e-5

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    problem_params = {
        'mu': 5.,
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
        'u0': np.array([0.99995, -0.00999985]),
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 3

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = vanderpol  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    Tend = 2.3752559741400825

    if strategy == 'adaptivity':
        controller_params['use_adaptivity'] = True
    elif strategy == 'HotRod':
        controller_params['use_HotRod'] = True
        controller_params['HotRod_tol'] = 5e-7
        level_params['e_tol'] = 1e-6
        step_params['maxiter'] = 4
    elif strategy == 'iterate':
        step_params['maxiter'] = 100
        level_params['restol'] = 9e-7

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
        controller.hooks.add_random_fault(time=1.1, rnd_args={'iteration': 3}, args={'target': 0})

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ProblemError:
        stats = controller.hooks.return_stats()

    return stats, controller, Tend


def my_plot(stats, controller):
    fig, axs = plt.subplots(2, 1, sharex=True)

    # convert filtered statistics to list of iterations count, sorted by process
    u = [m[1][0] for m in get_sorted(stats, type='u', recomputed=False)]
    p = [m[1][1] for m in get_sorted(stats, type='u', recomputed=False)]

    t = np.array([m[0] for m in get_sorted(stats, type='u', recomputed=False)])
    ts = np.array([m[0] for m in get_sorted(stats, type='e_em', recomputed=False)])

    e_em = np.array(get_sorted(stats, type='e_em', recomputed=False))[:, 1]
    e_ex = np.array(get_sorted(stats, type='e_ex', recomputed=False))[:, 1]
    dt = np.array(get_sorted(stats, type='dt', recomputed=False))[:, 1]
    restarts = get_sorted(stats, type='restart')

    print(f't: {t[-1]}')
    axs[0].plot(t, u)
    axs[0].plot(t, p)

    dt_ax = axs[1].twinx()
    dt_ax.plot(ts, dt, label='dt', color='black')
    dt_ax.set_yscale('log')
    dt_ax.legend(frameon=False)
    dt_ax.set_ylabel(r'$\Delta t$')

    hr_ready = np.logical_and(e_em != [None], e_ex != [None])
    e_ax = axs[1]
    e_ax.plot(ts, e_em, label=r'$\epsilon_\mathrm{embedded}$')
    e_ax.axhline(3e-5, color='grey')
    e_ax.plot(ts, e_ex, label=r'$\epsilon_\mathrm{extrapolation}$', marker='*')

    e_ax.plot(ts[hr_ready], abs(e_em[hr_ready] - e_ex[hr_ready]))
    e_ax.set_yscale('log')
    e_ax.legend(frameon=False)

    axs[1].set_xlabel('time')

    for ax in [axs[0], dt_ax]:
        for t in [me[0] for me in restarts if me[1]]:
            ax.axvline(t, color='grey', ls=':')

    fig.tight_layout()
    plt.show()


def get_error(stats, controller):
    u = get_sorted(stats, type='u')[-1][1]
    S = controller.MS[0]
    u_exact = S.levels[0].prob.u_exact(t=get_sorted(stats, type='u')[-1][0])
    k = [me[1] for me in get_sorted(stats, type='k', recomputed=False)]
    restarts = [me[1] for me in get_sorted(stats, type='restart', recomputed=False)]
    dt = [me[1] for me in get_sorted(stats, type='dt', recomputed=False)]
    print(f'Error: {abs(u - u_exact):.2e} in {sum(k)} iterations, with avg. dt: {np.mean(dt):.2e} and {sum(restarts)} \
restarts')


def get_order(strategy, k=4):
    force_params = {
        'controller_params': {'logger_level': 30},
        'step_params': {'maxiter': k},
        'level_params': {},
    }

    if strategy == 'adaptivity':
        key = 'e_tol'
        vals = [1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8, 1e-8]
    elif strategy == 'nothing':
        key = 'dt'
        vals = [5e-2, 4e-2, 3e-2, 2e-2, 1e-2]

    e = np.zeros(len(vals))
    dt = np.zeros(len(vals))

    for i in range(len(vals)):
        force_params['level_params'][key] = vals[i]
        stats, controller = run_vdp(strategy, None, False, force_params)

        u = get_sorted(stats, type='u')[-1][1]
        S = controller.MS[0]
        u_exact = S.levels[0].prob.u_exact(t=get_sorted(stats, type='u')[-1][0])
        e[i] = abs(u - u_exact)
        dt[i] = np.mean([me[1] for me in get_sorted(stats, type='dt')])

    order = np.log(e[:-1] / e[1:]) / np.log(dt[:-1] / dt[1:])
    fig, ax = plt.subplots(1, 1)
    ax.scatter(dt, e)
    ax.loglog(dt, e[0] * (dt / dt[0])**np.mean(order), label=f'order: {np.mean(order):.2f}')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f'data/order-{strategy}.pdf', transparent=True)


if __name__ == "__main__":
    # get_order('adaptivity', 4)
    force_params = {'controller_params': {'logger_level': 20}}
    stats, controller, Tend = run_vdp('iterate', np.random.RandomState(16), True, force_params)
    get_error(stats, controller)
    my_plot(stats, controller)
