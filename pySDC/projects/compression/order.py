import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.Resilience.advection import run_advection

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

from pySDC.projects.compression.compression_convergence_controller import Compression

MACHINEPRECISION = (
    1e-8  # generous tolerance below which we ascribe errors to floating point rounding errors rather than compression
)
LOGGER_LEVEL = 30


def single_run(problem, description=None, thresh=1e-10, Tend=2e-1, useMPI=False, num_procs=1):
    description = {} if description is None else description

    compressor_args = {}
    compressor_args['compressor_config'] = {'pressio:abs': thresh}

    if thresh > 0:
        description['convergence_controllers'] = {Compression: {'compressor_args': compressor_args}}

    controller_params = {
        'mssdc_jac': False,
        'logger_level': LOGGER_LEVEL,
    }

    stats, _, _ = problem(
        custom_description=description,
        hook_class=LogGlobalErrorPostRun,
        Tend=Tend,
        use_MPI=useMPI,
        num_procs=num_procs,
        custom_controller_params=controller_params,
    )

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None
    e = min([me[1] for me in get_sorted(stats, type='e_global_post_run', comm=comm)])
    return e


def multiple_runs(problem, values, expected_order, mode='dt', thresh=1e-10, useMPI=False, num_procs=1, **kwargs):
    errors = np.zeros_like(values)

    description = {
        'level_params': {},
        'problam_params': {},
        'step_params': {},
    }
    if mode == 'dt':
        description['step_params'] = {'maxiter': expected_order}
    elif mode == 'nvars':
        description['problem_params'] = {'order': expected_order}

    for i in range(len(values)):
        if mode == 'dt':
            description['level_params']['dt'] = values[i]
            Tend = values[i] * (5 if num_procs == 1 else 2 * num_procs)
        elif mode == 'nvars':
            description['problem_params']['nvars'] = values[i]
            Tend = 2e-1

        errors[i] = single_run(problem, description, thresh=thresh, Tend=Tend, useMPI=useMPI, num_procs=num_procs)
    return values, errors


def get_order(values, errors, thresh=1e-16, expected_order=None):
    values = np.array(values)
    idx = np.argsort(values)
    local_orders = np.log(errors[idx][1:] / errors[idx][:-1]) / np.log(values[idx][1:] / values[idx][:-1])
    order = np.mean(local_orders[errors[idx][1:] > max([thresh, MACHINEPRECISION])])
    if expected_order is not None:
        assert np.isclose(order, expected_order, atol=0.5), f"Expected order {expected_order}, but got {order:.2f}!"
    return order


def plot_order(values, errors, ax, thresh=1e-16, color='black', expected_order=None, **kwargs):
    values = np.array(values)
    order = get_order(values, errors, thresh=thresh, expected_order=expected_order)
    ax.scatter(values, errors, color=color, **kwargs)
    ax.loglog(values, errors[0] * (values / values[0]) ** order, color=color, label=f'p={order:.2f}', **kwargs)


def plot_order_in_time(ax, thresh, useMPI=False, num_procs=1):
    problem = run_advection

    base_configs_dt = {
        'values': np.array([2.0 ** (-i) for i in [2, 3, 4, 5, 6, 7, 8, 9]]),
        'mode': 'dt',
        'ax': ax,
        'thresh': thresh,
    }

    configs_dt = {}
    configs_dt[2] = {**base_configs_dt, 'color': 'black'}
    configs_dt[3] = {**base_configs_dt, 'color': 'magenta'}
    configs_dt[4] = {**base_configs_dt, 'color': 'teal'}
    configs_dt[5] = {**base_configs_dt, 'color': 'orange'}
    # configs_dt[6] = {**base_configs_dt, 'color': 'blue'}

    for key in configs_dt.keys():
        values, errors = multiple_runs(
            problem, expected_order=key, useMPI=useMPI, **configs_dt[key], num_procs=num_procs
        )
        plot_order(
            values,
            errors,
            ax=configs_dt[key]['ax'],
            thresh=configs_dt[key]['thresh'] * 1e2,
            color=configs_dt[key]['color'],
            expected_order=key + 1,
        )
    base_configs_dt['ax'].set_xlabel(r'$\Delta t$')
    base_configs_dt['ax'].set_ylabel('local error')
    base_configs_dt['ax'].axhline(
        base_configs_dt['thresh'], color='grey', ls='--', label=rf'$\|\delta\|={{{thresh:.0e}}}$'
    )
    base_configs_dt['ax'].legend(frameon=False)


def order_in_time_different_error_bounds():
    fig, axs = plt.subplots(
        2, 2, figsize=figsize_by_journal('Springer_Numerical_Algorithms', 1.0, 1.0), sharex=True, sharey=True
    )
    threshs = [1e-6, 1e-8, 1e-10, 1e-12]

    for i in range(len(threshs)):
        ax = axs.flatten()[i]
        plot_order_in_time(ax, threshs[i])
        if i != 2:
            ax.set_ylabel('')
            ax.set_xlabel('')
    fig.suptitle('Order in time for advection problem')
    fig.tight_layout()
    fig.savefig('compression-order-time.pdf')


if __name__ == '__main__':
    order_in_time_different_error_bounds()

    # base_configs_nvars = {
    #    'values': [128, 256, 512, 1024],
    #    # 'values': np.array([2**(i) for i in [4, 5, 6, 7, 8, 9]]),
    #    'mode': 'nvars',
    # }

    # configs_nvars = {}
    # configs_nvars[2] = {**base_configs_nvars, 'color': 'black'}
    # configs_nvars[4] = {**base_configs_nvars, 'color': 'magenta'}

    # for key in configs_nvars.keys():
    #    values, errors = multiple_runs(problem, expected_order=key, **configs_nvars[key])
    #    plot_order(values, errors, axs[1], color=configs_nvars[key]['color'])

    plt.show()
