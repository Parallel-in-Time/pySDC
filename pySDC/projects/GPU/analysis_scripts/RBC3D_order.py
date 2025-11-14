import os
import pickle
import numpy as np
from pySDC.helpers.fieldsIO import FieldsIO
from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.implementations.problem_classes.RayleighBenard3D import RayleighBenard3D
from mpi4py import MPI
import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.GPU.analysis_scripts.RBC3D_plotting_utils import get_plotting_style, savefig

step_sizes = {
    'RBC3DG4R4Ra1e5': [8e-2, 4e-2, 2e-2, 1e-2, 5e-3],
    'RBC3DG4R4SDC23Ra1e5': [5e-3 * 2**i for i in range(8)],
    'RBC3DG4R4SDC34Ra1e5': [5e-3 * 2**i for i in range(8)],
    'RBC3DG4R4SDC44Ra1e5': [5e-3 * 2**i for i in range(8)],
    'RBC3DG4R4RKRa1e5': [5e-3 * 2**i for i in range(8)],
    'RBC3DG4R4EulerRa1e5': [5e-3 * 2**i for i in range(8)],
}
n_freefall_times = {}


def no_logging_hook(*args, **kwargs):
    return None


def get_path(args):
    config = get_config(args)
    fname = config.get_file_name()
    return f'{fname[:fname.index('dt')]}order.pickle'


def compute_errors(args, dts, Tend):
    errors = {'u': [], 'v': [], 'w': [], 'T': [], 'p': [], 'dt': []}
    prob = RayleighBenard3D(nx=4, ny=4, nz=4, comm=MPI.COMM_SELF)

    dts = np.sort(dts)[::-1]
    ref = run(args, dts[-1], Tend)
    for dt in dts[:-1]:
        u = run(args, dt, Tend)
        e = u - ref
        for comp in ['u', 'v', 'w', 'T', 'p']:
            i = prob.index(comp)
            e_comp = np.max(np.abs(e[i])) / np.max(np.abs(ref[i]))
            e_comp = MPI.COMM_WORLD.allreduce(e_comp, op=MPI.MAX)
            errors[comp].append(float(e_comp))
        errors['dt'].append(dt)

    path = get_path(args)
    if MPI.COMM_WORLD.rank == 0:
        with open(path, 'wb') as file:
            pickle.dump(errors, file)
            print(f'Saved errors to {path}', flush=True)


def plot_error_all_components(args):  # pragma: no cover
    fig, ax = plt.subplots()
    with open(get_path(args), 'rb') as file:
        errors = pickle.load(file)

    for comp in ['u', 'v', 'w', 'T', 'p']:
        e = np.array(errors[comp])
        dt = np.array(errors['dt'])
        order = np.log(e[1:] / e[:-1]) / np.log(dt[1:] / dt[:-1])
        ax.loglog(errors['dt'], errors[comp], label=f'{comp} order {np.mean(order):.1f}')

    ax.loglog(errors['dt'], np.array(errors['dt']) ** 4, label='Order 4', ls='--')
    ax.loglog(errors['dt'], np.array(errors['dt']) ** 2, label='Order 2', ls='--')
    ax.legend()
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$e$')


def compare_order(Ra):  # pragma: no cover
    fig, ax = plt.subplots(figsize=figsize_by_journal('Nature_CS', 1, 0.6))
    if Ra == 1e5:
        names = ['RK', 'Euler', 'SDC23', 'SDC34', 'SDC44'][::-1]
        configs = [f'RBC3DG4R4{me}Ra1e5' for me in names]
        paths = [f'./data/RBC3DG4R4{me}Ra1e5-res-1-order.pickle' for me in names]

    else:
        raise NotImplementedError

    for path, config in zip(paths, configs, strict=True):
        with open(path, 'rb') as file:
            errors = pickle.load(file)

        e = np.array(errors['T'])
        dt = np.array(errors['dt'])
        order = np.log(e[1:] / e[:-1]) / np.log(dt[1:] / dt[:-1])
        print(f'{config}: order: mean={np.mean(order):.1f} / median={np.median(order):.1f}')
        ax.loglog(dt, e, **get_plotting_style(config))

    for _dt in dt:
        for i in [1, 3, 4]:
            ax.text(_dt, _dt**i, i, fontweight='bold', fontsize=14, ha='center', va='center')
            ax.loglog(dt, dt**i, ls=':', color='black')

    ax.legend(frameon=False)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$e$')
    savefig(fig, 'RBC3D_order_Ra1e5')


def run(args, dt, Tend):
    from pySDC.projects.GPU.run_experiment import run_experiment
    from pySDC.core.errors import ConvergenceError

    args['mode'] = 'run'
    args['dt'] = dt

    config = get_config(args)
    config.Tend = n_freefall_times.get(type(config).__name__, 3)

    desc = config.get_description(res=args['res'])
    prob = desc['problem_class'](**desc['problem_params'])

    ic_config_name = type(config).__name__
    for name in ['RK', 'Euler', 'O3', 'O4', 'SDC23', 'SDC34', 'SDC44']:
        ic_config_name = ic_config_name.replace(name, 'SDC34')

    ic_config = get_config({**args, 'config': ic_config_name})
    config.ic_config['config'] = type(ic_config)
    config.ic_config['res'] = ic_config.res
    config.ic_config['dt'] = ic_config.dt

    config.get_LogToFile = no_logging_hook
    config.Tend = Tend

    u_hat = run_experiment(args, config)
    u = prob.itransform(u_hat)

    return u


if __name__ == '__main__':
    from pySDC.projects.GPU.run_experiment import parse_args

    args = parse_args()

    if args['mode'] == 'run':
        config = get_config(args)
        dts = step_sizes[type(config).__name__]
        compute_errors(args, dts, max(dts))

    if args['config'] is not None:
        plot_error_all_components(args)

    compare_order(1e5)
    if MPI.COMM_WORLD.rank == 0:
        plt.show()
