from argparse import ArgumentParser
import json
import glob
import numpy as np
from mpi4py import MPI

import pySDC.helpers.plot_helper as plt_helper
import matplotlib.ticker as ticker

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex, allencahn_imex_timeforcing
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft

from pySDC.projects.AllenCahn_Bayreuth.AllenCahn_monitor import monitor


def run_simulation(name='', spectral=None, nprocs_space=None):
    """
    A test program to do PFASST runs for the AC equation with different forcing

    Args:
        name (str): name of the run, will be used to distinguish different setups
        spectral (bool): run in real or spectral space
        nprocs_space (int): number of processors in space (None if serial)
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # split world communicator to create space-communicators
    if nprocs_space is not None:
        color = int(world_rank / nprocs_space)
    else:
        color = int(world_rank / 1)
    space_comm = comm.Split(color=color)
    space_rank = space_comm.Get_rank()
    space_size = space_comm.Get_size()

    assert world_size == space_size, 'This script cannot run parallel-in-time with MPI, only spatial parallelism'

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 1e-03
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['L'] = 1.0
    problem_params['nvars'] = [(128, 128), (32, 32)]
    problem_params['eps'] = [0.04]
    problem_params['radius'] = 0.25
    problem_params['comm'] = space_comm
    problem_params['name'] = name
    problem_params['init_type'] = 'circle'
    problem_params['spectral'] = spectral

    if name == 'AC-test-constforce':
        problem_params['dw'] = [-23.59]

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30 if space_rank == 0 else 99  # set level depending on rank
    controller_params['hook_class'] = monitor
    controller_params['predict_type'] = 'pfasst_burnin'

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = fft_to_fft

    if name == 'AC-test-noforce' or name == 'AC-test-constforce':
        description['problem_class'] = allencahn_imex
    elif name == 'AC-test-timeforce':
        description['problem_class'] = allencahn_imex_timeforcing
    else:
        raise NotImplementedError(f'{name} is not implemented')

    # set time parameters
    t0 = 0.0
    Tend = 32 * 0.001

    if space_rank == 0:
        out = f'---------> Running {name} with spectral={spectral} and {space_size} process(es) in space...'
        print(out)

    # instantiate controller
    controller = controller_nonMPI(num_procs=8, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    if space_rank == 0:
        # convert filtered statistics to list of computed radii, sorted by time
        computed_radii = get_sorted(stats, type='computed_radius', sortby='time')
        exact_radii = get_sorted(stats, type='exact_radius', sortby='time')
        computed_vol = get_sorted(stats, type='computed_volume', sortby='time')
        exact_vol = get_sorted(stats, type='exact_volume', sortby='time')

        # print and store radii and error over time
        err_test = 0.0
        results = dict()
        for cr, er, cv, ev in zip(computed_radii, exact_radii, computed_vol, exact_vol):
            if name == 'AC-test-noforce':
                exrad = er[1]
                exvol = ev[1]
            else:
                exrad = computed_radii[0][1]
                exvol = computed_vol[0][1]
            if exrad > 0:
                errr = abs(cr[1] - exrad) / exrad
                errv = abs(cv[1] - exvol) / exvol
            else:
                errr = 1.0
                errv = 1.0
            if cr[0] == 0.025:
                err_test = errr
            out = f'Computed/exact/error radius for time {cr[0]:6.4f}: ' f'{cr[1]:8.6f} / {exrad:8.6f} / {errr:6.4e}'
            print(out)
            results[cr[0]] = (cr[1], exrad, errr, cv[1], exvol, errv)
        fname = f'./data/{name}_results.json'
        with open(fname, 'w') as fp:
            json.dump(results, fp, sort_keys=True, indent=4)

        print()

        # convert filtered statistics of iterations count, sorted by time
        iter_counts = get_sorted(stats, type='niter', sortby='time')
        niters = np.mean(np.array([item[1] for item in iter_counts]))
        out = f'Mean number of iterations: {niters:.4f}'
        print(out)

        # get setup time
        timing = get_sorted(stats, type='timing_setup', sortby='time')
        out = f'Setup time: {timing[0][1]:.4f} sec.'
        print(out)

        # get running time
        timing = get_sorted(stats, type='timing_run', sortby='time')
        out = f'Time to solution: {timing[0][1]:.4f} sec.'
        print(out)

        out = '...Done <---------\n'
        print(out)

        # Testing the output
        if name == 'AC-test-noforce':
            if spectral:
                exp_iters = 6.59375
                exp_err = 7.821e-02
            else:
                exp_iters = 7.8125
                exp_err = 7.85e-02
        elif name == 'AC-test-constforce':
            if spectral:
                exp_iters = 2.875
                exp_err = 4.678e-04
            else:
                exp_iters = 4.3125
                exp_err = 6.2384e-04
        elif name == 'AC-test-timeforce':
            if spectral:
                exp_iters = 1.65625
                exp_err = 6.2345e-04
            else:
                exp_iters = 2.40625
                exp_err = 6.2345e-04
        else:
            raise NotImplementedError(f'{name} is not implemented')

        assert niters == exp_iters, f'Got deviating iteration counts of {niters} instead of {exp_iters}'
        assert err_test < exp_err, f'Got deviating errors of {err_test} instead of {exp_err}'


def visualize_radii():
    """
    Routine to plot the radii of the runs vs. the exact radii
    """

    plt_helper.setup_mpl()

    filelist = glob.glob('./data/*_results.json')

    for file in filelist:
        # read in file with data
        with open(file, 'r') as fp:
            results = json.load(fp)

        print(f'Working on {file}...')

        # get times and radii
        xcoords = list(results)
        computed_radii = [v[0] for k, v in results.items()]
        exact_radii = [v[1] for k, v in results.items()]
        computed_vol = [v[3] for k, v in results.items()]
        exact_vol = [v[4] for k, v in results.items()]

        # compute bound for y-axis
        max_rad = max(max(computed_radii), max(exact_radii))
        max_vol = max(max(computed_vol), max(exact_vol))

        # set up plot for radii
        fig, ax = plt_helper.newfig(textwidth=238.96, scale=1.0)

        # and plot
        ax.plot(xcoords, computed_radii, label='Computed radius')
        ax.plot(xcoords, exact_radii, color='k', linestyle='--', linewidth=1, label='Exact radius')

        # beautify and save plot
        ax.set_ylim([-0.01, max_rad * 1.1])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
        # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.set_ylabel('radius')
        ax.set_xlabel('time')
        ax.grid()
        ax.legend(loc=3)
        # ax.set_title(file.split('/')[-1].replace('_results.json', ''))
        f = file.replace('_results.json', '_radii')
        plt_helper.savefig(f)

        # test if all went well
        assert glob.glob(f'{f}.pdf'), 'ERROR: plotting did not create PDF file'
        # assert glob.glob(f'{f}.pgf'), 'ERROR: plotting did not create PGF file'
        assert glob.glob(f'{f}.png'), 'ERROR: plotting did not create PNG file'

        # set up plot for volumes
        fig, ax = plt_helper.newfig(textwidth=238.96, scale=1.0)

        # and plot
        ax.plot(xcoords, computed_vol, label='Computed volume')
        ax.plot(xcoords, exact_vol, color='k', linestyle='--', linewidth=1, label='Exact volume')

        # beautify and save plot
        ax.set_ylim([-0.01, max_vol * 1.1])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
        # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.set_ylabel('radius')
        ax.set_xlabel('time')
        ax.grid()
        ax.legend(loc=3)
        # ax.set_title(file.split('/')[-1].replace('_results.json', ''))
        f = file.replace('_results.json', '_volume')
        plt_helper.savefig(f)

        # test if all went well
        assert glob.glob(f'{f}.pdf'), 'ERROR: plotting did not create PDF file'
        # assert glob.glob(f'{f}.pgf'), 'ERROR: plotting did not create PGF file'
        assert glob.glob(f'{f}.png'), 'ERROR: plotting did not create PNG file'


def main(nprocs_space=None):
    """
    Little helper routine to run the whole thing

    Args:
        nprocs_space (int): number of processors in space (None if serial)

    """
    name_list = ['AC-test-noforce', 'AC-test-constforce', 'AC-test-timeforce']

    for name in name_list:
        run_simulation(name=name, spectral=False, nprocs_space=nprocs_space)
        run_simulation(name=name, spectral=True, nprocs_space=nprocs_space)


if __name__ == "__main__":
    # Add parser to get number of processors in space (have to do this here to enable automatic testing)
    parser = ArgumentParser()
    parser.add_argument("-n", "--nprocs_space", help='Specifies the number of processors in space', type=int)
    args = parser.parse_args()

    main(nprocs_space=args.nprocs_space)
    visualize_radii()
