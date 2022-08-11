import numpy as np
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.AdvectionDiffusionEquation_1D_FFT import advectiondiffusion1d_implicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh_FFT import mesh_to_mesh_fft
from pySDC.playgrounds.fft.libpfasst_output import libpfasst_output


def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = 0.9 / 32
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [5]
    sweeper_params['QI'] = ['MIN2']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'
    sweeper_params['do_coll_update'] = False

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1.0  # diffusion coefficient
    problem_params['c'] = 10.0  # advection speed
    problem_params['freq'] = -1  # frequency for the test value
    problem_params['nvars'] = [256]  # number of degrees of freedom for each level
    problem_params['L'] = 1.0  # length of the interval [-L/2, L/2]

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = libpfasst_output
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    # description['problem_class'] = advectiondiffusion1d_imex # pass problem class
    description['problem_class'] = advectiondiffusion1d_implicit  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    # description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_fft  # pass spatial transfer class
    description['space_transfer_params'] = dict()  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = level_params['dt']
    num_proc = 1

    # instantiate controller
    controller = allinclusive_multigrid_nonMPI(
        num_procs=num_proc, controller_params=controller_params, description=description
    )

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # print(controller.MS[0].levels[0].sweep.coll.Qmat)
    # print(controller.MS[0].levels[0].sweep.QI)
    # exit()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

    # filter statistics by type (number of iterations)
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %2i' % item
        print(out)

    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % (int(np.argmax(niters)), int(np.argmin(niters)))
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    print(out)

    print('Error: %8.4e' % err)


if __name__ == "__main__":
    main()
