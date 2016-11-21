import numpy as np

import projects.node_failure.emulate_hard_faults as ft
from projects.node_failure.allinclusive_classic_nonMPI_hard_faults import allinclusive_classic_nonMPI_hard_faults
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott


# noinspection PyShadowingNames,PyShadowingBuiltins
def main(ft_strategies):
    """
    This routine generates the heatmaps showing the residual for node failures at different steps and iterations
    """

    num_procs = 32

    # setup parameters "in time"
    t0 = 0
    dt = 2.0
    Tend = 1280.0
    Nsteps = int((Tend - t0) / dt)

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-07
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['finter'] = True

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = 'LU'

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20

    # initialize problem parameters
    problem_params = dict()
    # problem_params['Du'] = 1.0
    # problem_params['Dv'] = 0.01
    # problem_params['A'] = 0.01
    # problem_params['B'] = 0.10
    # splitting pulses until steady state
    # problem_params['Du'] = 1.0
    # problem_params['Dv'] = 0.01
    # problem_params['A'] = 0.02
    # problem_params['B'] = 0.079
    # splitting pulses until steady state
    problem_params['Du'] = 1.0
    problem_params['Dv'] = 0.01
    problem_params['A'] = 0.09
    problem_params['B'] = 0.086

    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['c_nvars'] = [256]
    problem_params['family'] = 'CG'
    problem_params['order'] = [4]
    problem_params['refinements'] = [1, 0]

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = fenics_grayscott  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = fenics_mesh  # pass data type for u
    description['dtype_f'] = fenics_mesh  # pass data type for f
    description['sweeper_class'] = generic_LU  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_fenics  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    ft.hard_random = 0.03

    controller = allinclusive_classic_nonMPI_hard_faults(num_procs=num_procs,
                                                         controller_params=controller_params,
                                                         description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    for strategy in ft_strategies:

        print('------------------------------------------ working on strategy ', strategy)
        ft.strategy = strategy

        # read in reference data from clean run, will provide reproducable locations for faults
        if strategy is not 'NOFAULT':
            reffile = np.load('data/PFASST_GRAYSCOTT_stats_hf_NOFAULT_P16.npz')
            ft.refdata = reffile['hard_stats']

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # get residuals of the run
        extract_stats = filter_stats(stats, type='residual_post_iteration')

        # find boundaries for x-,y- and c-axis as well as arrays
        maxprocs = 0
        maxiter = 0
        minres = 0
        maxres = -99
        for k, v in extract_stats.items():
            maxprocs = max(maxprocs, getattr(k, 'process'))
            maxiter = max(maxiter, getattr(k, 'iter'))
            minres = min(minres, np.log10(v))
            maxres = max(maxres, np.log10(v))

        # grep residuals and put into array
        residual = np.zeros((maxiter, maxprocs + 1))
        residual[:] = -99
        for k, v in extract_stats.items():
            step = getattr(k, 'process')
            iter = getattr(k, 'iter')
            if iter is not -1:
                residual[iter - 1, step] = np.log10(v)

        # stats magic: get niter (probably redundant with maxiter)
        extract_stats = filter_stats(stats, level=-1, type='niter')
        sortedlist_stats = sort_stats(extract_stats, sortby='process')
        iter_count = np.zeros(Nsteps)
        for item in sortedlist_stats:
            iter_count[item[0]] = item[1]
        print(iter_count)

        np.savez('data/PFASST_GRAYSCOTT_stats_hf_' + ft.strategy + '_P' + str(num_procs), residual=residual,
                 iter_count=iter_count, hard_stats=ft.hard_stats)


if __name__ == "__main__":
    # ft_strategies = ['SPREAD', 'SPREAD_PREDICT', 'INTERP', 'INTERP_PREDICT']
    ft_strategies = ['NOFAULT']

    main(ft_strategies=ft_strategies)
