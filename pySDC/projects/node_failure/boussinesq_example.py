import numpy as np

import pySDC.projects.node_failure.emulate_hard_faults as ft
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.Boussinesq_2D_FD_imex import boussinesq_2d_imex
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh
from pySDC.projects.node_failure.controller_nonMPI_hard_faults import controller_nonMPI_hard_faults


# noinspection PyShadowingNames,PyShadowingBuiltins
def main(ft_strategies):
    """
    This routine generates the heatmaps showing the residual for node failures at different steps and iterations
    """

    num_procs = 16

    # setup parameters "in time"
    t0 = 0
    Tend = 960
    Nsteps = 320
    dt = Tend / float(Nsteps)

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-06
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['finter'] = True
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

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
    # problem_params['nvars'] = [(4, 450, 30), (4, 450, 30)]
    problem_params['nvars'] = [(4, 100, 10), (4, 100, 10)]
    problem_params['u_adv'] = 0.02
    problem_params['c_s'] = 0.3
    problem_params['Nfreq'] = 0.01
    problem_params['x_bounds'] = [(-150.0, 150.0)]
    problem_params['z_bounds'] = [(0.0, 10.0)]
    problem_params['order'] = [4, 2]
    problem_params['order_upw'] = [5, 1]
    problem_params['gmres_maxiter'] = [50, 50]
    problem_params['gmres_restart'] = [10, 10]
    problem_params['gmres_tol_limit'] = [1e-10, 1e-10]

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = boussinesq_2d_imex  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    ft.hard_random = 0.03

    controller = controller_nonMPI_hard_faults(
        num_procs=num_procs, controller_params=controller_params, description=description
    )

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    cfl_advection = P.params.u_adv * dt / P.h[0]
    cfl_acoustic_hor = P.params.c_s * dt / P.h[0]
    cfl_acoustic_ver = P.params.c_s * dt / P.h[1]
    print("CFL number of advection: %4.2f" % cfl_advection)
    print("CFL number of acoustics (horizontal): %4.2f" % cfl_acoustic_hor)
    print("CFL number of acoustics (vertical):   %4.2f" % cfl_acoustic_ver)

    for strategy in ft_strategies:

        print('------------------------------------------ working on strategy ', strategy)
        ft.strategy = strategy

        # read in reference data from clean run, will provide reproducable locations for faults
        if strategy != 'NOFAULT':
            reffile = np.load('data/PFASST_BOUSSINESQ_stats_hf_NOFAULT_P16.npz')
            ft.refdata = reffile['hard_stats']

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        P.report_log()

        # get residuals of the run
        extract_stats = filter_stats(stats, type='residual_post_iteration')

        # find boundaries for x-,y- and c-axis as well as arrays
        maxprocs = 0
        maxiter = 0
        minres = 0
        maxres = -99
        for k, v in extract_stats.items():
            maxprocs = max(maxprocs, k.process)
            maxiter = max(maxiter, k.iter)
            minres = min(minres, np.log10(v))
            maxres = max(maxres, np.log10(v))

        # grep residuals and put into array
        residual = np.zeros((maxiter, maxprocs + 1))
        residual[:] = -99
        for k, v in extract_stats.items():
            step = k.process
            iter = k.iter
            if iter != -1:
                residual[iter - 1, step] = np.log10(v)

        # stats magic: get niter (probably redundant with maxiter)
        extract_stats = filter_stats(stats, level=-1, type='niter')
        sortedlist_stats = sort_stats(extract_stats, sortby='process')
        iter_count = np.zeros(Nsteps)
        for item in sortedlist_stats:
            iter_count[item[0]] = item[1]
        print(iter_count)

        np.savez(
            'data/PFASST_BOUSSINESQ_stats_hf_' + ft.strategy + '_P' + str(num_procs),
            residual=residual,
            iter_count=iter_count,
            hard_stats=ft.hard_stats,
        )


if __name__ == "__main__":
    # ft_strategies = ['SPREAD', 'SPREAD_PREDICT', 'INTERP', 'INTERP_PREDICT']
    ft_strategies = ['NOFAULT']

    main(ft_strategies=ft_strategies)
