import numpy as np

import pySDC.projects.node_failure.emulate_hard_faults as ft
from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.node_failure.controller_nonMPI_hard_faults import controller_nonMPI_hard_faults


# noinspection PyShadowingNames,PyShadowingBuiltins
def main(ft_setups, ft_strategies):
    """
    This routine generates the heatmaps showing the residual for node failures at different steps and iterations
    """

    num_procs = 16

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-09

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
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    for setup in ft_setups:

        if setup == 'HEAT':

            # initialize problem parameters
            problem_params = dict()
            problem_params['nu'] = 0.5
            problem_params['freq'] = 1
            problem_params['nvars'] = [255, 127]

            level_params['dt'] = 0.5

            space_transfer_params['periodic'] = False

            # fill description dictionary for easy step instantiation
            description = dict()
            description['problem_class'] = heat1d_forced  # pass problem class
            description['problem_params'] = problem_params  # pass problem parameters
            description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
            description['sweeper_params'] = sweeper_params  # pass sweeper parameters
            description['level_params'] = level_params  # pass level parameters
            description['step_params'] = step_params  # pass step parameters
            description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
            description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

            # setup parameters "in time"
            t0 = 0.0
            Tend = 8.0

        elif setup == 'ADVECTION':

            # initialize problem parameters
            problem_params = dict()
            problem_params['c'] = 1.0
            problem_params['nvars'] = [256, 128]
            problem_params['freq'] = 2
            problem_params['order'] = 2

            level_params['dt'] = 0.125

            space_transfer_params['periodic'] = True

            # fill description dictionary for easy step instantiation
            description = dict()
            description['problem_class'] = advection1d  # pass problem class
            description['problem_params'] = problem_params  # pass problem parameters
            description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
            description['sweeper_params'] = sweeper_params  # pass sweeper parameters
            description['level_params'] = level_params  # pass level parameters
            description['step_params'] = step_params  # pass step parameters
            description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
            description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

            # setup parameters "in time"
            t0 = 0.0
            Tend = 2.0

        else:

            raise NotImplementedError('setup not implemented')

        # do a reference run without any faults to see how things would look like (and to get maxiter/ref_niter)
        ft.strategy = 'NOFAULT'

        controller = controller_nonMPI_hard_faults(
            num_procs=num_procs, controller_params=controller_params, description=description
        )

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # stats magic: get iteration counts to find maxiter/niter
        sortedlist_stats = get_sorted(stats, level=-1, type='niter', sortby='process')
        ref_niter = max([item[1] for item in sortedlist_stats])

        print('Will sweep over %i steps and %i iterations now...' % (num_procs, ref_niter))

        # loop over all strategies
        for strategy in ft_strategies:

            ft_iter = range(1, ref_niter + 1)
            ft_step = range(0, num_procs)

            print('------------------------------------------ working on strategy ', strategy)

            iter_count = np.zeros((len(ft_step), len(ft_iter)))

            # loop over all steps
            xcnt = -1
            for step in ft_step:

                xcnt += 1

                # loop over all iterations
                ycnt = -1
                for iter in ft_iter:
                    ycnt += 1

                    ft.hard_step = step
                    ft.hard_iter = iter
                    ft.strategy = strategy

                    # call main function to get things done...
                    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

                    # stats magic: get iteration counts to find maxiter/niter
                    sortedlist_stats = get_sorted(stats, level=-1, type='niter', sortby='process')
                    niter = max([item[1] for item in sortedlist_stats])
                    iter_count[xcnt, ycnt] = niter

            print(iter_count)

            np.savez(
                'data/' + setup + '_results_hf_' + strategy,
                iter_count=iter_count,
                description=description,
                ft_step=ft_step,
                ft_iter=ft_iter,
            )


if __name__ == "__main__":

    ft_strategies = ['SPREAD', 'SPREAD_PREDICT', 'INTERP', 'INTERP_PREDICT']
    ft_setups = ['ADVECTION', 'HEAT']

    # ft_strategies = ['NOFAULT']
    # ft_setups = ['HEAT']

    main(ft_setups=ft_setups, ft_strategies=ft_strategies)
