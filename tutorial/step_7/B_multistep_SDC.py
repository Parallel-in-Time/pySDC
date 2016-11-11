import os

from implementations.problem_classes.HeatEquation_1D_FD import heat1d
from implementations.datatype_classes.mesh import mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.generic_LU import generic_LU
from implementations.transfer_classes.TransferMesh import mesh_to_mesh
from implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats
from pySDC.plugins.visualization_tools import show_residual_across_simulation


def main():
    """
    A simple test program to do compare PFASST with multi-step SDC
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5E-10
    level_params['dt'] = 0.125

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 2  # frequency for the test value

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 40
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = mesh  # pass data type for f
    description['sweeper_class'] = generic_LU  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set up parameters for PFASST run
    problem_params['nvars'] = [63, 31]
    description['problem_params'] = problem_params.copy()
    description_pfasst = description.copy()

    # set up parameters for MSSDC run
    problem_params['nvars'] = [63]
    description['problem_params'] = problem_params.copy()
    description_mssdc = description.copy()

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # set up list of parallel time-steps to run PFASST/MSSDC with
    num_proc = 8

    # instantiate controllers
    controller_mssdc = allinclusive_classic_nonMPI(num_procs=num_proc, controller_params=controller_params,
                                                   description=description_mssdc)
    controller_pfasst = allinclusive_classic_nonMPI(num_procs=num_proc, controller_params=controller_params,
                                                    description=description_pfasst)

    # get initial values on finest level
    P = controller_mssdc.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend_pfasst, stats_pfasst = controller_pfasst.run(u0=uinit, t0=t0, Tend=Tend)
    uend_mssdc, stats_mssdc = controller_mssdc.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare for both runs
    uex = P.u_exact(Tend)
    err_mssdc = abs(uex - uend_mssdc)
    err_pfasst = abs(uex - uend_pfasst)
    diff = abs(uend_mssdc - uend_pfasst)

    f = open('step_7_B_out.txt', 'w')

    out = 'Error PFASST: %12.8e' % err_pfasst
    f.write(out + '\n')
    print(out)
    out = 'Error MSSDC: %12.8e' % err_mssdc
    f.write(out + '\n')
    print(out)
    out = 'Diff: %12.8e' % diff
    f.write(out + '\n')
    print(out)

    # filter statistics by type (number of iterations)
    filtered_stats_pfasst = filter_stats(stats_pfasst, type='niter')
    filtered_stats_mssdc = filter_stats(stats_mssdc, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts_pfasst = sort_stats(filtered_stats_pfasst, sortby='time')
    iter_counts_mssdc = sort_stats(filtered_stats_mssdc, sortby='time')

    # compute and print statistics
    for item_pfasst, item_mssdc in zip(iter_counts_pfasst, iter_counts_mssdc):
        out = 'Number of iterations for time %4.2f (PFASST/MSSDC): %1i / %1i' % \
              (item_pfasst[0], item_pfasst[1], item_mssdc[1])
        f.write(out + '\n')
        print(out)

    f.close()

    # call helper routine to produce residual plot
    show_residual_across_simulation(stats_mssdc, 'step_7_residuals_mssdc.png')

    assert os.path.isfile('step_7_residuals_mssdc.png')
    assert diff < 2.66E-09, "ERROR: difference between PFASST and MSSDC controller is too large, got %s" % diff


if __name__ == "__main__":
    main()
