import os
from pathlib import Path


from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.visualization_tools import show_residual_across_simulation
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def main():
    """
    A simple test program to do compare PFASST with multi-step SDC
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5e-10
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

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
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

    controller_params['mssdc_jac'] = True
    controller_params_jac = controller_params.copy()
    controller_params['mssdc_jac'] = False
    controller_params_gs = controller_params.copy()

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # set up list of parallel time-steps to run PFASST/MSSDC with
    num_proc = 8

    # instantiate controllers
    controller_mssdc_jac = controller_nonMPI(
        num_procs=num_proc, controller_params=controller_params_jac, description=description_mssdc
    )
    controller_mssdc_gs = controller_nonMPI(
        num_procs=num_proc, controller_params=controller_params_gs, description=description_mssdc
    )
    controller_pfasst = controller_nonMPI(
        num_procs=num_proc, controller_params=controller_params, description=description_pfasst
    )

    # get initial values on finest level
    P = controller_mssdc_jac.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend_pfasst, stats_pfasst = controller_pfasst.run(u0=uinit, t0=t0, Tend=Tend)
    uend_mssdc_jac, stats_mssdc_jac = controller_mssdc_jac.run(u0=uinit, t0=t0, Tend=Tend)
    uend_mssdc_gs, stats_mssdc_gs = controller_mssdc_gs.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare for both runs
    uex = P.u_exact(Tend)
    err_mssdc_jac = abs(uex - uend_mssdc_jac)
    err_mssdc_gs = abs(uex - uend_mssdc_gs)
    err_pfasst = abs(uex - uend_pfasst)
    diff_jac = abs(uend_mssdc_jac - uend_pfasst)
    diff_gs = abs(uend_mssdc_gs - uend_pfasst)
    diff_jac_gs = abs(uend_mssdc_gs - uend_mssdc_jac)

    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_8_B_out.txt', 'w')

    out = 'Error PFASST: %12.8e' % err_pfasst
    f.write(out + '\n')
    print(out)
    out = 'Error parallel MSSDC: %12.8e' % err_mssdc_jac
    f.write(out + '\n')
    print(out)
    out = 'Error serial MSSDC: %12.8e' % err_mssdc_gs
    f.write(out + '\n')
    print(out)
    out = 'Diff PFASST vs. parallel MSSDC: %12.8e' % diff_jac
    f.write(out + '\n')
    print(out)
    out = 'Diff PFASST vs. serial MSSDC: %12.8e' % diff_gs
    f.write(out + '\n')
    print(out)
    out = 'Diff parallel vs. serial MSSDC: %12.8e' % diff_jac_gs
    f.write(out + '\n')
    print(out)

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts_pfasst = get_sorted(stats_pfasst, type='niter', sortby='time')
    iter_counts_mssdc_jac = get_sorted(stats_mssdc_jac, type='niter', sortby='time')
    iter_counts_mssdc_gs = get_sorted(stats_mssdc_gs, type='niter', sortby='time')

    # compute and print statistics
    for item_pfasst, item_mssdc_jac, item_mssdc_gs in zip(
        iter_counts_pfasst, iter_counts_mssdc_jac, iter_counts_mssdc_gs
    ):
        out = 'Number of iterations for time %4.2f (PFASST/parMSSDC/serMSSDC): %2i / %2i / %2i' % (
            item_pfasst[0],
            item_pfasst[1],
            item_mssdc_jac[1],
            item_mssdc_gs[1],
        )
        f.write(out + '\n')
        print(out)

    f.close()

    # call helper routine to produce residual plot
    show_residual_across_simulation(stats_mssdc_jac, 'data/step_8_residuals_mssdc_jac.png')
    show_residual_across_simulation(stats_mssdc_gs, 'data/step_8_residuals_mssdc_gs.png')

    assert os.path.isfile('data/step_8_residuals_mssdc_jac.png')
    assert os.path.isfile('data/step_8_residuals_mssdc_gs.png')
    assert diff_jac < 3.1e-10, (
        "ERROR: difference between PFASST and parallel MSSDC controller is too large, got %s" % diff_jac
    )
    assert diff_gs < 3.1e-10, (
        "ERROR: difference between PFASST and serial MSSDC controller is too large, got %s" % diff_gs
    )
    assert diff_jac_gs < 3.1e-10, (
        "ERROR: difference between parallel and serial MSSDC controller is too large, got %s" % diff_jac_gs
    )


if __name__ == "__main__":
    main()
