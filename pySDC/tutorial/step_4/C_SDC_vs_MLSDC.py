from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def main():
    """
    A simple test program to compare SDC and MLSDC
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-09
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params_sdc = dict()
    sweeper_params_sdc['node_type'] = 'LEGENDRE'
    sweeper_params_sdc['quad_type'] = 'RADAU-RIGHT'
    sweeper_params_sdc['num_nodes'] = 5
    sweeper_params_sdc['QI'] = 'LU'

    sweeper_params_mlsdc = dict()
    sweeper_params_mlsdc['node_type'] = 'LEGENDRE'
    sweeper_params_mlsdc['quad_type'] = 'RADAU-RIGHT'
    sweeper_params_mlsdc['num_nodes'] = [5, 3, 2]
    sweeper_params_mlsdc['QI'] = 'LU'

    # initialize problem parameters
    problem_params_sdc = dict()
    problem_params_sdc['nu'] = 0.1  # diffusion coefficient
    problem_params_sdc['freq'] = 4  # frequency for the test value
    problem_params_sdc['nvars'] = 1023  # number of degrees of freedom for each level
    problem_params_sdc['bc'] = 'dirichlet-zero'  # boundary conditions

    problem_params_mlsdc = dict()
    problem_params_mlsdc['nu'] = 0.1  # diffusion coefficient
    problem_params_mlsdc['freq'] = 4  # frequency for the test value
    problem_params_mlsdc['nvars'] = [1023, 511, 255]  # number of degrees of freedom for each level
    problem_params_mlsdc['bc'] = 'dirichlet-zero'  # boundary conditions

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for SDC
    description_sdc = dict()
    description_sdc['problem_class'] = heatNd_unforced
    description_sdc['problem_params'] = problem_params_sdc
    description_sdc['sweeper_class'] = generic_implicit
    description_sdc['sweeper_params'] = sweeper_params_sdc
    description_sdc['level_params'] = level_params
    description_sdc['step_params'] = step_params

    # fill description dictionary for MLSDC
    description_mlsdc = dict()
    description_mlsdc['problem_class'] = heatNd_unforced
    description_mlsdc['problem_params'] = problem_params_mlsdc
    description_mlsdc['sweeper_class'] = generic_implicit
    description_mlsdc['sweeper_params'] = sweeper_params_mlsdc
    description_mlsdc['level_params'] = level_params
    description_mlsdc['step_params'] = step_params
    description_mlsdc['space_transfer_class'] = mesh_to_mesh
    description_mlsdc['space_transfer_params'] = space_transfer_params

    # instantiate the controller (no controller parameters used here)
    controller_sdc = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description_sdc)
    controller_mlsdc = controller_nonMPI(
        num_procs=1, controller_params=controller_params, description=description_mlsdc
    )

    # set time parameters
    t0 = 0.0
    Tend = 0.1

    # get initial values on finest level
    P = controller_sdc.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend_sdc, stats_sdc = controller_sdc.run(u0=uinit, t0=t0, Tend=Tend)
    uend_mlsdc, stats_mlsdc = controller_mlsdc.run(u0=uinit, t0=t0, Tend=Tend)

    # get number of iterations for both
    niter_sdc = get_sorted(stats_sdc, type='niter', sortby='time')[0][1]
    niter_mlsdc = get_sorted(stats_mlsdc, type='niter', sortby='time')[0][1]

    # compute exact solution and compare both
    uex = P.u_exact(Tend)
    err_sdc = abs(uex - uend_sdc)
    err_mlsdc = abs(uex - uend_mlsdc)
    diff = abs(uend_mlsdc - uend_sdc)

    # print out and check
    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_4_C_out.txt', 'a')
    out = 'Error SDC and MLSDC: %12.8e -- %12.8e' % (err_sdc, err_mlsdc)
    f.write(out + '\n')
    print(out)
    out = 'Difference SDC vs. MLSDC: %12.8e' % diff
    f.write(out + '\n')
    print(out)
    out = 'Number of iterations SDC and MLSDC: %2i -- %2i' % (niter_sdc, niter_mlsdc)
    f.write(out + '\n')
    print(out)

    assert diff < 6e-10, "ERROR: difference between MLSDC and SDC is higher than expected, got %s" % diff
    assert niter_sdc - niter_mlsdc <= 6, "ERROR: MLSDC required more iterations than expected, got %s" % niter_mlsdc


if __name__ == "__main__":
    main()
