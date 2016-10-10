
from implementations.problem_classes.HeatEquation_1D_FD import heat1d
from implementations.datatype_classes.mesh import mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.generic_LU import generic_LU
from implementations.transfer_classes.TransferMesh_1D import mesh_to_mesh_1d_dirichlet
from implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.Plugins.stats_helper import filter_stats, sort_stats

def main():
    """
    A simple test program to setup a full step hierarchy
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-09
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params_sdc = {}
    sweeper_params_sdc['collocation_class'] = CollGaussRadau_Right
    sweeper_params_sdc['num_nodes'] = 5

    sweeper_params_mlsdc = {}
    sweeper_params_mlsdc['collocation_class'] = CollGaussRadau_Right
    sweeper_params_mlsdc['num_nodes'] = [5, 3, 2]

    # initialize problem parameters
    problem_params_sdc = {}
    problem_params_sdc['nu'] = 0.1  # diffusion coefficient
    problem_params_sdc['freq'] = 4  # frequency for the test value
    problem_params_sdc['nvars'] = 1023  # number of degrees of freedom for each level

    problem_params_mlsdc = {}
    problem_params_mlsdc['nu'] = 0.1  # diffusion coefficient
    problem_params_mlsdc['freq'] = 4  # frequency for the test value
    problem_params_mlsdc['nvars'] = [1023, 511, 255]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30

    # fill description dictionary for SDC
    description_sdc = {}
    description_sdc['problem_class'] = heat1d                           # pass problem class
    description_sdc['problem_params'] = problem_params_sdc              # pass problem parameters
    description_sdc['dtype_u'] = mesh                                   # pass data type for u
    description_sdc['dtype_f'] = mesh                                   # pass data type for f
    description_sdc['sweeper_class'] = generic_LU                       # pass sweeper (see part B)
    description_sdc['sweeper_params'] = sweeper_params_sdc              # pass sweeper parameters
    description_sdc['level_params'] = level_params                      # pass level parameters
    description_sdc['step_params'] = step_params                        # pass step parameters

    # fill description dictionary for MLSDC
    description_mlsdc = {}
    description_mlsdc['problem_class'] = heat1d                           # pass problem class
    description_mlsdc['problem_params'] = problem_params_mlsdc            # pass problem parameters
    description_mlsdc['dtype_u'] = mesh                                   # pass data type for u
    description_mlsdc['dtype_f'] = mesh                                   # pass data type for f
    description_mlsdc['sweeper_class'] = generic_LU                       # pass sweeper (see part B)
    description_mlsdc['sweeper_params'] = sweeper_params_mlsdc            # pass sweeper parameters
    description_mlsdc['level_params'] = level_params                      # pass level parameters
    description_mlsdc['step_params'] = step_params                        # pass step parameters
    description_mlsdc['space_transfer_class'] = mesh_to_mesh_1d_dirichlet # pass spatial transfer class
    description_mlsdc['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    # instantiate the controller (no controller parameters used here)
    controller_sdc = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description_sdc)
    controller_mlsdc = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description_mlsdc)

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
    filtered_stats_sdc = filter_stats(stats_sdc, type='niter')
    filtered_stats_mlsdc = filter_stats(stats_mlsdc, type='niter')
    niter_sdc = sort_stats(filtered_stats_sdc,'time')[0][1]
    niter_mlsdc = sort_stats(filtered_stats_mlsdc,'time')[0][1]

    # compute exact solution and compare both
    uex = P.u_exact(Tend)
    err_sdc = abs(uex - uend_sdc)
    err_mlsdc = abs(uex - uend_mlsdc)
    diff = abs(uend_mlsdc-uend_sdc)
    print('Error SDC and MLSDC: %12.8e -- %12.8e' % (err_sdc, err_mlsdc))
    print('Difference SDC vs. MLSDC: %12.8e' % (diff))
    print('Number of iterations SDC and MLSDC: %2i -- %2i' %(niter_sdc, niter_mlsdc))

    assert diff < 6E-10, "ERROR: difference between MLSDC and SDC is higher than expected, got %s" %diff
    assert niter_sdc-niter_mlsdc >= 6, "ERROR: MLSDC required more iterations than expected, got %s" %niter_mlsdc


if __name__ == "__main__":
    main()
