import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.matrixPFASST.allinclusive_matrix_nonMPI import allinclusive_matrix_nonMPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def diffusion_setup(mu=0.0):
    """
    Setup routine for advection test

    Args:
        mu (float): parameter for controlling stiffness
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 0.25
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['spread'] = True

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = mu  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [127, 63]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = mesh  # pass data type for f
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    return description, controller_params


def advection_setup(mu=0.0):
    """
    Setup routine for advection test

    Args:
        mu (float): parameter for controlling stiffness
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 0.25
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['spread'] = True

    # initialize problem parameters
    problem_params = dict()
    problem_params['c'] = mu
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [128, 64]  # number of degrees of freedom for each level
    problem_params['order'] = 2
    problem_params['type'] = 'center'

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2
    space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = advection1d  # pass problem class
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = mesh  # pass data type for f
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    return description, controller_params


def compare_controllers(type=None, mu=0.0, f=None):
    """
    A simple test program to compare PFASST runs with matrix-based and matrix-free controllers

    Args:
        type (str): setup type
        mu (float) parameter for controlling stiffness
        f: file handler
    """

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    if type == 'diffusion':
        description, controller_params = diffusion_setup(mu)
    elif type == 'advection':
        description, controller_params = advection_setup(mu)
    else:
        raise ValueError('No valis setup type provided, aborting..')

    out = '\nWorking with %s setup and parameter %3.1e..' % (type, mu)
    f.write(out + '\n')
    print(out)

    # instantiate controller
    controller = allinclusive_matrix_nonMPI(num_procs=4, controller_params=controller_params,
                                            description=description)
    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uex = P.u_exact(Tend)

    # this is where the iteration is happening
    uend_mat, stats_mat = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats_mat = filter_stats(stats_mat, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts_mat = sort_stats(filtered_stats_mat, sortby='time')

    out = '  Iteration counts for matrix-based version: %s' % iter_counts_mat
    f.write(out + '\n')
    print(out)

    # filter only iteration counts and check for equality
    niters = [item[1] for item in iter_counts_mat]
    assert niters.count(niters[0]) == len(niters), 'ERROR: not all time-steps have the same number of iterations'
    niter = niters[0]

    # build propagation matrix using the prescribed number of iterations (or any other, if needed)
    prop = controller.build_propagation_matrix(niter=niter)

    err_prop_ex = np.linalg.norm(prop.dot(uinit.values) - uex.values)
    out = '  Difference between propagation and exact solution: %6.4e' % err_prop_ex
    f.write(out + '\n')
    print(out)
    err_mat_ex = np.linalg.norm(uend_mat.values - uex.values)
    out = '  Difference between matrix-PFASST and exact solution: %6.4e' % err_mat_ex
    f.write(out + '\n')
    print(out)
    err_mat_prop = np.linalg.norm(prop.dot(uinit.values) - uend_mat.values)
    out = '  Difference between matrix-PFASST and propagator: %6.4e' % err_mat_prop
    f.write(out + '\n')
    print(out)


def main():

    mu_list = [1E-02, 1.0, 1E+02]

    f = open('comparison_matrix_vs_propagator_detail.txt', 'a')
    for mu in mu_list:
        compare_controllers(type='diffusion', mu=mu, f=f)
        compare_controllers(type='advection', mu=mu, f=f)
    f.close()


if __name__ == "__main__":
    main()
