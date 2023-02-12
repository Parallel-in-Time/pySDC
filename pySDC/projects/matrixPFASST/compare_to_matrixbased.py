import numpy as np
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh as mesh_to_mesh_nocoarse
from pySDC.projects.matrixPFASST.controller_matrix_nonMPI import controller_matrix_nonMPI


def diffusion_setup(par=0.0):
    """
    Setup routine for advection test

    Args:
        par (float): parameter for controlling stiffness
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 0.25
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = par  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [127, 63]  # number of degrees of freedom for each level
    problem_params['bc'] = 'dirichlet-zero'  # boundary conditions

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
    controller_params['all_to_done'] = True

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_unforced  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    return description, controller_params


def advection_setup(par=0.0):
    """
    Setup routine for advection test

    Args:
        par (float): parameter for controlling stiffness
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 0.25
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['c'] = par
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = [128, 64]  # number of degrees of freedom for each level
    problem_params['order'] = 2
    problem_params['stencil_type'] = 'center'
    problem_params['bc'] = 'periodic'  # boundary conditions

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
    controller_params['all_to_done'] = True

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = advectionNd  # pass problem class
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    return description, controller_params


def testequation_setup():
    """
    Setup routine for the test equation

    Args:
        par (float): parameter for controlling stiffness
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 0.25
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3, 2]
    sweeper_params['QI'] = 'LU'
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['u0'] = 1.0  # initial value (for all instances)
    # use single values like this...
    # problem_params['lambdas'] = [[-1.0]]
    # .. or a list of values like this ...
    # problem_params['lambdas'] = [[-1.0, -2.0, 1j, -1j]]
    # .. or a whole block of values like this
    ilim_left = -11
    ilim_right = 0
    rlim_left = 0
    rlim_right = 11
    ilam = 1j * np.logspace(ilim_left, ilim_right, 11)
    rlam = -1 * np.logspace(rlim_left, rlim_right, 11)
    lambdas = []
    for rl in rlam:
        for il in ilam:
            lambdas.append(rl + il)
    problem_params['lambdas'] = [lambdas]
    # note: PFASST will do all of those at once, but without interaction (realized via diagonal matrix).
    # The propagation matrix will be diagonal too, corresponding to the respective lambda value.

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['all_to_done'] = True

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = testequation0d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_nocoarse  # pass spatial transfer class
    description['space_transfer_params'] = dict()  # pass paramters for spatial transfer

    return description, controller_params


def compare_controllers(type=None, par=0.0, f=None):
    """
    A simple test program to compare PFASST runs with matrix-based and matrix-free controllers

    Args:
        type (str): setup type
        par (float) parameter for controlling stiffness
        f: file handler
    """

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    if type == 'diffusion':
        description, controller_params = diffusion_setup(par)
    elif type == 'advection':
        description, controller_params = advection_setup(par)
    elif type == 'testequation':
        description, controller_params = testequation_setup()
    else:
        raise ValueError('No valis setup type provided, aborting..')

    out = '\nWorking with %s setup and parameter %3.1e..' % (type, par)
    f.write(out + '\n')
    print(out)

    # instantiate controller
    controller_mat = controller_matrix_nonMPI(num_procs=4, controller_params=controller_params, description=description)

    controller_nomat = controller_nonMPI(num_procs=4, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller_nomat.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uex = P.u_exact(Tend)

    # this is where the iteration is happening
    uend_mat, stats_mat = controller_mat.run(u0=uinit, t0=t0, Tend=Tend)
    uend_nomat, stats_nomat = controller_nomat.run(u0=uinit, t0=t0, Tend=Tend)

    diff = abs(uend_mat - uend_nomat)

    err_mat = abs(uend_mat - uex)
    err_nomat = abs(uend_nomat - uex)

    out = '  Error (mat/nomat) vs. exact solution: %6.4e -- %6.4e' % (err_mat, err_nomat)
    f.write(out + '\n')
    print(out)
    out = '  Difference between both results: %6.4e' % diff
    f.write(out + '\n')
    print(out)

    assert diff < 2.3e-15, 'ERROR: difference between matrix-based and matrix-free result is too large, got %s' % diff

    # get and convert statistics to list of iterations count, sorted by process
    iter_counts_mat = get_sorted(stats_mat, type='niter', sortby='time')
    iter_counts_nomat = get_sorted(stats_nomat, type='niter', sortby='time')

    out = '  Iteration counts for matrix-based version: %s' % iter_counts_mat
    f.write(out + '\n')
    print(out)
    out = '  Iteration counts for matrix-free version: %s' % iter_counts_nomat
    f.write(out + '\n')
    print(out)

    assert (
        iter_counts_nomat == iter_counts_mat
    ), 'ERROR: number of iterations differ between matrix-based and matrix-free controller'


def main():
    par_list = [1e-02, 1.0, 1e02]

    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/comparison_matrix_vs_nomat_detail.txt', 'w')
    for par in par_list:
        compare_controllers(type='diffusion', par=par, f=f)
        compare_controllers(type='advection', par=par, f=f)
    compare_controllers(type='testequation', par=0.0, f=f)
    f.close()


if __name__ == "__main__":
    main()
