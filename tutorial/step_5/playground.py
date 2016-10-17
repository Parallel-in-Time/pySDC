from pySDC.Step import step

from pySDC.implementations.problem_classes.AdvectionEquation_1D_FD import advection1d
from pySDC.implementations.problem_classes.VorticityVelocity_2D_FD import vortex2d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.collocation_classes.gauss_legendre import CollGaussLegendre
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh_1D import mesh_to_mesh_1d_periodic
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats


def main():
    """
    A simple test program to setup a full step hierarchy
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = 'GS'

    # initialize problem parameters
    problem_params = {}
    problem_params['c'] = 1.0  # advection coefficient
    problem_params['freq'] = 2  # frequency for the test value
    problem_params['nvars'] = [(4,4)]  # number of degrees of freedom for each level
    problem_params['order'] = [4]
    problem_params['type'] = 'center'

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    controller_params = {}
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = vortex2d                      # pass problem class
    description['problem_params'] = problem_params                  # pass problem parameters
    description['dtype_u'] = mesh                                   # pass data type for u
    description['dtype_f'] = mesh                                   # pass data type for f
    description['sweeper_class'] = generic_implicit                 # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params                  # pass sweeper parameters
    description['level_params'] = level_params                      # pass level parameters
    description['step_params'] = step_params                        # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_1d_periodic  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    controller = allinclusive_classic_nonMPI(num_procs=10, controller_params=controller_params, description=description)




    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)
    print('Error: %12.8e' % (err))

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by time
    iter_counts = sort_stats(filtered_stats, sortby='process')

    for item in iter_counts:
        print('Number of iterations for process %2i: %2i' % item)

if __name__ == "__main__":
    main()
