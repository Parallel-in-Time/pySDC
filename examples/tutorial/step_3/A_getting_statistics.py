from implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.Plugins.stats_helper import filter_stats, sort_stats, get_list_of_types

def main():

    stats = run_simulation()

    print('List of registered statistic types:', get_list_of_types(stats))

    # filter statistics by first time intervall and type (residual)
    filtered_stats = filter_stats(stats, time=0.1, type='residual')

    # sort and convert stats to list, sorted by iteration numbers
    residuals = sort_stats(filtered_stats, sortby='iter')

    for item in residuals:
        print('Residual in iteration %2i: %8.4e' %item)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by time
    iter_counts = sort_stats(filtered_stats, sortby='time')

    for item in iter_counts:
        print('Number of iterations at time %4.2f: %2i' %item)

    assert all([item[1] == 12 for item in iter_counts]), 'ERROR: number of iterations are not as expected, got %s' %iter_counts


def run_simulation():
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    # sweeper_params['do_LU'] = True      # for this sweeper we can use the LU trick for the implicit part!

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 1023  # number of degrees of freedom

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d_forced
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params={}, description=description)

    # set time parameters
    t0 = 0.1
    Tend = 0.9  # note that we are requesting 8 time steps here (dt is 0.1)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


if __name__ == "__main__":
    main()
