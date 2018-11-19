from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.helpers.stats_helper import filter_stats, sort_stats, get_list_of_types
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


def main():
    """
    A simple test program to describe how to get statistics of a run
    """

    # run simulation
    stats = run_simulation()

    f = open('step_3_A_out.txt', 'w')
    out = 'List of registered statistic types: %s' % get_list_of_types(stats)
    f.write(out + '\n')
    print(out)

    # filter statistics by first time intervall and type (residual)
    filtered_stats = filter_stats(stats, time=0.1, type='residual_post_iteration')

    # sort and convert stats to list, sorted by iteration numbers
    residuals = sort_stats(filtered_stats, sortby='iter')

    for item in residuals:
        out = 'Residual in iteration %2i: %8.4e' % item
        f.write(out + '\n')
        print(out)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by time
    iter_counts = sort_stats(filtered_stats, sortby='time')

    for item in iter_counts:
        out = 'Number of iterations at time %4.2f: %2i' % item
        f.write(out + '\n')
        print(out)

    f.close()

    assert all([item[1] == 12 for item in iter_counts]), \
        'ERROR: number of iterations are not as expected, got %s' % iter_counts


def run_simulation():
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 1023  # number of degrees of freedom

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters (<-- this is new!)
    controller_params = dict()
    controller_params['logger_level'] = 30  # reduce verbosity of each run

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = heat1d_forced
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.1
    Tend = 0.9

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


if __name__ == "__main__":
    main()
