from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def main(num_proc_list=None, fname=None, multi_level=True):
    """
    A simple test program to run PFASST

    Args:
        num_proc_list: list of number of processes to test with
        fname: filename/path for output
        multi_level (bool): do multi-level run or single-level
    """

    if multi_level:
        description, controller_params, t0, Tend = set_parameters_ml()
    else:
        assert all(num_proc == 1 for num_proc in num_proc_list), \
            'ERROR: single-level run can only use 1 processor, got %s' % num_proc_list
        description, controller_params, t0, Tend = set_parameters_sl()

    f = open(fname, 'w')
    # loop over different numbers of processes
    for num_proc in num_proc_list:

        out = 'Working with %2i processes...' % num_proc
        f.write(out + '\n')
        print(out)

        # instantiate controllers
        controller = controller_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main functions to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare with both results
        uex = P.u_exact(Tend)
        err = abs(uex - uend)

        out = 'Error vs. exact solution: %12.8e' % err
        f.write(out + '\n')
        print(out)

        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        # compute and print statistics
        for item in iter_counts:
            out = 'Number of iterations for time %4.2f: %1i ' % (item[0], item[1])
            f.write(out + '\n')
            print(out)

        f.write('\n')
        print()

        assert all([item[1] <= 8 for item in iter_counts]), "ERROR: weird iteration counts, got %s" % iter_counts

    f.close()


def set_parameters_ml():
    """
    Helper routine to set parameters for the following multi-level runs

    Returns:
        dict: dictionary containing the simulation parameters
        dict: dictionary containing the controller parameters
        float: starting time
        float: end time
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
    problem_params['nvars'] = [63, 31]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['all_to_done'] = True  # can ask the controller to keep iterating all steps until the end

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_LU  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    return description, controller_params, t0, Tend


def set_parameters_sl():
    """
    Helper routine to set parameters for the following multi-level runs

    Returns:
        dict: dictionary containing the simulation parameters
        dict: dictionary containing the controller parameters
        float: starting time
        float: end time
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5E-10
    level_params['dt'] = 0.125

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 2  # frequency for the test value
    problem_params['nvars'] = 63  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_LU  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    return description, controller_params, t0, Tend


if __name__ == "__main__":
    main(num_proc_list=[1], fname='step_6_A_sl_out.txt', multi_level=False)
    main(num_proc_list=[1, 2, 4, 8], fname='step_6_A_ml_out.txt', multi_level=True)
