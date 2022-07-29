import os

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.helpers.visualization_tools import show_residual_across_simulation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.tutorial.step_6.A_run_non_MPI_controller import set_parameters_ml


def main():
    """
    A simple test program to demonstrate residual visualization
    """

    # get parameters from Step 6, Part A
    description, controller_params, t0, Tend = set_parameters_ml()

    # use 8 processes here
    num_proc = 8

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare (for testing purposes only)
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    min_iter = 99
    max_iter = 0
    f = open('step_8_A_out.txt', 'w')
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %1i' % item
        f.write(out + '\n')
        print(out)
        min_iter = min(min_iter, item[1])
        max_iter = max(max_iter, item[1])
    f.close()

    # call helper routine to produce residual plot

    fname = 'step_8_residuals.png'
    show_residual_across_simulation(stats=stats, fname=fname)

    assert err < 6.1555e-05, 'ERROR: error is too large, got %s' % err
    assert os.path.isfile(fname), 'ERROR: residual plot has not been created'
    assert min_iter == 7 and max_iter == 7, "ERROR: number of iterations not as expected, got %s and %s" % (
        min_iter,
        max_iter,
    )


if __name__ == "__main__":
    main()
