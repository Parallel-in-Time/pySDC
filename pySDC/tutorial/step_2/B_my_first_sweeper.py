from pathlib import Path

from pySDC.core.Step import step
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order


def main():
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
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

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = heat1d_forced
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the step we are going to work on
    S = step(description=description)

    # run IMEX SDC test and check error, residual and number of iterations
    err, res, niter = run_imex_sdc(S)
    print('Error and residual: %12.8e -- %12.8e' % (err, res))

    assert err <= 1e-5, "ERROR: IMEX SDC iteration did not reduce the error enough, got %s" % err
    assert res <= level_params['restol'], "ERROR: IMEX SDC iteration did not reduce the residual enough, got %s" % res
    assert niter <= 12, "ERROR: IMEX SDC took too many iterations, got %s" % niter


def run_imex_sdc(S):
    """
    Routine to run IMEX SDC on a single time step

    Args:
        S: an instance of a step representing the time step

    Returns:
        the error of SDC vs. exact solution
        the residual after the SDC sweeps
        the number of iterations
    """
    # make shortcuts for the level and the problem
    L = S.levels[0]
    P = L.prob

    # set initial time in the status of the level
    L.status.time = 0.1
    # compute initial value (using the exact function here)
    L.u[0] = P.u_exact(L.time)

    # access the sweeper's predict routine to get things started
    # if we don't do this, the values at the nodes are not initialized
    L.sweep.predict()
    # compute the residual (we may be done already!)
    L.sweep.compute_residual()

    # reset iteration counter
    S.status.iter = 0
    # run the SDC iteration until either the maximum number of iterations is reached or the residual is small enough
    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_2_B_out.txt', 'w')
    while S.status.iter < S.params.maxiter and L.status.residual > L.params.restol:
        # this is where the nodes are actually updated according to the SDC formulas
        L.sweep.update_nodes()
        # compute/update the residual
        L.sweep.compute_residual()
        # increment the iteration counter
        S.status.iter += 1
        out = 'Time %4.2f of %s -- Iteration: %2i -- Residual: %12.8e' % (
            L.time,
            L.level_index,
            S.status.iter,
            L.status.residual,
        )
        f.write(out + '\n')
        print(out)
    f.close()

    # compute the interval's endpoint: this (and only this) will set uend, depending on the collocation nodes
    L.sweep.compute_end_point()
    # update the simulation time
    L.status.time += L.dt

    # compute exact solution and compare
    uex = P.u_exact(L.status.time)
    err = abs(uex - L.uend)

    return err, L.status.residual, S.status.iter


if __name__ == "__main__":
    main()
