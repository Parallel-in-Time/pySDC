"""
This script shows how to let ParaDiag choose its alpha by itself.

ParaDiag replaces the time-stepping matrix by an alpha-circulant approximation. Alpha trades two error
sources against each other: a small value approximates the original problem better and converges in
fewer iterations, but conditions the diagonalization worse, so round-off and inexact inner solves get
amplified. A single fixed alpha therefore has to be a compromise for the whole run, even though the
right balance shifts as the residual falls.

The `AdaptiveAlpha` convergence controller updates alpha after every iteration instead, following
`Caklovic et al. <https://doi.org/10.2140/camcos.2023.18.55>`_:

    gamma     = L * (3 * eps + tau)
    alpha_k   = sqrt(gamma * r_k / e_k)
    e_{k+1}   = 2 * sqrt(gamma * e_k * r_k)

with L the block size, eps machine precision, tau the inner solver tolerance, r_k the residual and
e_k a running bound on the error. Gamma is an accuracy floor: there is no point pushing alpha below
the level at which round-off and the inner solver dominate anyway.

We compare a few fixed alphas against the adaptive one on the same advection problem used in part D.
The interesting result is not that adaptive wins on iteration count -- it ties with the best fixed
value -- but that it gets there without being told, and while keeping alpha orders of magnitude
larger, which is exactly the margin that protects you once the inner solves are inexact.

Since alpha is a property of the method and not of the parallelization, the adaptive controller has to
give the same answer whether ParaDiag runs virtually or across MPI ranks. We check that too.
"""

import os
import subprocess

from pySDC.tutorial.step_9.D_paradiag_MPI import get_description, num_steps_total

# the fixed values we compare against, plus the adaptive strategy
alpha_settings = [1e-2, 1e-4, 1e-8, 'adaptive']


def get_controller_params(alpha):
    """
    Controller parameters for one alpha setting.

    Args:
        alpha: a number, or the string 'adaptive'

    Returns:
        tuple: the controller parameters and the extra description entries
    """
    from pySDC.implementations.convergence_controller_classes.adaptive_alpha import AdaptiveAlpha

    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['average_jacobian'] = False

    extra_description = {}
    if alpha == 'adaptive':
        # the adaptive controller overwrites this from the first iteration onwards, but ParaDiag needs
        # some alpha to build its first transform with
        controller_params['alpha'] = 1e-4
        extra_description['convergence_controllers'] = {AdaptiveAlpha: {}}
    else:
        controller_params['alpha'] = alpha

    return controller_params, extra_description


def format_result(mode, alpha, niter, error, final_alpha):
    """
    One line of output, in the same shape for both controllers so they can be compared.

    Args:
        mode (str): 'MPI' or 'virtual'
        alpha: the alpha setting used
        niter (int): number of iterations needed
        error (float): error against the exact solution
        final_alpha (float): the alpha in use when the run finished

    Returns:
        str: the formatted line
    """
    return (
        f'{mode:>7s}: alpha {str(alpha):>9s} -> {niter:2d} iterations, '
        f'error {error:.4e}, final alpha {final_alpha:.3e}'
    )


def run(alpha, block_size, comm=None):
    """
    Run the advection problem from part D with one alpha setting.

    Args:
        alpha: a number, or the string 'adaptive'
        block_size (int): number of time-steps in one block
        comm: MPI communicator, or None for the virtually parallel controller

    Returns:
        tuple: the end value, the iteration count, the error and the final alpha
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted

    controller_params, extra_description = get_controller_params(alpha)
    description = {**get_description(), **extra_description}

    if comm is None:
        from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI

        controller_params['mssdc_jac'] = False
        controller = controller_ParaDiag_nonMPI(
            controller_params=controller_params, description=description, num_procs=block_size
        )
        steps = controller.MS
    else:
        from pySDC.implementations.controller_classes.controller_ParaDiag_MPI import controller_ParaDiag_MPI

        controller = controller_ParaDiag_MPI(controller_params=controller_params, description=description, comm=comm)
        steps = [controller.S]

    # ParaDiag diagonalizes in time, so the solution becomes complex
    for S in steps:
        S.levels[0].prob.init = tuple([*S.levels[0].prob.init[:2]] + [np.dtype('complex128')])

    P = steps[0].levels[0].prob
    dt = steps[0].levels[0].params.dt
    Tend = num_steps_total * dt

    uend, stats = controller.run(u0=P.u_exact(0.0), t0=0.0, Tend=Tend)
    niter = max(int(me[1]) for me in get_sorted(stats, type='niter', sortby='time', comm=comm))

    return uend, niter, abs(uend - P.u_exact(Tend)), controller.params.alpha


def main(cwd):
    """
    Compare fixed and adaptive alpha, with both controllers.

    Args:
        cwd (str): current working directory
    """

    try:
        import mpi4py

        del mpi4py
    except ImportError as e:
        raise ImportError('ParaDiag with MPI needs mpi4py') from e

    import numpy as np

    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    block_size = num_steps_total

    fname = 'step_9_E_out.txt'
    f = open(cwd + '/../../../data/' + fname, 'w')
    f.close()

    # the MPI controller, one rank per time-step, all alpha settings in one run
    print('Running ParaDiag with %2i ranks...' % block_size)
    cmd = ('mpirun -np ' + str(block_size) + ' python playground_adaptive_alpha.py ../../../../data/' + fname).split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s' % p.returncode

    # and the same with the virtually parallel controller
    f = open(cwd + '/../../../data/' + fname, 'a')
    results = {}
    for alpha in alpha_settings:
        uend, niter, error, final_alpha = run(alpha, block_size)
        results[alpha] = (uend, niter)
        out = format_result('virtual', alpha, niter, error, final_alpha)
        f.write(out + '\n')
        print(out)
    f.close()

    # the adaptive strategy should need no more iterations than the best fixed alpha we tried
    best_fixed = min(results[a][1] for a in alpha_settings if a != 'adaptive')
    assert (
        results['adaptive'][1] <= best_fixed
    ), 'ERROR: adaptive alpha needed %s iterations, the best fixed alpha only %s' % (results['adaptive'][1], best_fixed)

    # alpha changes the iteration, not the problem, so all settings solve the same thing
    reference = results[alpha_settings[0]][0]
    for alpha in alpha_settings[1:]:
        assert np.allclose(results[alpha][0], reference, atol=1e-5), (
            'ERROR: alpha %s gives a different solution' % alpha
        )


if __name__ == "__main__":
    main('.')
