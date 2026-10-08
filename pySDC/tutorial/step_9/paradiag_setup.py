"""
The ParaDiag setup of tutorial step 9, parts D and E: the advection problem of part C, a few fixed alphas and the
adaptive one, and a `run` that takes a communicator or not. Part D runs it with the virtually parallel controller,
part E on MPI ranks, which is the whole point: the same function, serial or parallel.
"""

# we always do this many time-steps in total, no matter how many of them run in parallel
num_steps_total = 4


def get_description():
    """
    Set up the same advection problem as in Part C.

    Returns:
        dict: the description for the ParaDiag controller
    """
    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
    from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization

    level_params = {}
    level_params['dt'] = 0.1
    level_params['restol'] = 1e-6

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['initial_guess'] = 'copy'

    # Part C uses GMRES here to count linear solver work. We only care about the parallelism, and the
    # complex shifted systems ParaDiag produces are hard for GMRES, so we solve them directly instead.
    problem_params = {'nvars': 64, 'order': 8, 'c': 1, 'solver_type': 'direct'}

    step_params = {}
    step_params['maxiter'] = 99

    description = {}
    description['problem_class'] = advectionNd
    description['problem_params'] = problem_params
    description['sweeper_class'] = QDiagonalization
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description


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
        mode (str): which controller produced it, e.g. 'virtual' or 'MPI on 4'
        alpha: the alpha setting used
        niter (int): number of iterations needed
        error (float): error against the exact solution
        final_alpha (float): the alpha in use when the run finished

    Returns:
        str: the formatted line
    """
    return (
        f'{mode:>11s}: alpha {str(alpha):>9s} -> {niter:2d} iterations, '
        f'error {error:.4e}, final alpha {final_alpha:.3e}'
    )


def run(alpha, block_size, comm=None):
    """
    Run the advection problem with one alpha setting.

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
