import numpy as np
import pytest


def run_ParaDiag(alpha, L=4, M=3, N=2, dt=1e-1, restol=1e-10, maxiter=50, **cc_params):
    """
    Run a small Dahlquist problem with ParaDiag, either with a fixed or an adaptive alpha.

    Args:
        alpha: a number, or the string 'adaptive'
        L (int): number of steps in the block
        M (int): number of collocation nodes
        N (int): number of degrees of freedom in space
        dt (float): step size
        restol (float): residual tolerance
        maxiter (int): maximum number of iterations
        cc_params: parameters passed to the AdaptiveAlpha convergence controller

    Returns:
        tuple: the controller, the end value and the iteration count
    """
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI
    from pySDC.implementations.convergence_controller_classes.adaptive_alpha import AdaptiveAlpha
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization

    description = {
        'problem_class': testequation0d,
        'problem_params': {'lambdas': -1.0 * np.ones(shape=(N)), 'u0': 1},
        'sweeper_class': QDiagonalization,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': M, 'initial_guess': 'spread'},
        'level_params': {'dt': dt, 'restol': restol},
        'step_params': {'maxiter': maxiter},
    }
    controller_params = {
        'logger_level': 30,
        'alpha': 1e-4 if alpha == 'adaptive' else alpha,
        'average_jacobian': False,
        'mssdc_jac': False,
    }
    if alpha == 'adaptive':
        description['convergence_controllers'] = {AdaptiveAlpha: cc_params}

    controller = controller_ParaDiag_nonMPI(controller_params=controller_params, description=description, num_procs=L)
    for S in controller.MS:
        S.levels[0].prob.init = tuple([*S.levels[0].prob.init[:2]] + [np.dtype('complex128')])

    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(0), t0=0, Tend=L * dt)
    niter = max(int(me[1]) for me in get_sorted(stats, type='niter', sortby='time'))
    return controller, uend, niter


def get_convergence_controller(controller):
    """Fish the AdaptiveAlpha instance out of a controller."""
    from pySDC.implementations.convergence_controller_classes.adaptive_alpha import AdaptiveAlpha

    return [me for me in controller.convergence_controllers if type(me) == AdaptiveAlpha][0]


@pytest.mark.base
def test_adaptive_alpha_matches_best_fixed():
    """
    Adaptive alpha should need no more iterations than the best fixed alpha, without being told it.
    """
    fixed = {alpha: run_ParaDiag(alpha)[2] for alpha in [1e-2, 1e-4, 1e-8]}
    _, _, niter_adaptive = run_ParaDiag('adaptive')

    assert niter_adaptive <= min(
        fixed.values()
    ), f'adaptive needed {niter_adaptive} iterations, best fixed only {min(fixed.values())} ({fixed})'
    # and the badly chosen fixed alpha really is worse, or the test proves nothing
    assert max(fixed.values()) > min(fixed.values()), f'alpha made no difference at all: {fixed}'


@pytest.mark.base
def test_adaptive_alpha_stays_better_conditioned():
    """
    The point of adapting is not only the iteration count: alpha should end up much larger than the
    smallest fixed value that achieves the same, because a larger alpha is better conditioned.
    """
    controller, _, _ = run_ParaDiag('adaptive')
    assert controller.params.alpha > 1e-8, f'expected a well-conditioned alpha, got {controller.params.alpha}'
    assert (
        controller.params.alpha < 1e-2
    ), f'alpha should have been reduced from the start, got {controller.params.alpha}'


@pytest.mark.base
def test_alpha_history_advances_once_per_iteration():
    """
    The controller is called once per step in the virtual controller, but alpha must advance once per
    iteration, otherwise the recursion runs L times too fast.
    """
    controller, _, niter = run_ParaDiag('adaptive')
    alphas = get_convergence_controller(controller).alphas

    assert len(alphas) <= niter + 1, f'got {len(alphas)} alpha updates for {niter} iterations: {alphas}'
    assert len(alphas) > 1, 'alpha never changed'


@pytest.mark.base
def test_alpha_is_clamped():
    """Alpha has to stay inside the configured range."""
    controller, _, _ = run_ParaDiag('adaptive', alpha_min=1e-3, alpha_max=1e-2)
    alphas = get_convergence_controller(controller).alphas

    assert min(alphas) >= 1e-3, f'alpha fell below alpha_min: {alphas}'
    assert max(alphas) <= 1e-2, f'alpha rose above alpha_max: {alphas}'


@pytest.mark.base
def test_gamma_accounts_for_inner_tolerance():
    """
    Gamma is the accuracy floor. A looser inner solver means a larger floor, hence a larger alpha:
    there is no point resolving below the noise.
    """
    exact, _, _ = run_ParaDiag('adaptive', inner_tol=0.0)
    loose, _, _ = run_ParaDiag('adaptive', inner_tol=1e-6)

    cc_exact = get_convergence_controller(exact)
    cc_loose = get_convergence_controller(loose)

    assert cc_loose.get_gamma(loose) > cc_exact.get_gamma(exact), 'a looser inner solver must raise gamma'
    assert (
        cc_loose.alphas[0] > cc_exact.alphas[0]
    ), f'a looser inner solver must give a larger first alpha, got {cc_loose.alphas[0]} vs {cc_exact.alphas[0]}'


@pytest.mark.base
def test_alpha_does_not_change_the_solution():
    """Alpha changes the iteration, not the problem: every setting solves the same collocation problem."""
    _, u_fixed, _ = run_ParaDiag(1e-8)
    _, u_adaptive, _ = run_ParaDiag('adaptive')

    assert np.allclose(
        u_fixed, u_adaptive, atol=1e-8
    ), f'fixed and adaptive alpha give different answers, difference {abs(u_fixed - u_adaptive):.3e}'


@pytest.mark.base
def test_zero_residual_is_ignored():
    """A zero residual carries no information about alpha, and dividing by it would be worse."""
    controller, _, _ = run_ParaDiag('adaptive')
    cc = get_convergence_controller(controller)

    alpha_before = controller.params.alpha
    n_before = len(cc.alphas)

    for step in controller.steps:
        step.levels[0].status.residual = 0.0
    cc.last_iter = None
    cc.post_iteration_processing(controller, controller.steps[0])

    assert controller.params.alpha == alpha_before, 'ERROR: a zero residual moved alpha'
    assert len(cc.alphas) == n_before, 'ERROR: a zero residual was recorded as an update'
