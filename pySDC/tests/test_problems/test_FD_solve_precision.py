"""
``solve_precision`` and ``solver_type='FFT'`` on the finite-difference problems.

The solve runs at a precision of its own, independently of the state's ``dtype``. Each accuracy
check has a lower bound too: a solve that silently stayed in double precision would pass the upper
one, so only the lower one shows the reduced precision is real.
"""

import pytest

SIZES = [64, (32, 32)]


def _problem(name, **params):
    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    return {'heat': heatNd_unforced, 'advection': advectionNd}[name](lintol=1e-14, **params)


def _error(prob, factor):
    """Relative error of a solve against a direct one in double precision; tiny rhs, as a correction's."""
    import numpy as np
    from scipy.sparse.linalg import spsolve

    rhs = prob.dtype_u(prob.init)
    rng = np.random.default_rng(0)
    rhs[:] = rng.standard_normal(rhs.shape) * (1e-9 if rhs.dtype.kind == 'f' else (1 + 1j) * 1e-9)
    solution = prob.solve_system(rhs, factor, prob.dtype_u(prob.init, val=0.0), 0.0)
    matrix = (prob.Id - factor * prob.A).astype(rhs.dtype).tocsc()
    exact = spsolve(matrix, np.asarray(rhs).flatten()).reshape(rhs.shape)
    assert solution.dtype == rhs.dtype, 'the solve hands back the state precision, whatever it computed in'
    return np.abs(solution - exact).max() / np.abs(exact).max()


@pytest.mark.base
@pytest.mark.parametrize('nvars', SIZES)
@pytest.mark.parametrize('name', ['heat', 'advection'])
@pytest.mark.parametrize(
    'solver_type, precision, low, high',
    [
        ('FFT', None, 0, 1e-14),
        ('FFT', 'float32', 1e-9, 1e-6),
        ('FFT', 'float16', 1e-5, 3e-3),
        ('direct', 'float32', 1e-9, 1e-6),
        # a Krylov solve stops at 100 eps of its precision, so its error also carries the condition
        ('GMRES', 'float32', 1e-9, 1e-4),
    ],
)
def test_solve_precision(name, nvars, solver_type, precision, low, high):
    prob = _problem(name, nvars=nvars, solver_type=solver_type, solve_precision=precision)
    assert low <= _error(prob, 1e-3) < high


@pytest.mark.base
@pytest.mark.parametrize('solver_type, low, high', [('direct', 1e-9, 1e-5), ('FFT', 1e-9, 1e-5)])
def test_complex_state_and_factor(solver_type, low, high):
    """ParaDiag's case: a complex state, a complex factor, and the solve at complex64."""
    prob = _problem('heat', nvars=(32, 32), solver_type=solver_type, solve_precision='complex64', dtype='complex128')
    assert low <= _error(prob, 0.3 - 0.2j) < high


@pytest.mark.base
def test_misconfigurations_raise():
    from pySDC.core.errors import ProblemError

    with pytest.raises(ProblemError, match='single precision'):
        _problem('heat', nvars=64, solver_type='direct', solve_precision='float16')
    with pytest.raises(ProblemError, match='periodic'):
        _problem('heat', nvars=63, bc='dirichlet-zero', solver_type='FFT')


@pytest.mark.parametrize(
    'useGPU', [pytest.param(False, marks=pytest.mark.base), pytest.param(True, marks=pytest.mark.cupy)]
)
def test_half_precision_solve_reaches_double_precision_in_delta_form(useGPU):
    """
    What the option is for: the delta form solves for a correction, so a half-precision solve costs
    iterations, not accuracy. Stock SDC hands the same solve the state, and stalls -- the control.

    On a CPU the half-precision solve is an emulation; on a GPU it is cuFFT's half-precision
    arithmetic, which is the case the correction-solve contract on ``Problem`` is written for.
    """
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    def floor(sweeper_class, precision, **sweeper_extra):
        description = {
            'problem_class': heatNd_unforced,
            'problem_params': {
                'nvars': (32, 32),
                'nu': 0.1,
                'solver_type': 'FFT',
                'solve_precision': precision,
                'useGPU': useGPU,
            },
            'sweeper_class': sweeper_class,
            'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU', **sweeper_extra},
            'level_params': {'restol': -1, 'dt': 1e-2},
            'step_params': {'maxiter': 20},
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
        prob = controller.MS[0].levels[0].prob
        _, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=1e-2)
        return min(value for _, value in get_sorted(stats, type='residual_post_iteration'))

    full = floor(generic_implicit, None)
    assert full < 1e-13
    assert floor(delta_implicit, 'float16', linear_implicit=True) < 10 * full
    assert floor(generic_implicit, 'float16') > 1e-6, 'the control: the state itself through float16'


@pytest.mark.cupy
@pytest.mark.parametrize(
    'solver_type, precision, low, high', [('CG', 'float32', 1e-9, 1e-4), ('FFT', 'float16', 1e-5, 1e-2)]
)
def test_solve_precision_on_GPU(solver_type, precision, low, high):
    """On the device the reduced precision is genuine arithmetic, half precision included (cuFFT)."""
    import cupy as cp

    solves = {}
    for p in [None, precision]:
        prob = _problem('heat', nvars=(256, 256), solver_type=solver_type, solve_precision=p, useGPU=True)
        rhs = prob.dtype_u(prob.init)
        rhs[:] = cp.random.default_rng(0).standard_normal(prob.nvars) * 1e-9
        solves[p] = prob.solve_system(rhs, 1e-3, prob.dtype_u(prob.init, val=0.0), 0.0)
    error = float(abs(solves[precision] - solves[None]) / abs(solves[None]))
    assert low <= error < high, f'{precision} {solver_type} solve off by {error:.1e}'
    assert solves[precision].dtype == cp.float64
