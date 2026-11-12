import numpy as np
import pytest

UNIT_ROUNDOFF = {'fp32': 2.0**-24, 'fp16': 2.0**-11}


@pytest.mark.base
def test_genuine_lu_is_backward_stable():
    """The LU really runs in the format: backward error about u, forward error up to about kappa u."""
    from pySDC.projects.DeltaSDC.genuine import lu, lu_solve

    n = 64
    B = np.eye(n) + 100.0 * (2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    b = np.random.default_rng(0).standard_normal(n)
    b /= np.abs(b).max()
    for fmt in ('fp32', 'fp16'):
        y = lu_solve(lu(B, fmt), b, fmt)
        backward = np.abs(B @ y - b).max() / (np.abs(B).sum(1).max() * np.abs(y).max())
        assert 0.01 * UNIT_ROUNDOFF[fmt] < backward < 4 * UNIT_ROUNDOFF[fmt], fmt


@pytest.mark.base
@pytest.mark.parametrize('fmt', ['fp64', 'fp32', 'fp16'])
def test_genuine_solve(fmt):
    """A genuine solve of the heat equation agrees with the direct fp64 solve to about kappa u of its format."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.projects.DeltaSDC.genuine import genuine

    params = {'nvars': 63, 'nu': 1.0, 'freq': 2, 'bc': 'dirichlet-zero'}
    exact, low = heatNd_unforced(**params), genuine(heatNd_unforced)(solve_format=fmt, **params)
    rhs = exact.u_exact(0.0)
    reference = exact.solve_system(rhs, 1e-3, rhs, 0.0)  # kappa = 1 + 4 * 64^2 * 1e-3, about 17
    x = low.solve_system(rhs, 1e-3, rhs, 0.0)
    error = np.abs(x - reference).max() / np.abs(reference).max()
    assert error < {'fp64': 1e-14, 'fp32': 1e-5, 'fp16': 1e-1}[fmt], error
    assert fmt == 'fp64' or error > 0.0  # the format really is lower
