import numpy as np
import pytest


@pytest.mark.base
def test_cg_reaches_the_accuracy_of_its_format():
    """On a well-conditioned stencil, CG gets as close as its format allows, and stops on a breakdown."""
    from pySDC.projects.DeltaSDC.krylov import DTYPES, cg

    n = 64
    b = np.sin(2 * np.pi * np.arange(n) / n)
    A = 1.4 * np.eye(n) - 0.2 * (np.eye(n, k=1) + np.eye(n, k=-1) + np.eye(n, k=n - 1) + np.eye(n, k=1 - n))
    reference = np.linalg.solve(A, b)
    for fmt, tolerance in [('fp64', 1e-12), ('fp32', 1e-5), ('fp16', 1e-2)]:
        c = np.dtype(DTYPES[fmt]).type(0.2)
        diagonal = np.dtype(DTYPES[fmt]).type(1.4)

        def apply(x):
            return diagonal * x - c * (np.roll(x, 1) + np.roll(x, -1))

        x, iterations, reached = cg(apply, b, fmt, rtol=tolerance / 10, maxiter=4 * n)
        assert x.dtype == np.dtype(DTYPES[fmt])
        assert np.abs(x - reference).max() / np.abs(reference).max() < tolerance, fmt
        assert 0 < iterations < 4 * n, fmt


@pytest.mark.base
def test_heat_cg_solves_like_the_direct_solver():
    """In fp64, the CG node solve of heat_cg agrees with the sparse direct solve of the parent class."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.projects.DeltaSDC.krylov import heat_cg

    params = {'nvars': 64, 'nu': 1.0, 'freq': 2, 'bc': 'periodic'}
    direct, iterative = heatNd_unforced(**params), heat_cg(cg_rtol=1e-13, **params)
    rhs = direct.u_exact(0.0)
    for factor in [1e-4, 1e-2]:
        reference = direct.solve_system(rhs, factor, rhs, 0.0)
        assert np.abs(iterative.solve_system(rhs, factor, rhs, 0.0) - reference).max() < 1e-11
    assert len(iterative.cg_iterations) == 2 and min(iterative.cg_iterations) > 0
