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
