"""
The direct solve of the finite-difference problems reuses its factorization of ``I - factor * A``.

Reuse must not change a single bit against refactorizing with ``spsolve`` on every call, which is
what the solve did before; the cache has to stay bounded, too.
"""

import pytest


def _solve_twice(nvars, dtype='float64', rhs_dtype=None, solve_precision=None, factor=0.05):
    import numpy as np
    from scipy.sparse.linalg import spsolve

    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd

    prob = advectionNd(nvars=nvars, solver_type='direct', dtype=dtype, solve_precision=solve_precision)
    rng = np.random.default_rng(0)
    rhs = prob.dtype_u(prob.init)
    rhs[:] = rng.standard_normal(rhs.shape) + (1j * rng.standard_normal(rhs.shape) if rhs.dtype.kind == 'c' else 0)
    if rhs_dtype is not None:
        rhs = rhs.astype(rhs_dtype)

    first = prob.solve_system(rhs, factor, None, 0.0).copy()
    second = prob.solve_system(rhs, factor, None, 0.0)

    if solve_precision is None:
        matrix, b = prob.Id - factor * prob.A, rhs.flatten()
    else:
        precision = np.dtype(solve_precision)
        matrix, b = prob.Id.astype(precision) - precision.type(factor) * prob.A.astype(precision), rhs.flatten()
        b = b.astype(precision)
    expected = spsolve(matrix, np.asarray(b)).reshape(rhs.shape)
    return prob, first, second, expected


@pytest.mark.base
@pytest.mark.parametrize(
    'nvars, dtype, rhs_dtype, solve_precision',
    [
        (64, 'float64', None, None),
        ((16, 16), 'float64', None, None),
        (64, 'complex128', None, None),
        # a float64 numpy scalar widens a single-precision right-hand side, and spsolve promoted for it
        (64, 'float32', 'float64', None),
        (64, 'float64', None, 'float32'),
    ],
)
def test_cached_direct_solve_is_bitwise_spsolve(nvars, dtype, rhs_dtype, solve_precision):
    import numpy as np

    prob, first, second, expected = _solve_twice(nvars, dtype, rhs_dtype, solve_precision)
    assert len(prob._factorizations) == 1, 'the second solve with the same factor must reuse the factorization'
    assert np.array_equal(first, second)
    assert np.array_equal(first, expected.astype(first.dtype)), 'reuse changed the solution against spsolve'


@pytest.mark.base
def test_cached_direct_solve_is_bounded():
    import numpy as np

    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd

    prob = advectionNd(nvars=32, solver_type='direct')
    rhs = prob.dtype_u(prob.init, val=1.0)
    factors = [0.01 * (i + 1) for i in range(prob.max_cached_factorizations + 3)]
    for factor in factors:
        prob.solve_system(rhs, factor, None, 0.0)
    cached_factors = [factor for factor, _ in prob._factorizations]
    assert cached_factors == factors[-prob.max_cached_factorizations :], 'the oldest must go first'

    # an evicted factor is factorized afresh and still solves exactly
    sol = prob.solve_system(rhs, factors[0], None, 0.0)
    assert np.allclose((prob.Id - factors[0] * prob.A) @ sol.flatten(), rhs.flatten(), rtol=0, atol=1e-12)


@pytest.mark.base
def test_cached_direct_solve_mixed_rhs_precision():
    """One problem handed a single- and then a double-precision right-hand side for the same factor."""
    import numpy as np
    from scipy.sparse.linalg import spsolve

    from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd

    prob = advectionNd(nvars=64, solver_type='direct', dtype='float32')
    rhs = prob.dtype_u(prob.init)
    rhs[:] = np.random.default_rng(0).standard_normal(rhs.shape)
    matrix = prob.Id - 0.05 * prob.A
    for b in (rhs, rhs.astype('float64')):
        sol = prob.solve_system(b, 0.05, None, 0.0)
        assert np.array_equal(sol, spsolve(matrix, np.asarray(b.flatten())).reshape(b.shape).astype(sol.dtype))
