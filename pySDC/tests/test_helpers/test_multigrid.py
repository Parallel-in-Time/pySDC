"""
The geometric multigrid helper: convergence in 1D, 2D and 3D, the formats, the domain length, the guards, and on a
GPU the fused 3D kernels against the array operations.
"""

import numpy as np
import pytest

CASES = [(1, 4096), (2, 128), (3, 32)]  # coarsened three to four times each


def _residual(mg, x, b, s):
    """The relative residual in float64, with an operator independent of the solver's."""
    xp = mg.xp
    x, b = x.astype(np.float64), b.astype(np.float64)
    lap = sum(xp.roll(x, shift, axis) for axis in range(x.ndim) for shift in (1, -1)) - 2 * x.ndim * x
    r = b - (x - s * (mg.N / mg.L) ** 2 * lap)
    return float(xp.linalg.norm(r) / xp.linalg.norm(b))


@pytest.mark.base
@pytest.mark.parametrize('ndim, N', CASES)
@pytest.mark.parametrize('stiffness', [1e-2, 1e2, 1e5])
def test_vcycles_converge(ndim, N, stiffness):
    """A V-cycle reduces the residual by a factor of at most 0.3, whether the system is near the identity or stiff."""
    from pySDC.helpers.multigrid import GeometricMultigrid

    mg = GeometricMultigrid(N, ndim)
    s = stiffness / N**2  # s h^-2 = stiffness
    b = np.random.default_rng(0).standard_normal((N,) * ndim)
    x, cycles = mg.solve(b, s, rtol=1e-11, maxiter=50)
    assert _residual(mg, x, b, s) < 1e-11
    assert cycles <= np.ceil(np.log(1e-11) / np.log(0.3)), f'{cycles} V-cycles'


@pytest.mark.base
@pytest.mark.parametrize('ndim, N', CASES)
@pytest.mark.parametrize('dtype, low, high', [(np.float32, 1e-9, 1e-5), (np.float16, 1e-5, 3e-3)])
def test_formats(ndim, N, dtype, low, high):
    """
    In float32 and float16 the solve stops at the accuracy of the format, without being told it, and returns its
    format; float16 computes in float32, which the error to float64 shows (float16 arithmetic would give 1e-2 or
    nothing at these sizes, where h^-2 exceeds its range).
    """
    from pySDC.helpers.multigrid import GeometricMultigrid

    mg = GeometricMultigrid(N, ndim)
    s = 1e2 / N**2
    b = np.random.default_rng(1).standard_normal((N,) * ndim)
    exact, _ = mg.solve(b, s, rtol=1e-13, maxiter=50)
    x, cycles = mg.solve(b.astype(dtype), s, rtol=1e-13, maxiter=50)
    assert x.dtype == dtype
    assert cycles < 50, 'the stall rule ends the solve at the accuracy of the format'
    assert low <= np.abs(x - exact).max() / np.abs(exact).max() < high


@pytest.mark.base
def test_hopeless_format_returns_no_worse_than_zero():
    """A system far too ill-conditioned for float16 gives back its best iterate, not one worse than zero."""
    from pySDC.helpers.multigrid import GeometricMultigrid

    mg = GeometricMultigrid(8192, 1)
    b = np.random.default_rng(2).standard_normal(8192).astype(np.float16)
    x, cycles = mg.solve(b, 1.0, rtol=1e-10, maxiter=50)
    assert cycles < 50
    assert _residual(mg, x, b, 1.0) <= 1.0


@pytest.mark.base
def test_domain_length_and_laplacian():
    """The side length sets the grid spacing; ``laplacian`` is the operator the solve inverts."""
    from pySDC.helpers.multigrid import GeometricMultigrid

    N, L = 64, 2 * np.pi
    mg = GeometricMultigrid(N, 2, L=L)
    x = np.arange(N) * L / N
    u = np.sin(x)[:, None] * np.cos(2 * x)[None, :]
    h = L / N
    eigenvalue = -(2 - 2 * np.cos(h)) / h**2 - (2 - 2 * np.cos(2 * h)) / h**2  # of the discrete Laplacian, ~ -5
    assert np.abs(mg.laplacian(u) - eigenvalue * u).max() < 1e-12
    x_solved, _ = mg.solve(u - 0.1 * mg.laplacian(u), 0.1, rtol=1e-12, maxiter=50)
    assert np.abs(x_solved - u).max() < 1e-11


@pytest.mark.base
def test_guards():
    from pySDC.helpers.multigrid import GeometricMultigrid

    with pytest.raises(ValueError, match='coarsen'):
        GeometricMultigrid(22, 3)  # 11^3 > 1000 and 11 is odd
    with pytest.raises(ValueError, match='ndim'):
        GeometricMultigrid(16, 4)
    with pytest.raises(ValueError, match='shape'):
        GeometricMultigrid(16, 2).solve(np.ones(16), 1.0, 1e-8, 10)
    x, cycles = GeometricMultigrid(16, 2).solve(np.zeros((16, 16)), 1.0, 1e-8, 10)
    assert cycles == 0 and not x.any()


@pytest.mark.cupy
@pytest.mark.parametrize('dtype', ['float64', 'float32', 'float16'])
def test_fused_kernels_are_the_array_operations(dtype):
    """
    On a GPU, 3D grids run on fused kernels. Without FMA contraction, the stencils compute bit for bit what the
    array operations compute; restriction and interpolation sum in another order and agree to rounding.
    """
    import cupy as cp

    from pySDC.helpers.multigrid import _Ops

    fused, arrays = _Ops(cp), _Ops(cp)
    arrays.fused = lambda x: False  # the same operations, forced onto the array path
    n, s, h2inv = 64, 3e-3, 64.0**2
    rng = cp.random.default_rng(0)
    x, b = (rng.standard_normal((n, n, n)).astype(dtype) for _ in range(2))
    e = rng.standard_normal((n // 2,) * 3).astype(dtype)
    for kind in range(3):
        for section in (16, 64):  # sections that divide n and one that is all of it
            fused.section = section
            same = fused.stencil(kind, x, b, s * h2inv, 0.7), arrays.stencil(kind, x, b, s * h2inv, 0.7)
            assert cp.array_equal(*same), f'stencil {kind}, section {section}'
    bound = {'float64': 1e-14, 'float32': 1e-6, 'float16': 2e-3}[dtype]  # a few units in the last place
    for name, (a, r) in {
        'jacobi from zero': (fused.jacobi_from_zero(b, s, h2inv), arrays.jacobi_from_zero(b, s, h2inv)),
        'restrict': (fused.restrict(x), arrays.restrict(x)),
        'prolong': (fused.prolong_add(x.copy(), e), arrays.prolong_add(x.copy(), e)),
    }.items():
        assert a.dtype == r.dtype == x.dtype
        difference = float(cp.abs(a.astype(cp.float64) - r).max() / cp.abs(r.astype(cp.float64)).max())
        assert difference < bound, f'{name}: {difference:.1e}'


@pytest.mark.cupy
@pytest.mark.parametrize('ndim, N', CASES)
@pytest.mark.parametrize('fma', [False, True])
def test_solve_on_GPU(ndim, N, fma):
    """On a GPU, fused (3D) or not, the solve converges as on the CPU, with and without FMA contraction."""
    import cupy as cp

    from pySDC.helpers.multigrid import GeometricMultigrid

    s = 1e2 / N**2
    b = np.random.default_rng(3).standard_normal((N,) * ndim)
    x_cpu, cycles_cpu = GeometricMultigrid(N, ndim).solve(b, s, rtol=1e-11, maxiter=50)
    mg = GeometricMultigrid(N, ndim, xp=cp, fma=fma)
    x, cycles = mg.solve(cp.asarray(b), s, rtol=1e-11, maxiter=50)
    assert _residual(mg, x, cp.asarray(b), s) < 1e-11
    assert abs(cycles - cycles_cpu) <= 1
    # both within the tolerance, which bounds the error up to the condition number (about 1200 here)
    assert float(cp.abs(x - cp.asarray(x_cpu)).max()) < 1e-7 * float(np.abs(x_cpu).max())
