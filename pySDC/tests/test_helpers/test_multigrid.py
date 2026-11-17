"""
The geometric multigrid helper: convergence in 1D, 2D and 3D, periodic and with zero Dirichlet values, with and
without advection, the formats, the domain length, the guards, and on a GPU the fused 3D kernels against the array
operations.
"""

import numpy as np
import pytest

CASES = [(1, 4096), (2, 128), (3, 32)]  # coarsened three to four times each
DIRICHLET_CASES = [(1, 4095), (2, 127), (3, 31)]  # the same, N + 1 = 2^k
PECLET_CASES = [(1, 4095), (2, 63), (3, 15)]  # coarsest grid spacing 8, 2 and 2 times the finest
DIRECTIONS = (1.0, -0.5, 0.25)  # velocity components relative to the largest, of either sign


def _fd_operator(N, ndim, bc, nu=1.0, velocity=None):
    r"""
    :math:`\nu \Delta_h - c \cdot \nabla_h` assembled from pySDC's finite-difference matrices, independently of the
    solver: the second-order Laplacian and, per direction, first-order upwind differences.
    """
    import scipy.sparse as sp

    from pySDC.helpers.problem_helper import get_1d_grid, get_finite_difference_matrix

    dx, _ = get_1d_grid(N, bc)

    def matrix(derivative, order, stencil_type):
        return get_finite_difference_matrix(derivative, order, stencil_type, dx=dx, size=N, dim=1, bc=bc)[0]

    operator = 0
    for axis in range(ndim):
        one_d = nu * matrix(2, 2, 'center')
        c = 0.0 if velocity is None else velocity[axis]
        if c:  # pySDC's 'upwind' is the backward difference, for positive velocities
            one_d = one_d - c * matrix(1, 1, 'upwind' if c > 0 else 'forward')
        factors = [sp.eye(N)] * ndim
        factors[axis] = one_d
        term = factors[0]
        for factor in factors[1:]:
            term = sp.kron(term, factor)
        operator = operator + term
    return sp.csc_matrix(operator)


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
    # a first cycle that overflows float16 is not kept, though the first cycle is not judged by its residual otherwise
    mg = GeometricMultigrid(4096, 1)
    grid = np.arange(4096) / 4096
    b = (np.sin(np.pi * grid) + grid).astype(np.float16)
    x, cycles = mg.solve(b, 1e8 / 4096**2, rtol=1e-10, maxiter=50)
    assert cycles == 1 and not x.any() and mg.relative_residual == 1.0


@pytest.mark.base
def test_first_cycle_is_judged_by_the_second():
    """
    With advection, a smooth right-hand side and a stiff system, the first V-cycle from zero leaves its error in the
    outflow boundary layer and increases the residual, though at a coarsest-grid Peclet number of 0.06; the cycles
    after it converge, and the solve with them.
    """
    from pySDC.helpers.multigrid import GeometricMultigrid

    N, rtol = 1023, 1e-10
    mg = GeometricMultigrid(N, 1, bc='dirichlet-zero', velocity=(32.0,))  # H = 1/512
    grid = np.arange(1, N + 1) / (N + 1)
    b = np.sin(np.pi * grid) + grid
    s = 1e6 / (N + 1) ** 2
    x, cycles = mg.solve(b, s, rtol=rtol, maxiter=1)
    assert cycles == 1 and not x.any(), 'one cycle alone is worse than zero, and zero is returned'
    x, cycles = mg.solve(b, s, rtol=rtol, maxiter=50)
    assert np.linalg.norm(b - (x - mg.apply(x, s))) < rtol * np.linalg.norm(b)
    assert mg.relative_residual < rtol
    assert cycles <= np.ceil(np.log(rtol) / np.log(0.3)), f'{cycles} V-cycles'


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
def test_apply_and_laplacian():
    """
    ``apply`` is the whole operator, scaled: nu times the Laplacian less the upwind advection; ``laplacian`` stays the
    plain Laplacian, whatever nu and the velocity.
    """
    from pySDC.helpers.multigrid import GeometricMultigrid

    N, nu, velocity = 32, 0.1, (3.0, -2.0)
    diffusion = GeometricMultigrid(N, 2)
    mg = GeometricMultigrid(N, 2, nu=nu, velocity=velocity)
    x = np.random.default_rng(10).standard_normal((N, N))
    assert np.array_equal(mg.laplacian(x, 0.5), diffusion.laplacian(x, 0.5))
    upwind = velocity[0] * (x - np.roll(x, 1, 0)) * N + velocity[1] * (np.roll(x, -1, 1) - x) * N
    assert np.abs(mg.apply(x, 0.5) - 0.5 * (nu * mg.laplacian(x) - upwind)).max() < 1e-12 * np.abs(upwind).max()
    out = np.empty_like(x)
    assert mg.laplacian(x, 0.5, out=out) is out


@pytest.mark.base
@pytest.mark.parametrize('ndim, N', DIRICHLET_CASES)
@pytest.mark.parametrize('stiffness', [1e-2, 1e2, 1e5])
def test_dirichlet_vcycles_converge(ndim, N, stiffness):
    """With zero Dirichlet values, a V-cycle reduces the residual of pySDC's FD heat operator by at most 0.3."""
    from pySDC.helpers.multigrid import GeometricMultigrid

    mg = GeometricMultigrid(N, ndim, bc='dirichlet-zero')
    s = stiffness / (N + 1) ** 2  # s h^-2 = stiffness
    b = np.random.default_rng(4).standard_normal((N,) * ndim)
    x, cycles = mg.solve(b, s, rtol=1e-11, maxiter=50)
    A = _fd_operator(N, ndim, 'dirichlet-zero')
    assert np.linalg.norm(b.ravel() - (x.ravel() - s * (A @ x.ravel()))) < 1e-11 * np.linalg.norm(b)
    assert cycles <= np.ceil(np.log(1e-11) / np.log(0.3)), f'{cycles} V-cycles'


@pytest.mark.base
@pytest.mark.parametrize('bc, N', [('periodic', 16), ('dirichlet-zero', 15)])
@pytest.mark.parametrize('ndim', [1, 2, 3])
@pytest.mark.parametrize('advection', [False, True])
def test_operator_is_pySDCs_FD_matrix(bc, N, ndim, advection):
    """
    The operator is that of pySDC's finite differences, upwinded by the sign of each velocity component, and the
    solve agrees with a direct sparse solve of it.
    """
    from scipy.sparse import eye
    from scipy.sparse.linalg import spsolve

    from pySDC.helpers.multigrid import GeometricMultigrid

    nu, velocity = 0.1, (DIRECTIONS[:ndim] if advection else None)
    mg = GeometricMultigrid(N, ndim, bc=bc, nu=nu, velocity=velocity)
    A = _fd_operator(N, ndim, bc, nu, velocity)
    x = np.random.default_rng(5).standard_normal((N,) * ndim)
    assert np.abs(mg.apply(x).ravel() - A @ x.ravel()).max() < 1e-12 * np.abs(A @ x.ravel()).max()
    s = 0.05
    solved, _ = mg.solve(x, s, rtol=1e-13, maxiter=50)
    exact = spsolve(eye(N**ndim, format='csc') - s * A, x.ravel())
    assert np.abs(solved.ravel() - exact).max() < 1e-11 * np.abs(exact).max()


@pytest.mark.base
@pytest.mark.parametrize('ndim, N', PECLET_CASES)
@pytest.mark.parametrize('peclet', [0.5, 2])
@pytest.mark.parametrize('stiffness', [1e-2, 1e2, 1e5])
def test_advection_diffusion_converges(ndim, N, peclet, stiffness):
    """At grid Peclet numbers |c| h / nu of 0.5 and 2 on the finest grid, a V-cycle reduces the residual by 0.3."""
    from pySDC.helpers.multigrid import GeometricMultigrid

    nu, h = 0.1, 1 / (N + 1)
    velocity = [peclet * nu / h * d for d in DIRECTIONS[:ndim]]
    mg = GeometricMultigrid(N, ndim, bc='dirichlet-zero', nu=nu, velocity=velocity)
    s = stiffness * h**2 / nu
    b = np.random.default_rng(6).standard_normal((N,) * ndim)
    x, cycles = mg.solve(b, s, rtol=1e-11, maxiter=50)
    A = _fd_operator(N, ndim, 'dirichlet-zero', nu, velocity)
    assert np.linalg.norm(b.ravel() - (x.ravel() - s * (A @ x.ravel()))) < 1e-11 * np.linalg.norm(b)
    assert cycles <= np.ceil(np.log(1e-11) / np.log(0.3)), f'{cycles} V-cycles'


@pytest.mark.base
def test_advection_beyond_the_peclet_limit_stops_early():
    """
    The documented limit: at a grid Peclet number of 16 on the coarsest grid (2 on the finest, three coarsenings), a
    V-cycle reduces the residual by about 0.55, the stall rule takes that for the end of the format and stops, and the
    result is not converged, but no worse than zero, which ``relative_residual`` tells. A higher ``stall`` goes on to
    ``rtol``.
    """
    from pySDC.helpers.multigrid import GeometricMultigrid

    N, nu, rtol = 255, 1.0, 1e-10
    s = 1e4 / (N + 1) ** 2
    b = np.random.default_rng(7).standard_normal((N, N))
    for stall in (0.5, 0.8):
        mg = GeometricMultigrid(N, 2, bc='dirichlet-zero', nu=nu, velocity=(2.0 * (N + 1),) * 2, stall=stall)
        x, cycles = mg.solve(b, s, rtol=rtol, maxiter=50)
        residual = np.linalg.norm(b - (x - mg.apply(x, s))) / np.linalg.norm(b)
        assert mg.relative_residual == pytest.approx(residual, rel=1e-6)
        if stall == 0.5:
            assert cycles < 10
            assert 1e-4 < mg.relative_residual < 1
        else:
            assert cycles < 50
            assert mg.relative_residual < rtol


@pytest.mark.base
@pytest.mark.parametrize('velocity', [None, (4.0,)])
def test_coarsest(velocity):
    """``coarsest`` sets how far the grid is halved: N = 255 in 1D, solved directly by default, gets five levels."""
    from pySDC.helpers.multigrid import GeometricMultigrid

    N = 255
    assert GeometricMultigrid(N, 1, bc='dirichlet-zero').sizes == [N]
    mg = GeometricMultigrid(N, 1, bc='dirichlet-zero', velocity=velocity, coarsest=15)
    assert mg.sizes == [255, 127, 63, 31, 15]
    s = 1e4 / (N + 1) ** 2
    b = np.random.default_rng(12).standard_normal(N)
    x, cycles = mg.solve(b, s, rtol=1e-11, maxiter=50)
    assert np.linalg.norm(b - (x - mg.apply(x, s))) < 1e-11 * np.linalg.norm(b)
    assert cycles <= np.ceil(np.log(1e-11) / np.log(0.3)), f'{cycles} V-cycles'
    with pytest.raises(ValueError, match='at most 15 points'):
        GeometricMultigrid(64, 1, bc='dirichlet-zero', coarsest=15)


@pytest.mark.base
def test_relative_residual_is_set_by_every_solve():
    from pySDC.helpers.multigrid import GeometricMultigrid

    mg = GeometricMultigrid(64, 1)
    b = np.random.default_rng(11).standard_normal(64)
    mg.solve(b, 1e-3, rtol=1e-10, maxiter=50)
    assert 0 < mg.relative_residual < 1e-10
    mg.solve(b, 1e-3, rtol=1e-10, maxiter=0)
    assert mg.relative_residual == 1.0, 'no cycle: the residual of zero'
    mg.solve(np.zeros(64), 1e-3, rtol=1e-10, maxiter=50)
    assert mg.relative_residual == 0.0


@pytest.mark.base
@pytest.mark.parametrize('ndim, N', PECLET_CASES)
@pytest.mark.parametrize('dtype, low, high', [(np.float32, 1e-9, 1e-5), (np.float16, 1e-5, 3e-3)])
def test_formats_dirichlet_advection(ndim, N, dtype, low, high):
    """With zero Dirichlet values and advection, float32 and float16 end at the accuracy of their format too."""
    from pySDC.helpers.multigrid import GeometricMultigrid

    nu, h = 0.1, 1 / (N + 1)
    mg = GeometricMultigrid(N, ndim, bc='dirichlet-zero', nu=nu, velocity=[0.5 * nu / h * d for d in DIRECTIONS[:ndim]])
    s = 1e2 * h**2 / nu
    b = np.random.default_rng(8).standard_normal((N,) * ndim)
    exact, _ = mg.solve(b, s, rtol=1e-13, maxiter=50)
    x, cycles = mg.solve(b.astype(dtype), s, rtol=1e-13, maxiter=50)
    assert x.dtype == dtype
    assert cycles < 50, 'the stall rule ends the solve at the accuracy of the format'
    assert low <= np.abs(x - exact).max() / np.abs(exact).max() < high


@pytest.mark.base
def test_guards():
    from pySDC.helpers.multigrid import GeometricMultigrid

    with pytest.raises(ValueError, match='coarsen'):
        GeometricMultigrid(22, 3)  # 11^3 > 1000 and 11 is odd
    with pytest.raises(ValueError, match='coarsen'):
        GeometricMultigrid(64, 3, bc='dirichlet-zero')  # Dirichlet grids have N = 2^k - 1 interior points
    with pytest.raises(ValueError, match='bc'):
        GeometricMultigrid(64, 2, bc='neumann')
    with pytest.raises(ValueError, match='velocity'):
        GeometricMultigrid(64, 2, velocity=(1.0, 0.0, 0.0))
    assert GeometricMultigrid(64, 2, velocity=(0.0, 0.0)).velocity is None, 'zero velocity is pure diffusion'
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


@pytest.mark.cupy
@pytest.mark.parametrize('ndim, N', PECLET_CASES)
def test_dirichlet_advection_on_GPU(ndim, N):
    """With zero Dirichlet values and advection, the array operations on a GPU solve as those on the CPU."""
    import cupy as cp

    from pySDC.helpers.multigrid import GeometricMultigrid

    nu, h = 0.1, 1 / (N + 1)
    params = {'bc': 'dirichlet-zero', 'nu': nu, 'velocity': [0.5 * nu / h * d for d in DIRECTIONS[:ndim]]}
    s = 1e2 * h**2 / nu
    b = np.random.default_rng(9).standard_normal((N,) * ndim)
    x_cpu, cycles_cpu = GeometricMultigrid(N, ndim, **params).solve(b, s, rtol=1e-11, maxiter=50)
    mg = GeometricMultigrid(N, ndim, xp=cp, **params)
    x, cycles = mg.solve(cp.asarray(b), s, rtol=1e-11, maxiter=50)
    assert abs(cycles - cycles_cpu) <= 1
    assert float(cp.abs(x - cp.asarray(x_cpu)).max()) < 1e-8 * float(np.abs(x_cpu).max())
