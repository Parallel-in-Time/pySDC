import pytest


@pytest.mark.base
@pytest.mark.parametrize('nx', [16])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('axes', [(-2,), (-1,), (-2, -1)])
def test_integration_matrix2D(nx, nz, axes, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base='fft', N=nx)
    helper.add_axis(base='cheby', N=nz)
    helper.setup_fft()

    X, Z = helper.get_grid()

    conv = helper.get_basis_change_matrix()
    S = helper.get_integration_matrix(axes=axes)

    u = helper.u_init
    u[0, ...] = np.sin(X) * Z**2 + np.cos(X) * Z**3
    if axes == (-2,):
        expect = -np.cos(X) * Z**2 + np.sin(X) * Z**3
    elif axes == (-1,):
        expect = np.sin(X) * 1 / 3 * Z**3 + np.cos(X) * 1 / 4 * Z**4
    elif axes in [(-2, -1), (-1, -2)]:
        expect = -np.cos(X) * 1 / 3 * Z**3 + np.sin(X) * 1 / 4 * Z**4
    else:
        raise NotImplementedError

    u_hat = helper.transform(u, axes=(-2, -1))
    S_u_hat = (conv @ S @ u_hat.flatten()).reshape(u_hat.shape)
    S_u = helper.itransform(S_u_hat, axes=(-1, -2))

    assert np.allclose(S_u, expect, atol=1e-12)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [32])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('axes', [(-2,), (-1,), (-2, -1)])
@pytest.mark.parametrize('bx', ['cheby', 'fft'])
@pytest.mark.parametrize('bz', ['cheby', 'fft'])
def test_differentiation_matrix2D(nx, nz, axes, bx, bz, useGPU=False, **kwargs):
    from pySDC.helpers.spectral_helper import SpectralHelper

    helper = SpectralHelper(debug=True, useGPU=useGPU)
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=bz, N=nz)
    helper.setup_fft()

    np = helper.xp

    if useGPU:
        import cupy

        assert np == cupy

    X, Z = helper.get_grid()
    conv = helper.get_basis_change_matrix()
    D = helper.get_differentiation_matrix(axes)

    u = helper.u_init

    if bz == 'cheby':
        u[0, ...] = np.sin(X) * Z**2 + Z**3 + np.cos(2 * X)
        if axes == (-2,):
            expect = np.cos(X) * Z**2 - 2 * np.sin(2 * X)
        elif axes == (-1,):
            expect = np.sin(X) * Z * 2 + Z**2 * 3
        elif axes in [(-2, -1), (-1, -2)]:
            expect = np.cos(X) * 2 * Z
        else:
            raise NotImplementedError
    else:
        u[0, ...] = np.sin(X) * np.cos(2 * Z) + np.cos(2 * X) + np.sin(Z)
        if axes == (-2,):
            expect = np.cos(X) * np.cos(2 * Z) - 2 * np.sin(2 * X)
        elif axes == (-1,):
            expect = np.sin(X) * (-2) * np.sin(2 * Z) + np.cos(Z)
        elif axes in [(-2, -1), (-1, -2)]:
            expect = -2 * np.cos(X) * np.sin(2 * Z)
        else:
            raise NotImplementedError

    u_hat = helper.transform(u)
    D_u_hat = (conv @ D @ u_hat.flatten()).reshape(u_hat.shape)
    D_u = helper.itransform(D_u_hat).real

    assert np.allclose(D_u, expect, atol=1e-11)


@pytest.mark.cupy
@pytest.mark.parametrize('axes', [(-2,), (-1,), (-2, -1)])
@pytest.mark.parametrize('bx', ['cheby', 'fft'])
@pytest.mark.parametrize('bz', ['cheby', 'fft'])
def test_differentiation_matrix2D_GPU(bx, bz, axes):
    test_differentiation_matrix2D(32, 16, bx=bx, bz=bz, axes=axes, useGPU=True)


@pytest.mark.base
@pytest.mark.parametrize('nx', [32])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('bx', ['cheby', 'fft'])
def test_identity_matrix2D(nx, nz, bx, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    helper = SpectralHelper(debug=True)
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base='cheby', N=nz)
    helper.setup_fft()

    X, Z = helper.get_grid()
    conv = helper.get_basis_change_matrix()
    I = helper.get_Id()

    u = helper.u_init
    u[0, ...] = np.sin(X) * Z**2 + Z**3 + np.cos(2 * X)

    u_hat = helper.transform(u, axes=(-2, -1))
    I_u_hat = (conv @ I @ u_hat.flatten()).reshape(u_hat.shape)
    I_u = helper.itransform(I_u_hat, axes=(-1, -2))

    assert np.allclose(I_u, u, atol=1e-12)


@pytest.mark.base
@pytest.mark.parametrize('N', [4, 32])
@pytest.mark.parametrize('base', ['cheby'])
@pytest.mark.parametrize('type', ['diff', 'int'])
def test_matrix1D(N, base, type):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    coeffs = np.random.random(N)

    helper = SpectralHelper(debug=True)
    helper.add_axis(base=base, N=N)
    helper.setup_fft()

    x = helper.get_grid()

    if type == 'diff':
        D = helper.get_differentiation_matrix(axes=(-1,))
    elif type == 'int':
        D = helper.get_integration_matrix(axes=(-1,))

    C = helper.get_basis_change_matrix()

    u = helper.u_init
    u[0] = C @ D @ coeffs
    du = helper.itransform(u)

    if type == 'diff':
        exact = np.polynomial.Chebyshev(coeffs).deriv(1)(x)
    elif type == 'int':
        exact = np.polynomial.Chebyshev(coeffs).integ(1)(x)

    assert np.allclose(exact, du)


def _test_transform_dealias(
    bx,
    bz,
    axis,
    nx=2**4,
    nz=2**2,
    padding=3 / 2,
    useMPI=True,
    useGPU=False,
    **kwargs,
):
    from pySDC.helpers.spectral_helper import SpectralHelper
    import numpy as np

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True, useGPU=useGPU)
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=bz, N=nz)
    helper.setup_fft()
    xp = helper.xp

    if useGPU:
        import cupy

        assert xp == cupy

    _padding = tuple(
        [
            padding,
        ]
        * helper.ndim
    )

    helper_pad = SpectralHelper(comm=comm, debug=True, useGPU=useGPU)
    helper_pad.add_axis(base=bx, N=int(_padding[0] * nx))
    helper_pad.add_axis(base=bz, N=int(_padding[1] * nz))
    helper_pad.setup_fft()

    u_hat = helper.u_init_forward
    u2_hat_expect = helper.u_init_forward
    u_expect = helper.u_init
    u_expect_pad = helper_pad.u_init
    Kx, Kz = helper.get_wavenumbers()
    X, Z = helper.get_grid()
    X_pad, Z_pad = helper_pad.get_grid()

    if useGPU:
        X_CPU = X.get()
        Z_CPU = Z.get()
        X_pad_CPU = X_pad.get()
        Z_pad_CPU = Z_pad.get()
    else:
        X_CPU = X
        Z_CPU = Z
        X_pad_CPU = X_pad
        Z_pad_CPU = Z_pad

    if axis == -2:
        f = nx // 3
        u_hat[0][xp.logical_and(xp.abs(Kx) == f, Kz == 0)] += 1
        u2_hat_expect[0][xp.logical_and(xp.abs(Kx) == 2 * f, Kz == 0)] += 1 / nx
        u2_hat_expect[0][xp.logical_and(xp.abs(Kx) == 0, Kz == 0)] += 2 / nx
        u_expect[0] += xp.cos(f * X) * 2 / nx
        u_expect_pad[0] += xp.cos(f * X_pad) * 2 / nx
    elif axis == -1:

        f = nz // 2 + 1
        u_hat[0][xp.logical_and(xp.abs(Kz) == f, Kx == 0)] += 1
        u2_hat_expect[0][xp.logical_and(Kz == 2 * f, Kx == 0)] += 1 / (2 * nx)
        u2_hat_expect[0][xp.logical_and(Kz == 0, Kx == 0)] += 1 / (2 * nx)

        coef = np.zeros(nz)
        coef[f] = 1 / nx
        u_expect[0] = xp.array(np.polynomial.Chebyshev(coef)(Z_CPU))
        u_expect_pad[0] = xp.array(np.polynomial.Chebyshev(coef)(Z_pad_CPU))
    elif axis in [(-1, -2), (-2, -1)]:
        fx = nx // 3
        fz = nz // 2 + 1

        u_hat[0][xp.logical_and(xp.abs(Kx) == fx, Kz == 0)] += 1
        u_hat[0][xp.logical_and(xp.abs(Kz) == fz, Kx == 0)] += 1

        u2_hat_expect[0][xp.logical_and(xp.abs(Kx) == 2 * fx, Kz == 0)] += 1 / nx
        u2_hat_expect[0][xp.logical_and(xp.abs(Kx) == 0, Kz == 0)] += 2 / nx
        u2_hat_expect[0][xp.logical_and(Kz == 2 * fz, Kx == 0)] += 1 / (2 * nx)
        u2_hat_expect[0][xp.logical_and(Kz == 0, Kx == 0)] += 1 / (2 * nx)
        u2_hat_expect[0][xp.logical_and(Kz == fz, xp.abs(Kx) == fx)] += 2 / nx

        coef = np.zeros(nz)
        coef[fz] = 1 / nx

        u_expect[0] = xp.cos(fx * X) * 2 / nx + xp.array(np.polynomial.Chebyshev(coef)(Z_CPU))
        u_expect_pad[0] = xp.cos(fx * X_pad) * 2 / nx + xp.array(np.polynomial.Chebyshev(coef)(Z_pad_CPU))
    else:
        raise NotImplementedError

    assert bx == 'fft' and bz == 'cheby', 'This test is not implemented for the bases you are looking for'

    u_pad = helper.itransform(u_hat, padding=_padding)
    u = helper.itransform(u_hat).real

    assert not xp.allclose(u_pad.shape, u.shape) or padding == 1

    u2 = u**2
    u2_pad = u_pad**2

    assert xp.allclose(u, u_expect)
    assert xp.allclose(u_pad, u_expect_pad)

    u2_hat = helper.transform(u2_pad, padding=_padding)
    assert xp.allclose(u2_hat_expect, u2_hat)
    assert not xp.allclose(u2_hat_expect, helper.transform(u2)), 'Test is too boring, no dealiasing needed'


@pytest.mark.cupy
@pytest.mark.parametrize('axis', [-1, -2])
@pytest.mark.parametrize('bx', ['fft'])
@pytest.mark.parametrize('bz', ['cheby'])
def test_dealias_GPU(axis, bx, bz, **kwargs):
    _test_transform_dealias(axis=axis, bx=bx, bz=bz, **kwargs, useGPU=True)


@pytest.mark.base
@pytest.mark.parametrize('nx', [3, 8])
@pytest.mark.parametrize('ny', [3, 8])
@pytest.mark.parametrize('nz', [0, 3, 8])
@pytest.mark.parametrize('by', ['fft', 'cheby'])
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
@pytest.mark.parametrize('bx', ['fft', 'cheby'])
@pytest.mark.parametrize('axes', [(-1,), (-2,), (-3,), (-1, -2), (-2, -1), (-1, -2, -3)])
@pytest.mark.parametrize('padding', [1])
def test_transform(nx, ny, nz, bx, by, bz, axes, padding, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base=bx, N=nx)
    helper.add_axis(base=by, N=ny)
    if nz > 0:
        helper.add_axis(base=bz, N=nz)
    elif -3 in axes:
        return None

    helper.setup_fft()
    u = helper.u_init

    if nz > 0:
        u_all = np.random.random((1, nx, ny, nz)).astype(u.dtype)
    else:
        u_all = np.random.random((1, nx, ny)).astype(u.dtype)

    if useMPI:
        u_all = comm.bcast(u_all, root=0)

    u[...] = u_all[
        (
            0,
            *helper.local_slice(False),
        )
    ]

    axes_ordered = []
    for ax in axes:
        if 'FFT' in type(helper.axes[ax]).__name__:
            axes_ordered = axes_ordered + [ax]
        else:
            axes_ordered = [ax] + axes_ordered

    expect_trf = u_all.copy()
    for i in axes_ordered:
        base = helper.axes[i]
        expect_trf = base.transform(expect_trf, axes=(i,))

    trf = helper.transform(u, axes=axes)
    itrf = helper.itransform(trf, axes=axes)

    expect_local = expect_trf[
        (
            ...,
            *helper.local_slice(True),
        )
    ]
    if expect_local.shape != trf.shape:
        expect_local = expect_trf[
            (
                ...,
                *helper.local_slice(True),
            )
        ]

    assert np.allclose(expect_local, trf), 'Forward transform is unexpected'
    assert np.allclose(
        itrf,
        u_all[
            (
                0,
                *helper.local_slice(False),
            )
        ],
    ), 'Backward transform is unexpected'

    if padding > 1:
        _padding = (padding,) * helper.ndim
        u_pad = helper.itransform(trf, axes=axes, padding=_padding)
        trf2 = helper.transform(u_pad, axes=axes, padding=_padding)
        assert np.allclose(trf2, trf)
        assert sum(u_pad.shape) > sum(u.shape), f'{u_pad.shape}, {u.shape}'


@pytest.mark.mpi4py
@pytest.mark.mpi(ranks=[1, 2])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('ny', [4, 8])
@pytest.mark.parametrize('nz', [0, 8])
@pytest.mark.parametrize(
    'by',
    [
        'fft',
    ],
)
@pytest.mark.parametrize(
    'bx',
    [
        'fft',
    ],
)
@pytest.mark.parametrize('bz', ['fft', 'cheby'])
@pytest.mark.parametrize('axes', [(-1,), (-1, -2), (-2, -1, -3)])
@pytest.mark.parametrize('padding', [1, 1.5])
def test_transform_MPI(mpi_ranks, nx, ny, nz, bx, by, bz, axes, padding, **kwargs):
    test_transform(nx=nx, ny=ny, nz=nz, bx=bx, by=by, bz=bz, axes=axes, padding=padding, useMPI=True, **kwargs)


def run_MPI_test(num_procs, **kwargs):
    import os
    import subprocess

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_procs} python {__file__}"

    for key, value in kwargs.items():
        cmd += f' --{key}={value}'
    p = subprocess.Popen(cmd.split(), env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_procs,
    )


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [8])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('bx', ['fft'])
@pytest.mark.parametrize('num_procs', [2, 1])
@pytest.mark.parametrize('axes', ["-1", "-1,-2"])
def test_differentiation_MPI(nx, nz, bx, num_procs, axes):
    run_MPI_test(num_procs=num_procs, test='diff', nx=nx, nz=nz, bx=bx, bz='cheby', axes=axes)


@pytest.mark.mpi4py
@pytest.mark.parametrize('nx', [8])
@pytest.mark.parametrize('nz', [16])
@pytest.mark.parametrize('num_procs', [2, 1])
@pytest.mark.parametrize('axes', ["-1", "-1,-2"])
def test_integration_MPI(nx, nz, num_procs, axes):
    run_MPI_test(num_procs=num_procs, test='int', nx=nx, nz=nz, axes=axes)


@pytest.mark.base
@pytest.mark.parametrize('N', [8, 32])
@pytest.mark.parametrize('bc_val', [0, 3.1415])
def test_tau_method_integral(N, bc_val):
    test_tau_method(bc=0, N=N, bc_val=bc_val, kind='integral')


@pytest.mark.base
@pytest.mark.parametrize('bc', [-1, 0, 1])
@pytest.mark.parametrize('N', [8, 32])
@pytest.mark.parametrize('bc_val', [-99, 3.1415])
def test_tau_method(bc, N, bc_val, kind='Dirichlet', useGPU=False):
    '''
    solve u_x - u + tau P = 0, u(bc) = bc_val

    We choose P = T_N or U_N. We replace the last row in the matrix with the boundary condition to get a
    unique solution for the given resolution.

    The test verifies that the solution satisfies the perturbed equation and the boundary condition.
    '''
    from pySDC.helpers.spectral_helper import SpectralHelper
    import scipy.sparse as sp
    import numpy as np

    helper = SpectralHelper(debug=True, useGPU=useGPU)
    helper.add_component('u')
    helper.add_axis(base='cheby', N=N)
    helper.setup_fft()

    xp = helper.xp
    linalg = helper.linalg

    if useGPU:
        import cupy

        assert xp == cupy

    if kind == 'integral':
        helper.add_BC('u', 'u', 0, v=bc_val, kind='integral')
    else:
        helper.add_BC('u', 'u', 0, x=bc, v=bc_val, kind='Dirichlet')
    helper.setup_BCs()

    C = helper.get_basis_change_matrix()
    D = helper.get_differentiation_matrix(axes=(-1,))
    Id = helper.get_Id()

    _A = helper.get_empty_operator_matrix()
    helper.add_equation_lhs(_A, 'u', {'u': D - Id})
    A = helper.convert_operator_matrix_to_operator(_A)
    A = helper.put_BCs_in_matrix(A)

    rhs = helper.put_BCs_in_rhs(xp.zeros((1, N)))
    rhs_hat = helper.transform(rhs, axes=(-1,))

    sol_hat = linalg.spsolve(A, rhs_hat.flatten())

    if useGPU:
        C = C.get()
        sol_hat = sol_hat.get()

    sol_poly = np.polynomial.Chebyshev(sol_hat)
    d_sol_poly = sol_poly.deriv(1)
    x = xp.linspace(-1, 1, 100)

    if kind == 'integral':
        integral = sol_poly.integ(1, lbnd=-1)
        assert xp.isclose(integral(1), bc_val), 'Solution does not satisfy boundary condition'
    else:
        assert xp.isclose(sol_poly(bc), bc_val), 'Solution does not satisfy boundary condition'

    coef = np.append(np.zeros(N - 1), [1])
    P = np.polynomial.Chebyshev(C @ coef)
    tau = (d_sol_poly(x) - sol_poly(x)) / P(x)
    assert np.allclose(tau, tau[0]), 'Solution does not satisfy perturbed equation'


@pytest.mark.cupy
def test_tau_method_GPU():
    test_tau_method(-1, 8, 2.77, useGPU=True)


@pytest.mark.base
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
@pytest.mark.parametrize('bc_val', [-2, 1.0])
def test_tau_method2D(nz, nx, bc_val, bc=-1, plotting=False, useMPI=False, **kwargs):
    '''
    solve u_z - 0.1u_xx -u_x + tau P = 0, u(bc) = sin(bc_val*x) -> space-time discretization of advection-diffusion
    problem. We do FFT in x-direction and Chebychov in z-direction.
    '''
    from pySDC.helpers.spectral_helper import SpectralHelper
    import numpy as np

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis('fft', N=nx)
    helper.add_axis('cheby', N=nz)
    helper.add_component(['u'])
    helper.setup_fft()

    X, Z = helper.get_grid(forward_output=True)
    x = X[:, 0]
    z = Z[0, :]

    bcs = np.sin(bc_val * x)
    helper.add_BC('u', 'u', 1, kind='dirichlet', x=bc, v=bcs)
    helper.setup_BCs()

    # generate matrices
    Dz = helper.get_differentiation_matrix(axes=(1,))
    Dx = helper.get_differentiation_matrix(axes=(0,))
    Dxx = helper.get_differentiation_matrix(axes=(0,), p=2)

    # generate operator
    diag = True
    _A = helper.get_empty_operator_matrix(diag=diag)
    helper.add_equation_lhs(_A, 'u', {'u': Dz - Dxx * 1e-1 - Dx}, diag=diag)
    A = helper.convert_operator_matrix_to_operator(_A, diag=diag)

    # prepare system to solve
    A = helper.put_BCs_in_matrix(A)
    rhs_hat = helper.put_BCs_in_rhs_hat(helper.u_init_forward)

    # solve the system
    sol_hat = helper.u_init_forward
    sol_hat[0] = (helper.sparse_lib.linalg.spsolve(A, rhs_hat.flatten())).reshape(X.shape)
    sol = helper.redistribute(helper.itransform(sol_hat), True, 1).real

    # construct polynomials for testing
    sol_cheby = helper.transform(sol, axes=(-1,))
    polys = [np.polynomial.Chebyshev(sol_cheby[0, i, :]) for i in range(sol_cheby.shape[1])]

    Pz = np.polynomial.Chebyshev(np.append(np.zeros(nz - 1), [1]))
    tau_term, _ = np.meshgrid(Pz(z), np.ones(sol_hat.shape[1]))
    error = ((A @ sol_hat.flatten()).reshape(X.shape) / tau_term).real

    if plotting:
        import matplotlib.pyplot as plt

        _X, _Z = helper.get_grid(forward_output=False)
        im = plt.pcolormesh(_X, _Z, sol[0])
        plt.colorbar(im)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.show()

    for i in range(sol.shape[1]):

        assert np.isclose(polys[i](bc), bcs[i]), f'Solution does not satisfy boundary condition x={x[i]}'

        assert np.allclose(
            polys[i](z), sol[0, i, :]
        ), f'Solution is incorrectly transformed back to real space at x={x[i]}'

    assert np.allclose(error, error[0, 0]), 'Solution does not satisfy perturbed equation'


@pytest.mark.mpi4py
@pytest.mark.mpi(ranks=[2])
@pytest.mark.parametrize('nx', [4, 8])
@pytest.mark.parametrize('nz', [4, 8])
@pytest.mark.parametrize('bc_val', [-2])
@pytest.mark.parametrize('num_procs', [2, 1])
def test_tau_method2D_MPI(mpi_ranks, nz, nx, bc_val, num_procs, **kwargs):
    test_tau_method2D(nz=nz, nx=nx, bc_val=bc_val, num_procs=num_procs, test='tau', useMPI=True)


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_procs', [1])
@pytest.mark.parametrize('axis', [-1, -2])
@pytest.mark.parametrize('bx', ['fft'])
@pytest.mark.parametrize('bz', ['cheby'])
def test_dealias_MPI(num_procs, axis, bx, bz, nx=32, nz=64, **kwargs):
    run_MPI_test(num_procs=num_procs, axis=axis, nx=nx, nz=nz, bx=bx, bz=bz, test='dealias')


@pytest.mark.base
@pytest.mark.parametrize('nx', [8])
@pytest.mark.parametrize('ny', [16])
@pytest.mark.parametrize('nz', [32])
@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('bz', ['fft', 'cheby', 'ultraspherical'])
@pytest.mark.parametrize('axes', [(-1,), (-2,), (-3,), (-1, -2), (-2, -3), (-1, -3), (-1, -2, -3)])
def test_differentiation_matrix3D(nx, ny, nz, bz, axes, p, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base='fft', N=nx)
    helper.add_axis(base='fft', N=ny)
    helper.add_axis(base=bz, N=nz)
    helper.setup_fft()

    X, Y, Z = helper.get_grid()

    if bz == 'cheby' and p > 1:
        return None
    elif bz == 'ultraspherical' and -1 in axes:
        conv = helper.get_basis_change_matrix(p_out=0, p_in=p)
    else:
        conv = helper.get_basis_change_matrix()

    D = helper.get_differentiation_matrix(axes, p=p)

    u = helper.u_init

    z_part = np.sin(3 * Z) if bz == 'fft' else Z**3
    z_part_z = 3 * np.cos(3 * Z) if bz == 'fft' else 3 * Z**2
    z_part_zz = -9 * np.sin(3 * Z) if bz == 'fft' else 6 * Z

    u[0, ...] = np.sin(X) * np.cos(2 * Y) * z_part + np.cos(2 * X) + np.sin(Y) + np.sin(4 * Z)
    if axes == (-3,):
        if p == 1:
            expect = np.cos(X) * np.cos(2 * Y) * z_part - 2 * np.sin(2 * X)
        else:
            expect = -np.sin(X) * np.cos(2 * Y) * z_part - 4 * np.cos(2 * X)
    elif axes == (-2,):
        if p == 1:
            expect = np.sin(X) * (-2) * np.sin(2 * Y) * z_part + np.cos(Y)
        else:
            expect = np.sin(X) * (-4) * np.cos(2 * Y) * z_part - np.sin(Y)
    elif axes == (-1,):
        if p == 1:
            expect = np.sin(X) * np.cos(2 * Y) * z_part_z + 4 * np.cos(4 * Z)
        elif p == 2:
            expect = np.sin(X) * np.cos(2 * Y) * z_part_zz - 16 * np.sin(4 * Z)
    elif sorted(axes) == [-3, -2]:
        if p == 1:
            expect = -2 * np.cos(X) * np.sin(2 * Y) * z_part
        elif p == 2:
            expect = 4 * np.sin(X) * np.cos(2 * Y) * z_part
    elif sorted(axes) == [-2, -1]:
        if p == 1:
            expect = np.sin(X) * (-2) * np.sin(2 * Y) * z_part_z
        elif p == 2:
            expect = np.sin(X) * (-4) * np.cos(2 * Y) * z_part_zz
    elif sorted(axes) == [-3, -1]:
        if p == 1:
            expect = np.cos(X) * np.cos(2 * Y) * z_part_z
        elif p == 2:
            expect = -np.sin(X) * np.cos(2 * Y) * z_part_zz
    elif axes == (-1, -2, -3):
        if p == 1:
            expect = np.cos(X) * (-2) * np.sin(2 * Y) * z_part_z
        elif p == 2:
            expect = -np.sin(X) * (-4) * np.cos(2 * Y) * z_part_zz
    else:
        raise NotImplementedError

    u_hat = helper.transform(u)
    D_u_hat = (conv @ D @ u_hat.flatten()).reshape(u_hat.shape)
    D_u = helper.itransform(D_u_hat).real

    error = np.linalg.norm(D_u - expect)
    assert np.isclose(error, 0, atol=6e-8), f'Got {error=:.2e}'

    if useMPI:
        if comm.size == 2:
            assert u_hat.shape[1] < nx or u_hat.shape[2] < ny, 'Not distributed'
        elif comm.size > 2:
            assert u_hat.shape[1] < nx and u_hat.shape[2] < ny, 'Not distributed in pencils'


@pytest.mark.mpi4py
@pytest.mark.mpi(ranks=[2, 4])
@pytest.mark.parametrize('nx', [8])
@pytest.mark.parametrize('ny', [16])
@pytest.mark.parametrize('nz', [32])
@pytest.mark.parametrize('bz', ['fft', 'cheby', 'ultraspherical'])
@pytest.mark.parametrize('axes', [(-1,), (-2,), (-3,), (-1, -2), (-2, -3), (-1, -3), (-1, -2, -3)])
def test_differentiation_matrix3DMPI(mpi_ranks, nx, ny, nz, bz, axes, useMPI=True, **kwargs):
    test_differentiation_matrix3D(nx, ny, nz, bz, axes, p=1, **kwargs)


@pytest.mark.base
@pytest.mark.parametrize('nx', [32])
@pytest.mark.parametrize('ny', [0, 16])
@pytest.mark.parametrize('nz', [8])
@pytest.mark.parametrize('bx', ['cheby', 'fft'])
def test_identity_matrix_ND(nx, ny, nz, bx, useMPI=False, **kwargs):
    import numpy as np
    from pySDC.helpers.spectral_helper import SpectralHelper

    if useMPI:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        comm = None

    helper = SpectralHelper(comm=comm, debug=True)
    helper.add_axis(base=bx, N=nx)
    if ny > 0:
        helper.add_axis(base=bx, N=ny)
    helper.add_axis(base='cheby', N=nz)
    helper.setup_fft()

    grid = helper.get_grid()
    conv = helper.get_basis_change_matrix()
    I = helper.get_Id()

    u = helper.u_init
    u[0, ...] = np.sin(grid[-2]) * grid[-1] ** 2 + grid[-1] ** 3 + np.cos(2 * grid[-2])

    if ny > 0:
        u[0, ...] += np.cos(grid[-3])

    u_hat = helper.transform(u, axes=(-2, -1))
    I_u_hat = (conv @ I @ u_hat.flatten()).reshape(u_hat.shape)
    I_u = helper.itransform(I_u_hat, axes=(-1, -2))

    assert np.allclose(I_u, u, atol=1e-12)


@pytest.mark.base
def test_cache_decorator():
    from pySDC.helpers.spectral_helper import cache
    import numpy as np

    class Dummy:
        num_calls = 0

        @cache
        def increment(self, x):
            self.num_calls += 1
            return x + 1

    dummy = Dummy()
    values = [0, 1, 1, 0, 3, 1, 2]
    unique_vals = np.unique(values)

    for x in values:
        assert dummy.increment(x) == x + 1

    assert dummy.num_calls < len(values)
    assert dummy.num_calls == len(unique_vals)


@pytest.mark.base
def test_cache_memory_leaks():
    from pySDC.helpers.spectral_helper import cache

    track = [0, 0]

    class KeepTrack:

        def __init__(self):
            track[0] += 1
            track[1] = 0

        @cache
        def method(self, a, b, c=1, d=2):
            track[1] += 1
            return f"{a},{b},c={c},d={d}"

        def __del__(self):
            track[0] -= 1

    def function():
        obj = KeepTrack()
        for _ in range(10):
            obj.method(1, 2, d=2)
            assert track[0] == 1
            assert track[1] == 1

    for _ in range(3):
        function()

    assert track[0] == 0, "possible memory leak with the @cache"


@pytest.mark.base
def test_block_diagonal_operators(N=16):
    from pySDC.helpers.spectral_helper import SpectralHelper
    import numpy as np

    helper = SpectralHelper(comm=None, debug=True)
    helper.add_axis('fft', N=N)
    helper.add_axis('cheby', N=N)
    helper.add_component(['u', 'v'])
    helper.setup_fft()

    # generate matrices
    Dz = helper.get_differentiation_matrix(axes=(1,))
    Dx = helper.get_differentiation_matrix(axes=(0,))

    def get_operator(diag):
        _A = helper.get_empty_operator_matrix(diag=diag)
        helper.add_equation_lhs(_A, 'u', {'u': Dx}, diag=diag)
        helper.add_equation_lhs(_A, 'v', {'v': Dz}, diag=diag)
        return helper.convert_operator_matrix_to_operator(_A, diag=diag)

    AD = get_operator(True)
    A = get_operator(False)

    assert np.allclose(A.toarray(), AD.toarray()), 'Operators don\'t match'
    assert A.data.nbytes > AD.data.nbytes, 'Block diagonal operator did not conserve memory over general operator'


if __name__ == '__main__':
    str_to_bool = lambda me: False if me == 'False' else True
    str_to_tuple = lambda arg: tuple(int(me) for me in arg.split(','))

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, help='Dof in x direction')
    parser.add_argument('--nz', type=int, help='Dof in z direction')
    parser.add_argument('--axes', type=str_to_tuple, help='Axes over which to transform')
    parser.add_argument('--axis', type=int, help='Direction of the action')
    parser.add_argument('--bz', type=str, help='Base in z direction')
    parser.add_argument('--bx', type=str, help='Base in x direction')
    parser.add_argument('--bc_val', type=int, help='Value of boundary condition')
    parser.add_argument('--test', type=str, help='type of test', choices=['transform', 'diff', 'int', 'tau', 'dealias'])
    parser.add_argument('--useMPI', type=str_to_bool, help='use MPI or not', choices=[True, False], default=True)
    args = parser.parse_args()

    if args.test == 'transform':
        test_transform(**vars(args))
    elif args.test == 'diff':
        test_differentiation_matrix2D(**vars(args))
    elif args.test == 'int':
        test_integration_matrix2D(**vars(args))
    elif args.test == 'tau':
        test_tau_method2D(**vars(args))
    elif args.test == 'dealias':
        _test_transform_dealias(**vars(args))
    elif args.test is None:
        # test_differentiation_matrix3D(2, 2, 4, 'cheby', p=1, axes=(-1, -2, -3), useMPI=True)
        # test_differentiation_matrix3D(2, 2, 4, 'ultraspherical', p=1, axes=(-1, -2, -3), useMPI=True)
        # test_differentiation_matrix3D(32, 32, 32, 'fft', p=2, axes=(-1, -2), useMPI=True)
        # test_transform(4, 4, 8, 'fft', 'fft', 'cheby', axes=(-1,), padding=1.5, useMPI=True)
        # test_dealias_GPU(axis=(-1, -2), bx='fft', bz='cheby', padding=1.5)
        # test_differentiation_matrix2D(2**5, 2**5, 'T2U', bx='cheby', bz='fft', axes=(-2, -1))
        # test_matrix1D(4, 'cheby', 'diff')
        # test_tau_method(-1, 8, 99, kind='Dirichlet')
        # test_tau_method2D('T2U', 2**8, 2**8, -2, plotting=True, useMPI=True)
        # test_tau_method2D('T2U', 2**1, 2**2, -2, plotting=False, useMPI=True)
        # test_filter(6, 6, (0,))
        # _test_transform_dealias('fft', 'cheby', -1, nx=2**2, nz=5, padding=1.5)
        test_tau_method_GPU()
    else:
        raise NotImplementedError
    print('done')
